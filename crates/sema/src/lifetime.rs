//! Lifetime/deps + move tracking pass on HIR.
//!
//! Implements SPEC §6 (Lifetime and ownership): move semantics (§6.1),
//! auto-free at scope exit (§6.2), forced handling for `*T` deref (§6.3),
//! downstream invalidation on owner change (§6.4) via the kill_lid path —
//! deps sets, null-collapse for `*T` aliases, poisoning of `&T` borrows,
//! pointer narrowing — and heap-drop insertion at scope boundaries
//! (returns and block exit).
//!
//! Diagnostics produced here are reported alongside type errors.

use crate::hir::*;
use crate::{SemaError, SemaWarning};
use maka_lexer::Span;
use std::collections::HashSet;

/// Per-local state tracked through the function body.
#[derive(Debug, Clone)]
struct LocalState {
    /// `live` for stack values (the LID is the local itself); for heap bindings,
    /// `live`/`moved` per §6.1 (move semantics on owning types).
    moved: bool,
    /// For reference-like locals (`&T`, `*T`, `[]T`), the set of LIDs in deps.
    deps: HashSet<u32>,
    /// For `&T`-like strong references: poisoned if a needed LID is dead.
    poisoned: bool,
    /// Narrowing window: if Some(scope_depth), the next unwrap of this *T can skip null-check
    /// (set when inside `if (p != null)` branch).
    narrowed_until: Option<u32>,
    /// Statically known to be non-null at the current program point — set when the local
    /// was last assigned from `heap T(...)` or another `known_nonnull` source.  Cleared on
    /// any other reassignment.  Used in addition to `narrowed_until` to skip null-checks.
    known_nonnull: bool,
    /// Set when the lifetime pass auto-nulled this pointer at scope exit (its pointee
    /// went out of scope).  Cleared by a subsequent re-assignment to a known-non-null
    /// value on every code path.  Reads of a local with this flag still set emit a
    /// flow-sensitive warning — the runtime value is NULL even though the user never
    /// wrote `p = null;`.
    auto_nulled: bool,
    /// This local holds a borrow/view (`&T`, `*T`, or a `string` view from e.g.
    /// `as_str`) whose referent is a FUNCTION-LOCAL value that dies on return
    /// (a stack local or an owning local).  Set at assignment when the RHS is a
    /// dying borrow; cleared when re-assigned from a non-dying source.  The escape
    /// check rejects returning/escaping such a local - it makes `string v =
    /// s.as_str(); return v;` (a use-after-free of `s`'s freed buffer) a compile
    /// error, the same as the direct `return s.as_str();` already is.
    holds_dying_borrow: bool,
}

impl LocalState {
    fn fresh() -> Self {
        Self {
            moved: false,
            deps: HashSet::new(),
            poisoned: false,
            narrowed_until: None,
            known_nonnull: false,
            auto_nulled: false,
            holds_dying_borrow: false,
        }
    }
}

/// Result of one function's lifetime/null-proof analysis.
pub struct AnalyzeOk {
    pub warnings: Vec<SemaWarning>,
}

pub fn analyze_func(
    sym: &SymTab,
    f: &mut HFunc,
    summaries: &[bool],
    ret_borrows: &[Vec<bool>],
    initial_nonnull: &[LocalId],
) -> Result<AnalyzeOk, Vec<SemaError>> {
    let mut a = Analyzer::new(sym, f, summaries, ret_borrows);
    for lid in initial_nonnull {
        if let Some(st) = a.state.get_mut(lid.0 as usize) {
            st.known_nonnull = true;
        }
    }
    // analyze_block's `?` would early-return without surfacing warnings, but warnings
    // are only meaningful when analysis ran to completion, so this is fine.
    a.analyze_block(&mut Vec::new(), 0)?;
    let mut errors = std::mem::take(&mut a.errors);
    let warnings = std::mem::take(&mut a.warnings);
    // Materialize the per-statement `*T` auto-nulls the analysis recorded into
    // real `alias = NULL;` statements (in every block, incl. match-arm bodies),
    // BEFORE hoisting shifts statement indices.  A `*T` alias whose owner was
    // reassigned/moved is nulled at runtime so a later guard is honest.
    inject_stmt_nulls(&mut f.body, &f.locals);
    // Hoist owning temporaries consumed by a borrowing context into hidden
    // owning locals, so the auto-free machinery (below) drops them.
    hoist_owning_temps(sym, f);
    // Final: walk the body and fill heap_to_free for each block in reverse scope-decl order.
    fill_heap_drops(sym, f);
    // Scoped-borrow rule: a cross-thread closure may capture a borrowed reference
    // only if its handle is provably joined before scope exit.
    check_scoped_thread_borrows(f, &mut errors);
    if errors.is_empty() {
        Ok(AnalyzeOk { warnings })
    } else {
        Err(std::mem::take(&mut errors))
    }
}

/// True iff a value of this type can never be null at the language level —
/// i.e. it is not one of Maka's nullable pointer carriers (`*T`, `own *T`,
/// `raw *T`, `heap T`).  Used as the initial fact for interprocedural
/// return-value-non-null summaries.
fn return_type_nonnull(t: &HType) -> bool {
    !matches!(t, HType::Ptr { .. } | HType::OwnPtr { .. } | HType::RawPtr { .. } | HType::Heap { .. })
}

/// Compute per-function "never-returns-null" summaries by fixpoint over the
/// lowered HIR.  Result is a `Vec<bool>` indexed by `FuncId.0`, where `true`
/// means every return path in this function is provably non-null.
///
/// Algorithm: initialise from the static return type (non-pointer = NeverNull),
/// then iterate.  For each not-yet-NeverNull function, check whether every
/// `Return { value }` expression is `static_nonnull` *under the current
/// summary table*.  Functions with no explicit return on a fall-off path are
/// safe iff they return `unit` (which is non-pointer, so already true).
///
/// Conservative: a returned `Local(_)` of pointer type is not treated as
/// non-null here — flow-sensitive Local tracking lives in the per-function
/// pass that runs *after* summary computation.
// String builders that always allocate a fresh non-null buffer (they never
// return null), so their `own *string` result is known-non-null and derefs
// without a guard.  Compiler builtins: string concat `+` (and its
// free-operand variants), the `*_to_str` conversions, and `format`.
pub fn is_nonnull_string_builtin(id: u32) -> bool {
    id == u32::MAX - 5  || id == u32::MAX - 8  || id == u32::MAX - 9  || id == u32::MAX - 10
        || id == u32::MAX - 11 || id == u32::MAX - 12 || id == u32::MAX - 13 || id == u32::MAX - 14
        || id == u32::MAX - 59
}

// Runtime builder externs that likewise always allocate a non-null result.
// `__maka_rt_read_file` is deliberately ABSENT - it returns null on error.
fn is_nonnull_builder_extern(c_name: &str) -> bool {
    matches!(
        c_name,
        "__maka_rt_int_to_str"
            | "__maka_rt_substr_owned"
            | "__maka_rt_str_to_upper"
            | "__maka_rt_str_to_lower"
            | "__maka_rt_str_trim"
            | "__maka_rt_str_replace"
    )
}

pub fn compute_return_summaries(sym: &SymTab, funcs: &[HFunc]) -> Vec<bool> {
    let n = sym.sigs.len();
    let mut summaries: Vec<bool> = (0..n)
        .map(|i| {
            return_type_nonnull(&sym.sigs[i].ret)
                || (sym.sigs[i].is_extern && is_nonnull_builder_extern(&sym.sigs[i].c_name))
        })
        .collect();
    // Extern signatures stay at their type-derived value; only Maka bodies can be proven.
    loop {
        let mut changed = false;
        for f in funcs {
            let id = f.id.0 as usize;
            if summaries[id] { continue; }
            if function_returns_nonnull(f, &summaries) {
                summaries[id] = true;
                changed = true;
            }
        }
        if !changed { break; }
    }
    summaries
}

/// Per-function borrow provenance, indexed by `FuncId.0` (parallel to
/// `sym.sigs`): for a function whose return type is a borrowed view
/// (`string` / `&T` / `*T` / `raw *T`), the set of parameter indices its
/// return value may alias.  Everything else stays all-false.
///
/// This is what lets the escape walk follow a borrowed return value back
/// across a call to the argument it aliases - so `return localString.as_str()`
/// (a borrow of a local that dies on return) is rejected exactly like the
/// direct `return &local;`.  Computed by monotone fixpoint: provenance only
/// grows as nested callees' provenance becomes known, so it converges.
pub fn compute_return_borrows(sym: &SymTab, funcs: &[HFunc]) -> Vec<Vec<bool>> {
    let n = sym.sigs.len();
    let mut borrows: Vec<Vec<bool>> =
        (0..n).map(|i| vec![false; sym.sigs[i].param_tys.len()]).collect();
    loop {
        let mut changed = false;
        for f in funcs {
            let id = f.id.0 as usize;
            if id >= n || !ty_is_borrowed_view(&sym.sigs[id].ret) { continue; }
            let mut acc = borrows[id].clone();
            collect_ret_borrows_in_block(&f.body, f, &borrows, &mut acc);
            if acc != borrows[id] {
                borrows[id] = acc;
                changed = true;
            }
        }
        if !changed { break; }
    }
    borrows
}

/// A borrowed (non-owning) view type: returning one aliases data the function
/// does not own, so the result's lifetime is tied to wherever that data lives.
/// Owning carriers (`own *T`, `heap T`, value types) transfer ownership and so
/// never borrow a parameter.
fn ty_is_borrowed_view(t: &HType) -> bool {
    matches!(
        t,
        HType::Str | HType::Ref { .. } | HType::Ptr { .. } | HType::RawPtr { .. }
    )
}

fn collect_ret_borrows_in_block(b: &HBlock, f: &HFunc, borrows: &[Vec<bool>], acc: &mut Vec<bool>) {
    for s in &b.stmts {
        collect_ret_borrows_in_stmt(s, f, borrows, acc);
    }
}

fn collect_ret_borrows_in_stmt(s: &HStmt, f: &HFunc, borrows: &[Vec<bool>], acc: &mut Vec<bool>) {
    match s {
        HStmt::Return { value: Some(v), .. } => params_borrowed_by(v, f, borrows, acc),
        HStmt::If { then_b, else_b, .. } => {
            collect_ret_borrows_in_block(then_b, f, borrows, acc);
            if let Some(eb) = else_b {
                collect_ret_borrows_in_block(eb, f, borrows, acc);
            }
        }
        HStmt::While { body, .. } | HStmt::ForEach { body, .. } => {
            collect_ret_borrows_in_block(body, f, borrows, acc)
        }
        HStmt::ForC { body, .. } => collect_ret_borrows_in_block(body, f, borrows, acc),
        HStmt::Block(b) | HStmt::Unsafe(b, _) => collect_ret_borrows_in_block(b, f, borrows, acc),
        _ => {}
    }
}

/// Accumulate the parameter indices that `e` - a value of borrowed-view type
/// being returned - may alias.  Mirrors the escape walk: a borrow of a
/// parameter place, the parameter itself passed through, a field/element of
/// either, the borrowed return of a nested call (followed into the argument it
/// aliases), or a reinterpret/deref of any of those.  Anything else (a fresh
/// `alloc`, a string literal, a value the function owns) borrows no parameter.
fn params_borrowed_by(e: &HExpr, f: &HFunc, borrows: &[Vec<bool>], acc: &mut Vec<bool>) {
    use HExprKind::*;
    match &e.kind {
        AddrOfRef { place, .. } => {
            if let Some(root) = root_local(place) {
                if let Some(idx) = f.params.iter().position(|p| *p == root) {
                    if idx < acc.len() { acc[idx] = true; }
                }
            }
        }
        Local(id) => {
            if let Some(idx) = f.params.iter().position(|p| *p == *id) {
                if idx < acc.len() { acc[idx] = true; }
            }
        }
        Field { base, .. } | Index { base, .. } => params_borrowed_by(base, f, borrows, acc),
        Call { callee, args } | InlineCall { callee, args, .. } => {
            let cid = callee.0 as usize;
            if let Some(cb) = borrows.get(cid) {
                for (m, &b) in cb.iter().enumerate() {
                    if b {
                        if let Some(a) = args.get(m) {
                            params_borrowed_by(a, f, borrows, acc);
                        }
                    }
                }
            }
        }
        Cast { expr, .. } | CheckedCast { expr, .. } | DerefRef(expr) | Unwrap { expr, .. } => {
            params_borrowed_by(expr, f, borrows, acc)
        }
        _ => {}
    }
}

/// Result of walking a block / statement during summary computation.
#[derive(Clone, Copy)]
struct WalkRes {
    /// Every return seen on this path so far is non-null.
    ok: bool,
    /// Every path through this block exits (return / break / continue / propagate).
    /// Used so the post-`if` join state correctly carries the non-terminating
    /// branch's state instead of meeting against an unreachable continuation.
    terminates: bool,
}

fn function_returns_nonnull(f: &HFunc, summaries: &[bool]) -> bool {
    let mut state: Vec<bool> = vec![false; f.locals.len()];
    // Function parameters whose declared type is non-nullable (`&T`, `&mut T`,
    // value types) start known-non-null.  Pointer parameters stay false until
    // proven by flow.
    for (i, li) in f.locals.iter().enumerate() {
        if matches!(li.storage, StorageClass::Param) && return_type_nonnull(&li.ty) {
            state[i] = true;
        }
    }
    let mut any_return = false;
    let r = block_walk(&f.body, &mut state, summaries, &mut any_return);
    r.ok && any_return
}

fn block_walk(b: &HBlock, state: &mut Vec<bool>, summaries: &[bool], any: &mut bool) -> WalkRes {
    let mut terminates = false;
    for s in &b.stmts {
        let r = stmt_walk(s, state, summaries, any);
        if !r.ok { return WalkRes { ok: false, terminates: terminates || r.terminates }; }
        if r.terminates { terminates = true; break; }
    }
    WalkRes { ok: true, terminates }
}

fn stmt_walk(s: &HStmt, state: &mut Vec<bool>, summaries: &[bool], any: &mut bool) -> WalkRes {
    match s {
        HStmt::Let { local, init, .. } => {
            state[local.0 as usize] = expr_nonnull_flow(init, state, summaries);
            WalkRes { ok: true, terminates: false }
        }
        HStmt::Assign { place, value, .. } => {
            if let HExprKind::Local(id) = &place.kind {
                state[id.0 as usize] = expr_nonnull_flow(value, state, summaries);
            }
            WalkRes { ok: true, terminates: false }
        }
        HStmt::Return { value: Some(v), .. } => {
            *any = true;
            WalkRes { ok: expr_nonnull_flow(v, state, summaries), terminates: true }
        }
        HStmt::Return { value: None, .. } => {
            *any = true;
            WalkRes { ok: true, terminates: true }
        }
        HStmt::If { cond, then_b, else_b, .. } => {
            let (then_nn, else_nn) = detect_null_narrow(cond);
            // Walk then-branch with narrowed state.
            let mut then_state = state.clone();
            if let Some(id) = then_nn { then_state[id.0 as usize] = true; }
            let then_r = block_walk(then_b, &mut then_state, summaries, any);
            if !then_r.ok { return WalkRes { ok: false, terminates: false }; }

            // Walk else-branch (or treat absent else as falling through).
            let mut else_state = state.clone();
            if let Some(id) = else_nn { else_state[id.0 as usize] = true; }
            let (else_r, has_else) = match else_b {
                Some(eb) => (block_walk(eb, &mut else_state, summaries, any), true),
                None => (WalkRes { ok: true, terminates: false }, false),
            };
            if !else_r.ok { return WalkRes { ok: false, terminates: false }; }

            // Join: post-if state depends on which branches terminate.
            for i in 0..state.len() {
                state[i] = match (then_r.terminates, else_r.terminates) {
                    (true, true) => state[i], // both terminate; post-if unreachable
                    (true, false) => else_state[i], // only then terminates — else's state continues
                    (false, true) => then_state[i],
                    (false, false) => then_state[i] && else_state[i], // meet
                };
            }
            let terminates = if has_else { then_r.terminates && else_r.terminates } else { false };
            WalkRes { ok: true, terminates }
        }
        HStmt::While { body, .. } => {
            // Loop body might execute 0+ times.  Walk it to surface inner returns,
            // then meet its post-state with the pre-state (which holds if the loop
            // doesn't iterate).
            let mut body_state = state.clone();
            let r = block_walk(body, &mut body_state, summaries, any);
            if !r.ok { return WalkRes { ok: false, terminates: false }; }
            for i in 0..state.len() {
                state[i] = state[i] && body_state[i];
            }
            WalkRes { ok: true, terminates: false }
        }
        HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => {
            let mut body_state = state.clone();
            let r = block_walk(body, &mut body_state, summaries, any);
            if !r.ok { return WalkRes { ok: false, terminates: false }; }
            for i in 0..state.len() {
                state[i] = state[i] && body_state[i];
            }
            WalkRes { ok: true, terminates: false }
        }
        HStmt::Block(b) | HStmt::Unsafe(b, _) => block_walk(b, state, summaries, any),
        HStmt::Break { .. } | HStmt::Continue { .. } => WalkRes { ok: true, terminates: true },
        HStmt::Propagate { .. } => WalkRes { ok: true, terminates: true },
        HStmt::ExprStmt(_) => WalkRes { ok: true, terminates: false },
    }
}

/// Flow-aware non-null classifier — consults per-LID state for Local
/// references.  Used by the summary fixpoint.
fn expr_nonnull_flow(e: &HExpr, state: &[bool], summaries: &[bool]) -> bool {
    match &e.kind {
        HExprKind::HeapAlloc(_) => true,
        HExprKind::AddrOfRef { .. } => true,
        HExprKind::Local(id) => state.get(id.0 as usize).copied().unwrap_or(false),
        HExprKind::LitNull => false,
        HExprKind::Cast { expr, kind, .. } | HExprKind::CheckedCast { expr, kind, .. } => {
            if matches!(kind, CastKind::IntPtrToEnumPtrChecked) { return false; }
            expr_nonnull_flow(expr, state, summaries)
        }
        HExprKind::Call { callee, .. } | HExprKind::InlineCall { callee, .. } => {
            is_nonnull_string_builtin(callee.0)
                || summaries.get(callee.0 as usize).copied().unwrap_or(false)
        }
        HExprKind::DerefRef(inner) => expr_nonnull_flow(inner, state, summaries),
        _ => false,
    }
}

/// Pre-pass: walk a function's body with flow tracking and record
/// `(lifted FuncId, lifted LocalId)` pairs for every `Closure` expression
/// whose env_value is provably non-null at the capture site.  The driver
/// applies these as the lifted body's initial state.  Independent of the
/// per-func `Analyzer` so it can run before any function's main pass.
pub fn harvest_capture_nonnull(
    sym: &SymTab,
    f: &HFunc,
    summaries: &[bool],
    out: &mut std::collections::HashMap<u32, Vec<LocalId>>,
) {
    let _ = sym;
    let mut state: Vec<bool> = vec![false; f.locals.len()];
    for (i, li) in f.locals.iter().enumerate() {
        if matches!(li.storage, StorageClass::Param) && return_type_nonnull(&li.ty) {
            state[i] = true;
        }
    }
    harvest_block(&f.body, &mut state, summaries, out);
}

fn harvest_block(
    b: &HBlock,
    state: &mut Vec<bool>,
    summaries: &[bool],
    out: &mut std::collections::HashMap<u32, Vec<LocalId>>,
) {
    for s in &b.stmts { harvest_stmt(s, state, summaries, out); }
}

fn harvest_stmt(
    s: &HStmt,
    state: &mut Vec<bool>,
    summaries: &[bool],
    out: &mut std::collections::HashMap<u32, Vec<LocalId>>,
) {
    match s {
        HStmt::Let { local, init, .. } => {
            harvest_expr(init, state, summaries, out);
            state[local.0 as usize] = expr_nonnull_flow(init, state, summaries);
        }
        HStmt::Assign { place, value, .. } => {
            harvest_expr(place, state, summaries, out);
            harvest_expr(value, state, summaries, out);
            if let HExprKind::Local(id) = &place.kind {
                state[id.0 as usize] = expr_nonnull_flow(value, state, summaries);
            }
        }
        HStmt::ExprStmt(e) => harvest_expr(e, state, summaries, out),
        HStmt::Return { value: Some(v), .. } => harvest_expr(v, state, summaries, out),
        HStmt::Return { value: None, .. } => {}
        HStmt::If { cond, then_b, else_b, .. } => {
            harvest_expr(cond, state, summaries, out);
            let (then_nn, else_nn) = detect_null_narrow(cond);
            let mut then_state = state.clone();
            if let Some(id) = then_nn { then_state[id.0 as usize] = true; }
            harvest_block(then_b, &mut then_state, summaries, out);
            let mut else_state = state.clone();
            if let Some(id) = else_nn { else_state[id.0 as usize] = true; }
            if let Some(eb) = else_b {
                harvest_block(eb, &mut else_state, summaries, out);
            }
            for i in 0..state.len() {
                state[i] = then_state[i] && else_state[i];
            }
        }
        HStmt::While { cond, body, .. } => {
            harvest_expr(cond, state, summaries, out);
            let mut body_state = state.clone();
            harvest_block(body, &mut body_state, summaries, out);
            for i in 0..state.len() { state[i] = state[i] && body_state[i]; }
        }
        HStmt::Block(b) | HStmt::Unsafe(b, _) => harvest_block(b, state, summaries, out),
        HStmt::ForC { init, cond, step, body, .. } => {
            harvest_stmt(init, state, summaries, out);
            harvest_expr(cond, state, summaries, out);
            harvest_stmt(step, state, summaries, out);
            let mut body_state = state.clone();
            harvest_block(body, &mut body_state, summaries, out);
            for i in 0..state.len() { state[i] = state[i] && body_state[i]; }
        }
        HStmt::ForEach { src, body, .. } => {
            harvest_expr(src, state, summaries, out);
            let mut body_state = state.clone();
            harvest_block(body, &mut body_state, summaries, out);
            for i in 0..state.len() { state[i] = state[i] && body_state[i]; }
        }
        HStmt::Propagate { value: Some(v), .. } => harvest_expr(v, state, summaries, out),
        HStmt::Propagate { value: None, .. } | HStmt::Break { .. } | HStmt::Continue { .. } => {}
    }
}

fn harvest_expr(
    e: &HExpr,
    state: &[bool],
    summaries: &[bool],
    out: &mut std::collections::HashMap<u32, Vec<LocalId>>,
) {
    match &e.kind {
        HExprKind::Closure { lifted, env_values, capture_lids, .. } => {
            for (i, v) in env_values.iter().enumerate() {
                harvest_expr(v, state, summaries, out);
                if let Some(lid) = capture_lids.get(i) {
                    if expr_nonnull_flow(v, state, summaries) {
                        out.entry(lifted.0).or_default().push(*lid);
                    }
                }
            }
        }
        HExprKind::Bin { lhs, rhs, .. } => {
            harvest_expr(lhs, state, summaries, out);
            harvest_expr(rhs, state, summaries, out);
        }
        HExprKind::Un { expr, .. } => harvest_expr(expr, state, summaries, out),
        HExprKind::Unwrap { expr, .. } => harvest_expr(expr, state, summaries, out),
        HExprKind::AddrOfRef { place, .. } => harvest_expr(place, state, summaries, out),
        HExprKind::Field { base, .. } => harvest_expr(base, state, summaries, out),
        HExprKind::Index { base, idx } => { harvest_expr(base, state, summaries, out); harvest_expr(idx, state, summaries, out); }
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } | HExprKind::CallIndirect { args, .. } => {
            for a in args { harvest_expr(a, state, summaries, out); }
        }
        HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } => harvest_expr(expr, state, summaries, out),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => {
            for (_, fe) in fields { harvest_expr(fe, state, summaries, out); }
        }
        HExprKind::ArrayLit(elems) => for el in elems { harvest_expr(el, state, summaries, out); },
        HExprKind::DropWrite(inner) | HExprKind::DerefRef(inner) | HExprKind::HeapAlloc(inner)
            | HExprKind::Free(inner, _) | HExprKind::Transfer(inner) | HExprKind::SliceLen(inner)
            | HExprKind::EnumTag(inner) => harvest_expr(inner, state, summaries, out),
        HExprKind::ArrayToSlice { base, .. } => harvest_expr(base, state, summaries, out),
        HExprKind::Match { scrutinee, arms, .. } => {
            harvest_expr(scrutinee, state, summaries, out);
            for arm in arms {
                if let Some(g) = &arm.guard { harvest_expr(g, state, summaries, out); }
                harvest_block(&arm.body, &mut state.to_vec(), summaries, out);
            }
        }
        _ => {}
    }
}

/// Detect a `Local cmp null` shape and return which branch narrows it to non-null:
/// `p != null`  → then-branch narrows `p`
/// `p == null`  → else-branch narrows `p`
fn detect_null_narrow(cond: &HExpr) -> (Option<LocalId>, Option<LocalId>) {
    let HExprKind::Bin { op, lhs, rhs } = &cond.kind else { return (None, None); };
    let local_id = match (&lhs.kind, &rhs.kind) {
        (HExprKind::Local(id), HExprKind::LitNull) | (HExprKind::LitNull, HExprKind::Local(id)) => *id,
        _ => return (None, None),
    };
    match op {
        HBinOp::Ne => (Some(local_id), None),
        HBinOp::Eq => (None, Some(local_id)),
        _ => (None, None),
    }
}

/// A normalized non-null narrowing for a *place* that is not a bare local -
/// `xs[0]`, `s.field`, `xs[0].p` - keyed structurally so a guard `if (P != null)`
/// and a later deref `P!` match by identity.  Bare locals are handled by the
/// per-local `narrowed_until` flag; this handles the projected-place cases the
/// slot-based narrowing cannot (see SPEC 17).
#[derive(Clone, PartialEq, Eq)]
enum IdxKey {
    Const(i64),
    Local(u32),
}

#[derive(Clone, PartialEq, Eq)]
enum PlaceKey {
    Local(u32),
    Field(Box<PlaceKey>, usize),
    Index(Box<PlaceKey>, IdxKey),
}

/// An active place-narrowing fact: `key` is provably non-null while in scope.
struct PlaceFact {
    key: PlaceKey,
    /// Block depth this fact is scoped to; dropped when that block exits.
    depth: u32,
    /// Locals whose mutation could change the place (its root container and any
    /// index variables).  Touching any of them invalidates the fact.
    deps: Vec<u32>,
}

/// Detect a projected-place null guard: `P != null` (`want_ne`) or `P == null`
/// (`!want_ne`) where `P` is a Field/Index place (bare locals are handled by the
/// per-local narrowing path, so they are deliberately excluded here).
fn detect_place_narrow(cond: &HExpr, want_ne: bool) -> Option<(PlaceKey, Vec<u32>)> {
    let HExprKind::Bin { op, lhs, rhs } = &cond.kind else { return None };
    if !matches!((op, want_ne), (HBinOp::Ne, true) | (HBinOp::Eq, false)) {
        return None;
    }
    let place = match (&lhs.kind, &rhs.kind) {
        (_, HExprKind::LitNull) => lhs.as_ref(),
        (HExprKind::LitNull, _) => rhs.as_ref(),
        _ => return None,
    };
    if !matches!(place.kind, HExprKind::Field { .. } | HExprKind::Index { .. }) {
        return None;
    }
    // Only OWNING places (`own *T` / `own &T`).  An owning element's pointee lives
    // exactly as long as the element holds it, and every release path (move-out,
    // reassign, container drop, an intervening call) invalidates the fact - so the
    // narrowing stays sound.  A non-owning `*T`/`raw *T` element aliases memory
    // owned elsewhere that could be freed without touching this container (a UAF
    // we cannot see), and unlike a bare-local alias there is no dep edge to catch
    // it - so we refuse to narrow those.
    if !matches!(place.ty, HType::OwnPtr { .. } | HType::Heap { .. }) {
        return None;
    }
    place_key(place)
}

/// Normalize a narrowable place expression to a `(key, dep-locals)` pair, or
/// `None` if the expression is not a stable projected place rooted at a local
/// (e.g. it indexes by a computed expression, or derefs a pointer mid-path -
/// both of which we cannot cheaply prove stable, so we refuse to narrow them).
fn place_key(e: &HExpr) -> Option<(PlaceKey, Vec<u32>)> {
    match &e.kind {
        HExprKind::Local(id) => Some((PlaceKey::Local(id.0), vec![id.0])),
        HExprKind::Field { base, field } => {
            let (bk, deps) = place_key(base)?;
            Some((PlaceKey::Field(Box::new(bk), *field), deps))
        }
        HExprKind::Index { base, idx } => {
            let (bk, mut deps) = place_key(base)?;
            let ik = match &idx.kind {
                HExprKind::LitInt(n) => IdxKey::Const(*n),
                HExprKind::Local(id) => { deps.push(id.0); IdxKey::Local(id.0) }
                // A computed index (`xs[i+1]`, `xs[f()]`) is not tracked: proving
                // it unchanged between guard and deref needs value analysis we
                // don't do, so refuse rather than narrow unsoundly.
                _ => return None,
            };
            Some((PlaceKey::Index(Box::new(bk), ik), deps))
        }
        _ => None,
    }
}

struct Analyzer<'a> {
    #[allow(dead_code)]
    sym: &'a SymTab,
    f: *mut HFunc,
    /// One LocalState per LocalId.
    state: Vec<LocalState>,
    /// Active non-null narrowings for projected places (`xs[0]`, `s.p`).  A
    /// depth-scoped stack, separate from `state` so it needs no snapshotting:
    /// facts are pushed on entering a guard branch and dropped when that block
    /// exits or a dep local / any call could have changed the place.
    place_facts: Vec<PlaceFact>,
    errors: Vec<SemaError>,
    /// Non-fatal diagnostics — surfaced when an auto-nulled pointer is observed
    /// at a use site without intervening re-assignment on every code path.
    warnings: Vec<SemaWarning>,
    /// Interprocedural "never-returns-null" facts, indexed by `FuncId.0`.
    /// Populated by `compute_return_summaries` before the per-func pass runs.
    /// `expr_nonnull` consults this on `Call` / `InlineCall` arms.
    summaries: &'a [bool],
    /// Interprocedural borrow-provenance, indexed by `FuncId.0`: for each
    /// function whose return type is a borrowed view, the set of parameter
    /// indices its return value may alias.  Populated by
    /// `compute_return_borrows`.  The escape walk consults this to follow a
    /// borrowed return back across a call to the argument it aliases.
    ret_borrows: &'a [Vec<bool>],
    /// Closure locals (`let f = ...[caps]...;`) that transitively borrow-capture
    /// a local owning value the function frees on exit.  Returning or storing
    /// such a closure beyond the function would dangle (the captured owner is
    /// freed at scope exit while the escaped closure still points at it), so the
    /// escape walk rejects it - mirrors the `&local` escape rule for the borrow a
    /// capture really is.
    closure_holds_dying: std::collections::HashSet<LocalId>,
    /// Block nesting depth currently being walked.  Tracked as a field (set at
    /// every `walk_block` entry, restored on exit) so the expression walker can
    /// recover it when it must descend a `match` arm body - a statement block
    /// living inside an expression, which `walk_expr` reaches without the `depth`
    /// argument the statement walkers thread.
    cur_depth: u32,
    /// `*T` aliases auto-nulled by an owner reassign/move during the current
    /// statement's walk.  walk_block flushes these into `HBlock.stmt_nulls[i]`
    /// after statement `i`, so codegen emits a runtime `alias = NULL` right after
    /// it (making a guarded deref honest).  Drained per statement.
    pending_stmt_nulls: Vec<LocalId>,
}

impl<'a> Analyzer<'a> {
    fn new(sym: &'a SymTab, f: &mut HFunc, summaries: &'a [bool], ret_borrows: &'a [Vec<bool>]) -> Self {
        let n = f.locals.len();
        let closure_holds_dying = compute_closure_holds_dying(sym, f);
        Self {
            sym,
            f: f as *mut _,
            state: (0..n).map(|_| LocalState::fresh()).collect(),
            place_facts: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            summaries,
            ret_borrows,
            closure_holds_dying,
            cur_depth: 0,
            pending_stmt_nulls: Vec::new(),
        }
    }

    fn f(&self) -> &HFunc { unsafe { &*self.f } }
    fn f_mut(&self) -> &mut HFunc { unsafe { &mut *self.f } }

    fn err(&mut self, msg: impl Into<String>, span: Span) {
        self.errors.push(SemaError { msg: msg.into(), span });
    }

    fn warn(&mut self, msg: impl Into<String>, span: Span) {
        self.warnings.push(SemaWarning { msg: msg.into(), span });
    }

    /// Walk a block. `live_locals` is a stack of locals declared in enclosing scopes (most recent last);
    /// we use it to compute deps deaths at scope end.
    fn analyze_block(&mut self, parent_live: &mut Vec<LocalId>, depth: u32) -> Result<(), Vec<SemaError>> {
        // We don't recurse on a stored block reference here — analyze_block is called
        // from analyze_block_at via raw pointer style. Instead, the top-level driver
        // is just analyze_body; per-block walk happens in `walk_block`.
        // Keeping this as a placeholder.
        let _ = (parent_live, depth);
        // Walk the function body.
        let body_ptr = &mut self.f_mut().body as *mut HBlock;
        let body = unsafe { &mut *body_ptr };
        self.walk_block(body, &mut Vec::new(), 0);
        // Take errors out for the caller.
        if self.errors.is_empty() { Ok(()) } else { Err(std::mem::take(&mut self.errors)) }
    }

    fn walk_block(&mut self, b: &mut HBlock, live_outer: &mut Vec<LocalId>, depth: u32) {
        let prev_depth = self.cur_depth;
        self.cur_depth = depth;
        let mut declared_here: Vec<LocalId> = Vec::new();
        // Locals that have been narrowed by an early-exit guard like
        // `if (p == null) { return; }` — they revert to nullable at block exit.
        let mut guarded_here: Vec<LocalId> = Vec::new();

        // Per-statement runtime auto-nulls for THIS block.  Isolate the pending
        // buffer from any in-flight nulls of a PARENT statement (save/restore), so
        // a nested block never steals or drops them.  Reset the slots so the
        // loop-fixpoint re-walk is idempotent.
        let saved_pending = std::mem::take(&mut self.pending_stmt_nulls);
        b.stmt_nulls = vec![Vec::new(); b.stmts.len()];
        for i in 0..b.stmts.len() {
            self.walk_stmt(&mut b.stmts[i], &mut declared_here, live_outer, depth);
            if !self.pending_stmt_nulls.is_empty() {
                b.stmt_nulls[i] = std::mem::take(&mut self.pending_stmt_nulls);
            }
            if let Some(p) = self.detect_guard_return(&b.stmts[i]) {
                if self.state[p.0 as usize].narrowed_until.is_none() {
                    self.state[p.0 as usize].narrowed_until = Some(depth);
                    guarded_here.push(p);
                }
            }
        }
        self.pending_stmt_nulls = saved_pending;
        // Revert early-exit narrowing at block exit.
        for p in &guarded_here {
            self.state[p.0 as usize].narrowed_until = None;
        }

        // Scope exit: kill all LIDs declared here in reverse order.
        let mut collapsed: Vec<LocalId> = Vec::new();
        for id in declared_here.iter().rev() {
            // Scope-exit: auto-null aliases (codegen emits the runtime null below).
            let nulled = self.kill_lid(*id, b.span, false);
            for n in nulled { if !collapsed.contains(&n) { collapsed.push(n); } }
        }
        // Only null pointers that were declared OUTSIDE this scope (otherwise they're going away anyway).
        let outer_collapsed: Vec<LocalId> = collapsed.into_iter()
            .filter(|p| !declared_here.contains(p))
            .collect();
        b.ptr_nulls = outer_collapsed;
        // Locals declared in THIS block are dead once it exits (they no longer
        // exist in the emitted C).  Clear their alias deps so a PARENT scope's
        // later owner drop / reassign / move cannot collapse-null them out of
        // scope - e.g. a `*T` alias of a Vec element declared in a loop body must
        // not be nulled when the Vec is freed at the enclosing function's exit.
        for id in &declared_here {
            self.state[id.0 as usize].deps.clear();
        }
        let _ = depth;
        let _ = live_outer;
        self.cur_depth = prev_depth;
    }

    /// Walk a loop body with a move-state fixpoint: a value moved by the body is
    /// moved again on the next iteration, so after a first (silent) pass we fold
    /// the body's moves into the entry state and re-walk, surfacing a
    /// cross-iteration re-move / use-after-move that a single pass misses.  A
    /// value re-assigned before the move on each iteration (`while { x = alloc;
    /// eat(x); }`) is re-lived on the second pass, so it is correctly accepted.
    /// `depth` is the body's depth; `narrow` is a pointer proven non-null by the
    /// loop condition (narrowed inside the body).
    fn walk_loop_body(&mut self, body: &mut HBlock, live_outer: &mut Vec<LocalId>, depth: u32, narrow: Option<LocalId>) {
        self.walk_loop_body_rebound(body, live_outer, depth, narrow, None)
    }

    /// `rebound`: a loop binding (the for-each variable) that is FRESHLY assigned
    /// from the iterator at the top of every iteration.  Its move / poison /
    /// dying-borrow state must therefore reset each iteration and must NOT be
    /// folded into the cross-iteration entry state - otherwise consuming it in the
    /// body (e.g. `for (*Thread t in ts) { join(t); }`, which moves `t`) would make
    /// the next iteration see it as already-moved and wrongly reject the re-read.
    fn walk_loop_body_rebound(&mut self, body: &mut HBlock, live_outer: &mut Vec<LocalId>, depth: u32, narrow: Option<LocalId>, rebound: Option<LocalId>) {
        if let Some(r) = rebound { self.state[r.0 as usize] = LocalState::fresh(); }
        let pre = self.snapshot();
        let err_mark = self.errors.len();
        if let Some(p) = narrow { self.state[p.0 as usize].narrowed_until = Some(depth); }
        self.walk_block(body, live_outer, depth);
        if let Some(p) = narrow { self.state[p.0 as usize].narrowed_until = None; }
        let post = self.snapshot();
        // Re-walk if the body introduced any cross-iteration INVALIDATION that
        // re-enters at the top of the next iteration: a move (re-move/use-after-
        // move), OR a borrow invalidation that persists to the body end - an alias
        // poisoned by reassigning/moving its owner in the loop, or a binding that
        // now stashes a dying borrow.  A single pass checks the body against the
        // clean pre-loop state and misses the iteration-2+ dangling use (heap-UAF).
        // (auto_nulled is NOT folded: it is runtime-backed - codegen nulls the
        // pointer - so a guarded cross-iteration use is sound.)  Pass 2 is a
        // superset of pass 1's diagnostics, so discard pass-1 errors / capture
        // facts to avoid duplicates.
        let rebound_idx = rebound.map(|r| r.0 as usize);
        let new_invalidation = (0..self.state.len()).any(|i|
            Some(i) != rebound_idx && (
            (post[i].moved && !pre[i].moved)
            || (post[i].poisoned && !pre[i].poisoned)
            || (post[i].holds_dying_borrow && !pre[i].holds_dying_borrow)));
        if new_invalidation {
            self.errors.truncate(err_mark);
            let mut entry = pre;
            for i in 0..entry.len() {
                // The rebound loop variable is re-assigned at iteration top, so it
                // carries no invalidation forward - keep it fresh for pass 2.
                if Some(i) == rebound_idx { entry[i] = LocalState::fresh(); continue; }
                if post[i].moved { entry[i].moved = true; }
                if post[i].poisoned {
                    entry[i].poisoned = true;
                    entry[i].known_nonnull = false;
                    entry[i].narrowed_until = None;
                }
                if post[i].holds_dying_borrow { entry[i].holds_dying_borrow = true; }
            }
            self.restore(entry);
            if let Some(p) = narrow { self.state[p.0 as usize].narrowed_until = Some(depth); }
            self.walk_block(body, live_outer, depth);
            if let Some(p) = narrow { self.state[p.0 as usize].narrowed_until = None; }
        }
    }

    fn walk_stmt(&mut self, s: &mut HStmt, declared_here: &mut Vec<LocalId>, live_outer: &mut Vec<LocalId>, depth: u32) {
        match s {
            HStmt::Let { local, init, span } => {
                self.walk_expr(init);
                // Compute deps for init.  When the destination is a non-owning
                // slot (`*T`, `&T`), also fold in the LIDs of any `own *T` /
                // `own &T` Locals reachable through the init expression — those
                // are the owners that the new binding is aliasing, and we need
                // dep edges to them so the Assign-trigger / scope-exit
                // `kill_lid` can invalidate this alias if any of them mutates.
                let id = *local;
                let li_ty = self.f().locals[id.0 as usize].ty.clone();
                let mut init_deps = self.expr_deps(init);
                if matches!(li_ty, HType::Ptr { .. } | HType::Ref { .. }) {
                    self.collect_owner_aliases(init, &mut init_deps);
                }
                let init_nonnull = self.expr_nonnull(init);
                self.state[id.0 as usize].deps = init_deps;
                self.state[id.0 as usize].moved = false;
                self.state[id.0 as usize].poisoned = false;
                self.state[id.0 as usize].known_nonnull = init_nonnull;
                // Fresh local: it was never auto-nulled.  A re-bound name shadows
                // the prior state; explicit init beats any historical collapse.
                self.state[id.0 as usize].auto_nulled = false;
                // Track whether this binding stashes a borrow of a dying local, so
                // returning it later is caught as the use-after-free it is.
                let dying = self.expr_is_dying_borrow(init);
                self.state[id.0 as usize].holds_dying_borrow = dying;
                declared_here.push(id);

                // Move semantics: binding a bare owning Local (`own *T`/`heap T`
                // or an owning VALUE like String/Vec/owning struct) moves it.
                let _ = li_ty;
                if let HExprKind::Local(src) = init.kind {
                    if ty_owns_heap(self.sym, &init.ty) {
                        let sp = *span;
                        self.mark_moved(src, sp);
                    }
                } else if ty_owns_heap(self.sym, &init.ty)
                    && matches!(init.kind, HExprKind::Field { .. } | HExprKind::Index { .. })
                {
                    // `own *T e = xs[0];` moves the element out and auto-nulls the
                    // source slot at runtime, so a prior `xs[0] != null` narrowing
                    // is now stale - invalidate it (else `xs[0]!` would deref null).
                    if let Some(r) = root_local(init) { self.invalidate_place_facts(r.0); }
                }
            }
            HStmt::Assign { op, place, value, drop_old, span } => {
                let _ = op;
                // Conservative escape check on assigning into a struct field:
                // (a) explicit `b.p = &local` — caught by check_no_local_ref_escape;
                // (b) `b.p = borrow_param` — without lifetime annotations we don't
                //     know if `borrow_param`'s source outlives `b`.  When the place
                //     reaches back to a parameter, `b` survives the call, so the
                //     stash would escape.  Reject conservatively.
                // A GlobalRef place (`g = &local`) stores into a module global,
                // which OUTLIVES every function local, so a borrow of a local
                // escaping into it dangles forever - escape-check it like the
                // field-store case.  (A global is never auto-nulled by the
                // scope-exit kill_lid path, so the dangle persists.)
                if matches!(place.kind, HExprKind::Field { .. } | HExprKind::Index { .. } | HExprKind::Unwrap { .. } | HExprKind::GlobalRef(_)) {
                    self.check_no_local_ref_escape(value, false);
                    let place_root_is_param = root_local(place).map(|id|
                        matches!(self.f().locals[id.0 as usize].storage, StorageClass::Param)
                    ).unwrap_or(false);
                    let value_is_borrow = matches!(value.ty, HType::Ref { .. });
                    if place_root_is_param && value_is_borrow {
                        self.err(
                            "storing a borrow into a struct field reachable through a parameter would escape the borrow's lifetime — without lifetime annotations the compiler can't prove the borrow's source outlives the struct".to_string(),
                            *span,
                        );
                    }
                }
                // Bare-Local on the LHS is a write, not a read — don't fire the auto-null
                // warning on it.  For complex places (Field/Index/Unwrap), the inner reads
                // should still emit normally.
                if !matches!(place.kind, HExprKind::Local(_)) {
                    self.walk_expr(place);
                }
                self.walk_expr(value);
                // A bare owning Local on the RHS is moved by the assignment.  A
                // conditional move (e.g. `if (k != null) { nk[i] = k; }`) no
                // longer hard-errors at the branch join - join_branches auto-nulls
                // it per SPEC 6.4 - so it is safe to flag the move here too; a
                // straight-line `x = y; y.use()` is then a proper use-after-move.
                self.mark_owning_move(value);
                // The write (to `xs = ..`, `xs[i] = ..`, `s.f = ..`, or an index
                // variable `i = ..`) can change any narrowed place rooted at the
                // assigned local, so invalidate those facts.  Done after walking
                // both sides, so a deref in `value`/`place` is still proven.
                if let Some(r) = root_local(place) {
                    self.invalidate_place_facts(r.0);
                }
                // Reassigning a pointer-like or borrow-bearing local overwrites
                // its deps (§3.8).  Pointers (`HType::Ptr`) participate in the
                // null-collapse path; anything else (refs, structs that contain
                // refs, etc.) just refreshes deps and clears poison.
                if let HExprKind::Local(id) = place.kind {
                    let ty = self.f().locals[id.0 as usize].ty.clone();
                    let is_ptr = matches!(ty, HType::Ptr { .. });
                    let is_owner = matches!(ty, HType::OwnPtr { .. } | HType::Heap { .. });
                    // The local is freshly assigned, so it is live again even if
                    // the RHS consumed its previous value (`root = ins(root, k)`):
                    // `walk_expr(value)` above marked the *old* value moved; the
                    // assignment re-lives the binding.
                    self.state[id.0 as usize].moved = false;
                    // Downstream invalidation: if the LHS is an owner whose
                    // pointee is being replaced (or null-assigned), every live
                    // alias of the OLD pointee is now stale.  Reuse the same
                    // kill_lid path the scope-exit case uses: `*T` aliases get
                    // auto-NULLed in flow state (deref needs fresh proof);
                    // `&T` borrows get poisoned (use is a compile error).
                    // The owner itself is unconstrained — kill_lid skips its
                    // own LID, so its state is refreshed normally below.
                    if is_owner {
                        // Reassigning the owner frees its OLD pointee (drop_old),
                        // so its `*T` aliases are stale: AUTO-NULL them and emit a
                        // runtime `alias = NULL` right after this statement (via
                        // stmt_nulls) so a later `if (a != null)` guard is honest.
                        let n = self.kill_lid(id, *span, true);
                        self.pending_stmt_nulls.extend(n);
                    }
                    if is_ptr {
                        // `*T` slot: also fold in owner-aliases reachable from
                        // the RHS so a fresh alias relationship is tracked.
                        let mut d = self.expr_deps(value);
                        self.collect_owner_aliases(value, &mut d);
                        let nn = self.expr_nonnull(value);
                        self.state[id.0 as usize].deps = d;
                        self.state[id.0 as usize].known_nonnull = nn;
                        self.state[id.0 as usize].narrowed_until = None;
                        self.state[id.0 as usize].auto_nulled = false;
                        // Re-pointing the alias to a fresh source is the sound
                        // recovery from an invalidation - clear poison too (an
                        // owner reassign/move poisons aliases; this re-establishes
                        // a valid pointee), mirroring the auto_nulled clear above.
                        self.state[id.0 as usize].poisoned = false;
                    } else if is_owner {
                        // Reassigning an `own *T` / heap owner re-establishes its
                        // nullness from the RHS (`alloc` => non-null, `null` =>
                        // null).  This is what clears an auto-null left by a prior
                        // conditional move (SPEC 6.4: only an explicit user
                        // assignment clears it / re-proves non-null), so a deref
                        // after re-setting on every path is accepted again while a
                        // partial re-set still fails the join's `known_nonnull`.
                        let nn = self.expr_nonnull(value);
                        let d = self.expr_deps(value);
                        self.state[id.0 as usize].deps = d;
                        self.state[id.0 as usize].known_nonnull = nn;
                        self.state[id.0 as usize].narrowed_until = None;
                        self.state[id.0 as usize].auto_nulled = false;
                        self.state[id.0 as usize].poisoned = false;
                    } else {
                        // For ref-shaped, owner-bearing, or borrow-bearing
                        // struct locals, refresh the deps so subsequent
                        // kill_lid cycles target the right source set.  Clear
                        // poison so the new value is usable.  For `&T` / `&mut T`
                        // also fold in owner-aliases (a borrow of `head.field`
                        // depends on the chain rooted at `head`).
                        let mut d = self.expr_deps(value);
                        if matches!(ty, HType::Ref { .. }) {
                            self.collect_owner_aliases(value, &mut d);
                        }
                        self.state[id.0 as usize].deps = d;
                        self.state[id.0 as usize].poisoned = false;
                    }
                    // Re-assignment refreshes whether the binding stashes a dying
                    // borrow: a fresh non-dying source clears it, a dying one sets it.
                    let dying = self.expr_is_dying_borrow(value);
                    self.state[id.0 as usize].holds_dying_borrow = dying;
                }
                // Reassigning an OWNING-POINTER field/element (`b.p = ...` /
                // `arr[i] = ...` where the field/element is `own *T` / `own &T`)
                // frees the OLD pointee, so a *T alias of it - which depends on the
                // container root (recorded by collect_owner_aliases) - now dangles.
                // AUTO-NULL the container's *T aliases and emit a runtime
                // `alias = NULL` after this statement (via stmt_nulls).
                //
                // Keyed strictly on the place being an `own *T`/`own &T`
                // (HType::OwnPtr): ONLY an owning pointer owns heap that gets
                // freed here.  A place is an lvalue, so its type is the true
                // field/element type.  `drop_old` alone is wrong - it can be set
                // for a non-owning element (`v[i] = char`), which frees nothing and
                // must NOT kill a `string`/borrow view of the container.
                let _ = drop_old;
                if matches!(place.kind, HExprKind::Field { .. } | HExprKind::Index { .. })
                    && matches!(place.ty, HType::OwnPtr { .. })
                {
                    if let Some(root) = root_local(place) {
                        let n = self.kill_lid(root, *span, true);
                        self.pending_stmt_nulls.extend(n);
                    }
                }
                let _ = span;
            }
            HStmt::ExprStmt(e) => self.walk_expr(e),
            HStmt::Return { value, heap_drops, span } => {
                if let Some(v) = value { self.walk_expr(v); }
                // Move-on-return: `return a;` where `a` is a heap binding.
                if let Some(v) = value {
                    if matches!(v.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        if let HExprKind::Local(id) = v.kind { self.mark_moved(id, *span); }
                    }
                    // UAF check: returning ANY value that embeds `&local` would
                    // dangle as soon as the caller dereferences the borrow.  Covers
                    // both `return &x;` directly and the escape-via-struct-field
                    // case `return Container { p = &x };`.
                    self.check_no_local_ref_escape(v, false);
                }
                let _ = heap_drops;
            }
            HStmt::If { cond, then_b, else_b, .. } => {
                self.walk_expr(cond);
                // narrowing: detect `p != null` / `null != p` for an immediate
                // Local(p), and `P != null` for a projected place P (`xs[0]`, `s.p`).
                let then_narrow = self.detect_not_null_narrow(cond);
                let else_narrow = self.detect_is_null_narrow(cond);
                let then_place = detect_place_narrow(cond, true);
                let else_place = detect_place_narrow(cond, false);

                // Snapshot state for branch join
                let snap = self.snapshot();

                // then branch
                if let Some(p) = then_narrow {
                    self.state[p.0 as usize].narrowed_until = Some(depth + 1);
                }
                if let Some((k, deps)) = &then_place {
                    self.place_facts.push(PlaceFact { key: k.clone(), depth: depth + 1, deps: deps.clone() });
                }
                self.walk_block(then_b, live_outer, depth + 1);
                if let Some(p) = then_narrow {
                    self.state[p.0 as usize].narrowed_until = None;
                }
                // Drop place-facts scoped to this branch (this guard's, plus any
                // from nested guards inside it that weren't already invalidated).
                self.place_facts.retain(|f| f.depth <= depth);
                let then_state = self.snapshot();
                self.restore(snap.clone());

                // else branch
                if let Some(b) = else_b {
                    if let Some(p) = else_narrow {
                        self.state[p.0 as usize].narrowed_until = Some(depth + 1);
                    }
                    if let Some((k, deps)) = &else_place {
                        self.place_facts.push(PlaceFact { key: k.clone(), depth: depth + 1, deps: deps.clone() });
                    }
                    self.walk_block(b, live_outer, depth + 1);
                    if let Some(p) = else_narrow {
                        self.state[p.0 as usize].narrowed_until = None;
                    }
                    self.place_facts.retain(|f| f.depth <= depth);
                }
                let else_state = self.snapshot();

                // Join: a heap binding moved in both branches is moved after.
                // Inconsistent move state is an error per §11.7 — but only for
                // bindings that are *live at the join*.  A branch that always
                // exits (return/break/continue/propagate/panic) never reaches the
                // join, so its moves don't count; and a binding declared *inside*
                // a branch is out of scope after the `if`, so its move state is
                // irrelevant.  In-scope-at-the-if = enclosing scopes (`live_outer`)
                // plus this block's locals declared before the `if`.
                let then_exits = block_always_exits(then_b);
                let else_exits = else_b.as_ref().map_or(false, |b| block_always_exits(b));
                if then_exits && !else_exits {
                    self.restore(else_state);
                } else if else_exits && !then_exits {
                    self.restore(then_state);
                } else if then_exits && else_exits {
                    // both diverge: code after the `if` is unreachable
                    self.restore(else_state);
                } else {
                    let in_scope: std::collections::HashSet<u32> = live_outer.iter()
                        .chain(declared_here.iter()).map(|l| l.0).collect();
                    self.join_branches(&then_state, &else_state, &in_scope);
                }
            }
            HStmt::While { cond, body, span } => {
                let _ = span;
                self.walk_expr(cond);
                // narrow inside the body when the condition is `p != null`
                let body_narrow = self.detect_not_null_narrow(cond);
                self.walk_loop_body(body, live_outer, depth + 1, body_narrow);
            }
            HStmt::Block(b) => self.walk_block(b, live_outer, depth + 1),
            HStmt::Unsafe(b, _) => self.walk_block(b, live_outer, depth + 1),
            HStmt::Break { .. } | HStmt::Continue { .. } => {}
            HStmt::ForC { init, cond, step, body, .. } => {
                self.walk_stmt(init, declared_here, live_outer, depth);
                self.walk_expr(cond);
                self.walk_stmt(step, declared_here, live_outer, depth);
                self.walk_loop_body(body, live_outer, depth + 1, None);
            }
            HStmt::ForEach { var, src, body, .. } => {
                self.walk_expr(src);
                self.walk_loop_body_rebound(body, live_outer, depth + 1, None, Some(*var));
            }
            HStmt::Propagate { value: Some(v), .. } => self.walk_expr(v),
            HStmt::Propagate { value: None, .. } => {}
        }
    }

    /// Detect the shape `if (p == null) { return ...; }` (or `break/continue/propagate`)
    /// at a sibling statement; the rest of the enclosing block sees `p` as non-null.
    fn detect_guard_return(&self, stmt: &HStmt) -> Option<LocalId> {
        if let HStmt::If { cond, then_b, else_b, .. } = stmt {
            // Only one-sided guards count — an `else` would let us fall through with p still nullable.
            if else_b.is_some() { return None; }
            let p = self.detect_is_null_narrow(cond)?;
            if block_always_exits(then_b) {
                return Some(p);
            }
        }
        None
    }

    fn detect_not_null_narrow(&self, cond: &HExpr) -> Option<LocalId> {
        match &cond.kind {
            HExprKind::Bin { op: HBinOp::Ne, lhs, rhs } => {
                if let (HExprKind::Local(id), HExprKind::LitNull) = (&lhs.kind, &rhs.kind) {
                    return Some(*id);
                }
                if let (HExprKind::LitNull, HExprKind::Local(id)) = (&lhs.kind, &rhs.kind) {
                    return Some(*id);
                }
            }
            _ => {}
        }
        None
    }
    fn detect_is_null_narrow(&self, cond: &HExpr) -> Option<LocalId> {
        // For else-branch of `if (p == null)`, p is non-null.
        match &cond.kind {
            HExprKind::Bin { op: HBinOp::Eq, lhs, rhs } => {
                if let (HExprKind::Local(id), HExprKind::LitNull) = (&lhs.kind, &rhs.kind) {
                    return Some(*id);
                }
                if let (HExprKind::LitNull, HExprKind::Local(id)) = (&lhs.kind, &rhs.kind) {
                    return Some(*id);
                }
            }
            _ => {}
        }
        None
    }

    fn walk_expr(&mut self, e: &mut HExpr) {
        // Set `skip_check` on Unwrap if narrowed.
        match &mut e.kind {
            HExprKind::Unwrap { expr, skip_check } => {
                self.walk_expr(expr);
                // Forced-handling rule: a `*T` deref is only permitted when the lifetime
                // pass can prove the pointer is non-null. Codegen no longer emits a runtime
                // null-check, so we must reject every unproven site here.
                let proven = self.expr_nonnull(expr);
                if proven {
                    *skip_check = true;
                } else {
                    let sp = expr.span;
                    let hint = match &expr.kind {
                        // A `mut` global can be reassigned by any other call between the
                        // guard and the deref, so a `if (G == null)` guard on the global
                        // itself can never be proven to hold across the deref.  Suggest
                        // the local-rebind pattern rather than the (already-written) guard.
                        HExprKind::GlobalRef(gid) if self.sym.globals[gid.0 as usize].is_mut => {
                            let name = self.sym.globals[gid.0 as usize].name.clone();
                            format!(
                                "cannot prove the `mut` global `{n}` is non-null here: another call \
                                 could reassign it between the guard and the deref. Copy it to a local \
                                 first, then guard and deref the local: \
                                 `<type> p = {n}; if (p == null) {{ return ...; }} unsafe {{ p!... }}`",
                                n = name
                            )
                        }
                        HExprKind::Local(id) => {
                            let name = self.f().locals[id.0 as usize].name.clone();
                            format!(
                                "cannot prove `{}` is non-null here; guard with `if ({} != null) {{ ... }}` \
                                 or exit early with `if ({} == null) {{ return ...; }}` before dereferencing",
                                name, name, name
                            )
                        }
                        _ => "cannot prove this pointer is non-null here; guard the access with \
                              `if (p != null) { ... }` first".to_string(),
                    };
                    self.err(hint, sp);
                }
            }
            HExprKind::Bin { op: _, lhs, rhs } => {
                // No null-cmp suppression: comparing an auto-nulled pointer with `null`
                // produces a tautological result whose meaning the user did not write —
                // they need to see the warning so they can re-assign before checking.
                self.walk_expr(lhs);
                self.walk_expr(rhs);
            }
            HExprKind::Un { expr, .. } => self.walk_expr(expr),
            HExprKind::AddrOfRef { place, .. } => self.walk_expr(place),
            HExprKind::Field { base, .. } => self.walk_expr(base),
            HExprKind::Index { base, idx } => { self.walk_expr(base); self.walk_expr(idx); }
            HExprKind::Call { callee, args } => {
                let spawn = is_spawn_callee(*callee);
                let reap = is_reaping_callee(*callee);
                // `spawn_on`/`job_on` (MAX-70) submit work to a Pool passed as
                // args[1]; they BORROW it (the pool outlives the fibers - its drop
                // drains+joins the workers), so it must not be moved/consumed.
                let pool_borrow_arg = if callee.0 == u32::MAX - 70 { 1usize } else { usize::MAX };
                for (i, a) in args.iter_mut().enumerate() {
                    // Detect move-in for any owning value passed by value.
                    self.walk_expr(a);
                    if i != pool_borrow_arg {
                        self.mark_owning_move(a);
                    }
                    // A closure passed to spawn/thread/job MOVES its owning
                    // captures into the thread, so using them afterwards is a
                    // use-after-move (the thread now owns and frees them).
                    if spawn {
                        if let HExprKind::Closure { env_values, .. } = &a.kind {
                            for v in env_values {
                                if ty_owns_heap(self.sym, &v.ty) {
                                    if let HExprKind::Local(id) = v.kind { self.mark_moved(id, v.span); }
                                }
                            }
                        }
                    }
                }
                // A terminal thread-handle reap (join / detach / cancel)
                // unconditionally drops the handle's spawner ref - the runtime
                // Thread struct is freed once its refcount hits zero.  So the
                // `*Thread` handle is CONSUMED: a second reap, or any later use,
                // is a heap-use-after-free / double-free.  Mark it moved (unless
                // its read above already flagged it moved, to avoid a double
                // diagnostic) so reuse is rejected at compile time.
                if reap {
                    if let Some(a) = args.first() {
                        if let Some(id) = root_local(a) {
                            if !self.state[id.0 as usize].moved {
                                self.mark_moved(id, a.span);
                            }
                        }
                    }
                }
                // push(container, element): storing a borrow of a local into a
                // container that OUTLIVES this function (one reached through a
                // parameter, or a global) leaves a dangling reference once the
                // local dies.  Escape-check the pushed element like a returned
                // borrow.  (A same-scope local container is not checked here - it
                // dies with the borrow; only a provably-outliving container is.)
                if callee.0 == u32::MAX - 60 && args.len() == 2 {
                    let container_outlives = root_local(&args[0]).map(|id|
                        matches!(self.f().locals[id.0 as usize].storage, StorageClass::Param)
                    ).unwrap_or_else(|| matches!(args[0].kind, HExprKind::GlobalRef(_)));
                    if container_outlives {
                        self.check_no_local_ref_escape(&args[1], false);
                    }
                }
                // A call can mutate any container reachable through a `&mut`
                // argument (push/pop/clear/swap...), which could change a narrowed
                // element's value, so conservatively drop all place-facts after the
                // args are evaluated (a deref passed AS an arg is already proven).
                self.place_facts.clear();
            }
            HExprKind::Cast { expr, .. } => self.walk_expr(expr),
            HExprKind::CheckedCast { expr, .. } => self.walk_expr(expr),
            // Aggregate literals move an owning value into the new aggregate (it
            // now owns it), so the source local is moved - later use is rejected.
            HExprKind::Struct { fields, .. } => for (_, fe) in fields { self.walk_expr(fe); self.mark_owning_move(fe); },
            HExprKind::ArrayLit(elems) => for e in elems { self.walk_expr(e); self.mark_owning_move(e); },
            HExprKind::DropWrite(inner) => self.walk_expr(inner),
            HExprKind::ArrayToSlice { base, .. } => self.walk_expr(base),
            HExprKind::DerefRef(inner) => self.walk_expr(inner),
            HExprKind::HeapAlloc(inner) => self.walk_expr(inner),
            HExprKind::Free(inner, _) => self.walk_expr(inner),
            HExprKind::CallIndirect { callee, args } => {
                self.walk_expr(callee);
                for a in args { self.walk_expr(a); }
                self.place_facts.clear();
            }
            HExprKind::InlineCall { args, .. } => {
                // InlineCall must mirror Call's move semantics: an `own *T`/`heap T`
                // value passed by value transfers ownership into the inline's
                // parameter local.  Without this, the outer binding stays "live"
                // and main()'s scope-exit drops the same pointer the inline's
                // expansion drops — a double-free for any inline that takes an
                // owning parameter.
                for a in args {
                    self.walk_expr(a);
                    self.mark_owning_move(a);
                }
                self.place_facts.clear();
            }
            HExprKind::Closure { env_values, .. } => {
                for v in env_values.iter_mut() {
                    self.walk_expr(v);
                }
            }
            HExprKind::Transfer(inner) => {
                // Treat the source-local as moved (use after this point is a compile error).
                self.walk_expr(inner);
                if let Some(id) = root_local(inner) {
                    let span = inner.span;
                    // mark_moved poisons the owner's aliases (single move choke
                    // point): transferring across a gate frees the value, so a *T
                    // alias of it dangles.
                    self.mark_moved(id, span);
                }
            }
            HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => self.walk_expr(inner),
            HExprKind::FnRef(_) => {}
            HExprKind::VariantCtor { fields, .. } => for (_, fe) in fields { self.walk_expr(fe); self.mark_owning_move(fe); },
            HExprKind::Match { scrutinee, arms, .. } => {
                self.walk_expr(scrutinee);
                for a in arms.iter() {
                    // A guard runs speculatively, before the arm commits, on a
                    // binding that aliases the scrutinee's payload.  Moving an
                    // owning binding out of a guard (e.g. passing it to a by-value
                    // parameter) frees a value the scrutinee still owns, so a
                    // failed guard or the scrutinee's later drop double-frees.
                    // Guards may only borrow - reject the move.
                    if let Some(g) = &a.guard {
                        if let HArmKind::Variant { bindings, .. } = &a.kind {
                            for b in bindings.iter().flatten() {
                                if ty_owns_heap(self.sym, &self.f().locals[b.0 as usize].ty)
                                    && expr_moves_owning_local(self.sym, g, *b)
                                {
                                    let name = self.f().locals[b.0 as usize].name.clone();
                                    self.err(format!("cannot move `{}` out of a match guard: a guard may only borrow the matched value (moving it would double-free, since the scrutinee still owns it)", name), g.span);
                                }
                            }
                        }
                    }
                }
                // Arms are alternatives: walk each from the post-scrutinee
                // baseline so one arm's narrowing/moves never contaminate a
                // sibling.  Crucially this descends the arm BODY (a statement
                // block living inside the match expression) through walk_block, so
                // the borrow/null/escape checks fire there too - previously the
                // body was skipped entirely, accepting unproven `*T` derefs (null
                // deref at runtime) and borrows-of-locals escaping via a returned
                // yield (stack-use-after-return).  Order matches evaluation:
                // guard, then body, then the yielded value.
                let body_depth = self.cur_depth + 1;
                let snap = self.snapshot();
                let n = self.state.len();
                let mut moved_union = vec![false; n];
                let mut poison_union = vec![false; n];
                let mut dying_union = vec![false; n];
                for a in arms.iter_mut() {
                    self.restore(snap.clone());
                    if let Some(g) = &mut a.guard.clone() { self.walk_expr(g); }
                    let mut arm_live: Vec<LocalId> = Vec::new();
                    self.walk_block(&mut a.body, &mut arm_live, body_depth);
                    if let Some(v) = &mut a.value.clone() {
                        self.walk_expr(v);
                        // A terse arm `pattern value` YIELDS its value out of the
                        // match (into the match result), so a bare owning local
                        // yielded here is MOVED - mark it, exactly like the Assign/
                        // return/struct-init move sites.  The block-`yield` and
                        // if-expression forms desugar to an Assign which already
                        // marks the move; the terse arm-value form did not, so a
                        // later use of the yielded owner was accepted (UAF).
                        self.mark_owning_move(v);
                    }
                    // A diverging arm (its body always returns/breaks/continues/
                    // propagates) never reaches the join after the match, so its
                    // invalidations must NOT propagate there - else a value it
                    // consumed looks moved on a sibling fall-through path that never
                    // touched it (false "use of moved value").  Mirrors the if/else
                    // join's `block_always_exits` divergence handling.
                    if !block_always_exits(&a.body) {
                        for (i, s) in self.state.iter().enumerate() {
                            if s.moved { moved_union[i] = true; }
                            // An arm can also INVALIDATE an enclosing borrow: moving
                            // the owner of a `*T` alias inside the arm poisons the
                            // alias, and stashing a dying borrow sets holds_dying.
                            // These must reach the post-match state too, else a use
                            // of the now-dangling alias after the match is missed
                            // (heap-UAF).
                            if s.poisoned { poison_union[i] = true; }
                            if s.holds_dying_borrow { dying_union[i] = true; }
                        }
                    }
                }
                // Post-match state: the unconditional scrutinee moves (in `snap`)
                // plus, conservatively, any invalidation on ANY fall-through arm -
                // so a use after the match is flagged if some arm consumed/
                // invalidated it.  Narrowing reverts with `snap`; clearing it on a
                // freshly-poisoned binding keeps a stale `!= null` proof from
                // re-validating a dangling pointer.
                self.restore(snap);
                for i in 0..n {
                    if moved_union[i] { self.state[i].moved = true; }
                    if dying_union[i] { self.state[i].holds_dying_borrow = true; }
                    if poison_union[i] {
                        self.state[i].poisoned = true;
                        self.state[i].known_nonnull = false;
                        self.state[i].narrowed_until = None;
                    }
                }
            }
            HExprKind::Local(id) => {
                let name = self.f().locals[id.0 as usize].name.clone();
                let st = &self.state[id.0 as usize];
                let moved = st.moved;
                let poisoned = st.poisoned;
                let auto_nulled = st.auto_nulled;
                let narrowed = st.narrowed_until.is_some();
                let known_nn = st.known_nonnull;
                let is_ptr = matches!(self.f().locals[id.0 as usize].ty, HType::Ptr { .. });
                if moved {
                    self.err(format!("use of moved value `{}`", name), e.span);
                }
                if poisoned {
                    self.err(format!("use of poisoned reference `{}`", name), e.span);
                }
                // The user wants this fired even when the access is gated by an
                // `if (p != null)` check — gating proves the runtime value is
                // non-null, but the user's *intended* value was silently overwritten.
                // They still need to know.  We do drop the warning if the user
                // explicitly re-assigned `p` since auto-null, which clears the flag
                // below in the Assign branch.
                let _ = (narrowed, known_nn);
                if is_ptr && auto_nulled {
                    self.warn(
                        format!(
                            "pointer `{}` was auto-nulled when its pointee went out of scope and \
                             has not been explicitly re-assigned on every code path since; \
                             this use observes that silent overwrite — re-assign `{}` yourself \
                             before reading it",
                            name, name
                        ),
                        e.span,
                    );
                }
            }
            _ => {}
        }
    }

    /// Walk an expression tree to catch every `&local` whose root is a
    /// function-scope stack binding — used to reject escape-via-return through
    /// any shape: direct, struct field, array element, variant payload, etc.
    fn check_no_local_ref_escape(&mut self, e: &HExpr, under_deref: bool) {
        use HExprKind::*;
        match &e.kind {
            AddrOfRef { place, .. } => {
                if let Some(root) = root_local(place) {
                    let li = &self.f().locals[root.0 as usize];
                    // A borrow rooted in a non-owning reference parameter
                    // (`&T` / `*T` / `raw *T`) targets the caller's data, which
                    // outlives the call, so returning it is sound (cf. Rust's
                    // elided `fn(&Box) -> &int`).  Stack locals, value parameters,
                    // and OWNING parameters (`own &T` / `own *T`, whose referent
                    // the function frees on exit) all die on return - reject those.
                    let escapes = match li.storage {
                        StorageClass::Stack => true,
                        StorageClass::Param => !matches!(
                            li.ty,
                            HType::Ref { .. } | HType::Ptr { .. } | HType::RawPtr { .. }
                        ),
                        _ => false,
                    };
                    if escapes {
                        let name = li.name.clone();
                        let span = e.span;
                        self.err(
                            format!(
                                "reference to local `{}` escapes its scope (via a return, a store into a global, or a push into a longer-lived container) — the local dies here, so the reference would dangle",
                                name
                            ),
                            span,
                        );
                    }
                }
            }
            Struct { fields, .. } | VariantCtor { fields, .. } => {
                // Aggregates PRESERVE a stored borrow, so propagate under_deref.
                for (_, fe) in fields { self.check_no_local_ref_escape(fe, under_deref); }
            }
            ArrayLit(elems) => for el in elems { self.check_no_local_ref_escape(el, under_deref); }
            HeapAlloc(inner) | Free(inner, _) => self.check_no_local_ref_escape(inner, under_deref),
            // Cast / DropWrite preserve borrow-ness; a DEREF (DerefRef/Unwrap) or a
            // unary op CONSUMES the borrow into a plain value, so a holds-dying-borrow
            // local under it does not escape AS a borrow - mark under_deref.
            Cast { expr, .. } | CheckedCast { expr, .. } | DropWrite(expr) => self.check_no_local_ref_escape(expr, under_deref),
            DerefRef(expr) | Un { expr, .. } | Unwrap { expr, .. } => self.check_no_local_ref_escape(expr, true),
            // Extracting a borrow/view FIELD out of a local that holds a dying
            // borrow (`Holder h = Holder { q = &x }; return h.q;`) escapes that
            // borrow - the field may be the dying one.  `holds_dying_borrow` is set
            // on the struct local exactly when some field is a DYING borrow (a field
            // borrowing a parameter leaves it false), so this is sound; it can only
            // over-reject the exotic mixed case (one dying field + one param-borrow
            // field, returning the safe one), which is restructurable.  Reading the
            // field under a deref (`h.q!`) consumes the borrow, so skip then.
            Field { base, .. } => {
                if !under_deref
                    && matches!(e.ty, HType::Ref { .. } | HType::Ptr { .. })
                {
                    if let HExprKind::Local(id) = base.kind {
                        if self.state[id.0 as usize].holds_dying_borrow {
                            let name = self.f().locals[id.0 as usize].name.clone();
                            let span = e.span;
                            self.err(
                                format!(
                                    "a borrow field of `{}` (which holds a borrow of a local the function frees on exit) escapes via the returned value, leaving a dangling reference. Return an owned value instead",
                                    name
                                ),
                                span,
                            );
                        }
                    }
                }
                // The base is read into (not itself escaping as a borrow), so walk
                // it under_deref - this also catches a dying field nested deeper.
                self.check_no_local_ref_escape(base, true);
            }
            Bin { lhs, rhs, .. } => {
                // Arithmetic/comparison consumes the borrow into a value.
                self.check_no_local_ref_escape(lhs, true);
                self.check_no_local_ref_escape(rhs, true);
            }
            // A call whose borrowed return value aliases one of its arguments
            // (per `compute_return_borrows`): the result borrows whatever that
            // argument borrows, so follow into it.  This catches a borrow that
            // escapes through a call - e.g. `return localString.as_str()`, where
            // `as_str(&String) -> string` returns a borrow of its receiver, so
            // the result dangles on a local receiver just like `return &local;`.
            Call { callee, args } | InlineCall { callee, args, .. } => {
                let cid = callee.0 as usize;
                let borrowed: Vec<usize> = self
                    .ret_borrows
                    .get(cid)
                    .map(|cb| cb.iter().enumerate().filter(|(_, &b)| b).map(|(m, _)| m).collect())
                    .unwrap_or_default();
                for m in borrowed {
                    if let Some(a) = args.get(m) {
                        // The call re-borrows this arg into its result, preserving
                        // borrow-ness, so propagate under_deref.
                        self.check_no_local_ref_escape(a, under_deref);
                    }
                }
            }
            // Returning/storing a closure LOCAL that borrow-captures a local the
            // function frees on exit dangles: the captured owner is freed at
            // scope exit while the escaped closure still points at it.
            Local(id) => {
                if self.closure_holds_dying.contains(id) {
                    let name = self.f().locals[id.0 as usize].name.clone();
                    let span = e.span;
                    self.err(
                        format!(
                            "closure `{}` captures a local that the function frees on exit, so returning it would leave a dangling capture. Move the captured owner's ownership out, or build the value the closure needs in the caller",
                            name
                        ),
                        span,
                    );
                }
                // A local that holds a borrow/view of a dying function-local value
                // (e.g. `string v = s.as_str();`) dangles once returned - the same
                // use-after-free as `return s.as_str();`, just routed through `v`.
                // Only when the borrow VALUE escapes: a deref (`p!`) reads the
                // pointee and does not escape the borrow, so skip under_deref.
                if !under_deref && self.state[id.0 as usize].holds_dying_borrow {
                    let name = self.f().locals[id.0 as usize].name.clone();
                    let span = e.span;
                    self.err(
                        format!(
                            "`{}` holds a borrow of a local that the function frees on exit, so returning it would leave a dangling reference. Return an owned value (e.g. transfer ownership) instead of a borrow of a local",
                            name
                        ),
                        span,
                    );
                }
            }
            // The same for a closure LITERAL appearing directly in the escaping
            // value (`return unit() [b] { ... };`): each capture of a dying owner
            // (or of a closure that itself holds one) would dangle.
            Closure { env_values, .. } => {
                for v in env_values {
                    if let Some(c) = root_local(v) {
                        if local_is_dying_owner(self.sym, self.f(), c)
                            || self.closure_holds_dying.contains(&c)
                        {
                            let name = self.f().locals[c.0 as usize].name.clone();
                            let span = e.span;
                            self.err(
                                format!(
                                    "returned closure captures local `{}`, which the function frees on exit, so the caller would observe a dangling capture",
                                    name
                                ),
                                span,
                            );
                        }
                    }
                }
            }
            // A `match` used as the escaping/returned value: a borrow of a local
            // can escape through any arm's yielded value, so check each arm.  Match
            // fell into the catch-all below, so `return match c { 0 { yield S { p =
            // &x } } ... }` returned a dangling `&x` undetected (stack-use-after-
            // return).  Mirrors the if/else sugar already handled via Struct/Call
            // recursion.  The scrutinee/guards are checked defensively; arm bodies
            // are statement blocks whose own returns are checked by the statement
            // walk, and the escaping yield value lives in `a.value`.
            Match { scrutinee, arms, .. } => {
                self.check_no_local_ref_escape(scrutinee, under_deref);
                for a in arms {
                    if let Some(g) = &a.guard { self.check_no_local_ref_escape(g, under_deref); }
                    if let Some(v) = &a.value { self.check_no_local_ref_escape(v, under_deref); }
                }
            }
            _ => {}
        }
    }

    /// Non-erroring predicate: does `e` evaluate to a borrow/view whose referent is
    /// a FUNCTION-LOCAL value that dies on return (a stack local or an owning
    /// local)?  Mirrors the dying cases of `check_no_local_ref_escape` (an `&local`
    /// or a borrowing call whose borrowed arg is itself dying), and additionally
    /// propagates through a local already flagged `holds_dying_borrow`.  Used to set
    /// that flag at an assignment so a borrow stashed in a local is caught when the
    /// local later escapes, not only when the borrow is returned directly.
    fn expr_is_dying_borrow(&self, e: &HExpr) -> bool {
        use HExprKind::*;
        match &e.kind {
            AddrOfRef { place, .. } => root_local(place).map_or(false, |root| {
                let li = &self.f().locals[root.0 as usize];
                match li.storage {
                    StorageClass::Stack => true,
                    StorageClass::Param => !matches!(
                        li.ty,
                        HType::Ref { .. } | HType::Ptr { .. } | HType::RawPtr { .. }
                    ),
                    _ => false,
                }
            }),
            Call { callee, args } | InlineCall { callee, args, .. } => {
                let cid = callee.0 as usize;
                self.ret_borrows.get(cid).map_or(false, |cb| {
                    cb.iter().enumerate().any(|(m, &b)| {
                        b && args.get(m).map_or(false, |a| self.expr_is_dying_borrow(a))
                    })
                })
            }
            Local(id) => self.state[id.0 as usize].holds_dying_borrow,
            Cast { expr, .. } | CheckedCast { expr, .. } | DerefRef(expr)
                | Unwrap { expr, .. } | DropWrite(expr) => self.expr_is_dying_borrow(expr),
            // `alloc X { ... }` of an aggregate that holds a dying borrow yields an
            // owning pointer whose pointee still dangles, so a local bound to it
            // (`own *Container c = alloc Container { p = &x };`) carries the dying
            // borrow too.  (`Un`/`Bin` consume a borrow into a plain value and a
            // `Closure` has its own closure_holds_dying path, so they stay omitted.)
            HeapAlloc(inner) => self.expr_is_dying_borrow(inner),
            // An aggregate HOLDS a dying borrow if any field/element is one, so a
            // local bound to it (`Container c = Container { p = &x };`) is flagged
            // and returning the whole aggregate is caught - the via-local form of
            // the `return Container { p = &x };` escape.  (Reading a non-borrow
            // field of `c` does not escape the borrow: the escape check has no
            // Field arm, so `return c.value_field` is never routed here.)
            Struct { fields, .. } | VariantCtor { fields, .. } =>
                fields.iter().any(|(_, fe)| self.expr_is_dying_borrow(fe)),
            ArrayLit(elems) => elems.iter().any(|el| self.expr_is_dying_borrow(el)),
            // A match RESULT is a dying borrow if any arm yields one (`string v =
            // match c { 0 { yield s.as_str() } ... }` makes v a view of local s).
            // Without this, the borrow flowing out of the match into v is missed
            // and `return v` dangles - the direct `return match ...` form is caught
            // via the arm-body walk, but routing it through v needs this.
            Match { arms, .. } => arms.iter().any(|a|
                a.value.as_ref().map_or(false, |v| self.expr_is_dying_borrow(v))),
            // Extracting a borrow/view FIELD from a local that holds a dying borrow
            // yields a dying borrow (the field may be the dying one), so `&int r =
            // h.q` propagates holds_dying_borrow to r and a later `return r` is
            // caught - the via-local twin of the Field arm in check_no_local_ref_escape.
            Field { base, .. } => matches!(e.ty, HType::Ref { .. } | HType::Ptr { .. })
                && matches!(&base.kind, HExprKind::Local(id) if self.state[id.0 as usize].holds_dying_borrow),
            _ => false,
        }
    }

    fn mark_moved(&mut self, id: LocalId, sp: Span) {
        // Heap-typed bindings honor the v1.2 move semantics; explicit `transfer X` invalidates any binding.
        let (name, is_capture, owns) = {
            let li = &self.f().locals[id.0 as usize];
            (li.name.clone(), li.is_capture, ty_owns_heap(self.sym, &li.ty))
        };
        // An owning closure capture is owned by the heap env and freed on env
        // drop; moving it OUT of the body hands ownership to a second owner while
        // the env still frees it -> double-free.  Reject (borrow it in place, or
        // don't move the owner out).  Covers every move-out path (by-value call,
        // struct/array init, return, transfer, move into another local).
        if is_capture && owns {
            self.err(format!("cannot move captured value `{}` out of the closure body: the closure's environment owns it and frees it when the closure is dropped, so moving it out would double-free. Read it in place (borrow), or restructure so the owner is not moved out of the closure.", name), sp);
            return;
        }
        if self.state[id.0 as usize].moved {
            self.err(format!("use of moved value `{}`", name), sp);
            return;
        }
        self.state[id.0 as usize].moved = true;
        // A move invalidates any place-narrowing rooted at (or indexed by) this
        // local: the container/element it names is now owned elsewhere.
        self.invalidate_place_facts(id.0);
        // Moving an owner POINTER (by-value into a call/struct/array/return, into
        // another owning local, or via `transfer` across a gate) transfers its heap
        // to the new owner, which frees it - any `*T`/`&T` alias of it now dangles.
        // Invalidate the aliases here, at the single move choke point, so EVERY
        // move site is covered uniformly: `*T` aliases AUTO-NULL (a runtime
        // `alias = NULL` is emitted after this statement via stmt_nulls, so a
        // guarded deref is honest), `&T` borrows poison.  Non-pointer owners
        // (String/Vec values) have no `*T` aliases, so kill_lid is a no-op there.
        if matches!(
            self.f().locals[id.0 as usize].ty,
            HType::OwnPtr { .. } | HType::Heap { .. }
        ) {
            let n = self.kill_lid(id, sp, true);
            self.pending_stmt_nulls.extend(n);
        }
    }

    /// Mark `a` moved if it is a bare owning Local consumed BY VALUE - the same
    /// rule the drop pass uses (`ty_owns_heap`) to skip a value's scope-exit
    /// free, applied to the use side so a later read is rejected as a
    /// use-after-move.  Covers owning VALUES (String, Vec, owning struct/enum),
    /// not just `own *T`/`heap T`; borrows (`&T`/`*T`) and bare `FnPtr` locals
    /// own no heap here, so a borrowed receiver (`s.len()` -> `&s`) is not a
    /// move.  Call AFTER walking `a`, so the move itself is not flagged.
    fn mark_owning_move(&mut self, a: &HExpr) {
        // Packing `Vec<T> -> Vec<some X>` moves the underlying Vec into the column,
        // so peel the coercion and mark the source Vec moved (use-after-pack errors).
        let a = match &a.kind {
            HExprKind::Cast { expr, kind, .. } if matches!(kind, CastKind::PackSomeVec { .. }) => expr.as_ref(),
            _ => a,
        };
        if ty_owns_heap(self.sym, &a.ty) {
            if let HExprKind::Local(id) = a.kind {
                // mark_moved poisons the owner's aliases (single move choke point).
                self.mark_moved(id, a.span);
            } else if matches!(a.kind, HExprKind::Field { .. } | HExprKind::Index { .. }) {
                // A projected owning place moved out (`f(xs[0])`, `S { g = s.p }`)
                // auto-nulls the source at runtime (SPEC 6.4), so any place-narrowing
                // rooted there is now stale.
                if let Some(r) = root_local(a) { self.invalidate_place_facts(r.0); }
            }
        }
    }

    /// Compute the deps set of an expression.
    fn expr_deps(&self, e: &HExpr) -> HashSet<u32> {
        let mut out = HashSet::new();
        self.collect_deps(e, &mut out);
        out
    }

    /// Is this RHS expression statically known to produce a non-null pointer?
    /// True for `heap T(...)` and for locals already known non-null.
    /// A projected place (`xs[0]`, `s.p`) is proven non-null iff a live fact
    /// keys to it exactly.
    fn place_nonnull(&self, key: &PlaceKey) -> bool {
        self.place_facts.iter().any(|f| &f.key == key)
    }

    /// Drop every place-fact whose dependency set mentions `local` - it was
    /// mutated / moved / freed / went out of scope, so any narrowing riding on
    /// it is no longer sound.
    fn invalidate_place_facts(&mut self, local: u32) {
        if !self.place_facts.is_empty() {
            self.place_facts.retain(|f| !f.deps.contains(&local));
        }
    }

    fn expr_nonnull(&self, e: &HExpr) -> bool {
        match &e.kind {
            HExprKind::HeapAlloc(_) => true,
            HExprKind::AddrOfRef { .. } => true,
            HExprKind::Local(id) => {
                let st = &self.state[id.0 as usize];
                st.known_nonnull || st.narrowed_until.is_some()
            }
            // A projected place guarded by `if (P != null)`: proven while the
            // matching place-fact is live (see `place_facts`).
            HExprKind::Field { .. } | HExprKind::Index { .. } => {
                place_key(e).map_or(false, |(k, _)| self.place_nonnull(&k))
            }
            HExprKind::Cast { expr, kind, .. } | HExprKind::CheckedCast { expr, kind, .. } => {
                // The tag-checked `*int → *Enum` cast can yield null even when
                // the source pointer is non-null — failure rides in the result
                // type (§6.6).  Bail out so `(p as *Enum)!` is rejected unless
                // the cast result is first null-checked.
                if matches!(kind, CastKind::IntPtrToEnumPtrChecked) {
                    return false;
                }
                self.expr_nonnull(expr)
            }
            // Interprocedural: a call result is proven non-null when the global
            // summary for that callee says so (every return path in the callee
            // is non-null).  `compute_return_summaries` runs before the per-func
            // pass populates these facts.
            HExprKind::Call { callee, .. } | HExprKind::InlineCall { callee, .. } => {
                is_nonnull_string_builtin(callee.0)
                    || self.summaries.get(callee.0 as usize).copied().unwrap_or(false)
            }
            _ => false,
        }
    }
    /// Walk an expression and add the LID of every `own *T` / `own &T` Local
    /// reachable through field / index / unwrap / cast nesting to `out`.
    /// Used by `Let` and `Assign` when the destination is a non-owning slot
    /// (`*T`, `&T`): the new binding aliases the heap allocation rooted at
    /// each owner LID, and `kill_lid` uses the dep edge to invalidate the
    /// alias when the owner mutates, moves, frees, or goes out of scope.
    fn collect_owner_aliases(&self, e: &HExpr, out: &mut HashSet<u32>) {
        match &e.kind {
            HExprKind::Local(id) => {
                if matches!(self.f().locals[id.0 as usize].ty, HType::OwnPtr { .. } | HType::Heap { .. }) {
                    out.insert(id.0);
                }
            }
            HExprKind::Unwrap { expr, .. }
            | HExprKind::Cast { expr, .. }
            | HExprKind::CheckedCast { expr, .. }
            | HExprKind::DropWrite(expr) => self.collect_owner_aliases(expr, out),
            HExprKind::Field { base, .. } | HExprKind::Index { base, .. } => {
                // Aliasing into a by-value CONTAINER that owns heap (`*int a = b.p`
                // where `b: Box{own*int p}`, or `arr[0]` where `arr: [N]own*int`)
                // makes `a` depend on the container Local: reassigning the owning
                // field/element (drop_old frees the old pointee) or invalidating
                // the container must poison the alias.  The container's own type
                // is a struct/array (not OwnPtr/Heap), so the Local arm records
                // nothing - and the projected type may be coerced from own*T to
                // *T here, so key on the container OWNING heap rather than the
                // projected type.  Coarse (container-level) but sound.
                if let Some(r) = root_local(e) {
                    if ty_owns_heap(self.sym, &self.f().locals[r.0 as usize].ty) {
                        out.insert(r.0);
                    }
                }
                self.collect_owner_aliases(base, out);
            }
            HExprKind::DerefRef(inner) => self.collect_owner_aliases(inner, out),
            _ => {}
        }
    }

    fn collect_deps(&self, e: &HExpr, out: &mut HashSet<u32>) {
        match &e.kind {
            HExprKind::AddrOfRef { place, .. } => {
                // root local of the place is the LID
                if let Some(id) = root_local(place) {
                    out.insert(id.0);
                }
            }
            HExprKind::Local(id) => {
                // For a reference-like local, propagate its current deps
                if matches!(self.f().locals[id.0 as usize].ty, HType::Ref { .. } | HType::Ptr { .. } | HType::Slice { .. }) {
                    for d in &self.state[id.0 as usize].deps { out.insert(*d); }
                }
            }
            HExprKind::Unwrap { expr, .. } => self.collect_deps(expr, out),
            HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) => self.collect_deps(expr, out),
            HExprKind::ArrayToSlice { base, .. } => {
                if let Some(id) = root_local(base) { out.insert(id.0); }
                self.collect_deps(base, out);
            }
            HExprKind::DerefRef(inner) => self.collect_deps(inner, out),
            HExprKind::HeapAlloc(_) => {
                // Fresh heap LID; no deps from source. We model it as `{}`.
            }
            HExprKind::Free(_, _) => {
                // `free p;` returns unit; deps don't propagate from the freed pointer.
            }
            HExprKind::Field { base, .. } | HExprKind::Index { base, .. } => self.collect_deps(base, out),
            // Struct/variant literals and array literals propagate their fields'
            // borrow deps up to the containing value.  Without this, building a
            // `Holder { x = &y }` produces a Holder whose deps are empty, and
            // when `y` dies the holder stays unpoisoned — leaving a dangling
            // read undetected.  With this, the holder's deps include y; when y
            // dies `kill_lid` poisons the holder; reads through `holder.x`
            // recurse to the (poisoned) base local and error.
            HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => {
                for (_, fe) in fields { self.collect_deps(fe, out); }
            }
            HExprKind::ArrayLit(elems) => {
                for el in elems { self.collect_deps(el, out); }
            }
            HExprKind::LitNull => {} // empty
            _ => {}
        }
    }

    /// Kill a LID. Returns pointers whose deps became empty (need runtime null-collapse).
    /// Also marks every such pointer as `auto_nulled` so subsequent reads are flagged
    /// unless the user explicitly re-assigned on every code path.
    /// Invalidate every `*T`/`&T` alias of owner `id` whose alias-set collapses to
    /// nothing once `id` is removed.  `poison_ptrs` selects how a `*T` alias dies:
    /// - false (scope-exit): AUTO-NULL it.  Codegen emits a runtime `alias = NULL`
    ///   (HBlock.ptr_nulls), so a later `if (alias != null)` guard correctly sees
    ///   null and the deref is sound.
    /// - true (owner REASSIGN / MOVE): POISON it.  No runtime null is emitted at a
    ///   reassign/move site, so the alias still holds the freed address; a guard
    ///   would wrongly re-validate it against freed memory (heap-UAF).  Poison makes
    ///   any use a hard compile error until the alias is re-assigned (which clears
    ///   poison) - the only sound recovery.
    fn kill_lid(&mut self, id: LocalId, _span: Span, _poison_ptrs: bool) -> Vec<LocalId> {
        let lid_num = id.0;
        let types: Vec<HType> = self.f().locals.iter().map(|l| l.ty.clone()).collect();
        let mut nulled = Vec::new();
        for (i, st) in self.state.iter_mut().enumerate() {
            if i as u32 == lid_num { continue; }
            if st.deps.remove(&lid_num) {
                if let HType::Ptr { .. } = types[i] {
                    if st.deps.is_empty() {
                        // A `*T` nullable alias ALWAYS AUTO-NULLS, NEVER poisons.
                        // The caller emits a runtime `alias = NULL` at the site
                        // (scope-exit -> block.ptr_nulls; mid-block reassign/move ->
                        // block.stmt_nulls) so a later `if (a != null)` guard sees
                        // the real null.  A dangling `*T` errors ONLY at a USE site
                        // (a deref of the unproven alias), never here.
                        nulled.push(LocalId(i as u32));
                        st.auto_nulled = true;
                        st.known_nonnull = false;
                        st.narrowed_until = None;
                    }
                } else {
                    // `&T` / `&mut T` borrows POISON: a borrow has no null state to
                    // reset to, so a dangling-borrow use is a hard error.
                    st.poisoned = true;
                }
            }
        }
        // Freeing / moving / scope-exiting the killed local invalidates any
        // place-narrowing rooted at (or indexed by) it.
        self.invalidate_place_facts(lid_num);
        nulled
    }

    fn snapshot(&self) -> Vec<LocalState> { self.state.clone() }
    fn restore(&mut self, s: Vec<LocalState>) { self.state = s; }

    fn join_branches(&mut self, then_s: &[LocalState], else_s: &[LocalState], in_scope: &std::collections::HashSet<u32>) {
        let n = self.state.len();
        for i in 0..n {
            // Classify the local up front so `li`'s borrow ends before we mutate
            // `self.state[i]` below (the owner pointee vs owning value vs other).
            let (is_owner_ptr, owns_heap_value) = {
                let li = &self.f().locals[i];
                let is_ptr = matches!(li.ty, HType::Heap { .. } | HType::OwnPtr { .. });
                (is_ptr, !is_ptr && ty_owns_heap(self.sym, &li.ty))
            };
            if is_owner_ptr {
                let tm = then_s[i].moved;
                let em = else_s[i].moved;
                // Conditionally moved (moved on exactly one branch): per SPEC 6.4
                // a path that frees/moves the owner AUTO-NULLs it; this is NOT a
                // hard error.  SPEC 6.1 only invalidates a later *use*, and a
                // conditional move with no later use is sound (e.g. hm_grow's
                // `own *string k = m.keys[j]; if (k != null) { nk[i] = k; }`).
                // After the merge the owner is no longer provably non-null, so a
                // later deref needs fresh proof (SPEC 6.3); the owner is only
                // "definitely moved" when BOTH paths moved it.
                let cond_moved = tm != em && in_scope.contains(&(i as u32));
                self.state[i].moved = tm && em;
                self.state[i].known_nonnull = then_s[i].known_nonnull && else_s[i].known_nonnull && !cond_moved;
                self.state[i].auto_nulled = then_s[i].auto_nulled || else_s[i].auto_nulled;
                let mut deps = then_s[i].deps.clone();
                for d in &else_s[i].deps { deps.insert(*d); }
                self.state[i].deps = deps;
                self.state[i].poisoned = then_s[i].poisoned || else_s[i].poisoned;
                self.state[i].holds_dying_borrow = then_s[i].holds_dying_borrow || else_s[i].holds_dying_borrow;
            } else {
                // Reference/pointer: union deps; poison if poisoned in either reachable branch.
                let mut deps = then_s[i].deps.clone();
                for d in &else_s[i].deps { deps.insert(*d); }
                self.state[i].deps = deps;
                self.state[i].poisoned = then_s[i].poisoned || else_s[i].poisoned;
                self.state[i].holds_dying_borrow = then_s[i].holds_dying_borrow || else_s[i].holds_dying_borrow;
                // auto_nulled persists if EITHER branch left it set — the user
                // must re-assign on every code path to deterministically clear it.
                self.state[i].auto_nulled = then_s[i].auto_nulled || else_s[i].auto_nulled;
                // known_nonnull holds only if BOTH branches established it.
                self.state[i].known_nonnull = then_s[i].known_nonnull && else_s[i].known_nonnull;
                // An owning VALUE (String/Vec/owning struct) is move-tracked but
                // has no null state to fall back on like `own *T` does, so a move
                // on EITHER reachable branch makes it moved after the join - a
                // later use is then a use-after-move.  Both branches reach the
                // join here (the caller filters out diverging branches), so the
                // union is sound and does not over-reject a value consumed on a
                // path that returns.
                if owns_heap_value {
                    self.state[i].moved = then_s[i].moved || else_s[i].moved;
                }
            }
        }
    }
}

/// Does every path through this block exit the enclosing scope (return / break /
/// continue / propagate / `panic(...)` call)?  Used to validate early-exit
/// null-guards: only an unconditional exit makes the rest of the parent block
/// see `p` as narrowed.
fn block_always_exits(b: &HBlock) -> bool {
    match b.stmts.last() {
        Some(HStmt::Return { .. }) => true,
        Some(HStmt::Break { .. }) | Some(HStmt::Continue { .. }) => true,
        Some(HStmt::Propagate { .. }) => true,
        Some(HStmt::ExprStmt(e)) => matches!(
            &e.kind,
            HExprKind::Call { callee, .. } if callee.0 == u32::MAX - 2  // builtin panic(...)
        ),
        Some(HStmt::If { then_b, else_b: Some(eb), .. }) => block_always_exits(then_b) && block_always_exits(eb),
        Some(HStmt::Block(b)) => block_always_exits(b),
        _ => false,
    }
}

fn root_local(e: &HExpr) -> Option<LocalId> {
    match &e.kind {
        HExprKind::Local(id) => Some(*id),
        HExprKind::Field { base, .. } | HExprKind::Index { base, .. } | HExprKind::Unwrap { expr: base, .. } => root_local(base),
        HExprKind::AddrOfRef { place, .. } => root_local(place),
        _ => None,
    }
}

/// Whether the function frees this local's value on exit, so capturing it by
/// reference into an escaping closure would dangle: it owns heap AND is not a
/// borrowed reference parameter (whose referent is the caller's, outliving the
/// call).  Stack owning values, `own *T`/heap locals, and owning value/`own`
/// parameters all die on return.
fn local_is_dying_owner(sym: &SymTab, f: &HFunc, id: LocalId) -> bool {
    let li = &f.locals[id.0 as usize];
    if !ty_owns_heap(sym, &li.ty) { return false; }
    // A borrowed-reference param (`&T`/`*T`/`raw *T`) targets caller data; it
    // does not own heap anyway, but guard explicitly for clarity.
    if matches!(li.storage, StorageClass::Param)
        && matches!(li.ty, HType::Ref { .. } | HType::Ptr { .. } | HType::RawPtr { .. })
    {
        return false;
    }
    true
}

/// Collect every `let d = ...closure-literal[caps]...;` in the function as
/// `(d, [outer locals captured])`, descending into nested blocks.
fn collect_closure_caps_block(b: &HBlock, out: &mut Vec<(LocalId, Vec<LocalId>)>) {
    for s in &b.stmts {
        if let HStmt::Let { local, init, .. } = s {
            if let HExprKind::Closure { env_values, .. } = &init.kind {
                let caps: Vec<LocalId> = env_values.iter().filter_map(root_local).collect();
                if !caps.is_empty() { out.push((*local, caps)); }
            }
        }
        for_each_child_block(s, &mut |cb| collect_closure_caps_block(cb, out));
    }
}

/// Closure locals that transitively borrow-capture a local the function frees on
/// exit.  Fixpoint over capture edges so a chain (outer captures inner captures
/// an owner) is fully propagated.
fn compute_closure_holds_dying(sym: &SymTab, f: &HFunc) -> std::collections::HashSet<LocalId> {
    let mut closures: Vec<(LocalId, Vec<LocalId>)> = Vec::new();
    collect_closure_caps_block(&f.body, &mut closures);
    let mut set: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
    if closures.is_empty() { return set; }
    loop {
        let mut changed = false;
        for (d, caps) in &closures {
            if set.contains(d) { continue; }
            if caps.iter().any(|c| local_is_dying_owner(sym, f, *c) || set.contains(c)) {
                set.insert(*d);
                changed = true;
            }
        }
        if !changed { break; }
    }
    set
}

/// Hoist owning temporaries that are consumed by a borrowing context - a call
/// argument that was coerced to a borrow/view, or a discarded expression
/// statement - into hidden owning locals.  The existing auto-free + move
/// analysis then drops or keeps each one correctly: the coercion already
/// encodes borrow-vs-own in the argument's type, so a temp passed to a borrow
/// param stays owning-typed-locally and is freed at scope, while one passed to
/// an `own` param is read as owning and marked moved (the callee frees it).
///
/// This removes the "owning temporary consumed inline leaks" gap, e.g.
/// `log(format(...))` or `f(a + b)`.  Concat-builtin arguments are left alone
/// (the concat helpers free them, so hoisting would double-free), and only
/// fresh heap producers are hoisted (never literals, statics like
/// `bool_to_str`, or named locals).
fn hoist_owning_temps(sym: &SymTab, f: &mut HFunc) {
    let HFunc { body, locals, .. } = f;
    hoist_block_temps(sym, body, locals);
}

fn hoist_block_temps(sym: &SymTab, b: &mut HBlock, locals: &mut Vec<LocalInfo>) {
    let stmts = std::mem::take(&mut b.stmts);
    let mut out: Vec<HStmt> = Vec::with_capacity(stmts.len());
    for mut s in stmts {
        // Recurse into nested blocks first.
        match &mut s {
            HStmt::If { then_b, else_b, .. } => {
                hoist_block_temps(sym, then_b, locals);
                if let Some(e) = else_b { hoist_block_temps(sym, e, locals); }
            }
            HStmt::While { cond, body, span } => {
                hoist_block_temps(sym, body, locals);
                desugar_loop_cond_temps(sym, cond, body, *span, locals);
            }
            HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => {
                hoist_block_temps(sym, body, locals);
            }
            HStmt::Block(bb) | HStmt::Unsafe(bb, _) => hoist_block_temps(sym, bb, locals),
            _ => {}
        }
        // Hoist owning temporaries out of this statement's once-evaluated
        // expression positions.  A LOOP condition (while/for) is re-evaluated, so
        // hoisting it before the loop would be wrong - left alone.  An IF condition
        // is evaluated exactly once, so its temporaries ARE safe to hoist before the
        // if (needed for, e.g., `if (a == make_temp())`, where the auto-borrowed
        // operand temp must become a real lvalue).
        let mut pre: Vec<HStmt> = Vec::new();
        match &mut s {
            HStmt::ExprStmt(e) => {
                hoist_in_expr(sym, e, locals, &mut pre);
                if is_owning_temp(sym, e) { hoist_one(sym, e, locals, &mut pre); }
            }
            HStmt::If { cond, .. } => hoist_in_expr(sym, cond, locals, &mut pre),
            HStmt::Let { init, .. } => hoist_in_expr(sym, init, locals, &mut pre),
            HStmt::Assign { place, value, .. } => {
                hoist_in_expr(sym, place, locals, &mut pre);
                hoist_in_expr(sym, value, locals, &mut pre);
            }
            HStmt::Return { value: Some(v), .. } => hoist_in_expr(sym, v, locals, &mut pre),
            HStmt::Propagate { value: Some(v), .. } => hoist_in_expr(sym, v, locals, &mut pre),
            HStmt::ForEach { src, .. } => {
                hoist_in_expr(sym, src, locals, &mut pre);
                // A temporary owning CONTAINER as the loop source (`for (x in
                // make_vec())`) has no binding, so nothing drops it - it leaks on
                // normal loop exit AND on any early return/break out of the body.
                // Bind it to a hidden owning local so the scope-exit / early-exit
                // drop machinery frees it exactly once on every path (shares the
                // hoist mechanism with owning call-argument temps).
                if is_droppable_temp(sym, src) { hoist_one(sym, src, locals, &mut pre); }
            }
            _ => {}
        }
        out.extend(pre);
        out.push(s);
    }
    b.stmts = out;
}

/// Hoist owning temporaries out of a re-evaluated loop condition.  A `while`
/// condition can't be hoisted *before* the loop (it must run each iteration), so
/// move the temp bindings and a `break`-if-false to the TOP of the loop body and
/// make the condition unconditionally true.  The temps then bind and free once per
/// iteration (correct re-evaluation), and `&<temp>` becomes a valid lvalue instead
/// of invalid C.  No-op when the condition has no temps to hoist.
fn desugar_loop_cond_temps(sym: &SymTab, cond: &mut HExpr, body: &mut HBlock, span: Span, locals: &mut Vec<LocalInfo>) {
    let mut cond_pre: Vec<HStmt> = Vec::new();
    hoist_in_expr(sym, cond, locals, &mut cond_pre);
    if cond_pre.is_empty() { return; }
    // `if (!cond) { break; }` — cond now references the hoisted locals.
    let neg = HExpr {
        kind: HExprKind::Un { op: HUnOp::Not, expr: Box::new(cond.clone()) },
        ty: HType::Bool,
        span,
    };
    let if_break = HStmt::If {
        cond: neg,
        then_b: HBlock { stmts: vec![HStmt::Break { heap_drops: Vec::new(), span }],
                         heap_to_free: Vec::new(), ptr_nulls: Vec::new(), stmt_nulls: Vec::new(), span },
        else_b: None,
        span,
    };
    let mut new_body: Vec<HStmt> = Vec::with_capacity(cond_pre.len() + 1 + body.stmts.len());
    new_body.append(&mut cond_pre);
    new_body.push(if_break);
    new_body.append(&mut body.stmts);
    body.stmts = new_body;
    *cond = HExpr { kind: HExprKind::LitBool(true), ty: HType::Bool, span };
}

/// Recurse through an expression hoisting owning-temporary call arguments.
fn hoist_in_expr(sym: &SymTab, e: &mut HExpr, locals: &mut Vec<LocalInfo>, pre: &mut Vec<HStmt>) {
    match &mut e.kind {
        HExprKind::Call { callee, args } => {
            // Concat helpers free their own string args; hoisting would double-free.
            let c = callee.0;
            let is_concat = c == u32::MAX - 5 || c == u32::MAX - 8 || c == u32::MAX - 9 || c == u32::MAX - 10;
            for a in args.iter_mut() { hoist_in_expr(sym, a, locals, pre); }
            if !is_concat {
                for a in args.iter_mut() {
                    if is_owning_temp(sym, a) { hoist_one(sym, a, locals, pre); }
                }
            }
        }
        HExprKind::CallIndirect { callee, args } => {
            hoist_in_expr(sym, callee, locals, pre);
            for a in args.iter_mut() { hoist_in_expr(sym, a, locals, pre); }
            for a in args.iter_mut() {
                if is_owning_temp(sym, a) { hoist_one(sym, a, locals, pre); }
            }
        }
        HExprKind::Bin { lhs, rhs, .. } => { hoist_in_expr(sym, lhs, locals, pre); hoist_in_expr(sym, rhs, locals, pre); }
        // Deref of a temporary (`make()!`, e.g. `("a"+"b")!`): the temp owns its
        // buffer and `!` only reads the value out, so the temp has no owner and
        // would leak.  Bind it to a hidden local (dropped at scope exit), exactly
        // like the field/element projections below.
        HExprKind::Unwrap { expr, .. } => {
            hoist_in_expr(sym, expr, locals, pre);
            if is_droppable_temp(sym, expr) { hoist_one(sym, expr, locals, pre); }
        }
        HExprKind::Un { expr, .. }
        | HExprKind::Cast { expr, .. }
        | HExprKind::CheckedCast { expr, .. }
        | HExprKind::DropWrite(expr)
        | HExprKind::DerefRef(expr)
        | HExprKind::HeapAlloc(expr)
        | HExprKind::Free(expr, _)
        | HExprKind::SliceLen(expr)
        | HExprKind::EnumTag(expr)
        | HExprKind::ArrayToSlice { base: expr, .. } => hoist_in_expr(sym, expr, locals, pre),
        // Borrowing a temporary (`&make()`, or an auto-borrowed method receiver
        // `make().method()`): the temp has no owner and `&<rvalue>` is not a
        // valid C lvalue, so bind it to a hidden local that lives to scope exit.
        HExprKind::AddrOfRef { place, .. } => {
            hoist_in_expr(sym, place, locals, pre);
            // Any temporary in `&` position needs binding: an owning one to free
            // it, and even a non-owning rvalue because `&<rvalue>` is not a valid
            // C lvalue.  A `dyn` value collapses to its fat pointer at codegen
            // (no real address taken), so leave it alone.
            if !matches!(place.ty, HType::Dyn { .. })
                && (is_droppable_temp(sym, place) || !is_lvalue_expr(place))
            {
                hoist_one(sym, place, locals, pre);
            }
        }
        // Projecting a field/element off a temporary (`make().field`,
        // `make()[i]`): nothing owns the temp, so bind it or it leaks after the
        // access.  The hidden local is a normal owning local from here on, so the
        // standard move-tracking + scope-exit drop handle it.
        HExprKind::Field { base, .. } => {
            hoist_in_expr(sym, base, locals, pre);
            if is_droppable_temp(sym, base) { hoist_one(sym, base, locals, pre); }
        }
        HExprKind::Index { base, idx } => {
            hoist_in_expr(sym, base, locals, pre);
            hoist_in_expr(sym, idx, locals, pre);
            if is_droppable_temp(sym, base) { hoist_one(sym, base, locals, pre); }
        }
        HExprKind::ArrayLit(es) => { for e2 in es.iter_mut() { hoist_in_expr(sym, e2, locals, pre); } }
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => {
            // Field initializers are moved into the aggregate (owned by it), so
            // they are not hoisted; only their nested call-args are.
            for (_, fe) in fields.iter_mut() { hoist_in_expr(sym, fe, locals, pre); }
        }
        HExprKind::Match { scrutinee, arms, .. } => {
            // The scrutinee is evaluated once, so hoist its nested temporaries
            // (e.g. an auto-borrowed operand temp in `match (a == make_temp()) {}`).
            // The scrutinee-as-a-whole owning temp is handled by the match's own
            // __s copy + field-nulling, so this only touches nested temps.
            hoist_in_expr(sym, scrutinee, locals, pre);
            for arm in arms.iter_mut() {
                hoist_block_temps(sym, &mut arm.body, locals);
                // The arm `value` (how `if`/`match`-as-value and `yield` lower)
                // runs AFTER the body, so hoist its rvalue operand temporaries into
                // the END of the arm body - not the outer `pre`, which would
                // evaluate them unconditionally before the match.  Without this an
                // `&`-taking overload with rvalue operands inside the yielded value
                // (`yield string_from("a") + string_from("b")`) emits `&(rvalue)`,
                // not a valid C lvalue.  The arm body is a drop scope (see
                // visit_matches_in_expr in fill_heap_drops), so these owning temps
                // are freed at arm exit rather than leaking.
                if let Some(v) = &mut arm.value {
                    hoist_in_expr(sym, v, locals, &mut arm.body.stmts);
                }
            }
        }
        // Inline splices and closures capture/move their operands; leave them.
        _ => {}
    }
}

/// A freshly-allocated owning value that no binding owns yet.  Precise: fresh
/// heap producers only - never `bool_to_str` (a static), literals, or locals.
fn is_owning_temp(sym: &SymTab, e: &HExpr) -> bool {
    match &e.kind {
        HExprKind::HeapAlloc(_) => true,
        HExprKind::Call { callee, .. } => {
            let c = callee.0;
            // malloc'ing string builtins: concat / read_line / int|float|char_to_str / format1
            if c == u32::MAX - 5 || c == u32::MAX - 8 || c == u32::MAX - 9 || c == u32::MAX - 10
                || c == u32::MAX - 6
                || c == u32::MAX - 11 || c == u32::MAX - 13 || c == u32::MAX - 14
                || c == u32::MAX - 59
            {
                return true;
            }
            // a regular function that returns an owning value (ownership transfers out)
            if (c as usize) < sym.sigs.len() {
                return matches!(sym.func_sig(*callee).ret, HType::OwnPtr { .. } | HType::Heap { .. });
            }
            false
        }
        _ => false,
    }
}

/// The owning type to declare the hidden local with (so it auto-frees).  Owning
/// pointers keep their type; string temps (`Str`-typed but a malloc'd `char*`)
/// become `own *char`.
fn owning_type_of(sym: &SymTab, e: &HExpr) -> HType {
    // Recover the PRODUCER's true owning type, not the expression's `e.ty`: an
    // `own *T` temp passed to a `*T` borrow parameter has been coerced to the
    // non-owning alias `*T` at the arg position, so `e.ty` is `*T` (which owns
    // nothing).  Declaring the hidden local with that coerced type would leave it
    // un-owning and un-dropped -> leak (`peek(make())`).  A call's owning return
    // type is the authority.
    if let HExprKind::Call { callee, .. } = &e.kind {
        if (callee.0 as usize) < sym.sigs.len() {
            let ret = &sym.func_sig(*callee).ret;
            if matches!(ret, HType::OwnPtr { .. } | HType::Heap { .. }) {
                return ret.clone();
            }
        }
    }
    match &e.ty {
        HType::OwnPtr { .. } | HType::Heap { .. } => e.ty.clone(),
        // A `Str`-typed temp is really a malloc'd `char*` (e.g. a concat result
        // coerced to the borrowed `string` view), so the hidden local must own it.
        HType::Str => HType::owned_string(),
        // An owning value type (struct / Vec / enum returned by value): declare
        // the hidden local with the real type so its drop glue frees it.
        other => other.clone(),
    }
}

/// Whether `e` is a C lvalue (so `&e` is valid without binding it first).
/// Places - locals, globals, fields, indices, and pointer derefs - are
/// lvalues; call/match/arithmetic/literal results are not.
fn is_lvalue_expr(e: &HExpr) -> bool {
    matches!(
        &e.kind,
        HExprKind::Local(_)
            | HExprKind::GlobalRef(_)
            | HExprKind::Field { .. }
            | HExprKind::Index { .. }
            | HExprKind::Unwrap { .. }
            | HExprKind::DerefRef(_)
    )
}

/// A freshly-produced value that owns heap and has no binding yet, for the
/// projecting/borrowing positions (`make().field`, `make()[i]`, `&make()`).
/// Broader than `is_owning_temp` (pointer-only, used for the by-value move-arg
/// case): also covers owning structs/Vecs/enums returned by value, which leak
/// when projected or borrowed without being bound.
fn is_droppable_temp(sym: &SymTab, e: &HExpr) -> bool {
    if is_owning_temp(sym, e) { return true; }
    match &e.kind {
        HExprKind::Call { callee, .. } if (callee.0 as usize) < sym.sigs.len() => ty_owns_heap(sym, &e.ty),
        HExprKind::Match { .. } => ty_owns_heap(sym, &e.ty),
        _ => false,
    }
}

/// Replace `e` with a read of a fresh owning local, and append the binding to
/// `pre`.  The read keeps `e`'s original (possibly coerced) type so downstream
/// dispatch (e.g. `log`) and the move analysis behave exactly as before.
fn hoist_one(sym: &SymTab, e: &mut HExpr, locals: &mut Vec<LocalInfo>, pre: &mut Vec<HStmt>) {
    let lid = LocalId(locals.len() as u32);
    let own_ty = owning_type_of(sym, e);
    let orig_ty = e.ty.clone();
    let sp = e.span;
    locals.push(LocalInfo {
        name: format!("__tmp{}", lid.0),
        ty: own_ty,
        storage: StorageClass::Heap,
        mut_payload: true,
        reassignable: true,
        thread_local: false,
        is_capture: false,
        span: sp,
    });
    let original = std::mem::replace(e, HExpr { kind: HExprKind::Local(lid), ty: orig_ty, span: sp });
    pre.push(HStmt::Let { local: lid, init: original, span: sp });
}

/// Fill in `HBlock::heap_to_free` and `HStmt::Return::heap_drops` with the heap locals
/// declared in their respective scopes (excluding moved ones).
///
/// Conservative version: drop heap locals at the *end* of the block in which they were
/// declared, in reverse declaration order, unless they were moved (which we approximate
/// by skipping any local that appears as the direct value of a `return` or as a move
/// argument inside the block). For v1 we adopt the safe always-drop-if-not-moved-at-end
/// reading and rely on the move tracker to keep this consistent.
/// Does a value of this type transitively own heap storage (so it needs a drop
/// at scope exit, and transfers ownership when passed/returned by value)?
/// Mirrors codegen's `drop_ty_owns`.  Owning pointers short-circuit, so
/// recursive owning types (`data Node { own *Node next }`) terminate; a `seen`
/// set guards against any pathological by-value cycle.
/// Builtin spawn/thread/job/spawn_pool callee ids (a closure arg to one of
/// these moves its owning captures into the thread).
fn is_spawn_callee(c: FuncId) -> bool {
    // spawn / thread / job / spawn_pool / spawn_on -- all move owning captures
    // into the closure env (so the source local must not also be freed).
    c.0 == u32::MAX - 3 || c.0 == u32::MAX - 15 || c.0 == u32::MAX - 16
        || c.0 == u32::MAX - 37 || c.0 == u32::MAX - 70
}

/// Terminal thread-handle reaps: `join` / `detach` / `cancel` unconditionally
/// drop the handle's spawner ref (see codegen __maka_join_result / __maka_detach
/// / __maka_cancel), so the `*Thread` handle is consumed - a second reap or any
/// later use is a use-after-free / double-free.  `try_join` / `join_timeout`
/// reap only on success (they poll), so they are NOT unconditional consumers.
fn is_reaping_callee(c: FuncId) -> bool {
    c.0 == u32::MAX - 4     // join
        || c.0 == u32::MAX - 33 // detach
        || c.0 == u32::MAX - 23 // cancel
}

/// CROSS-thread spawn tiers (thread / job / spawn_pool), excluding the
/// same-thread fiber `spawn`.  Only these can outlive the spawning scope, so a
/// borrowed-reference capture into one is sound only with a proven join.
fn is_cross_thread_callee(c: FuncId) -> bool {
    // thread / job / spawn_pool / spawn_on -- all cross an OS-thread boundary.
    // (spawn_on is emitted closure-first, so the closure is args[0] like the rest.)
    c.0 == u32::MAX - 15 || c.0 == u32::MAX - 16 || c.0 == u32::MAX - 37 || c.0 == u32::MAX - 70
}

/// If `e` is a cross-thread spawn whose closure captures a borrowed reference
/// (`&T`/`&mut T`), return that capture's span (the borrow that needs a join).
/// Per-function map: a local bound to a closure -> the borrows (root local +
/// `&mut`?) that closure carries in its env, TRANSITIVELY (a closure that
/// captures another borrow-carrying closure by value inherits its borrows).
/// Lets the cross-thread scans see through a nested closure that smuggles a
/// borrow across the boundary (`bump = unit()[&mut x]{...}; thread(unit()[bump]{bump()})`).
type ClosureCaps = std::collections::HashMap<LocalId, Vec<(LocalId, bool)>>;

/// Append the borrows a closure's env carries: a direct `&T`/`&mut T` capture,
/// AND the transitive borrows of any captured closure value (looked up in `caps`).
fn closure_env_borrows(env_values: &[HExpr], caps: &ClosureCaps, out: &mut Vec<(LocalId, bool)>) {
    for v in env_values {
        let root = root_local(v);
        if let HType::Ref { mutable, .. } = &v.ty {
            if let Some(r) = root { out.push((r, *mutable)); }
        }
        // A captured closure VALUE (FnPtr) - or a reference to one - drags its
        // own borrow captures across the boundary even though its type is FnPtr,
        // invisible to the direct `Ref` check above.  Inherit them.
        if let Some(r) = root {
            if let Some(inner) = caps.get(&r) { out.extend(inner.iter().copied()); }
        }
    }
}

/// Unwrap `alloc`/`heap`/deref wrappers to the underlying `Closure { env_values }`.
fn closure_env_of(e: &HExpr) -> Option<&[HExpr]> {
    let mut cur = e;
    loop {
        match &cur.kind {
            HExprKind::Closure { env_values, .. } => return Some(env_values),
            HExprKind::HeapAlloc(i) | HExprKind::DropWrite(i)
            | HExprKind::DerefRef(i) | HExprKind::Transfer(i) => cur = i,
            _ => return None,
        }
    }
}

/// Build the per-function closure-capture map (forward pass: a closure can only
/// capture locals bound earlier, so one forward walk resolves transitivity).
fn build_closure_caps(f: &HFunc) -> ClosureCaps {
    fn walk(b: &HBlock, caps: &mut ClosureCaps) {
        for s in &b.stmts {
            if let HStmt::Let { local, init, .. } = s {
                if let Some(env) = closure_env_of(init) {
                    let mut borrows = Vec::new();
                    closure_env_borrows(env, caps, &mut borrows);
                    caps.insert(*local, borrows);
                }
            }
            for_each_child_block(s, &mut |cb| walk(cb, caps));
        }
    }
    let mut caps = ClosureCaps::new();
    walk(&f.body, &mut caps);
    caps
}

fn cross_thread_borrow_capture(e: &HExpr, caps: &ClosureCaps) -> Option<Span> {
    let HExprKind::Call { callee, args } = &e.kind else { return None };
    if !is_cross_thread_callee(*callee) { return None; }
    let env = closure_env_of(args.first()?)?;
    for v in env {
        if matches!(v.ty, HType::Ref { .. }) { return Some(v.span); }
        if let Some(r) = root_local(v) {
            if caps.get(&r).map_or(false, |b| !b.is_empty()) { return Some(v.span); }
        }
    }
    None
}

/// Is `e` a `join(handle)` call (the single-handle blocking join, MAX-4)?
fn is_join_of(e: &HExpr, handle: LocalId) -> bool {
    if let HExprKind::Call { callee, args } = &e.kind {
        if callee.0 == u32::MAX - 4 {
            if let Some(a) = args.first() { return root_local(a) == Some(handle); }
        }
    }
    false
}

/// Within `b`, is the spawn at `spawn_idx` (binding `handle`) UNCONDITIONALLY
/// joined later in the same block before any branch or early-exit?  Only
/// straight-line `let`/expr statements may precede the `join(handle)`; any
/// Assign / If / While / For / Match / Return / Break / Continue / Propagate in
/// between means the join cannot be proven to dominate the block's exit (and
/// thus every in-scope local's drop), so we answer no (caller rejects).
fn handle_joined_in_block(b: &HBlock, spawn_idx: usize, handle: LocalId) -> bool {
    for s in &b.stmts[spawn_idx + 1..] {
        match s {
            HStmt::Let { .. } => continue,
            HStmt::ExprStmt(e) => { if is_join_of(e, handle) { return true; } }
            _ => return false,
        }
    }
    false
}

/// Reject a cross-thread spawn (thread/job/spawn_pool) whose closure captures a
/// borrowed reference UNLESS the handle is unconditionally joined before scope
/// exit (Rust `thread::scope`-style scoped borrow).  Conservative by design:
/// when a join cannot be proven, reject - an under-rejection here would be a
/// use-after-free, so "unsure" must mean "no".
/// The locals a cross-thread closure captures BY REFERENCE, with each borrow's
/// mutability (`&mut` = true).  While the (un-joined) thread holds these borrows,
/// the parent must not alias them - reading/writing them races with the thread.
fn cross_thread_captured_borrows(e: &HExpr, caps: &ClosureCaps) -> Vec<(LocalId, bool)> {
    let mut out = Vec::new();
    let HExprKind::Call { callee, args } = &e.kind else { return out };
    if !is_cross_thread_callee(*callee) { return out; }
    let Some(a) = args.first() else { return out };
    if let Some(env) = closure_env_of(a) {
        closure_env_borrows(env, caps, &mut out);
    }
    out
}

/// Does statement `s` mention local `target` anywhere in its expressions
/// (read/write/borrow)?  Used to reject a parent aliasing a thread-borrowed local.
fn expr_mentions_local(e: &HExpr, t: LocalId) -> bool {
    use HExprKind::*;
    match &e.kind {
        Local(id) => *id == t,
        AddrOfRef { place, .. } => expr_mentions_local(place, t),
        Field { base, .. } | ArrayToSlice { base, .. } => expr_mentions_local(base, t),
        Index { base, idx } => expr_mentions_local(base, t) || expr_mentions_local(idx, t),
        Bin { lhs, rhs, .. } => expr_mentions_local(lhs, t) || expr_mentions_local(rhs, t),
        Un { expr, .. } | Unwrap { expr, .. } | Cast { expr, .. } | CheckedCast { expr, .. }
        | DropWrite(expr) => expr_mentions_local(expr, t),
        DerefRef(i) | HeapAlloc(i) | Free(i, _) | Transfer(i) | SliceLen(i) | EnumTag(i) => expr_mentions_local(i, t),
        Call { args, .. } | InlineCall { args, .. } => args.iter().any(|a| expr_mentions_local(a, t)),
        CallIndirect { callee, args } => expr_mentions_local(callee, t) || args.iter().any(|a| expr_mentions_local(a, t)),
        Struct { fields, .. } | VariantCtor { fields, .. } => fields.iter().any(|(_, fe)| expr_mentions_local(fe, t)),
        ArrayLit(es) => es.iter().any(|e2| expr_mentions_local(e2, t)),
        Closure { env_values, .. } => env_values.iter().any(|v| expr_mentions_local(v, t)),
        Match { scrutinee, arms, .. } => expr_mentions_local(scrutinee, t)
            || arms.iter().any(|a| a.guard.as_ref().map_or(false, |g| expr_mentions_local(g, t))
                || a.value.as_ref().map_or(false, |v| expr_mentions_local(v, t))
                || a.body.stmts.iter().any(|s| stmt_mentions_local(s, t))),
        _ => false,
    }
}
fn stmt_mentions_local(s: &HStmt, t: LocalId) -> bool {
    match s {
        HStmt::Let { init, .. } => expr_mentions_local(init, t),
        HStmt::Assign { place, value, .. } => expr_mentions_local(place, t) || expr_mentions_local(value, t),
        HStmt::ExprStmt(e) | HStmt::Return { value: Some(e), .. } | HStmt::Propagate { value: Some(e), .. } => expr_mentions_local(e, t),
        HStmt::If { cond, then_b, else_b, .. } => expr_mentions_local(cond, t)
            || then_b.stmts.iter().any(|s| stmt_mentions_local(s, t))
            || else_b.as_ref().map_or(false, |b| b.stmts.iter().any(|s| stmt_mentions_local(s, t))),
        HStmt::While { cond, body, .. } => expr_mentions_local(cond, t) || body.stmts.iter().any(|s| stmt_mentions_local(s, t)),
        HStmt::ForC { cond, body, .. } => expr_mentions_local(cond, t) || body.stmts.iter().any(|s| stmt_mentions_local(s, t)),
        HStmt::ForEach { src, body, .. } => expr_mentions_local(src, t) || body.stmts.iter().any(|s| stmt_mentions_local(s, t)),
        HStmt::Block(b) | HStmt::Unsafe(b, _) => b.stmts.iter().any(|s| stmt_mentions_local(s, t)),
        _ => false,
    }
}

/// Materialize `HBlock.stmt_nulls` (filled by the lifetime pass) into real
/// `alias = NULL;` statements right after their statement, in EVERY block
/// (including match-arm bodies nested in expressions), so all codegen
/// block-emission paths (block, match arm, inline) emit the runtime auto-null
/// for a `*T` alias invalidated by an owner reassign/move.  Runs once after the
/// analysis (indices still valid, before hoisting).
fn inject_stmt_nulls(b: &mut HBlock, locals: &[LocalInfo]) {
    let stmt_nulls = std::mem::take(&mut b.stmt_nulls);
    let old = std::mem::take(&mut b.stmts);
    let mut out = Vec::with_capacity(old.len());
    for (i, mut s) in old.into_iter().enumerate() {
        inject_nulls_in_stmt(&mut s, locals);
        out.push(s);
        if let Some(nulls) = stmt_nulls.get(i) {
            for alias in nulls {
                let ty = locals[alias.0 as usize].ty.clone();
                out.push(HStmt::Assign {
                    op: HAssignOp::Assign,
                    place: HExpr { kind: HExprKind::Local(*alias), ty: ty.clone(), span: b.span },
                    value: HExpr { kind: HExprKind::LitNull, ty, span: b.span },
                    drop_old: false,
                    span: b.span,
                });
            }
        }
    }
    b.stmts = out;
}
fn inject_nulls_in_stmt(s: &mut HStmt, locals: &[LocalInfo]) {
    match s {
        HStmt::Let { init, .. } => inject_nulls_in_expr(init, locals),
        HStmt::Assign { place, value, .. } => { inject_nulls_in_expr(place, locals); inject_nulls_in_expr(value, locals); }
        HStmt::ExprStmt(e) => inject_nulls_in_expr(e, locals),
        HStmt::Return { value: Some(e), .. } | HStmt::Propagate { value: Some(e), .. } => inject_nulls_in_expr(e, locals),
        HStmt::If { cond, then_b, else_b, .. } => { inject_nulls_in_expr(cond, locals); inject_stmt_nulls(then_b, locals); if let Some(eb) = else_b { inject_stmt_nulls(eb, locals); } }
        HStmt::While { cond, body, .. } => { inject_nulls_in_expr(cond, locals); inject_stmt_nulls(body, locals); }
        HStmt::ForC { init, cond, step, body, .. } => { inject_nulls_in_stmt(init, locals); inject_nulls_in_expr(cond, locals); inject_nulls_in_stmt(step, locals); inject_stmt_nulls(body, locals); }
        HStmt::ForEach { src, body, .. } => { inject_nulls_in_expr(src, locals); inject_stmt_nulls(body, locals); }
        HStmt::Block(b) | HStmt::Unsafe(b, _) => inject_stmt_nulls(b, locals),
        HStmt::Return { value: None, .. } | HStmt::Propagate { value: None, .. }
        | HStmt::Break { .. } | HStmt::Continue { .. } => {}
    }
}
fn inject_nulls_in_expr(e: &mut HExpr, locals: &[LocalInfo]) {
    use HExprKind::*;
    match &mut e.kind {
        Match { scrutinee, arms, .. } => {
            inject_nulls_in_expr(scrutinee, locals);
            for a in arms {
                if let Some(g) = &mut a.guard { inject_nulls_in_expr(g, locals); }
                inject_stmt_nulls(&mut a.body, locals);
                if let Some(v) = &mut a.value { inject_nulls_in_expr(v, locals); }
            }
        }
        Call { args, .. } | InlineCall { args, .. } => for a in args { inject_nulls_in_expr(a, locals); },
        CallIndirect { callee, args } => { inject_nulls_in_expr(callee, locals); for a in args { inject_nulls_in_expr(a, locals); } }
        Bin { lhs, rhs, .. } => { inject_nulls_in_expr(lhs, locals); inject_nulls_in_expr(rhs, locals); }
        Un { expr, .. } | Unwrap { expr, .. } | Cast { expr, .. } | CheckedCast { expr, .. }
        | DropWrite(expr) | DerefRef(expr) | HeapAlloc(expr) | Free(expr, _) | Transfer(expr)
        | SliceLen(expr) | EnumTag(expr) => inject_nulls_in_expr(expr, locals),
        AddrOfRef { place, .. } => inject_nulls_in_expr(place, locals),
        Field { base, .. } | ArrayToSlice { base, .. } => inject_nulls_in_expr(base, locals),
        Index { base, idx } => { inject_nulls_in_expr(base, locals); inject_nulls_in_expr(idx, locals); }
        Struct { fields, .. } | VariantCtor { fields, .. } => for (_, fe) in fields { inject_nulls_in_expr(fe, locals); },
        ArrayLit(es) => for e2 in es { inject_nulls_in_expr(e2, locals); },
        Closure { env_values, .. } => for v in env_values { inject_nulls_in_expr(v, locals); },
        LitInt(..) | LitFloat(..) | LitBool(..) | LitChar(..) | LitStr(..) | LitNull | LitUnit
        | ZeroInit | Local(..) | EnumVariant { .. } | FnRef(..) | GlobalRef(..) => {}
    }
}

fn check_scoped_thread_borrows(f: &HFunc, errors: &mut Vec<SemaError>) {
    fn walk(b: &HBlock, locals: &[LocalInfo], caps: &ClosureCaps, errors: &mut Vec<SemaError>) {
        for (i, s) in b.stmts.iter().enumerate() {
            match s {
                HStmt::Let { local, init, .. } => {
                    if let Some(bspan) = cross_thread_borrow_capture(init, caps) {
                        if !handle_joined_in_block(b, i, *local) {
                            let h = &locals[local.0 as usize].name;
                            errors.push(SemaError {
                                msg: format!(
                                    "a cross-thread closure captures a borrowed reference, which is sound only if the thread finishes before the borrowed data's scope ends; add an unconditional `join({})` after the spawn in this block (no branch or early return in between), or capture by value to move ownership into the thread",
                                    h
                                ),
                                span: bspan,
                            });
                        } else {
                            // Joined - but the parent must not ALIAS a `&mut`-captured
                            // local (or write a `&const`-captured one) during the
                            // spawn..join window: the thread holds the borrow, so
                            // concurrent parent access is an unsynchronized data race.
                            // (A `&const` capture allows concurrent parent READS, but
                            // proving read-only is hard here; conservatively any
                            // mention of a &mut-captured local in the window is rejected.)
                            let join_idx = b.stmts[i + 1..].iter()
                                .position(|s2| matches!(s2, HStmt::ExprStmt(e) if is_join_of(e, *local)))
                                .map(|p| i + 1 + p);
                            if let Some(ji) = join_idx {
                                for (cl, is_mut) in cross_thread_captured_borrows(init, caps) {
                                    if !is_mut { continue; }
                                    for s2 in &b.stmts[i + 1..ji] {
                                        if stmt_mentions_local(s2, cl) {
                                            let cn = &locals[cl.0 as usize].name;
                                            errors.push(SemaError {
                                                msg: format!(
                                                    "`{}` is mutably borrowed by a spawned thread until `join`; the parent must not access it during the spawn..join window (that is an unsynchronized data race). Move access after the `join`, or synchronize via an atomic/Mutex",
                                                    cn
                                                ),
                                                span: bspan,
                                            });
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                HStmt::ExprStmt(e) => {
                    if let Some(bspan) = cross_thread_borrow_capture(e, caps) {
                        errors.push(SemaError {
                            msg: "a cross-thread closure captures a borrowed reference but its handle is discarded, so it can never be joined; bind the handle and `join` it before scope exit, or capture by value".to_string(),
                            span: bspan,
                        });
                    }
                }
                _ => {}
            }
            for_each_child_block(s, &mut |cb| walk(cb, locals, caps, errors));
        }
    }
    let caps = build_closure_caps(f);
    walk(&f.body, &f.locals, &caps, errors);
}

/// Does the nominal type named `name` carry a user `Drop` impl?  Such a type is
/// move-only and gets scope-exit drop glue even if it owns no heap fields (the
/// `drop` is the resource release).  Mirrors codegen's `type_impls_drop`.
pub(crate) fn type_impls_drop(sym: &SymTab, name: &str) -> bool {
    sym.trait_impls.get("Drop").is_some_and(|s| s.contains(name))
}

pub(crate) fn ty_owns_heap(sym: &SymTab, ty: &HType) -> bool {
    fn go(sym: &SymTab, ty: &HType, seen: &mut Vec<u64>) -> bool {
        match ty {
            HType::Heap { .. } | HType::OwnPtr { .. } => true,
            // A by-value `Vec<T>` owns its malloc'd buffer.
            HType::Vec { .. } => true,
            // `Rust<T>` owns a boxed Rust value (dropped via a generated shim).
            HType::RustOpaque(_) => true,
            HType::Struct(id) => {
                // A `has Drop` type is owning/move-only by its destructor alone.
                if type_impls_drop(sym, &sym.struct_info(*id).name) { return true; }
                let k = id.0 as u64;
                if seen.contains(&k) { return false; }
                seen.push(k);
                // A `FnPtr` field is owning *as a field*: a closure stored in a
                // struct owns its heap env, which must be freed when the struct
                // drops.  (A bare `FnPtr` local is not owning here - those are
                // handled by the dedicated closure-drop pass, so counting them
                // would double-free.)
                let r = sym.struct_info(*id).fields.iter().any(|fi| matches!(fi.ty, HType::FnPtr { .. }) || go(sym, &fi.ty, seen));
                seen.pop();
                r
            }
            HType::Enum(id) => {
                if type_impls_drop(sym, &sym.enum_info(*id).name) { return true; }
                let k = (id.0 as u64) | (1u64 << 32);
                if seen.contains(&k) { return false; }
                seen.push(k);
                let r = sym.enum_info(*id).variants.iter().any(|v| v.fields.iter().any(|fi| matches!(fi.ty, HType::FnPtr { .. }) || go(sym, &fi.ty, seen)));
                seen.pop();
                r
            }
            HType::Array { elem, .. } => go(sym, elem, seen),
            _ => false,
        }
    }
    go(sym, ty, &mut Vec::new())
}

/// Is `ty` a by-value composite (struct/enum/array) that owns heap - i.e. one
/// that the directly-owning `Heap`/`OwnPtr` checks would miss?
fn is_owning_value_composite(sym: &SymTab, ty: &HType) -> bool {
    matches!(ty, HType::Struct(_) | HType::Enum(_) | HType::Array { .. } | HType::Vec { .. } | HType::RustOpaque(_)) && ty_owns_heap(sym, ty)
}

// True if `e` moves the owning local `target` out of itself: `target` (an owning
// value) appears in a by-value owning position - a call/inline-call argument, a
// `transfer`, or an aggregate field.  A focused bool form of the move-detection
// in `moved_locals_in_expr`, used to reject moving an owning binding out of a
// match guard (a guard may only borrow; moving double-frees).
fn expr_moves_owning_local(sym: &SymTab, e: &HExpr, target: LocalId) -> bool {
    let is_move = |a: &HExpr| ty_owns_heap(sym, &a.ty) && matches!(a.kind, HExprKind::Local(id) if id == target);
    match &e.kind {
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } =>
            args.iter().any(|a| is_move(a) || expr_moves_owning_local(sym, a, target)),
        HExprKind::CallIndirect { callee, args } =>
            expr_moves_owning_local(sym, callee, target) || args.iter().any(|a| expr_moves_owning_local(sym, a, target)),
        HExprKind::Bin { lhs, rhs, .. } => expr_moves_owning_local(sym, lhs, target) || expr_moves_owning_local(sym, rhs, target),
        HExprKind::Index { base, idx } => expr_moves_owning_local(sym, base, target) || expr_moves_owning_local(sym, idx, target),
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } | HExprKind::Cast { expr, .. }
        | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr)
        | HExprKind::AddrOfRef { place: expr, .. } | HExprKind::Field { base: expr, .. }
        | HExprKind::HeapAlloc(expr) | HExprKind::Free(expr, _) | HExprKind::SliceLen(expr)
        | HExprKind::EnumTag(expr) | HExprKind::ArrayToSlice { base: expr, .. } => expr_moves_owning_local(sym, expr, target),
        HExprKind::Transfer(inner) => matches!(inner.kind, HExprKind::Local(id) if id == target) || expr_moves_owning_local(sym, inner, target),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } =>
            fields.iter().any(|(_, fe)| is_move(fe) || expr_moves_owning_local(sym, fe, target)),
        HExprKind::ArrayLit(es) => es.iter().any(|x| is_move(x) || expr_moves_owning_local(sym, x, target)),
        HExprKind::Match { scrutinee, arms, .. } =>
            expr_moves_owning_local(sym, scrutinee, target) || arms.iter().any(|a|
                a.guard.as_ref().is_some_and(|g| expr_moves_owning_local(sym, g, target))
                || a.value.as_ref().is_some_and(|v| expr_moves_owning_local(sym, v, target))),
        _ => false,
    }
}

fn fill_heap_drops(sym: &SymTab, f: &mut HFunc) {
    // We need to walk and, for each block, accumulate `Let { local }` whose local has heap storage.
    // For `return` statements inside that block, the same locals are dropped *before* returning,
    // minus any local being returned (moved-out via the return expression).

    fn moved_locals_in_expr(sym: &SymTab, e: &HExpr, out: &mut std::collections::HashSet<LocalId>) {
        // local appearing as an owning value at the top of any call arg or return is moved
        match &e.kind {
            HExprKind::Local(_) => {
                // A bare Local read is NOT a move — it's only a move when it appears
                // directly as a call argument (handled below) or as a return value
                // (handled in the caller).  Field/Unwrap/etc. accesses through a
                // Heap/OwnPtr local are reads of the inner value, not transfers.
            }
            HExprKind::Call { callee, args } => {
                let spawn = is_spawn_callee(*callee);
                for a in args {
                    if ty_owns_heap(sym, &a.ty) {
                        if let HExprKind::Local(id) = a.kind { out.insert(id); }
                    }
                    // A closure passed to spawn/thread/job MOVES its owning
                    // captures into the thread (the thread's env owns them and
                    // frees them via env_drop); mark them moved so the enclosing
                    // scope no longer frees them (else a double-free).
                    if spawn {
                        if let HExprKind::Closure { env_values, .. } = &a.kind {
                            for v in env_values {
                                if ty_owns_heap(sym, &v.ty) {
                                    if let HExprKind::Local(id) = v.kind { out.insert(id); }
                                }
                            }
                        }
                    }
                    moved_locals_in_expr(sym, a, out);
                }
            }
            HExprKind::Bin { lhs, rhs, .. } => { moved_locals_in_expr(sym, lhs, out); moved_locals_in_expr(sym, rhs, out); }
            HExprKind::Un { expr, .. } => moved_locals_in_expr(sym, expr, out),
            HExprKind::Unwrap { expr, .. } => moved_locals_in_expr(sym, expr, out),
            HExprKind::AddrOfRef { place, .. } => moved_locals_in_expr(sym, place, out),
            HExprKind::Field { base, .. } => moved_locals_in_expr(sym, base, out),
            HExprKind::Index { base, idx } => { moved_locals_in_expr(sym, base, out); moved_locals_in_expr(sym, idx, out); }
            HExprKind::Cast { expr, kind, .. } => {
                // Packing `Vec<T> -> Vec<some X>` MOVES the buffer into the column,
                // so the source Vec must not also be dropped (double free).
                if matches!(kind, CastKind::PackSomeVec { .. }) {
                    if let HExprKind::Local(id) = expr.kind { out.insert(id); }
                }
                moved_locals_in_expr(sym, expr, out);
            }
            HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) => moved_locals_in_expr(sym, expr, out),
            HExprKind::ArrayToSlice { base, .. } => moved_locals_in_expr(sym, base, out),
            HExprKind::DerefRef(inner) => moved_locals_in_expr(sym, inner, out),
            HExprKind::HeapAlloc(inner) => moved_locals_in_expr(sym, inner, out),
            HExprKind::Free(inner, _) => moved_locals_in_expr(sym, inner, out),
            HExprKind::CallIndirect { callee, args } => {
                moved_locals_in_expr(sym, callee, out);
                for a in args { moved_locals_in_expr(sym, a, out); }
            }
            HExprKind::InlineCall { args, .. } => {
                // Mirror the Call arm — owning args at the InlineCall site transfer
                // ownership into the inline's parameter local, so the outer binding
                // must be marked moved for the auto-free at scope exit to skip it.
                for a in args {
                    if ty_owns_heap(sym, &a.ty) {
                        if let HExprKind::Local(id) = a.kind { out.insert(id); }
                    }
                    moved_locals_in_expr(sym, a, out);
                }
            }
            HExprKind::Closure { env_values, .. } => {
                for v in env_values { moved_locals_in_expr(sym, v, out); }
            }
            HExprKind::Transfer(inner) => {
                if let HExprKind::Local(id) = inner.kind { out.insert(id); }
                moved_locals_in_expr(sym, inner, out);
            }
            HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => moved_locals_in_expr(sym, inner, out),
            HExprKind::FnRef(_) => {}
            // Aggregate literals MOVE an owning value into the new struct/variant/
            // array (it now owns it), so the source must be marked moved - else it
            // is freed at scope AND owned by the aggregate (double free).
            HExprKind::VariantCtor { fields, .. } => for (_, fe) in fields {
                if ty_owns_heap(sym, &fe.ty) { if let HExprKind::Local(id) = fe.kind { out.insert(id); } }
                moved_locals_in_expr(sym, fe, out);
            },
            HExprKind::Match { scrutinee, arms, .. } => {
                moved_locals_in_expr(sym, scrutinee, out);
                for a in arms {
                    if let Some(g) = &a.guard { moved_locals_in_expr(sym, g, out); }
                    if let Some(v) = &a.value { moved_locals_in_expr(sym, v, out); }
                }
            }
            HExprKind::Struct { fields, .. } => for (_, fe) in fields {
                if ty_owns_heap(sym, &fe.ty) { if let HExprKind::Local(id) = fe.kind { out.insert(id); } }
                moved_locals_in_expr(sym, fe, out);
            },
            HExprKind::ArrayLit(es) => for e in es {
                if ty_owns_heap(sym, &e.ty) { if let HExprKind::Local(id) = e.kind { out.insert(id); } }
                moved_locals_in_expr(sym, e, out);
            },
            _ => {}
        }
    }

    // A `match` (and its `if`/`if-expr` sugar) lowers to an expression, so its arm
    // bodies live INSIDE an expression rather than as child statement blocks - the
    // statement walk in visit_block / for_each_child_block never reaches them, so
    // owning locals declared in an arm body (including operand temporaries hoisted
    // into it for an `&`-taking yield) would never be scheduled for drop and would
    // leak.  This walks every expression position (mirroring moved_locals_in_expr
    // so no nesting is missed) and, for each Match arm body, runs visit_block to
    // fill its heap_to_free as a real nested scope.  A local moved OUT by the arm
    // value (`yield s`) runs after the body and is excluded so it is not also
    // dropped here (which would double-free its new owner).
    fn visit_matches_in_expr(sym: &SymTab, locals: &[LocalInfo], e: &mut HExpr, scope_chain: &mut Vec<Vec<LocalId>>, loop_start: Option<usize>, moved: &mut std::collections::HashSet<LocalId>, owning_params: &std::collections::HashSet<LocalId>) {
        match &mut e.kind {
            HExprKind::Match { scrutinee, arms, .. } => {
                // Scrutinee evaluates first; its moves are in effect for every arm.
                visit_matches_in_expr(sym, locals, scrutinee, scope_chain, loop_start, moved, owning_params);
                // Arms are alternatives, so each starts from the same baseline state.
                let baseline = moved.clone();
                // Enclosing owning values an arm could move out, which the parent
                // would otherwise double-free at scope exit: enclosing owning locals
                // (the scope chain) PLUS owning parameters.  Params are not in the
                // scope chain (they are dropped by the separate param-append pass),
                // so without including them here a param consumed in one arm gets no
                // compensating drop on the sibling arms and leaks there (the mirror
                // of the if/else handler, which already covers params via raw esc).
                let outer: std::collections::HashSet<LocalId> =
                    scope_chain.iter().flatten().copied().chain(owning_params.iter().copied()).collect();
                let mut arm_moves: Vec<(std::collections::HashSet<LocalId>, bool)> = Vec::with_capacity(arms.len());
                for a in arms.iter_mut() {
                    if let Some(g) = &mut a.guard {
                        visit_matches_in_expr(sym, locals, g, scope_chain, loop_start, moved, owning_params);
                    }
                    let (esc, div) = visit_block(sym, locals, &mut a.body, scope_chain, loop_start, &baseline, owning_params);
                    // Outer owning locals this arm moves out (body moves + value moves).
                    let mut mv: std::collections::HashSet<LocalId> = esc.intersection(&outer).copied().collect();
                    if let Some(v) = &mut a.value {
                        // The arm value runs after the body and may move a local out
                        // (`yield s`): drop an arm-declared one from the body's drop
                        // set, and record an outer one as moved by this arm.
                        let mut vmoved = std::collections::HashSet::new();
                        moved_locals_in_expr(sym, v, &mut vmoved);
                        if ty_owns_heap(sym, &v.ty) {
                            if let HExprKind::Local(id) = v.kind { vmoved.insert(id); }
                        }
                        if !vmoved.is_empty() {
                            a.body.heap_to_free.retain(|id| !vmoved.contains(id));
                        }
                        for id in vmoved.intersection(&outer) { mv.insert(*id); }
                        visit_matches_in_expr(sym, locals, v, scope_chain, loop_start, moved, owning_params);
                    }
                    arm_moves.push((mv, div));
                }
                // An outer owning local moved on SOME fall-through arm escapes the
                // match (a diverging arm freed its own at its exit, never reaching
                // the join, so it does not contribute).
                let mut union: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
                for (mv, div) in &arm_moves {
                    if !div { for id in mv { union.insert(*id); } }
                }
                // Compensating drop: on every fall-through arm that does NOT move L,
                // free L at arm exit so L is freed exactly once on every path; the
                // parent's scope-exit drop is then skipped (L is marked moved below),
                // so there is no double-free and no leak.
                for (i, a) in arms.iter_mut().enumerate() {
                    let (mv, div) = &arm_moves[i];
                    if *div { continue; }
                    for &l in &union {
                        if !mv.contains(&l) && !a.body.heap_to_free.contains(&l) {
                            a.body.heap_to_free.push(l);
                        }
                    }
                }
                for id in union { moved.insert(id); }
            }
            HExprKind::Call { args, .. } => {
                for a in args { visit_matches_in_expr(sym, locals, a, scope_chain, loop_start, moved, owning_params); }
            }
            HExprKind::InlineCall { args, propagate_drops, loop_jump_drops, .. } => {
                // A `propagate` inside the inlined body early-returns the CALLER's C
                // frame, so it must free the caller's owning locals live here - the
                // same set a `return` at this point drops (all in-scope owning locals
                // minus those already moved).  scope_chain holds only owning locals.
                let mut drops = Vec::new();
                for scope in scope_chain.iter() {
                    for id in scope.iter().rev() {
                        if moved.contains(id) { continue; }
                        drops.push(*id);
                    }
                }
                *propagate_drops = drops;
                // A caller-targeting `break`/`continue` spliced from the inline exits
                // the CALLER's enclosing loop, so it must free the caller's loop-body
                // owning locals (the scope chain from the loop start), not the whole
                // function.  Only meaningful when this call is inside a loop; if it is
                // not and the inline jumps, the call-site check reports an error.
                let mut ldrops = Vec::new();
                if let Some(start) = loop_start {
                    // The jump targets THIS enclosing loop: free the loop-body locals.
                    for scope in scope_chain.iter().skip(start) {
                        for id in scope.iter().rev() {
                            if moved.contains(id) { continue; }
                            ldrops.push(*id);
                        }
                    }
                } else {
                    // No enclosing loop here: a jump from the inline passes THROUGH
                    // this frame to an outer one, abandoning this whole frame, so free
                    // all its owning locals (like a propagate frame).  For a non-inline
                    // caller this is the call-outside-a-loop case, rejected by
                    // check_inline_loop_jumps; for an intermediate inline frame it is
                    // exactly the chaining that frees that frame's locals on the jump.
                    for scope in scope_chain.iter() {
                        for id in scope.iter().rev() {
                            if moved.contains(id) { continue; }
                            ldrops.push(*id);
                        }
                    }
                }
                *loop_jump_drops = ldrops;
                for a in args { visit_matches_in_expr(sym, locals, a, scope_chain, loop_start, moved, owning_params); }
            }
            HExprKind::CallIndirect { callee, args } => {
                visit_matches_in_expr(sym, locals, callee, scope_chain, loop_start, moved, owning_params);
                for a in args { visit_matches_in_expr(sym, locals, a, scope_chain, loop_start, moved, owning_params); }
            }
            HExprKind::Bin { lhs, rhs, .. } => {
                visit_matches_in_expr(sym, locals, lhs, scope_chain, loop_start, moved, owning_params);
                visit_matches_in_expr(sym, locals, rhs, scope_chain, loop_start, moved, owning_params);
            }
            HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. }
            | HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. }
            | HExprKind::DropWrite(expr) => visit_matches_in_expr(sym, locals, expr, scope_chain, loop_start, moved, owning_params),
            HExprKind::AddrOfRef { place, .. } => visit_matches_in_expr(sym, locals, place, scope_chain, loop_start, moved, owning_params),
            HExprKind::Field { base, .. } | HExprKind::ArrayToSlice { base, .. } => visit_matches_in_expr(sym, locals, base, scope_chain, loop_start, moved, owning_params),
            HExprKind::Index { base, idx } => {
                visit_matches_in_expr(sym, locals, base, scope_chain, loop_start, moved, owning_params);
                visit_matches_in_expr(sym, locals, idx, scope_chain, loop_start, moved, owning_params);
            }
            HExprKind::DerefRef(inner) | HExprKind::HeapAlloc(inner) | HExprKind::Free(inner, _)
            | HExprKind::Transfer(inner) | HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => visit_matches_in_expr(sym, locals, inner, scope_chain, loop_start, moved, owning_params),
            HExprKind::Closure { env_values, .. } => {
                for v in env_values { visit_matches_in_expr(sym, locals, v, scope_chain, loop_start, moved, owning_params); }
            }
            HExprKind::VariantCtor { fields, .. } | HExprKind::Struct { fields, .. } => {
                for (_, fe) in fields { visit_matches_in_expr(sym, locals, fe, scope_chain, loop_start, moved, owning_params); }
            }
            HExprKind::ArrayLit(es) => {
                for e2 in es { visit_matches_in_expr(sym, locals, e2, scope_chain, loop_start, moved, owning_params); }
            }
            _ => {}
        }
    }

    fn visit_block(sym: &SymTab, locals: &[LocalInfo], b: &mut HBlock, scope_chain: &mut Vec<Vec<LocalId>>, loop_start: Option<usize>, inherited: &std::collections::HashSet<LocalId>, owning_params: &std::collections::HashSet<LocalId>) -> (std::collections::HashSet<LocalId>, bool) {
        scope_chain.push(Vec::new());
        // Track moves up to each position in the block.  Start from the moves
        // already in effect at block entry (made by enclosing statements), so a
        // `return` inside a nested block does not re-drop a value the parent
        // already moved out (e.g. an owning temporary push'd into a Vec before a
        // loop, then an early return inside that loop - that would double-free).
        let mut moved: std::collections::HashSet<LocalId> = inherited.clone();
        // Whether control can fall through the END of this block.  A block
        // DIVERGES (cannot fall through) once it hits a return/break/continue or
        // an `if` whose every arm diverges.  A diverging branch never reaches the
        // join after an enclosing `if`, so its escaped moves must not propagate
        // there nor force a compensating drop in a sibling branch - that branch
        // already freed its own locals at its divergent exit.
        let mut diverges = false;
        for s in &mut b.stmts {
            match s {
                HStmt::Let { local, init, .. } => {
                    // Check init for moves
                    moved_locals_in_expr(sym, init, &mut moved);
                    // Fill drops for any match-expr arm bodies inside the init.
                    visit_matches_in_expr(sym, locals, init, scope_chain, loop_start, &mut moved, owning_params);
                    // A direct-Local init of an owning type transfers ownership.
                    if ty_owns_heap(sym, &init.ty) {
                        if let HExprKind::Local(id) = init.kind { moved.insert(id); }
                    }
                    let lty = &locals[local.0 as usize].ty;
                    let schedule = if matches!(lty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        matches!(locals[local.0 as usize].storage, StorageClass::Heap)
                    } else {
                        // A by-value struct/enum/array local that owns heap needs its
                        // recursive drop run at scope exit too.
                        is_owning_value_composite(sym, lty)
                    };
                    if schedule {
                        scope_chain.last_mut().unwrap().push(*local);
                    }
                }
                HStmt::Assign { place, value, drop_old, .. } => {
                    moved_locals_in_expr(sym, value, &mut moved);
                    visit_matches_in_expr(sym, locals, value, scope_chain, loop_start, &mut moved, owning_params);
                    if ty_owns_heap(sym, &value.ty) {
                        if let HExprKind::Local(id) = value.kind { moved.insert(id); }
                    }
                    // The assigned-to local is redefined here, so it is live again -
                    // even if its old value was just consumed into the RHS (e.g.
                    // `x = Cons { tail = x }` or `left = Bin { l = left, ... }`).
                    if let HExprKind::Local(id) = place.kind {
                        // If the owner was already moved out (here or by an earlier
                        // statement / a conditional move), its old value is gone -
                        // codegen must NOT free it before storing the new one, or it
                        // double-frees the moved-out value.  No-runtime-null
                        // counterpart to scope-exit drops skipping moved locals.
                        if moved.contains(&id) { *drop_old = false; }
                        moved.remove(&id);
                    }
                }
                HStmt::ExprStmt(e) => {
                    moved_locals_in_expr(sym, e, &mut moved);
                    visit_matches_in_expr(sym, locals, e, scope_chain, loop_start, &mut moved, owning_params);
                }
                HStmt::Return { value, heap_drops, .. } => {
                    // The set of heap locals to drop at return = union of all scopes minus the return value.
                    let mut returning: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
                    if let Some(v) = value {
                        if ty_owns_heap(sym, &v.ty) {
                            if let HExprKind::Local(id) = v.kind {
                                returning.insert(id);
                            }
                        }
                        moved_locals_in_expr(sym, v, &mut moved);
                        visit_matches_in_expr(sym, locals, v, scope_chain, loop_start, &mut moved, owning_params);
                    }
                    let mut drops = Vec::new();
                    for scope in scope_chain.iter() {
                        for id in scope.iter().rev() {
                            if returning.contains(id) || moved.contains(id) { continue; }
                            drops.push(*id);
                        }
                    }
                    *heap_drops = drops;
                    diverges = true;
                }
                HStmt::If { cond, then_b, else_b, .. } => {
                    visit_matches_in_expr(sym, locals, cond, scope_chain, loop_start, &mut moved, owning_params);
                    let (then_esc, then_div) = visit_block(sym, locals, then_b, scope_chain, loop_start, &mut moved, owning_params);
                    let (else_esc, else_div) = match else_b {
                        Some(eb) => visit_block(sym, locals, eb, scope_chain, loop_start, &moved, owning_params),
                        None => (std::collections::HashSet::new(), false),
                    };
                    // Drop elaboration over the paths that actually FALL THROUGH
                    // to the join after the `if`.  A diverging branch is excluded:
                    // it already freed its own locals at its return/break/continue,
                    // so its moves neither propagate past the if nor force a
                    // compensating drop in the sibling.
                    match (then_div, else_div) {
                        // Both diverge: nothing reaches the join.
                        (true, true) => {}
                        // Only one branch reaches the join; its moves are simply
                        // the post-if moves, no compensation needed.
                        (true, false) => { for id in else_esc { moved.insert(id); } }
                        (false, true) => { for id in then_esc { moved.insert(id); } }
                        // Both fall through.  A local consumed on EVERY path
                        // becomes moved for the parent (no scope-exit drop); a
                        // local consumed on only SOME paths gets a compensating
                        // drop on the paths that did NOT consume it (synthesizing
                        // an else if there is none), so every path frees it exactly
                        // once and the parent's later drop never double-frees it.
                        (false, false) => {
                            let mut union: Vec<LocalId> = then_esc.iter().chain(else_esc.iter()).copied().collect();
                            union.sort_by_key(|id| id.0);
                            union.dedup();
                            for id in union {
                                let in_then = then_esc.contains(&id);
                                let in_else = else_esc.contains(&id);
                                if !(in_then && in_else) {
                                    if !in_then { then_b.heap_to_free.push(id); }
                                    if !in_else {
                                        let eb = else_b.get_or_insert_with(|| HBlock {
                                            stmts: Vec::new(),
                                            heap_to_free: Vec::new(),
                                            ptr_nulls: Vec::new(), stmt_nulls: Vec::new(),
                                            span: then_b.span,
                                        });
                                        eb.heap_to_free.push(id);
                                    }
                                }
                                moved.insert(id);
                            }
                        }
                    }
                    // The `if` diverges only if EVERY arm does (an absent else
                    // falls through, so else_div is false there).
                    if then_div && else_div { diverges = true; }
                }
                HStmt::While { cond, body, .. } => {
                    visit_matches_in_expr(sym, locals, cond, scope_chain, loop_start, &mut moved, owning_params);
                    let _ = visit_block(sym, locals, body, scope_chain, Some(scope_chain.len()), &moved, owning_params);
                }
                // An unconditional nested block always runs, so its escaped moves
                // are moves for the parent too, and its divergence is the parent's.
                HStmt::Block(b) => {
                    let (esc, div) = visit_block(sym, locals, b, scope_chain, loop_start, &mut moved, owning_params);
                    for id in esc { moved.insert(id); }
                    if div { diverges = true; }
                }
                HStmt::Unsafe(b, _) => {
                    let (esc, div) = visit_block(sym, locals, b, scope_chain, loop_start, &mut moved, owning_params);
                    for id in esc { moved.insert(id); }
                    if div { diverges = true; }
                }
                HStmt::Break { heap_drops, .. } | HStmt::Continue { heap_drops, .. } => {
                    // Free owning locals declared inside the enclosing loop before
                    // the jump leaves the loop-body scope chain (same idea as the
                    // return drops, but bounded to the loop body, not the whole fn).
                    if let Some(start) = loop_start {
                        let mut drops = Vec::new();
                        for scope in scope_chain.iter().skip(start).rev() {
                            for id in scope.iter().rev() {
                                if moved.contains(id) { continue; }
                                drops.push(*id);
                            }
                        }
                        *heap_drops = drops;
                    } else {
                        // No enclosing loop in THIS function: a caller-targeting
                        // break/continue inside an `inline` body, which is spliced
                        // into the caller's loop.  It abandons the whole inline
                        // frame, so free every live owning local exactly like a
                        // return does.  (In a non-inline fn a break here is invalid
                        // and rejected elsewhere, so this only fires for inlines.)
                        let mut drops = Vec::new();
                        for scope in scope_chain.iter() {
                            for id in scope.iter().rev() {
                                if moved.contains(id) { continue; }
                                drops.push(*id);
                            }
                        }
                        *heap_drops = drops;
                    }
                    diverges = true;
                }
                HStmt::ForC { cond, body, .. } => {
                    visit_matches_in_expr(sym, locals, cond, scope_chain, loop_start, &mut moved, owning_params);
                    let _ = visit_block(sym, locals, body, scope_chain, Some(scope_chain.len()), &moved, owning_params);
                }
                HStmt::ForEach { src, body, .. } => {
                    visit_matches_in_expr(sym, locals, src, scope_chain, loop_start, &mut moved, owning_params);
                    let _ = visit_block(sym, locals, body, scope_chain, Some(scope_chain.len()), &moved, owning_params);
                }
                HStmt::Propagate { value, heap_drops, .. } => {
                    // `propagate` early-returns the enclosing frame, so it must free
                    // this frame's live owning locals - exactly like a `return`,
                    // minus a propagated bare-local value (which moves out).  Without
                    // this the frame's owning locals leaked on the propagate path.
                    let mut returning: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
                    if let Some(v) = value {
                        if ty_owns_heap(sym, &v.ty) {
                            if let HExprKind::Local(id) = v.kind { returning.insert(id); }
                        }
                        moved_locals_in_expr(sym, v, &mut moved);
                        visit_matches_in_expr(sym, locals, v, scope_chain, loop_start, &mut moved, owning_params);
                    }
                    let mut drops = Vec::new();
                    for scope in scope_chain.iter() {
                        for id in scope.iter().rev() {
                            if returning.contains(id) || moved.contains(id) { continue; }
                            drops.push(*id);
                        }
                    }
                    *heap_drops = drops;
                    diverges = true;
                }
            }
        }
        // Fill heap_to_free: locals declared in this scope, in reverse order, skipping moved.
        let scope = scope_chain.pop().unwrap_or_default();
        let scope_set: std::collections::HashSet<LocalId> = scope.iter().copied().collect();
        let mut to_free = Vec::new();
        for id in scope.iter().rev() {
            if moved.contains(id) { continue; }
            to_free.push(*id);
        }
        b.heap_to_free = to_free;
        // Moves of locals NOT declared in this block escape to the enclosing
        // scope so the caller can reconcile them: a value moved in only some
        // branches of an `if` needs a compensating drop on the paths that did
        // not move it, and a value moved in an unconditional nested block is
        // simply moved for the parent too.  Paired with `diverges` so the caller
        // knows whether this block reaches the join after it.
        let escaped: std::collections::HashSet<LocalId> =
            moved.difference(inherited).copied().filter(|id| !scope_set.contains(id)).collect();
        (escaped, diverges)
    }

    let mut chain = Vec::new();
    let locals = f.locals.clone();
    // Owning params: needed by the match drop-elaboration so a param consumed in
    // one arm gets a compensating drop on the sibling arms (the if/else handler
    // already covers params via raw escaped-move sets; the match handler filters
    // by an `outer` set that must therefore include the owning params too).
    let owning_params: std::collections::HashSet<LocalId> = f.params.iter().copied()
        .filter(|pid| {
            let li = &locals[pid.0 as usize];
            (matches!(li.storage, StorageClass::Heap)
                && matches!(li.ty, HType::Heap { .. } | HType::OwnPtr { .. }))
                || is_owning_value_composite(sym, &li.ty)
        })
        .collect();
    let _ = visit_block(sym, &locals, &mut f.body, &mut chain, None, &std::collections::HashSet::new(), &owning_params);

    // Append owning parameters to the body's heap_to_free so they auto-free at
    // function scope-exit — unless they were transferred out somewhere in the
    // body.  Covers `own *T` / `own &T` (heap-storage) params and by-value
    // struct/enum/array params that transitively own heap.  Without this, an
    // owning param would leak (no drop is otherwise emitted for it; the caller's
    // drop is suppressed because the call site marks the source as moved).
    let mut param_moved: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
    collect_param_moves_block(sym, &f.body, &mut param_moved);
    for &pid in f.params.iter().rev() {
        let li = &locals[pid.0 as usize];
        let owns = (matches!(li.storage, StorageClass::Heap)
                && matches!(li.ty, HType::Heap { .. } | HType::OwnPtr { .. }))
            || is_owning_value_composite(sym, &li.ty);
        if owns && !param_moved.contains(&pid) {
            f.body.heap_to_free.push(pid);
        }
    }
    // Also append undropped params to every Return's heap_drops list — early
    // returns must still free param-owned values.
    append_param_drops_to_returns(sym, &mut f.body, &f.params, &locals, &param_moved);

    // Free the capture env of a non-escaping closure local.  A capturing closure
    // (`let f = int(int n) [x] ...;`) malloc's an env that nothing frees.  We can
    // free it at scope exit ONLY when the closure provably does not escape - it
    // appears solely as the target of an indirect call.  This excludes closures
    // passed to spawn (the fiber frees that env), returned, stored, or aliased,
    // so no double free.
    let mut cands: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
    collect_closure_cands(&f.body, &locals, &mut cands);
    if !cands.is_empty() {
        let mut escaped: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
        mark_closure_escapes_block(&f.body, &cands, &mut escaped);
        let freeable: std::collections::HashSet<LocalId> = cands.difference(&escaped).copied().collect();
        if !freeable.is_empty() {
            add_closure_drops_block(&mut f.body, &freeable);
            // A `return` exits before the declaring block's end, bypassing the
            // block heap_to_free above, so also free in-scope freeable closures
            // at each return (mirrors append_param_drops_to_returns for params).
            add_closure_drops_to_returns(&mut f.body, &freeable, &mut Vec::new());
        }
    }
}

/// For each parameter of `f`, whether it is a `FnPtr` that is "consume-only" -
/// it appears in the body solely as the target of an indirect call (`p(...)`),
/// never stored, returned, aliased, or re-passed.  Such a function can safely
/// free a *one-shot* closure-literal argument's env after the call returns (the
/// callee does not retain it).  Used by codegen to free `apply(<closure>)` envs.
pub fn consume_only_fnptr_params(f: &HFunc) -> Vec<bool> {
    let mut cands: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
    let mut fnptr: Vec<(LocalId, bool)> = Vec::new();
    for &pid in &f.params {
        let is_fp = matches!(f.locals[pid.0 as usize].ty, HType::FnPtr { .. });
        fnptr.push((pid, is_fp));
        if is_fp { cands.insert(pid); }
    }
    let mut escaped: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
    if !cands.is_empty() { mark_closure_escapes_block(&f.body, &cands, &mut escaped); }
    fnptr.iter().map(|(pid, is_fp)| *is_fp && !escaped.contains(pid)).collect()
}

/// Locals bound directly to a capturing closure literal (`let f = ...[caps]...;`).
fn collect_closure_cands(b: &HBlock, locals: &[LocalInfo], out: &mut std::collections::HashSet<LocalId>) {
    for s in &b.stmts {
        if let HStmt::Let { local, init, .. } = s {
            if matches!(&init.kind, HExprKind::Closure { env_values, .. } if !env_values.is_empty())
                && matches!(locals[local.0 as usize].ty, HType::FnPtr { .. }) {
                out.insert(*local);
            }
        }
        for_each_child_block(s, &mut |cb| collect_closure_cands(cb, locals, out));
    }
}

/// Mark a candidate closure local as escaped if it appears anywhere other than as
/// the callee of an indirect call (i.e. anything but `f(...)`).
///
/// Exception (borrow, not escape): capturing a candidate `c` into the env of
/// ANOTHER candidate closure `d` (`let d = ...[c]...;`) does not escape `c` on
/// its own - `d`'s env merely holds a pointer to `c`'s env, and `d` is freeable
/// only if it itself does not escape.  Since `c` is in scope wherever `d` is
/// declared, `c`'s env always outlives `d`'s, so freeing `c` at its own scope is
/// sound.  We record `c -> d` as a deferred edge and let `c` escape only if `d`
/// transitively escapes (fixpoint below).  This is what lets a non-escaping
/// nested closure (`outer` capturing `inner`) free both envs at scope exit
/// instead of leaking `inner`'s.
fn mark_closure_escapes_block(b: &HBlock, cands: &std::collections::HashSet<LocalId>, escaped: &mut std::collections::HashSet<LocalId>) {
    fn ex(e: &HExpr, cands: &std::collections::HashSet<LocalId>, escaped: &mut std::collections::HashSet<LocalId>) {
        match &e.kind {
            HExprKind::Local(id) => { if cands.contains(id) { escaped.insert(*id); } }
            HExprKind::CallIndirect { callee, args } => {
                // `f(...)` - the callee being a bare candidate local is a call, not
                // an escape; still scan a non-trivial callee and all args.
                if !matches!(&callee.kind, HExprKind::Local(_)) { ex(callee, cands, escaped); }
                for a in args { ex(a, cands, escaped); }
            }
            HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => { for a in args { ex(a, cands, escaped); } }
            HExprKind::Bin { lhs, rhs, .. } => { ex(lhs, cands, escaped); ex(rhs, cands, escaped); }
            HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } | HExprKind::Cast { expr, .. }
            | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr)
            | HExprKind::HeapAlloc(expr) | HExprKind::Free(expr, _) | HExprKind::SliceLen(expr)
            | HExprKind::EnumTag(expr) | HExprKind::ArrayToSlice { base: expr, .. } | HExprKind::Transfer(expr) => ex(expr, cands, escaped),
            HExprKind::AddrOfRef { place, .. } => ex(place, cands, escaped),
            HExprKind::Field { base, .. } => ex(base, cands, escaped),
            HExprKind::Index { base, idx } => { ex(base, cands, escaped); ex(idx, cands, escaped); }
            HExprKind::Closure { env_values, .. } => { for v in env_values { ex(v, cands, escaped); } }
            HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => { for (_, fe) in fields { ex(fe, cands, escaped); } }
            HExprKind::ArrayLit(es) => { for e in es { ex(e, cands, escaped); } }
            HExprKind::Match { scrutinee, arms, .. } => {
                ex(scrutinee, cands, escaped);
                for a in arms {
                    if let Some(g) = &a.guard { ex(g, cands, escaped); }
                    if let Some(v) = &a.value { ex(v, cands, escaped); }
                }
            }
            _ => {}
        }
    }
    // Exhaustive statement walk: every expression position must be scanned, or a
    // missed use would leave an escaped closure wrongly freeable.  `edges`
    // collects deferred `c -> d` capture borrows (see the function doc).
    fn scan_stmt(
        s: &HStmt,
        cands: &std::collections::HashSet<LocalId>,
        escaped: &mut std::collections::HashSet<LocalId>,
        edges: &mut Vec<(LocalId, LocalId)>,
    ) {
        match s {
            HStmt::Let { local, init, .. } => {
                // `let d = ...[c]...;` where both d and c are candidate closures:
                // record c -> d as a deferred borrow instead of escaping c.  Any
                // non-candidate capture or nested expr in the env still escapes
                // normally via ex().
                if cands.contains(local) {
                    if let HExprKind::Closure { env_values, .. } = &init.kind {
                        for v in env_values {
                            if let HExprKind::Local(c) = &v.kind {
                                if cands.contains(c) { edges.push((*c, *local)); continue; }
                            }
                            ex(v, cands, escaped);
                        }
                        return;
                    }
                }
                ex(init, cands, escaped);
            }
            HStmt::Assign { place, value, .. } => { ex(place, cands, escaped); ex(value, cands, escaped); }
            HStmt::ExprStmt(e) => ex(e, cands, escaped),
            HStmt::Return { value, .. } => { if let Some(v) = value { ex(v, cands, escaped); } }
            HStmt::Propagate { value, .. } => { if let Some(v) = value { ex(v, cands, escaped); } }
            HStmt::If { cond, then_b, else_b, .. } => {
                ex(cond, cands, escaped);
                for st in &then_b.stmts { scan_stmt(st, cands, escaped, edges); }
                if let Some(b) = else_b { for st in &b.stmts { scan_stmt(st, cands, escaped, edges); } }
            }
            HStmt::While { cond, body, .. } => { ex(cond, cands, escaped); for st in &body.stmts { scan_stmt(st, cands, escaped, edges); } }
            HStmt::Block(b) | HStmt::Unsafe(b, _) => { for st in &b.stmts { scan_stmt(st, cands, escaped, edges); } }
            HStmt::ForC { init, cond, step, body, .. } => {
                scan_stmt(init, cands, escaped, edges); ex(cond, cands, escaped); scan_stmt(step, cands, escaped, edges);
                for st in &body.stmts { scan_stmt(st, cands, escaped, edges); }
            }
            HStmt::ForEach { src, body, .. } => { ex(src, cands, escaped); for st in &body.stmts { scan_stmt(st, cands, escaped, edges); } }
            HStmt::Break { .. } | HStmt::Continue { .. } => {}
        }
    }
    let mut edges: Vec<(LocalId, LocalId)> = Vec::new();
    for s in &b.stmts { scan_stmt(s, cands, escaped, &mut edges); }
    // Fixpoint: a deferred capture `c -> d` escapes `c` only if `d` escapes.
    // Iterate until stable so chains (outer captures middle captures inner)
    // propagate fully.
    loop {
        let mut changed = false;
        for (c, d) in &edges {
            if escaped.contains(d) && !escaped.contains(c) { escaped.insert(*c); changed = true; }
        }
        if !changed { break; }
    }
}

/// Add freeable closure locals to the heap_to_free of the block that declares them.
fn add_closure_drops_block(b: &mut HBlock, freeable: &std::collections::HashSet<LocalId>) {
    let mut to_add: Vec<LocalId> = Vec::new();
    for s in &b.stmts {
        if let HStmt::Let { local, .. } = s {
            if freeable.contains(local) && !b.heap_to_free.contains(local) { to_add.push(*local); }
        }
    }
    for id in to_add { b.heap_to_free.push(id); }
    for s in b.stmts.iter_mut() {
        for_each_child_block_mut(s, &mut |cb| add_closure_drops_block(cb, freeable));
    }
}

/// Add freeable closure locals to the `heap_drops` of every `return` that lies
/// within their scope (after their declaration), so a function that returns
/// before its block ends still frees the closure env.  `in_scope` is the stack
/// of freeable closures declared by enclosing blocks, in declaration order.
fn add_closure_drops_to_returns(
    b: &mut HBlock,
    freeable: &std::collections::HashSet<LocalId>,
    in_scope: &mut Vec<LocalId>,
) {
    let mark = in_scope.len();
    for s in b.stmts.iter_mut() {
        match s {
            HStmt::Let { local, .. } => {
                if freeable.contains(local) { in_scope.push(*local); }
            }
            HStmt::Return { value, heap_drops, .. } => {
                // A closure returned as the value escapes (and is excluded from
                // `freeable`), but guard against dropping the returned local anyway.
                let ret_local = match value {
                    Some(v) => if let HExprKind::Local(id) = v.kind { Some(id) } else { None },
                    None => None,
                };
                for &c in in_scope.iter() {
                    if Some(c) != ret_local && !heap_drops.contains(&c) {
                        heap_drops.push(c);
                    }
                }
            }
            HStmt::If { then_b, else_b, .. } => {
                add_closure_drops_to_returns(then_b, freeable, in_scope);
                if let Some(eb) = else_b { add_closure_drops_to_returns(eb, freeable, in_scope); }
            }
            HStmt::While { body, .. } | HStmt::Block(body) | HStmt::Unsafe(body, _)
            | HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => {
                add_closure_drops_to_returns(body, freeable, in_scope);
            }
            _ => {}
        }
    }
    in_scope.truncate(mark);
}

/// Run `g` on each immediate child block of a statement (read-only).
// A `match`/`if`-expr arm BODY is a real child block, but it lives inside an
// EXPRESSION, so the statement-level cases below would miss it.  These yield every
// match-arm body reachable through a statement's expressions so block-walking
// passes (closure-capture collection, freeable-closure drops, the cross-thread
// borrow check) also descend into arm bodies - otherwise e.g. a closure declared
// in an arm body never gets its env freed (leak).
fn each_arm_body_in_expr(e: &HExpr, g: &mut dyn FnMut(&HBlock)) {
    match &e.kind {
        HExprKind::Match { scrutinee, arms, .. } => {
            each_arm_body_in_expr(scrutinee, g);
            for a in arms {
                if let Some(gd) = &a.guard { each_arm_body_in_expr(gd, g); }
                g(&a.body);
                if let Some(v) = &a.value { each_arm_body_in_expr(v, g); }
            }
        }
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => { for a in args { each_arm_body_in_expr(a, g); } }
        HExprKind::CallIndirect { callee, args } => { each_arm_body_in_expr(callee, g); for a in args { each_arm_body_in_expr(a, g); } }
        HExprKind::Bin { lhs, rhs, .. } => { each_arm_body_in_expr(lhs, g); each_arm_body_in_expr(rhs, g); }
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } | HExprKind::Cast { expr, .. }
        | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) => each_arm_body_in_expr(expr, g),
        HExprKind::AddrOfRef { place, .. } => each_arm_body_in_expr(place, g),
        HExprKind::Field { base, .. } | HExprKind::ArrayToSlice { base, .. } => each_arm_body_in_expr(base, g),
        HExprKind::Index { base, idx } => { each_arm_body_in_expr(base, g); each_arm_body_in_expr(idx, g); }
        HExprKind::DerefRef(inner) | HExprKind::HeapAlloc(inner) | HExprKind::Free(inner, _)
        | HExprKind::Transfer(inner) | HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => each_arm_body_in_expr(inner, g),
        HExprKind::Closure { env_values, .. } => { for v in env_values { each_arm_body_in_expr(v, g); } }
        HExprKind::VariantCtor { fields, .. } | HExprKind::Struct { fields, .. } => { for (_, fe) in fields { each_arm_body_in_expr(fe, g); } }
        HExprKind::ArrayLit(es) => { for e2 in es { each_arm_body_in_expr(e2, g); } }
        _ => {}
    }
}
fn each_arm_body_in_expr_mut(e: &mut HExpr, g: &mut dyn FnMut(&mut HBlock)) {
    match &mut e.kind {
        HExprKind::Match { scrutinee, arms, .. } => {
            each_arm_body_in_expr_mut(scrutinee, g);
            for a in arms.iter_mut() {
                if let Some(gd) = &mut a.guard { each_arm_body_in_expr_mut(gd, g); }
                g(&mut a.body);
                if let Some(v) = &mut a.value { each_arm_body_in_expr_mut(v, g); }
            }
        }
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => { for a in args { each_arm_body_in_expr_mut(a, g); } }
        HExprKind::CallIndirect { callee, args } => { each_arm_body_in_expr_mut(callee, g); for a in args { each_arm_body_in_expr_mut(a, g); } }
        HExprKind::Bin { lhs, rhs, .. } => { each_arm_body_in_expr_mut(lhs, g); each_arm_body_in_expr_mut(rhs, g); }
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } | HExprKind::Cast { expr, .. }
        | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) => each_arm_body_in_expr_mut(expr, g),
        HExprKind::AddrOfRef { place, .. } => each_arm_body_in_expr_mut(place, g),
        HExprKind::Field { base, .. } | HExprKind::ArrayToSlice { base, .. } => each_arm_body_in_expr_mut(base, g),
        HExprKind::Index { base, idx } => { each_arm_body_in_expr_mut(base, g); each_arm_body_in_expr_mut(idx, g); }
        HExprKind::DerefRef(inner) | HExprKind::HeapAlloc(inner) | HExprKind::Free(inner, _)
        | HExprKind::Transfer(inner) | HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => each_arm_body_in_expr_mut(inner, g),
        HExprKind::Closure { env_values, .. } => { for v in env_values { each_arm_body_in_expr_mut(v, g); } }
        HExprKind::VariantCtor { fields, .. } | HExprKind::Struct { fields, .. } => { for (_, fe) in fields { each_arm_body_in_expr_mut(fe, g); } }
        HExprKind::ArrayLit(es) => { for e2 in es { each_arm_body_in_expr_mut(e2, g); } }
        _ => {}
    }
}
fn for_each_child_block(s: &HStmt, g: &mut dyn FnMut(&HBlock)) {
    match s {
        HStmt::If { cond, then_b, else_b, .. } => { each_arm_body_in_expr(cond, g); g(then_b); if let Some(b) = else_b { g(b); } }
        HStmt::While { cond, body, .. } => { each_arm_body_in_expr(cond, g); g(body); }
        HStmt::Block(body) | HStmt::Unsafe(body, _) => g(body),
        HStmt::ForC { cond, body, .. } => { each_arm_body_in_expr(cond, g); g(body); }
        HStmt::ForEach { src, body, .. } => { each_arm_body_in_expr(src, g); g(body); }
        HStmt::Let { init, .. } => each_arm_body_in_expr(init, g),
        HStmt::Assign { place, value, .. } => { each_arm_body_in_expr(place, g); each_arm_body_in_expr(value, g); }
        HStmt::ExprStmt(e) => each_arm_body_in_expr(e, g),
        HStmt::Return { value: Some(v), .. } | HStmt::Propagate { value: Some(v), .. } => each_arm_body_in_expr(v, g),
        _ => {}
    }
}
fn for_each_child_block_mut(s: &mut HStmt, g: &mut dyn FnMut(&mut HBlock)) {
    match s {
        HStmt::If { cond, then_b, else_b, .. } => { each_arm_body_in_expr_mut(cond, g); g(then_b); if let Some(b) = else_b { g(b); } }
        HStmt::While { cond, body, .. } => { each_arm_body_in_expr_mut(cond, g); g(body); }
        HStmt::Block(body) | HStmt::Unsafe(body, _) => g(body),
        HStmt::ForC { cond, body, .. } => { each_arm_body_in_expr_mut(cond, g); g(body); }
        HStmt::ForEach { src, body, .. } => { each_arm_body_in_expr_mut(src, g); g(body); }
        HStmt::Let { init, .. } => each_arm_body_in_expr_mut(init, g),
        HStmt::Assign { place, value, .. } => { each_arm_body_in_expr_mut(place, g); each_arm_body_in_expr_mut(value, g); }
        HStmt::ExprStmt(e) => each_arm_body_in_expr_mut(e, g),
        HStmt::Return { value: Some(v), .. } | HStmt::Propagate { value: Some(v), .. } => each_arm_body_in_expr_mut(v, g),
        _ => {}
    }
}

fn collect_param_moves_block(sym: &SymTab, b: &HBlock, out: &mut std::collections::HashSet<LocalId>) {
    for s in &b.stmts { collect_param_moves_stmt(sym, s, out); }
}
fn collect_param_moves_stmt(sym: &SymTab, s: &HStmt, out: &mut std::collections::HashSet<LocalId>) {
    match s {
        HStmt::Let { init, .. } => {
            // A direct owning-Local init moves it (`own *N cur = head;`).
            if ty_owns_heap(sym, &init.ty) {
                if let HExprKind::Local(id) = init.kind { out.insert(id); }
            }
            collect_param_moves_expr(sym, init, out);
        }
        HStmt::Assign { value, .. } => {
            // Assigning an owning local into a slot moves it (`x.next = prev;`).
            if ty_owns_heap(sym, &value.ty) {
                if let HExprKind::Local(id) = value.kind { out.insert(id); }
            }
            collect_param_moves_expr(sym, value, out);
        }
        HStmt::ExprStmt(e) => collect_param_moves_expr(sym, e, out),
        HStmt::Return { value: Some(v), .. } => {
            // NOTE: do NOT mark a param moved just because it is returned here.
            // `param_moved` is function-wide, but a return-move is path-specific:
            // a param returned on one branch must still be dropped on a sibling
            // branch that returns something else.  append_param_drops_to_returns
            // already excludes the per-return `returning_id`, so marking it here
            // would leak the param on every other return path (e.g. list delete
            // that does `return rest;` on the match and `return head;` otherwise).
            collect_param_moves_expr(sym, v, out);
        }
        HStmt::Return { .. } => {}
        HStmt::If { cond, then_b, else_b, .. } => {
            collect_param_moves_expr(sym, cond, out);
            collect_param_moves_block(sym, then_b, out);
            if let Some(b) = else_b { collect_param_moves_block(sym, b, out); }
        }
        HStmt::While { cond, body, .. } => {
            collect_param_moves_expr(sym, cond, out);
            collect_param_moves_block(sym, body, out);
        }
        HStmt::Block(b) | HStmt::Unsafe(b, _) => collect_param_moves_block(sym, b, out),
        HStmt::ForC { init, cond, step, body, .. } => {
            collect_param_moves_stmt(sym, init, out);
            collect_param_moves_expr(sym, cond, out);
            collect_param_moves_stmt(sym, step, out);
            collect_param_moves_block(sym, body, out);
        }
        HStmt::ForEach { src, body, .. } => {
            collect_param_moves_expr(sym, src, out);
            collect_param_moves_block(sym, body, out);
        }
        HStmt::Propagate { value: Some(v), .. } => collect_param_moves_expr(sym, v, out),
        HStmt::Propagate { value: None, .. } => {}
        HStmt::Break { .. } | HStmt::Continue { .. } => {}
    }
}
fn collect_param_moves_expr(sym: &SymTab, e: &HExpr, out: &mut std::collections::HashSet<LocalId>) {
    match &e.kind {
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => {
            for a in args {
                if ty_owns_heap(sym, &a.ty) {
                    if let HExprKind::Local(id) = a.kind { out.insert(id); }
                }
                collect_param_moves_expr(sym, a, out);
            }
        }
        HExprKind::Bin { lhs, rhs, .. } => {
            collect_param_moves_expr(sym, lhs, out);
            collect_param_moves_expr(sym, rhs, out);
        }
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } => collect_param_moves_expr(sym, expr, out),
        HExprKind::AddrOfRef { place, .. } => collect_param_moves_expr(sym, place, out),
        HExprKind::Field { base, .. } => collect_param_moves_expr(sym, base, out),
        HExprKind::Index { base, idx } => {
            collect_param_moves_expr(sym, base, out);
            collect_param_moves_expr(sym, idx, out);
        }
        HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. }
        | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr) => collect_param_moves_expr(sym, expr, out),
        HExprKind::ArrayToSlice { base, .. } => collect_param_moves_expr(sym, base, out),
        HExprKind::HeapAlloc(inner) | HExprKind::Free(inner, _) => collect_param_moves_expr(sym, inner, out),
        HExprKind::CallIndirect { callee, args } => {
            collect_param_moves_expr(sym, callee, out);
            for a in args { collect_param_moves_expr(sym, a, out); }
        }
        HExprKind::Closure { env_values, .. } => for v in env_values { collect_param_moves_expr(sym, v, out); },
        HExprKind::Transfer(inner) => {
            if let Some(id) = root_local(inner) { out.insert(id); }
            collect_param_moves_expr(sym, inner, out);
        }
        HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => collect_param_moves_expr(sym, inner, out),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => {
            // Storing an owning local/param into a struct or variant field moves it
            // into the aggregate (e.g. `alloc Node { next = tl }` consumes `tl`).
            // Mirror the call-argument arm so the moved value is not also dropped.
            for (_, fe) in fields {
                if ty_owns_heap(sym, &fe.ty) {
                    if let HExprKind::Local(id) = fe.kind { out.insert(id); }
                }
                collect_param_moves_expr(sym, fe, out);
            }
        }
        HExprKind::ArrayLit(es) => for e in es { collect_param_moves_expr(sym, e, out); },
        // A `match` lowers to an expression, so a param consumed inside it
        // (scrutinee, a guard, an arm value, OR an arm body statement) was
        // invisible to the param-move scan - the missing arm meant the param
        // fell through to `_ => {}`.  append_param_drops_to_returns would then
        // auto-drop it at the function's return even though an arm already
        // consumed it -> double free.  Recurse every match position; the arm
        // body is a real statement block, so descend it via the block walker
        // (this path has no separate visit_matches mechanism like the main flow).
        HExprKind::Match { scrutinee, arms, .. } => {
            collect_param_moves_expr(sym, scrutinee, out);
            for a in arms {
                if let Some(g) = &a.guard { collect_param_moves_expr(sym, g, out); }
                collect_param_moves_block(sym, &a.body, out);
                if let Some(v) = &a.value { collect_param_moves_expr(sym, v, out); }
            }
        }
        _ => {}
    }
}

fn append_param_drops_to_returns(
    sym: &SymTab,
    b: &mut HBlock,
    params: &[LocalId],
    locals: &[LocalInfo],
    param_moved: &std::collections::HashSet<LocalId>,
) {
    for s in b.stmts.iter_mut() {
        match s {
            HStmt::Return { value, heap_drops, .. } => {
                let returning_id = value.as_ref().and_then(|v| {
                    if ty_owns_heap(sym, &v.ty) {
                        if let HExprKind::Local(id) = v.kind { Some(id) } else { None }
                    } else { None }
                });
                for &pid in params.iter().rev() {
                    let li = &locals[pid.0 as usize];
                    let owns = (matches!(li.storage, StorageClass::Heap)
                            && matches!(li.ty, HType::Heap { .. } | HType::OwnPtr { .. }))
                        || is_owning_value_composite(sym, &li.ty);
                    if owns
                        && !param_moved.contains(&pid)
                        && returning_id != Some(pid)
                        && !heap_drops.contains(&pid)
                    {
                        heap_drops.push(pid);
                    }
                }
            }
            HStmt::If { then_b, else_b, .. } => {
                append_param_drops_to_returns(sym, then_b, params, locals, param_moved);
                if let Some(eb) = else_b { append_param_drops_to_returns(sym, eb, params, locals, param_moved); }
            }
            HStmt::While { body, .. } | HStmt::Block(body) | HStmt::Unsafe(body, _) => {
                append_param_drops_to_returns(sym, body, params, locals, param_moved);
            }
            HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => {
                append_param_drops_to_returns(sym, body, params, locals, param_moved);
            }
            _ => {}
        }
    }
}
