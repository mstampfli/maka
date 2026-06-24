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
        }
    }
}

/// Result of one function's lifetime/null-proof analysis.
pub struct AnalyzeOk {
    pub warnings: Vec<SemaWarning>,
    /// `(lifted FuncId, LocalId inside lifted body)` pairs harvested from
    /// `Closure` expressions in this function whose env_value was provably
    /// non-null at the capture site.  The driver applies these to the
    /// corresponding lifted closure's initial state before its own pass runs.
    pub capture_nonnull: Vec<(FuncId, LocalId)>,
}

pub fn analyze_func(
    sym: &SymTab,
    f: &mut HFunc,
    summaries: &[bool],
    initial_nonnull: &[LocalId],
) -> Result<AnalyzeOk, Vec<SemaError>> {
    let mut a = Analyzer::new(sym, f, summaries);
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
    let capture_nonnull = std::mem::take(&mut a.capture_nonnull);
    // Hoist owning temporaries consumed by a borrowing context into hidden
    // owning locals, so the auto-free machinery (below) drops them.
    hoist_owning_temps(sym, f);
    // Final: walk the body and fill heap_to_free for each block in reverse scope-decl order.
    fill_heap_drops(sym, f);
    if errors.is_empty() {
        Ok(AnalyzeOk { warnings, capture_nonnull })
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
pub fn compute_return_summaries(sym: &SymTab, funcs: &[HFunc]) -> Vec<bool> {
    let n = sym.sigs.len();
    let mut summaries: Vec<bool> = (0..n).map(|i| return_type_nonnull(&sym.sigs[i].ret)).collect();
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
        HStmt::Break(_) | HStmt::Continue(_) => WalkRes { ok: true, terminates: true },
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
            summaries.get(callee.0 as usize).copied().unwrap_or(false)
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
        HStmt::Propagate { value: None, .. } | HStmt::Break(_) | HStmt::Continue(_) => {}
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
            | HExprKind::Free(inner) | HExprKind::Transfer(inner) | HExprKind::SliceLen(inner)
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

struct Analyzer<'a> {
    #[allow(dead_code)]
    sym: &'a SymTab,
    f: *mut HFunc,
    /// One LocalState per LocalId.
    state: Vec<LocalState>,
    errors: Vec<SemaError>,
    /// Non-fatal diagnostics — surfaced when an auto-nulled pointer is observed
    /// at a use site without intervening re-assignment on every code path.
    warnings: Vec<SemaWarning>,
    /// Interprocedural "never-returns-null" facts, indexed by `FuncId.0`.
    /// Populated by `compute_return_summaries` before the per-func pass runs.
    /// `expr_nonnull` consults this on `Call` / `InlineCall` arms.
    summaries: &'a [bool],
    /// Capture-site non-null facts harvested while walking `Closure` exprs:
    /// `(lifted FuncId, LocalId inside the lifted body)` pairs whose
    /// corresponding `env_value` was provably non-null at the capture site.
    /// The driver applies these as the lifted function's initial state before
    /// running its per-func pass — bridges (§6.3) capture flow into the
    /// synthesized closure body.
    capture_nonnull: Vec<(FuncId, LocalId)>,
}

impl<'a> Analyzer<'a> {
    fn new(sym: &'a SymTab, f: &mut HFunc, summaries: &'a [bool]) -> Self {
        let n = f.locals.len();
        Self {
            sym,
            f: f as *mut _,
            state: (0..n).map(|_| LocalState::fresh()).collect(),
            errors: Vec::new(),
            warnings: Vec::new(),
            summaries,
            capture_nonnull: Vec::new(),
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
        let mut declared_here: Vec<LocalId> = Vec::new();
        // Locals that have been narrowed by an early-exit guard like
        // `if (p == null) { return; }` — they revert to nullable at block exit.
        let mut guarded_here: Vec<LocalId> = Vec::new();

        for stmt in &mut b.stmts {
            self.walk_stmt(stmt, &mut declared_here, live_outer, depth);
            if let Some(p) = self.detect_guard_return(stmt) {
                if self.state[p.0 as usize].narrowed_until.is_none() {
                    self.state[p.0 as usize].narrowed_until = Some(depth);
                    guarded_here.push(p);
                }
            }
        }
        // Revert early-exit narrowing at block exit.
        for p in &guarded_here {
            self.state[p.0 as usize].narrowed_until = None;
        }

        // Scope exit: kill all LIDs declared here in reverse order.
        let mut collapsed: Vec<LocalId> = Vec::new();
        for id in declared_here.iter().rev() {
            let nulled = self.kill_lid(*id, b.span);
            for n in nulled { if !collapsed.contains(&n) { collapsed.push(n); } }
        }
        // Only null pointers that were declared OUTSIDE this scope (otherwise they're going away anyway).
        let outer_collapsed: Vec<LocalId> = collapsed.into_iter()
            .filter(|p| !declared_here.contains(p))
            .collect();
        b.ptr_nulls = outer_collapsed;
        let _ = depth;
        let _ = live_outer;
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
                declared_here.push(id);

                // Move semantics: `heap T b = a;` or `own *T b = a;` moves `a`.
                if matches!(li_ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                    if let HExprKind::Local(src) = init.kind {
                        let src_ty = self.f().locals[src.0 as usize].ty.clone();
                        if matches!(src_ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                            self.mark_moved(src, *span);
                        }
                    }
                }
            }
            HStmt::Assign { op, place, value, span } => {
                let _ = op;
                // Conservative escape check on assigning into a struct field:
                // (a) explicit `b.p = &local` — caught by check_no_local_ref_escape;
                // (b) `b.p = borrow_param` — without lifetime annotations we don't
                //     know if `borrow_param`'s source outlives `b`.  When the place
                //     reaches back to a parameter, `b` survives the call, so the
                //     stash would escape.  Reject conservatively.
                if matches!(place.kind, HExprKind::Field { .. } | HExprKind::Index { .. } | HExprKind::Unwrap { .. }) {
                    self.check_no_local_ref_escape(value);
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
                        let _ = self.kill_lid(id, *span);
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
                    self.check_no_local_ref_escape(v);
                }
                let _ = heap_drops;
            }
            HStmt::If { cond, then_b, else_b, span } => {
                self.walk_expr(cond);
                // narrowing: detect `p != null` or `null != p` for an immediate Local(p)
                let then_narrow = self.detect_not_null_narrow(cond);
                let else_narrow = self.detect_is_null_narrow(cond);

                // Snapshot state for branch join
                let snap = self.snapshot();

                // then branch
                if let Some(p) = then_narrow {
                    self.state[p.0 as usize].narrowed_until = Some(depth + 1);
                }
                self.walk_block(then_b, live_outer, depth + 1);
                if let Some(p) = then_narrow {
                    self.state[p.0 as usize].narrowed_until = None;
                }
                let then_state = self.snapshot();
                self.restore(snap.clone());

                // else branch
                if let Some(b) = else_b {
                    if let Some(p) = else_narrow {
                        self.state[p.0 as usize].narrowed_until = Some(depth + 1);
                    }
                    self.walk_block(b, live_outer, depth + 1);
                    if let Some(p) = else_narrow {
                        self.state[p.0 as usize].narrowed_until = None;
                    }
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
                    self.join_branches(&then_state, &else_state, *span, &in_scope);
                }
            }
            HStmt::While { cond, body, span } => {
                let _ = span;
                self.walk_expr(cond);
                // narrow inside the body when the condition is `p != null`
                let body_narrow = self.detect_not_null_narrow(cond);
                if let Some(p) = body_narrow {
                    self.state[p.0 as usize].narrowed_until = Some(depth + 1);
                }
                // For simplicity we don't do a fixpoint; we walk once.
                self.walk_block(body, live_outer, depth + 1);
                if let Some(p) = body_narrow {
                    self.state[p.0 as usize].narrowed_until = None;
                }
            }
            HStmt::Block(b) => self.walk_block(b, live_outer, depth + 1),
            HStmt::Unsafe(b, _) => self.walk_block(b, live_outer, depth + 1),
            HStmt::Break(_) | HStmt::Continue(_) => {}
            HStmt::ForC { init, cond, step, body, .. } => {
                self.walk_stmt(init, declared_here, live_outer, depth);
                self.walk_expr(cond);
                self.walk_stmt(step, declared_here, live_outer, depth);
                self.walk_block(body, live_outer, depth + 1);
            }
            HStmt::ForEach { src, body, .. } => {
                self.walk_expr(src);
                self.walk_block(body, live_outer, depth + 1);
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
            HExprKind::Call { args, .. } => {
                for a in args {
                    // Detect move-in for heap parameters
                    self.walk_expr(a);
                    if matches!(a.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        if let HExprKind::Local(id) = a.kind {
                            self.mark_moved(id, a.span);
                        }
                    }
                }
            }
            HExprKind::Cast { expr, .. } => self.walk_expr(expr),
            HExprKind::CheckedCast { expr, .. } => self.walk_expr(expr),
            HExprKind::Struct { fields, .. } => for (_, fe) in fields { self.walk_expr(fe); },
            HExprKind::ArrayLit(elems) => for e in elems { self.walk_expr(e); },
            HExprKind::DropWrite(inner) => self.walk_expr(inner),
            HExprKind::ArrayToSlice { base, .. } => self.walk_expr(base),
            HExprKind::DerefRef(inner) => self.walk_expr(inner),
            HExprKind::HeapAlloc(inner) => self.walk_expr(inner),
            HExprKind::Free(inner) => self.walk_expr(inner),
            HExprKind::CallIndirect { callee, args } => {
                self.walk_expr(callee);
                for a in args { self.walk_expr(a); }
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
                    if matches!(a.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        if let HExprKind::Local(id) = a.kind {
                            self.mark_moved(id, a.span);
                        }
                    }
                }
            }
            HExprKind::Closure { env_values, lifted, capture_lids, .. } => {
                for (i, v) in env_values.iter_mut().enumerate() {
                    self.walk_expr(v);
                    // Propagate capture-site non-null facts into the lifted body
                    // so its per-func pass starts with `state[capture_lid]` proven
                    // when the captured expression is provably non-null here.
                    if let Some(lid) = capture_lids.get(i) {
                        if self.expr_nonnull(v) {
                            self.capture_nonnull.push((*lifted, *lid));
                        }
                    }
                }
            }
            HExprKind::Transfer(inner) => {
                // Treat the source-local as moved (use after this point is a compile error).
                self.walk_expr(inner);
                if let Some(id) = root_local(inner) {
                    let span = inner.span;
                    self.mark_moved(id, span);
                }
            }
            HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => self.walk_expr(inner),
            HExprKind::FnRef(_) => {}
            HExprKind::VariantCtor { fields, .. } => for (_, fe) in fields { self.walk_expr(fe); },
            HExprKind::Match { scrutinee, arms, .. } => {
                self.walk_expr(scrutinee);
                for a in arms {
                    if let Some(g) = &mut a.guard.clone() { self.walk_expr(g); }
                    if let Some(v) = &mut a.value.clone() { self.walk_expr(v); }
                    // Body stmts walked separately via the block.
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
    fn check_no_local_ref_escape(&mut self, e: &HExpr) {
        use HExprKind::*;
        match &e.kind {
            AddrOfRef { place, .. } => {
                if let Some(root) = root_local(place) {
                    let li = &self.f().locals[root.0 as usize];
                    // Stack locals AND value-class parameters both die when the
                    // function returns — escaping a borrow of either is unsafe.
                    if matches!(li.storage, StorageClass::Stack | StorageClass::Param) {
                        let name = li.name.clone();
                        let span = e.span;
                        self.err(
                            format!(
                                "reference to local `{}` escapes the function via the returned value — the local dies on return, so the caller would observe a dangling reference",
                                name
                            ),
                            span,
                        );
                    }
                }
            }
            Struct { fields, .. } | VariantCtor { fields, .. } => {
                for (_, fe) in fields { self.check_no_local_ref_escape(fe); }
            }
            ArrayLit(elems) => for el in elems { self.check_no_local_ref_escape(el); }
            HeapAlloc(inner) | Free(inner) => self.check_no_local_ref_escape(inner),
            Cast { expr, .. } | CheckedCast { expr, .. } | DerefRef(expr) | DropWrite(expr)
                | Un { expr, .. } | Unwrap { expr, .. } => self.check_no_local_ref_escape(expr),
            Bin { lhs, rhs, .. } => {
                self.check_no_local_ref_escape(lhs);
                self.check_no_local_ref_escape(rhs);
            }
            _ => {}
        }
    }

    fn mark_moved(&mut self, id: LocalId, sp: Span) {
        // Heap-typed bindings honor the v1.2 move semantics; explicit `transfer X` invalidates any binding.
        let name = self.f().locals[id.0 as usize].name.clone();
        if self.state[id.0 as usize].moved {
            self.err(format!("use of moved value `{}`", name), sp);
            return;
        }
        self.state[id.0 as usize].moved = true;
    }

    /// Compute the deps set of an expression.
    fn expr_deps(&self, e: &HExpr) -> HashSet<u32> {
        let mut out = HashSet::new();
        self.collect_deps(e, &mut out);
        out
    }

    /// Is this RHS expression statically known to produce a non-null pointer?
    /// True for `heap T(...)` and for locals already known non-null.
    fn expr_nonnull(&self, e: &HExpr) -> bool {
        match &e.kind {
            HExprKind::HeapAlloc(_) => true,
            HExprKind::AddrOfRef { .. } => true,
            HExprKind::Local(id) => {
                let st = &self.state[id.0 as usize];
                st.known_nonnull || st.narrowed_until.is_some()
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
                self.summaries.get(callee.0 as usize).copied().unwrap_or(false)
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
            HExprKind::Field { base, .. } | HExprKind::Index { base, .. } => self.collect_owner_aliases(base, out),
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
            HExprKind::Free(_) => {
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
    fn kill_lid(&mut self, id: LocalId, _span: Span) -> Vec<LocalId> {
        let lid_num = id.0;
        let types: Vec<HType> = self.f().locals.iter().map(|l| l.ty.clone()).collect();
        let mut nulled = Vec::new();
        for (i, st) in self.state.iter_mut().enumerate() {
            if i as u32 == lid_num { continue; }
            if st.deps.remove(&lid_num) {
                if let HType::Ptr { .. } = types[i] {
                    if st.deps.is_empty() {
                        nulled.push(LocalId(i as u32));
                        // The runtime value of this pointer is about to be overwritten with
                        // NULL by codegen.  Past non-null proofs no longer hold.
                        st.auto_nulled = true;
                        st.known_nonnull = false;
                        st.narrowed_until = None;
                    }
                } else {
                    st.poisoned = true;
                }
            }
        }
        nulled
    }

    fn snapshot(&self) -> Vec<LocalState> { self.state.clone() }
    fn restore(&mut self, s: Vec<LocalState>) { self.state = s; }

    fn join_branches(&mut self, then_s: &[LocalState], else_s: &[LocalState], span: Span, in_scope: &std::collections::HashSet<u32>) {
        let n = self.state.len();
        for i in 0..n {
            let li = &self.f().locals[i];
            if matches!(li.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                let tm = then_s[i].moved;
                let em = else_s[i].moved;
                if tm != em && in_scope.contains(&(i as u32)) {
                    self.err(format!("`{}` is moved on one branch but not the other", li.name), span);
                }
                self.state[i].moved = tm && em;
            } else {
                // Reference/pointer: union deps; poison if poisoned in either reachable branch.
                let mut deps = then_s[i].deps.clone();
                for d in &else_s[i].deps { deps.insert(*d); }
                self.state[i].deps = deps;
                self.state[i].poisoned = then_s[i].poisoned || else_s[i].poisoned;
                // auto_nulled persists if EITHER branch left it set — the user
                // must re-assign on every code path to deterministically clear it.
                self.state[i].auto_nulled = then_s[i].auto_nulled || else_s[i].auto_nulled;
                // known_nonnull holds only if BOTH branches established it.
                self.state[i].known_nonnull = then_s[i].known_nonnull && else_s[i].known_nonnull;
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
        Some(HStmt::Break(_)) | Some(HStmt::Continue(_)) => true,
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
            HStmt::While { body, .. } | HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => {
                hoist_block_temps(sym, body, locals);
            }
            HStmt::Block(bb) | HStmt::Unsafe(bb, _) => hoist_block_temps(sym, bb, locals),
            _ => {}
        }
        // Hoist owning temporaries out of this statement's once-evaluated
        // expression positions.  (Loop/if conditions are re-evaluated, so they
        // are left alone.)
        let mut pre: Vec<HStmt> = Vec::new();
        match &mut s {
            HStmt::ExprStmt(e) => {
                hoist_in_expr(sym, e, locals, &mut pre);
                if is_owning_temp(sym, e) { hoist_one(e, locals, &mut pre); }
            }
            HStmt::Let { init, .. } => hoist_in_expr(sym, init, locals, &mut pre),
            HStmt::Assign { place, value, .. } => {
                hoist_in_expr(sym, place, locals, &mut pre);
                hoist_in_expr(sym, value, locals, &mut pre);
            }
            HStmt::Return { value: Some(v), .. } => hoist_in_expr(sym, v, locals, &mut pre),
            HStmt::Propagate { value: Some(v), .. } => hoist_in_expr(sym, v, locals, &mut pre),
            HStmt::ForEach { src, .. } => hoist_in_expr(sym, src, locals, &mut pre),
            _ => {}
        }
        out.extend(pre);
        out.push(s);
    }
    b.stmts = out;
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
                    if is_owning_temp(sym, a) { hoist_one(a, locals, pre); }
                }
            }
        }
        HExprKind::CallIndirect { callee, args } => {
            hoist_in_expr(sym, callee, locals, pre);
            for a in args.iter_mut() { hoist_in_expr(sym, a, locals, pre); }
            for a in args.iter_mut() {
                if is_owning_temp(sym, a) { hoist_one(a, locals, pre); }
            }
        }
        HExprKind::Bin { lhs, rhs, .. } => { hoist_in_expr(sym, lhs, locals, pre); hoist_in_expr(sym, rhs, locals, pre); }
        HExprKind::Un { expr, .. }
        | HExprKind::Unwrap { expr, .. }
        | HExprKind::Cast { expr, .. }
        | HExprKind::CheckedCast { expr, .. }
        | HExprKind::DropWrite(expr)
        | HExprKind::DerefRef(expr)
        | HExprKind::HeapAlloc(expr)
        | HExprKind::Free(expr)
        | HExprKind::SliceLen(expr)
        | HExprKind::EnumTag(expr)
        | HExprKind::ArrayToSlice { base: expr, .. }
        | HExprKind::AddrOfRef { place: expr, .. } => hoist_in_expr(sym, expr, locals, pre),
        HExprKind::Field { base, .. } => hoist_in_expr(sym, base, locals, pre),
        HExprKind::Index { base, idx } => { hoist_in_expr(sym, base, locals, pre); hoist_in_expr(sym, idx, locals, pre); }
        HExprKind::ArrayLit(es) => { for e2 in es.iter_mut() { hoist_in_expr(sym, e2, locals, pre); } }
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => {
            // Field initializers are moved into the aggregate (owned by it), so
            // they are not hoisted; only their nested call-args are.
            for (_, fe) in fields.iter_mut() { hoist_in_expr(sym, fe, locals, pre); }
        }
        HExprKind::Match { arms, .. } => {
            // Arm bodies are real statement blocks; their conditional values are
            // left to their own scopes.
            for arm in arms.iter_mut() { hoist_block_temps(sym, &mut arm.body, locals); }
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
fn owning_type_of(e: &HExpr) -> HType {
    match &e.ty {
        HType::OwnPtr { .. } | HType::Heap { .. } => e.ty.clone(),
        _ => HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) },
    }
}

/// Replace `e` with a read of a fresh owning local, and append the binding to
/// `pre`.  The read keeps `e`'s original (possibly coerced) type so downstream
/// dispatch (e.g. `log`) and the move analysis behave exactly as before.
fn hoist_one(e: &mut HExpr, locals: &mut Vec<LocalInfo>, pre: &mut Vec<HStmt>) {
    let lid = LocalId(locals.len() as u32);
    let own_ty = owning_type_of(e);
    let orig_ty = e.ty.clone();
    let sp = e.span;
    locals.push(LocalInfo {
        name: format!("__tmp{}", lid.0),
        ty: own_ty,
        storage: StorageClass::Heap,
        mut_payload: true,
        reassignable: true,
        thread_local: false,
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
pub(crate) fn ty_owns_heap(sym: &SymTab, ty: &HType) -> bool {
    fn go(sym: &SymTab, ty: &HType, seen: &mut Vec<u64>) -> bool {
        match ty {
            HType::Heap { .. } | HType::OwnPtr { .. } => true,
            // A by-value `Vec<T>` owns its malloc'd buffer.
            HType::Vec { .. } => true,
            // `Rust<T>` owns a boxed Rust value (dropped via a generated shim).
            HType::RustOpaque(_) => true,
            HType::Struct(id) => {
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
            HExprKind::Call { args, .. } => {
                for a in args {
                    if ty_owns_heap(sym, &a.ty) {
                        if let HExprKind::Local(id) = a.kind { out.insert(id); }
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
            HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) => moved_locals_in_expr(sym, expr, out),
            HExprKind::ArrayToSlice { base, .. } => moved_locals_in_expr(sym, base, out),
            HExprKind::DerefRef(inner) => moved_locals_in_expr(sym, inner, out),
            HExprKind::HeapAlloc(inner) => moved_locals_in_expr(sym, inner, out),
            HExprKind::Free(inner) => moved_locals_in_expr(sym, inner, out),
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

    fn visit_block(sym: &SymTab, locals: &[LocalInfo], b: &mut HBlock, scope_chain: &mut Vec<Vec<LocalId>>) {
        scope_chain.push(Vec::new());
        // Track moves up to each position in the block.
        let mut moved: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
        for s in &mut b.stmts {
            match s {
                HStmt::Let { local, init, .. } => {
                    // Check init for moves
                    moved_locals_in_expr(sym, init, &mut moved);
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
                HStmt::Assign { place, value, .. } => {
                    moved_locals_in_expr(sym, value, &mut moved);
                    if ty_owns_heap(sym, &value.ty) {
                        if let HExprKind::Local(id) = value.kind { moved.insert(id); }
                    }
                    // The assigned-to local is redefined here, so it is live again -
                    // even if its old value was just consumed into the RHS (e.g.
                    // `x = Cons { tail = x }` or `left = Bin { l = left, ... }`).
                    if let HExprKind::Local(id) = place.kind { moved.remove(&id); }
                }
                HStmt::ExprStmt(e) => {
                    moved_locals_in_expr(sym, e, &mut moved);
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
                    }
                    let mut drops = Vec::new();
                    for scope in scope_chain.iter() {
                        for id in scope.iter().rev() {
                            if returning.contains(id) || moved.contains(id) { continue; }
                            drops.push(*id);
                        }
                    }
                    *heap_drops = drops;
                }
                HStmt::If { then_b, else_b, .. } => {
                    visit_block(sym, locals, then_b, scope_chain);
                    if let Some(b) = else_b { visit_block(sym, locals, b, scope_chain); }
                }
                HStmt::While { body, .. } => visit_block(sym, locals, body, scope_chain),
                HStmt::Block(b) => visit_block(sym, locals, b, scope_chain),
                HStmt::Unsafe(b, _) => visit_block(sym, locals, b, scope_chain),
                HStmt::Break(_) | HStmt::Continue(_) => {}
                HStmt::ForC { body, .. } => visit_block(sym, locals, body, scope_chain),
                HStmt::ForEach { body, .. } => visit_block(sym, locals, body, scope_chain),
                HStmt::Propagate { value: Some(v), .. } => {
                    moved_locals_in_expr(sym, v, &mut moved);
                }
                HStmt::Propagate { value: None, .. } => {}
            }
        }
        // Fill heap_to_free: locals declared in this scope, in reverse order, skipping moved.
        let scope = scope_chain.pop().unwrap_or_default();
        let mut to_free = Vec::new();
        for id in scope.iter().rev() {
            if moved.contains(id) { continue; }
            to_free.push(*id);
        }
        b.heap_to_free = to_free;
    }

    let mut chain = Vec::new();
    let locals = f.locals.clone();
    visit_block(sym, &locals, &mut f.body, &mut chain);

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
        if !freeable.is_empty() { add_closure_drops_block(&mut f.body, &freeable); }
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
            | HExprKind::HeapAlloc(expr) | HExprKind::Free(expr) | HExprKind::SliceLen(expr)
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
    // missed use would leave an escaped closure wrongly freeable.
    fn scan_stmt(s: &HStmt, cands: &std::collections::HashSet<LocalId>, escaped: &mut std::collections::HashSet<LocalId>) {
        match s {
            HStmt::Let { init, .. } => ex(init, cands, escaped),
            HStmt::Assign { place, value, .. } => { ex(place, cands, escaped); ex(value, cands, escaped); }
            HStmt::ExprStmt(e) => ex(e, cands, escaped),
            HStmt::Return { value, .. } => { if let Some(v) = value { ex(v, cands, escaped); } }
            HStmt::Propagate { value, .. } => { if let Some(v) = value { ex(v, cands, escaped); } }
            HStmt::If { cond, then_b, else_b, .. } => {
                ex(cond, cands, escaped);
                for st in &then_b.stmts { scan_stmt(st, cands, escaped); }
                if let Some(b) = else_b { for st in &b.stmts { scan_stmt(st, cands, escaped); } }
            }
            HStmt::While { cond, body, .. } => { ex(cond, cands, escaped); for st in &body.stmts { scan_stmt(st, cands, escaped); } }
            HStmt::Block(b) | HStmt::Unsafe(b, _) => { for st in &b.stmts { scan_stmt(st, cands, escaped); } }
            HStmt::ForC { init, cond, step, body, .. } => {
                scan_stmt(init, cands, escaped); ex(cond, cands, escaped); scan_stmt(step, cands, escaped);
                for st in &body.stmts { scan_stmt(st, cands, escaped); }
            }
            HStmt::ForEach { src, body, .. } => { ex(src, cands, escaped); for st in &body.stmts { scan_stmt(st, cands, escaped); } }
            HStmt::Break(_) | HStmt::Continue(_) => {}
        }
    }
    for s in &b.stmts { scan_stmt(s, cands, escaped); }
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

/// Run `g` on each immediate child block of a statement (read-only).
fn for_each_child_block(s: &HStmt, g: &mut dyn FnMut(&HBlock)) {
    match s {
        HStmt::If { then_b, else_b, .. } => { g(then_b); if let Some(b) = else_b { g(b); } }
        HStmt::While { body, .. } | HStmt::Block(body) | HStmt::Unsafe(body, _)
        | HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => g(body),
        _ => {}
    }
}
fn for_each_child_block_mut(s: &mut HStmt, g: &mut dyn FnMut(&mut HBlock)) {
    match s {
        HStmt::If { then_b, else_b, .. } => { g(then_b); if let Some(b) = else_b { g(b); } }
        HStmt::While { body, .. } | HStmt::Block(body) | HStmt::Unsafe(body, _)
        | HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => g(body),
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
            // A return that yields an owning local moves it out.
            if ty_owns_heap(sym, &v.ty) {
                if let HExprKind::Local(id) = v.kind { out.insert(id); }
            }
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
        HStmt::Break(_) | HStmt::Continue(_) => {}
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
        HExprKind::HeapAlloc(inner) | HExprKind::Free(inner) => collect_param_moves_expr(sym, inner, out),
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
            for (_, fe) in fields { collect_param_moves_expr(sym, fe, out); }
        }
        HExprKind::ArrayLit(es) => for e in es { collect_param_moves_expr(sym, e, out); },
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
