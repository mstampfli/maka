//! Lifetime/deps + move tracking pass on HIR.
//!
//! Implements §11: LIDs, deps sets, poisoning of strong references,
//! null-collapse for `*T`, move tracking for `heap T`, pointer narrowing,
//! and heap-drop insertion at scope boundaries (returns and block exit).
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
    /// `live`/`moved` per §11.7.
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

pub fn analyze_func(
    sym: &SymTab,
    f: &mut HFunc,
) -> Result<Vec<SemaWarning>, Vec<SemaError>> {
    let mut a = Analyzer::new(sym, f);
    // analyze_block's `?` would early-return without surfacing warnings, but warnings
    // are only meaningful when analysis ran to completion, so this is fine.
    a.analyze_block(&mut Vec::new(), 0)?;
    let mut errors = std::mem::take(&mut a.errors);
    let warnings = std::mem::take(&mut a.warnings);
    // Final: walk the body and fill heap_to_free for each block in reverse scope-decl order.
    fill_heap_drops(sym, f);
    if errors.is_empty() { Ok(warnings) } else { Err(std::mem::take(&mut errors)) }
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
}

impl<'a> Analyzer<'a> {
    fn new(sym: &'a SymTab, f: &mut HFunc) -> Self {
        let n = f.locals.len();
        Self {
            sym,
            f: f as *mut _,
            state: (0..n).map(|_| LocalState::fresh()).collect(),
            errors: Vec::new(),
            warnings: Vec::new(),
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
                // Compute deps for init
                let init_deps = self.expr_deps(init);
                let init_nonnull = self.expr_nonnull(init);
                let id = *local;
                self.state[id.0 as usize].deps = init_deps;
                self.state[id.0 as usize].moved = false;
                self.state[id.0 as usize].poisoned = false;
                self.state[id.0 as usize].known_nonnull = init_nonnull;
                // Fresh local: it was never auto-nulled.  A re-bound name shadows
                // the prior state; explicit init beats any historical collapse.
                self.state[id.0 as usize].auto_nulled = false;
                declared_here.push(id);

                // Move semantics: `heap T b = a;` or `own *T b = a;` moves `a`.
                let li_ty = self.f().locals[id.0 as usize].ty.clone();
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
                // Reassigning a pointer overwrites its deps (§3.8).
                if let HExprKind::Local(id) = place.kind {
                    if matches!(self.f().locals[id.0 as usize].ty, HType::Ptr { .. }) {
                        let d = self.expr_deps(value);
                        let nn = self.expr_nonnull(value);
                        self.state[id.0 as usize].deps = d;
                        self.state[id.0 as usize].known_nonnull = nn;
                        // Reassignment also clears any active narrowing window — the
                        // previously-checked value is no longer there.
                        self.state[id.0 as usize].narrowed_until = None;
                        // ANY explicit re-assignment by the user clears auto_nulled —
                        // the silent compiler overwrite has been replaced by an intentional
                        // act.  The new value may itself be NULL or unproven; that's caught
                        // separately by the forced-handling deref rule.
                        self.state[id.0 as usize].auto_nulled = false;
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
                // Inconsistent move state is an error per §11.7.
                self.join_branches(&then_state, &else_state, *span);
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
            HExprKind::Closure { env_values, .. } => {
                for v in env_values { self.walk_expr(v); }
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
            HeapAlloc(inner) => self.check_no_local_ref_escape(inner),
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
            HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } => self.expr_nonnull(expr),
            _ => false,
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
            HExprKind::Field { base, .. } | HExprKind::Index { base, .. } => self.collect_deps(base, out),
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

    fn join_branches(&mut self, then_s: &[LocalState], else_s: &[LocalState], span: Span) {
        let n = self.state.len();
        for i in 0..n {
            let li = &self.f().locals[i];
            if matches!(li.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                let tm = then_s[i].moved;
                let em = else_s[i].moved;
                if tm != em {
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

/// Fill in `HBlock::heap_to_free` and `HStmt::Return::heap_drops` with the heap locals
/// declared in their respective scopes (excluding moved ones).
///
/// Conservative version: drop heap locals at the *end* of the block in which they were
/// declared, in reverse declaration order, unless they were moved (which we approximate
/// by skipping any local that appears as the direct value of a `return` or as a move
/// argument inside the block). For v1 we adopt the safe always-drop-if-not-moved-at-end
/// reading and rely on the move tracker to keep this consistent.
fn fill_heap_drops(_sym: &SymTab, f: &mut HFunc) {
    // We need to walk and, for each block, accumulate `Let { local }` whose local has heap storage.
    // For `return` statements inside that block, the same locals are dropped *before* returning,
    // minus any local being returned (moved-out via the return expression).

    fn moved_locals_in_expr(e: &HExpr, out: &mut std::collections::HashSet<LocalId>) {
        // local appearing as a heap value at the top of any call arg or return is moved
        match &e.kind {
            HExprKind::Local(_) => {
                // A bare Local read is NOT a move — it's only a move when it appears
                // directly as a call argument (handled below) or as a return value
                // (handled in the caller).  Field/Unwrap/etc. accesses through a
                // Heap/OwnPtr local are reads of the inner value, not transfers.
            }
            HExprKind::Call { args, .. } => {
                for a in args {
                    if matches!(a.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        if let HExprKind::Local(id) = a.kind { out.insert(id); }
                    }
                    moved_locals_in_expr(a, out);
                }
            }
            HExprKind::Bin { lhs, rhs, .. } => { moved_locals_in_expr(lhs, out); moved_locals_in_expr(rhs, out); }
            HExprKind::Un { expr, .. } => moved_locals_in_expr(expr, out),
            HExprKind::Unwrap { expr, .. } => moved_locals_in_expr(expr, out),
            HExprKind::AddrOfRef { place, .. } => moved_locals_in_expr(place, out),
            HExprKind::Field { base, .. } => moved_locals_in_expr(base, out),
            HExprKind::Index { base, idx } => { moved_locals_in_expr(base, out); moved_locals_in_expr(idx, out); }
            HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) => moved_locals_in_expr(expr, out),
            HExprKind::ArrayToSlice { base, .. } => moved_locals_in_expr(base, out),
            HExprKind::DerefRef(inner) => moved_locals_in_expr(inner, out),
            HExprKind::HeapAlloc(inner) => moved_locals_in_expr(inner, out),
            HExprKind::CallIndirect { callee, args } => {
                moved_locals_in_expr(callee, out);
                for a in args { moved_locals_in_expr(a, out); }
            }
            HExprKind::InlineCall { args, .. } => {
                // Mirror the Call arm — heap/own-ptr args at the InlineCall site
                // transfer ownership into the inline's parameter local, so the
                // outer binding must be marked moved for the auto-free at scope
                // exit to skip it.
                for a in args {
                    if matches!(a.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        if let HExprKind::Local(id) = a.kind { out.insert(id); }
                    }
                    moved_locals_in_expr(a, out);
                }
            }
            HExprKind::Closure { env_values, .. } => {
                for v in env_values { moved_locals_in_expr(v, out); }
            }
            HExprKind::Transfer(inner) => {
                if let HExprKind::Local(id) = inner.kind { out.insert(id); }
                moved_locals_in_expr(inner, out);
            }
            HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => moved_locals_in_expr(inner, out),
            HExprKind::FnRef(_) => {}
            HExprKind::VariantCtor { fields, .. } => for (_, fe) in fields { moved_locals_in_expr(fe, out); },
            HExprKind::Match { scrutinee, arms, .. } => {
                moved_locals_in_expr(scrutinee, out);
                for a in arms {
                    if let Some(g) = &a.guard { moved_locals_in_expr(g, out); }
                    if let Some(v) = &a.value { moved_locals_in_expr(v, out); }
                }
            }
            HExprKind::Struct { fields, .. } => for (_, fe) in fields { moved_locals_in_expr(fe, out); },
            HExprKind::ArrayLit(es) => for e in es { moved_locals_in_expr(e, out); },
            _ => {}
        }
    }

    fn visit_block(locals: &[LocalInfo], b: &mut HBlock, scope_chain: &mut Vec<Vec<LocalId>>) {
        scope_chain.push(Vec::new());
        // Track moves up to each position in the block.
        let mut moved: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
        for s in &mut b.stmts {
            match s {
                HStmt::Let { local, init, .. } => {
                    // Check init for moves
                    moved_locals_in_expr(init, &mut moved);
                    // A direct-Local init of an owning type transfers ownership.
                    if matches!(init.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        if let HExprKind::Local(id) = init.kind { moved.insert(id); }
                    }
                    if matches!(locals[local.0 as usize].ty, HType::Heap { .. } | HType::OwnPtr { .. })
                        && matches!(locals[local.0 as usize].storage, StorageClass::Heap) {
                        scope_chain.last_mut().unwrap().push(*local);
                    }
                }
                HStmt::Assign { value, .. } => {
                    moved_locals_in_expr(value, &mut moved);
                    if matches!(value.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        if let HExprKind::Local(id) = value.kind { moved.insert(id); }
                    }
                }
                HStmt::ExprStmt(e) => {
                    moved_locals_in_expr(e, &mut moved);
                }
                HStmt::Return { value, heap_drops, .. } => {
                    // The set of heap locals to drop at return = union of all scopes minus the return value.
                    let mut returning: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
                    if let Some(v) = value {
                        if matches!(v.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                            if let HExprKind::Local(id) = v.kind {
                                returning.insert(id);
                            }
                        }
                        moved_locals_in_expr(v, &mut moved);
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
                    visit_block(locals, then_b, scope_chain);
                    if let Some(b) = else_b { visit_block(locals, b, scope_chain); }
                }
                HStmt::While { body, .. } => visit_block(locals, body, scope_chain),
                HStmt::Block(b) => visit_block(locals, b, scope_chain),
                HStmt::Unsafe(b, _) => visit_block(locals, b, scope_chain),
                HStmt::Break(_) | HStmt::Continue(_) => {}
                HStmt::ForC { body, .. } => visit_block(locals, body, scope_chain),
                HStmt::ForEach { body, .. } => visit_block(locals, body, scope_chain),
                HStmt::Propagate { value: Some(v), .. } => {
                    moved_locals_in_expr(v, &mut moved);
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
    visit_block(&locals, &mut f.body, &mut chain);

    // Append heap-storage parameters to the body's heap_to_free so they auto-free
    // at function scope-exit — unless they were transferred out somewhere in the
    // body.  Without this, `own *T` / `own &T` parameters would leak (no drop is
    // ever emitted for them; the caller's drop is suppressed because the call
    // site marks the source as moved).
    let mut param_moved: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
    collect_param_moves_block(&f.body, &mut param_moved);
    for &pid in f.params.iter().rev() {
        let li = &locals[pid.0 as usize];
        if matches!(li.storage, StorageClass::Heap)
            && matches!(li.ty, HType::Heap { .. } | HType::OwnPtr { .. })
            && !param_moved.contains(&pid)
        {
            f.body.heap_to_free.push(pid);
        }
    }
    // Also append undropped params to every Return's heap_drops list — early
    // returns must still free param-owned values.
    append_param_drops_to_returns(&mut f.body, &f.params, &locals, &param_moved);
}

fn collect_param_moves_block(b: &HBlock, out: &mut std::collections::HashSet<LocalId>) {
    for s in &b.stmts { collect_param_moves_stmt(s, out); }
}
fn collect_param_moves_stmt(s: &HStmt, out: &mut std::collections::HashSet<LocalId>) {
    match s {
        HStmt::Let { init, .. } => collect_param_moves_expr(init, out),
        HStmt::Assign { value, .. } => collect_param_moves_expr(value, out),
        HStmt::ExprStmt(e) => collect_param_moves_expr(e, out),
        HStmt::Return { value: Some(v), .. } => {
            // A return that yields an owning local moves it out.
            if matches!(v.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                if let HExprKind::Local(id) = v.kind { out.insert(id); }
            }
            collect_param_moves_expr(v, out);
        }
        HStmt::Return { .. } => {}
        HStmt::If { cond, then_b, else_b, .. } => {
            collect_param_moves_expr(cond, out);
            collect_param_moves_block(then_b, out);
            if let Some(b) = else_b { collect_param_moves_block(b, out); }
        }
        HStmt::While { cond, body, .. } => {
            collect_param_moves_expr(cond, out);
            collect_param_moves_block(body, out);
        }
        HStmt::Block(b) | HStmt::Unsafe(b, _) => collect_param_moves_block(b, out),
        HStmt::ForC { init, cond, step, body, .. } => {
            collect_param_moves_stmt(init, out);
            collect_param_moves_expr(cond, out);
            collect_param_moves_stmt(step, out);
            collect_param_moves_block(body, out);
        }
        HStmt::ForEach { src, body, .. } => {
            collect_param_moves_expr(src, out);
            collect_param_moves_block(body, out);
        }
        HStmt::Propagate { value: Some(v), .. } => collect_param_moves_expr(v, out),
        HStmt::Propagate { value: None, .. } => {}
        HStmt::Break(_) | HStmt::Continue(_) => {}
    }
}
fn collect_param_moves_expr(e: &HExpr, out: &mut std::collections::HashSet<LocalId>) {
    match &e.kind {
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => {
            for a in args {
                if matches!(a.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                    if let HExprKind::Local(id) = a.kind { out.insert(id); }
                }
                collect_param_moves_expr(a, out);
            }
        }
        HExprKind::Bin { lhs, rhs, .. } => {
            collect_param_moves_expr(lhs, out);
            collect_param_moves_expr(rhs, out);
        }
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } => collect_param_moves_expr(expr, out),
        HExprKind::AddrOfRef { place, .. } => collect_param_moves_expr(place, out),
        HExprKind::Field { base, .. } => collect_param_moves_expr(base, out),
        HExprKind::Index { base, idx } => {
            collect_param_moves_expr(base, out);
            collect_param_moves_expr(idx, out);
        }
        HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. }
        | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr) => collect_param_moves_expr(expr, out),
        HExprKind::ArrayToSlice { base, .. } => collect_param_moves_expr(base, out),
        HExprKind::HeapAlloc(inner) => collect_param_moves_expr(inner, out),
        HExprKind::CallIndirect { callee, args } => {
            collect_param_moves_expr(callee, out);
            for a in args { collect_param_moves_expr(a, out); }
        }
        HExprKind::Closure { env_values, .. } => for v in env_values { collect_param_moves_expr(v, out); },
        HExprKind::Transfer(inner) => {
            if let Some(id) = root_local(inner) { out.insert(id); }
            collect_param_moves_expr(inner, out);
        }
        HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => collect_param_moves_expr(inner, out),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => {
            for (_, fe) in fields { collect_param_moves_expr(fe, out); }
        }
        HExprKind::ArrayLit(es) => for e in es { collect_param_moves_expr(e, out); },
        _ => {}
    }
}

fn append_param_drops_to_returns(
    b: &mut HBlock,
    params: &[LocalId],
    locals: &[LocalInfo],
    param_moved: &std::collections::HashSet<LocalId>,
) {
    for s in b.stmts.iter_mut() {
        match s {
            HStmt::Return { value, heap_drops, .. } => {
                let returning_id = value.as_ref().and_then(|v| {
                    if matches!(v.ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                        if let HExprKind::Local(id) = v.kind { Some(id) } else { None }
                    } else { None }
                });
                for &pid in params.iter().rev() {
                    let li = &locals[pid.0 as usize];
                    if matches!(li.storage, StorageClass::Heap)
                        && matches!(li.ty, HType::Heap { .. } | HType::OwnPtr { .. })
                        && !param_moved.contains(&pid)
                        && returning_id != Some(pid)
                        && !heap_drops.contains(&pid)
                    {
                        heap_drops.push(pid);
                    }
                }
            }
            HStmt::If { then_b, else_b, .. } => {
                append_param_drops_to_returns(then_b, params, locals, param_moved);
                if let Some(eb) = else_b { append_param_drops_to_returns(eb, params, locals, param_moved); }
            }
            HStmt::While { body, .. } | HStmt::Block(body) | HStmt::Unsafe(body, _) => {
                append_param_drops_to_returns(body, params, locals, param_moved);
            }
            HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => {
                append_param_drops_to_returns(body, params, locals, param_moved);
            }
            _ => {}
        }
    }
}
