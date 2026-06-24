//! Type checking pass: AST → HIR with type annotations.
//!
//! Mutability/reference/pointer/cast rules per §1–§4, §7–§10.
//! Lifetime/deps/move passes happen later in `lifetime.rs`.

use crate::hir::*;
use crate::resolve::{resolve_type, resolve_type_in, check_type_visibility};
use crate::SemaError;
use maka_ast::{self as ast, Mutness};
use maka_lexer::Span;

pub struct TypeChecker<'a> {
    sym: &'a SymTab,
    locals: Vec<LocalInfo>,
    scopes: Vec<Scope>,
    errors: Vec<SemaError>,
    /// `Rust<T>` types observed at `transfer` / `spawn` sites — emitted as
    /// `assert_send::<T>()` probes by the rust-bridge after sema completes.
    pub send_probes: Vec<String>,
    /// `Rust<T>` types observed at `share` sites — emitted as
    /// `assert_sync::<T>()` probes.
    pub sync_probes: Vec<String>,
    /// current function return type (for `return` validation)
    cur_ret: HType,
    /// Type parameters of the function currently being checked, so local
    /// variable annotations inside a generic body (e.g. `Vec<V> nv = [];`) can
    /// resolve the type vars.
    cur_type_params: Vec<String>,
    /// Stack of `(synthetic local, type)` targets for `yield`.  When non-empty,
    /// `yield e` lowers to `<top> = e` (assigning the enclosing match-arm /
    /// if-expression result) instead of a discarded `ExprStmt` - so a `yield`
    /// nested inside an `if`/`while`/block statement still produces the value.
    yield_target: Vec<(LocalId, HType)>,
    /// The expected type of the call expression currently being dispatched, set
    /// by `check_expr` before `check_call`.  Lets a generic call infer a type
    /// parameter that appears only in the return type from its context, e.g.
    /// `Stack<int> s = snew();` binds T=int from the expected `Stack<int>`.
    call_ret_expected: Option<HType>,
    /// Whether the current function is `inline` (governs `propagate` legality).
    cur_is_inline: bool,
    /// Dotted module path of the function currently being checked.  Used to enforce
    /// `pub` at call sites: calls to a non-`pub` callee from a different module are
    /// rejected.
    cur_module: Vec<String>,
    /// Imports visible at the current function's source file — used to require
    /// cross-module references to be explicitly imported.
    cur_imports: Vec<(Vec<String>, String)>,
    /// `use Mod.Type.Attr;` declarations visible at the current function's site.
    cur_has_imports: Vec<maka_ast::HasImport>,
    /// Where-bounds on the function currently being checked.  Used to disambiguate
    /// calls whose receiver type is a TyVar that has a multi-arg attribute bound:
    /// `demo<T: Convert<int>>(&T x) { x.to(); }` — the bound says "pick the
    /// `Convert<int>` impl of `to`", which narrows the candidate set.
    cur_where_bounds: Vec<(String, Vec<HType>, Vec<(String, HType)>)>,
    /// Are we inside an `unsafe` block? (Currently unused except as a flag.)
    in_unsafe: u32,
    /// If we're checking a function in a `logic` block, this is the logic's name.
    cur_logic: Option<String>,
    /// >0 if we're inside a Call's argument expression — allows `transfer`/`share`.
    call_arg_depth: u32,
    /// True when the immediate enclosing call's callee is a `gate` function.
    cur_call_is_gate: bool,
    /// §10.5 attr-qualified call (`Attr::method`).  When `Some(_)`, the next
    /// `check_call_inner` skips local-shadow lookup on the callee's qualifier
    /// — the `::` parse already committed to the qualified form.
    force_qualifier: Option<String>,
    /// §10.5: set together with `force_qualifier` for the postfix form
    /// `recv.Attr::method(args)`.  When true the synthesized call is treated
    /// as postfix (auto-borrow on arg 0).
    force_postfix: bool,
    /// Substitution for generic type parameters of the current function being checked.
    pub subst: std::collections::HashMap<String, HType>,
    /// Instantiation requests queued during type checking; drained by `analyze`.
    pub instantiation_requests: Vec<InstantiationReq>,
    /// Synthetic struct decls (lambda envs) emitted during typechecking.
    pub synth_structs: Vec<StructInfo>,
    /// Synthetic function sigs.
    pub synth_sigs: Vec<FuncSig>,
    /// Synthetic typed function bodies (lifted lambdas).
    pub synth_funcs: Vec<HFunc>,
}

#[derive(Debug, Clone)]
pub struct InstantiationReq {
    pub template_fid: FuncId,
    pub args: Vec<HType>,
    /// Module path of the caller that issued this instantiation, captured so
    /// downstream bound checks can filter `has` impls by visibility (file-private
    /// vs `pub` vs explicitly `use`-imported).
    pub caller_module: Vec<String>,
    /// `use Mod.Type.Attr;` imports active at the caller's site.
    pub caller_has_imports: Vec<maka_ast::HasImport>,
}

#[derive(Debug, Clone, Default)]
pub struct SynthDecls {
    pub structs: Vec<StructInfo>,
    pub sigs: Vec<FuncSig>,
    pub funcs: Vec<HFunc>,
    /// `Rust<T>` type names that must satisfy `Send`.  Collected at
    /// `transfer` / `spawn` sites.  Bubbled up to `SymTab.send_probes`
    /// after this function is checked.
    pub send_probes: Vec<String>,
    /// `Rust<T>` type names that must satisfy `Sync` — collected at
    /// `share` sites.
    pub sync_probes: Vec<String>,
}

#[derive(Default)]
struct Scope {
    names: Vec<(String, LocalId)>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(sym: &'a SymTab) -> Self {
        Self::new_with_logic(sym, None)
    }

    /// Reject `*unit` (the untyped opaque pointer) anywhere it would surface
    /// in safe user code — fn signatures, let bindings, struct fields.
    /// `*unit` is only meaningful at the FFI boundary, so it stays allowed
    /// in (a) the stdlib module, where typed handles wrap a single `*unit`
    /// field, (b) `extern` declarations, and (c) `unsafe { }` blocks for
    /// the same reason `raw *T` is.  Mutable `*mut unit` is still allowed
    /// everywhere — it's the canonical byte-buffer type for I/O until proper
    /// slice-of-byte lands.
    fn ban_unit_ptr_in_user_code(&mut self, ty: &HType, where_: &str, sp: Span) {
        if self.in_unsafe > 0 { return; }
        if self.cur_module.as_slice() == ["std".to_string()].as_slice() { return; }
        if matches!(ty, HType::Ptr { inner, .. } if matches!(**inner, HType::Unit)) {
            self.err(
                format!(
                    "`*unit` is not allowed in safe code ({0}); use a typed handle (e.g. `Mutex`, \
                     `Atomic`, `TlsConn`) from the stdlib, or wrap the use in an `unsafe {{ }}` block",
                    where_
                ),
                sp,
            );
        }
    }

    /// Reject captures that can't safely cross a thread boundary when used
    /// with `thread()` / `job()` / `spawn_pool()`.  v1: borrowed references
    /// (`&T`, `&mut T`) are tied to a scope on the spawning thread; the
    /// captured ref could outlive its source when the closure resumes on
    /// another thread.  Other types are conservatively allowed — users own
    /// the safety of `*T` to thread-local data.
    fn check_cross_thread_captures(&mut self, tier: &str, arg: &HExpr, sp: Span) {
        let mut cur = arg;
        let env_values = loop {
            match &cur.kind {
                HExprKind::Closure { env_values, .. } => break env_values.as_slice(),
                HExprKind::HeapAlloc(inner) | HExprKind::DropWrite(inner)
                | HExprKind::DerefRef(inner) | HExprKind::Transfer(inner) => cur = inner,
                _ => return,
            }
        };
        // Spawn-tier rule: every captured value must be safe to cross a
        // thread boundary.  Three classes are allowed:
        //
        //   (a) owning types (`own *T`, `own &T`) — by-value capture moves
        //       ownership to the thread (transfer semantics).
        //   (b) Shareable types (per is_shareable) — by-value copy / share.
        //   (c) Unit.
        //
        // Borrowed references (`&T`, `&mut T`) and non-Shareable pointers
        // (`*T`, `raw *T`, mutable slices) are rejected — their pointee
        // lifetime is tied to the caller's scope, and the thread can
        // outlive that scope.  This is the spawn-tier analogue of the
        // gate / transfer / share check (§7.1) — gate is just inlined
        // into the spawn-tier handler instead of being explicit.
        for v in env_values {
            let ty = &v.ty;
            if matches!(ty, HType::Unit) { continue; }
            // (a) owning types — capture by value moves ownership.
            if matches!(ty, HType::OwnPtr { .. } | HType::Heap { .. }) {
                continue;
            }
            // Borrows are the most common foot-gun; specific diagnostic.
            if matches!(ty, HType::Ref { .. }) {
                self.err(
                    format!(
                        "`{0}` captures a borrowed reference (`{1}`) which can't cross a thread \
                         boundary — the borrow's lifetime is tied to this scope but the thread can \
                         outlive it.  Capture by value `[name]` (moves ownership for `own *T` / \
                         `own &T`, copies Shareable values), or restructure to pass an `own *T`.",
                        tier, type_str(ty)
                    ),
                    sp,
                );
                continue;
            }
            // (b) anything else must be Shareable.
            if !self.is_shareable(ty) {
                self.err(
                    format!(
                        "`{0}` captures a value of type `{1}` which is not Shareable and can't \
                         cross a thread boundary — its pointee lifetime is unknown to the lifetime \
                         pass.  Capture an owning binding (`own *T` / `own &T`) so ownership \
                         transfers, or use a Shareable handle (e.g. an atomic, a `Mutex`, or \
                         `&const T` where `T: Shareable`).",
                        tier, type_str(ty)
                    ),
                    sp,
                );
            }
        }
    }

    /// Walk a spawn'd closure (possibly wrapped in `alloc` / `heap`) and
    /// record a `Send` probe for every captured `Rust<T>` value.  The
    /// captures live in `HExprKind::Closure { env_values }`; we scan
    /// each one for opaque types.
    fn collect_send_from_closure(&mut self, arg: &HExpr) {
        let mut cur = arg;
        loop {
            match &cur.kind {
                HExprKind::Closure { env_values, .. } => {
                    for v in env_values {
                        self.collect_send_from_expr(v);
                    }
                    return;
                }
                HExprKind::HeapAlloc(inner) | HExprKind::DropWrite(inner)
                | HExprKind::DerefRef(inner) | HExprKind::Transfer(inner) => {
                    cur = inner;
                }
                _ => return,
            }
        }
    }

    /// Recursively scan an expression tree for `Rust<T>` types, accumulating
    /// Send probes.  Conservative — we record every observed label so even
    /// values that flow through arithmetic / casts get covered.
    fn collect_send_from_expr(&mut self, e: &HExpr) {
        if let HType::RustOpaque(label) = &e.ty {
            self.send_probes.push(label.clone());
        }
        match &e.kind {
            HExprKind::Bin { lhs, rhs, .. } => {
                self.collect_send_from_expr(lhs);
                self.collect_send_from_expr(rhs);
            }
            HExprKind::Un { expr: inner, .. }
            | HExprKind::Cast { expr: inner, .. }
            | HExprKind::CheckedCast { expr: inner, .. }
            | HExprKind::DerefRef(inner)
            | HExprKind::DropWrite(inner)
            | HExprKind::HeapAlloc(inner)
            | HExprKind::ArrayToSlice { base: inner, .. }
            | HExprKind::Transfer(inner)
            | HExprKind::SliceLen(inner)
            | HExprKind::EnumTag(inner) => self.collect_send_from_expr(inner),
            HExprKind::Call { args, .. }
            | HExprKind::CallIndirect { args, .. }
            | HExprKind::InlineCall { args, .. } => {
                for a in args { self.collect_send_from_expr(a); }
            }
            HExprKind::ArrayLit(elems) => {
                for e in elems { self.collect_send_from_expr(e); }
            }
            HExprKind::Closure { env_values, .. } => {
                for v in env_values { self.collect_send_from_expr(v); }
            }
            HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => {
                for (_, e) in fields { self.collect_send_from_expr(e); }
            }
            _ => {}
        }
    }

    pub fn new_with_logic(sym: &'a SymTab, logic: Option<&str>) -> Self {
        Self {
            sym,
            locals: Vec::new(),
            scopes: Vec::new(),
            errors: Vec::new(),
            send_probes: Vec::new(),
            sync_probes: Vec::new(),
            cur_ret: HType::Unit,
            cur_type_params: Vec::new(),
            yield_target: Vec::new(),
            call_ret_expected: None,
            cur_is_inline: false,
            cur_module: Vec::new(),
            cur_imports: Vec::new(),
            cur_has_imports: Vec::new(),
            cur_where_bounds: Vec::new(),
            in_unsafe: 0,
            cur_logic: logic.map(|s| s.to_string()),
            call_arg_depth: 0,
            cur_call_is_gate: false,
            force_qualifier: None,
            force_postfix: false,
            subst: std::collections::HashMap::new(),
            instantiation_requests: Vec::new(),
            synth_structs: Vec::new(),
            synth_sigs: Vec::new(),
            synth_funcs: Vec::new(),
        }
    }

    /// Resolve a type annotation appearing inside a function body (local var,
    /// for-each binding, lambda param).  Uses the current function's type params
    /// so generic type vars resolve, then applies the instantiation substitution
    /// and concretizes any resulting `Name<concrete..>` pattern to its struct.
    fn resolve_local_ty(&mut self, t: &ast::Type) -> HType {
        let tps = self.cur_type_params.clone();
        let raw = resolve_type_in(self.sym, t, &tps, &mut self.errors);
        concretize_generic_patterns(&raw.subst(&self.subst), self.sym)
    }

    pub fn with_subst(mut self, subst: std::collections::HashMap<String, HType>) -> Self {
        self.subst = subst;
        self
    }

    pub fn check_func(self, f: &ast::FuncDecl) -> Result<HFunc, Vec<SemaError>> {
        self.check_func_with_id(f, None).map(|(hf, _, _)| hf)
    }

    /// `forced_fid`: when monomorphizing, use this FuncId instead of looking up by name.
    /// Returns the typed function, instantiation requests, and any synthetic decls.
    pub fn check_func_with_id(mut self, f: &ast::FuncDecl, forced_fid: Option<FuncId>) -> Result<(HFunc, Vec<InstantiationReq>, SynthDecls), Vec<SemaError>> {
        let fid = if let Some(id) = forced_fid {
            id
        } else {
            match &self.cur_logic {
                Some(l) => self.sym.func_by_qualified(l, &f.name).expect("sig collected").0,
                None => self.sym.func_by_name(&f.name).expect("sig collected").0,
            }
        };
        let ret_t = resolve_type_in(self.sym, &f.ret, &f.type_params, &mut self.errors);
        // After substituting the instantiation's type args, resolve any
        // `Name<concrete..>` pattern to its monomorphized struct/enum so the
        // body sees a real Struct/Enum (not a GenericPattern) - needed for
        // field access, codegen c_type, etc. on a generic param like `&mut Map<int>`.
        self.cur_ret = concretize_generic_patterns(&ret_t.subst(&self.subst), self.sym);
        self.cur_type_params = f.type_params.clone();
        self.cur_is_inline = f.is_inline;
        // Read the module path off the function's already-resolved sig so we can
        // check it against callees' modules later.
        self.cur_module = self.sym.func_sig(fid).module_path.clone();
        self.cur_imports = self.sym.func_sig(fid).imports.clone();
        self.cur_has_imports = self.sym.func_sig(fid).has_imports.clone();
        self.cur_where_bounds = self.sym.func_sig(fid).where_bounds.clone();

        // Lifted no-capture lambdas inherit the unsafe scope of their lexical
        // call site, but the lifted top-level function loses that context.
        // The AST-level lift encodes the call-site `unsafe { }` state in the
        // synthetic name (`__lambda_unsafe_N` vs `__lambda_N`) so we can
        // either skip or apply the `*unit` ban accordingly.
        let lifted_in_unsafe = f.name.starts_with("__lambda_unsafe_");
        if !lifted_in_unsafe {
            self.ban_unit_ptr_in_user_code(&self.cur_ret.clone(), "function return type", f.span);
        }

        self.enter_scope();
        let mut param_ids = Vec::new();
        for p in &f.params {
            let raw = resolve_type_in(self.sym, &p.ty, &f.type_params, &mut self.errors);
            let ty = concretize_generic_patterns(&raw.subst(&self.subst), self.sym);
            if !lifted_in_unsafe {
                self.ban_unit_ptr_in_user_code(&ty, "function parameter", p.span);
            }
            // Owning params (both `own &T`/Heap and `own *T`/OwnPtr) carry
            // ownership in from the caller — storage=Heap so the lifetime pass
            // auto-frees them at function scope-exit unless the callee moves
            // them out (return, transfer, onward call).  Without this, `own *T`
            // params leaked AND double-freed depending on call shape.
            let storage = if matches!(ty, HType::Heap { .. } | HType::OwnPtr { .. }) {
                StorageClass::Heap
            } else {
                StorageClass::Param
            };
            let id = self.fresh_local(p.name.clone(), ty.clone(), storage, /*mut_payload*/ true, /*reassignable*/ false, p.span);
            self.bind_name(&p.name, id);
            param_ids.push(id);
        }

        let body = self.check_block(&f.body);
        self.leave_scope();

        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let synth = SynthDecls {
            structs: self.synth_structs,
            sigs: self.synth_sigs,
            funcs: self.synth_funcs,
            send_probes: self.send_probes,
            sync_probes: self.sync_probes,
        };
        Ok((HFunc {
            id: fid,
            name: f.name.clone(),
            params: param_ids,
            ret: self.cur_ret.clone(),
            locals: self.locals,
            body,
            span: f.span,
        }, self.instantiation_requests, synth))
    }

    fn fresh_local(&mut self, name: String, ty: HType, storage: StorageClass, mut_payload: bool, reassignable: bool, span: Span) -> LocalId {
        let id = LocalId(self.locals.len() as u32);
        self.locals.push(LocalInfo { name, ty, storage, mut_payload, reassignable, thread_local: false, span });
        id
    }

    fn enter_scope(&mut self) { self.scopes.push(Scope::default()); }
    fn leave_scope(&mut self) { self.scopes.pop(); }
    fn bind_name(&mut self, n: &str, id: LocalId) {
        self.scopes.last_mut().unwrap().names.push((n.to_string(), id));
    }
    fn lookup(&self, n: &str) -> Option<LocalId> {
        for s in self.scopes.iter().rev() {
            for (nm, id) in s.names.iter().rev() {
                if nm == n { return Some(*id); }
            }
        }
        None
    }
    fn local(&self, id: LocalId) -> &LocalInfo { &self.locals[id.0 as usize] }

    fn err(&mut self, msg: impl Into<String>, span: Span) {
        self.errors.push(SemaError { msg: msg.into(), span });
    }

    // ---- block / stmts ----
    fn check_block(&mut self, b: &ast::Block) -> HBlock {
        self.enter_scope();
        let mut stmts = Vec::new();
        for s in &b.stmts {
            stmts.push(self.check_stmt(s));
        }
        self.leave_scope();
        HBlock { stmts, heap_to_free: Vec::new(), ptr_nulls: Vec::new(), span: b.span }
    }

    fn check_stmt(&mut self, s: &ast::Stmt) -> HStmt {
        match s {
            ast::Stmt::Let { mutness, ty, name, init, thread_local, span } => self.check_let(mutness.clone(), ty, name, init, *thread_local, *span),
            ast::Stmt::Assign { op, place, value, span } => self.check_assign(*op, place, value, *span),
            ast::Stmt::ExprStmt(e, span) => {
                let h = self.check_expr(e, None);
                // Non-unit values cannot be silently discarded.
                if !matches!(h.ty, HType::Unit) {
                    self.err(format!("non-unit return value discarded; use `_ = expr;` to discard"), *span);
                }
                HStmt::ExprStmt(h)
            }
            ast::Stmt::Return(e, span) => {
                let ret_ty = self.cur_ret.clone();
                let v = e.as_ref().map(|e| self.check_expr_coerce(e, &ret_ty));
                // Lambda-escape rule: a capturing lambda returned by value escapes its
                // creating scope, which would dangle. Force the user to allocate it.
                if let Some(hv) = &v {
                    if let HExprKind::Closure { env_values, .. } = &hv.kind {
                        if !env_values.is_empty() && !matches!(ret_ty, HType::Heap { .. }) {
                            self.err("capturing lambda escapes; the return type must be `alloc <fn-type>`".to_string(), *span);
                        }
                    }
                }
                HStmt::Return { value: v, heap_drops: Vec::new(), span: *span }
            }
            ast::Stmt::If { cond, then_block, else_block, span } => {
                let c = self.check_expr_coerce(cond, &HType::Bool);
                let then_b = self.check_block(then_block);
                let else_b = else_block.as_ref().map(|b| self.check_block(b));
                HStmt::If { cond: c, then_b, else_b, span: *span }
            }
            ast::Stmt::While { cond, body, span } => {
                let c = self.check_expr_coerce(cond, &HType::Bool);
                let b = self.check_block(body);
                HStmt::While { cond: c, body: b, span: *span }
            }
            ast::Stmt::Block(b) => HStmt::Block(self.check_block(b)),
            ast::Stmt::Match { scrutinee, arms, span } => {
                let hm = self.check_match(scrutinee, arms, /*as_stmt=*/true, *span, None);
                HStmt::ExprStmt(hm)
            }
            ast::Stmt::Yield(e, span) => {
                // If we're inside a value-producing block (a match arm or an
                // if-expression branch), `yield` assigns the enclosing result
                // target.  This makes a `yield` nested inside an `if`/`while`/
                // block statement work, not just a trailing one.  Otherwise fall
                // back to a plain expression statement (a trailing `yield` whose
                // arm has no active target is captured by `extract_yield_value`).
                if let Some((tgt, tgt_ty)) = self.yield_target.last().cloned() {
                    let h = self.check_expr_coerce(e, &tgt_ty);
                    let place = HExpr { kind: HExprKind::Local(tgt), ty: tgt_ty, span: *span };
                    HStmt::Assign { op: HAssignOp::Assign, place, value: h, span: *span }
                } else {
                    let h = self.check_expr(e, None);
                    HStmt::ExprStmt(h)
                }
            }
            ast::Stmt::Propagate(opt_e, span) => {
                if !self.cur_is_inline {
                    self.err("`propagate` is only valid inside an `inline` function", *span);
                }
                let value = opt_e.as_ref().map(|e| self.check_expr(e, None));
                HStmt::Propagate { value, span: *span }
            }
            ast::Stmt::ForRange { var_ty, var_name, start, end, inclusive, body, span } => {
                self.check_for_range(var_ty, var_name, start, end, *inclusive, body, *span)
            }
            ast::Stmt::ForEach { var_ty, var_name, src, body, span } => {
                self.check_for_each(var_ty, var_name, src, body, *span)
            }
            ast::Stmt::InlineFor { var_name, iter, body, span } => {
                self.check_inline_for(var_name, iter, body, *span)
            }
            ast::Stmt::Break(span) => HStmt::Break { heap_drops: Vec::new(), span: *span },
            ast::Stmt::Continue(span) => HStmt::Continue { heap_drops: Vec::new(), span: *span },
            ast::Stmt::Unsafe(b, span) => {
                self.in_unsafe += 1;
                let bb = self.check_block(b);
                self.in_unsafe -= 1;
                HStmt::Unsafe(bb, *span)
            }
        }
    }

    fn check_let(&mut self, mutness: Mutness, ty: &ast::Type, name: &str, init: &ast::Expr, thread_local: bool, span: Span) -> HStmt {
        if matches!(mutness, Mutness::Mut) {
            if let ast::Type::Ptr { .. } = ty {
                self.err("`mut *T` is invalid: pointer bindings are always reassignable", span);
            }
        }
        let mut declared = self.resolve_local_ty(ty);
        let cur_module = self.cur_module.clone();
        let cur_imports = self.cur_imports.clone();
        // Skip the visibility check while monomorphizing a generic body: a local
        // like `Vec<V>` becomes `Vec<own *Box>` where Box is the *caller's* type,
        // legitimately visible at the call site.  The template was already checked
        // with the type vars in place (subst empty), which catches real leaks.
        if self.subst.is_empty() {
            check_type_visibility(self.sym, &declared, &cur_module, &cur_imports, span, &mut self.errors);
        }
        self.ban_unit_ptr_in_user_code(&declared, "let binding type", span);
        // If the user declared `string` but the initializer produces an owning
        // value (e.g. `string + string` → `own *char`, `read_line()` → `own *char`),
        // bind the slot AS the owning type so the lifetime pass auto-frees it.
        // Use-site coercion (`own *char` → `string`) keeps source code ergonomic.
        if matches!(declared, HType::Str) {
            if let Some(probed_ty) = Self::probe_init_ty(self.sym, init) {
                if matches!(&probed_ty, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char)) {
                    declared = probed_ty;
                }
            }
        }
        let (storage, mut_payload, reassignable) = self.binding_kind(&declared, &mutness);
        let init_h = self.check_expr_coerce(init, &declared);
        let id = self.fresh_local_with_tls(name.to_string(), declared, storage, mut_payload, reassignable, thread_local, span);
        self.bind_name(name, id);
        HStmt::Let { local: id, init: init_h, span }
    }

    /// Light-weight type probe for the limited cases where `check_let` needs to
    /// upgrade `string` → `own *char`.  Only handles the two known producers; any
    /// other shape returns `None` and the standard coercion path runs.
    fn probe_init_ty(_sym: &SymTab, init: &ast::Expr) -> Option<HType> {
        match init {
            ast::Expr::Bin { op: ast::BinOp::Add, .. } => {
                // Conservative: assume any `+` MIGHT be string concat; if it isn't,
                // the type still resolves to a numeric and `binding_kind` is fine.
                // We only consult this result when declared == Str, so a numeric
                // init at a Str slot will error normally via coerce.
                Some(HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) })
            }
            ast::Expr::Call { callee, .. } => {
                // Builtins that return a freshly-owned `own *char`: binding their
                // result to a `string` slot should own + free it.
                if matches!(callee.as_ref(), ast::Expr::Ident(n, _) if n == "read_line" || n == "format") {
                    Some(HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn fresh_local_with_tls(&mut self, name: String, ty: HType, storage: StorageClass, mut_payload: bool, reassignable: bool, thread_local: bool, span: Span) -> LocalId {
        let id = LocalId(self.locals.len() as u32);
        self.locals.push(LocalInfo { name, ty, storage, mut_payload, reassignable, thread_local, span });
        id
    }

    /// Classify a binding given its declared type and the outer mut/const modifier.
    fn binding_kind(&self, ty: &HType, mutness: &Mutness) -> (StorageClass, bool, bool) {
        let storage = match ty {
            HType::Heap { .. } | HType::OwnPtr { .. } => StorageClass::Heap,
            _ => StorageClass::Stack,
        };
        let mut_payload = match ty {
            // For ptrs and refs, the *pointee* mutness is encoded in the type,
            // not in the outer `mut`/`const`. We accept any outer modifier here.
            HType::Ptr { .. } | HType::Ref { .. } | HType::RawPtr { .. } | HType::OwnPtr { .. } => true, /* pointer/ref binding is always "writable handle" insofar as data ops permit */
            HType::Slice { mutable, .. } => *mutable,
            HType::Array { .. } => matches!(mutness, Mutness::Mut),
            HType::Vec { .. } => true,
            HType::Heap { inner } => match inner.as_ref() {
                HType::Vec { .. } => true,
                _ => !matches!(mutness, Mutness::Const),
            },
            _ => matches!(mutness, Mutness::Mut),
        };
        let reassignable = match ty {
            // pointers are always reassignable
            HType::Ptr { .. } | HType::RawPtr { .. } | HType::OwnPtr { .. } => true,
            // refs, slices, arrays, vectors, heap bindings: handle is fixed
            HType::Ref { .. } | HType::Slice { .. } | HType::Array { .. } | HType::Vec { .. } | HType::Heap { .. } => false,
            // plain values: reassignable iff mut
            _ => matches!(mutness, Mutness::Mut),
        };
        (storage, mut_payload, reassignable)
    }

    fn check_assign(&mut self, op: ast::AssignOp, place: &ast::Expr, value: &ast::Expr, span: Span) -> HStmt {
        // Discard pattern: `_ = expr;` — accept any type, throw away the result.
        if let ast::Expr::Ident(n, _) = place {
            if n == "_" {
                let v = self.check_expr(value, None);
                return HStmt::ExprStmt(v);
            }
        }
        // Determine the place's type and check it is mutable.
        let place_h = self.check_expr(place, None);
        // The *value* type written through this place.
        // If the place's HIR type is `&mut T` and it's a local (e.g. a captured by-mut-ref
        // binding inside a closure body), we write the referent type T transparently.
        let writes_via_ref = matches!(&place_h.kind, HExprKind::Local(_))
            && matches!(&place_h.ty, HType::Ref { mutable: true, .. });
        let pty = if writes_via_ref {
            match &place_h.ty {
                HType::Ref { inner, .. } => (**inner).clone(),
                _ => place_h.ty.clone(),
            }
        } else {
            place_h.ty.clone()
        };
        if !writes_via_ref {
            if let Err(reason) = self.diagnose_place_mutability(&place_h) {
                self.err(reason, span);
            }
        }
        let value_h = self.check_expr_coerce(value, &pty);

        // For compound assignment, the place type must be numeric (basic check).
        let hop = match op {
            ast::AssignOp::Assign => HAssignOp::Assign,
            ast::AssignOp::AddAssign => HAssignOp::Add,
            ast::AssignOp::SubAssign => HAssignOp::Sub,
            ast::AssignOp::MulAssign => HAssignOp::Mul,
            ast::AssignOp::DivAssign => HAssignOp::Div,
            ast::AssignOp::ModAssign => HAssignOp::Mod,
        };
        HStmt::Assign { op: hop, place: place_h, value: value_h, span }
    }

    /// Same shape as `is_place_mutable` but on failure returns a specific
    /// reason naming the offending binding or field, so the user sees
    /// "binding `dead` is not declared `mut`" instead of the generic
    /// "cannot assign to an immutable place".
    fn diagnose_place_mutability(&self, e: &HExpr) -> Result<(), String> {
        match &e.kind {
            HExprKind::GlobalRef(gid) => {
                let g = &self.sym.globals[gid.0 as usize];
                if g.is_mut { Ok(()) } else {
                    Err(format!("cannot assign to global `{}` - it was declared without `mut`", g.name))
                }
            }
            HExprKind::Local(id) => {
                let li = self.local(*id);
                let ok = match &li.ty {
                    HType::Ptr { .. } | HType::RawPtr { .. } | HType::OwnPtr { .. } => true,
                    HType::Ref { .. } => false,
                    HType::Heap { .. } => true,
                    _ => li.mut_payload && li.reassignable,
                };
                if ok { Ok(()) } else {
                    if matches!(li.ty, HType::Ref { .. }) {
                        Err(format!("cannot assign through `{}` - it is an immutable reference (`&T`); use `&mut T` to write through it", li.name))
                    } else {
                        Err(format!("cannot assign to local `{}` - it was declared without `mut`", li.name))
                    }
                }
            }
            HExprKind::Field { base, field } => {
                // Walk the base first - if the base itself is non-writable,
                // surface that reason verbatim.
                self.diagnose_place_target(base)?;
                let sid = match struct_id_of(&base.ty) {
                    Some(id) => id,
                    None => return Err("field access on non-struct value".to_string()),
                };
                let f = &self.sym.struct_info(sid).fields[*field];
                if let HType::Ptr { .. } = &f.ty { return Ok(()); }
                if f.mut_payload { Ok(()) } else {
                    Err(format!(
                        "cannot assign to field `{}` of `{}` - the field is not declared `mut` in `data {}`",
                        f.name,
                        self.sym.struct_info(sid).name,
                        self.sym.struct_info(sid).name,
                    ))
                }
            }
            HExprKind::Index { base, .. } => self.diagnose_place_target(base),
            HExprKind::Unwrap { expr, .. } => match &expr.ty {
                HType::Ptr { mutable: true, .. }
                | HType::RawPtr { mutable: true, .. }
                | HType::OwnPtr { mutable: true, .. } => Ok(()),
                _ => Err("cannot assign through `*const T` - the pointee is immutable".to_string()),
            },
            _ => Err("cannot assign to this expression - it is not a writable place".to_string()),
        }
    }

    /// Lower-level: is the storage reachable through `e` mutable?  Used by
    /// `diagnose_place_mutability` for Field/Index bases.
    fn diagnose_place_target(&self, e: &HExpr) -> Result<(), String> {
        match &e.kind {
            HExprKind::Local(id) => {
                let li = self.local(*id);
                let ok = match &li.ty {
                    HType::Ref { mutable, .. } => *mutable,
                    HType::Ptr { mutable, .. } => *mutable,
                    HType::RawPtr { mutable, .. } => *mutable,
                    HType::OwnPtr { mutable, .. } => *mutable,
                    HType::Slice { mutable, .. } => *mutable,
                    HType::Array { .. } => li.mut_payload,
                    HType::Heap { .. } => li.mut_payload,
                    _ => li.mut_payload,
                };
                if ok { Ok(()) } else {
                    if matches!(li.ty, HType::Ref { mutable: false, .. }) {
                        Err(format!("cannot write through `{}` - it is `&T` (immutable borrow); use `&mut T`", li.name))
                    } else {
                        Err(format!("cannot write through `{}` - it was declared without `mut`", li.name))
                    }
                }
            }
            HExprKind::Unwrap { expr, .. } => match &expr.ty {
                HType::Ptr { mutable: true, .. }
                | HType::RawPtr { mutable: true, .. }
                | HType::OwnPtr { mutable: true, .. } => Ok(()),
                _ => Err("cannot write through a `*const T`".to_string()),
            },
            HExprKind::Field { base, field } => {
                self.diagnose_place_target(base)?;
                let sid = match struct_id_of(&base.ty) {
                    Some(id) => id,
                    None => return Err("field access on non-struct value".to_string()),
                };
                let f = &self.sym.struct_info(sid).fields[*field];
                if f.mut_payload { Ok(()) } else {
                    Err(format!(
                        "field `{}` of `{}` is not declared `mut`",
                        f.name,
                        self.sym.struct_info(sid).name,
                    ))
                }
            }
            HExprKind::Index { base, .. } => self.diagnose_place_target(base),
            _ => Err("storage is not writable from this expression".to_string()),
        }
    }

    fn is_place_mutable(&self, e: &HExpr) -> bool {
        match &e.kind {
            HExprKind::GlobalRef(gid) => self.sym.globals[gid.0 as usize].is_mut,
            HExprKind::Local(id) => {
                let li = self.local(*id);
                // For pointer bindings: handle is always reassignable.
                // For other types: must be mut binding.
                match &li.ty {
                    HType::Ptr { .. } | HType::RawPtr { .. } | HType::OwnPtr { .. } => true,
                    HType::Ref { .. } => false,
                    HType::Heap { .. } => true, // heap binding reassignment allowed (replaces value in place)
                    _ => li.mut_payload && li.reassignable,
                }
            }
            HExprKind::Field { base, field } => {
                let base_mut = self.deref_target_mut(base);
                let sid = match struct_id_of(&base.ty) {
                    Some(id) => id,
                    None => return false,
                };
                let f = &self.sym.struct_info(sid).fields[*field];
                // Pointer (`*T`) fields are always reassignable per §15.1 (handles).
                // The pointee mutability matters at deref time, not at field reassignment.
                if let HType::Ptr { .. } = &f.ty { return base_mut; }
                base_mut && f.mut_payload
            }
            HExprKind::Index { base, .. } => self.deref_target_mut(base),
            HExprKind::Unwrap { expr, .. } => match &expr.ty {
                HType::Ptr { mutable, .. } => *mutable,
                HType::RawPtr { mutable, .. } => *mutable,
                HType::OwnPtr { mutable, .. } => *mutable,
                _ => false,
            },
            _ => false,
        }
    }

    /// Is the *underlying storage* through this expression mutable?
    fn deref_target_mut(&self, e: &HExpr) -> bool {
        match &e.kind {
            HExprKind::Local(id) => {
                let li = self.local(*id);
                match &li.ty {
                    HType::Ref { mutable, .. } => *mutable,
                    HType::Ptr { mutable, .. } => *mutable,
                    HType::RawPtr { mutable, .. } => *mutable,
                    HType::OwnPtr { mutable, .. } => *mutable,
                    HType::Slice { mutable, .. } => *mutable,
                    HType::Array { .. } => li.mut_payload,
                    HType::Heap { .. } => li.mut_payload,
                    _ => li.mut_payload,
                }
            }
            HExprKind::Unwrap { expr, .. } => matches!(
                expr.ty,
                HType::Ptr { mutable: true, .. }
                | HType::RawPtr { mutable: true, .. }
                | HType::OwnPtr { mutable: true, .. }
            ),
            HExprKind::Field { base, field } => {
                let base_mut = self.deref_target_mut(base);
                let sid = match struct_id_of(&base.ty) {
                    Some(id) => id,
                    None => return false,
                };
                let f = &self.sym.struct_info(sid).fields[*field];
                base_mut && f.mut_payload
            }
            HExprKind::Index { base, .. } => self.deref_target_mut(base),
            _ => false,
        }
    }

    // ---- expressions ----

    fn check_expr_coerce(&mut self, e: &ast::Expr, target: &HType) -> HExpr {
        let h = self.check_expr(e, Some(target));
        self.coerce(h, target)
    }

    fn check_expr(&mut self, e: &ast::Expr, expected: Option<&HType>) -> HExpr {
        match e {
            ast::Expr::Lit(l, sp) => self.check_lit(l, expected, *sp),
            ast::Expr::Ident(n, sp) => self.check_ident(n, *sp),
            ast::Expr::Bin { op, lhs, rhs, span } => self.check_bin(*op, lhs, rhs, *span),
            ast::Expr::Un { op, expr, span } => self.check_un(*op, expr, *span),
            ast::Expr::Unwrap { expr, span } => self.check_unwrap(expr, *span),
            ast::Expr::Ref { mutness, expr, span } => self.check_ref(matches!(mutness, Mutness::Mut), expr, *span, expected),
            ast::Expr::Field { base, name, span } => self.check_field(base, name, expected, *span),
            ast::Expr::Index { base, idx, span } => self.check_index(base, idx, *span),
            ast::Expr::Call { callee, args, span } => {
                self.call_ret_expected = expected.cloned();
                self.check_call(callee, args, *span)
            }
            ast::Expr::AttrCall { attr, name, receiver, args, span } => {
                self.check_attr_call(attr, name, receiver.as_deref(), args, *span)
            }
            ast::Expr::Cast { expr, ty, span } => self.check_cast(expr, ty, false, *span),
            ast::Expr::CheckedCast { expr, ty, span } => self.check_cast(expr, ty, true, *span),
            ast::Expr::Struct { ty, fields, span } => self.check_struct_lit(ty.as_deref(), fields, expected, *span),
            ast::Expr::ArrayLit { elems, span } => self.check_array_lit(elems, expected, *span),
            ast::Expr::VariantCtor { enum_name, variant, fields, span } => {
                self.check_variant_ctor(enum_name, variant, fields, expected, *span)
            }
            ast::Expr::Match { scrutinee, arms, span } => {
                self.check_match(scrutinee, arms, false, *span, expected)
            }
            ast::Expr::Lambda { ret, params, captures, body, span } => {
                // Lambdas with captures: synthesize an env struct + lifted fn here.
                // Lambdas without captures should have been lifted at AST level.
                self.check_capturing_lambda(ret, params, captures, body, *span)
            }
            ast::Expr::WallMod { mode, expr, span } => {
                if self.call_arg_depth == 0 {
                    self.err("`transfer`/`share` are only valid as direct call arguments at gate crossings", *span);
                } else if !self.cur_call_is_gate {
                    self.err("`transfer`/`share` require the callee to be declared `gate`", *span);
                }
                let h = self.check_expr(expr, expected);
                match mode {
                    ast::WallMode::Share => {
                        // `Rust<T>` is opaque to Maka's Shareable check; delegate
                        // to rustc by emitting a `Sync` probe.  Native Maka types
                        // still go through the Shareable rule.
                        if let HType::RustOpaque(label) = &h.ty {
                            self.sync_probes.push(label.clone());
                        } else if !self.is_shareable(&h.ty) {
                            self.err(format!("type `{}` is not Shareable; cannot `share` across a gate", type_str(&h.ty)), *span);
                        }
                        h
                    }
                    ast::WallMode::Transfer => {
                        // `Rust<T>` transferred across a gate needs `Send`.
                        if let HType::RustOpaque(label) = &h.ty {
                            self.send_probes.push(label.clone());
                        }
                        let ty = h.ty.clone();
                        HExpr {
                            kind: HExprKind::Transfer(Box::new(h)),
                            ty,
                            span: *span,
                        }
                    }
                }
            }
            ast::Expr::HeapAlloc { value, span } => {
                // `alloc value`: must land in an owning slot (`own *T` or `own &T`).
                // Letting an alloc'd pointer flow into a plain `*T` (non-owning) slot
                // would create an untracked allocation that nobody auto-frees — a
                // memory leak waiting to happen.  Landing in `raw *T` is allowed
                // ONLY inside `unsafe { ... }` (the manual-memory escape hatch,
                // paired with `free p;` for teardown).
                let inner_expected = match expected {
                    Some(HType::OwnPtr { inner, .. })
                    | Some(HType::Heap { inner })
                    | Some(HType::RawPtr { inner, .. }) => Some((**inner).clone()),
                    _ => None,
                };
                let h = if let Some(ie) = inner_expected.as_ref() {
                    self.check_expr_coerce(value, ie)
                } else {
                    self.check_expr(value, None)
                };
                let inner_ty = h.ty.clone();
                let result_ty = match expected {
                    Some(HType::OwnPtr { mutable, .. }) => HType::OwnPtr { mutable: *mutable, inner: Box::new(inner_ty) },
                    Some(HType::Heap { .. })            => HType::Heap   { inner: Box::new(inner_ty) },
                    Some(HType::RawPtr { mutable, .. }) => {
                        if self.in_unsafe == 0 {
                            self.err(
                                "`alloc` into `raw *T` is the manual-memory escape hatch and requires \
                                 `unsafe { ... }` — landing an allocation in a `raw *T` opts out of \
                                 the auto-free machinery, so the caller must release it explicitly \
                                 (`free p;`) inside the same `unsafe` block.",
                                *span,
                            );
                        }
                        HType::RawPtr { mutable: *mutable, inner: Box::new(inner_ty) }
                    }
                    Some(HType::Ptr { .. }) => {
                        self.err(
                            "`alloc value` must land in an owning slot (`own *T` or `own &T`) — \
                             assigning an allocation to a non-owning `*T` would leak with no auto-free. \
                             Declare the binding as `own *T` or downgrade explicitly later.",
                            *span,
                        );
                        // Continue with Ptr so we don't cascade errors.
                        HType::Ptr { mutable: true, inner: Box::new(inner_ty) }
                    }
                    // No explicit target type — default to nullable owning so the
                    // result is captured and auto-freed.
                    _ => HType::OwnPtr { mutable: true, inner: Box::new(inner_ty) },
                };
                HExpr {
                    kind: HExprKind::HeapAlloc(Box::new(h)),
                    ty: result_ty,
                    span: *span,
                }
            }
            ast::Expr::Free { value, span } => {
                // `free value`: bare-word deallocator for `raw *T`, only inside `unsafe { }`.
                let h = self.check_expr(value, None);
                let ok_ty = matches!(h.ty, HType::RawPtr { .. });
                if !ok_ty {
                    self.err(
                        format!(
                            "`free` only accepts `raw *T` (the manual-memory escape hatch); got `{}`. \
                             Maka-managed allocations (`own *T` / `own &T`) auto-free at scope exit — \
                             assign `null` to the owner to release one early.",
                            type_str(&h.ty),
                        ),
                        *span,
                    );
                }
                if self.in_unsafe == 0 {
                    self.err(
                        "`free p;` requires `unsafe { ... }` — manual deallocation through a `raw *T` \
                         is unsafe (the lifetime pass can't prove the pointer is still live or that \
                         no other view aliases it).",
                        *span,
                    );
                }
                HExpr {
                    kind: HExprKind::Free(Box::new(h)),
                    ty: HType::Unit,
                    span: *span,
                }
            }
        }
    }

    fn check_lit(&mut self, l: &ast::Lit, expected: Option<&HType>, sp: Span) -> HExpr {
        match l {
            ast::Lit::Int(n) => {
                // promote to float if expected
                if matches!(expected, Some(HType::Float)) {
                    HExpr { kind: HExprKind::LitFloat(*n as f64), ty: HType::Float, span: sp }
                } else if matches!(expected, Some(HType::Char)) {
                    HExpr { kind: HExprKind::LitChar((*n as u8) as char), ty: HType::Char, span: sp }
                } else if let Some(t @ HType::SizedInt { .. }) = expected {
                    HExpr { kind: HExprKind::LitInt(*n), ty: t.clone(), span: sp }
                } else {
                    HExpr { kind: HExprKind::LitInt(*n), ty: HType::Int, span: sp }
                }
            }
            ast::Lit::Float(f) => {
                // Sized-float literals (`1.5f32`) coerce to the expected
                // SizedFloat width.  Default is `HType::Float` (= C double).
                if let Some(t @ HType::SizedFloat { .. }) = expected {
                    HExpr { kind: HExprKind::LitFloat(*f), ty: t.clone(), span: sp }
                } else {
                    HExpr { kind: HExprKind::LitFloat(*f), ty: HType::Float, span: sp }
                }
            }
            ast::Lit::Bool(b) => HExpr { kind: HExprKind::LitBool(*b), ty: HType::Bool, span: sp },
            ast::Lit::Char(c) => HExpr { kind: HExprKind::LitChar(*c), ty: HType::Char, span: sp },
            ast::Lit::Str(s) => HExpr { kind: HExprKind::LitStr(s.clone()), ty: HType::Str, span: sp },
            ast::Lit::Null => HExpr { kind: HExprKind::LitNull, ty: HType::NullT, span: sp },
            ast::Lit::Unit => HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp },
        }
    }

    fn check_ident(&mut self, n: &str, sp: Span) -> HExpr {
        if let Some(id) = self.lookup(n) {
            let ty = self.local(id).ty.clone();
            return HExpr { kind: HExprKind::Local(id), ty, span: sp };
        }
        // Function name used as a value (function pointer).
        if let Some((fid, sig)) = self.sym.func_by_name(n) {
            let ty = HType::FnPtr {
                ret: Box::new(sig.ret.clone()),
                params: sig.param_tys.clone(),
            };
            return HExpr { kind: HExprKind::FnRef(fid), ty, span: sp };
        }
        // Constexpr lookup: a name in any module that defines `pub constexpr int N = ...;`
        // and is imported (or in the same module) becomes a literal int at this site.
        // In-file constexprs were already inlined as int literals by the parser's
        // fold-map, so they never reach this code path - but they show up here when
        // referenced cross-module.
        if let Some(ce) = self.find_constexpr(n) {
            return HExpr { kind: HExprKind::LitInt(ce.value), ty: HType::Int, span: sp };
        }
        // Module-scope `mut` / immutable globals: same lookup rules as constexprs
        // (same-module always visible; cross-module needs pub + import).
        if let Some((gid, info)) = self.find_global(n) {
            return HExpr { kind: HExprKind::GlobalRef(gid), ty: info.ty.clone(), span: sp };
        }
        self.err(format!("unknown identifier `{}`", n), sp);
        HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp }
    }

    /// Type-check a `format(fmt_lit, args...)` call.  Parses the format string
    /// at compile time, splits on `{}` placeholders, validates arg count, and
    /// lowers each placeholder to a per-type "value to string" converter -
    /// then the whole thing chains through string concat so the result is an
    /// `own *char` auto-freed at scope exit.
    fn check_format(&mut self, args: &[ast::Expr], sp: Span) -> HExpr {
        if args.is_empty() {
            self.err("format requires at least a format-string argument", sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        }
        let fmt_str = match &args[0] {
            ast::Expr::Lit(ast::Lit::Str(s), _) => s.clone(),
            _ => {
                self.err("format's first argument must be a string literal", sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
        };
        // Fast path: all-scalar placeholders lower to a single `__maka_format1`
        // (one allocation) instead of a chain of per-arg concat mallocs.  Returns
        // `own *char`, auto-freed like the concat result.
        if let Some((pf, vals)) = self.build_printf_parts(&fmt_str, &args[1..]) {
            let mut hargs = Vec::with_capacity(vals.len() + 1);
            hargs.push(HExpr { kind: HExprKind::LitStr(pf), ty: HType::Str, span: sp });
            hargs.extend(vals);
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 59), args: hargs },
                ty: HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) },
                span: sp,
            };
        }
        let parts: Vec<&str> = fmt_str.split("{}").collect();
        let expected = parts.len().saturating_sub(1);
        let provided = args.len() - 1;
        if provided != expected {
            self.err(format!(
                "format expected {} placeholder argument(s), got {}",
                expected, provided,
            ), sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        }
        // Build concat tree.  Start with the first literal segment.
        let mut acc = HExpr { kind: HExprKind::LitStr(parts[0].to_string()), ty: HType::Str, span: sp };
        for (i, arg) in args[1..].iter().enumerate() {
            let arg_h = self.check_expr(arg, None);
            let converted = self.format_arg_to_string(arg_h, sp);
            acc = self.format_concat(acc, converted, sp);
            let next_lit = HExpr { kind: HExprKind::LitStr(parts[i + 1].to_string()), ty: HType::Str, span: sp };
            acc = self.format_concat(acc, next_lit, sp);
        }
        acc
    }

    /// Translate a `{}` format template + args into a printf format string (no
    /// trailing newline) and the checked value exprs - shared by the
    /// `log(format(...))->printf` and `format(...)->snprintf` lowerings.  Returns
    /// None if the placeholder count mismatches or any arg isn't a printf-able
    /// scalar, so the caller falls back to the (correctly-erroring) concat path.
    fn build_printf_parts(&mut self, fmt: &str, fmt_args: &[ast::Expr]) -> Option<(String, Vec<HExpr>)> {
        let parts: Vec<&str> = fmt.split("{}").collect();
        if parts.len().saturating_sub(1) != fmt_args.len() { return None; }
        let mut pf = String::new();
        let mut vals: Vec<HExpr> = Vec::with_capacity(fmt_args.len());
        for (i, arg) in fmt_args.iter().enumerate() {
            pf.push_str(&parts[i].replace('%', "%%"));
            let mut h = self.check_expr(arg, None);
            h = match &h.ty {
                HType::Ref { inner, .. } if matches!(**inner, HType::Int | HType::SizedInt { .. } | HType::Float | HType::Bool | HType::Char) => self.auto_deref(h),
                _ => h,
            };
            // An owned `own *char` argument is consumed as a borrowed view here
            // (printf/snprintf reads it, doesn't take ownership).  Retype it as
            // `string` so the move analysis treats it as borrowed - the value is
            // still a fresh-temp Call, so the lifetime pass hoists and frees it.
            if matches!(&h.ty, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char)) {
                h.ty = HType::Str;
            }
            let spec = match &h.ty {
                HType::Int | HType::SizedInt { .. } => "%lld",
                HType::Float | HType::SizedFloat { .. } => "%g",
                HType::Bool => "%s",
                HType::Char => "%c",
                HType::Str => "%s",
                HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char) => "%s",
                _ => return None,
            };
            pf.push_str(spec);
            vals.push(h);
        }
        pf.push_str(&parts[parts.len() - 1].replace('%', "%%"));
        Some((pf, vals))
    }

    /// Lower `log(format(fmt, args...))` to a printf builtin (no allocation).
    fn try_log_format_print(&mut self, fmt: &str, fmt_args: &[ast::Expr], sp: Span) -> Option<HExpr> {
        let (mut pf, vals) = self.build_printf_parts(fmt, fmt_args)?;
        pf.push('\n'); // log() appends a newline
        let mut hargs = Vec::with_capacity(vals.len() + 1);
        hargs.push(HExpr { kind: HExprKind::LitStr(pf), ty: HType::Str, span: sp });
        hargs.extend(vals);
        Some(HExpr { kind: HExprKind::Call { callee: FuncId(u32::MAX - 58), args: hargs }, ty: HType::Unit, span: sp })
    }

    /// Convert one HExpr arg to a stringy HExpr suitable for `format_concat`.
    /// Each primitive type maps to a reserved builtin FuncId; strings pass
    /// through unchanged.
    fn format_arg_to_string(&mut self, h: HExpr, sp: Span) -> HExpr {
        let result_ty = HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) };
        match &h.ty {
            HType::Int | HType::SizedInt { .. } => HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 11), args: vec![h] },
                ty: result_ty, span: sp,
            },
            HType::Bool => HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 12), args: vec![h] },
                ty: HType::Str, span: sp,
            },
            HType::Float => HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 13), args: vec![h] },
                ty: result_ty, span: sp,
            },
            HType::Char => HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 14), args: vec![h] },
                ty: result_ty, span: sp,
            },
            HType::Str | HType::OwnPtr { .. } => h,
            _ => {
                self.err(format!(
                    "format placeholder for type `{}` is not supported (int/float/bool/char/string only)",
                    type_str(&h.ty),
                ), sp);
                HExpr { kind: HExprKind::LitStr("".to_string()), ty: HType::Str, span: sp }
            }
        }
    }

    /// Concatenate two stringy HExprs using the same `__maka_str_concat`
    /// family used by `a + b`.  Picks the right freeing variant based on
    /// which side owns its buffer.
    fn format_concat(&self, l: HExpr, r: HExpr, sp: Span) -> HExpr {
        let is_owning_char = |t: &HType| matches!(t, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char));
        let l_owns = is_owning_char(&l.ty);
        let r_owns = is_owning_char(&r.ty);
        let helper_id = match (l_owns, r_owns) {
            (false, false) => u32::MAX - 5,
            (true,  false) => u32::MAX - 8,
            (false, true ) => u32::MAX - 9,
            (true,  true ) => u32::MAX - 10,
        };
        HExpr {
            kind: HExprKind::Call { callee: FuncId(helper_id), args: vec![l, r] },
            ty: HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) },
            span: sp,
        }
    }

    /// Type-check a module-scope global's initializer expression and coerce it
    /// to the declared type.  Used by `analyze()` before any function bodies
    /// are processed, so globals are visible from every function.
    pub fn check_global_init(mut self, init: &ast::Expr, ty: &HType) -> Result<HExpr, Vec<SemaError>> {
        let h = self.check_expr_coerce(init, ty);
        if self.errors.is_empty() { Ok(h) } else { Err(std::mem::take(&mut self.errors)) }
    }

    /// Find a module-scope global visible from the current file, returning
    /// its id alongside the info.  Same visibility rules as constexprs.
    fn find_global(&self, n: &str) -> Option<(GlobalId, &GlobalInfo)> {
        for (i, g) in self.sym.globals.iter().enumerate() {
            if g.name != n { continue; }
            if g.module_path == self.cur_module {
                return Some((GlobalId(i as u32), g));
            }
            if g.is_pub && self.cur_imports.iter().any(|(p, name)| p == &g.module_path && (name == n || name == "*")) {
                return Some((GlobalId(i as u32), g));
            }
        }
        None
    }

    /// Find a `pub constexpr` named `n` visible from the current module: either
    /// declared here (regardless of pub) or in another module and explicitly
    /// imported via `import path.Name;`.
    fn find_constexpr(&self, n: &str) -> Option<&ConstexprInfo> {
        // Same-module: any visibility.
        if let Some(ce) = self.sym.constexprs.iter().find(|c| c.name == n && c.module_path == self.cur_module) {
            return Some(ce);
        }
        // Cross-module: must be pub AND imported.
        self.sym.constexprs.iter().find(|c| {
            c.name == n && c.is_pub
                && self.cur_imports.iter().any(|(p, name)| p == &c.module_path && (name == n || name == "*"))
        })
    }

    fn check_bin(&mut self, op: ast::BinOp, lhs: &ast::Expr, rhs: &ast::Expr, sp: Span) -> HExpr {
        use ast::BinOp::*;
        match op {
            Add | Sub | Mul | Div | Mod => {
                let l = self.check_expr(lhs, None);
                let r = self.check_expr(rhs, Some(&l.ty));
                // String concat: `a + b` where each operand is either `string` or
                // `own *char` (a previous concat result).  Lowers to a runtime helper
                // that mallocs a NUL-terminated buffer and returns it.  Result type
                // is `own *char` — auto-freed at scope exit, coerces back to `string`
                // anywhere a borrowed view is wanted.  Chained concats `a + b + c`
                // therefore work end-to-end and free each intermediate.
                let is_owning_char = |t: &HType| matches!(t, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char));
                let is_strish = |t: &HType| matches!(t, HType::Str) || is_owning_char(t);
                if matches!(op, Add) && is_strish(&l.ty) && is_strish(&r.ty) {
                    // Pick the helper that frees whichever operands were owning
                    // intermediates, so chained `a + b + c` doesn't leak.
                    let l_owns = is_owning_char(&l.ty);
                    let r_owns = is_owning_char(&r.ty);
                    let helper_id = match (l_owns, r_owns) {
                        (false, false) => u32::MAX - 5,    // both borrowed
                        (true,  false) => u32::MAX - 8,    // free left  → _freel
                        (false, true ) => u32::MAX - 9,    // free right → _freer
                        (true,  true ) => u32::MAX - 10,   // free both  → _freeb
                    };
                    let result_ty = HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) };
                    return HExpr {
                        kind: HExprKind::Call {
                            callee: FuncId(helper_id),
                            args: vec![l, r],
                        },
                        ty: result_ty,
                        span: sp,
                    };
                }
                // If either operand is non-primitive, try operator overload dispatch.
                if !self.is_numeric(&l.ty) || !self.is_numeric(&r.ty) {
                    if let Some(hir) = self.try_op_overload(op, &l, &r, sp) {
                        return hir;
                    }
                    self.err("arithmetic on non-numeric types", sp);
                }
                let l = self.auto_deref(l);
                let r = self.auto_deref(r);
                let r = self.coerce(r, &l.ty);
                let ty = l.ty.clone();
                HExpr { kind: HExprKind::Bin { op: bin_to_hir(op), lhs: Box::new(l), rhs: Box::new(r) }, ty, span: sp }
            }
            Lt | Le | Gt | Ge => {
                let l = self.check_expr(lhs, None);
                let r = self.check_expr(rhs, Some(&l.ty));
                if !self.is_numeric(&l.ty) || !self.is_numeric(&r.ty) {
                    if let Some(hir) = self.try_op_overload(op, &l, &r, sp) {
                        return hir;
                    }
                    self.err("ordering on non-numeric types", sp);
                }
                let l = self.auto_deref(l);
                let r = self.auto_deref(r);
                let r = self.coerce(r, &l.ty);
                HExpr { kind: HExprKind::Bin { op: bin_to_hir(op), lhs: Box::new(l), rhs: Box::new(r) }, ty: HType::Bool, span: sp }
            }
            Eq | Ne => {
                let l = self.check_expr(lhs, None);
                let r = self.check_expr(rhs, Some(&l.ty));
                let l_is_ptr = matches!(l.ty, HType::Ptr { .. });
                let r_is_null = matches!(r.ty, HType::NullT);
                let r_is_ptr = matches!(r.ty, HType::Ptr { .. });
                let l_is_null = matches!(l.ty, HType::NullT);
                let l_is_prim = self.is_primitive(&l.ty);
                let r_is_prim = self.is_primitive(&r.ty);
                // Enum-to-enum comparison: two values of the same enum type can
                // be compared directly.  For simple (payload-less) enums this
                // is just an integer tag compare; for tagged enums the codegen
                // peels off `.tag` on both sides.
                let l_is_enum = matches!(l.ty, HType::Enum(_));
                let r_is_enum = matches!(r.ty, HType::Enum(_));
                let same_enum = l_is_enum && r_is_enum && type_eq(&l.ty, &r.ty);
                if !l_is_prim && !r_is_null && !l_is_ptr && !same_enum {
                    if let Some(hir) = self.try_op_overload(op, &l, &r, sp) {
                        return hir;
                    }
                }
                let _ = r_is_prim;
                let (lh, rh) = if l_is_ptr && r_is_null {
                    let tgt = l.ty.clone();
                    (l, self.coerce(r, &tgt))
                } else if l_is_null && r_is_ptr {
                    let tgt = r.ty.clone();
                    (self.coerce(l, &tgt), r)
                } else {
                    let tgt = l.ty.clone();
                    (l, self.coerce(r, &tgt))
                };
                HExpr { kind: HExprKind::Bin { op: bin_to_hir(op), lhs: Box::new(lh), rhs: Box::new(rh) }, ty: HType::Bool, span: sp }
            }
            And | Or => {
                let l = self.check_expr_coerce(lhs, &HType::Bool);
                let r = self.check_expr_coerce(rhs, &HType::Bool);
                HExpr { kind: HExprKind::Bin { op: bin_to_hir(op), lhs: Box::new(l), rhs: Box::new(r) }, ty: HType::Bool, span: sp }
            }
            BitAnd | BitOr | BitXor | Shl | Shr => {
                let l = self.check_expr(lhs, None);
                let r = self.check_expr(rhs, Some(&l.ty));
                let lprim = self.is_primitive(&l.ty);
                let rprim = self.is_primitive(&r.ty);
                if !lprim || !rprim {
                    if let Some(h) = self.try_op_overload(op, &l, &r, sp) {
                        return h;
                    }
                    self.err("bitwise operator on non-numeric types", sp);
                }
                let l_int = matches!(l.ty, HType::Int | HType::SizedInt { .. });
                let r_int = matches!(r.ty, HType::Int | HType::SizedInt { .. });
                if !l_int || !r_int {
                    self.err("bitwise operators require integer types", sp);
                }
                let ty = l.ty.clone();
                HExpr { kind: HExprKind::Bin { op: bin_to_hir(op), lhs: Box::new(l), rhs: Box::new(r) }, ty, span: sp }
            }
        }
    }

    fn is_primitive(&self, t: &HType) -> bool {
        matches!(t,
            HType::Int | HType::SizedInt { .. } | HType::Float | HType::Bool
            | HType::Char | HType::Unit | HType::Str | HType::NullT)
    }

    /// `Shareable` per §D11: primitives, sync primitives, all-Shareable structs/enums,
    /// `&const T` to a Shareable T are shareable. `*T` to mutable data is NOT shareable.
    pub fn is_shareable(&self, t: &HType) -> bool {
        match t {
            HType::Int | HType::SizedInt { .. } | HType::Float | HType::SizedFloat { .. } | HType::Bool
            | HType::Char | HType::Unit | HType::Str | HType::NullT => true,
            HType::GenericPattern { .. } => false,
            // Sync primitives recognized by name (auto-recognized stdlib types).
            HType::Struct(id) => {
                let info = self.sym.struct_info(*id);
                // For an instantiation (e.g. `Atomic<int>`), match the template
                // base name ("Atomic"), not the mangled instantiation name.
                let n = info.template.as_deref().unwrap_or(info.name.as_str());
                if matches!(n,
                    "Mutex" | "RwLock" | "Spinlock" | "Channel" | "Thread"
                    | "Atomic" | "WaitGroup" | "Once" | "IntChan" | "FloatChan" | "ByteChan"
                    | "TlsConn") {
                    return true;
                }
                // Auto-derive: all fields must be Shareable.
                info.fields.iter().all(|f| self.is_shareable(&f.ty))
            }
            HType::Enum(id) => {
                let info = self.sym.enum_info(*id);
                info.variants.iter().all(|v| v.fields.iter().all(|f| self.is_shareable(&f.ty)))
            }
            HType::Ref { mutable, inner } => {
                // &const T is shareable iff T is Shareable. &mut T is NOT shareable.
                !mutable && self.is_shareable(inner)
            }
            // *T (mutable pointer) is NOT shareable. *const T is shareable iff target is.
            HType::Ptr { mutable, inner } => {
                !mutable && self.is_shareable(inner)
            }
            // raw *T: NEVER shareable.  The provenance is unknown, so even immutable
            // raw pointers may alias data being mutated outside Maka's tracking.
            HType::RawPtr { .. } => false,
            // own *T is sole-owner; sharing means two threads might free it.
            HType::OwnPtr { .. } => false,
            // Heap, Array, Slice, Vec: shareable iff element/inner is shareable AND not mutable
            // (we conservatively reject Vec since it has mutable internals).
            HType::Heap { inner } => self.is_shareable(inner),
            HType::Array { elem, .. } => self.is_shareable(elem),
            HType::Slice { mutable, elem } => !mutable && self.is_shareable(elem),
            HType::Vec { .. } => false,
            HType::Dyn { .. } => false,
            HType::FnPtr { .. } => true,
            HType::TyVar(_) => false,
            // Unresolved associated-type — Shareable check is per concrete
            // instantiation; abstract form has no verdict, conservatively false.
            HType::AssocType { .. } => false,
            // A `Rust<T>` is `own *mut unit` semantically: sole-owner, never
            // shareable.  Probe routing for `share` is delegated to rustc's
            // `Sync` bound; see RUST_INTEROP.md §6.
            HType::RustOpaque(_) => false,
        }
    }

    fn local_mut_invalidate(&mut self, id: LocalId) {
        // We mark the local as "moved" for downstream uses. We piggy-back on the
        // existing `mut_payload`/`reassignable` flags by setting both to false.
        let li = &mut self.locals[id.0 as usize];
        li.mut_payload = false;
        li.reassignable = false;
    }

    /// Try to dispatch `op` as a call to a user-defined logic block.
    /// Returns Some(hir) on success.
    fn try_op_overload(&mut self, op: ast::BinOp, l: &HExpr, r: &HExpr, sp: Span) -> Option<HExpr> {
        use ast::BinOp::*;
        let (logic_name, fn_name) = match op {
            Add => ("Add", "add"),
            Sub => ("Sub", "sub"),
            Mul => ("Mul", "mul"),
            Div => ("Div", "div"),
            Mod => ("Mod", "mod"),
            Eq | Ne => ("Eq", "eq"),
            Lt | Le | Gt | Ge => ("Ord", "cmp"),
            BitAnd => ("BitAnd", "band"),
            BitOr => ("BitOr", "bor"),
            BitXor => ("BitXor", "bxor"),
            Shl => ("Shl", "shl"),
            Shr => ("Shr", "shr"),
            And | Or => return None,
        };
        let logic_info = self.sym.logic_by_name(logic_name)?.clone();
        // Find a candidate matching by name + arity + first-param compatibility.
        let cand = logic_info.funcs.iter().find_map(|fid| {
            let sig = self.sym.func_sig(*fid);
            if sig.name == fn_name && sig.param_tys.len() == 2 {
                if param_compatible(&sig.param_tys[0], &l.ty, &sig.type_params)
                    && param_compatible(&sig.param_tys[1], &r.ty, &sig.type_params) {
                    return Some((*fid, sig.clone()));
                }
            }
            None
        })?;
        let (fid, sig) = cand;
        let ret = sig.ret.clone();
        let lh = self.coerce(l.clone(), &sig.param_tys[0]);
        let rh = self.coerce(r.clone(), &sig.param_tys[1]);
        let call = HExpr {
            kind: HExprKind::Call { callee: fid, args: vec![lh, rh] },
            ty: ret.clone(),
            span: sp,
        };
        // For comparison ops, post-process Ord.cmp into a comparison.
        match op {
            Eq => Some(call),
            Ne => Some(HExpr {
                kind: HExprKind::Un { op: HUnOp::Not, expr: Box::new(call) },
                ty: HType::Bool,
                span: sp,
            }),
            Lt | Le | Gt | Ge => {
                let cmp_op = match op {
                    Lt => HBinOp::Lt,
                    Le => HBinOp::Le,
                    Gt => HBinOp::Gt,
                    Ge => HBinOp::Ge,
                    _ => unreachable!(),
                };
                let zero = HExpr { kind: HExprKind::LitInt(0), ty: HType::Int, span: sp };
                Some(HExpr {
                    kind: HExprKind::Bin { op: cmp_op, lhs: Box::new(call), rhs: Box::new(zero) },
                    ty: HType::Bool,
                    span: sp,
                })
            }
            _ => Some(call),
        }
    }

    fn is_numeric(&self, t: &HType) -> bool {
        match t {
            // `char` and `u8` are the same byte type (see resolve.rs), so byte
            // arithmetic / ordering (`'A' + 25`, `a < 'B'`, u8 math) is allowed.
            HType::Int | HType::SizedInt { .. } | HType::SizedFloat { .. }
            | HType::Float | HType::Char => true,
            HType::Ref { inner, .. } => self.is_numeric(inner),
            _ => false,
        }
    }

    /// If `e` is a `&T` or `&mut T` to a numeric, insert an implicit deref.
    fn auto_deref(&self, e: HExpr) -> HExpr {
        match &e.ty {
            HType::Ref { inner, .. } => {
                let inner_ty = (**inner).clone();
                let span = e.span;
                HExpr { kind: HExprKind::DerefRef(Box::new(e)), ty: inner_ty, span }
            }
            _ => e,
        }
    }

    fn check_un(&mut self, op: ast::UnOp, e: &ast::Expr, sp: Span) -> HExpr {
        match op {
            ast::UnOp::Neg => {
                let h = self.check_expr(e, None);
                if !self.is_numeric(&h.ty) {
                    // Try Neg overload.
                    if let Some(hir) = self.try_unary_overload("Neg", "neg", &h, sp) {
                        return hir;
                    }
                    self.err("negation of non-numeric value", sp);
                }
                let ty = h.ty.clone();
                HExpr { kind: HExprKind::Un { op: HUnOp::Neg, expr: Box::new(h) }, ty, span: sp }
            }
            ast::UnOp::Not => {
                let h = self.check_expr_coerce(e, &HType::Bool);
                HExpr { kind: HExprKind::Un { op: HUnOp::Not, expr: Box::new(h) }, ty: HType::Bool, span: sp }
            }
        }
    }

    fn try_unary_overload(&mut self, logic_name: &str, fn_name: &str, h: &HExpr, sp: Span) -> Option<HExpr> {
        let info = self.sym.logic_by_name(logic_name)?.clone();
        let cand = info.funcs.iter().find_map(|fid| {
            let sig = self.sym.func_sig(*fid);
            if sig.name == fn_name && sig.param_tys.len() == 1 {
                if param_compatible(&sig.param_tys[0], &h.ty, &sig.type_params) {
                    return Some((*fid, sig.clone()));
                }
            }
            None
        })?;
        let (fid, sig) = cand;
        let arg = self.coerce(h.clone(), &sig.param_tys[0]);
        let ret = sig.ret.clone();
        Some(HExpr {
            kind: HExprKind::Call { callee: fid, args: vec![arg] },
            ty: ret,
            span: sp,
        })
    }

    fn check_unwrap(&mut self, e: &ast::Expr, sp: Span) -> HExpr {
        let h = self.check_expr(e, None);
        match &h.ty {
            HType::Ptr { mutable: _, inner } => {
                let inner_ty = (**inner).clone();
                // Spec says p! has type &mut T / &const T, but we treat it as a value/place of T
                // (with assignment writing through it). Codegen emits `*p`.
                HExpr { kind: HExprKind::Unwrap { expr: Box::new(h), skip_check: false }, ty: inner_ty, span: sp }
            }
            HType::RawPtr { mutable: _, inner } => {
                if self.in_unsafe == 0 {
                    self.err(
                        "deref of `raw *T` requires `unsafe { ... }` — this pointer's provenance \
                         is unknown to the lifetime pass, so the user must vouch for its validity"
                            .to_string(),
                        sp,
                    );
                }
                let inner_ty = (**inner).clone();
                HExpr { kind: HExprKind::Unwrap { expr: Box::new(h), skip_check: true }, ty: inner_ty, span: sp }
            }
            HType::OwnPtr { mutable: _, inner } => {
                // own *T deref follows the same forced-handling rule as plain *T —
                // the value can be null, so the user must guard before deref.
                let inner_ty = (**inner).clone();
                HExpr { kind: HExprKind::Unwrap { expr: Box::new(h), skip_check: false }, ty: inner_ty, span: sp }
            }
            _ => {
                self.err("`!` only valid on pointers", sp);
                HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp }
            }
        }
    }

    fn check_ref(&mut self, want_mut: bool, e: &ast::Expr, sp: Span, expected: Option<&HType>) -> HExpr {
        // Determine result type from the expression's storage mutability.
        let h = self.check_expr(e, None);
        let place_mut = self.is_place_addr_mut(&h);
        if want_mut && !place_mut {
            self.err("cannot take `&mut` to immutable storage", sp);
        }
        // Ref-peel: a place whose type is already `&T` / `&mut T` reads as the
        // pointee value (auto-deref applies), so taking its address gives back
        // the address it stores — NOT the address of the binding.  Emit
        // `&(*place)` at codegen via DerefRef so the result has the runtime
        // value of `place` itself (an address), not `&place` (address of the
        // local that holds the address).  Consistent with how `b.v` already
        // auto-derefs through the same Ref.
        let (place_h, inner) = match &h.ty.clone() {
            // Peel only for "thin" referents (plain T).  Fat-pointer kinds
            // (`dyn Trait`, slices) carry extra metadata in the ref value;
            // `&(*m)` doesn't round-trip for them — fall through to the
            // existing reborrow/coerce path.
            HType::Ref { inner, .. } if !matches!(**inner, HType::Dyn { .. } | HType::Slice { .. }) => {
                let deref = HExpr {
                    kind: HExprKind::DerefRef(Box::new(h.clone())),
                    ty: (**inner).clone(),
                    span: sp,
                };
                (deref, (**inner).clone())
            }
            // `own &T` is a non-null pointer to a heap T.  Borrowing the binding
            // borrows the pointee, yielding `&T`/`&mut T` whose runtime value is
            // the owner's pointer itself (emitted as `&(*owner)` == owner), not
            // the address of the local that holds it.  This makes heap-owned
            // values usable with ordinary `&T`/`&mut T` functions, exactly like
            // stack values - there is no distinct "borrow of a heap thing".
            HType::Heap { inner } if !matches!(**inner, HType::Dyn { .. } | HType::Slice { .. }) => {
                let deref = HExpr {
                    kind: HExprKind::DerefRef(Box::new(h.clone())),
                    ty: (**inner).clone(),
                    span: sp,
                };
                (deref, (**inner).clone())
            }
            _ => (h.clone(), h.ty.clone()),
        };
        // If expected is a pointer, produce a pointer.
        let result = match expected {
            Some(HType::Ptr { mutable, .. }) => {
                let mutable = want_mut && *mutable;
                HType::Ptr { mutable, inner: Box::new(inner) }
            }
            _ => HType::Ref { mutable: want_mut, inner: Box::new(inner) },
        };
        HExpr { kind: HExprKind::AddrOfRef { mutable: want_mut, place: Box::new(place_h) }, ty: result, span: sp }
    }

    fn is_place_addr_mut(&self, e: &HExpr) -> bool {
        match &e.kind {
            HExprKind::Local(id) => {
                let li = self.local(*id);
                // The storage itself is mutable iff the binding's payload is mut and the
                // binding actually owns/references mutable data.
                match &li.ty {
                    HType::Ref { mutable, .. } => *mutable,
                    HType::Ptr { mutable, .. } => *mutable,
                    _ => li.mut_payload,
                }
            }
            HExprKind::Unwrap { expr, .. } => matches!(expr.ty, HType::Ptr { mutable: true, .. }),
            HExprKind::Field { base, field } => {
                let base_mut = self.is_place_addr_mut(base);
                let sid = match struct_id_of(&base.ty) {
                    Some(id) => id,
                    None => return false,
                };
                let f = &self.sym.struct_info(sid).fields[*field];
                base_mut && f.mut_payload
            }
            HExprKind::Index { base, .. } => self.is_place_addr_mut(base),
            _ => false,
        }
    }

    fn check_field(&mut self, base: &ast::Expr, name: &str, expected: Option<&HType>, sp: Span) -> HExpr {
        // Special-case: `EnumName.Variant`.  If the context expects a specific
        // monomorphized enum (e.g. `Option<int>`), produce THAT EnumId instead of
        // the template's — `Option.None` typed as `Option<int>` must yield
        // `enum#OptionInt`, not the generic template.
        if let ast::Expr::Ident(n, _) = base {
            if let Some((tmpl_eid, info)) = self.sym.enum_by_name(n) {
                if let Some(vi) = info.variant_index(name) {
                    let (eid, _info_inst) = if let Some(HType::Enum(want_eid)) = expected {
                        let want_info = self.sym.enum_info(*want_eid);
                        let template_name = want_info.name.split("__").next().unwrap_or(&want_info.name);
                        if template_name == n { (*want_eid, want_info.clone()) }
                        else { (tmpl_eid, info.clone()) }
                    } else { (tmpl_eid, info.clone()) };
                    return HExpr { kind: HExprKind::EnumVariant(eid, vi), ty: HType::Enum(eid), span: sp };
                } else {
                    self.err(format!("enum `{}` has no variant `{}`", n, name), sp);
                    return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
                }
            }
        }
        let bh = self.check_expr(base, None);

        // Built-in `.tag` on an enum value - returns the discriminant as `int`.
        // Works on both simple and tagged enums; simple ones lower to identity,
        // tagged ones to a `.tag` field read.
        if name == "tag" {
            if matches!(&bh.ty, HType::Enum(_)) {
                return HExpr {
                    kind: HExprKind::EnumTag(Box::new(bh)),
                    ty: HType::Int,
                    span: sp,
                };
            }
        }

        // Built-in `.len` on slice / fixed-array / vector — lowers to the HIR's
        // SliceLen node so codegen emits the appropriate length expression.
        if name == "len" {
            let underlying = match &bh.ty {
                HType::Ref { inner, .. } => inner.as_ref(),
                HType::Heap { inner } => inner.as_ref(),
                other => other,
            };
            if matches!(underlying, HType::Slice { .. } | HType::Array { .. } | HType::Vec { .. }) {
                return HExpr {
                    kind: HExprKind::SliceLen(Box::new(bh)),
                    ty: HType::SizedInt { signed: false, bits: 0 },
                    span: sp,
                };
            }
        }

        // Auto-deref: if base is a pointer, we need explicit unwrap. We do NOT auto-deref pointers.
        // But for references and heap bindings, field access goes through transparently.
        // For Ref<Struct>, we treat it as struct access (codegen will deref).
        let struct_id = match &bh.ty {
            HType::Struct(id) => *id,
            HType::Ref { inner, .. } => match inner.as_ref() {
                HType::Struct(id) => *id,
                _ => { self.err("field access on non-struct reference", sp); return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp }; }
            },
            HType::Heap { inner } => match inner.as_ref() {
                HType::Struct(id) => *id,
                _ => { self.err("field access on a non-struct `own &` value", sp); return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp }; }
            },
            HType::Ptr { .. } => {
                self.err("dereference a `*T` with `!` before accessing fields", sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
            HType::RawPtr { .. } => {
                self.err("dereference a `raw *T` with `!` (inside `unsafe { ... }`) before accessing fields", sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
            HType::OwnPtr { .. } => {
                self.err("dereference an `own *T` with `!` before accessing fields", sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
            _ => {
                self.err("field access on non-struct value", sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
        };
        // Keep bh.ty as-is so codegen can choose `.` vs `->`.
        // Direct lookup first.
        let info = self.sym.struct_info(struct_id);
        if let Some((idx, f)) = info.fields.iter().enumerate().find(|(_, f)| f.name == name) {
            let ty = f.ty.clone();
            return HExpr { kind: HExprKind::Field { base: Box::new(bh), field: idx }, ty, span: sp };
        }
        // Promoted lookup: search any embedded field whose embedded type contains `name`.
        match self.find_promoted_field_count(struct_id, name) {
            Ok(Some((path, ty))) => {
                let mut cur = bh;
                for idx in path {
                    let info = match struct_id_of(&cur.ty) {
                        Some(id) => self.sym.struct_info(id),
                        None => break,
                    };
                    let f_ty = info.fields[idx].ty.clone();
                    cur = HExpr { kind: HExprKind::Field { base: Box::new(cur), field: idx }, ty: f_ty, span: sp };
                }
                cur.ty = ty;
                return cur;
            }
            Err(n) => {
                self.err(
                    format!(
                        "field `{}` is ambiguous in `{}` — {} embedded paths reach it; qualify with the embed name (e.g. `value.embed_field.{}`)",
                        name, info.name, n, name,
                    ),
                    sp,
                );
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
            Ok(None) => {}
        }
        self.err(format!("struct `{}` has no field `{}`", info.name, name), sp);
        HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp }
    }

    /// Find a promoted field chain for `name` starting from `start`.
    /// Returns a path of field indices to traverse and the final HType.
    /// If the name is reachable through more than one distinct embed chain,
    /// returns `Err(N)` where N is the number of distinct chains found — the
    /// caller raises an ambiguity diagnostic instead of silently picking one.
    fn find_promoted_field(&self, start: StructId, name: &str) -> Option<(Vec<usize>, HType)> {
        match self.find_promoted_field_count(start, name) {
            Ok(hit) => hit,
            Err(_) => None,   // ambiguous → caller falls through to the "no field" error
        }
    }

    /// Like `find_promoted_field` but counts every matching chain so ambiguity
    /// can be reported.  Returns `Ok(Some(...))` for a single hit, `Ok(None)`
    /// for no hits, and `Err(n)` for n>1 hits.
    fn find_promoted_field_count(&self, start: StructId, name: &str) -> Result<Option<(Vec<usize>, HType)>, usize> {
        let info = self.sym.struct_info(start);
        let mut hits: Vec<(Vec<usize>, HType)> = Vec::new();
        for (i, f) in info.fields.iter().enumerate() {
            if !f.is_embed { continue; }
            let Some(sid) = struct_id_of(&f.ty) else { continue; };
            let sub = self.sym.struct_info(sid);
            if let Some((j, fj)) = sub.fields.iter().enumerate().find(|(_, x)| x.name == name) {
                hits.push((vec![i, j], fj.ty.clone()));
                continue;
            }
            match self.find_promoted_field_count(sid, name) {
                Ok(Some((mut path, ty))) => {
                    let mut full = vec![i];
                    full.append(&mut path);
                    hits.push((full, ty));
                }
                Ok(None) => {}
                Err(n) => return Err(n + hits.len()),
            }
        }
        match hits.len() {
            0 => Ok(None),
            1 => Ok(Some(hits.remove(0))),
            n => Err(n),
        }
    }

    /// Build an HExpr that walks an embed-field chain from `base` through `path`.
    /// Each index in `path` is a field index in the current struct; the result
    /// has the type of the innermost field.
    fn drill_embed_path(&self, base: HExpr, path: &[usize]) -> HExpr {
        let span = base.span;
        let mut cur = base;
        for &idx in path {
            let sid = match struct_id_of(&cur.ty) {
                Some(id) => id,
                None => break,
            };
            let f_ty = self.sym.struct_info(sid).fields[idx].ty.clone();
            cur = HExpr {
                kind: HExprKind::Field { base: Box::new(cur), field: idx },
                ty: f_ty,
                span,
            };
        }
        cur
    }

    /// Find a chain of embed-field indices that drills from `start` down to a
    /// nested struct of id `target`.  Returns the chain (possibly empty if
    /// start == target) or `None` if no embed path exists.
    fn find_embed_path(&self, start: StructId, target: StructId) -> Option<Vec<usize>> {
        if start == target { return Some(Vec::new()); }
        let info = self.sym.struct_info(start);
        for (i, f) in info.fields.iter().enumerate() {
            if !f.is_embed { continue; }
            let Some(sid) = struct_id_of(&f.ty) else { continue; };
            if let Some(mut rest) = self.find_embed_path(sid, target) {
                let mut full = vec![i];
                full.append(&mut rest);
                return Some(full);
            }
        }
        None
    }

    fn check_index(&mut self, base: &ast::Expr, idx: &ast::Expr, sp: Span) -> HExpr {
        let bh = self.check_expr(base, None);
        // Don't pre-coerce the index — it might be a different type for an overloaded Index.
        let ih_probe = self.check_expr(idx, None);
        let elem_ty = match &bh.ty {
            HType::Array { elem, .. } => (**elem).clone(),
            HType::Slice { elem, .. } => (**elem).clone(),
            HType::Vec { elem } => (**elem).clone(),
            HType::Heap { inner } => match inner.as_ref() {
                HType::Array { elem, .. } | HType::Vec { elem } => (**elem).clone(),
                HType::Ptr { inner, .. } => match inner.as_ref() {
                    HType::Vec { elem } | HType::Array { elem, .. } => (**elem).clone(),
                    _ => return self.try_index_overload_or_err(bh, ih_probe, sp),
                },
                _ => return self.try_index_overload_or_err(bh, ih_probe, sp),
            },
            HType::Ptr { inner, .. } => match inner.as_ref() {
                HType::Vec { elem } | HType::Array { elem, .. } | HType::Slice { elem, .. } => (**elem).clone(),
                _ => return self.try_index_overload_or_err(bh, ih_probe, sp),
            },
            // A borrow of an array/vector/slice (`&[N]T`, `&[*]T`, `&[]T`) indexes
            // through to the element - e.g. a non-owning pointer to a stack array.
            HType::Ref { inner, .. } => match inner.as_ref() {
                HType::Vec { elem } | HType::Array { elem, .. } | HType::Slice { elem, .. } => (**elem).clone(),
                _ => return self.try_index_overload_or_err(bh, ih_probe, sp),
            },
            _ => return self.try_index_overload_or_err(bh, ih_probe, sp),
        };
        // The index may be `int` or `usize` (the latter is what `.len` yields,
        // so `arr[i]` in a `0..arr.len` loop type-checks); codegen casts it to
        // maka_int either way.  Any other type coerces to int (or errors).
        let ih = if matches!(&ih_probe.ty, HType::SizedInt { signed: false, .. }) {
            ih_probe // usize / u8..u64: codegen casts to maka_int
        } else {
            self.coerce(ih_probe, &HType::Int)
        };
        HExpr { kind: HExprKind::Index { base: Box::new(bh), idx: Box::new(ih) }, ty: elem_ty, span: sp }
    }

    fn try_index_overload_or_err(&mut self, bh: HExpr, ih: HExpr, sp: Span) -> HExpr {
        if let Some(info) = self.sym.logic_by_name("Index").cloned() {
            let cand = info.funcs.iter().find_map(|fid| {
                let sig = self.sym.func_sig(*fid);
                if sig.name == "index" && sig.param_tys.len() == 2 {
                    if param_compatible(&sig.param_tys[0], &bh.ty, &sig.type_params)
                        && param_compatible(&sig.param_tys[1], &ih.ty, &sig.type_params) {
                        return Some((*fid, sig.clone()));
                    }
                }
                None
            });
            if let Some((fid, sig)) = cand {
                let lh = self.coerce(bh, &sig.param_tys[0]);
                let rh = self.coerce(ih, &sig.param_tys[1]);
                let ret = sig.ret.clone();
                return HExpr {
                    kind: HExprKind::Call { callee: fid, args: vec![lh, rh] },
                    ty: ret,
                    span: sp,
                };
            }
        }
        self.err("indexing on non-array/slice/vector (no `Index.index` overload found)", sp);
        HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp }
    }

    fn check_call(&mut self, callee: &ast::Expr, args: &[ast::Expr], sp: Span) -> HExpr {
        self.call_arg_depth += 1;
        // Pre-resolve: if the callee names a function declared `gate`, enable `transfer`/`share`
        // for this argument list only.
        let saved_gate = self.cur_call_is_gate;
        self.cur_call_is_gate = self.callee_is_gate(callee);
        let result = self.check_call_inner(callee, args, sp);
        self.cur_call_is_gate = saved_gate;
        self.call_arg_depth -= 1;
        result
    }

    /// §10.5 attr-qualified call: `Attr::method(args)` or
    /// `receiver.Attr::method(args)`.  Synthesizes a Field callee shaped
    /// like the legacy `Attr.method` form and routes through `check_call`,
    /// but unlike a bare `Attr.method` it bypasses the local-shadow check
    /// inside `check_call_inner` (the `::` parse guaranteed the user meant
    /// the attr).  Postfix receiver becomes arg 0 with auto-borrow.
    fn check_attr_call(
        &mut self,
        attr: &str,
        name: &str,
        receiver: Option<&ast::Expr>,
        args: &[ast::Expr],
        sp: Span,
    ) -> HExpr {
        if self.sym.attr_by_name(attr).is_none() && self.sym.logic_by_name(attr).is_none() {
            self.err(format!("unknown attr `{}` in qualified call `{}::{}`", attr, attr, name), sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        }
        let mut all_args: Vec<ast::Expr> = Vec::new();
        if let Some(recv) = receiver { all_args.push(recv.clone()); }
        all_args.extend(args.iter().cloned());
        // Synthesise the existing qualified-call shape so the rest of
        // dispatch picks up unchanged.  The local-shadow check in
        // `check_call_inner` only fires when the bare-Ident lookup
        // succeeds; here we pre-tag this as a qualified call by setting
        // `force_qualifier`, which suppresses the shadow lookup.
        let callee = ast::Expr::Field {
            base: Box::new(ast::Expr::Ident(attr.to_string(), sp)),
            name: name.to_string(),
            span: sp,
        };
        self.force_qualifier = Some(attr.to_string());
        self.force_postfix = receiver.is_some();
        let r = self.check_call(&callee, &all_args, sp);
        self.force_qualifier = None;
        self.force_postfix = false;
        r
    }

    /// Is `n` the last segment of any module declared in this build?  Used to
    /// distinguish module-qualified calls (`mod.func()`) from postfix method calls.
    fn is_module_name(&self, n: &str) -> bool {
        self.sym.sigs.iter().any(|s| s.module_path.last().map(|p| p == n).unwrap_or(false))
            || self.sym.structs.iter().any(|s| s.module_path.last().map(|p| p == n).unwrap_or(false))
            || self.sym.enums.iter().any(|e| e.module_path.last().map(|p| p == n).unwrap_or(false))
    }

    fn callee_is_gate(&self, callee: &ast::Expr) -> bool {
        let (name, qualifier) = match callee {
            ast::Expr::Ident(n, _) => (n.clone(), None),
            ast::Expr::Field { base, name, .. } => {
                if let ast::Expr::Ident(logic_name, _) = base.as_ref() {
                    if self.sym.logic_by_name(logic_name).is_some() {
                        (name.clone(), Some(logic_name.clone()))
                    } else { return false; }
                } else { return false; }
            }
            _ => return false,
        };
        self.sym.funcs_by_qualified(qualifier.as_deref(), &name)
            .iter().any(|(_, sig)| sig.is_gate)
    }

    fn check_call_inner(&mut self, callee: &ast::Expr, args: &[ast::Expr], sp: Span) -> HExpr {
        // Capture (and clear, so nested arg calls don't inherit it) the expected
        // return type for return-position generic inference below.
        let ret_expected = self.call_ret_expected.take();
        // Indirect call: `f(args)` where `f` is a local of FnPtr type, or a
        // pointer/heap to a FnPtr (a heap-allocated / escaped closure, whose type
        // is `own *T(..)` / `own &T(..)`).
        if let ast::Expr::Ident(n, _) = callee {
            if let Some(id) = self.lookup(n) {
                let lty = self.local(id).ty.clone();
                let (fnptr_ty, via_deref) = match &lty {
                    HType::FnPtr { .. } => (Some(lty.clone()), false),
                    HType::OwnPtr { inner, .. } | HType::Ptr { inner, .. } | HType::Heap { inner }
                        if matches!(inner.as_ref(), HType::FnPtr { .. }) => (Some((**inner).clone()), true),
                    _ => (None, false),
                };
                if let Some(HType::FnPtr { ret, params }) = fnptr_ty {
                    let ret_ty = (*ret).clone();
                    let hargs: Vec<HExpr> = args.iter().enumerate().map(|(i, a)| {
                        let want = params.get(i).cloned().unwrap_or(HType::Int);
                        self.check_expr_coerce(a, &want)
                    }).collect();
                    let fnptr = HType::FnPtr { ret: Box::new(ret_ty.clone()), params };
                    let callee_h = if via_deref {
                        let local_h = HExpr { kind: HExprKind::Local(id), ty: lty.clone(), span: sp };
                        HExpr { kind: HExprKind::DerefRef(Box::new(local_h)), ty: fnptr, span: sp }
                    } else {
                        HExpr { kind: HExprKind::Local(id), ty: fnptr, span: sp }
                    };
                    return HExpr {
                        kind: HExprKind::CallIndirect { callee: Box::new(callee_h), args: hargs },
                        ty: ret_ty,
                        span: sp,
                    };
                }
            }
        }
        // Indirect call through a fn-pointer / closure stored in a struct field:
        // `recv.field(args)` where `recv` is an in-scope value and `field` has a
        // FnPtr type.  Without this, `recv.field(...)` is misread as a postfix
        // method call `field(recv, ...)` ("unknown function field").
        if let ast::Expr::Field { base, name: fname, span: fsp } = callee {
            let base_ty = match base.as_ref() {
                ast::Expr::Ident(bn, _) => self.lookup(bn).map(|id| self.local(id).ty.clone()),
                _ => None,
            };
            if let Some(bty) = base_ty {
                fn peel_struct(t: &HType) -> Option<StructId> {
                    match t {
                        HType::Struct(id) => Some(*id),
                        HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. }
                        | HType::OwnPtr { inner, .. } | HType::Heap { inner } => peel_struct(inner),
                        _ => None,
                    }
                }
                if let Some(sid) = peel_struct(&bty) {
                    let field = self.sym.struct_info(sid).fields.iter().enumerate()
                        .find(|(_, f)| &f.name == fname)
                        .map(|(i, f)| (i, f.ty.clone()));
                    if let Some((fidx, HType::FnPtr { ret, params })) = field {
                        let base_h = self.check_expr(base, None);
                        let ret_ty = (*ret).clone();
                        let hargs: Vec<HExpr> = args.iter().enumerate().map(|(i, a)| {
                            let want = params.get(i).cloned().unwrap_or(HType::Int);
                            self.check_expr_coerce(a, &want)
                        }).collect();
                        let field_ty = HType::FnPtr { ret: Box::new(ret_ty.clone()), params };
                        let callee_h = HExpr {
                            kind: HExprKind::Field { base: Box::new(base_h), field: fidx },
                            ty: field_ty, span: *fsp,
                        };
                        return HExpr {
                            kind: HExprKind::CallIndirect { callee: Box::new(callee_h), args: hargs },
                            ty: ret_ty, span: sp,
                        };
                    }
                }
            }
        }
        // Decide the call kind based on the callee shape.
        // 1. `name(args)`                  — top-level or in-scope call
        // 2. `Logic.fn(args)`              — qualified call to a logic / attr block
        // 3. `module.fn(args)`             — module-qualified call (NEW)
        // 4. `receiver.fn(args)`           — postfix call: rewrite as `fn(receiver, args)`
        let mut module_qualifier: Option<String> = None;
        let forced_qual = self.force_qualifier.take();
        let forced_postfix = std::mem::take(&mut self.force_postfix);
        let (name, qualifier, postfix_receiver) = match callee {
            ast::Expr::Ident(n, _) => (n.clone(), None, None),
            ast::Expr::Field { base, name, .. } => {
                // §10.5: `Attr::method` (and `recv.Attr::method`) skip the
                // local-shadow check — the `::` form unambiguously names a
                // qualified attr method.  In the postfix case, the receiver
                // was already prepended to the args list by `check_attr_call`,
                // and we set `forced_postfix` so auto-borrow can fire.
                if let Some(q) = forced_qual.clone() {
                    if forced_postfix {
                        // Receiver lives at args[0]; treat as postfix without
                        // a separate `postfix_receiver` slot — the args list
                        // already carries it.
                        (name.clone(), Some(q), None)
                    } else {
                        (name.clone(), Some(q), None)
                    }
                } else if let ast::Expr::Ident(qual, _) = base.as_ref() {
                    // Locals shadow `logic`/`attr` names: when `qual` resolves
                    // to a value in scope, `qual.method(args)` is a postfix
                    // call on the instance, NOT an attr-qualified call.  This
                    // matches Maka's normal scoping rule and removes the
                    // ambiguity between `LogicName.fn(args)` (legacy qualified
                    // call) and `instance.fn(args)` when `instance` happens to
                    // share its name with a registered attr/logic.  Users who
                    // need the unambiguous qualified form can write
                    // `Attr::fn(args)` (§10.5).
                    if let Some(id) = self.lookup(qual) {
                        // Local variable as receiver: postfix call.  When the
                        // receiver is a `Rust<T>` opaque, the bridge emits its
                        // methods as free functions named `T_<method>` — look
                        // there first so `rng.pick(...)` dispatches without
                        // requiring the user to spell out `Rng_pick(rng, ...)`.
                        let local_ty = self.local(id).ty.clone();
                        if let HType::RustOpaque(label) = &local_ty {
                            let mangled = format!("{}_{}", label, name);
                            if self.sym.func_by_name(&mangled).is_some() {
                                (mangled, None, Some((**base).clone()))
                            } else {
                                (name.clone(), None, Some((**base).clone()))
                            }
                        } else {
                            (name.clone(), None, Some((**base).clone()))
                        }
                    } else if self.sym.logic_by_name(qual).is_some() || self.sym.attr_by_name(qual).is_some() {
                        (name.clone(), Some(qual.clone()), None)
                    } else if self.is_module_name(qual) {
                        // Module-qualified call: filter candidates by module path.
                        module_qualifier = Some(qual.clone());
                        (name.clone(), None, None)
                    } else {
                        // Unknown identifier — treat as postfix; if it's truly
                        // unresolved, downstream dispatch reports the error.
                        (name.clone(), None, Some((**base).clone()))
                    }
                } else {
                    // Receiver is some non-Ident expression (e.g., field access chain).
                    (name.clone(), None, Some((**base).clone()))
                }
            }
            _ => {
                // General indirect call: any callee expression whose type is a
                // FnPtr (or a pointer/heap to one) - e.g. `fs[i](x)` calling a
                // closure stored in a Vec/array element.
                let callee_h = self.check_expr(callee, None);
                let (fnptr_ty, via_deref) = match &callee_h.ty {
                    HType::FnPtr { .. } => (Some(callee_h.ty.clone()), false),
                    HType::OwnPtr { inner, .. } | HType::Ptr { inner, .. } | HType::Heap { inner }
                        if matches!(inner.as_ref(), HType::FnPtr { .. }) => (Some((**inner).clone()), true),
                    _ => (None, false),
                };
                if let Some(HType::FnPtr { ret, params }) = fnptr_ty {
                    let ret_ty = (*ret).clone();
                    let hargs: Vec<HExpr> = args.iter().enumerate().map(|(i, a)| {
                        let want = params.get(i).cloned().unwrap_or(HType::Int);
                        self.check_expr_coerce(a, &want)
                    }).collect();
                    let callee_final = if via_deref {
                        let fnptr = HType::FnPtr { ret: Box::new(ret_ty.clone()), params };
                        HExpr { kind: HExprKind::DerefRef(Box::new(callee_h)), ty: fnptr, span: sp }
                    } else {
                        callee_h
                    };
                    return HExpr {
                        kind: HExprKind::CallIndirect { callee: Box::new(callee_final), args: hargs },
                        ty: ret_ty,
                        span: sp,
                    };
                }
                self.err("only direct function calls are supported", sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
        };

        // If this is a postfix call, prepend the receiver to args and treat as a normal call by name.
        // For `recv.Attr::method(args)` the receiver was already prepended by
        // `check_attr_call` and `forced_postfix` flags this so auto-borrow can fire.
        let (args_owned, is_postfix): (Vec<ast::Expr>, bool) = if let Some(recv) = postfix_receiver {
            let mut v = vec![recv];
            v.extend(args.iter().cloned());
            (v, true)
        } else if forced_postfix {
            (args.iter().cloned().collect(), true)
        } else {
            (args.iter().cloned().collect(), false)
        };
        let args: &[ast::Expr] = &args_owned;

        // Built-in `panic(msg)` aborts the program.
        if name == "panic" && qualifier.is_none() {
            let mut hargs = Vec::new();
            for a in args { hargs.push(self.check_expr(a, None)); }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 2), args: hargs },
                ty: HType::Unit,
                span: sp,
            };
        }
        // Built-in `free` — manual deallocation for non-owning `*T`.  Owning types
        // (`own *T`, `own &T`) are auto-freed at scope exit; calling free() on them
        // would double-free.
        // Built-in `spawn(closure)` / `thread(closure)` / `job(closure)` — three
        // concurrency tiers (fiber / OS-thread / work-item respectively).  All
        // three currently lower to `pthread_create`; the real fiber and job
        // runtimes will replace the backings without changing the surface.
        // See CONCURRENCY.md for the full spec.
        //
        // The closure must be a `unit()` callable; captures need `alloc` so the
        // env lives on the heap (the lambda-escape rule applies to all three).
        if (name == "spawn" || name == "thread" || name == "job" || name == "spawn_pool") && qualifier.is_none() {
            let mut hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 1 {
                self.err(format!("{} expects exactly one closure argument", name), sp);
            } else {
                let arg = &hargs[0];
                let inner = match &arg.ty {
                    HType::FnPtr { .. } => Some(&arg.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok = matches!(inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Unit) && params.is_empty());
                if !ok {
                    self.err(format!("{} expects a `unit()` closure, got `{}`", name, type_str(&arg.ty)), sp);
                }
                // Closure captures of `Rust<T>` cross the concurrency boundary —
                // record `T` for a `Send` probe in the sidecar.
                self.collect_send_from_closure(arg);
                // For cross-thread tiers, reject captures whose type can't
                // safely cross threads — borrowed references are tied to a
                // scope on the spawning thread.  Fiber-tier `spawn` runs on
                // the same thread, so refs are fine there.
                let cross_thread = matches!(name.as_str(), "thread" | "job" | "spawn_pool");
                if cross_thread {
                    self.check_cross_thread_captures(name.as_str(), arg, sp);
                }
            }
            // Thread handle type: *Thread (lookup the builtin struct).
            let thread_id = self.sym.struct_by_name("Thread").map(|(id, _)| id).expect("Thread struct registered");
            let ret_ty = HType::Ptr { mutable: true, inner: Box::new(HType::Struct(thread_id)) };
            // Pick the right runtime entry by tier — codegen recognizes the
            // FuncIds and emits __maka_spawn_thread / _fiber / _job.
            let fid = match name.as_str() {
                "thread"     => FuncId(u32::MAX - 15),
                "job"        => FuncId(u32::MAX - 16),
                "spawn_pool" => FuncId(u32::MAX - 37),
                _            => FuncId(u32::MAX - 3),    // spawn (fiber) — keeps legacy id
            };
            return HExpr {
                kind: HExprKind::Call { callee: fid, args: hargs },
                ty: ret_ty,
                span: sp,
            };
        }
        // Built-in `join(*Thread)` — blocks until that handle finishes.
        // Also accepts `join(&[]*Thread)` or `join([]*Thread)` to wait for an
        // entire slice of handles (homogeneous, all the same backing tier).
        // `select(&[]*Thread)` is the race variant — first handle to finish
        // wins; the rest are cancelled.
        if (name == "join" || name == "select") && qualifier.is_none() {
            let mut hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 1 {
                self.err(format!("{} expects exactly one argument", name), sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
            // Sniff the arg's type to choose the variant.
            let arg_ty = hargs[0].ty.clone();
            let is_thread_ptr = matches!(&arg_ty,
                HType::Ptr { inner, .. } if matches!(**inner, HType::Struct(id) if {
                    let info = self.sym.struct_info(id);
                    info.name == "Thread"
                }));
            let is_thread_slice = matches!(&arg_ty,
                HType::Slice { elem, .. } if matches!(
                    elem.as_ref(),
                    HType::Ptr { inner, .. } if matches!(**inner, HType::Struct(id) if {
                        let info = self.sym.struct_info(id);
                        info.name == "Thread"
                    })
                ));
            let is_thread_slice_ref = matches!(&arg_ty,
                HType::Ref { inner, .. } if matches!(
                    inner.as_ref(),
                    HType::Slice { elem, .. } if matches!(
                        elem.as_ref(),
                        HType::Ptr { inner: i2, .. } if matches!(**i2, HType::Struct(id) if {
                            let info = self.sym.struct_info(id);
                            info.name == "Thread"
                        })
                    )
                ));

            if name == "join" && is_thread_ptr {
                // Single-handle join — existing path.
                return HExpr {
                    kind: HExprKind::Call { callee: FuncId(u32::MAX - 4), args: hargs },
                    ty: HType::Unit,
                    span: sp,
                };
            }
            if (is_thread_slice || is_thread_slice_ref) && (name == "join" || name == "select") {
                // Slice path: codegen recognises the FuncId and emits a call
                // to the runtime's __maka_join_all_i64 / __maka_select_first_i64.
                let fid = if name == "join" {
                    FuncId(u32::MAX - 17)   // join_all
                } else {
                    FuncId(u32::MAX - 18)   // select_first
                };
                return HExpr {
                    kind: HExprKind::Call { callee: fid, args: hargs },
                    ty: HType::Unit,    // result-value capture deferred (closures return unit today)
                    span: sp,
                };
            }
            self.err(
                format!(
                    "{} expects `*Thread`, `[]*Thread`, or `&[]*Thread`; got `{}`",
                    name,
                    type_str(&arg_ty)
                ),
                sp,
            );
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        }
        // ===================================================================
        // Concurrency primitives (the "irreducible base").
        //
        // These are the operations Maka can't express in pure Maka — atomic
        // memory ops, thread blocking/waking, memory fences, syscalls.  The
        // stdlib's `Atomic`, `Mutex`, `RwLock`, `WaitGroup`, `Once`, the
        // `*Chan` family, and friends are written in pure Maka source on
        // top of these.
        //
        // The CAS primitive alone is enough to derive `atomic_load` / `store`
        // / `fetch_add` / `fetch_sub` / `fetch_and` / `fetch_or` / `fetch_xor`
        // via CAS-loops — but each of those is provided as a direct builtin
        // for performance (on x86 they collapse to a single instruction;
        // CAS-looped equivalents waste ~30 cycles per call).
        // ===================================================================
        if (name == "atomic_cas"
            || name == "atomic_load"
            || name == "atomic_store"
            || name == "atomic_fetch_add"
            || name == "atomic_fetch_sub"
            || name == "atomic_fetch_and"
            || name == "atomic_fetch_or"
            || name == "atomic_fetch_xor"
            || name == "atomic_fence"
            || name == "futex_wait"
            || name == "futex_wake"
            || name == "thread_yield"
            || name == "syscall") && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            // Atomic-compatible scalar: any word-sized type the hardware can
            // load/store atomically.  Ints (native + sized), bool (lowers to
            // u8), and pointers all qualify.  References don't — they carry
            // a borrow lifetime that can't safely cross an atomic.
            let atomic_t = |t: &HType| matches!(
                t,
                HType::Int
                | HType::SizedInt { .. }
                | HType::Bool
                | HType::Ptr { .. }
                | HType::RawPtr { .. }
            );
            // Helper: pull the inner scalar T out of a `&T` / `&mut T` / `&const T`.
            let inner_int = |e: &HExpr| -> Option<HType> {
                match &e.ty {
                    HType::Ref { inner, .. } if atomic_t(inner) => Some((**inner).clone()),
                    _ => None,
                }
            };
            let int_t = atomic_t;

            match name.as_str() {
                "atomic_cas" => {
                    // atomic_cas(&mut T p, T expected, T new) -> T  (returns old value).
                    if hargs.len() != 3 {
                        self.err("`atomic_cas` expects 3 args: `&mut T`, `T`, `T`", sp);
                    }
                    let t = hargs.first().and_then(inner_int).unwrap_or(HType::Int);
                    if !int_t(&t) {
                        self.err(format!("`atomic_cas`: first arg must be `&mut T` where T is an integer; got `{}`", hargs.first().map(|h| type_str(&h.ty)).unwrap_or_default()), sp);
                    }
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 45), args: hargs },
                        ty: t,
                        span: sp,
                    };
                }
                "atomic_load" => {
                    // atomic_load(&const T p) -> T
                    if hargs.len() != 1 {
                        self.err("`atomic_load` expects 1 arg: `&const T`", sp);
                    }
                    let t = hargs.first().and_then(inner_int).unwrap_or(HType::Int);
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 46), args: hargs },
                        ty: t,
                        span: sp,
                    };
                }
                "atomic_store" => {
                    // atomic_store(&mut T p, T v)
                    if hargs.len() != 2 {
                        self.err("`atomic_store` expects 2 args: `&mut T`, `T`", sp);
                    }
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 47), args: hargs },
                        ty: HType::Unit,
                        span: sp,
                    };
                }
                "atomic_fetch_add" | "atomic_fetch_sub" | "atomic_fetch_and"
                | "atomic_fetch_or" | "atomic_fetch_xor" => {
                    // atomic_fetch_*(&mut T p, T delta) -> T  (returns old value)
                    if hargs.len() != 2 {
                        self.err(format!("`{}` expects 2 args: `&mut T`, `T`", name), sp);
                    }
                    let t = hargs.first().and_then(inner_int).unwrap_or(HType::Int);
                    let fid = match name.as_str() {
                        "atomic_fetch_add" => u32::MAX - 48,
                        "atomic_fetch_sub" => u32::MAX - 49,
                        "atomic_fetch_and" => u32::MAX - 50,
                        "atomic_fetch_or"  => u32::MAX - 51,
                        _                  => u32::MAX - 52,    // xor
                    };
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(fid), args: hargs },
                        ty: t,
                        span: sp,
                    };
                }
                "atomic_fence" => {
                    // atomic_fence(int order).  Order: 1=acquire, 2=release,
                    // 3=acq_rel, 4=seq_cst (matches C11 __ATOMIC_* enum).
                    if hargs.len() != 1 {
                        self.err("`atomic_fence` expects 1 arg: `int` (memory order)", sp);
                    }
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 53), args: hargs },
                        ty: HType::Unit,
                        span: sp,
                    };
                }
                "futex_wait" => {
                    // futex_wait(&const int addr, int expected) -> int
                    if hargs.len() != 2 {
                        self.err("`futex_wait` expects 2 args: `&const int`, `int`", sp);
                    }
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 54), args: hargs },
                        ty: HType::Int,
                        span: sp,
                    };
                }
                "futex_wake" => {
                    // futex_wake(&const int addr, int n) -> int
                    if hargs.len() != 2 {
                        self.err("`futex_wake` expects 2 args: `&const int`, `int`", sp);
                    }
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 55), args: hargs },
                        ty: HType::Int,
                        span: sp,
                    };
                }
                "thread_yield" => {
                    // thread_yield() — sched_yield equivalent.
                    if !hargs.is_empty() {
                        self.err("`thread_yield` takes no arguments", sp);
                    }
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 56), args: hargs },
                        ty: HType::Unit,
                        span: sp,
                    };
                }
                "syscall" => {
                    // syscall(int n, int a1..a6) -> int.  All args are int;
                    // missing args codegen to 0.
                    if hargs.is_empty() || hargs.len() > 7 {
                        self.err("`syscall` expects 1..7 args (syscall number + up to 6 int args)", sp);
                    }
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 57), args: hargs },
                        ty: HType::Int,
                        span: sp,
                    };
                }
                _ => unreachable!(),
            }
        }
        // Built-in `detach(*Thread)` — caller opts out of join; runtime auto-
        // reaps the handle when the fiber/thread/job completes.
        if name == "detach" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 1 {
                self.err("detach expects exactly one `*Thread` argument", sp);
            } else {
                let is_thread_ptr = matches!(&hargs[0].ty,
                    HType::Ptr { inner, .. } if matches!(**inner, HType::Struct(id) if {
                        let info = self.sym.struct_info(id);
                        info.name == "Thread"
                    }));
                if !is_thread_ptr {
                    self.err(format!("detach: expected `*Thread`, got `{}`", type_str(&hargs[0].ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 33), args: hargs },
                ty: HType::Unit,
                span: sp,
            };
        }
        // Built-in `cancel(*Thread)` — user-callable cancellation.  No-op for
        // jobs (run to completion); pthread_cancel for threads; queue-walk
        // removal for fibers.  Frees the handle.
        if name == "cancel" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 1 {
                self.err("cancel expects exactly one `*Thread` argument", sp);
            } else {
                let is_thread_ptr = matches!(&hargs[0].ty,
                    HType::Ptr { inner, .. } if matches!(**inner, HType::Struct(id) if {
                        let info = self.sym.struct_info(id);
                        info.name == "Thread"
                    }));
                if !is_thread_ptr {
                    self.err(format!("cancel: expected `*Thread`, got `{}`", type_str(&hargs[0].ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 23), args: hargs },
                ty: HType::Unit,
                span: sp,
            };
        }
        // Built-in `try_join(*Thread) -> bool` — non-blocking poll.  Returns
        // true if the handle had finished (and reclaims it); false otherwise.
        if name == "try_join" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 1 {
                self.err("try_join expects exactly one `*Thread` argument", sp);
            } else {
                let is_thread_ptr = matches!(&hargs[0].ty,
                    HType::Ptr { inner, .. } if matches!(**inner, HType::Struct(id) if {
                        let info = self.sym.struct_info(id);
                        info.name == "Thread"
                    }));
                if !is_thread_ptr {
                    self.err(format!("try_join: expected `*Thread`, got `{}`", type_str(&hargs[0].ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 24), args: hargs },
                ty: HType::Bool,
                span: sp,
            };
        }
        // Built-in `join_timeout(*Thread, int ms) -> bool` — returns true if
        // the handle joined within the deadline, false if the deadline expired.
        if name == "join_timeout" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("join_timeout expects (`*Thread`, `int ms`)", sp);
            } else {
                let is_thread_ptr = matches!(&hargs[0].ty,
                    HType::Ptr { inner, .. } if matches!(**inner, HType::Struct(id) if {
                        let info = self.sym.struct_info(id);
                        info.name == "Thread"
                    }));
                if !is_thread_ptr {
                    self.err(format!("join_timeout: first arg must be `*Thread`, got `{}`", type_str(&hargs[0].ty)), sp);
                }
                if !matches!(&hargs[1].ty, HType::Int) {
                    self.err(format!("join_timeout: second arg must be `int` (milliseconds), got `{}`", type_str(&hargs[1].ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 25), args: hargs },
                ty: HType::Bool,
                span: sp,
            };
        }
        // Built-in `select_timeout(slice, int ms) -> int` — returns the index
        // of the first handle to finish, or -1 on timeout.  Losers are
        // cancelled exactly like plain `select`.
        if name == "select_timeout" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("select_timeout expects (`[]*Thread`, `int ms`)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(
                        elem.as_ref(),
                        HType::Ptr { inner, .. } if matches!(**inner, HType::Struct(id) if {
                            let info = self.sym.struct_info(id);
                            info.name == "Thread"
                        })
                    ));
                let is_slice_ref = matches!(&hargs[0].ty,
                    HType::Ref { inner, .. } if matches!(
                        inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(
                            elem.as_ref(),
                            HType::Ptr { inner: i2, .. } if matches!(**i2, HType::Struct(id) if {
                                let info = self.sym.struct_info(id);
                                info.name == "Thread"
                            })
                        )
                    ));
                if !is_slice && !is_slice_ref {
                    self.err(format!("select_timeout: first arg must be `[]*Thread` or `&[]*Thread`, got `{}`", type_str(&hargs[0].ty)), sp);
                }
                if !matches!(&hargs[1].ty, HType::Int) {
                    self.err(format!("select_timeout: second arg must be `int` (milliseconds), got `{}`", type_str(&hargs[1].ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 26), args: hargs },
                ty: HType::Int,
                span: sp,
            };
        }
        // Built-in `par_map_int` — two shapes:
        //   par_map_int(start, end, fn) -> []int          // integer range
        //   par_map_int(slice, fn)      -> []int          // slice form
        // Body in both cases is `int(int)`.  Chunks distributed across the job pool.
        if name == "par_map_int" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            let first_is_int_slice = hargs.first().is_some_and(|a| matches!(&a.ty,
                HType::Slice { elem, .. } if matches!(**elem, HType::Int)) ||
                matches!(&a.ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                    HType::Slice { elem, .. } if matches!(**elem, HType::Int))));
            if first_is_int_slice {
                if hargs.len() != 2 {
                    self.err("par_map_int(slice, body): expected 2 arguments", sp);
                } else {
                    let body = &hargs[1];
                    let body_inner = match &body.ty {
                        HType::FnPtr { .. } => Some(&body.ty),
                        HType::Heap { inner } => Some(inner.as_ref()),
                        HType::Ptr { inner, .. } => Some(inner.as_ref()),
                        HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                        _ => None,
                    };
                    let ok_body = matches!(body_inner,
                        Some(HType::FnPtr { ret, params })
                            if matches!(**ret, HType::Int) && params.len() == 1 && matches!(params[0], HType::Int));
                    if !ok_body { self.err(format!("par_map_int(slice) body must be `int(int)`, got `{}`", type_str(&body.ty)), sp); }
                }
                return HExpr {
                    kind: HExprKind::Call { callee: FuncId(u32::MAX - 28), args: hargs },
                    ty: HType::Vec { elem: Box::new(HType::Int) },
                    span: sp,
                };
            }
            if hargs.len() != 3 {
                self.err("par_map_int expects (int start, int end, int(int) f) or ([]int slice, int(int) f)", sp);
            } else {
                let ok_int = matches!(&hargs[0].ty, HType::Int) && matches!(&hargs[1].ty, HType::Int);
                let body = &hargs[2];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Int) && params.len() == 1 && matches!(params[0], HType::Int));
                if !ok_int { self.err("par_map_int: start/end must be `int`", sp); }
                if !ok_body { self.err(format!("par_map_int body must be `int(int)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 22), args: hargs },
                ty: HType::Vec { elem: Box::new(HType::Int) },
                span: sp,
            };
        }
        // once_do(*unit o, unit() init) — call init() once across all callers.
        // Builtin so codegen can split the Callable's code/env at the call site
        // (the runtime entry takes raw void pointers, avoiding a C type mismatch
        // with the synthesized Callable_unit_ struct).
        if name == "once_do" && qualifier.is_none() {
            let mut hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            // Accept either the legacy `*unit` handle or the typed `Once`
            // struct from the stdlib; for the latter, auto-extract `.h`.
            if let Some(first) = hargs.first_mut() {
                if let HType::Struct(id) = &first.ty {
                    let info = self.sym.struct_info(*id);
                    if info.name == "Once" {
                        let span = first.span;
                        let field_idx = info.fields.iter().position(|f| f.name == "h").unwrap_or(0);
                        let field_ty = info.fields.get(field_idx).map(|f| f.ty.clone())
                            .unwrap_or(HType::Ptr { mutable: true, inner: Box::new(HType::Unit) });
                        let base = first.clone();
                        *first = HExpr {
                            kind: HExprKind::Field { base: Box::new(base), field: field_idx },
                            ty: field_ty,
                            span,
                        };
                    }
                }
            }
            if hargs.len() != 2 {
                self.err("once_do expects (`Once o` or `*unit o`, `unit() init`)", sp);
            } else {
                let body = &hargs[1];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Unit) && params.is_empty());
                if !ok_body {
                    self.err(format!("once_do init must be `unit()`, got `{}`", type_str(&body.ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 32), args: hargs },
                ty: HType::Unit,
                span: sp,
            };
        }
        // par_map_bytes(*mut unit in, int n, int in_sz, int out_sz, body)
        // body: unit(*mut unit in_item, *mut unit out_item)
        if name == "par_map_bytes" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 5 {
                self.err("par_map_bytes expects (*mut unit in, int n, int in_sz, int out_sz, body)", sp);
            } else {
                let body = &hargs[4];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Unit) && params.len() == 2);
                if !ok_body {
                    self.err(format!("par_map_bytes body must be `unit(*mut unit, *mut unit)`, got `{}`", type_str(&body.ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 38), args: hargs },
                ty: HType::Ptr { mutable: true, inner: Box::new(HType::Unit) },
                span: sp,
            };
        }
        // par_for_each_float(slice, body) — float-slice iteration; body is `unit(float)`.
        if name == "par_for_each_float" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("par_for_each_float expects ([]float slice, unit(float) body)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(**elem, HType::Float)) ||
                    matches!(&hargs[0].ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(**elem, HType::Float)));
                let body = &hargs[1];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Unit) && params.len() == 1 && matches!(params[0], HType::Float));
                if !is_slice { self.err(format!("par_for_each_float: first arg must be `[]float`, got `{}`", type_str(&hargs[0].ty)), sp); }
                if !ok_body { self.err(format!("par_for_each_float body must be `unit(float)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 34), args: hargs },
                ty: HType::Unit,
                span: sp,
            };
        }
        // par_map_float(slice, fn) — float slice in/out; fn is `float(float)`.
        if name == "par_map_float" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("par_map_float expects ([]float slice, float(float) f)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(**elem, HType::Float)) ||
                    matches!(&hargs[0].ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(**elem, HType::Float)));
                let body = &hargs[1];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Float) && params.len() == 1 && matches!(params[0], HType::Float));
                if !is_slice { self.err(format!("par_map_float: first arg must be `[]float`, got `{}`", type_str(&hargs[0].ty)), sp); }
                if !ok_body { self.err(format!("par_map_float body must be `float(float)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 35), args: hargs },
                ty: HType::Vec { elem: Box::new(HType::Float) },
                span: sp,
            };
        }
        // par_reduce_float(slice, init, combine) — fold over float slice.
        if name == "par_reduce_float" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 3 {
                self.err("par_reduce_float expects ([]float slice, float init, float(float, float) combine)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(**elem, HType::Float)) ||
                    matches!(&hargs[0].ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(**elem, HType::Float)));
                let body = &hargs[2];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_init = matches!(&hargs[1].ty, HType::Float);
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Float) && params.len() == 2 &&
                            matches!(params[0], HType::Float) && matches!(params[1], HType::Float));
                if !is_slice { self.err(format!("par_reduce_float: first arg must be `[]float`, got `{}`", type_str(&hargs[0].ty)), sp); }
                if !ok_init { self.err("par_reduce_float: init must be `float`", sp); }
                if !ok_body { self.err(format!("par_reduce_float combine must be `float(float, float)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 36), args: hargs },
                ty: HType::Float,
                span: sp,
            };
        }
        // par_for_each(slice, body) — runs body(elem) for every elem; chunked.
        if name == "par_for_each" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("par_for_each expects ([]int slice, unit(int) body)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(**elem, HType::Int)) ||
                    matches!(&hargs[0].ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(**elem, HType::Int)));
                let body = &hargs[1];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Unit) && params.len() == 1 && matches!(params[0], HType::Int));
                if !is_slice { self.err(format!("par_for_each: first arg must be `[]int` or `&[]int`, got `{}`", type_str(&hargs[0].ty)), sp); }
                if !ok_body { self.err(format!("par_for_each body must be `unit(int)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 27), args: hargs },
                ty: HType::Unit,
                span: sp,
            };
        }
        // par_filter_int(slice, pred) — bool(int) predicate; returns filtered []int.
        if name == "par_filter_int" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("par_filter_int expects ([]int slice, bool(int) pred)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(**elem, HType::Int)) ||
                    matches!(&hargs[0].ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(**elem, HType::Int)));
                let body = &hargs[1];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Bool) && params.len() == 1 && matches!(params[0], HType::Int));
                if !is_slice { self.err(format!("par_filter_int: first arg must be `[]int`, got `{}`", type_str(&hargs[0].ty)), sp); }
                if !ok_body { self.err(format!("par_filter_int pred must be `bool(int)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 30), args: hargs },
                ty: HType::Vec { elem: Box::new(HType::Int) },
                span: sp,
            };
        }
        // file_listdir(path) -> []string.  Compiler builtin since the runtime
        // returns the array + count via an out-pointer and we need to wrap
        // it into a Slice_str literal at the call site.
        if name == "file_listdir" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 1 {
                self.err("file_listdir expects (string path)", sp);
            } else if !matches!(hargs[0].ty, HType::Str) {
                self.err(format!("file_listdir: arg must be string, got `{}`", type_str(&hargs[0].ty)), sp);
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 43), args: hargs },
                ty: HType::Slice { mutable: false, elem: Box::new(HType::Str) },
                span: sp,
            };
        }
        // str_split(s, sep) -> []string.  Same compiler-builtin trick.
        if name == "str_split" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("str_split expects (string s, string sep)", sp);
            } else if !matches!(hargs[0].ty, HType::Str) || !matches!(hargs[1].ty, HType::Str) {
                self.err("str_split: both args must be string", sp);
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 44), args: hargs },
                ty: HType::Slice { mutable: false, elem: Box::new(HType::Str) },
                span: sp,
            };
        }
        // par_filter_bytes(*mut unit in, int n, int item_sz, &mut int out_n, bool(*unit) pred)
        // -> *mut unit.  Same shape as par_map_bytes but for filtering.
        if name == "par_filter_bytes" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 5 {
                self.err("par_filter_bytes expects (*mut unit in, int n, int item_sz, &mut int out_n, bool(*unit) pred)", sp);
            } else {
                let body = &hargs[4];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Bool) && params.len() == 1);
                if !ok_body {
                    self.err(format!("par_filter_bytes pred must be `bool(*mut unit)`, got `{}`", type_str(&body.ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 41), args: hargs },
                ty: HType::Ptr { mutable: true, inner: Box::new(HType::Unit) },
                span: sp,
            };
        }
        // par_scan_bytes(*mut unit in, int n, int item_sz, unit(*unit acc, *unit cur, *unit out) combine)
        // -> *mut unit.  Generic inclusive scan over arbitrary-sized items.
        if name == "par_scan_bytes" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 4 {
                self.err("par_scan_bytes expects (*mut unit in, int n, int item_sz, unit(*unit acc, *unit cur, *unit out) combine)", sp);
            } else {
                let body = &hargs[3];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Unit) && params.len() == 3);
                if !ok_body {
                    self.err(format!("par_scan_bytes combine must be `unit(*mut unit, *mut unit, *mut unit)`, got `{}`", type_str(&body.ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 42), args: hargs },
                ty: HType::Ptr { mutable: true, inner: Box::new(HType::Unit) },
                span: sp,
            };
        }
        // par_filter_float(slice, pred) — bool(float) predicate over []float.
        if name == "par_filter_float" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("par_filter_float expects ([]float slice, bool(float) pred)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(**elem, HType::Float)) ||
                    matches!(&hargs[0].ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(**elem, HType::Float)));
                let body = &hargs[1];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Bool) && params.len() == 1 && matches!(params[0], HType::Float));
                if !is_slice { self.err(format!("par_filter_float: first arg must be `[]float`, got `{}`", type_str(&hargs[0].ty)), sp); }
                if !ok_body { self.err(format!("par_filter_float pred must be `bool(float)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 39), args: hargs },
                ty: HType::Slice { mutable: false, elem: Box::new(HType::Float) },
                span: sp,
            };
        }
        // par_scan_float(slice, combine) — inclusive prefix scan over []float.
        if name == "par_scan_float" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("par_scan_float expects ([]float slice, float(float, float) combine)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(**elem, HType::Float)) ||
                    matches!(&hargs[0].ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(**elem, HType::Float)));
                let body = &hargs[1];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Float) && params.len() == 2 &&
                            matches!(params[0], HType::Float) && matches!(params[1], HType::Float));
                if !is_slice { self.err(format!("par_scan_float: first arg must be `[]float`, got `{}`", type_str(&hargs[0].ty)), sp); }
                if !ok_body { self.err(format!("par_scan_float combine must be `float(float, float)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 40), args: hargs },
                ty: HType::Slice { mutable: false, elem: Box::new(HType::Float) },
                span: sp,
            };
        }
        // par_scan_int(slice, combine) — inclusive prefix scan with associative combine.
        if name == "par_scan_int" && qualifier.is_none() {
            let hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 2 {
                self.err("par_scan_int expects ([]int slice, int(int, int) combine)", sp);
            } else {
                let is_slice = matches!(&hargs[0].ty,
                    HType::Slice { elem, .. } if matches!(**elem, HType::Int)) ||
                    matches!(&hargs[0].ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                        HType::Slice { elem, .. } if matches!(**elem, HType::Int)));
                let body = &hargs[1];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Int) && params.len() == 2 &&
                            matches!(params[0], HType::Int) && matches!(params[1], HType::Int));
                if !is_slice { self.err(format!("par_scan_int: first arg must be `[]int`, got `{}`", type_str(&hargs[0].ty)), sp); }
                if !ok_body { self.err(format!("par_scan_int combine must be `int(int, int)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 31), args: hargs },
                ty: HType::Vec { elem: Box::new(HType::Int) },
                span: sp,
            };
        }
        // Built-in `yield_now()` — cooperative yield from a fiber back to
        // the scheduler.  No-op outside a fiber context.
        if name == "yield_now" && qualifier.is_none() {
            if !args.is_empty() {
                self.err("yield_now takes no arguments", sp);
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 20), args: Vec::new() },
                ty: HType::Unit,
                span: sp,
            };
        }
        // Built-in `par_reduce_int` — two shapes:
        //   par_reduce_int(start, end, init, combine) -> int   // range
        //   par_reduce_int(slice, init, combine)      -> int   // slice
        // Combine in both cases is `int(int, int)`.
        if name == "par_reduce_int" && qualifier.is_none() {
            let hargs_pre: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            let first_is_int_slice = hargs_pre.first().is_some_and(|a| matches!(&a.ty,
                HType::Slice { elem, .. } if matches!(**elem, HType::Int)) ||
                matches!(&a.ty, HType::Ref { inner, .. } if matches!(inner.as_ref(),
                    HType::Slice { elem, .. } if matches!(**elem, HType::Int))));
            if first_is_int_slice {
                if hargs_pre.len() != 3 {
                    self.err("par_reduce_int(slice, init, combine): expected 3 arguments", sp);
                } else {
                    let body = &hargs_pre[2];
                    let body_inner = match &body.ty {
                        HType::FnPtr { .. } => Some(&body.ty),
                        HType::Heap { inner } => Some(inner.as_ref()),
                        HType::Ptr { inner, .. } => Some(inner.as_ref()),
                        HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                        _ => None,
                    };
                    let ok_init = matches!(&hargs_pre[1].ty, HType::Int);
                    let ok_body = matches!(body_inner,
                        Some(HType::FnPtr { ret, params })
                            if matches!(**ret, HType::Int) && params.len() == 2 &&
                                matches!(params[0], HType::Int) && matches!(params[1], HType::Int));
                    if !ok_init { self.err("par_reduce_int(slice): init must be `int`", sp); }
                    if !ok_body { self.err(format!("par_reduce_int(slice) combine must be `int(int, int)`, got `{}`", type_str(&body.ty)), sp); }
                }
                return HExpr {
                    kind: HExprKind::Call { callee: FuncId(u32::MAX - 29), args: hargs_pre },
                    ty: HType::Int,
                    span: sp,
                };
            }
            let mut hargs: Vec<HExpr> = hargs_pre;
            if hargs.len() != 4 {
                self.err("par_reduce_int expects (int start, int end, int init, int(int, int) combine) or ([]int slice, int init, int(int, int) combine)", sp);
            } else {
                let ok_int = matches!(&hargs[0].ty, HType::Int)
                    && matches!(&hargs[1].ty, HType::Int)
                    && matches!(&hargs[2].ty, HType::Int);
                let body = &hargs[3];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(
                    body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Int)
                            && params.len() == 2
                            && matches!(params[0], HType::Int)
                            && matches!(params[1], HType::Int)
                );
                if !ok_int { self.err("par_reduce_int: start/end/init must be `int`", sp); }
                if !ok_body { self.err(format!("par_reduce_int body must be `int(int, int)`, got `{}`", type_str(&body.ty)), sp); }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 21), args: hargs },
                ty: HType::Int,
                span: sp,
            };
        }
        // Built-in growable-vector ops on `Vec<T>`:
        //   push(v, x)  - append x, growing the buffer (realloc) if needed
        //   pop(v) -> T - remove and return the last element (panics if empty)
        // `v` must be a mutable variable/field/element of `Vec<T>` type; it is
        // taken by mutable reference so the growth is visible to the caller.
        if (name == "push" || name == "pop") && qualifier.is_none() {
            let vh = self.check_expr(&args[0], None);
            if let HType::Vec { elem } = vh.ty.clone() {
                let elem = (*elem).clone();
                if !matches!(&vh.kind, HExprKind::Local(_) | HExprKind::Field { .. } | HExprKind::Index { .. } | HExprKind::GlobalRef(_)) {
                    self.err(format!("`{}` target must be a `Vec` variable, field, or element", name), sp);
                }
                let vref = HExpr {
                    kind: HExprKind::AddrOfRef { mutable: true, place: Box::new(vh) },
                    ty: HType::Ref { mutable: true, inner: Box::new(HType::Vec { elem: Box::new(elem.clone()) }) },
                    span: sp,
                };
                if name == "push" {
                    if args.len() != 2 { self.err("push expects (Vec v, T value)", sp); }
                    let xh = if args.len() == 2 { self.check_expr_coerce(&args[1], &elem) } else { HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp } };
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 60), args: vec![vref, xh] },
                        ty: HType::Unit, span: sp,
                    };
                } else {
                    if args.len() != 1 { self.err("pop expects (Vec v)", sp); }
                    return HExpr {
                        kind: HExprKind::Call { callee: FuncId(u32::MAX - 61), args: vec![vref] },
                        ty: elem, span: sp,
                    };
                }
            }
            // Not a Vec - fall through to ordinary function resolution.
        }

        // Built-in `par_for_range(start, end, closure)` — runs `closure(i)`
        // for every i in [start, end), chunked across the job-pool's
        // workers.  Body must be a `unit(int)` closure.
        if name == "par_for_range" && qualifier.is_none() {
            let mut hargs: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            if hargs.len() != 3 {
                self.err("par_for_range expects (int start, int end, unit(int) body)", sp);
            } else {
                let ok_int_a = matches!(&hargs[0].ty, HType::Int);
                let ok_int_b = matches!(&hargs[1].ty, HType::Int);
                let body = &hargs[2];
                let body_inner = match &body.ty {
                    HType::FnPtr { .. } => Some(&body.ty),
                    HType::Heap { inner } => Some(inner.as_ref()),
                    HType::Ptr { inner, .. } => Some(inner.as_ref()),
                    HType::OwnPtr { inner, .. } => Some(inner.as_ref()),
                    _ => None,
                };
                let ok_body = matches!(
                    body_inner,
                    Some(HType::FnPtr { ret, params })
                        if matches!(**ret, HType::Unit) && params.len() == 1 && matches!(params[0], HType::Int)
                );
                if !ok_int_a || !ok_int_b {
                    self.err("par_for_range: first two args must be `int`", sp);
                }
                if !ok_body {
                    self.err(format!("par_for_range body must be `unit(int)`, got `{}`", type_str(&body.ty)), sp);
                }
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 19), args: hargs },
                ty: HType::Unit,
                span: sp,
            };
        }
        // Built-in `read_line() -> own *char`.  Returns a heap NUL-terminated line
        // from stdin (without the trailing `\n`), or `null` on EOF.  Caller owns;
        // auto-freed at scope exit.
        if name == "read_line" && qualifier.is_none() {
            if !args.is_empty() { self.err("read_line takes no arguments", sp); }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 6), args: Vec::new() },
                ty: HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) },
                span: sp,
            };
        }
        // Built-in `read_int() -> int`.  Reads one base-10 integer from stdin;
        // panics on malformed input.
        if name == "read_int" && qualifier.is_none() {
            if !args.is_empty() { self.err("read_int takes no arguments", sp); }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX - 7), args: Vec::new() },
                ty: HType::Int,
                span: sp,
            };
        }
        // Built-in `format(fmt, ...) -> String`.  fmt is a string literal with
        // `{}` placeholders; each `{}` consumes one trailing arg.  Returns an
        // `own *char` so the lifetime pass auto-frees the result at scope exit.
        // Implemented via a snprintf-into-malloc helper - see the prelude for
        // the helper's C body (registered as `__maka_format` extern below).
        if name == "format" && qualifier.is_none() {
            return self.check_format(args, sp);
        }
        // Built-in `log` accepts any single arg and returns unit.
        if name == "log" {
            // Optimization: `log(format(LIT, args...))` lowers straight to a
            // printf-style call (no intermediate malloc'd String).  Only when the
            // format string is a literal and every arg is a supported scalar;
            // otherwise fall through to the normal format()+log path.
            if args.len() == 1 {
                if let ast::Expr::Call { callee: fc, args: fa, .. } = &args[0] {
                    if matches!(fc.as_ref(), ast::Expr::Ident(n, _) if n == "format") && !fa.is_empty() {
                        if let ast::Expr::Lit(ast::Lit::Str(fmt), _) = &fa[0] {
                            let fmt = fmt.clone();
                            if let Some(h) = self.try_log_format_print(&fmt, &fa[1..], sp) {
                                return h;
                            }
                        }
                    }
                }
            }
            // Optimization: `log(a + b + ...)` string concat -> printf with a %s
            // per piece, no per-concat malloc.  (`a + b` lowers to a chain of
            // __maka_str_concat calls; flatten it back to the operands.)
            if args.len() == 1 {
                let h = self.check_expr(&args[0], None);
                let is_concat = matches!(&h.kind, HExprKind::Call { callee, .. }
                    if matches!(callee.0, c if c == u32::MAX - 5 || c == u32::MAX - 8 || c == u32::MAX - 9 || c == u32::MAX - 10));
                if is_concat {
                    let mut leaves = Vec::new();
                    flatten_str_concat(h, &mut leaves);
                    let mut pf = String::with_capacity(leaves.len() * 2 + 1);
                    for leaf in &mut leaves {
                        // Owned pieces become borrowed string views (printf reads
                        // them); fresh-temp pieces are freed by the lifetime pass.
                        if matches!(&leaf.ty, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char)) {
                            leaf.ty = HType::Str;
                        }
                        pf.push_str("%s");
                    }
                    pf.push('\n');
                    let mut hargs = Vec::with_capacity(leaves.len() + 1);
                    hargs.push(HExpr { kind: HExprKind::LitStr(pf), ty: HType::Str, span: sp });
                    hargs.extend(leaves);
                    return HExpr { kind: HExprKind::Call { callee: FuncId(u32::MAX - 58), args: hargs }, ty: HType::Unit, span: sp };
                }
                // Not a concat: reuse the already-checked value for the normal path.
                let h = match &h.ty {
                    HType::Ref { inner, .. } if matches!(**inner, HType::Int | HType::SizedInt { .. } | HType::Float | HType::Bool | HType::Char | HType::Enum(_)) => self.auto_deref(h),
                    _ => h,
                };
                let h = if matches!(&h.ty, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char)) {
                    self.coerce(h, &HType::Str)
                } else { h };
                return HExpr { kind: HExprKind::Call { callee: FuncId(u32::MAX), args: vec![h] }, ty: HType::Unit, span: sp };
            }
            let mut hargs = Vec::new();
            for a in args {
                let h = self.check_expr(a, None);
                // Auto-deref &T for primitive/enum types so log prints the value.
                let h = match &h.ty {
                    HType::Ref { inner, .. } if matches!(**inner, HType::Int | HType::SizedInt { .. } | HType::Float | HType::Bool | HType::Char | HType::Enum(_)) => self.auto_deref(h),
                    _ => h,
                };
                // Coerce `own *char` to `string` so log dispatches to the string
                // helper, not the pointer helper.
                let h = if matches!(&h.ty, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char)) {
                    self.coerce(h, &HType::Str)
                } else {
                    h
                };
                hargs.push(h);
            }
            return HExpr {
                kind: HExprKind::Call { callee: FuncId(u32::MAX), args: hargs },
                ty: HType::Unit,
                span: sp,
            };
        }

        // Gather candidates: qualified or unqualified.
        let mut candidates: Vec<(FuncId, FuncSig)> = if let Some(modq) = &module_qualifier {
            // Module-qualified call: filter to sigs declared in a module whose last
            // path segment matches the qualifier.
            self.sym.sigs.iter().enumerate()
                .filter(|(_, s)| s.name == name
                    && s.logic.is_none()
                    && s.module_path.last().map(|p| p == modq).unwrap_or(false))
                .map(|(i, s)| (FuncId(i as u32), s.clone()))
                .collect()
        } else { match &qualifier {
            Some(logic) => self.sym.funcs_by_qualified(Some(logic), &name)
                .into_iter().map(|(f, s)| (f, s.clone())).collect(),
            None => {
                let mut v: Vec<(FuncId, FuncSig)> = self.sym.funcs_by_qualified(None, &name)
                    .into_iter().map(|(f, s)| (f, s.clone())).collect();
                if v.is_empty() {
                    if let Some(l) = self.cur_logic.clone() {
                        v = self.sym.funcs_by_qualified(Some(&l), &name)
                            .into_iter().map(|(f, s)| (f, s.clone())).collect();
                    }
                }
                // Open dispatch for postfix method calls: when we have no top-level
                // candidate but the call has the shape `receiver.method(args)`, look
                // across every attr / logic namespace for a method with this name —
                // overload resolution by first-arg type then picks the right one.
                if v.is_empty() {
                    v = self.sym.sigs.iter().enumerate()
                        .filter(|(_, s)| s.name == name && s.logic.is_some())
                        .map(|(i, s)| (FuncId(i as u32), s.clone()))
                        .collect();
                }
                v
            }
        } };
        if candidates.is_empty() {
            let full = if let Some(m) = &module_qualifier {
                format!("{}.{}", m, name)
            } else {
                qualifier.as_ref().map(|q| format!("{}.{}", q, name)).unwrap_or(name.clone())
            };
            self.err(format!("unknown function `{}`", full), sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        }

        // Probe arg types once for overload resolution.  When there is a single
        // candidate by name and arity, there is no overload ambiguity, so probe
        // each argument against that candidate's parameter type: this pushes the
        // expected type down into generic struct/enum literals in argument
        // position (bidirectional inference), e.g. `f(Option.Some{ value = ... })`
        // where `f` expects a concrete `Option<Shape<int>>`.  A type hint only
        // affects literals/constructors (idents and other exprs ignore it), so
        // this is safe even for receiver/auto-borrow slots.  With multiple
        // candidates the expected type is ambiguous, so fall back to no hint.
        let single_expect: Option<Vec<HType>> = if candidates.len() == 1 {
            let (_, sig) = &candidates[0];
            if !sig.is_variadic && sig.param_tys.len() == args.len() {
                Some(sig.param_tys.clone())
            } else { None }
        } else { None };
        let mut probed: Vec<HExpr> = match &single_expect {
            Some(ptys) => args.iter().enumerate()
                .map(|(i, a)| self.check_expr(a, Some(&ptys[i]))).collect(),
            None => args.iter().map(|a| self.check_expr(a, None)).collect(),
        };

        // Dynamic dispatch path: if the receiver is a `dyn Trait` (possibly via &/*),
        // lower to an indirect call through the vtable. Skip overload resolution.
        if !probed.is_empty() {
            let is_dyn = matches!(strip_to_dyn(&probed[0].ty), Some(_));
            if is_dyn {
                // Find a signature in the trait logic with matching name and arity.
                let trait_name = match strip_to_dyn(&probed[0].ty) { Some(traits) => traits[0].clone(), None => String::new() };
                let linfo = match self.sym.logic_by_name(&trait_name) {
                    Some(li) => li.clone(),
                    None => {
                        self.err(format!("unknown trait `{}`", trait_name), sp);
                        return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
                    }
                };
                // Pick the first overload by name and arity.
                let chosen = linfo.funcs.iter().find_map(|fid| {
                    let s = self.sym.func_sig(*fid).clone();
                    if s.name == name && s.param_tys.len() == probed.len() { Some((*fid, s)) } else { None }
                });
                let Some((fid, sig)) = chosen else {
                    self.err(format!("trait `{}` has no method `{}` with arity {}", trait_name, name, probed.len()), sp);
                    return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
                };
                // Type-check args (no coercion needed for arg 0 since dyn).
                let mut hargs = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    if i == 0 {
                        hargs.push(probed[0].clone());
                    } else {
                        let want = sig.param_tys.get(i).cloned().unwrap_or(HType::Unit);
                        hargs.push(self.check_expr_coerce(a, &want));
                    }
                }
                return HExpr {
                    kind: HExprKind::Call { callee: fid, args: hargs },
                    ty: sig.ret.clone(),
                    span: sp,
                };
            }
        }

        // Snapshot pre-filter candidates so we can produce a useful error if
        // overload resolution ends up empty.
        let pre_filter: Vec<(FuncId, FuncSig)> = candidates.clone();

        // Step 3: filter by arity + parameter compatibility (allow exact or via unify for generics).
        // Variadic externs (e.g. `printf`) only require their fixed-arity prefix to match; any
        // extra args after that are passed through to the C variadic ABI.
        candidates.retain(|(_, sig)| {
            if sig.is_variadic {
                if probed.len() < sig.param_tys.len() { return false; }
            } else {
                if sig.param_tys.len() != probed.len() { return false; }
            }
            for (i, ph) in probed.iter().take(sig.param_tys.len()).enumerate() {
                if param_compatible_with_sym(&sig.param_tys[i], &ph.ty, &sig.type_params, self.sym) {
                    continue;
                }
                // Auto-borrow on the receiver slot: accepted in both call
                // shapes that name an attr-style method on a value:
                //   * postfix `f.method()` — receiver becomes arg 0
                //   * attr-qualified `Attr.method(f)` / `Attr::method(f)` —
                //     same dispatch, just a different surface
                // The arg-build step below inserts the `AddrOfRef`.  Only the
                // receiver slot benefits — non-receiver args still require an
                // explicit `&`.
                if (is_postfix || qualifier.is_some()) && i == 0 {
                    if let HType::Ref { inner, .. } = &sig.param_tys[i] {
                        if param_compatible_with_sym(inner, &ph.ty, &sig.type_params, self.sym) {
                            continue;
                        }
                    }
                }
                return false;
            }
            true
        });

        // Embed-promotion fallback: when no direct candidate matches but the
        // receiver/first-arg's struct embeds a type that does, drill into the
        // embed field and retry.  Covers both postfix `x.method()` calls and
        // direct calls like `f(&x)` where x embeds the target.  Ambiguous matches
        // (the embed chain reaches more than one candidate, OR more than one
        // embed field reaches the same target) are rejected.
        //
        // If a rewrite happens, `embed_promoted_first` carries the rewritten HExpr
        // so the final argument-building step uses it instead of re-checking the
        // AST (which would produce an un-drilled receiver).
        let mut embed_promoted_first: Option<HExpr> = None;
        let _ = is_postfix;   // accepted in both call shapes now
        if candidates.is_empty() && !probed.is_empty() {
            if let Some(recv_sid) = struct_id_of(&probed[0].ty) {
                // Pre-filter the sig list: candidates may include the unqualified
                // top-level pool we already gathered.  For embed promotion we re-scan
                // every sig (top-level + attr-namespaced) whose name matches.
                let mut hits: Vec<(FuncId, FuncSig, Vec<usize>, StructId)> = Vec::new();
                for (idx, sig) in self.sym.sigs.iter().enumerate() {
                    if sig.name != name { continue; }
                    if sig.param_tys.is_empty() { continue; }
                    if sig.param_tys.len() != probed.len() { continue; }
                    let want_first = &sig.param_tys[0];
                    let Some(target_sid) = struct_id_of(want_first) else { continue; };
                    if target_sid == recv_sid { continue; } // already a direct match
                    let Some(path) = self.find_embed_path(recv_sid, target_sid) else { continue; };
                    if path.is_empty() { continue; }
                    // Verify the remaining args match without embed promotion.
                    let mut rest_ok = true;
                    for (i, ph) in probed.iter().enumerate().skip(1) {
                        if !param_compatible(&sig.param_tys[i], &ph.ty, &sig.type_params) {
                            rest_ok = false; break;
                        }
                    }
                    if !rest_ok { continue; }
                    hits.push((FuncId(idx as u32), sig.clone(), path, target_sid));
                }
                match hits.len() {
                    0 => {}
                    1 => {
                        let (fid, sig, path, _target) = hits.remove(0);
                        // Drill the embed chain into probed[0].
                        let drilled = self.drill_embed_path(probed[0].clone(), &path);
                        // Wrap in AddrOfRef if the target wants a reference / borrow.
                        let want_first = &sig.param_tys[0];
                        let new_first = match want_first {
                            HType::Ref { mutable, inner: _ } => HExpr {
                                kind: HExprKind::AddrOfRef { mutable: *mutable, place: Box::new(drilled.clone()) },
                                ty: HType::Ref { mutable: *mutable, inner: Box::new(drilled.ty.clone()) },
                                span: sp,
                            },
                            _ => drilled,
                        };
                        embed_promoted_first = Some(new_first.clone());
                        probed[0] = new_first;
                        candidates = vec![(fid, sig)];
                    }
                    n => {
                        self.err(
                            format!(
                                "call to `{}` is ambiguous via embed promotion — {} reachable impls; qualify the receiver to pick one (e.g. `value.embed_field.{}()`)",
                                name, n, name,
                            ),
                            sp,
                        );
                        return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
                    }
                }
            }
        }

        // Step 5: specificity. Rank: 0 = concrete (no type_params), 1 = generic.
        candidates.sort_by_key(|(_, sig)| if sig.type_params.is_empty() { 0 } else { 1 });
        let top_rank = candidates.first().map(|(_, s)| if s.type_params.is_empty() { 0 } else { 1 }).unwrap_or(99);
        let tied: Vec<_> = candidates.iter().take_while(|(_, s)| (if s.type_params.is_empty() { 0 } else { 1 }) == top_rank).cloned().collect();

        if tied.is_empty() {
            // Build a diagnostic that names the candidates considered and the
            // call-site argument shape, so the user can see at a glance which
            // arg failed which signature.
            let arg_shape: String = probed.iter()
                .map(|h| type_str(&h.ty))
                .collect::<Vec<_>>()
                .join(", ");
            let mut msg = format!(
                "no matching overload for `{}` called with ({})",
                name, arg_shape,
            );
            if pre_filter.is_empty() {
                msg.push_str(" - no function by that name found in scope");
            } else {
                msg.push_str("; candidates were:");
                for (_, sig) in pre_filter.iter().take(8) {
                    let params: Vec<String> = sig.param_tys.iter().map(type_str).collect();
                    msg.push_str(&format!("\n    {}({}) -> {}",
                        sig.name, params.join(", "), type_str(&sig.ret)));
                }
                if pre_filter.len() > 8 {
                    msg.push_str(&format!("\n    ... and {} more", pre_filter.len() - 8));
                }
            }
            self.err(msg, sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        }
        // Bound-aware narrowing: when ambiguous, look at the current function's
        // `where T has Attr<U>` clauses.  After substituting via `self.subst`,
        // if the bound's receiver type matches the call's receiver, use the
        // bound's attr_args to keep only impls whose `has_impl.attr_args` match.
        // Works for both template-check (TyVars) and post-monomorph (concrete).
        let mut tied = tied;
        if tied.len() > 1 && !probed.is_empty() {
            let recv_key = receiver_key(&probed[0].ty, self.sym);
            let bound_args_by_attr: Vec<(String, Vec<HType>)> = self.cur_where_bounds.iter()
                .filter_map(|(trait_name, args, _bindings)| {
                    if args.is_empty() { return None; }
                    let bound_recv = args[0].subst(&self.subst);
                    let bound_recv_key = receiver_key(&bound_recv, self.sym);
                    if bound_recv_key.is_none() || bound_recv_key != recv_key { return None; }
                    let attr_args: Vec<HType> = args[1..].iter().map(|t| t.subst(&self.subst)).collect();
                    Some((trait_name.clone(), attr_args))
                })
                .collect();
            if !bound_args_by_attr.is_empty() {
                let filtered: Vec<(FuncId, FuncSig)> = tied.iter().filter(|(fid, sig)| {
                    let Some(impl_record) = self.sym.has_impls.iter().find(|h| h.func_ids.contains(fid)) else {
                        return true;
                    };
                    let _ = sig;
                    bound_args_by_attr.iter().any(|(attr, attr_args)| {
                        *attr == impl_record.attr_name
                            && attr_args.len() == impl_record.attr_args.len()
                            && attr_args.iter().zip(impl_record.attr_args.iter()).all(|(a, b)| type_eq(a, b))
                    })
                }).cloned().collect();
                if !filtered.is_empty() { tied = filtered; }
            }
        }
        // Subject-based narrowing: a no-self attr method (e.g. `from_word(w) -> T`)
        // is selected by its arguments only as a last resort.  When still tied and
        // a `where T has Attr` clause is in scope, keep the impl whose type matches
        // the bound subject T (post-subst) - that is the impl being called for.
        if tied.len() > 1 {
            let subjects: Vec<(String, String)> = self.cur_where_bounds.iter()
                .filter_map(|(trait_name, args, _)| {
                    if args.is_empty() { return None; }
                    let subj = args[0].subst(&self.subst);
                    crate::resolve::underlying_struct_key(self.sym, &subj).map(|k| (trait_name.clone(), k))
                })
                .collect();
            if !subjects.is_empty() {
                let filtered: Vec<(FuncId, FuncSig)> = tied.iter().filter(|(fid, _)| {
                    match self.sym.has_impls.iter().find(|h| h.func_ids.contains(fid)) {
                        Some(h) => subjects.iter().any(|(attr, k)| *attr == h.attr_name && *k == h.type_key),
                        None => true,
                    }
                }).cloned().collect();
                if !filtered.is_empty() && filtered.len() < tied.len() { tied = filtered; }
            }
        }
        if tied.len() > 1 {
            self.err(format!("ambiguous call to `{}`: {} candidates", name, tied.len()), sp);
        }
        let (fid, sig) = tied[0].clone();
        let _ = candidates;
        // Resolve parameter types by re-reading the AST function.
        // We need access to ast::FuncDecl for parameter types — keep a side map.
        // For now, look up the sig and re-extract from the parser via name.
        // We approximate using HFunc params not yet built; so re-resolve from the AST module via callback.
        // Hack: we get param types from the existing HFuncs already typed (impossible across funcs).
        // Instead, store param HTypes on FuncSig — done below in a helper.
        let template_param_tys = sig.param_tys.clone();
        let template_ret = sig.ret.clone();
        let type_params = sig.type_params.clone();
        let is_variadic = sig.is_variadic;
        if is_variadic {
            if args.len() < template_param_tys.len() {
                self.err(format!("function `{}` expects at least {} args, got {}", name, template_param_tys.len(), args.len()), sp);
            }
        } else if args.len() != template_param_tys.len() {
            self.err(format!("function `{}` expects {} args, got {}", name, template_param_tys.len(), args.len()), sp);
        }

        // If the function is generic, infer substitution from arg types.
        let mut env: std::collections::HashMap<String, HType> = std::collections::HashMap::new();
        if !type_params.is_empty() {
            // First do a probe: type-check each arg without coercion to learn its type.
            let probed: Vec<HExpr> = args.iter().map(|a| self.check_expr(a, None)).collect();
            for (i, ph) in probed.iter().enumerate() {
                if let Some(want) = template_param_tys.get(i) {
                    unify_with_sym(want, &ph.ty, &mut env, self.sym);
                }
            }
            // Bind type params that appear only in the return type from the call's
            // expected type (e.g. `Stack<int> s = snew();` infers T=int even with
            // no arguments to unify against).
            if type_params.iter().any(|tp| !env.contains_key(tp)) {
                if let Some(exp) = &ret_expected {
                    unify_with_sym(&template_ret, exp, &mut env, self.sym);
                }
            }
            // Ensure all type params got substitutions.
            for tp in &type_params {
                if !env.contains_key(tp) {
                    self.err(format!("cannot infer type parameter `{}` for `{}`", tp, name), sp);
                    env.insert(tp.clone(), HType::Int);
                }
            }
        }

        // Cross-module visibility checks:
        //   1. Callee must be `pub`.
        //   2. Callee's (module_path, name) must appear in the caller file's imports
        //      OR the callee is an extern (extern decls are linked externally and have
        //      no source-level module to import from).
        if let Some(callee_sig) = self.sym.sigs.get(fid.0 as usize) {
            if callee_sig.module_path != self.cur_module {
                if !callee_sig.is_pub {
                    self.err(
                        format!(
                            "`{}` is private to module `{}`; mark it `pub` to call from `{}`",
                            callee_sig.name,
                            if callee_sig.module_path.is_empty() { "<root>".to_string() } else { callee_sig.module_path.join(".") },
                            if self.cur_module.is_empty() { "<root>".to_string() } else { self.cur_module.join(".") },
                        ),
                        sp,
                    );
                } else if !callee_sig.is_extern {
                    let imported = self.cur_imports.iter().any(|(m, n)| {
                        m == &callee_sig.module_path && (n == &callee_sig.name || n == "*")
                    });
                    // `use Mod.Type.Attr;` also authorizes any method call on that
                    // attr-namespaced impl — the impl is explicitly propagated, so its
                    // methods come along.
                    let authorized_by_use = callee_sig.logic.as_ref().is_some_and(|attr|
                        self.cur_has_imports.iter().any(|imp|
                            imp.module_path == callee_sig.module_path && imp.attr_name == *attr
                        )
                    );
                    if !imported && !authorized_by_use {
                        self.err(
                            format!(
                                "`{}` is in module `{}` and must be imported (`import {}.{};`) to call from `{}`",
                                callee_sig.name,
                                if callee_sig.module_path.is_empty() { "<root>".to_string() } else { callee_sig.module_path.join(".") },
                                if callee_sig.module_path.is_empty() { "<root>".to_string() } else { callee_sig.module_path.join(".") },
                                callee_sig.name,
                                if self.cur_module.is_empty() { "<root>".to_string() } else { self.cur_module.join(".") },
                            ),
                            sp,
                        );
                    }
                }
            }
        }
        // Compute the final FuncId: if generic, queue an instantiation. The analyze pass
        // will allocate the final FuncId and rewrite the placeholder later.
        let (final_fid, final_param_tys, final_ret) = if type_params.is_empty() {
            // The resolved callee may be an already-instantiated generic whose stored
            // signature still carries un-concretized GenericPattern types (e.g. the
            // target instantiation was not yet registered when the request was first
            // recorded). Re-concretize at the call site now that all instantiations
            // exist, so argument coercion sees the concrete Struct/Enum id.
            let cp: Vec<HType> = template_param_tys.iter()
                .map(|t| concretize_generic_patterns(t, self.sym))
                .collect();
            let cr = concretize_generic_patterns(&template_ret, self.sym);
            (fid, cp, cr)
        } else {
            let new_param_tys: Vec<HType> = template_param_tys.iter()
                .map(|t| concretize_generic_patterns(&t.subst(&env), self.sym))
                .collect();
            let new_ret = concretize_generic_patterns(&template_ret.subst(&env), self.sym);
            let req_idx = self.instantiation_requests.len();
            self.instantiation_requests.push(InstantiationReq {
                template_fid: fid,
                args: type_params.iter().map(|tp| env[tp].clone()).collect(),
                caller_module: self.cur_module.clone(),
                caller_has_imports: self.cur_has_imports.clone(),
            });
            // Placeholder FuncId — value encodes the request index. Analyze rewrites it.
            (FuncId(crate::PLACEHOLDER_FID_BASE - req_idx as u32), new_param_tys, new_ret)
        };

        let mut hargs = Vec::new();
        for (i, a) in args.iter().enumerate() {
            if i == 0 {
                if let Some(promoted) = embed_promoted_first.take() {
                    hargs.push(promoted);
                    continue;
                }
                // Auto-borrow on receiver slot (matches the candidate filter
                // above): postfix call OR attr-qualified call.
                if is_postfix || qualifier.is_some() {
                    if let Some(want) = final_param_tys.get(0) {
                        if let HType::Ref { mutable, inner } = want {
                            let h = self.check_expr(a, None);
                            if type_eq(&h.ty, inner) {
                                let sp = h.span;
                                hargs.push(HExpr {
                                    kind: HExprKind::AddrOfRef { mutable: *mutable, place: Box::new(h.clone()) },
                                    ty: HType::Ref { mutable: *mutable, inner: Box::new(h.ty.clone()) },
                                    span: sp,
                                });
                                continue;
                            }
                        }
                    }
                }
            }
            if i < final_param_tys.len() {
                let want = final_param_tys[i].clone();
                hargs.push(self.check_expr_coerce(a, &want));
            } else {
                // Variadic trailing arg: type-check without coercion, accept whatever the
                // user gave us. C's varargs ABI handles default promotions itself.
                hargs.push(self.check_expr(a, None));
            }
        }
        // Inline function: emit an InlineCall so codegen splices the body at this call site.
        // Read inline-ness from the *template* fid - for a generic call `final_fid`
        // is a placeholder with no sig yet, so it would otherwise look non-inline
        // and be lowered to a (never-emitted) direct call.
        let is_inline = self.sym.sigs.get(fid.0 as usize)
            .map(|s| s.is_inline).unwrap_or(false);
        let kind = if is_inline && final_fid.0 < u32::MAX - 1024 {
            // The propagate-vs-caller-return check happens as a post-pass in `analyze()`
            // because the inline function's HFunc may not yet be in `sym.funcs` here.
            HExprKind::InlineCall { callee: final_fid, args: hargs }
        } else {
            HExprKind::Call { callee: final_fid, args: hargs }
        };
        HExpr { kind, ty: final_ret, span: sp }
    }

    fn check_cast(&mut self, expr: &ast::Expr, ty: &ast::Type, _checked: bool, sp: Span) -> HExpr {
        let h = self.check_expr(expr, None);
        // Resolve the cast target with the current fn's type params + instantiation
        // subst, so `x as *T` works inside a generic body (e.g. atomic pointer get).
        let to = self.resolve_local_ty(ty);

        // `as dyn Trait` — special: produces a dyn fat pointer or `&dyn` / `&mut dyn`.
        if let HType::Dyn { traits } = &to {
            return self.check_to_dyn(h, traits.clone(), false, sp);
        }
        // `own &T` (Heap) is never a valid cast target — synthesizing an
        // owning binding from somewhere else would create a phantom free
        // obligation.  `&T` targets are handled per-arm by classify_cast
        // (only allowed from `own &T`; nullable sources must use `&(p!)`).
        if matches!(to, HType::Heap { .. }) {
            self.err("cannot cast to `own &T` — owning bindings come only from `alloc` or moves", sp);
        }
        // Bool ↔ int forbidden per §7.4
        let from = &h.ty;
        let bool_int_forbidden = match (from, &to) {
            (HType::Bool, HType::Int) | (HType::Int, HType::Bool) => true,
            _ => false,
        };
        if bool_int_forbidden {
            self.err("bool ↔ int conversion is not allowed; use if/else", sp);
        }
        let kind = self.classify_cast(from, &to, false, sp);
        HExpr { kind: HExprKind::Cast { expr: Box::new(h), kind, to: to.clone() }, ty: to, span: sp }
    }

    /// Handle `expr as dyn Trait`. The source must be a reference or a value whose underlying
    /// concrete type is known; verify trait satisfaction and produce a Dyn fat pointer.
    fn check_to_dyn(&mut self, h: HExpr, traits: Vec<String>, _checked: bool, sp: Span) -> HExpr {
        // Determine the concrete struct id from `h.ty`. Allow `&T`, `&mut T`, `T`.
        let (struct_id, want_mut_ref) = match &h.ty {
            HType::Ref { mutable, inner } => match inner.as_ref() {
                HType::Struct(id) => (*id, Some(*mutable)),
                _ => { self.err("`as dyn Trait` requires a struct or reference-to-struct source", sp);
                    return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp }; }
            },
            HType::Struct(id) => (*id, None),
            _ => { self.err("`as dyn Trait` requires a struct or reference-to-struct source", sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp }; }
        };

        // For each trait in `traits`, verify the concrete type satisfies it: every function
        // in the logic must have a matching overload for the concrete struct as receiver.
        for tn in &traits {
            let Some(linfo) = self.sym.logic_by_name(tn) else {
                self.err(format!("unknown trait/logic `{}`", tn), sp);
                continue;
            };
            // Group by name; we only require that each *distinct* name has at least one overload
            // accepting the concrete type as its first parameter.
            let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut satisfied_names: std::collections::HashSet<String> = std::collections::HashSet::new();
            for fid in &linfo.funcs {
                let sig = self.sym.func_sig(*fid);
                seen_names.insert(sig.name.clone());
                if sig.param_tys.is_empty() { continue; }
                let first = &sig.param_tys[0];
                if let HType::Ref { inner, .. } = first {
                    if let HType::Struct(sid) = inner.as_ref() {
                        if *sid == struct_id {
                            satisfied_names.insert(sig.name.clone());
                        }
                    }
                }
            }
            for n in &seen_names {
                if !satisfied_names.contains(n) {
                    self.err(format!("type does not satisfy trait `{}`: missing overload of `{}`",
                                     tn, n), sp);
                }
            }
        }

        // Result type: keep `dyn Trait` value form; downstream coercion adapts to `&dyn`/`&mut dyn`.
        let dyn_ty = HType::Dyn { traits: traits.clone() };
        let kind = CastKind::ToDyn { trait_name: traits[0].clone(), struct_id };
        let result_ty = match want_mut_ref {
            Some(true) => HType::Ref { mutable: true, inner: Box::new(dyn_ty.clone()) },
            Some(false) => HType::Ref { mutable: false, inner: Box::new(dyn_ty.clone()) },
            None => dyn_ty.clone(),
        };
        HExpr {
            kind: HExprKind::Cast { expr: Box::new(h), kind, to: dyn_ty },
            ty: result_ty,
            span: sp,
        }
    }

    fn classify_cast(&mut self, from: &HType, to: &HType, checked: bool, sp: Span) -> CastKind {
        use HType::*;
        if !checked {
            match (from, to) {
                (Int, Int) | (Float, Float) | (Int, Float) | (Float, Int) => CastKind::Numeric,
                (Int, SizedInt { .. }) | (SizedInt { .. }, Int)
                    | (SizedInt { .. }, SizedInt { .. })
                    | (SizedInt { .. }, Float) | (Float, SizedInt { .. }) => CastKind::Numeric,
                // f32 (SizedFloat) numeric casts: to/from int, float, sized int,
                // and f64/f32.  Without this `f32 as int` etc. were rejected.
                (SizedFloat { .. }, SizedFloat { .. })
                    | (SizedFloat { .. }, Float) | (Float, SizedFloat { .. })
                    | (SizedFloat { .. }, Int) | (Int, SizedFloat { .. })
                    | (SizedFloat { .. }, SizedInt { .. }) | (SizedInt { .. }, SizedFloat { .. }) => CastKind::Numeric,
                (Enum(_), Int) => CastKind::EnumToInt,
                // §3 `int as Enum` — runtime bounds-checked against the
                // variant count, panics on out-of-range (same shape as
                // array indexing).  Result is the Enum value itself, NOT
                // a nullable pointer.
                (Int, Enum(_)) | (SizedInt { .. }, Enum(_)) => CastKind::IntToEnumChecked,
                (Char, Int) | (Int, Char) => CastKind::CharIntInt,
                (Char, SizedInt { .. }) | (SizedInt { .. }, Char) => CastKind::CharIntInt,
                // Reinterpret cast.  Three flavors:
                //   *T → integer     — safe (reading an address is harmless).
                //   *T → *U          — §3 prefix rule: safe iff U is a structural
                //                       prefix of T (data → data, U's fields match
                //                       T's first |U.fields| fields by name and
                //                       type).  Otherwise must be inside `unsafe { }`.
                //   integer → *T     — UNSAFE: synthesizes an arbitrary pointer with no
                //                       dep tracking.  Must be inside `unsafe { }`.
                (Ptr { .. }, SizedInt { bits: 0, .. }) | (Ptr { .. }, Int) => CastKind::Reinterpret,
                (Ptr { inner: from_inner, .. }, Ptr { inner: to_inner, .. }) => {
                    // §6.6 safe pointer-to-pointer cast cases (between
                    // different inner types — identity is already covered
                    // via the structural-prefix helper):
                    //   `*int → *Enum`  — runtime tag check, null on fail.
                    //   `*Enum → *int`  — always valid (every variant has
                    //                     a valid int representation).
                    //   `*T → *U` between `data` structs where U is a
                    //   structural prefix of T.
                    // Everything else requires `unsafe { ... }`.
                    if matches!(from_inner.as_ref(), Int) && matches!(to_inner.as_ref(), Enum(_)) {
                        return CastKind::IntPtrToEnumPtrChecked;
                    }
                    if matches!(from_inner.as_ref(), Enum(_)) && matches!(to_inner.as_ref(), Int) {
                        return CastKind::Reinterpret;
                    }
                    if self.in_unsafe == 0 && !self.is_structural_prefix(from_inner, to_inner) {
                        self.err(
                            format!(
                                "cast `{}` → `{}` is not a structural prefix and requires \
                                 `unsafe {{ ... }}` — target must be a `data` whose fields are \
                                 a prefix (same names, types, order) of the source's `data`",
                                type_str(from), type_str(to),
                            ),
                            sp,
                        );
                    }
                    CastKind::Reinterpret
                }
                // References can be read as addresses — safe, just looking at the pointer value.
                (Ref { .. }, SizedInt { bits: 0, .. }) | (Ref { .. }, Int) => CastKind::Reinterpret,
                // &T  →  *T  is the FFI bridge: same address, narrower lifetime tracking
                //               drops to "non-owning". The pointee still aliases the
                //               source binding's memory, so this is no less safe than
                //               *T → *U (which is already allowed without unsafe).
                (Ref { .. }, Ptr { .. }) => CastKind::Reinterpret,
                // &T → raw *T is the explicit FFI escape; require `unsafe { }` so the
                // call site visibly opts out of borrow tracking.
                (Ref { .. }, RawPtr { .. }) => {
                    if self.in_unsafe == 0 {
                        self.err(
                            "casting a reference to a `raw *T` drops borrow tracking; \
                             wrap the cast in an `unsafe { ... }` block".to_string(),
                            sp,
                        );
                    }
                    CastKind::Reinterpret
                }
                // own *T and own &T can also expose their address (no ownership transfer).
                (OwnPtr { .. }, SizedInt { bits: 0, .. }) | (OwnPtr { .. }, Int) => CastKind::Reinterpret,
                (Heap { .. }, SizedInt { bits: 0, .. }) | (Heap { .. }, Int) => CastKind::Reinterpret,
                // Explicit pointer-kind casts that mirror the safe-direction
                // implicit coercions in `check_expr_coerce` (loosening: drop
                // owning, drop tracking, drop non-null).  All are codegen
                // no-ops; the cast is purely a re-tag.  Tightening directions
                // (e.g. `*T as &T`) are intentionally NOT here — they need
                // a null proof, written `&(p!)`.
                (OwnPtr { .. }, Ptr { .. }) => {
                    if let (OwnPtr { mutable: am, inner: ai }, Ptr { mutable: bm, inner: bi }) = (from, to) {
                        if !(*am || !*bm) || !type_eq(ai, bi) {
                            self.err(format!("invalid cast: {:?} as {:?}", from, to), sp);
                        }
                    }
                    CastKind::Reinterpret
                }
                (OwnPtr { .. }, RawPtr { .. }) => {
                    if let (OwnPtr { mutable: am, inner: ai }, RawPtr { mutable: bm, inner: bi }) = (from, to) {
                        if !(*am || !*bm) || !type_eq(ai, bi) {
                            self.err(format!("invalid cast: {:?} as {:?}", from, to), sp);
                        }
                    }
                    CastKind::Reinterpret
                }
                (Heap { .. }, Ptr { .. }) => {
                    if let (Heap { inner: ai }, Ptr { mutable: _, inner: bi }) = (from, to) {
                        if !type_eq(ai, bi) {
                            self.err(format!("invalid cast: {:?} as {:?}", from, to), sp);
                        }
                    }
                    CastKind::Reinterpret
                }
                (Heap { .. }, RawPtr { .. }) => {
                    if let (Heap { inner: ai }, RawPtr { mutable: _, inner: bi }) = (from, to) {
                        if !type_eq(ai, bi) {
                            self.err(format!("invalid cast: {:?} as {:?}", from, to), sp);
                        }
                    }
                    CastKind::Reinterpret
                }
                (Heap { .. }, Ref { .. }) => {
                    if let (Heap { inner: ai }, Ref { mutable: _, inner: bi }) = (from, to) {
                        if !type_eq(ai, bi) {
                            self.err(format!("invalid cast: {:?} as {:?}", from, to), sp);
                        }
                    }
                    CastKind::Reinterpret
                }
                (SizedInt { bits: 0, .. }, Ptr { .. }) | (Int, Ptr { .. }) => {
                    if self.in_unsafe == 0 {
                        self.err(
                            "synthesizing a pointer from an integer requires an `unsafe { ... }` \
                             block — this is the one operation that can produce a dangling pointer \
                             the lifetime pass cannot track".to_string(),
                            sp,
                        );
                    }
                    CastKind::Reinterpret
                }
                _ => {
                    if std::mem::discriminant(from) == std::mem::discriminant(to) {
                        CastKind::Identity
                    } else {
                        self.err(format!("invalid cast: {:?} as {:?}", from, to), sp);
                        CastKind::Identity
                    }
                }
            }
        } else {
            // `as?` is gone — checked casts are now spelled `as` and
            // dispatched by the type pair (see int → Enum above).
            self.err("internal: classify_cast called with checked=true after `as?` removal", sp);
            CastKind::Identity
        }
    }

    /// §3 prefix rule: returns true iff a `*from → *to` reinterpret is safe.
    /// Identity (`type_eq(from, to)`) trivially is — that's a mutness-only
    /// adjustment, not a real reinterpret.  Different structs are safe iff
    /// `to`'s field list is a prefix of `from`'s (same names + types + order).
    /// All other type-pair combinations return false and the cast falls
    /// through to the `unsafe` requirement.
    fn is_structural_prefix(&self, from: &HType, to: &HType) -> bool {
        if type_eq(from, to) { return true; }
        let (HType::Struct(from_id), HType::Struct(to_id)) = (from, to) else { return false; };
        let from_info = self.sym.struct_info(*from_id);
        let to_info = self.sym.struct_info(*to_id);
        if to_info.fields.len() > from_info.fields.len() { return false; }
        for (i, tf) in to_info.fields.iter().enumerate() {
            let ff = &from_info.fields[i];
            if ff.name != tf.name { return false; }
            if !type_eq(&ff.ty, &tf.ty) { return false; }
        }
        true
    }

    fn check_capturing_lambda(&mut self,
        ret_ty: &ast::Type,
        params: &[ast::Param],
        captures: &[ast::LambdaCapture],
        body: &ast::LambdaBody,
        sp: Span,
    ) -> HExpr {
        if captures.is_empty() {
            // Should have been lifted at AST level.
            self.err("internal: no-capture lambda not lifted", sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        }
        // 1. Resolve each capture's name to a local in the current scope, getting its type.
        struct ResolvedCap { name: String, mode: char, ty: HType, source: LocalId }
        let mut caps: Vec<ResolvedCap> = Vec::new();
        for c in captures {
            let Some(lid) = self.lookup(&c.name) else {
                self.err(format!("unknown capture `{}`", c.name), c.span);
                continue;
            };
            let lty = self.local(lid).ty.clone();
            // The captured field's type depends on mode:
            //  'v' → copy by value
            //  'r' → store a &T reference (read-only)
            //  'm' → store a &mut T reference
            let cap_ty = match c.mode {
                'v' => lty,
                'r' => HType::Ref { mutable: false, inner: Box::new(lty) },
                'm' => HType::Ref { mutable: true, inner: Box::new(lty) },
                _ => lty,
            };
            caps.push(ResolvedCap { name: c.name.clone(), mode: c.mode, ty: cap_ty, source: lid });
        }

        // 2. Synthesize the env struct.
        // StructId index = current structs in sym.structs + already-synthesized ones in this checker.
        let env_idx = self.sym.structs.len() + self.synth_structs.len();
        let env_struct_id = StructId(env_idx as u32);
        let env_name = format!("__LambdaEnv_{}", env_idx);
        let env_fields: Vec<FieldInfo> = caps.iter().map(|c| FieldInfo {
            name: c.name.clone(),
            ty: c.ty.clone(),
            mut_payload: true,
            default: None,
            is_embed: false,
            span: sp,
        }).collect();
        self.synth_structs.push(StructInfo {
            name: env_name.clone(),
            type_params: Vec::new(),
            template: None,
            template_args: Vec::new(),
            fields: env_fields,
            is_pub: false,
            module_path: Vec::new(),
            span: sp,
            where_bounds: Vec::new(),
        });

        // 3. Resolve lambda's params and ret types in a fresh scope, then typecheck body
        //    with captures bound as fresh locals reading from `env.<name>`.
        let resolved_ret = resolve_type_in(self.sym, ret_ty, &[], &mut self.errors);
        let resolved_params: Vec<HType> = params.iter().map(|p|
            self.resolve_local_ty(&p.ty)
        ).collect();

        // Build a sub-TypeChecker for the lifted function body.
        let lifted_fid = FuncId((self.sym.sigs.len() + self.synth_sigs.len()) as u32);
        let lifted_name = format!("__lambda_cap_{}", env_idx);
        let env_ref_ty = HType::Ref { mutable: false, inner: Box::new(HType::Struct(env_struct_id)) };
        let mut sub = TypeChecker::new_with_logic(self.sym, None);
        // Inherit the enclosing function's imports / has-imports / module path
        // so closure bodies can call the same names that worked in the outer
        // scope (e.g. `sleep_ms` imported from std).
        sub.cur_module = self.cur_module.clone();
        sub.cur_imports = self.cur_imports.clone();
        sub.cur_has_imports = self.cur_has_imports.clone();
        sub.cur_where_bounds = self.cur_where_bounds.clone();
        // Lambda body is lexically inside the caller's `unsafe { }` scope (if any).
        sub.in_unsafe = self.in_unsafe;
        sub.enter_scope();

        // First param: the env reference.
        let env_param_id = sub.fresh_local("__env".to_string(), env_ref_ty.clone(), StorageClass::Param, true, false, sp);
        sub.bind_name("__env", env_param_id);
        // Lambda's params follow.
        let mut lambda_param_ids = Vec::new();
        for (i, p) in params.iter().enumerate() {
            let pty = resolved_params[i].clone();
            let pid = sub.fresh_local(p.name.clone(), pty.clone(), StorageClass::Param, true, false, sp);
            sub.bind_name(&p.name, pid);
            lambda_param_ids.push(pid);
        }
        // Captures are bound as ordinary locals; at codegen we initialize them from __env.
        let mut capture_local_ids: Vec<LocalId> = Vec::new();
        for c in &caps {
            let id = sub.fresh_local(c.name.clone(), c.ty.clone(), StorageClass::Stack, true, false, sp);
            sub.bind_name(&c.name, id);
            capture_local_ids.push(id);
        }

        sub.cur_ret = resolved_ret.clone();
        let body_block: ast::Block = match body {
            ast::LambdaBody::Block(b) => b.clone(),
            ast::LambdaBody::Expr(e) => ast::Block {
                stmts: vec![ast::Stmt::Return(Some((**e).clone()), sp)],
                span: sp,
            },
        };
        let hir_body = sub.check_block(&body_block);
        sub.leave_scope();
        let sub_errors = std::mem::take(&mut sub.errors);
        self.errors.extend(sub_errors);

        // The synthetic HFunc params = [env, lambda_params...].
        let mut params_list: Vec<LocalId> = vec![env_param_id];
        params_list.extend(lambda_param_ids.iter().copied());
        let synth_func = HFunc {
            id: lifted_fid,
            name: lifted_name.clone(),
            params: params_list,
            ret: resolved_ret.clone(),
            locals: sub.locals,
            body: hir_body,
            span: sp,
        };
        // Track the capture local ids so codegen can emit env-extraction at function entry.
        // Encode via a side channel: store on the FuncSig's param_names.
        let mut sig_param_names: Vec<String> = vec!["__env".to_string()];
        for p in params { sig_param_names.push(p.name.clone()); }
        for c in &caps { sig_param_names.push(format!("__capture_{}", c.name)); }
        // FuncSig: caller's perspective is `(env*, lambda_params...)`.
        let mut sig_param_tys: Vec<HType> = vec![env_ref_ty];
        sig_param_tys.extend(resolved_params.clone());
        self.synth_sigs.push(FuncSig {
            name: lifted_name.clone(),
            param_tys: sig_param_tys,
            param_names: sig_param_names,
            ret: resolved_ret.clone(),
            is_extern: false,
            c_name: lifted_name,
            logic: None,
            type_params: Vec::new(),
            is_inline: false,
            is_gate: false,
            is_variadic: false,
            is_pub: false,
            module_path: self.cur_module.clone(),
            imports: self.cur_imports.clone(),
            has_imports: self.cur_has_imports.clone(),
            where_bounds: self.cur_where_bounds.clone(),
        });
        self.synth_funcs.push(synth_func);

        // Build the closure HExpr: env-init values from the current scope's locals.
        let env_inits: Vec<HExpr> = caps.iter().map(|c| {
            let src_ty = self.local(c.source).ty.clone();
            let src = HExpr { kind: HExprKind::Local(c.source), ty: src_ty.clone(), span: sp };
            match c.mode {
                'r' => HExpr { kind: HExprKind::AddrOfRef { mutable: false, place: Box::new(src) }, ty: c.ty.clone(), span: sp },
                'm' => HExpr { kind: HExprKind::AddrOfRef { mutable: true, place: Box::new(src) }, ty: c.ty.clone(), span: sp },
                _ => src,
            }
        }).collect();

        let fn_ty = HType::FnPtr { ret: Box::new(resolved_ret), params: resolved_params };
        HExpr {
            kind: HExprKind::Closure {
                lifted: lifted_fid,
                env_struct: env_struct_id,
                env_values: env_inits,
                capture_lids: capture_local_ids,
            },
            ty: fn_ty,
            span: sp,
        }
    }

    fn check_for_each(&mut self,
        var_ty: &ast::Type, var_name: &str,
        src: &ast::Expr, body: &ast::Block, sp: Span,
    ) -> HStmt {
        let declared = self.resolve_local_ty(var_ty);
        let src_probe = self.check_expr(src, None);

        // User-iterator path: when `src` is a struct value with a `next` method
        // returning `Option<elem_ty>`, lower the for-each to an AST-level while
        // loop and re-typecheck:
        //   { mut Src __it = <src>;
        //     while (true) {
        //       Option<T> __cur = (&mut __it).next();
        //       match (__cur) {
        //         Some{value} { <body with var_name bound to value> }
        //         None { break; }
        //       }
        //     } }
        // No language-level attr is required; "has a next() returning Option" is
        // the iterator protocol.
        if let Some(sid) = struct_id_of(&src_probe.ty) {
            if self.struct_has_iter_method(sid, &declared) {
                let struct_name = self.sym.struct_info(sid).name.clone();
                let elem_ty_ast = var_ty.clone();
                let desugared = build_iter_desugaring(&struct_name, &elem_ty_ast, var_name, src, body, sp);
                // The desugaring references `Option<T>` from `std`.  Inject a
                // synthetic `import std.Option;` for the duration of this body
                // so user code isn't forced to write the import just because
                // they iterate a custom type.  Same model `for x in slice`
                // uses (no `import std.SliceLen;` required).
                self.cur_imports.push((vec!["std".to_string()], "Option".to_string()));
                let result = self.check_block_stmt(desugared);
                self.cur_imports.pop();
                return result;
            }
        }

        self.enter_scope();
        let id_var = self.fresh_local_with_tls(var_name.to_string(), declared.clone(), StorageClass::Stack, true, true, false, sp);
        self.bind_name(var_name, id_var);
        let body_h = self.check_block(body);
        self.leave_scope();
        HStmt::ForEach { var: id_var, src: src_probe, body: body_h, span: sp }
    }

    /// Check an already-built `ast::Block` and return it wrapped as an HStmt::Block.
    fn check_block_stmt(&mut self, b: ast::Block) -> HStmt {
        let h = self.check_block(&b);
        HStmt::Block(h)
    }

    fn empty_block_stmt(&self, sp: Span) -> HStmt {
        HStmt::Block(HBlock { stmts: Vec::new(), heap_to_free: Vec::new(), ptr_nulls: Vec::new(), span: sp })
    }

    /// Tier-2 compile-time reflection: `inline for (f in fields(value)) { body }`.
    /// The body is unrolled once per field of `value`'s struct type, with
    /// `f.name`/`f.value`/`f.index`/`f.type` substituted per field, then checked
    /// as an ordinary block.  Lowers entirely to `HStmt::Block`, so codegen never
    /// sees an inline-for.
    fn check_inline_for(&mut self, var_name: &str, iter: &ast::Expr, body: &ast::Block, sp: Span) -> HStmt {
        // The iterable must be `fields(receiver)`.
        let recv = match iter {
            ast::Expr::Call { callee, args, .. } => match callee.as_ref() {
                ast::Expr::Ident(fname, _) if fname == "fields" && args.len() == 1 => Some(&args[0]),
                _ => None,
            },
            _ => None,
        };
        let Some(recv) = recv else {
            self.err("`inline for` expects `fields(value)` as its iterable", sp);
            return self.empty_block_stmt(sp);
        };
        // The receiver is re-read once per field (as `recv.<field>`), so restrict
        // it to a plain variable to avoid duplicating side effects.
        if !matches!(recv, ast::Expr::Ident(_, _)) {
            self.err("`fields(...)` argument must be a variable", recv.span());
            return self.empty_block_stmt(sp);
        }

        let recv_h = self.check_expr(recv, None);
        let ty = concretize_generic_patterns(&recv_h.ty.subst(&self.subst), self.sym);
        let Some(sid) = struct_id_of(&ty) else {
            // A still-generic receiver means we're checking the generic template
            // before monomorphization; the real unroll happens once the function
            // is instantiated with a concrete type.  A concrete non-struct is a
            // genuine error.
            if !has_tyvar(&ty) {
                self.err("`inline for` over `fields(...)` requires a struct value", recv.span());
            }
            return self.empty_block_stmt(sp);
        };

        let fields = self.sym.struct_info(sid).fields.clone();
        let mut stmts: Vec<ast::Stmt> = Vec::with_capacity(fields.len());
        for (i, fld) in fields.iter().enumerate() {
            let type_str = htype_display(self.sym, &fld.ty);
            let mut b = body.clone();
            let ctx = FieldCtx { var: var_name, recv, fname: &fld.name, index: i as i64, tystr: &type_str };
            rewrite_field_refs(&mut b, &ctx);
            // Each field's body gets its own block so `let` bindings don't clash
            // across iterations.
            stmts.push(ast::Stmt::Block(b));
        }
        let unrolled = ast::Block { stmts, span: sp };
        self.check_block_stmt(unrolled)
    }

    /// Is there a `next` method registered for the given struct whose return is
    /// `Option<elem_ty>`?  The iterator protocol is: receiver = `&mut Self`,
    /// return = `Option<T>` where T matches the loop variable's declared type.
    fn struct_has_iter_method(&self, sid: StructId, elem_ty: &HType) -> bool {
        let key = self.sym.struct_info(sid).name.clone();
        let expected_enum_name = format!("Option__{}", elem_ty.key());
        for s in &self.sym.sigs {
            if s.name != "next" { continue; }
            if s.param_tys.len() != 1 { continue; }
            let Some(first_sid) = struct_id_of(&s.param_tys[0]) else { continue; };
            if self.sym.struct_info(first_sid).name != key { continue; }
            if let HType::Enum(eid) = &s.ret {
                if self.sym.enum_info(*eid).name == expected_enum_name { return true; }
            }
        }
        false
    }

    fn check_for_range(&mut self,
        var_ty: &ast::Type, var_name: &str,
        start: &ast::Expr, end: &ast::Expr, inclusive: bool,
        body: &ast::Block, sp: Span,
    ) -> HStmt {
        // Lower to native C-style `for (init; cond; step) body` so `continue` triggers `step`.
        let declared = self.resolve_local_ty(var_ty);

        self.enter_scope();
        let start_h = self.check_expr_coerce(start, &declared);
        let id_var = self.fresh_local(var_name.to_string(), declared.clone(), StorageClass::Stack, true, true, sp);
        self.bind_name(var_name, id_var);
        let end_h = self.check_expr_coerce(end, &declared);
        let var_expr = HExpr { kind: HExprKind::Local(id_var), ty: declared.clone(), span: sp };
        let op = if inclusive { HBinOp::Le } else { HBinOp::Lt };
        let one = HExpr { kind: HExprKind::LitInt(1), ty: HType::Int, span: sp };
        let plus = HExpr {
            kind: HExprKind::Bin { op: HBinOp::Add, lhs: Box::new(var_expr.clone()), rhs: Box::new(one) },
            ty: declared.clone(), span: sp,
        };
        let step = HStmt::Assign { op: HAssignOp::Assign, place: var_expr.clone(), value: plus, span: sp };

        // A constant or `slice.len` end is inlined directly into the condition
        // (`i < N` / `i < s.len`) rather than hoisted into a `__end` local, so
        // codegen can recognize the counted-loop bound and drop bounds checks on
        // `arr[i]` (N <= len) or `s[i]` (same slice).  Both ends are cheap and
        // (for the elided case) loop-invariant, so re-evaluating them is fine.
        let inline_bound = matches!(end_h.kind, HExprKind::LitInt(_) | HExprKind::SliceLen(_));

        let body_h = self.check_block(body);
        self.leave_scope();
        let init = HStmt::Let { local: id_var, init: start_h, span: sp };

        if inline_bound {
            let cond = HExpr {
                kind: HExprKind::Bin { op, lhs: Box::new(var_expr.clone()), rhs: Box::new(end_h) },
                ty: HType::Bool, span: sp,
            };
            HStmt::ForC { init: Box::new(init), cond, step: Box::new(step), body: body_h, span: sp }
        } else {
            let id_end = self.fresh_local("__end".to_string(), declared.clone(), StorageClass::Stack, true, true, sp);
            let end_expr = HExpr { kind: HExprKind::Local(id_end), ty: declared.clone(), span: sp };
            let cond = HExpr {
                kind: HExprKind::Bin { op, lhs: Box::new(var_expr.clone()), rhs: Box::new(end_expr) },
                ty: HType::Bool, span: sp,
            };
            // Wrap: { let __end = end; for (i = start; i op __end; i += 1) body }
            HStmt::Block(HBlock {
                stmts: vec![
                    HStmt::Let { local: id_end, init: end_h, span: sp },
                    HStmt::ForC { init: Box::new(init), cond, step: Box::new(step), body: body_h, span: sp },
                ],
                heap_to_free: Vec::new(),
                ptr_nulls: Vec::new(),
                span: sp,
            })
        }
    }

    fn check_match(&mut self, scrutinee: &ast::Expr, arms: &[ast::MatchArm], as_stmt: bool, sp: Span, expected: Option<&HType>) -> HExpr {
        let scrut_h = self.check_expr(scrutinee, None);
        let scrut_ty = scrut_h.ty.clone();

        // What enum (if any) is the scrutinee?
        let enum_id = match self.peel_to_enum(&scrut_ty) {
            Some(id) => Some(id),
            None => None,
        };

        let mut h_arms: Vec<HMatchArm> = Vec::new();
        let mut result_ty: Option<HType> = None;
        let mut covered_variants: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut has_else = false;

        for arm in arms {
            self.enter_scope();
            let (kind, scrut_binding) = self.lower_pattern(&arm.pattern, enum_id, &scrut_ty, sp);
            // Track exhaustiveness for variant patterns on enum scrutinee.
            if let HArmKind::Variant { variant, .. } = &kind {
                covered_variants.insert(*variant);
            }
            if matches!(kind, HArmKind::Else) {
                has_else = true;
            }
            let guard_h = arm.guard.as_ref().map(|g| self.check_expr_coerce(g, &HType::Bool));
            let (body, value) = match &arm.body {
                ast::ArmBody::Expr(e) => {
                    let h = self.check_expr(e, result_ty.as_ref());
                    let block = HBlock { stmts: Vec::new(), heap_to_free: Vec::new(), ptr_nulls: Vec::new(), span: arm.span };
                    (block, Some(h))
                }
                ast::ArmBody::Block(b) => {
                    // When the arm's result type is known (from a prior arm or
                    // the surrounding context) and is a plain non-owning value,
                    // route `yield` through a synthetic result local.  This
                    // captures yields nested inside `if`/`while`/block statements,
                    // not just a trailing `yield` (which `extract_yield_value`
                    // alone handles).  Owning results keep the old path: a
                    // synthetic local would be freed at arm scope *and* flow out
                    // as the value (double free), since a bare-local arm value is
                    // not tracked as a move.
                    let arm_ty = result_ty.clone().or_else(|| expected.cloned());
                    let target = if as_stmt { None } else {
                        arm_ty.as_ref()
                            .filter(|t| !crate::lifetime::ty_owns_heap(self.sym, t))
                            .and_then(|t| self.zero_expr(t, arm.span).map(|z| (t.clone(), z)))
                    };
                    if let Some((t, init)) = target {
                        let yv = self.fresh_local("__yield".to_string(), t.clone(), StorageClass::Stack, true, true, arm.span);
                        self.yield_target.push((yv, t.clone()));
                        let mut block = self.check_block(b);
                        self.yield_target.pop();
                        block.stmts.insert(0, HStmt::Let { local: yv, init, span: arm.span });
                        let value = HExpr { kind: HExprKind::Local(yv), ty: t, span: arm.span };
                        (block, Some(value))
                    } else {
                        let block = self.check_block(b);
                        // Statement-form matches don't extract a value from the
                        // arm body — extracting would double-emit the last stmt.
                        let value = if as_stmt { None } else { self.extract_yield_value(&block) };
                        (block, value)
                    }
                }
            };
            if let Some(v) = &value {
                if result_ty.is_none() { result_ty = Some(v.ty.clone()); }
            }
            self.leave_scope();
            h_arms.push(HMatchArm { kind, guard: guard_h, body, value, scrut_binding });
        }

        // Exhaustiveness check for enum scrutinees.
        if let Some(eid) = enum_id {
            if !has_else {
                let info = self.sym.enum_info(eid);
                let n = info.variants.len();
                let all_covered = (0..n).all(|i| covered_variants.contains(&i));
                if !all_covered {
                    let missing: Vec<String> = (0..n).filter(|i| !covered_variants.contains(i))
                        .map(|i| info.variants[i].name.clone()).collect();
                    self.err(format!("non-exhaustive match: missing variants {}", missing.join(", ")), sp);
                }
            }
        } else {
            // Primitive scrutinees require an else.
            if !has_else {
                self.err("non-exhaustive match: require `else` for primitive scrutinee", sp);
            }
        }

        let ty = result_ty.unwrap_or(HType::Unit);
        HExpr {
            kind: HExprKind::Match {
                scrutinee: Box::new(scrut_h),
                arms: h_arms,
                result_ty: ty.clone(),
            },
            ty,
            span: sp,
        }
    }

    fn peel_to_enum(&self, t: &HType) -> Option<EnumId> {
        match t {
            HType::Enum(id) => Some(*id),
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::Heap { inner } => self.peel_to_enum(inner),
            _ => None,
        }
    }

    /// Returns (HArmKind, optional scrut-binding local).
    fn lower_pattern(&mut self, pat: &ast::Pattern, enum_id: Option<EnumId>, scrut_ty: &HType, sp: Span)
        -> (HArmKind, Option<LocalId>)
    {
        match pat {
            ast::Pattern::Else(_) => (HArmKind::Else, None),
            ast::Pattern::Null(_) => (HArmKind::Null, None),
            ast::Pattern::Lit(l, _) => {
                let h = self.check_lit(l, None, sp);
                (HArmKind::Lit(h), None)
            }
            ast::Pattern::Ident(name, _) => {
                // If matches an enum variant of the inferred enum, treat as tag-only variant.
                if let Some(eid) = enum_id {
                    let info = self.sym.enum_info(eid).clone();
                    if let Some(vi) = info.variant_index(name) {
                        let n = info.variants[vi].fields.len();
                        let bindings: Vec<Option<LocalId>> = (0..n).map(|_| None).collect();
                        let lit_checks: Vec<Option<HExpr>> = (0..n).map(|_| None).collect();
                        return (HArmKind::Variant { enum_id: eid, variant: vi, bindings, lit_checks }, None);
                    }
                }
                // Otherwise treat as a variable binding to the scrutinee value.
                let local = self.fresh_local(name.clone(), scrut_ty.clone(), StorageClass::Stack, true, true, sp);
                self.bind_name(name, local);
                (HArmKind::Else, Some(local))
            }
            ast::Pattern::Variant { enum_name: _, variant, .. } => {
                let Some(eid) = enum_id else {
                    self.err("variant pattern requires an enum scrutinee", sp);
                    return (HArmKind::Else, None);
                };
                let info = self.sym.enum_info(eid).clone();
                let Some(vi) = info.variant_index(variant) else {
                    self.err(format!("enum has no variant `{}`", variant), sp);
                    return (HArmKind::Else, None);
                };
                let n = info.variants[vi].fields.len();
                let bindings: Vec<Option<LocalId>> = (0..n).map(|_| None).collect();
                let lit_checks: Vec<Option<HExpr>> = (0..n).map(|_| None).collect();
                (HArmKind::Variant { enum_id: eid, variant: vi, bindings, lit_checks }, None)
            }
            ast::Pattern::VariantDestructure { enum_name: _, variant, fields, .. } => {
                let Some(eid) = enum_id else {
                    self.err("variant pattern requires an enum scrutinee", sp);
                    return (HArmKind::Else, None);
                };
                let info = self.sym.enum_info(eid).clone();
                let Some(vi) = info.variant_index(variant) else {
                    self.err(format!("enum has no variant `{}`", variant), sp);
                    return (HArmKind::Else, None);
                };
                let vinfo = info.variants[vi].clone();
                let mut bindings: Vec<Option<LocalId>> = vec![None; vinfo.fields.len()];
                let mut lit_checks: Vec<Option<HExpr>> = (0..vinfo.fields.len()).map(|_| None).collect();
                for pf in fields {
                    let Some((fi, finfo)) = vinfo.fields.iter().enumerate().find(|(_, f)| f.name == pf.field) else {
                        self.err(format!("variant has no field `{}`", pf.field), pf.span);
                        continue;
                    };
                    if let Some(lit) = &pf.literal {
                        let h = self.check_lit(lit, Some(&finfo.ty), pf.span);
                        lit_checks[fi] = Some(h);
                    } else {
                        let binding_name = pf.binding.clone().unwrap_or_else(|| pf.field.clone());
                        let local = self.fresh_local(binding_name.clone(), finfo.ty.clone(), StorageClass::Stack, true, false, pf.span);
                        self.bind_name(&binding_name, local);
                        bindings[fi] = Some(local);
                    }
                }
                (HArmKind::Variant { enum_id: eid, variant: vi, bindings, lit_checks }, None)
            }
            ast::Pattern::Or(_, _) => {
                self.err("or-patterns not supported yet", sp);
                (HArmKind::Else, None)
            }
        }
    }

    /// A zero/default initializer for `t`, used to declare the synthetic result
    /// local of a value-producing match arm.  Returns `None` for types that have
    /// no simple literal zero (enums, structs, arrays, refs, closures): those
    /// keep the trailing-`yield` extraction path instead of a synthetic local.
    fn zero_expr(&self, t: &HType, span: Span) -> Option<HExpr> {
        let kind = match t {
            HType::Int | HType::SizedInt { .. } => HExprKind::LitInt(0),
            HType::Float | HType::SizedFloat { .. } => HExprKind::LitFloat(0.0),
            HType::Bool => HExprKind::LitBool(false),
            HType::Char => HExprKind::LitChar('\0'),
            HType::Str => HExprKind::LitStr(String::new()),
            HType::Ptr { .. } | HType::RawPtr { .. } => HExprKind::LitNull,
            _ => return None,
        };
        Some(HExpr { kind, ty: t.clone(), span })
    }

    /// If a block has trailing `yield expr;` or expression-statement, extract that as the value.
    fn extract_yield_value(&self, block: &HBlock) -> Option<HExpr> {
        // Simple heuristic: last statement, if it's ExprStmt, becomes the block's value.
        if let Some(HStmt::ExprStmt(e)) = block.stmts.last() {
            return Some(e.clone());
        }
        None
    }

    fn check_variant_ctor(
        &mut self,
        enum_name: &str,
        variant: &str,
        fields: &[(String, ast::Expr)],
        expected: Option<&HType>,
        sp: Span,
    ) -> HExpr {
        // If the context wants a specific monomorphized enum (e.g. `Option<int>`),
        // prefer that EnumId — the template's fields contain `TyVar`s and would
        // fail to coerce against concrete arg expressions.
        let (mut eid, mut info) = if let Some(HType::Enum(want_eid)) = expected {
            let want_info = self.sym.enum_info(*want_eid).clone();
            let template_name = want_info.name.split("__").next().unwrap_or(&want_info.name);
            if template_name == enum_name {
                (*want_eid, want_info)
            } else if let Some((e, i)) = self.sym.enum_by_name(enum_name) {
                (e, i.clone())
            } else {
                self.err(format!("unknown enum `{}`", enum_name), sp);
                return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
            }
        } else if let Some((e, i)) = self.sym.enum_by_name(enum_name) {
            (e, i.clone())
        } else {
            self.err(format!("unknown enum `{}`", enum_name), sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        };
        let Some(vi) = info.variant_index(variant) else {
            self.err(format!("enum `{}` has no variant `{}`", enum_name, variant), sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        };
        // When the resolved enum is still a generic *template* (expected did not
        // pin a concrete instantiation), infer the instantiation from the field
        // values plus the enclosing function's type-param substitution.  This is
        // what makes `propagate Result.Err { err = err }` work inside an inline
        // generic: `err` fixes E, and the surrounding monomorphization fixes the
        // remaining params (e.g. T) by name.
        if !info.type_params.is_empty() {
            fn is_concrete(t: &HType) -> bool {
                match t {
                    HType::TyVar(_) => false,
                    HType::GenericPattern { args, .. } => args.iter().all(is_concrete),
                    HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. }
                    | HType::OwnPtr { inner, .. } | HType::Heap { inner } => is_concrete(inner),
                    HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => is_concrete(elem),
                    HType::FnPtr { ret, params } => is_concrete(ret) && params.iter().all(is_concrete),
                    _ => true,
                }
            }
            let tmpl_v = info.variants[vi].clone();
            let mut env: std::collections::HashMap<String, HType> = std::collections::HashMap::new();
            for (fname, fv) in fields {
                if let Some((_, finfo)) = tmpl_v.fields.iter().enumerate().find(|(_, f)| f.name == *fname) {
                    let h = self.check_expr(fv, None);
                    unify_with_sym(&finfo.ty, &h.ty, &mut env, self.sym);
                }
            }
            for tp in &info.type_params {
                if !env.contains_key(tp) {
                    if let Some(t) = self.subst.get(tp) { env.insert(tp.clone(), t.clone()); }
                }
            }
            let args: Vec<HType> = info.type_params.iter()
                .map(|tp| concretize_generic_patterns(&env.get(tp).cloned().unwrap_or(HType::TyVar(tp.clone())), self.sym))
                .collect();
            if args.iter().all(is_concrete) {
                let pat = HType::GenericPattern { template_name: enum_name.to_string(), args, is_enum: true };
                if let HType::Enum(mono) = concretize_generic_patterns(&pat, self.sym) {
                    eid = mono;
                    info = self.sym.enum_info(mono).clone();
                }
            }
        }
        let vi = info.variant_index(variant).unwrap_or(vi);
        let v = &info.variants[vi];
        let mut provided: Vec<Option<HExpr>> = (0..v.fields.len()).map(|_| None).collect();
        for (fname, fv) in fields {
            let Some((idx, finfo)) = v.fields.iter().enumerate().find(|(_, f)| f.name == *fname) else {
                self.err(format!("variant `{}.{}` has no field `{}`", enum_name, variant, fname), sp);
                continue;
            };
            let h = self.check_expr_coerce(fv, &finfo.ty);
            provided[idx] = Some(h);
        }
        let mut out = Vec::new();
        for (i, f) in v.fields.iter().enumerate() {
            if let Some(h) = provided[i].take() {
                out.push((i, h));
            } else if matches!(f.ty, HType::Ptr { .. }) {
                let null = HExpr { kind: HExprKind::LitNull, ty: f.ty.clone(), span: sp };
                out.push((i, null));
            } else {
                self.err(format!("missing field `{}` in variant `{}.{}`", f.name, enum_name, variant), sp);
            }
        }
        HExpr {
            kind: HExprKind::VariantCtor { enum_id: eid, variant: vi, fields: out },
            ty: HType::Enum(eid),
            span: sp,
        }
    }

    fn check_struct_lit(&mut self, name: Option<&str>, fields: &[(String, ast::Expr)], expected: Option<&HType>, sp: Span) -> HExpr {
        // Peel pointer/ref layers off a (possibly concretized) expected type down
        // to a Struct id.
        fn peel_to_struct_id(t: &HType) -> Option<StructId> {
            match t {
                HType::Struct(id) => Some(*id),
                HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::OwnPtr { inner, .. }
                | HType::RawPtr { inner, .. } | HType::Heap { inner } => peel_to_struct_id(inner),
                _ => None,
            }
        }
        // A concrete instantiation suggested by the expected type (e.g.
        // `alloc Node { .. }` landing in `own *Node<int>` -> Node<int>).
        let expected_inst = expected
            .map(|e| concretize_generic_patterns(e, self.sym))
            .as_ref()
            .and_then(peel_to_struct_id);

        // Determine struct id either from `name` or expected.
        let sid = if let Some(n) = name {
            let named = self.sym.struct_by_name(n).map(|(id, _)| id);
            // When `n` is a generic *template* and the expected type is a concrete
            // instantiation of it, build the instantiation - so field types are the
            // concrete ones (`int`), not the template's TyVars.  This makes
            // `alloc Node { .. }` work in a `own *Node<int>` slot.
            let named_is_template = named.map_or(false, |id| !self.sym.struct_info(id).type_params.is_empty());
            let inst = expected_inst.filter(|eid| self.sym.struct_info(*eid).template.as_deref() == Some(n));
            if named_is_template { inst.or(named) } else { named }
        } else {
            expected_inst
        };
        let Some(sid) = sid else {
            self.err("cannot infer struct type for struct literal", sp);
            return HExpr { kind: HExprKind::LitUnit, ty: HType::Unit, span: sp };
        };
        let info = self.sym.struct_info(sid).clone();
        let mut provided = vec![None; info.fields.len()];
        for (fname, fv) in fields {
            let Some((idx, finfo)) = info.fields.iter().enumerate().find(|(_, f)| &f.name == fname) else {
                self.err(format!("struct `{}` has no field `{}`", info.name, fname), sp);
                continue;
            };
            let h = self.check_expr_coerce(fv, &finfo.ty);
            provided[idx] = Some(h);
        }
        // Fill defaults / required-not-supplied checks
        let mut out = Vec::new();
        for (i, f) in info.fields.iter().enumerate() {
            if let Some(h) = provided[i].take() {
                out.push((i, h));
                continue;
            }
            // Default: pointer fields default to null, otherwise use literal default or error
            match &f.ty {
                HType::Ptr { .. } => {
                    let null = HExpr { kind: HExprKind::LitNull, ty: f.ty.clone(), span: sp };
                    out.push((i, null));
                }
                _ => {
                    self.err(format!("missing field `{}` (no default available)", f.name), sp);
                }
            }
        }
        HExpr { kind: HExprKind::Struct { id: sid, fields: out }, ty: HType::Struct(sid), span: sp }
    }

    fn check_array_lit(&mut self, elems: &[ast::Expr], expected: Option<&HType>, sp: Span) -> HExpr {
        let elem_target = expected.and_then(|t| match t {
            HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => Some((**elem).clone()),
            HType::Heap { inner } => match inner.as_ref() {
                HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => Some((**elem).clone()),
                _ => None,
            },
            _ => None,
        });
        let mut hs = Vec::new();
        let mut tyo: Option<HType> = elem_target.clone();
        for e in elems {
            let h = if let Some(t) = &tyo { self.check_expr_coerce(e, t) } else { self.check_expr(e, None) };
            if tyo.is_none() { tyo = Some(h.ty.clone()); }
            hs.push(h);
        }
        let elem_ty = tyo.unwrap_or(HType::Int);
        let result = match expected {
            Some(HType::Vec { .. }) | Some(HType::Heap { .. }) => HType::Vec { elem: Box::new(elem_ty.clone()) },
            Some(HType::Slice { mutable, .. }) => HType::Slice { mutable: *mutable, elem: Box::new(elem_ty.clone()) },
            _ => HType::Array { len: hs.len() as i64, elem: Box::new(elem_ty.clone()) },
        };
        HExpr { kind: HExprKind::ArrayLit(hs), ty: result, span: sp }
    }

    // ---- coercion ----

    fn coerce(&mut self, e: HExpr, target: &HType) -> HExpr {
        if type_eq(&e.ty, target) { return e; }

        // Implicit reborrow: `&mut &mut T` → `&mut T` (and `&&T` → `&T`).  This
        // is what makes `func(&mut g)` work when `g` is already `&mut T` inside
        // a helper - users write the borrow naturally and the compiler peels
        // the redundant layer.
        if let (HType::Ref { mutable: pm, inner: pi }, HType::Ref { mutable: am, inner: ai }) = (target, &e.ty) {
            if pm == am {
                if let HType::Ref { mutable: imut, inner: iinner } = ai.as_ref() {
                    if pm == imut && type_eq(pi, iinner) {
                        if let HExprKind::AddrOfRef { place, .. } = &e.kind {
                            // Peel `&mut <place>` where place is already `&mut T`
                            // - the inner place IS the borrow we want to pass.
                            return (**place).clone();
                        }
                        return HExpr { ty: target.clone(), ..e };
                    }
                }
            }
        }

        // Auto-deref &T -> T when target is the referent type.
        if let HType::Ref { inner, .. } = &e.ty {
            if type_eq(inner, target) {
                return self.auto_deref(e);
            }
        }

        // null → *T (any pointer)
        if matches!(e.ty, HType::NullT) && matches!(target, HType::Ptr { .. }) {
            return HExpr { ty: target.clone(), ..e };
        }
        // `own *char` → `string` — read-only view of a heap-allocated NUL-terminated buffer.
        if matches!(target, HType::Str)
            && matches!(&e.ty, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char)) {
            return HExpr { ty: HType::Str, ..e };
        }
        // null → raw *T (raw pointers are also nullable)
        if matches!(e.ty, HType::NullT) && matches!(target, HType::RawPtr { .. }) {
            return HExpr { ty: target.clone(), ..e };
        }
        // null → own *T (nullable owning pointer accepts null)
        if matches!(e.ty, HType::NullT) && matches!(target, HType::OwnPtr { .. }) {
            return HExpr { ty: target.clone(), ..e };
        }
        // alloc value (HType::Heap = strict owning) → own *T (nullable owning).
        // This lets `own *T x = alloc T{};` work just like `own &T x = alloc T{};`.
        if let (HType::Heap { inner: ai }, HType::OwnPtr { mutable: _, inner: bi }) = (&e.ty, target) {
            if type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        // own *T → *T (downgrade ownership to a non-owning view).
        if let (HType::OwnPtr { mutable: am, inner: ai }, HType::Ptr { mutable: bm, inner: bi }) = (&e.ty, target) {
            if (*am || !*bm) && type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        // own &T (= Heap) → *T (downgrade strict-owner to nullable view).
        if let (HType::Heap { inner: ai }, HType::Ptr { mutable: _bm, inner: bi }) = (&e.ty, target) {
            if type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        // own *T → raw *T  /  own &T → raw *T  (drop tracking; observation still
        // requires `unsafe { }`, but the conversion itself is safe-direction).
        if let (HType::OwnPtr { mutable: am, inner: ai }, HType::RawPtr { mutable: bm, inner: bi }) = (&e.ty, target) {
            if (*am || !*bm) && type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        if let (HType::Heap { inner: ai }, HType::RawPtr { mutable: _bm, inner: bi }) = (&e.ty, target) {
            if type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        // &T / &mut T → *T  (drop borrow tracking; same address, alias still
        // depends on the borrow's source via the Q-D dep chain).
        if let (HType::Ref { mutable: am, inner: ai }, HType::Ptr { mutable: bm, inner: bi }) = (&e.ty, target) {
            if (*am || !*bm) && type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        // Note: there is intentionally NO implicit `own *T → &T` / `*T → &T`
        // coercion.  Those sources are NULLABLE — the conversion has a null
        // obligation to discharge — so users write the explicit form at the
        // source site:
        //
        //     &Box b = &(owner!);   // unwrap (proves non-null) + borrow
        //
        // The unifying rule is: `!` exists only to discharge a null obligation.
        // For non-nullable sources (own &T, &T, &mut T) the conversion to
        // `&T` is just a retype — see the `Heap → Ref` arm below.
        //
        // own &T (= Heap, non-null) → &T  /  &mut T  — safe retype, no proof.
        // Mutability check: an immutable owner can't produce `&mut T`.
        if let (HType::Heap { inner: ai }, HType::Ref { mutable: _bm, inner: bi }) = (&e.ty, target) {
            if type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        // Rust<T> → *T (any inner type the target wants).  Lossy at the type
        // level — we trust the bridge-generated extern signatures.
        if let (HType::RustOpaque(_), HType::Ptr { .. }) = (&e.ty, target) {
            return HExpr { ty: target.clone(), ..e };
        }
        // own &T (= Heap) → own *T (loosen strict to nullable).
        if let (HType::Heap { inner: ai }, HType::OwnPtr { mutable: _, inner: bi }) = (&e.ty, target) {
            if type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        // *T → raw *T (safe direction: handing a tracked pointer to a raw-typed slot
        // is always fine, since we're discarding tracking, not synthesizing it).
        if let (HType::Ptr { mutable: am, inner: ai }, HType::RawPtr { mutable: bm, inner: bi }) = (&e.ty, target) {
            if (*am || !*bm) && type_eq(ai, bi) {
                return HExpr { ty: target.clone(), ..e };
            }
        }
        // &mut T → &T (drop write capability)
        if let (HType::Ref { mutable: true, inner: ai }, HType::Ref { mutable: false, inner: bi }) = (&e.ty, target) {
            if type_eq(ai, bi) {
                let mut e = e; e.ty = target.clone();
                return HExpr { kind: HExprKind::DropWrite(Box::new(e.clone())), ..e };
            }
        }
        // *T → *const T
        if let (HType::Ptr { mutable: true, inner: ai }, HType::Ptr { mutable: false, inner: bi }) = (&e.ty, target) {
            if type_eq(ai, bi) {
                let mut e = e; e.ty = target.clone();
                return HExpr { kind: HExprKind::DropWrite(Box::new(e.clone())), ..e };
            }
        }
        // []mut T → []T
        if let (HType::Slice { mutable: true, elem: ai }, HType::Slice { mutable: false, elem: bi }) = (&e.ty, target) {
            if type_eq(ai, bi) {
                let mut e = e; e.ty = target.clone();
                return HExpr { kind: HExprKind::DropWrite(Box::new(e.clone())), ..e };
            }
        }
        // Array → slice
        if let (HType::Array { len, elem: ai }, HType::Slice { mutable: _bm, elem: bi }) = (&e.ty.clone(), target) {
            if type_eq(ai, bi) {
                let len = *len;
                let span = e.span;
                return HExpr { kind: HExprKind::ArrayToSlice { base: Box::new(e), len }, ty: target.clone(), span };
            }
        }
        // Heap T -> ... only via move; otherwise allow heap T binding-context to accept value T
        // We handle this on the let-binding side: `heap T x = T-valued expr`.
        if let HType::Heap { inner } = target {
            if type_eq(&e.ty, inner) {
                let mut e = e; e.ty = target.clone();
                return e;
            }
        }
        // Same for `own *T x = T-valued expr` — the value gets heap-allocated implicitly.
        if let HType::OwnPtr { inner, .. } = target {
            if type_eq(&e.ty, inner) {
                let mut e = e; e.ty = target.clone();
                return e;
            }
        }
        // Int ↔ Float numeric promotion in const expressions: only at lit level, handled in check_lit.
        // Otherwise allow implicit Int→Float? Spec requires `as` so no.
        // Identity by structural sub-match for refs/ptrs where mutability is identical
        if type_eq(&e.ty, target) { return e; }

        // Otherwise, type mismatch — report.
        self.err(format!("type mismatch: expected {}, got {}", type_str(target), type_str(&e.ty)), e.span);
        // keep going with `target` to suppress cascade
        HExpr { ty: target.clone(), ..e }
    }
}


fn bin_to_hir(op: ast::BinOp) -> HBinOp {
    match op {
        ast::BinOp::Add => HBinOp::Add,
        ast::BinOp::Sub => HBinOp::Sub,
        ast::BinOp::Mul => HBinOp::Mul,
        ast::BinOp::Div => HBinOp::Div,
        ast::BinOp::Mod => HBinOp::Mod,
        ast::BinOp::Eq => HBinOp::Eq,
        ast::BinOp::Ne => HBinOp::Ne,
        ast::BinOp::Lt => HBinOp::Lt,
        ast::BinOp::Le => HBinOp::Le,
        ast::BinOp::Gt => HBinOp::Gt,
        ast::BinOp::Ge => HBinOp::Ge,
        ast::BinOp::And => HBinOp::And,
        ast::BinOp::Or => HBinOp::Or,
        ast::BinOp::BitAnd => HBinOp::BitAnd,
        ast::BinOp::BitOr => HBinOp::BitOr,
        ast::BinOp::BitXor => HBinOp::BitXor,
        ast::BinOp::Shl => HBinOp::Shl,
        ast::BinOp::Shr => HBinOp::Shr,
    }
}

/// Build an AST-level desugaring of `for (T x in source) { body }` when `source`
/// is a user iterator (struct with a `next() -> Option<T>` method).  The result
/// is a Block that the caller re-typechecks.
///
/// Shape:
///   { mut Src __it = src;
///     mut bool __broke = false;
///     while (true) {
///       Option<T> __cur = (&mut __it).next();
///       mut bool __none = false;
///       match (__cur) {
///         Some{value} { let T x = value; <body, with break rewritten> }
///         None        { __none = true; }
///       }
///       if (__none) { break; }
///       if (__broke) { break; }
///     } }
///
/// Top-level `break` in the user body is rewritten to
/// `{ __broke = true; break; }` so the inner break exits the match's
/// do-while-0 wrapper AND the outer check exits the while.  `continue` and
/// `return` work naturally — continue falls through both flag checks to the
/// next iteration, return exits the C function entirely.  Breaks/continues
/// inside nested loops in the body are left untouched.
fn build_iter_desugaring(
    struct_name: &str,
    elem_ty: &ast::Type,
    var_name: &str,
    src: &ast::Expr,
    body: &ast::Block,
    sp: maka_lexer::Span,
) -> ast::Block {
    use ast::*;
    let it_name = "__it";
    let cur_name = "__cur";
    let none_flag = "__none";
    let broke_flag = "__broke";
    // Rewrite the user's body: top-level `break;` becomes `{ __broke = true; break; }`.
    let body_rewritten = rewrite_breaks_for_iter(body, broke_flag, sp);
    let it_ty = Type::Named(struct_name.to_string(), sp);
    let let_it = Stmt::Let {
        mutness: Mutness::Mut, ty: it_ty,
        name: it_name.to_string(), init: src.clone(),
        thread_local: false, span: sp,
    };
    let let_broke = Stmt::Let {
        mutness: Mutness::Mut, ty: Type::Named("bool".to_string(), sp),
        name: broke_flag.to_string(), init: Expr::Lit(Lit::Bool(false), sp),
        thread_local: false, span: sp,
    };
    let it_ref = Expr::Ref {
        mutness: Mutness::Mut,
        expr: Box::new(Expr::Ident(it_name.to_string(), sp)),
        span: sp,
    };
    let call_next = Expr::Call {
        callee: Box::new(Expr::Field { base: Box::new(it_ref), name: "next".to_string(), span: sp }),
        args: Vec::new(),
        span: sp,
    };
    let opt_ty = Type::Generic { name: "Option".to_string(), args: vec![elem_ty.clone()], span: sp };
    let let_cur = Stmt::Let {
        mutness: Mutness::Default, ty: opt_ty,
        name: cur_name.to_string(), init: call_next,
        thread_local: false, span: sp,
    };
    let let_none = Stmt::Let {
        mutness: Mutness::Mut, ty: Type::Named("bool".to_string(), sp),
        name: none_flag.to_string(), init: Expr::Lit(Lit::Bool(false), sp),
        thread_local: false, span: sp,
    };
    let some_pat = Pattern::VariantDestructure {
        enum_name: None, variant: "Some".to_string(),
        fields: vec![PatField {
            field: "value".to_string(),
            binding: Some(var_name.to_string()),
            literal: None, span: sp,
        }],
        span: sp,
    };
    let some_arm = MatchArm {
        pattern: some_pat, guard: None,
        body: ArmBody::Block(body_rewritten),
        span: sp,
    };
    let none_arm = MatchArm {
        pattern: Pattern::Variant { enum_name: None, variant: "None".to_string(), span: sp },
        guard: None,
        body: ArmBody::Block(Block {
            stmts: vec![Stmt::Assign {
                op: AssignOp::Assign,
                place: Expr::Ident(none_flag.to_string(), sp),
                value: Expr::Lit(Lit::Bool(true), sp),
                span: sp,
            }],
            span: sp,
        }),
        span: sp,
    };
    let match_stmt = Stmt::Match { scrutinee: Expr::Ident(cur_name.to_string(), sp), arms: vec![some_arm, none_arm], span: sp };
    let if_none = Stmt::If {
        cond: Expr::Ident(none_flag.to_string(), sp),
        then_block: Block { stmts: vec![Stmt::Break(sp)], span: sp },
        else_block: None, span: sp,
    };
    let if_broke = Stmt::If {
        cond: Expr::Ident(broke_flag.to_string(), sp),
        then_block: Block { stmts: vec![Stmt::Break(sp)], span: sp },
        else_block: None, span: sp,
    };
    let while_stmt = Stmt::While {
        cond: Expr::Lit(Lit::Bool(true), sp),
        body: Block { stmts: vec![let_cur, let_none, match_stmt, if_none, if_broke], span: sp },
        span: sp,
    };
    Block { stmts: vec![let_it, let_broke, while_stmt], span: sp }
}

/// Recursively rewrite a body so that top-level `break;` statements become
/// `{ __broke = true; break; }`.  Doesn't descend into nested loops — their
/// breaks belong to those loops, not the surrounding for-each.
fn rewrite_breaks_for_iter(b: &ast::Block, flag_name: &str, sp: maka_lexer::Span) -> ast::Block {
    use ast::*;
    let mut out = Vec::with_capacity(b.stmts.len());
    for s in &b.stmts {
        match s {
            Stmt::Break(_) => {
                out.push(Stmt::Block(Block {
                    stmts: vec![
                        Stmt::Assign {
                            op: AssignOp::Assign,
                            place: Expr::Ident(flag_name.to_string(), sp),
                            value: Expr::Lit(Lit::Bool(true), sp),
                            span: sp,
                        },
                        Stmt::Break(sp),
                    ],
                    span: sp,
                }));
            }
            Stmt::If { cond, then_block, else_block, span } => {
                out.push(Stmt::If {
                    cond: cond.clone(),
                    then_block: rewrite_breaks_for_iter(then_block, flag_name, sp),
                    else_block: else_block.as_ref().map(|eb| rewrite_breaks_for_iter(eb, flag_name, sp)),
                    span: *span,
                });
            }
            Stmt::Block(b) => out.push(Stmt::Block(rewrite_breaks_for_iter(b, flag_name, sp))),
            Stmt::Unsafe(b, sp2) => out.push(Stmt::Unsafe(rewrite_breaks_for_iter(b, flag_name, sp), *sp2)),
            Stmt::Match { scrutinee, arms, span } => {
                let new_arms = arms.iter().map(|a| MatchArm {
                    pattern: a.pattern.clone(),
                    guard: a.guard.clone(),
                    body: match &a.body {
                        ArmBody::Block(b) => ArmBody::Block(rewrite_breaks_for_iter(b, flag_name, sp)),
                        ArmBody::Expr(e) => ArmBody::Expr(e.clone()),
                    },
                    span: a.span,
                }).collect();
                out.push(Stmt::Match { scrutinee: scrutinee.clone(), arms: new_arms, span: *span });
            }
            // Don't recurse into While / ForRange / ForEach — their breaks belong to them.
            other => out.push(other.clone()),
        }
    }
    Block { stmts: out, span: b.span }
}

/// Compute a stable "receiver key" used for matching where-bound receivers
/// against call sites.  Peels references/pointers and returns the underlying
/// struct's name (for concrete types) or the TyVar name (for unsubstituted).
fn receiver_key(t: &HType, sym: &SymTab) -> Option<String> {
    match t {
        HType::Struct(id) => Some(sym.struct_info(*id).name.clone()),
        HType::Enum(id) => Some(sym.enum_info(*id).name.clone()),
        HType::TyVar(n) => Some(format!("@{}", n)),
        HType::Ref { inner, .. }
        | HType::Ptr { inner, .. }
        | HType::RawPtr { inner, .. }
        | HType::OwnPtr { inner, .. }
        | HType::Heap { inner } => receiver_key(inner, sym),
        _ => None,
    }
}

/// Flatten a string-concat chain (`__maka_str_concat*` calls) into its leaf
/// operands, left to right.  Used to lower `log(a + b + ...)` to a single printf.
fn flatten_str_concat(h: HExpr, out: &mut Vec<HExpr>) {
    let is_concat = matches!(&h.kind, HExprKind::Call { callee, args }
        if (callee.0 == u32::MAX - 5 || callee.0 == u32::MAX - 8 || callee.0 == u32::MAX - 9 || callee.0 == u32::MAX - 10) && args.len() == 2);
    if is_concat {
        if let HExprKind::Call { args, .. } = h.kind {
            let mut it = args.into_iter();
            flatten_str_concat(it.next().unwrap(), out);
            flatten_str_concat(it.next().unwrap(), out);
            return;
        }
    }
    out.push(h);
}

fn struct_id_of(t: &HType) -> Option<StructId> {
    match t {
        HType::Struct(id) => Some(*id),
        HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::Heap { inner } => struct_id_of(inner),
        _ => None,
    }
}

/// Does this type still contain an unresolved generic parameter?  Used by the
/// `inline for` unroll to tell a not-yet-monomorphized template (skip) apart
/// from a concrete non-struct receiver (error).
fn has_tyvar(t: &HType) -> bool {
    match t {
        HType::TyVar(_) | HType::GenericPattern { .. } => true,
        HType::Ref { inner, .. }
        | HType::Ptr { inner, .. }
        | HType::RawPtr { inner, .. }
        | HType::OwnPtr { inner, .. }
        | HType::Heap { inner } => has_tyvar(inner),
        HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => has_tyvar(elem),
        _ => false,
    }
}

/// Human-readable rendering of a type, used for the `f.type` reflection string.
fn htype_display(sym: &SymTab, t: &HType) -> String {
    match t {
        HType::Int => "int".to_string(),
        HType::Float => "float".to_string(),
        HType::Bool => "bool".to_string(),
        HType::Char => "char".to_string(),
        HType::Unit => "unit".to_string(),
        HType::Str => "string".to_string(),
        HType::SizedInt { signed, bits } => format!("{}{}", if *signed { "i" } else { "u" }, bits),
        HType::SizedFloat { bits } => format!("f{}", bits),
        HType::Struct(id) => sym.struct_info(*id).name.clone(),
        HType::Enum(id) => sym.enum_info(*id).name.clone(),
        HType::Ref { mutable, inner } => format!("&{}{}", if *mutable { "mut " } else { "" }, htype_display(sym, inner)),
        HType::Ptr { inner, .. } => format!("*{}", htype_display(sym, inner)),
        HType::OwnPtr { inner, .. } => format!("own *{}", htype_display(sym, inner)),
        HType::RawPtr { inner, .. } => format!("raw *{}", htype_display(sym, inner)),
        HType::Heap { inner } => format!("own &{}", htype_display(sym, inner)),
        HType::Array { len, elem } => format!("[{}]{}", len, htype_display(sym, elem)),
        HType::Slice { elem, .. } => format!("[]{}", htype_display(sym, elem)),
        HType::Vec { elem } => format!("[*]{}", htype_display(sym, elem)),
        HType::TyVar(n) => n.clone(),
        _ => "_".to_string(),
    }
}

/// One field's substitution context for an `inline for` unroll.
struct FieldCtx<'a> {
    var: &'a str,
    recv: &'a ast::Expr,
    fname: &'a str,
    index: i64,
    tystr: &'a str,
}

/// Replace `var.name` / `var.value` / `var.index` / `var.type` throughout an
/// AST block with the current field's name literal, field access, index, and
/// type string respectively.
fn rewrite_field_refs(b: &mut ast::Block, c: &FieldCtx) {
    for s in &mut b.stmts {
        rw_stmt(s, c);
    }
}

fn rw_stmt(s: &mut ast::Stmt, c: &FieldCtx) {
    use ast::Stmt::*;
    match s {
        Let { init, .. } => rw_expr(init, c),
        Assign { place, value, .. } => { rw_expr(place, c); rw_expr(value, c); }
        ExprStmt(e, _) => rw_expr(e, c),
        Return(Some(e), _) => rw_expr(e, c),
        Return(None, _) => {}
        If { cond, then_block, else_block, .. } => {
            rw_expr(cond, c);
            rewrite_field_refs(then_block, c);
            if let Some(eb) = else_block { rewrite_field_refs(eb, c); }
        }
        While { cond, body, .. } => { rw_expr(cond, c); rewrite_field_refs(body, c); }
        Block(b) => rewrite_field_refs(b, c),
        Unsafe(b, _) => rewrite_field_refs(b, c),
        Match { scrutinee, arms, .. } => {
            rw_expr(scrutinee, c);
            for a in arms {
                if let Some(g) = &mut a.guard { rw_expr(g, c); }
                match &mut a.body {
                    ast::ArmBody::Expr(e) => rw_expr(e, c),
                    ast::ArmBody::Block(b) => rewrite_field_refs(b, c),
                }
            }
        }
        Yield(e, _) => rw_expr(e, c),
        Propagate(Some(e), _) => rw_expr(e, c),
        Propagate(None, _) => {}
        ForRange { start, end, body, .. } => { rw_expr(start, c); rw_expr(end, c); rewrite_field_refs(body, c); }
        ForEach { src, body, .. } => { rw_expr(src, c); rewrite_field_refs(body, c); }
        InlineFor { iter, body, .. } => { rw_expr(iter, c); rewrite_field_refs(body, c); }
        Break(_) | Continue(_) => {}
    }
}

fn rw_expr(e: &mut ast::Expr, c: &FieldCtx) {
    use ast::Expr::*;
    // The magic `var.{name,value,index,type}` field accesses.
    if let Field { base, name, span } = e {
        if let Ident(b, _) = base.as_ref() {
            if b == c.var {
                let sp = *span;
                match name.as_str() {
                    "name" => { *e = ast::Expr::Lit(ast::Lit::Str(c.fname.to_string()), sp); return; }
                    "index" => { *e = ast::Expr::Lit(ast::Lit::Int(c.index), sp); return; }
                    // `ty` not `type` - `type` is a reserved keyword and won't parse after `.`.
                    "ty" => { *e = ast::Expr::Lit(ast::Lit::Str(c.tystr.to_string()), sp); return; }
                    "value" => { *e = ast::Expr::Field { base: Box::new(c.recv.clone()), name: c.fname.to_string(), span: sp }; return; }
                    _ => {}
                }
            }
        }
    }
    match e {
        Lit(..) | Ident(..) => {}
        Bin { lhs, rhs, .. } => { rw_expr(lhs, c); rw_expr(rhs, c); }
        Un { expr, .. } => rw_expr(expr, c),
        Unwrap { expr, .. } => rw_expr(expr, c),
        Ref { expr, .. } => rw_expr(expr, c),
        Field { base, .. } => rw_expr(base, c),
        Index { base, idx, .. } => { rw_expr(base, c); rw_expr(idx, c); }
        Call { callee, args, .. } => { rw_expr(callee, c); for a in args { rw_expr(a, c); } }
        Cast { expr, .. } => rw_expr(expr, c),
        CheckedCast { expr, .. } => rw_expr(expr, c),
        Struct { fields, .. } => { for (_, fe) in fields { rw_expr(fe, c); } }
        ArrayLit { elems, .. } => { for el in elems { rw_expr(el, c); } }
        HeapAlloc { value, .. } => rw_expr(value, c),
        Free { value, .. } => rw_expr(value, c),
        VariantCtor { fields, .. } => { for (_, fe) in fields { rw_expr(fe, c); } }
        Match { scrutinee, arms, .. } => {
            rw_expr(scrutinee, c);
            for a in arms {
                if let Some(g) = &mut a.guard { rw_expr(g, c); }
                match &mut a.body {
                    ast::ArmBody::Expr(e2) => rw_expr(e2, c),
                    ast::ArmBody::Block(b) => rewrite_field_refs(b, c),
                }
            }
        }
        Lambda { body, .. } => match body {
            ast::LambdaBody::Expr(e2) => rw_expr(e2, c),
            ast::LambdaBody::Block(b) => rewrite_field_refs(b, c),
        },
        WallMod { expr, .. } => rw_expr(expr, c),
        AttrCall { receiver, args, .. } => {
            if let Some(r) = receiver { rw_expr(r, c); }
            for a in args { rw_expr(a, c); }
        }
    }
}

pub fn type_eq(a: &HType, b: &HType) -> bool {
    use HType::*;
    match (a, b) {
        (Int, Int) | (Float, Float) | (Bool, Bool) | (Char, Char) | (Unit, Unit) | (Str, Str) | (NullT, NullT) => true,
        (SizedInt { signed: a, bits: b }, SizedInt { signed: c, bits: d }) => a == c && b == d,
        (SizedFloat { bits: a }, SizedFloat { bits: b }) => a == b,
        // `float` (binary64) ≡ `f64`: same C `double` ABI, source-name alias.
        (Float, SizedFloat { bits: 64 }) | (SizedFloat { bits: 64 }, Float) => true,
        (Struct(a), Struct(b)) => a == b,
        (Enum(a), Enum(b)) => a == b,
        (Ref { mutable: am, inner: ai }, Ref { mutable: bm, inner: bi }) => am == bm && type_eq(ai, bi),
        (Ptr { mutable: am, inner: ai }, Ptr { mutable: bm, inner: bi }) => am == bm && type_eq(ai, bi),
        (RawPtr { mutable: am, inner: ai }, RawPtr { mutable: bm, inner: bi }) => am == bm && type_eq(ai, bi),
        (OwnPtr { mutable: am, inner: ai }, OwnPtr { mutable: bm, inner: bi }) => am == bm && type_eq(ai, bi),
        (Heap { inner: ai }, Heap { inner: bi }) => type_eq(ai, bi),
        (Array { len: an, elem: ai }, Array { len: bn, elem: bi }) => an == bn && type_eq(ai, bi),
        (Slice { mutable: am, elem: ai }, Slice { mutable: bm, elem: bi }) => am == bm && type_eq(ai, bi),
        (Vec { elem: ai }, Vec { elem: bi }) => type_eq(ai, bi),
        (Dyn { traits: a }, Dyn { traits: b }) => a == b,
        (FnPtr { ret: ar, params: ap }, FnPtr { ret: br, params: bp }) => {
            type_eq(ar, br) && ap.len() == bp.len() && ap.iter().zip(bp.iter()).all(|(a, b)| type_eq(a, b))
        }
        (TyVar(a), TyVar(b)) => a == b,
        // Two `Rust<T>` are type-equal regardless of label — the label is
        // metadata, not part of the layout.  A `Rust<T>` is also equal to
        // a plain `own *mut unit` so coercion at extern call sites works
        // without ceremony.
        (RustOpaque(_), RustOpaque(_)) => true,
        (RustOpaque(_), OwnPtr { mutable: true, inner }) | (OwnPtr { mutable: true, inner }, RustOpaque(_))
            if matches!(**inner, HType::Unit) => true,
        _ => false,
    }
}

pub fn type_str(t: &HType) -> String {
    match t {
        HType::Int => "int".into(),
        HType::SizedInt { signed, bits: 0 } => if *signed { "isize".into() } else { "usize".into() },
        HType::SizedInt { signed, bits } => format!("{}{}", if *signed {"i"} else {"u"}, bits),
        HType::Float => "float".into(),
        HType::SizedFloat { bits } => format!("f{}", bits),
        HType::Bool => "bool".into(),
        HType::Char => "char".into(),
        HType::Unit => "unit".into(),
        HType::Str => "string".into(),
        HType::NullT => "null".into(),
        HType::Struct(i) => format!("struct#{}", i.0),
        HType::Enum(i) => format!("enum#{}", i.0),
        HType::Ref { mutable, inner } => format!("&{}{}", if *mutable {"mut "} else {""}, type_str(inner)),
        HType::Ptr { mutable, inner } => format!("*{}{}", if *mutable {""} else {"const "}, type_str(inner)),
        HType::RawPtr { mutable, inner } => format!("raw *{}{}", if *mutable {""} else {"const "}, type_str(inner)),
        HType::OwnPtr { mutable, inner } => format!("own *{}{}", if *mutable {""} else {"const "}, type_str(inner)),
        HType::Heap { inner } => format!("own &{}", type_str(inner)),
        HType::Array { len, elem } => format!("[{}]{}", len, type_str(elem)),
        HType::Slice { mutable, elem } => format!("[]{}{}", if *mutable {"mut "} else {""}, type_str(elem)),
        HType::Vec { elem } => format!("[*]{}", type_str(elem)),
        HType::Dyn { traits } => format!("dyn {}", traits.join("+")),
        HType::FnPtr { ret, params } => {
            let parts: Vec<String> = params.iter().map(type_str).collect();
            format!("{}({})", type_str(ret), parts.join(", "))
        }
        HType::TyVar(n) => format!("'{}", n),
        HType::AssocType { on, segment, .. } => format!("{}::{}", type_str(on), segment),
        HType::GenericPattern { template_name, args, .. } => {
            let inner: Vec<String> = args.iter().map(type_str).collect();
            format!("{}<{}>", template_name, inner.join(", "))
        }
        HType::RustOpaque(label) => format!("Rust<{}>", label),
    }
}

/// If `t` is a `dyn Trait` (possibly behind &/&mut/*/heap), return the traits list.
pub fn strip_to_dyn(t: &HType) -> Option<Vec<String>> {
    match t {
        HType::Dyn { traits } => Some(traits.clone()),
        HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::Heap { inner } => strip_to_dyn(inner),
        _ => None,
    }
}

/// After type-variable substitution, replace any `GenericPattern { template, args }`
/// whose args are all concrete with the matching concrete `Struct(id)` / `Enum(id)`
/// from `sym.struct_instantiations` / `sym.enum_instantiations`.  Returns the input
/// unchanged when no canonical instantiation exists.  Walks the type tree.
pub fn concretize_generic_patterns(t: &HType, sym: &SymTab) -> HType {
    fn fully_concrete(t: &HType) -> bool {
        match t {
            HType::TyVar(_) | HType::GenericPattern { .. } => false,
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. }
            | HType::OwnPtr { inner, .. } | HType::Heap { inner } => fully_concrete(inner),
            HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => fully_concrete(elem),
            HType::FnPtr { ret, params } => fully_concrete(ret) && params.iter().all(fully_concrete),
            HType::AssocType { on, .. } => fully_concrete(on),
            _ => true,
        }
    }
    match t {
        HType::GenericPattern { template_name, args, is_enum } => {
            let cargs: Vec<HType> = args.iter().map(|a| concretize_generic_patterns(a, sym)).collect();
            if cargs.iter().all(fully_concrete) {
                let key: String = cargs.iter().map(|t| t.key()).collect::<Vec<_>>().join(",");
                if *is_enum {
                    if let Some(eid) = sym.enum_instantiations.get(&(template_name.clone(), key)) {
                        return HType::Enum(EnumId(*eid));
                    }
                } else {
                    if let Some(sid) = sym.struct_instantiations.get(&(template_name.clone(), key)) {
                        return HType::Struct(StructId(*sid));
                    }
                }
            }
            HType::GenericPattern {
                template_name: template_name.clone(),
                args: cargs,
                is_enum: *is_enum,
            }
        }
        HType::Ref { mutable, inner } => HType::Ref { mutable: *mutable, inner: Box::new(concretize_generic_patterns(inner, sym)) },
        HType::Ptr { mutable, inner } => HType::Ptr { mutable: *mutable, inner: Box::new(concretize_generic_patterns(inner, sym)) },
        HType::RawPtr { mutable, inner } => HType::RawPtr { mutable: *mutable, inner: Box::new(concretize_generic_patterns(inner, sym)) },
        HType::OwnPtr { mutable, inner } => HType::OwnPtr { mutable: *mutable, inner: Box::new(concretize_generic_patterns(inner, sym)) },
        HType::Heap { inner } => HType::Heap { inner: Box::new(concretize_generic_patterns(inner, sym)) },
        HType::Array { len, elem } => HType::Array { len: *len, elem: Box::new(concretize_generic_patterns(elem, sym)) },
        HType::Slice { mutable, elem } => HType::Slice { mutable: *mutable, elem: Box::new(concretize_generic_patterns(elem, sym)) },
        HType::Vec { elem } => HType::Vec { elem: Box::new(concretize_generic_patterns(elem, sym)) },
        HType::FnPtr { ret, params } => HType::FnPtr {
            ret: Box::new(concretize_generic_patterns(ret, sym)),
            params: params.iter().map(|p| concretize_generic_patterns(p, sym)).collect(),
        },
        _ => t.clone(),
    }
}

/// Check if an argument's type is compatible with a parameter type (possibly generic).
pub fn param_compatible(param: &HType, actual: &HType, type_params: &[String]) -> bool {
    param_compatible_impl(param, actual, type_params, None)
}

/// SymTab-aware variant — needed when the param contains `GenericPattern`
/// (impl receiver pattern for `Result<T, E> has Foo`) and the actual is a
/// concrete `Enum(id)` / `Struct(id)` of an instantiation: we have to read
/// the concrete's `template` and `template_args` from sym to match arg-by-arg.
pub fn param_compatible_with_sym(param: &HType, actual: &HType, type_params: &[String], sym: &SymTab) -> bool {
    param_compatible_impl(param, actual, type_params, Some(sym))
}

fn param_compatible_impl(param: &HType, actual: &HType, type_params: &[String], sym: Option<&SymTab>) -> bool {
    // A TyVar matches anything.
    if let HType::TyVar(_) = param { return true; }
    // §10.4 receiver-pattern dispatch: the impl stored `Result<T, E>` as
    // GenericPattern; the call site has a concrete instantiation Enum(id).
    // Look up the concrete's template_name + template_args and match.
    if let HType::GenericPattern { template_name: pn, args: pargs, is_enum: pe } = param {
        if let Some(sym) = sym {
            match actual {
                HType::Struct(sid) if !pe => {
                    let info = sym.struct_info(*sid);
                    if info.template.as_deref() != Some(pn.as_str()) { return false; }
                    if info.template_args.len() != pargs.len() { return false; }
                    return pargs.iter().zip(info.template_args.iter())
                        .all(|(p, a)| param_compatible_impl(p, a, type_params, Some(sym)));
                }
                HType::Enum(eid) if *pe => {
                    let info = sym.enum_info(*eid);
                    if info.template.as_deref() != Some(pn.as_str()) { return false; }
                    if info.template_args.len() != pargs.len() { return false; }
                    return pargs.iter().zip(info.template_args.iter())
                        .all(|(p, a)| param_compatible_impl(p, a, type_params, Some(sym)));
                }
                HType::GenericPattern { template_name: an, args: aargs, is_enum: ae } => {
                    if pn != an || pe != ae || pargs.len() != aargs.len() { return false; }
                    return pargs.iter().zip(aargs.iter())
                        .all(|(p, a)| param_compatible_impl(p, a, type_params, Some(sym)));
                }
                _ => return false,
            }
        }
        return false;
    }
    match (param, actual) {
        (HType::Ref { mutable: pm, inner: pi }, HType::Ref { mutable: am, inner: ai })
            if pm == am => { if param_compatible_impl(pi, ai, type_params, sym) { return true; } }
        (HType::Ptr { mutable: pm, inner: pi }, HType::Ptr { mutable: am, inner: ai })
            if pm == am => { if param_compatible_impl(pi, ai, type_params, sym) { return true; } }
        (HType::RawPtr { mutable: pm, inner: pi }, HType::RawPtr { mutable: am, inner: ai })
            if pm == am => { if param_compatible_impl(pi, ai, type_params, sym) { return true; } }
        (HType::OwnPtr { mutable: pm, inner: pi }, HType::OwnPtr { mutable: am, inner: ai })
            if pm == am => { if param_compatible_impl(pi, ai, type_params, sym) { return true; } }
        // Function-pointer / closure params: recurse so a generic `'T('T)`
        // parameter accepts a concrete `int(int)` argument (higher-order
        // generics).  Contravariance is not modelled - exact arity, recursive
        // compatibility on return + each param.
        (HType::FnPtr { ret: pr, params: pp }, HType::FnPtr { ret: ar, params: ap })
            if pp.len() == ap.len() => {
                if param_compatible_impl(pr, ar, type_params, sym)
                    && pp.iter().zip(ap.iter()).all(|(p, a)| param_compatible_impl(p, a, type_params, sym)) {
                    return true;
                }
            }
        _ => {}
    }
    // Allow trivial implicit conversions (struct embedding upcast deferred).
    if type_eq(param, actual) { return true; }
    // Allow null → ptr.
    if matches!(actual, HType::NullT) && matches!(param, HType::Ptr { .. }) { return true; }
    // Implicit reborrow: writing `&mut g` where `g` is already `&mut T` produces
    // `&mut &mut T`; at the call site, strip the outer borrow so callers don't
    // have to remember whether to pass `g` or `&mut g`.  Same for `&T`.
    if let (HType::Ref { mutable: pm, inner: pi }, HType::Ref { mutable: am, inner: ai }) = (param, actual) {
        if pm == am {
            if let HType::Ref { mutable: imut, inner: iinner } = ai.as_ref() {
                if pm == imut && type_eq(pi, iinner) { return true; }
            }
        }
    }
    // Allow `own *char` → `string` (heap-allocated NUL-terminated buffer
    // produced by string concat / read_line, viewed read-only).
    if matches!(param, HType::Str)
        && matches!(actual, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Char)) {
        return true;
    }
    // Allow ref-mut → ref-const.
    if let (HType::Ref { mutable: false, inner: pi }, HType::Ref { mutable: true, inner: ai }) = (param, actual) {
        if type_eq(pi, ai) { return true; }
    }
    // Allow ptr-mut → ptr-const.
    if let (HType::Ptr { mutable: false, inner: pi }, HType::Ptr { mutable: true, inner: ai }) = (param, actual) {
        if type_eq(pi, ai) { return true; }
    }
    // Allow own *T → *T downgrade (passing an owner as a borrow).  Recurse into
    // the inner so a generic borrow param accepts an owning concrete argument
    // (e.g. `len<T>(*Node<T>)` called with an `own *Node<int>`).
    if let (HType::Ptr { mutable: pm, inner: pi }, HType::OwnPtr { mutable: am, inner: ai }) = (param, actual) {
        if (*am || !*pm) && param_compatible_impl(pi, ai, type_params, sym) { return true; }
    }
    // Allow own &T (Heap) -> *T downgrade as a borrow too.
    if let (HType::Ptr { inner: pi, .. }, HType::Heap { inner: ai }) = (param, actual) {
        if param_compatible_impl(pi, ai, type_params, sym) { return true; }
    }
    // `Rust<T>` is one pointer at the ABI; allow it to pass as any `*T`
    // (raw pointer) parameter — bridge-generated externs use that shape
    // for &T / &mut T params.
    if matches!((param, actual), (HType::Ptr { .. }, HType::RustOpaque(_))) {
        return true;
    }
    // Allow own &T → &T downgrade (recurse into the inner for generics).
    if let (HType::Ref { mutable: _, inner: pi }, HType::Heap { inner: ai }) = (param, actual) {
        if param_compatible_impl(pi, ai, type_params, sym) { return true; }
    }
    // Allow own *T (OwnPtr) -> &T downgrade.
    if let (HType::Ref { inner: pi, .. }, HType::OwnPtr { inner: ai, .. }) = (param, actual) {
        if param_compatible_impl(pi, ai, type_params, sym) { return true; }
    }
    // Allow slice-mut → slice-const.
    if let (HType::Slice { mutable: false, elem: pi }, HType::Slice { mutable: true, elem: ai }) = (param, actual) {
        if type_eq(pi, ai) { return true; }
    }
    // Allow array → slice (with element type match).
    if let (HType::Slice { elem: pi, .. }, HType::Array { elem: ai, .. }) = (param, actual) {
        if type_eq(pi, ai) { return true; }
    }
    // Containers with TyVars inside: rough match by skeleton.
    match (param, actual) {
        (HType::Ref { inner: pi, .. }, HType::Ref { inner: ai, .. }) => param_compatible(pi, ai, type_params),
        (HType::Ptr { inner: pi, .. }, HType::Ptr { inner: ai, .. }) => param_compatible(pi, ai, type_params),
        (HType::Heap { inner: pi }, HType::Heap { inner: ai }) => param_compatible(pi, ai, type_params),
        (HType::Array { elem: pi, .. }, HType::Array { elem: ai, .. }) => param_compatible(pi, ai, type_params),
        (HType::Slice { elem: pi, .. }, HType::Slice { elem: ai, .. }) => param_compatible(pi, ai, type_params),
        (HType::Vec { elem: pi }, HType::Vec { elem: ai }) => param_compatible(pi, ai, type_params),
        _ => false,
    }
}

/// Best-effort unification: collect bindings for TyVars by walking pattern (`pat`) vs actual.
pub fn unify(pat: &HType, actual: &HType, env: &mut std::collections::HashMap<String, HType>) {
    unify_impl(pat, actual, env, None)
}

pub fn unify_with_sym(pat: &HType, actual: &HType, env: &mut std::collections::HashMap<String, HType>, sym: &SymTab) {
    unify_impl(pat, actual, env, Some(sym))
}

fn unify_impl(pat: &HType, actual: &HType, env: &mut std::collections::HashMap<String, HType>, sym: Option<&SymTab>) {
    match (pat, actual) {
        (HType::TyVar(n), other) => {
            // A bare `null` argument carries no type information: binding the type
            // var to NullT would wrongly pin it (e.g. `hashmap(null)` -> V=NullT).
            // Leave it unbound so return-position / other-argument inference wins.
            if !matches!(other, HType::NullT) {
                env.entry(n.clone()).or_insert_with(|| other.clone());
            }
            return;
        }
        _ => {}
    }
    // §10.4: GenericPattern (impl receiver Result<T, E>) vs concrete Enum/Struct.
    // Recurse into the concrete's template_args to bind T and E.
    if let HType::GenericPattern { args: pargs, is_enum, .. } = pat {
        if let Some(sym) = sym {
            let actual_args: Option<&Vec<HType>> = match actual {
                HType::Enum(id) if *is_enum => Some(&sym.enum_info(*id).template_args),
                HType::Struct(id) if !*is_enum => Some(&sym.struct_info(*id).template_args),
                HType::GenericPattern { args, .. } => Some(args),
                _ => None,
            };
            if let Some(aargs) = actual_args {
                if pargs.len() == aargs.len() {
                    for (p, a) in pargs.iter().zip(aargs.iter()) {
                        unify_impl(p, a, env, Some(sym));
                    }
                }
                return;
            }
        }
        return;
    }
    match (pat, actual) {
        (HType::Ref { inner: pi, .. }, HType::Ref { inner: ai, .. }) => unify_impl(pi, ai, env, sym),
        (HType::Ptr { inner: pi, .. }, HType::Ptr { inner: ai, .. }) => unify_impl(pi, ai, env, sym),
        (HType::Heap { inner: pi }, HType::Heap { inner: ai }) => unify_impl(pi, ai, env, sym),
        (HType::Array { elem: pi, .. }, HType::Array { elem: ai, .. }) => unify_impl(pi, ai, env, sym),
        (HType::Slice { elem: pi, .. }, HType::Slice { elem: ai, .. }) => unify_impl(pi, ai, env, sym),
        (HType::Vec { elem: pi }, HType::Vec { elem: ai }) => unify_impl(pi, ai, env, sym),
        (HType::OwnPtr { inner: pi, .. }, HType::OwnPtr { inner: ai, .. }) => unify_impl(pi, ai, env, sym),
        (HType::RawPtr { inner: pi, .. }, HType::RawPtr { inner: ai, .. }) => unify_impl(pi, ai, env, sym),
        // own->borrow coercion at a generic param: a `*T` / `&T` parameter
        // receiving an `own *U` / `own &U` argument binds T from U's inner.
        (HType::Ptr { inner: pi, .. }, HType::OwnPtr { inner: ai, .. })
        | (HType::Ptr { inner: pi, .. }, HType::Heap { inner: ai })
        | (HType::Ref { inner: pi, .. }, HType::OwnPtr { inner: ai, .. })
        | (HType::Ref { inner: pi, .. }, HType::Heap { inner: ai }) => unify_impl(pi, ai, env, sym),
        // Bind type vars through function-pointer / closure types so a generic
        // arg that appears only inside a `'T('T)` parameter is still inferred.
        (HType::FnPtr { ret: pr, params: pp }, HType::FnPtr { ret: ar, params: ap }) => {
            unify_impl(pr, ar, env, sym);
            for (p, a) in pp.iter().zip(ap.iter()) { unify_impl(p, a, env, sym); }
        }
        // Also accept passing `T` value for `&T`/`&mut T` parameter.
        (HType::Ref { inner: pi, .. }, other) => unify_impl(pi, other, env, sym),
        (HType::Ptr { inner: pi, .. }, other) => unify_impl(pi, other, env, sym),
        _ => {}
    }
}

