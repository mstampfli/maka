//! Maka semantic analysis: types, lifetimes, deps tracking, move tracking.
//! Spec v1.2 §1, §2, §6, §7, §8, §9, §10, §11.
//!
//! Produces a typed HIR consumed by the codegen.

pub mod hir;
mod resolve;
mod typeck;
mod lifetime;
pub use lifetime::consume_only_fnptr_params;

pub use hir::*;
pub use resolve::*;
pub use typeck::*;

use maka_lexer::Span;

#[derive(Debug, Clone)]
pub struct SemaError {
    pub msg: String,
    pub span: Span,
}

impl std::fmt::Display for SemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sema error at {}: {}", self.span, self.msg)
    }
}

pub type SemaResult<T> = Result<T, SemaError>;

/// A diagnostic that does not stop compilation but warns the user about
/// suspicious-but-defined behavior — e.g. using a pointer that was auto-nulled
/// by the lifetime pass to prevent use-after-free, without having explicitly
/// re-assigned it on every code path since.
#[derive(Debug, Clone)]
pub struct SemaWarning {
    pub msg: String,
    pub span: Span,
}

impl std::fmt::Display for SemaWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "warning at {}: {}", self.span, self.msg)
    }
}

/// Walks the AST module and lifts every `Expr::Lambda` to a fresh top-level function.
/// Replaces each lambda expression with `Expr::Ident("__lambda_<N>")`.
fn lambda_lift(mut m: maka_ast::Module) -> maka_ast::Module {
    let mut counter: u32 = 0;
    // Group the lifted lambdas by the enclosing item's index so that we can
    // copy that item's module path + imports onto the synthesized FuncDecl.
    // Without this, closure bodies can't call any non-`pub`/non-imported name
    // that the enclosing function relied on (e.g. `sleep_ms` from std).
    let mut lifted: Vec<(usize, maka_ast::FuncDecl)> = Vec::new();
    for (idx, item) in m.items.iter_mut().enumerate() {
        let mut bucket: Vec<maka_ast::FuncDecl> = Vec::new();
        match item {
            maka_ast::Item::Func(f) => {
                lift_block(&mut f.body, &mut counter, &mut bucket, false);
            }
            _ => {}
        }
        for nf in bucket { lifted.push((idx, nf)); }
    }
    for (src_idx, nf) in lifted {
        let module_path = m.item_modules.get(src_idx).cloned().unwrap_or_default();
        let imports = m.item_imports.get(src_idx).cloned().unwrap_or_default();
        let has_imports = m.item_has_imports.get(src_idx).cloned().unwrap_or_default();
        m.items.push(maka_ast::Item::Func(nf));
        m.item_modules.push(module_path);
        m.item_imports.push(imports);
        m.item_has_imports.push(has_imports);
    }
    m
}

fn lift_block(b: &mut maka_ast::Block, counter: &mut u32, out: &mut Vec<maka_ast::FuncDecl>, in_unsafe: bool) {
    for s in &mut b.stmts { lift_stmt(s, counter, out, in_unsafe); }
}

fn lift_stmt(s: &mut maka_ast::Stmt, counter: &mut u32, out: &mut Vec<maka_ast::FuncDecl>, in_unsafe: bool) {
    use maka_ast::Stmt::*;
    match s {
        Let { init, .. } => lift_expr(init, counter, out, in_unsafe),
        LetTuple { init, .. } => lift_expr(init, counter, out, in_unsafe),
        Assign { place, value, .. } => { lift_expr(place, counter, out, in_unsafe); lift_expr(value, counter, out, in_unsafe); }
        ExprStmt(e, _) => lift_expr(e, counter, out, in_unsafe),
        Return(Some(e), _) => lift_expr(e, counter, out, in_unsafe),
        Return(None, _) => {}
        If { cond, then_block, else_block, .. } => {
            lift_expr(cond, counter, out, in_unsafe);
            lift_block(then_block, counter, out, in_unsafe);
            if let Some(b) = else_block { lift_block(b, counter, out, in_unsafe); }
        }
        While { cond, body, .. } => { lift_expr(cond, counter, out, in_unsafe); lift_block(body, counter, out, in_unsafe); }
        Block(b) => lift_block(b, counter, out, in_unsafe),
        Unsafe(b, _) => lift_block(b, counter, out, /*in_unsafe*/ true),
        Match { scrutinee, arms, .. } => {
            lift_expr(scrutinee, counter, out, in_unsafe);
            for a in arms {
                if let Some(g) = &mut a.guard { lift_expr(g, counter, out, in_unsafe); }
                match &mut a.body {
                    maka_ast::ArmBody::Expr(e) => lift_expr(e, counter, out, in_unsafe),
                    maka_ast::ArmBody::Block(b) => lift_block(b, counter, out, in_unsafe),
                }
            }
        }
        Yield(e, _) => lift_expr(e, counter, out, in_unsafe),
        Propagate(opt, _) => if let Some(e) = opt { lift_expr(e, counter, out, in_unsafe); },
        ForEach { src, body, .. } => {
            lift_expr(src, counter, out, in_unsafe);
            lift_block(body, counter, out, in_unsafe);
        }
        InlineFor { iter, body, .. } => {
            lift_expr(iter, counter, out, in_unsafe);
            lift_block(body, counter, out, in_unsafe);
        }
        ForRange { start, end, body, .. } => {
            lift_expr(start, counter, out, in_unsafe);
            lift_expr(end, counter, out, in_unsafe);
            lift_block(body, counter, out, in_unsafe);
        }
        Break(_) | Continue(_) => {}
    }
}

fn lift_expr(e: &mut maka_ast::Expr, counter: &mut u32, out: &mut Vec<maka_ast::FuncDecl>, in_unsafe: bool) {
    // Post-order: lift children first.
    match e {
        maka_ast::Expr::Bin { lhs, rhs, .. } => { lift_expr(lhs, counter, out, in_unsafe); lift_expr(rhs, counter, out, in_unsafe); }
        maka_ast::Expr::Un { expr, .. } => lift_expr(expr, counter, out, in_unsafe),
        maka_ast::Expr::Unwrap { expr, .. } => lift_expr(expr, counter, out, in_unsafe),
        maka_ast::Expr::Ref { expr, .. } => lift_expr(expr, counter, out, in_unsafe),
        maka_ast::Expr::Field { base, .. } => lift_expr(base, counter, out, in_unsafe),
        maka_ast::Expr::Index { base, idx, .. } => { lift_expr(base, counter, out, in_unsafe); lift_expr(idx, counter, out, in_unsafe); }
        maka_ast::Expr::Call { callee, args, .. } => {
            lift_expr(callee, counter, out, in_unsafe);
            for a in args { lift_expr(a, counter, out, in_unsafe); }
        }
        maka_ast::Expr::Cast { expr, .. } | maka_ast::Expr::CheckedCast { expr, .. } => lift_expr(expr, counter, out, in_unsafe),
        maka_ast::Expr::Struct { fields, .. } => for (_, fe) in fields { lift_expr(fe, counter, out, in_unsafe); },
        maka_ast::Expr::VariantCtor { fields, .. } => for (_, fe) in fields { lift_expr(fe, counter, out, in_unsafe); },
        maka_ast::Expr::ArrayLit { elems, .. } => for ee in elems { lift_expr(ee, counter, out, in_unsafe); },
        maka_ast::Expr::HeapAlloc { value, .. } => lift_expr(value, counter, out, in_unsafe),
        maka_ast::Expr::Match { scrutinee, arms, .. } => {
            lift_expr(scrutinee, counter, out, in_unsafe);
            for a in arms {
                if let Some(g) = &mut a.guard { lift_expr(g, counter, out, in_unsafe); }
                match &mut a.body {
                    maka_ast::ArmBody::Expr(e) => lift_expr(e, counter, out, in_unsafe),
                    maka_ast::ArmBody::Block(b) => lift_block(b, counter, out, in_unsafe),
                }
            }
        }
        maka_ast::Expr::WallMod { expr, .. } => {
            lift_expr(expr, counter, out, in_unsafe);
        }
        maka_ast::Expr::Lambda { ret, params, captures, body, span } => {
            // Lift body's children first.
            match body {
                maka_ast::LambdaBody::Block(b) => lift_block(b, counter, out, in_unsafe),
                maka_ast::LambdaBody::Expr(e) => lift_expr(e, counter, out, in_unsafe),
            }
            // Capturing lambdas are handled at sema time (need types of captures).
            if !captures.is_empty() { return; }
            let id = *counter;
            *counter += 1;
            // Encode the lift-time unsafe context into the synthetic name so
            // check_func can decide whether to apply the `*unit` ban or not.
            let name = if in_unsafe {
                format!("__lambda_unsafe_{}", id)
            } else {
                format!("__lambda_{}", id)
            };
            let block = match body {
                maka_ast::LambdaBody::Block(b) => b.clone(),
                maka_ast::LambdaBody::Expr(e) => maka_ast::Block {
                    stmts: vec![maka_ast::Stmt::Return(Some((**e).clone()), *span)],
                    span: *span,
                },
            };
            let decl = maka_ast::FuncDecl {
                name: name.clone(),
                type_params: Vec::new(),
                params: params.clone(),
                ret: ret.clone(),
                body: block,
                is_inline: false,
                is_gate: false,
                is_pub: false,
                is_export: false,
                where_clauses: Vec::new(),
                span: *span,
            };
            out.push(decl);
            *e = maka_ast::Expr::Ident(name, *span);
        }
        _ => {}
    }
}

pub fn analyze(m: &maka_ast::Module) -> Result<HirModule, Vec<SemaError>> {
    let mut errors = Vec::new();
    let mut warnings: Vec<SemaWarning> = Vec::new();
    // Lift lambdas to top-level functions.
    let m_lifted = lambda_lift(m.clone());
    let m = &m_lifted;
    let mut sym = match SymTab::collect(m) {
        Ok(s) => s,
        Err(es) => return Err(es),
    };
    let mut funcs: Vec<HFunc> = Vec::new();

    // Stored per HFunc-index: any instantiation requests queued during its checking.
    let mut pending_reqs: Vec<Vec<InstantiationReq>> = Vec::new();

    // Globals: type-check each initializer and register the GlobalInfo before
    // any function body sees a reference.  Each module's globals live in the
    // file's module path; init expressions must be constant in the C sense
    // (codegen passes them through as static initializers).
    for (idx, item) in m.items.iter().enumerate() {
        let maka_ast::Item::Global(g) = item else { continue; };
        let item_module: Vec<String> = m.item_modules.get(idx).cloned().unwrap_or_default();
        let tc = TypeChecker::new_with_trait(&sym, None);
        let resolved_ty = resolve::resolve_type(&sym, &g.ty, &mut errors);
        let init_h = match tc.check_global_init(&g.init, &resolved_ty) {
            Ok(h) => h,
            Err(es) => { errors.extend(es); continue; }
        };
        let c_name = format!("__maka_global__{}", g.name);
        sym.globals.push(hir::GlobalInfo {
            name: g.name.clone(),
            c_name,
            ty: resolved_ty,
            init: init_h,
            is_mut: g.is_mut,
            is_pub: g.is_pub,
            module_path: item_module,
            span: g.span,
        });
    }

    // Maps a parent HFunc index -> the range of `funcs` indices holding the
    // lifted CAPTURING-closure bodies it produced.  Those synth funcs carry
    // placeholder FuncIds (for generic calls inside the closure body) that were
    // re-based into the PARENT's instantiation_requests by check_capturing_lambda,
    // so they must be rewritten with the parent's mapping - but they are pushed
    // with EMPTY pending_reqs, so the fixpoint loop never visits them on their
    // own.  We rewrite them alongside their parent instead.
    let mut synth_of_parent: std::collections::HashMap<usize, std::ops::Range<usize>> = std::collections::HashMap::new();

    // First pass: non-generic top-level functions.
    let mut seen_has_pairs: std::collections::HashMap<(String, String), usize> = std::collections::HashMap::new();
    for item in &m.items {
        match item {
            maka_ast::Item::Func(f) => {
                if !f.type_params.is_empty() { continue; }
                let tc = TypeChecker::new_with_trait(&sym, None);
                match tc.check_func_with_id(f, None) {
                    Ok((hfunc, reqs, synth)) => {
                        for s in synth.structs { sym.structs.push(s); }
                        for s in synth.sigs { sym.sigs.push(s); }
                        let __ss = funcs.len();
                        for fnc in synth.funcs { funcs.push(fnc); pending_reqs.push(Vec::new()); }
                        if funcs.len() > __ss { synth_of_parent.insert(funcs.len(), __ss..funcs.len()); }
                        sym.send_probes.extend(synth.send_probes);
                        sym.sync_probes.extend(synth.sync_probes);
                        pending_reqs.push(reqs);
                        funcs.push(hfunc);
                    }
                    Err(es) => errors.extend(es),
                }
            }
            maka_ast::Item::Has(h) => {
                // Find the HasImpl record this AST item produced.  Match by
                // (attr_name, type_name) AND by resolved attr_args so two
                // `Type has Attr<X>` / `Type has Attr<Y>` decls land on
                // their respective impls.  We can't resolve attr_args here
                // (no SymTab in this scope), so fall back to a per-decl
                // counter to pick the Nth matching impl in source order.
                let nth_for_pair: usize = {
                    let target = (h.attr_name.clone(), h.type_name.clone());
                    seen_has_pairs.entry(target).and_modify(|c| *c += 1).or_insert(0).clone()
                };
                let matches: Vec<usize> = sym.has_impls.iter().enumerate()
                    .filter_map(|(i, hi)| {
                        if hi.attr_name == h.attr_name && hi.type_key == h.type_name { Some(i) } else { None }
                    })
                    .collect();
                let Some(&idx) = matches.get(nth_for_pair) else { continue; };
                if sym.has_impls[idx].func_ids.is_empty() { continue; }
                let func_ids = sym.has_impls[idx].func_ids.clone();
                for fid in func_ids {
                    let f_decl = match sym.ast_funcs.get(&fid.0).cloned() {
                        Some(f) => f,
                        None => continue,
                    };
                    if !f_decl.type_params.is_empty() { continue; }
                    let tc = TypeChecker::new_with_trait(&sym, Some(&h.attr_name));
                    match tc.check_func_with_id(&f_decl, Some(fid)) {
                        Ok((hfunc, reqs, synth)) => {
                            for s in synth.structs { sym.structs.push(s); }
                            for s in synth.sigs { sym.sigs.push(s); }
                            let __ss = funcs.len();
                        for fnc in synth.funcs { funcs.push(fnc); pending_reqs.push(Vec::new()); }
                        if funcs.len() > __ss { synth_of_parent.insert(funcs.len(), __ss..funcs.len()); }
                            sym.send_probes.extend(synth.send_probes);
                            sym.sync_probes.extend(synth.sync_probes);
                            pending_reqs.push(reqs);
                            funcs.push(hfunc);
                        }
                        Err(es) => errors.extend(es),
                    }
                }
            }
            _ => {}
        }
    }

    // Process instantiations to a fixpoint. Each request needs a fresh FuncId+sig in `sym`,
    // and we need to rewrite placeholder FuncIds in the corresponding caller body.
    loop {
        // Are there any pending requests left to drain?
        let any = pending_reqs.iter().any(|v| !v.is_empty());
        if !any { break; }

        // For each function's pending requests, dedupe and allocate.
        // We process one HFunc index at a time to keep indices stable.
        let n = pending_reqs.len();
        for i in 0..n {
            let reqs = std::mem::take(&mut pending_reqs[i]);
            if reqs.is_empty() { continue; }

            // For each request: dedupe via sym.instantiations
            let mut mapping: Vec<u32> = Vec::with_capacity(reqs.len());
            for req in &reqs {
                let arg_keys: Vec<String> = req.args.iter().map(|t| t.key()).collect();
                let key_s = arg_keys.join(",");
                let dedup_key = (req.template_fid.0, key_s.clone());
                if let Some(existing) = sym.instantiations.get(&dedup_key) {
                    mapping.push(*existing);
                    continue;
                }
                // Allocate new FuncId.
                let template_sig = sym.func_sig(req.template_fid).clone();
                let new_fid = FuncId(sym.sigs.len() as u32);
                let mut env: std::collections::HashMap<String, HType> = std::collections::HashMap::new();
                for (tp, t) in template_sig.type_params.iter().zip(req.args.iter()) {
                    env.insert(tp.clone(), t.clone());
                }
                // Resolve any `T::Assoc` to the impl's concrete type after
                // substituting the type args, so the instantiated signature (and
                // thus codegen and overload resolution at the call site) sees the
                // real type, not an unresolved associated-type placeholder.
                let new_param_tys: Vec<HType> = template_sig.param_tys.iter()
                    .map(|t| resolve::resolve_assoc_types_in(&sym, &t.subst(&env))).collect();
                let new_ret = resolve::resolve_assoc_types_in(&sym, &template_sig.ret.subst(&env));
                // Enforce where-clause bounds at this instantiation, filtering `has`
                // impls by visibility: a non-`pub` impl is only usable in its own
                // module; a `pub` impl is only usable in modules that opted in via
                // `use Mod.Type.Attr;`.
                for (trait_name, type_args, assoc_bindings) in &template_sig.where_bounds {
                    // Multi-arg bound semantics: type_args[0] is the receiver type
                    // (T in `where T has Attr<U>`), type_args[1..] are the attr's
                    // type-args.  Match against a HasImpl whose type_key equals the
                    // receiver AND whose attr_args match the rest.
                    if type_args.is_empty() { continue; }
                    let recv_concrete = type_args[0].subst(&env);
                    let attr_args_concrete: Vec<HType> = type_args[1..].iter().map(|a| a.subst(&env)).collect();
                    let key = resolve::underlying_struct_key(&sym, &recv_concrete);
                    // §10.5 bounded assoc types: if the bound has any
                    // `Slot = ConcreteT` bindings, the picked impl's
                    // `type Slot = R` (after substitution via receiver
                    // unification) must type_eq ConcreteT.
                    let bindings_concrete: Vec<(String, hir::HType)> = assoc_bindings.iter()
                        .map(|(n, t)| (n.clone(), t.subst(&env)))
                        .collect();
                    let satisfied = match key.as_ref() {
                        None => false,
                        Some(k) => sym.has_impls.iter().any(|h| {
                            if h.attr_name != *trait_name { return false; }
                            // Primary match: the impl's type_key string equals the
                            // receiver's underlying name (the proven non-generic
                            // path).  Fallback: for a generic-struct receiver the
                            // impl is keyed by its full form (`Box<T>` / `Box<int>`)
                            // which never string-equals the bare name, so unify the
                            // impl's receiver pattern against the concrete receiver.
                            if h.type_key != *k
                                && hir::receiver_unify_with_sym(&h.receiver_pattern, &recv_concrete, &h.receiver_tyvars, &sym).is_none() {
                                return false;
                            }
                            // A bound may omit trait type params that default to
                            // `_` (Self) - e.g. `T has Add` against `int has Add`,
                            // where the impl carries the defaulted `R = int`.  The
                            // impl must have at least as many attr args as the
                            // bound; each supplied one must match, and any extra
                            // (defaulted) one must equal the receiver (Self).
                            if h.attr_args.len() < attr_args_concrete.len() { return false; }
                            if !h.attr_args.iter().enumerate().all(|(i, ha)| {
                                let want = attr_args_concrete.get(i).unwrap_or(&recv_concrete);
                                typeck::type_eq(ha, want)
                            }) { return false; }
                            // The bound is part of the generic's contract: satisfy
                            // it if the impl is visible from the generic's DEFINING
                            // module (it knows its own impls - e.g. std's
                            // `int has AtomicWord` for a std generic) OR from the
                            // caller's module.  Without the former, callers would
                            // have to `use` an impl they never name.
                            if !has_impl_visible(h, &template_sig.module_path, &template_sig.has_imports)
                                && !has_impl_visible(h, &req.caller_module, &req.caller_has_imports) { return false; }
                            // Validate assoc-type bindings: substitute the impl's
                            // type vars (via receiver_unify against recv_concrete),
                            // then compare each binding's value.
                            if !bindings_concrete.is_empty() {
                                let env = match hir::receiver_unify_with_sym(&h.receiver_pattern, &recv_concrete, &h.receiver_tyvars, &sym) {
                                    Some(e) => e,
                                    None => return false,
                                };
                                for (bname, bvalue) in &bindings_concrete {
                                    let impl_def = h.assoc_type_defs.iter().find(|(n, _)| n == bname);
                                    let resolved = match impl_def {
                                        Some((_, raw)) => raw.subst(&env),
                                        None => return false,
                                    };
                                    if !typeck::type_eq(&resolved, bvalue) { return false; }
                                }
                            }
                            true
                        }),
                    };
                    if !satisfied {
                        let span = sym.ast_funcs.get(&req.template_fid.0)
                            .map(|f| f.span)
                            .unwrap_or_else(maka_lexer::Span::dummy);
                        let pretty = key.unwrap_or_else(|| typeck::type_str(&recv_concrete));
                        let exists = sym.has_impls.iter().any(|h|
                            h.attr_name == *trait_name && h.type_key == pretty
                        );
                        let msg = if exists {
                            format!(
                                "type `{}` has a `has {}` impl but it is not visible here — either move it to this module, mark it `pub`, or add `use {}.{}.{};`",
                                pretty, trait_name,
                                sym.has_impls.iter().find(|h| h.attr_name == *trait_name && h.type_key == pretty)
                                    .map(|h| if h.module_path.is_empty() { "<root>".to_string() } else { h.module_path.join(".") })
                                    .unwrap_or_default(),
                                pretty, trait_name,
                            )
                        } else {
                            format!(
                                "type `{}` does not satisfy `{}` bound at instantiation of `{}`",
                                pretty, trait_name, template_sig.name,
                            )
                        };
                        errors.push(SemaError { msg, span });
                    }
                }
                // Mangle: name__keys.
                let mangle = arg_keys.join("_");
                let c_name = format!("{}__{}", template_sig.c_name, mangle);
                sym.sigs.push(FuncSig {
                    name: template_sig.name.clone(),
                    param_tys: new_param_tys,
                    param_names: template_sig.param_names.clone(),
                    ret: new_ret,
                    is_extern: false,
                    c_name,
                    trait_name: template_sig.trait_name.clone(),
                    type_params: Vec::new(),
                    is_inline: template_sig.is_inline,
                    is_gate: template_sig.is_gate,
                    is_variadic: template_sig.is_variadic,
                    is_pub: template_sig.is_pub,
                    is_export: false,
                    module_path: template_sig.module_path.clone(),
                    imports: template_sig.imports.clone(),
                    has_imports: template_sig.has_imports.clone(),
                    where_bounds: template_sig.where_bounds.clone(),
                });
                sym.instantiations.insert(dedup_key, new_fid.0);
                mapping.push(new_fid.0);
            }

            // Rewrite placeholder FuncIds in the caller's body.
            rewrite_placeholders(&mut funcs[i], &mapping);
            // ...and in the lifted capturing-closure bodies this function produced
            // (their placeholders were re-based into this function's requests, so
            // the same mapping applies).  Without this a generic call inside a
            // capturing closure keeps its placeholder FuncId into codegen and
            // panics on an out-of-bounds signature lookup.
            if let Some(range) = synth_of_parent.get(&i).cloned() {
                for j in range { rewrite_placeholders(&mut funcs[j], &mapping); }
            }

            // Type-check the new instantiation bodies (each may produce more requests).
            for (req, new_id) in reqs.iter().zip(mapping.iter()) {
                let new_fid = FuncId(*new_id);
                // If sym.funcs already has it (from a sibling instantiation cycle), skip.
                if funcs.iter().any(|f| f.id == new_fid) { continue; }
                let template_sig = sym.func_sig(req.template_fid).clone();
                let f_ast = sym.ast_funcs.get(&req.template_fid.0).cloned();
                let Some(f_ast) = f_ast else { continue; };
                let mut env: std::collections::HashMap<String, HType> = std::collections::HashMap::new();
                for (tp, t) in template_sig.type_params.iter().zip(req.args.iter()) {
                    env.insert(tp.clone(), t.clone());
                }
                let tc = TypeChecker::new_with_trait(&sym, template_sig.trait_name.as_deref()).with_subst(env);
                match tc.check_func_with_id(&f_ast, Some(new_fid)) {
                    Ok((hf, reqs2, synth)) => {
                        for s in synth.structs { sym.structs.push(s); }
                        for s in synth.sigs { sym.sigs.push(s); }
                        let __ss = funcs.len();
                        for fnc in synth.funcs { funcs.push(fnc); pending_reqs.push(Vec::new()); }
                        if funcs.len() > __ss { synth_of_parent.insert(funcs.len(), __ss..funcs.len()); }
                        sym.send_probes.extend(synth.send_probes);
                        sym.sync_probes.extend(synth.sync_probes);
                        pending_reqs.push(reqs2);
                        funcs.push(hf);
                    }
                    Err(es) => errors.extend(es),
                }
            }
        }
    }

    // Detect recursion in inline functions (forbidden per spec).
    detect_inline_recursion(&sym, &funcs, &mut errors);
    // Validate every `propagate X;` against the *caller's* return type at each InlineCall site.
    check_inline_propagate_compat(&sym, &funcs, &mut errors);
    // An inline whose body has a `break`/`continue` targeting an enclosing loop must
    // only be called from inside a loop; reject calling it at loop depth 0.
    check_inline_loop_jumps(&sym, &funcs, &mut errors);
    // A `mut` module global written (transitively) from a real OS-thread / job /
    // pool body is an unsynchronized data race: globals are referenced by name, not
    // captured, so the cross-thread capture check never sees them.  Reject it.
    check_cross_thread_global_races(&sym, &funcs, &mut errors);
    // `export` functions emit a stable unmangled C symbol callable from C/Rust,
    // so their signature must cross the C ABI cleanly.
    check_exports(&sym, &mut errors);

    // Interprocedural pass — runs after every function (including instantiations)
    // is fully lowered.  First compute "never returns null" summaries by fixpoint
    // over the lowered HIR (§6.3 forced-handling proof obligation), then run the
    // per-function lifetime/deps/move pass with those summaries in hand so
    // `(some_fn())!` is accepted iff every return path in `some_fn` is provably
    // non-null.  Capture-site non-null facts harvested from `Closure` exprs are
    // forwarded into the corresponding lifted body so its own pass starts with
    // those captures proven (§6.3 closure-capture flow).
    let summaries = lifetime::compute_return_summaries(&sym, &funcs);
    // Interprocedural borrow provenance: which parameters each function's
    // borrowed return value aliases, so the escape pass can follow a borrow
    // back across a call to the local it would dangle on.
    let ret_borrows = lifetime::compute_return_borrows(&sym, &funcs);
    // Pre-pass to harvest capture-site non-null facts.  Lifted closure funcs
    // live *earlier* in `funcs[]` than their parent (they're pushed during the
    // parent's type-check, before the parent's own push), so a single forward
    // walk wouldn't see the parent's facts in time for the child.  This pre-pass
    // walks every function looking only for `Closure` exprs and records their
    // non-null env_values; the main pass below then applies the collected facts
    // as each function's initial state regardless of vec order.
    let mut capture_inits: std::collections::HashMap<u32, Vec<hir::LocalId>> = std::collections::HashMap::new();
    for hf in &funcs {
        lifetime::harvest_capture_nonnull(&sym, hf, &summaries, &mut capture_inits);
    }
    for hf in &mut funcs {
        let initial: Vec<hir::LocalId> = capture_inits.remove(&hf.id.0).unwrap_or_default();
        match lifetime::analyze_func(&sym, hf, &summaries, &ret_borrows, &initial) {
            Ok(ok) => warnings.extend(ok.warnings),
            Err(es) => errors.extend(es),
        }
    }

    if !errors.is_empty() {
        sym.funcs = funcs;
        return Err(errors);
    }

    sym.funcs = funcs;
    // Forward the cinclude / cblock directives verbatim — they're not type-checked,
    // they're just pasted into the generated C in source order.
    let mut cincludes: Vec<String> = Vec::new();
    let mut cblocks: Vec<String> = Vec::new();
    let mut clinks: Vec<String> = Vec::new();
    for it in &m.items {
        match it {
            maka_ast::Item::CInclude(h, _) => cincludes.push(h.clone()),
            maka_ast::Item::CBlock(b, _)   => cblocks.push(b.clone()),
            maka_ast::Item::CLink(f, _)    => clinks.push(f.clone()),
            _ => {}
        }
    }
    Ok(HirModule { sym, warnings, cincludes, cblocks, clinks })
}

/// Walks each non-inline function and, at every InlineCall, scans the inline body for
/// `propagate X;` and verifies the value type matches the caller's return type.
fn check_inline_propagate_compat(sym: &SymTab, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
    for caller in funcs {
        let caller_sig = sym.func_sig(caller.id);
        if caller_sig.is_inline { continue; }
        let ret = caller_sig.ret.clone();
        walk_block(sym, &caller.body, &ret, funcs, errors);
    }
    fn walk_block(sym: &SymTab, b: &HBlock, ret: &HType, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
        for s in &b.stmts { walk_stmt(sym, s, ret, funcs, errors); }
    }
    fn walk_stmt(sym: &SymTab, s: &HStmt, ret: &HType, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
        match s {
            HStmt::Let { init, .. } => walk_expr(sym, init, ret, funcs, errors),
            HStmt::Assign { place, value, .. } => { walk_expr(sym, place, ret, funcs, errors); walk_expr(sym, value, ret, funcs, errors); }
            HStmt::ExprStmt(e) => walk_expr(sym, e, ret, funcs, errors),
            HStmt::Return { value: Some(v), .. } => walk_expr(sym, v, ret, funcs, errors),
            HStmt::Return { .. } => {}
            HStmt::If { cond, then_b, else_b, .. } => {
                walk_expr(sym, cond, ret, funcs, errors);
                walk_block(sym, then_b, ret, funcs, errors);
                if let Some(b) = else_b { walk_block(sym, b, ret, funcs, errors); }
            }
            HStmt::While { cond, body, .. } => { walk_expr(sym, cond, ret, funcs, errors); walk_block(sym, body, ret, funcs, errors); }
            HStmt::Block(b) | HStmt::Unsafe(b, _) => walk_block(sym, b, ret, funcs, errors),
            HStmt::Propagate { value: Some(v), .. } => walk_expr(sym, v, ret, funcs, errors),
            HStmt::Propagate { value: None, .. } => {}
            HStmt::ForC { init, cond, step, body, .. } => {
                walk_stmt(sym, init, ret, funcs, errors);
                walk_expr(sym, cond, ret, funcs, errors);
                walk_stmt(sym, step, ret, funcs, errors);
                walk_block(sym, body, ret, funcs, errors);
            }
            HStmt::ForEach { src, body, .. } => {
                walk_expr(sym, src, ret, funcs, errors);
                walk_block(sym, body, ret, funcs, errors);
            }
            HStmt::Break { .. } | HStmt::Continue { .. } => {}
        }
    }
    fn walk_expr(sym: &SymTab, e: &HExpr, ret: &HType, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
        if let HExprKind::InlineCall { callee, .. } = &e.kind {
            // Walk the inline's body for `propagate` — transitively through any nested
            // InlineCalls.  Cycle protection comes from the inline-recursion detector
            // (a cycle would already have errored), but we still track `visited` so an
            // unrelated DAG with shared children doesn't repeat-report the same error.
            if let Some(inline_f) = funcs.iter().find(|hf| hf.id == *callee) {
                let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
                visited.insert(callee.0);
                check_propagate_in_block(&inline_f.body, ret, errors, e.span, funcs, &mut visited);
            }
        }
        // Recurse:
        match &e.kind {
            HExprKind::Bin { lhs, rhs, .. } => { walk_expr(sym, lhs, ret, funcs, errors); walk_expr(sym, rhs, ret, funcs, errors); }
            HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } => walk_expr(sym, expr, ret, funcs, errors),
            HExprKind::AddrOfRef { place, .. } => walk_expr(sym, place, ret, funcs, errors),
            HExprKind::Field { base, .. } => walk_expr(sym, base, ret, funcs, errors),
            HExprKind::Index { base, idx } => { walk_expr(sym, base, ret, funcs, errors); walk_expr(sym, idx, ret, funcs, errors); }
            HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => for a in args { walk_expr(sym, a, ret, funcs, errors); },
            HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr) => walk_expr(sym, expr, ret, funcs, errors),
            HExprKind::ArrayToSlice { base, .. } => walk_expr(sym, base, ret, funcs, errors),
            HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => for (_, fe) in fields { walk_expr(sym, fe, ret, funcs, errors); },
            HExprKind::ArrayLit(es) => for ee in es { walk_expr(sym, ee, ret, funcs, errors); },
            HExprKind::HeapAlloc(inner) => walk_expr(sym, inner, ret, funcs, errors),
            HExprKind::CallIndirect { callee, args } => { walk_expr(sym, callee, ret, funcs, errors); for a in args { walk_expr(sym, a, ret, funcs, errors); } }
            HExprKind::Closure { env_values, .. } => for v in env_values { walk_expr(sym, v, ret, funcs, errors); },
            HExprKind::Transfer(inner) | HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => walk_expr(sym, inner, ret, funcs, errors),
            _ => {}
        }
    }
    fn check_propagate_in_block(
        b: &HBlock, ret: &HType, errs: &mut Vec<SemaError>, call_span: maka_lexer::Span,
        funcs: &[HFunc], visited: &mut std::collections::HashSet<u32>,
    ) {
        for s in &b.stmts { check_propagate_in_stmt(s, ret, errs, call_span, funcs, visited); }
    }
    fn check_propagate_in_stmt(
        s: &HStmt, ret: &HType, errs: &mut Vec<SemaError>, call_span: maka_lexer::Span,
        funcs: &[HFunc], visited: &mut std::collections::HashSet<u32>,
    ) {
        match s {
            HStmt::Propagate { value, span, .. } => {
                match value {
                    Some(v) => {
                        if !type_eq(&v.ty, ret) {
                            errs.push(SemaError {
                                msg: format!(
                                    "`propagate` value type `{}` does not match the caller's return type `{}`",
                                    type_str(&v.ty), type_str(ret)
                                ),
                                span: if span.line != 0 { *span } else { call_span },
                            });
                        }
                        check_propagate_in_expr(v, ret, errs, call_span, funcs, visited);
                    }
                    None => {
                        // value-less `propagate;` is only valid when the caller returns unit.
                        if !matches!(ret, HType::Unit) {
                            errs.push(SemaError {
                                msg: format!(
                                    "`propagate;` (no value) requires the caller to return `unit`, but it returns `{}`",
                                    type_str(ret),
                                ),
                                span: if span.line != 0 { *span } else { call_span },
                            });
                        }
                    }
                }
            }
            HStmt::If { cond, then_b, else_b, .. } => {
                check_propagate_in_expr(cond, ret, errs, call_span, funcs, visited);
                check_propagate_in_block(then_b, ret, errs, call_span, funcs, visited);
                if let Some(b) = else_b { check_propagate_in_block(b, ret, errs, call_span, funcs, visited); }
            }
            HStmt::While { cond, body, .. } => {
                check_propagate_in_expr(cond, ret, errs, call_span, funcs, visited);
                check_propagate_in_block(body, ret, errs, call_span, funcs, visited);
            }
            HStmt::Block(body) | HStmt::Unsafe(body, _) => check_propagate_in_block(body, ret, errs, call_span, funcs, visited),
            HStmt::ForC { init, cond, step, body, .. } => {
                check_propagate_in_stmt(init, ret, errs, call_span, funcs, visited);
                check_propagate_in_expr(cond, ret, errs, call_span, funcs, visited);
                check_propagate_in_stmt(step, ret, errs, call_span, funcs, visited);
                check_propagate_in_block(body, ret, errs, call_span, funcs, visited);
            }
            HStmt::ForEach { src, body, .. } => {
                check_propagate_in_expr(src, ret, errs, call_span, funcs, visited);
                check_propagate_in_block(body, ret, errs, call_span, funcs, visited);
            }
            HStmt::Let { init, .. } => check_propagate_in_expr(init, ret, errs, call_span, funcs, visited),
            HStmt::Assign { place, value, .. } => {
                check_propagate_in_expr(place, ret, errs, call_span, funcs, visited);
                check_propagate_in_expr(value, ret, errs, call_span, funcs, visited);
            }
            HStmt::ExprStmt(e) => check_propagate_in_expr(e, ret, errs, call_span, funcs, visited),
            HStmt::Return { value: Some(v), .. } => check_propagate_in_expr(v, ret, errs, call_span, funcs, visited),
            _ => {}
        }
    }
    /// Walk an expression looking for nested `InlineCall`s; recurse into each
    /// inline callee's body so transitively-propagated values are still checked
    /// against the outermost non-inline caller's return type.
    fn check_propagate_in_expr(
        e: &HExpr, ret: &HType, errs: &mut Vec<SemaError>, call_span: maka_lexer::Span,
        funcs: &[HFunc], visited: &mut std::collections::HashSet<u32>,
    ) {
        if let HExprKind::InlineCall { callee, .. } = &e.kind {
            if visited.insert(callee.0) {
                if let Some(inline_f) = funcs.iter().find(|hf| hf.id == *callee) {
                    check_propagate_in_block(&inline_f.body, ret, errs, call_span, funcs, visited);
                }
            }
        }
        match &e.kind {
            HExprKind::Bin { lhs, rhs, .. } => {
                check_propagate_in_expr(lhs, ret, errs, call_span, funcs, visited);
                check_propagate_in_expr(rhs, ret, errs, call_span, funcs, visited);
            }
            HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } => check_propagate_in_expr(expr, ret, errs, call_span, funcs, visited),
            HExprKind::AddrOfRef { place, .. } => check_propagate_in_expr(place, ret, errs, call_span, funcs, visited),
            HExprKind::Field { base, .. } => check_propagate_in_expr(base, ret, errs, call_span, funcs, visited),
            HExprKind::Index { base, idx } => {
                check_propagate_in_expr(base, ret, errs, call_span, funcs, visited);
                check_propagate_in_expr(idx, ret, errs, call_span, funcs, visited);
            }
            HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => {
                for a in args { check_propagate_in_expr(a, ret, errs, call_span, funcs, visited); }
            }
            HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. }
            | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr) => check_propagate_in_expr(expr, ret, errs, call_span, funcs, visited),
            HExprKind::ArrayToSlice { base, .. } => check_propagate_in_expr(base, ret, errs, call_span, funcs, visited),
            HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => {
                for (_, fe) in fields { check_propagate_in_expr(fe, ret, errs, call_span, funcs, visited); }
            }
            HExprKind::ArrayLit(es) => for ee in es { check_propagate_in_expr(ee, ret, errs, call_span, funcs, visited); },
            HExprKind::HeapAlloc(inner) => check_propagate_in_expr(inner, ret, errs, call_span, funcs, visited),
            HExprKind::CallIndirect { callee, args } => {
                check_propagate_in_expr(callee, ret, errs, call_span, funcs, visited);
                for a in args { check_propagate_in_expr(a, ret, errs, call_span, funcs, visited); }
            }
            HExprKind::Closure { env_values, .. } => for v in env_values { check_propagate_in_expr(v, ret, errs, call_span, funcs, visited); },
            HExprKind::Transfer(inner) | HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => check_propagate_in_expr(inner, ret, errs, call_span, funcs, visited),
            _ => {}
        }
    }
}

/// Does this inline body contain a `break`/`continue` that targets an enclosing
/// loop (i.e. one NOT inside a loop within the inline)?  Such a jump is only valid
/// when the inline is spliced into a caller's loop.  Recurses into if/block/unsafe
/// and match-arm bodies (a jump there targets the enclosing loop) but NOT into a
/// nested loop's body (its breaks are that loop's).  Closures are separate
/// functions, so their captured exprs are scanned but not their bodies.
fn inline_jumps_out_block(b: &HBlock) -> bool { b.stmts.iter().any(inline_jumps_out_stmt) }
fn inline_jumps_out_stmt(s: &HStmt) -> bool {
    match s {
        HStmt::Break { .. } | HStmt::Continue { .. } => true,
        HStmt::If { cond, then_b, else_b, .. } => inline_jumps_out_expr(cond) || inline_jumps_out_block(then_b) || else_b.as_ref().is_some_and(inline_jumps_out_block),
        HStmt::Block(b) | HStmt::Unsafe(b, _) => inline_jumps_out_block(b),
        HStmt::While { cond, .. } | HStmt::ForC { cond, .. } => inline_jumps_out_expr(cond),
        HStmt::ForEach { src, .. } => inline_jumps_out_expr(src),
        HStmt::Let { init, .. } => inline_jumps_out_expr(init),
        HStmt::Assign { place, value, .. } => inline_jumps_out_expr(place) || inline_jumps_out_expr(value),
        HStmt::ExprStmt(e) => inline_jumps_out_expr(e),
        HStmt::Return { value, .. } | HStmt::Propagate { value, .. } => value.as_ref().is_some_and(inline_jumps_out_expr),
    }
}
fn inline_jumps_out_expr(e: &HExpr) -> bool {
    match &e.kind {
        HExprKind::Match { scrutinee, arms, .. } => inline_jumps_out_expr(scrutinee) || arms.iter().any(|a|
            a.guard.as_ref().is_some_and(inline_jumps_out_expr) || inline_jumps_out_block(&a.body) || a.value.as_ref().is_some_and(inline_jumps_out_expr)),
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => args.iter().any(inline_jumps_out_expr),
        HExprKind::CallIndirect { callee, args } => inline_jumps_out_expr(callee) || args.iter().any(inline_jumps_out_expr),
        HExprKind::Bin { lhs, rhs, .. } => inline_jumps_out_expr(lhs) || inline_jumps_out_expr(rhs),
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } | HExprKind::Cast { expr, .. }
        | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr)
        | HExprKind::HeapAlloc(expr) | HExprKind::Free(expr, _) | HExprKind::SliceLen(expr)
        | HExprKind::EnumTag(expr) | HExprKind::Transfer(expr) | HExprKind::ArrayToSlice { base: expr, .. }
        | HExprKind::AddrOfRef { place: expr, .. } | HExprKind::Field { base: expr, .. } => inline_jumps_out_expr(expr),
        HExprKind::Index { base, idx } => inline_jumps_out_expr(base) || inline_jumps_out_expr(idx),
        HExprKind::Closure { env_values, .. } => env_values.iter().any(inline_jumps_out_expr),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => fields.iter().any(|(_, fe)| inline_jumps_out_expr(fe)),
        HExprKind::ArrayLit(es) => es.iter().any(inline_jumps_out_expr),
        _ => false,
    }
}

/// An inline whose body break/continues out to an enclosing loop is a loop fragment:
/// it is only valid spliced into a loop.  Reject calling it at loop depth 0 in a
/// (non-inline) caller, with a clear message at the call site - otherwise the
/// spliced jump becomes a C `break`/`continue` with no enclosing loop (an opaque
/// backend error).  Inline callers are skipped: a jump there just propagates to
/// THEIR caller, and is checked when the chain bottoms out at a non-inline call.
fn check_inline_loop_jumps(sym: &SymTab, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
    for caller in funcs {
        if sym.func_sig(caller.id).is_inline { continue; }
        wblock(&caller.body, 0, funcs, errors);
    }
    fn wblock(b: &HBlock, depth: usize, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
        for s in &b.stmts { wstmt(s, depth, funcs, errors); }
    }
    fn wstmt(s: &HStmt, depth: usize, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
        match s {
            HStmt::Let { init, .. } => wexpr(init, depth, funcs, errors),
            HStmt::Assign { place, value, .. } => { wexpr(place, depth, funcs, errors); wexpr(value, depth, funcs, errors); }
            HStmt::ExprStmt(e) => wexpr(e, depth, funcs, errors),
            HStmt::Return { value: Some(v), .. } | HStmt::Propagate { value: Some(v), .. } => wexpr(v, depth, funcs, errors),
            HStmt::If { cond, then_b, else_b, .. } => { wexpr(cond, depth, funcs, errors); wblock(then_b, depth, funcs, errors); if let Some(b) = else_b { wblock(b, depth, funcs, errors); } }
            HStmt::While { cond, body, .. } => { wexpr(cond, depth, funcs, errors); wblock(body, depth + 1, funcs, errors); }
            HStmt::ForC { init, cond, step, body, .. } => { wstmt(init, depth, funcs, errors); wexpr(cond, depth, funcs, errors); wstmt(step, depth, funcs, errors); wblock(body, depth + 1, funcs, errors); }
            HStmt::ForEach { src, body, .. } => { wexpr(src, depth, funcs, errors); wblock(body, depth + 1, funcs, errors); }
            HStmt::Block(b) | HStmt::Unsafe(b, _) => wblock(b, depth, funcs, errors),
            _ => {}
        }
    }
    fn wexpr(e: &HExpr, depth: usize, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
        if let HExprKind::InlineCall { callee, .. } = &e.kind {
            if depth == 0 {
                if let Some(inline_f) = funcs.iter().find(|hf| hf.id == *callee) {
                    if inline_jumps_out_block(&inline_f.body) {
                        errors.push(SemaError {
                            msg: "this inline function uses a `break`/`continue` that targets an enclosing loop, but it is called outside any loop here; call it inside a `for`/`while`, or make the break/continue stay within a loop inside the inline".to_string(),
                            span: e.span,
                        });
                    }
                }
            }
        }
        match &e.kind {
            HExprKind::Match { scrutinee, arms, .. } => {
                wexpr(scrutinee, depth, funcs, errors);
                for a in arms {
                    if let Some(g) = &a.guard { wexpr(g, depth, funcs, errors); }
                    wblock(&a.body, depth, funcs, errors);
                    if let Some(v) = &a.value { wexpr(v, depth, funcs, errors); }
                }
            }
            HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => for a in args { wexpr(a, depth, funcs, errors); },
            HExprKind::CallIndirect { callee, args } => { wexpr(callee, depth, funcs, errors); for a in args { wexpr(a, depth, funcs, errors); } }
            HExprKind::Bin { lhs, rhs, .. } => { wexpr(lhs, depth, funcs, errors); wexpr(rhs, depth, funcs, errors); }
            HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } | HExprKind::Cast { expr, .. }
            | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr)
            | HExprKind::HeapAlloc(expr) | HExprKind::Free(expr, _) | HExprKind::SliceLen(expr)
            | HExprKind::EnumTag(expr) | HExprKind::Transfer(expr) | HExprKind::ArrayToSlice { base: expr, .. }
            | HExprKind::AddrOfRef { place: expr, .. } | HExprKind::Field { base: expr, .. } => wexpr(expr, depth, funcs, errors),
            HExprKind::Index { base, idx } => { wexpr(base, depth, funcs, errors); wexpr(idx, depth, funcs, errors); }
            HExprKind::Closure { env_values, .. } => for v in env_values { wexpr(v, depth, funcs, errors); },
            HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => for (_, fe) in fields { wexpr(fe, depth, funcs, errors); },
            HExprKind::ArrayLit(es) => for ee in es { wexpr(ee, depth, funcs, errors); },
            _ => {}
        }
    }
}

/// Can a value of this type cross the C ABI by value as an `export` parameter or
/// return - i.e. does Rust/C see a type it can declare and pass?  Scalars, `char`,
/// `unit` (-> void), `string` (-> char*), and any pointer/reference (a raw address)
/// qualify.  Owning pointers (ownership can't transfer across the ABI), by-value
/// aggregates (Vec / String / structs / arrays / slices), enums, closures, and
/// existentials do NOT - pass those behind a `*T`/`raw *T` pointer instead.
fn is_c_abi_type(t: &HType) -> bool {
    matches!(t,
        HType::Int | HType::Float | HType::Bool | HType::Char | HType::Unit
        | HType::SizedInt { .. } | HType::SizedFloat { .. } | HType::Str
        | HType::Ptr { .. } | HType::RawPtr { .. } | HType::Ref { .. })
}

/// Verify every `export` function has a C-ABI-crossable signature and no feature
/// that would defeat a stable, single, unmangled C symbol (generics monomorphize
/// to many mangled symbols; `inline` emits no standalone symbol).
fn check_exports(sym: &SymTab, errors: &mut Vec<SemaError>) {
    for sig in &sym.sigs {
        if !sig.is_export { continue; }
        let sp = Span { start: 0, end: 0, line: 0, col: 0 };
        if !sig.type_params.is_empty() {
            errors.push(SemaError { msg: format!(
                "`export` function `{}` cannot be generic - a monomorphized function has no single stable C symbol",
                sig.name), span: sp });
        }
        if sig.is_inline {
            errors.push(SemaError { msg: format!(
                "`export` function `{}` cannot be `inline` - an inlined function emits no standalone C symbol",
                sig.name), span: sp });
        }
        for (pt, pn) in sig.param_tys.iter().zip(sig.param_names.iter()) {
            if !is_c_abi_type(pt) {
                errors.push(SemaError { msg: format!(
                    "`export` function `{}`: parameter `{}` has type `{}`, which does not cross the C ABI. Use a scalar, `string`, or a `*T`/`raw *T` pointer (owning/aggregate/enum/closure types can't be passed by value to C).",
                    sig.name, pn, type_str(pt)), span: sp });
            }
        }
        if !is_c_abi_type(&sig.ret) {
            errors.push(SemaError { msg: format!(
                "`export` function `{}`: return type `{}` does not cross the C ABI. Return a scalar, `string`, or a `*T`/`raw *T` pointer.",
                sig.name, type_str(&sig.ret)), span: sp });
        }
    }
}

/// The root module global a place ultimately reads/writes through, if any.
fn place_root_global(e: &HExpr) -> Option<u32> {
    match &e.kind {
        HExprKind::GlobalRef(g) => Some(g.0),
        HExprKind::Field { base, .. } | HExprKind::Index { base, .. } => place_root_global(base),
        HExprKind::Unwrap { expr, .. } | HExprKind::DerefRef(expr) => place_root_global(expr),
        _ => None,
    }
}

/// Is `c` a REAL-parallelism spawn tier (own OS thread / job / background pool)?
/// The fiber tier (`spawn`, MAX-3) multiplexes cooperatively on one thread and
/// never races, so it is excluded.
fn is_real_thread_callee(c: FuncId) -> bool {
    c.0 == u32::MAX - 15 || c.0 == u32::MAX - 16 || c.0 == u32::MAX - 37
}

/// The lifted body FuncId a spawn's callable argument runs, unwrapping the
/// Callable coercion / transfer wrapper.  A capturing closure is `Closure { lifted }`;
/// a non-capturing one is a bare `FnRef(fid)`.
fn spawn_target_fid(e: &HExpr) -> Option<u32> {
    match &e.kind {
        HExprKind::Closure { lifted, .. } => Some(lifted.0),
        HExprKind::FnRef(fid) => Some(fid.0),
        HExprKind::Cast { expr, .. } | HExprKind::Transfer(expr) => spawn_target_fid(expr),
        _ => None,
    }
}

/// Collect, for one function body: the mut-global ids it DIRECTLY writes (an
/// assignment to a global place, or a `&mut` borrow of one - a borrow hands out
/// write capability), the real functions it DIRECTLY calls (for the transitive
/// fixpoint), and every real-thread spawn site as `(lifted closure fid, span)`.
fn collect_thread_effects(
    sym: &SymTab, b: &HBlock,
    writes: &mut std::collections::HashSet<u32>,
    calls: &mut std::collections::HashSet<u32>,
    spawns: &mut Vec<(u32, Span)>,
) {
    for s in &b.stmts { collect_thread_effects_stmt(sym, s, writes, calls, spawns); }
}

fn collect_thread_effects_stmt(
    sym: &SymTab, s: &HStmt,
    writes: &mut std::collections::HashSet<u32>,
    calls: &mut std::collections::HashSet<u32>,
    spawns: &mut Vec<(u32, Span)>,
) {
    match s {
        HStmt::Let { init, .. } => collect_thread_effects_expr(sym, init, writes, calls, spawns),
        HStmt::Assign { place, value, .. } => {
            if let Some(g) = place_root_global(place) { writes.insert(g); }
            collect_thread_effects_expr(sym, place, writes, calls, spawns);
            collect_thread_effects_expr(sym, value, writes, calls, spawns);
        }
        HStmt::ExprStmt(e) => collect_thread_effects_expr(sym, e, writes, calls, spawns),
        HStmt::Return { value: Some(v), .. } | HStmt::Propagate { value: Some(v), .. } =>
            collect_thread_effects_expr(sym, v, writes, calls, spawns),
        HStmt::Return { .. } | HStmt::Propagate { .. } | HStmt::Break { .. } | HStmt::Continue { .. } => {}
        HStmt::If { cond, then_b, else_b, .. } => {
            collect_thread_effects_expr(sym, cond, writes, calls, spawns);
            collect_thread_effects(sym, then_b, writes, calls, spawns);
            if let Some(b) = else_b { collect_thread_effects(sym, b, writes, calls, spawns); }
        }
        HStmt::While { cond, body, .. } => {
            collect_thread_effects_expr(sym, cond, writes, calls, spawns);
            collect_thread_effects(sym, body, writes, calls, spawns);
        }
        HStmt::Block(b) | HStmt::Unsafe(b, _) => collect_thread_effects(sym, b, writes, calls, spawns),
        HStmt::ForC { init, cond, step, body, .. } => {
            collect_thread_effects_stmt(sym, init, writes, calls, spawns);
            collect_thread_effects_expr(sym, cond, writes, calls, spawns);
            collect_thread_effects_stmt(sym, step, writes, calls, spawns);
            collect_thread_effects(sym, body, writes, calls, spawns);
        }
        HStmt::ForEach { src, body, .. } => {
            collect_thread_effects_expr(sym, src, writes, calls, spawns);
            collect_thread_effects(sym, body, writes, calls, spawns);
        }
    }
}

fn collect_thread_effects_expr(
    sym: &SymTab, e: &HExpr,
    writes: &mut std::collections::HashSet<u32>,
    calls: &mut std::collections::HashSet<u32>,
    spawns: &mut Vec<(u32, Span)>,
) {
    // A `&mut` borrow of a global place grants write capability across whatever
    // receives it, so treat it as a write of that global.
    if let HExprKind::AddrOfRef { mutable: true, place } = &e.kind {
        if let Some(g) = place_root_global(place) { writes.insert(g); }
    }
    match &e.kind {
        HExprKind::Call { callee, args } => {
            // Real-thread spawn site: the callable argument's body is what runs on
            // the new thread.  A capturing closure is a `Closure { lifted }`; a
            // non-capturing one lowers to a bare `FnRef(fid)`.
            if is_real_thread_callee(*callee) {
                if let Some(fid) = args.first().and_then(spawn_target_fid) {
                    spawns.push((fid, e.span));
                }
            } else if (callee.0 as usize) < sym.sigs.len() {
                calls.insert(callee.0);
            }
            for a in args { collect_thread_effects_expr(sym, a, writes, calls, spawns); }
        }
        HExprKind::InlineCall { callee, args, .. } => {
            if (callee.0 as usize) < sym.sigs.len() { calls.insert(callee.0); }
            for a in args { collect_thread_effects_expr(sym, a, writes, calls, spawns); }
        }
        HExprKind::CallIndirect { callee, args } => {
            collect_thread_effects_expr(sym, callee, writes, calls, spawns);
            for a in args { collect_thread_effects_expr(sym, a, writes, calls, spawns); }
        }
        HExprKind::Bin { lhs, rhs, .. } => {
            collect_thread_effects_expr(sym, lhs, writes, calls, spawns);
            collect_thread_effects_expr(sym, rhs, writes, calls, spawns);
        }
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. }
        | HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. }
        | HExprKind::DropWrite(expr) => collect_thread_effects_expr(sym, expr, writes, calls, spawns),
        HExprKind::AddrOfRef { place, .. } => collect_thread_effects_expr(sym, place, writes, calls, spawns),
        HExprKind::Field { base, .. } | HExprKind::ArrayToSlice { base, .. } =>
            collect_thread_effects_expr(sym, base, writes, calls, spawns),
        HExprKind::Index { base, idx } => {
            collect_thread_effects_expr(sym, base, writes, calls, spawns);
            collect_thread_effects_expr(sym, idx, writes, calls, spawns);
        }
        HExprKind::DerefRef(inner) | HExprKind::HeapAlloc(inner) | HExprKind::Free(inner, _)
        | HExprKind::Transfer(inner) | HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) =>
            collect_thread_effects_expr(sym, inner, writes, calls, spawns),
        HExprKind::Closure { env_values, .. } =>
            for v in env_values { collect_thread_effects_expr(sym, v, writes, calls, spawns); },
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } =>
            for (_, fe) in fields { collect_thread_effects_expr(sym, fe, writes, calls, spawns); },
        HExprKind::ArrayLit(es) => for x in es { collect_thread_effects_expr(sym, x, writes, calls, spawns); },
        HExprKind::Match { scrutinee, arms, .. } => {
            collect_thread_effects_expr(sym, scrutinee, writes, calls, spawns);
            for a in arms {
                if let Some(g) = &a.guard { collect_thread_effects_expr(sym, g, writes, calls, spawns); }
                collect_thread_effects(sym, &a.body, writes, calls, spawns);
                if let Some(v) = &a.value { collect_thread_effects_expr(sym, v, writes, calls, spawns); }
            }
        }
        _ => {}
    }
}

/// A `mut` module global written (transitively) from inside a real OS-thread /
/// job / pool body is an unsynchronized data race: the cross-thread capture check
/// never inspects globals (they are referenced by name, not captured), so two
/// threads doing an unlocked read-modify-write on one static otherwise compiles
/// clean.  Compute each function's transitive mut-global write set by fixpoint
/// over the call graph, then reject any real-thread spawn whose body writes one.
fn check_cross_thread_global_races(sym: &SymTab, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
    use std::collections::{HashMap, HashSet};
    let mut writes: HashMap<u32, HashSet<u32>> = HashMap::new();
    let mut calls: HashMap<u32, HashSet<u32>> = HashMap::new();
    let mut spawns: Vec<(u32, Span)> = Vec::new();
    for f in funcs {
        let mut w = HashSet::new();
        let mut c = HashSet::new();
        collect_thread_effects(sym, &f.body, &mut w, &mut c, &mut spawns);
        writes.insert(f.id.0, w);
        calls.insert(f.id.0, c);
    }
    // Fixpoint: a function transitively writes what it writes directly plus what
    // anything it calls transitively writes.  Iterate to convergence (handles
    // mutual recursion).
    let mut changed = true;
    while changed {
        changed = false;
        let fids: Vec<u32> = calls.keys().copied().collect();
        for fid in fids {
            let callee_writes: HashSet<u32> = calls[&fid].iter()
                .filter_map(|c| writes.get(c))
                .flatten()
                .copied()
                .collect();
            let entry = writes.get_mut(&fid).unwrap();
            for g in callee_writes {
                if entry.insert(g) { changed = true; }
            }
        }
    }
    // Report each racy global once, at the first spawn site that writes it.
    let mut reported: HashSet<u32> = HashSet::new();
    for (lifted, span) in &spawns {
        if let Some(gs) = writes.get(lifted) {
            let mut gids: Vec<u32> = gs.iter().copied().collect();
            gids.sort_unstable();
            for gid in gids {
                let g = &sym.globals[gid as usize];
                if g.is_mut && reported.insert(gid) {
                    errors.push(SemaError {
                        msg: format!(
                            "data race: mutable global `{}` is written from a thread body. \
                             Concurrent unsynchronized access to a non-atomic mutable global is undefined behavior. \
                             Make it an `Atomic<{}>`, pass the state through a channel, or confine it to one thread.",
                            g.name, type_str(&g.ty)),
                        span: *span,
                    });
                }
            }
        }
    }
}

/// Build a directed call graph over inline functions (callee is inline). Detect cycles.
fn detect_inline_recursion(sym: &SymTab, funcs: &[HFunc], errors: &mut Vec<SemaError>) {
    use std::collections::HashMap;
    // Edges: inline_callee → inline_caller (we record edges from caller to callee).
    let mut graph: HashMap<u32, Vec<u32>> = HashMap::new();
    for f in funcs {
        if !sym.func_sig(f.id).is_inline { continue; }
        let mut callees = Vec::new();
        collect_inline_callees(sym, &f.body, &mut callees);
        graph.insert(f.id.0, callees);
    }
    // Cycle detection via DFS.
    let mut visited: HashMap<u32, u8> = HashMap::new();
    fn dfs(graph: &HashMap<u32, Vec<u32>>, node: u32, visited: &mut HashMap<u32, u8>) -> bool {
        if let Some(&c) = visited.get(&node) {
            if c == 1 { return true; } // cycle
            if c == 2 { return false; } // already cleared
        }
        visited.insert(node, 1);
        if let Some(succs) = graph.get(&node) {
            for &s in succs {
                if dfs(graph, s, visited) { return true; }
            }
        }
        visited.insert(node, 2);
        false
    }
    for &start in graph.keys() {
        if dfs(&graph, start, &mut visited) {
            let f = funcs.iter().find(|f| f.id.0 == start).unwrap();
            errors.push(SemaError {
                msg: format!("inline function `{}` participates in a cycle; inline recursion is forbidden", f.name),
                span: f.span,
            });
            // Don't keep reporting the same cycle.
            break;
        }
    }
}

fn collect_inline_callees(sym: &SymTab, b: &HBlock, out: &mut Vec<u32>) {
    for s in &b.stmts { collect_inline_callees_stmt(sym, s, out); }
}
fn collect_inline_callees_stmt(sym: &SymTab, s: &HStmt, out: &mut Vec<u32>) {
    match s {
        HStmt::Let { init, .. } => collect_inline_callees_expr(sym, init, out),
        HStmt::Assign { place, value, .. } => { collect_inline_callees_expr(sym, place, out); collect_inline_callees_expr(sym, value, out); }
        HStmt::ExprStmt(e) => collect_inline_callees_expr(sym, e, out),
        HStmt::Return { value: Some(v), .. } => collect_inline_callees_expr(sym, v, out),
        HStmt::Return { .. } => {}
        HStmt::If { cond, then_b, else_b, .. } => {
            collect_inline_callees_expr(sym, cond, out);
            collect_inline_callees(sym, then_b, out);
            if let Some(b) = else_b { collect_inline_callees(sym, b, out); }
        }
        HStmt::While { cond, body, .. } => { collect_inline_callees_expr(sym, cond, out); collect_inline_callees(sym, body, out); }
        HStmt::Block(b) | HStmt::Unsafe(b, _) => collect_inline_callees(sym, b, out),
        HStmt::Propagate { value: Some(v), .. } => collect_inline_callees_expr(sym, v, out),
        HStmt::Propagate { value: None, .. } => {}
        HStmt::ForC { init, cond, step, body, .. } => {
            collect_inline_callees_stmt(sym, init, out);
            collect_inline_callees_expr(sym, cond, out);
            collect_inline_callees_stmt(sym, step, out);
            collect_inline_callees(sym, body, out);
        }
        HStmt::ForEach { src, body, .. } => {
            collect_inline_callees_expr(sym, src, out);
            collect_inline_callees(sym, body, out);
        }
        HStmt::Break { .. } | HStmt::Continue { .. } => {}
    }
}
fn collect_inline_callees_expr(sym: &SymTab, e: &HExpr, out: &mut Vec<u32>) {
    match &e.kind {
        HExprKind::Call { callee, args } | HExprKind::InlineCall { callee, args, .. } => {
            if (callee.0 as usize) < sym.sigs.len() && sym.func_sig(*callee).is_inline {
                out.push(callee.0);
            }
            for a in args { collect_inline_callees_expr(sym, a, out); }
        }
        HExprKind::Bin { lhs, rhs, .. } => { collect_inline_callees_expr(sym, lhs, out); collect_inline_callees_expr(sym, rhs, out); }
        HExprKind::Un { expr, .. } => collect_inline_callees_expr(sym, expr, out),
        HExprKind::Unwrap { expr, .. } => collect_inline_callees_expr(sym, expr, out),
        HExprKind::AddrOfRef { place, .. } => collect_inline_callees_expr(sym, place, out),
        HExprKind::Field { base, .. } => collect_inline_callees_expr(sym, base, out),
        HExprKind::Index { base, idx } => { collect_inline_callees_expr(sym, base, out); collect_inline_callees_expr(sym, idx, out); }
        HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr) => collect_inline_callees_expr(sym, expr, out),
        HExprKind::ArrayToSlice { base, .. } => collect_inline_callees_expr(sym, base, out),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => for (_, e) in fields { collect_inline_callees_expr(sym, e, out); },
        HExprKind::ArrayLit(es) => for e in es { collect_inline_callees_expr(sym, e, out); },
        HExprKind::Match { scrutinee, arms, .. } => {
            collect_inline_callees_expr(sym, scrutinee, out);
            for a in arms {
                if let Some(g) = &a.guard { collect_inline_callees_expr(sym, g, out); }
                // Arm BODY statements too, not just the arm value: a generic inline
                // called inside an arm body (`int x = id(5); yield x;`) must still be
                // collected for monomorphization, or codegen can't find the instance
                // and emits a broken (MAKA_UNIT) expansion.
                for s in &a.body.stmts { collect_inline_callees_stmt(sym, s, out); }
                if let Some(v) = &a.value { collect_inline_callees_expr(sym, v, out); }
            }
        }
        HExprKind::HeapAlloc(inner) => collect_inline_callees_expr(sym, inner, out),
        HExprKind::CallIndirect { callee, args } => {
            collect_inline_callees_expr(sym, callee, out);
            for a in args { collect_inline_callees_expr(sym, a, out); }
        }
        HExprKind::Closure { env_values, .. } => for v in env_values { collect_inline_callees_expr(sym, v, out); },
        HExprKind::Transfer(inner) | HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => collect_inline_callees_expr(sym, inner, out),
        _ => {}
    }
}

/// Base for generic-instantiation placeholder FuncIds.  Placeholders count DOWN
/// from here (`BASE - req_idx`).  Kept far below the builtin sentinel range
/// (`u32::MAX - 0..~1024`, used for log/panic/concat/format/...) so a generic
/// call's placeholder is never mistaken for a builtin by a sentinel check.
pub const PLACEHOLDER_FID_BASE: u32 = u32::MAX - 0x0010_0000;

/// Replace placeholder FuncIds inside Call expressions with real ones.
fn rewrite_placeholders(f: &mut HFunc, mapping: &[u32]) {
    fn rw_block(b: &mut HBlock, mapping: &[u32]) {
        for s in &mut b.stmts {
            rw_stmt(s, mapping);
        }
    }
    fn rw_stmt(s: &mut HStmt, mapping: &[u32]) {
        match s {
            HStmt::Let { init, .. } => rw_expr(init, mapping),
            HStmt::Assign { place, value, .. } => { rw_expr(place, mapping); rw_expr(value, mapping); }
            HStmt::ExprStmt(e) => rw_expr(e, mapping),
            HStmt::Return { value, .. } => if let Some(v) = value { rw_expr(v, mapping); },
            HStmt::If { cond, then_b, else_b, .. } => {
                rw_expr(cond, mapping);
                rw_block(then_b, mapping);
                if let Some(b) = else_b { rw_block(b, mapping); }
            }
            HStmt::While { cond, body, .. } => { rw_expr(cond, mapping); rw_block(body, mapping); }
            HStmt::Block(b) | HStmt::Unsafe(b, _) => rw_block(b, mapping),
            HStmt::Break { .. } | HStmt::Continue { .. } => {}
            HStmt::ForC { init, cond, step, body, .. } => {
                rw_stmt(init, mapping);
                rw_expr(cond, mapping);
                rw_stmt(step, mapping);
                rw_block(body, mapping);
            }
            HStmt::ForEach { src, body, .. } => {
                rw_expr(src, mapping);
                rw_block(body, mapping);
            }
            HStmt::Propagate { value: Some(v), .. } => rw_expr(v, mapping),
            HStmt::Propagate { value: None, .. } => {}
        }
    }
    fn rw_expr(e: &mut HExpr, mapping: &[u32]) {
        match &mut e.kind {
            HExprKind::Call { callee, args } => {
                let v = callee.0;
                if v <= PLACEHOLDER_FID_BASE && v + (mapping.len() as u32) > PLACEHOLDER_FID_BASE {
                    let idx = (PLACEHOLDER_FID_BASE - v) as usize;
                    if idx < mapping.len() {
                        *callee = FuncId(mapping[idx]);
                    }
                }
                for a in args { rw_expr(a, mapping); }
            }
            HExprKind::Bin { lhs, rhs, .. } => { rw_expr(lhs, mapping); rw_expr(rhs, mapping); }
            HExprKind::Un { expr, .. } => rw_expr(expr, mapping),
            HExprKind::Unwrap { expr, .. } => rw_expr(expr, mapping),
            HExprKind::AddrOfRef { place, .. } => rw_expr(place, mapping),
            HExprKind::Field { base, .. } => rw_expr(base, mapping),
            HExprKind::Index { base, idx } => { rw_expr(base, mapping); rw_expr(idx, mapping); }
            HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr) => rw_expr(expr, mapping),
            HExprKind::ArrayToSlice { base, .. } => rw_expr(base, mapping),
            HExprKind::Struct { fields, .. } => for (_, fe) in fields { rw_expr(fe, mapping); },
            HExprKind::VariantCtor { fields, .. } => for (_, fe) in fields { rw_expr(fe, mapping); },
            HExprKind::Match { scrutinee, arms, .. } => {
                rw_expr(scrutinee, mapping);
                for a in arms {
                    if let Some(g) = &mut a.guard { rw_expr(g, mapping); }
                    // Rewrite the arm BODY too: a placeholder FuncId for a generic
                    // call inside an arm body (`int x = id(5);`) must be remapped to
                    // its instantiated id, or codegen sees an unresolved placeholder
                    // and emits a broken (MAKA_UNIT) expansion.
                    rw_block(&mut a.body, mapping);
                    if let Some(v) = &mut a.value { rw_expr(v, mapping); }
                }
            }
            // Free's operand is walked too: a generic call inside `free X;`
            // (HExprKind::Free) otherwise keeps its placeholder callee FuncId,
            // which codegen indexes out of bounds -> compiler panic.
            HExprKind::HeapAlloc(inner) | HExprKind::Free(inner, _) => rw_expr(inner, mapping),
            HExprKind::CallIndirect { callee, args } => {
                rw_expr(callee, mapping);
                for a in args { rw_expr(a, mapping); }
            }
            HExprKind::InlineCall { callee, args, .. } => {
                let v = callee.0;
                if v <= PLACEHOLDER_FID_BASE && v + (mapping.len() as u32) > PLACEHOLDER_FID_BASE {
                    let idx = (PLACEHOLDER_FID_BASE - v) as usize;
                    if idx < mapping.len() { *callee = FuncId(mapping[idx]); }
                }
                for a in args { rw_expr(a, mapping); }
            }
            HExprKind::Closure { env_values, .. } => {
                for v in env_values { rw_expr(v, mapping); }
            }
            HExprKind::Transfer(inner) => rw_expr(inner, mapping),
            HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => rw_expr(inner, mapping),
            HExprKind::ArrayLit(es) => for e in es { rw_expr(e, mapping); },
            _ => {}
        }
    }
    rw_block(&mut f.body, mapping);
}

/// Decide whether a `has` impl is visible to a caller in module `from_module` with
/// `caller_has_imports` declared.  Same-module impls are always visible; cross-module
/// requires both `is_pub` AND an explicit `use Mod.Type.Attr;` in the caller's file.
/// Type keys that name a primitive.  A `has` impl on one of these is inherent
/// and universally visible - you can't meaningfully `use` (or fail to `use`) the
/// `Add` impl for `int` the way you would for a user-defined type, and overlap
/// rules keep it unique, so it is always in scope.
fn is_primitive_type_key(k: &str) -> bool {
    matches!(k,
        "int" | "bool" | "char" | "string" | "float" | "unit"
        | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"
        | "isize" | "usize" | "f32" | "f64")
}

fn has_impl_visible(
    h: &hir::HasImpl,
    from_module: &[String],
    caller_has_imports: &[maka_ast::HasImport],
) -> bool {
    if h.is_pub && is_primitive_type_key(&h.type_key) { return true; }
    if h.module_path == from_module { return true; }
    if !h.is_pub { return false; }
    caller_has_imports.iter().any(|imp|
        imp.module_path == h.module_path
            && imp.type_name == h.type_key
            && imp.attr_name == h.attr_name
    )
}
