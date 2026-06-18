//! First pass: collect data/enum decls and function signatures.

use crate::hir::*;
use crate::SemaError;
use maka_ast::{self as ast, Mutness};

pub fn resolve_type(sym: &SymTab, t: &ast::Type, errors: &mut Vec<SemaError>) -> HType {
    resolve_type_in(sym, t, &[], errors)
}

/// Peel references/pointers/heap to find the underlying struct/enum and return its
/// canonical name as a string key, or `None` if the type isn't a nominal type.
/// Used to record trait impls: the first parameter's underlying nominal type is the
/// receiver of the trait.
/// Collect the names of type variables introduced by a `has` impl's
/// receiver pattern.  Any `Type::Named(n, _)` whose `n` is NOT a known
/// struct/enum and NOT a primitive name is treated as a type variable.
/// This is the impl-local convention — the variable's scope is the
/// receiver and the impl body.
pub fn collect_receiver_tyvars(sym: &SymTab, t: &maka_ast::Type) -> Vec<String> {
    fn is_struct_or_enum(sym: &SymTab, n: &str) -> bool {
        sym.struct_by_name(n).is_some() || sym.enum_by_name(n).is_some()
    }
    fn is_primitive(n: &str) -> bool {
        matches!(n,
            "int" | "bool" | "char" | "string" | "float" | "unit" | "String"
            | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"
            | "isize" | "usize" | "f32" | "f64")
    }
    let mut out: Vec<String> = Vec::new();
    fn walk(sym: &SymTab, t: &maka_ast::Type, out: &mut Vec<String>) {
        match t {
            maka_ast::Type::Named(n, _) => {
                if !is_struct_or_enum(sym, n) && !is_primitive(n) && n != "_" {
                    if !out.contains(n) { out.push(n.clone()); }
                }
            }
            maka_ast::Type::Ref { inner, .. } | maka_ast::Type::Ptr { inner, .. }
            | maka_ast::Type::RawPtr { inner, .. } | maka_ast::Type::OwnPtr { inner, .. }
            | maka_ast::Type::Heap { inner, .. } => walk(sym, inner, out),
            maka_ast::Type::Array { elem, .. } | maka_ast::Type::Slice { elem, .. }
            | maka_ast::Type::Vec { elem, .. } => walk(sym, elem, out),
            maka_ast::Type::Generic { args, .. } => {
                for a in args { walk(sym, a, out); }
            }
            maka_ast::Type::FnPtr { ret, params, .. } => {
                walk(sym, ret, out);
                for p in params { walk(sym, p, out); }
            }
            maka_ast::Type::AssocPath { base, .. } => walk(sym, base, out),
            _ => {}
        }
        let _ = sym;
    }
    walk(sym, t, &mut out);
    out
}

/// Resolve `on::segment` (an HType::AssocType placeholder) to its concrete
/// type by looking up the impl whose receiver pattern unifies with `on` and
/// reading the impl's `type segment = ...` definition with the unification
/// env applied.  Returns `None` if no impl matches or the impl doesn't
/// define the segment.  Reports `attr_hint` (if Some) to disambiguate when
/// multiple bounds could match.
pub fn resolve_assoc_type(
    sym: &SymTab,
    on: &HType,
    segment: &str,
    attr_hint: Option<&str>,
) -> Option<HType> {
    let matching: Vec<&HasImpl> = sym.has_impls.iter()
        .filter(|h| {
            if let Some(a) = attr_hint { if h.attr_name != a { return false; } }
            // Must declare a segment of this name in the matching attr.
            let attr = sym.attr_by_name(&h.attr_name);
            let declares = attr.map(|a| a.assoc_type_decls.iter().any(|(n, _, _)| n == segment)).unwrap_or(false);
            if !declares { return false; }
            // Receiver pattern must unify with `on`.
            receiver_unify(&h.receiver_pattern, on, &h.receiver_tyvars).is_some()
        })
        .collect();
    if matching.is_empty() { return None; }
    // Pick the first matching impl (coherence guarantees uniqueness once
    // the overlap checker is in; until then we tolerate first-match).
    let h = matching[0];
    let env = receiver_unify(&h.receiver_pattern, on, &h.receiver_tyvars)?;
    let (_, raw) = h.assoc_type_defs.iter().find(|(n, _)| n == segment)?;
    Some(raw.subst(&env))
}

/// Walk an HType, replacing every `HType::AssocType { on, segment, .. }`
/// whose `on` is concrete (no TyVar inside it) with the resolved type.
/// Recurses into the resolved type so chained assoc-types collapse.
pub fn resolve_assoc_types_in(sym: &SymTab, t: &HType) -> HType {
    fn has_tyvar(t: &HType) -> bool {
        match t {
            HType::TyVar(_) => true,
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. }
            | HType::OwnPtr { inner, .. } | HType::Heap { inner } => has_tyvar(inner),
            HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => has_tyvar(elem),
            HType::FnPtr { ret, params } => has_tyvar(ret) || params.iter().any(has_tyvar),
            HType::AssocType { on, .. } => has_tyvar(on),
            _ => false,
        }
    }
    match t {
        HType::AssocType { on, segment, attr_hint } => {
            let on_resolved = resolve_assoc_types_in(sym, on);
            if has_tyvar(&on_resolved) {
                return HType::AssocType {
                    on: Box::new(on_resolved),
                    segment: segment.clone(),
                    attr_hint: attr_hint.clone(),
                };
            }
            match resolve_assoc_type(sym, &on_resolved, segment, attr_hint.as_deref()) {
                Some(r) => resolve_assoc_types_in(sym, &r),
                None => t.clone(),
            }
        }
        HType::Ref { mutable, inner } => HType::Ref { mutable: *mutable, inner: Box::new(resolve_assoc_types_in(sym, inner)) },
        HType::Ptr { mutable, inner } => HType::Ptr { mutable: *mutable, inner: Box::new(resolve_assoc_types_in(sym, inner)) },
        HType::RawPtr { mutable, inner } => HType::RawPtr { mutable: *mutable, inner: Box::new(resolve_assoc_types_in(sym, inner)) },
        HType::OwnPtr { mutable, inner } => HType::OwnPtr { mutable: *mutable, inner: Box::new(resolve_assoc_types_in(sym, inner)) },
        HType::Heap { inner } => HType::Heap { inner: Box::new(resolve_assoc_types_in(sym, inner)) },
        HType::Array { len, elem } => HType::Array { len: *len, elem: Box::new(resolve_assoc_types_in(sym, elem)) },
        HType::Slice { mutable, elem } => HType::Slice { mutable: *mutable, elem: Box::new(resolve_assoc_types_in(sym, elem)) },
        HType::Vec { elem } => HType::Vec { elem: Box::new(resolve_assoc_types_in(sym, elem)) },
        HType::FnPtr { ret, params } => HType::FnPtr {
            ret: Box::new(resolve_assoc_types_in(sym, ret)),
            params: params.iter().map(|p| resolve_assoc_types_in(sym, p)).collect(),
        },
        _ => t.clone(),
    }
}

pub fn underlying_struct_key(sym: &SymTab, ty: &HType) -> Option<String> {
    match ty {
        HType::Struct(id) => Some(sym.struct_info(*id).name.clone()),
        HType::Enum(id) => Some(sym.enum_info(*id).name.clone()),
        HType::Ref { inner, .. }
        | HType::Ptr { inner, .. }
        | HType::RawPtr { inner, .. }
        | HType::OwnPtr { inner, .. }
        | HType::Heap { inner } => underlying_struct_key(sym, inner),
        // Primitives can be `has`-implementors (§10.4) — return their
        // canonical name as the key so bound-check lookup matches the
        // string the parser stored in HasDecl.type_name.
        HType::Int => Some("int".into()),
        HType::Bool => Some("bool".into()),
        HType::Char => Some("char".into()),
        HType::Str => Some("string".into()),
        HType::Float => Some("float".into()),
        HType::Unit => Some("unit".into()),
        HType::SizedInt { signed, bits } => {
            let pref = if *signed { "i" } else { "u" };
            match bits { 8 | 16 | 32 | 64 => Some(format!("{}{}", pref, bits)), _ => None }
        }
        _ => None,
    }
}

/// Walk a resolved type, reporting an error for every Struct or Enum reference
/// whose declaring module differs from `from_module` and which is either not
/// `pub` or not imported by the current file.  Same rule that already applies
/// to cross-module function calls.
///
/// `from_imports` is the current file's `(module_path, name)` import list.
/// For an imported name to satisfy a type reference, both the path and the
/// name (the type's name) must match.  The instantiation-mangled name of a
/// generic enum (e.g. `Option__int`) is unwrapped to its template name
/// (`Option`) for the import check.
pub fn check_type_visibility(
    sym: &SymTab,
    ty: &HType,
    from_module: &[String],
    from_imports: &[(Vec<String>, String)],
    span: maka_lexer::Span,
    errors: &mut Vec<SemaError>,
) {
    fn template_name(full: &str) -> &str {
        full.split("__").next().unwrap_or(full)
    }
    fn is_imported(imports: &[(Vec<String>, String)], path: &[String], name: &str) -> bool {
        let tmpl = template_name(name);
        imports.iter().any(|(p, n)| p.as_slice() == path && (n == name || n == tmpl || n == "*"))
    }
    fn walk(sym: &SymTab, ty: &HType, from: &[String], from_imports: &[(Vec<String>, String)], sp: maka_lexer::Span, errs: &mut Vec<SemaError>) {
        match ty {
            HType::Struct(id) => {
                let info = sym.struct_info(*id);
                if info.module_path.as_slice() != from {
                    if !info.is_pub {
                        errs.push(SemaError {
                            msg: format!(
                                "data type `{}` is private to module `{}`; mark it `pub` to use from `{}`",
                                info.name,
                                if info.module_path.is_empty() { "<root>".to_string() } else { info.module_path.join(".") },
                                if from.is_empty() { "<root>".to_string() } else { from.join(".") },
                            ),
                            span: sp,
                        });
                    } else if !is_imported(from_imports, &info.module_path, &info.name) {
                        let tmpl = template_name(&info.name).to_string();
                        errs.push(SemaError {
                            msg: format!(
                                "data type `{}` is in module `{}` and must be imported (`import {}.{};`) to use from `{}`",
                                tmpl,
                                if info.module_path.is_empty() { "<root>".to_string() } else { info.module_path.join(".") },
                                if info.module_path.is_empty() { "<root>".to_string() } else { info.module_path.join(".") },
                                tmpl,
                                if from.is_empty() { "<root>".to_string() } else { from.join(".") },
                            ),
                            span: sp,
                        });
                    }
                }
            }
            HType::Enum(id) => {
                let info = sym.enum_info(*id);
                if info.module_path.as_slice() != from {
                    if !info.is_pub {
                        errs.push(SemaError {
                            msg: format!(
                                "enum `{}` is private to module `{}`; mark it `pub` to use from `{}`",
                                info.name,
                                if info.module_path.is_empty() { "<root>".to_string() } else { info.module_path.join(".") },
                                if from.is_empty() { "<root>".to_string() } else { from.join(".") },
                            ),
                            span: sp,
                        });
                    } else if !is_imported(from_imports, &info.module_path, &info.name) {
                        let tmpl = template_name(&info.name).to_string();
                        errs.push(SemaError {
                            msg: format!(
                                "enum `{}` is in module `{}` and must be imported (`import {}.{};`) to use from `{}`",
                                tmpl,
                                if info.module_path.is_empty() { "<root>".to_string() } else { info.module_path.join(".") },
                                if info.module_path.is_empty() { "<root>".to_string() } else { info.module_path.join(".") },
                                tmpl,
                                if from.is_empty() { "<root>".to_string() } else { from.join(".") },
                            ),
                            span: sp,
                        });
                    }
                }
            }
            HType::Ref { inner, .. }
            | HType::Ptr { inner, .. }
            | HType::RawPtr { inner, .. }
            | HType::OwnPtr { inner, .. }
            | HType::Heap { inner } => walk(sym, inner, from, from_imports, sp, errs),
            HType::Array { elem, .. }
            | HType::Slice { elem, .. }
            | HType::Vec { elem } => walk(sym, elem, from, from_imports, sp, errs),
            HType::FnPtr { ret, params } => {
                walk(sym, ret, from, from_imports, sp, errs);
                for p in params { walk(sym, p, from, from_imports, sp, errs); }
            }
            _ => {}
        }
    }
    walk(sym, ty, from_module, from_imports, span, errors);
}

pub fn resolve_signature(
    sym: &SymTab,
    params: &[ast::Param],
    ret: &ast::Type,
    type_params: &[String],
    errors: &mut Vec<SemaError>,
) -> (Vec<HType>, HType) {
    let param_tys = params.iter().map(|p| resolve_type_in(sym, &p.ty, type_params, errors)).collect();
    let ret_ty = resolve_type_in(sym, ret, type_params, errors);
    (param_tys, ret_ty)
}

/// Like resolve_type but treats identifiers in `type_params` as TyVars.
pub fn resolve_type_in(
    sym: &SymTab,
    t: &ast::Type,
    type_params: &[String],
    errors: &mut Vec<SemaError>,
) -> HType {
    match t {
        ast::Type::Named(n, sp) => match n.as_str() {
            "int" => HType::Int,
            "i8"  => HType::SizedInt { signed: true,  bits: 8  },
            "i16" => HType::SizedInt { signed: true,  bits: 16 },
            "i32" => HType::SizedInt { signed: true,  bits: 32 },
            "i64" => HType::SizedInt { signed: true,  bits: 64 },
            "isize" => HType::SizedInt { signed: true, bits: 0 },
            "u16" => HType::SizedInt { signed: false, bits: 16 },
            "u32" => HType::SizedInt { signed: false, bits: 32 },
            "u64" => HType::SizedInt { signed: false, bits: 64 },
            "usize" => HType::SizedInt { signed: false, bits: 0 },
            // Per D5: char and u8 are the same byte type.
            "u8" | "char" => HType::Char,
            "float" | "f64" => HType::Float,
            "f32" => HType::SizedFloat { bits: 32 },
            "bool" => HType::Bool,
            "unit" => HType::Unit,
            "string" => HType::Str,
            // `String` is the owned heap-text name.  Internally it's `own *char`
            // (NUL-terminated, mutable, auto-freed at scope exit).  String literals
            // and borrowed views keep the primitive `string` type; constructors
            // (`a + b`, `read_line()`) and any function that allocates returns
            // `String`.  Coerces to `string` for borrowed reads.
            "String" => HType::OwnPtr { mutable: true, inner: Box::new(HType::Char) },
            other => {
                if other == "_" {
                    errors.push(SemaError {
                        msg: "`_` placeholder type is only valid inside `attr` / `has` blocks — it refers to the implementing type".into(),
                        span: *sp,
                    });
                    return HType::Int;
                }
                if type_params.iter().any(|tp| tp == other) {
                    return HType::TyVar(other.to_string());
                }
                if let Some((id, _)) = sym.struct_by_name(other) {
                    HType::Struct(id)
                } else if let Some((id, _)) = sym.enum_by_name(other) {
                    HType::Enum(id)
                } else {
                    errors.push(SemaError { msg: format!("unknown type `{}`", other), span: *sp });
                    HType::Int
                }
            }
        },
        ast::Type::Unit(_) => HType::Unit,
        ast::Type::Ref { mutness, inner, .. } => {
            let mutable = matches!(mutness, Mutness::Mut);
            let inner = resolve_type_in(sym, inner, type_params, errors);
            HType::Ref { mutable, inner: Box::new(inner) }
        }
        ast::Type::Ptr { mutness, inner, .. } => {
            let mutable = match mutness {
                Mutness::Const => false,
                Mutness::Mut | Mutness::Default => true,
            };
            let inner = resolve_type_in(sym, inner, type_params, errors);
            HType::Ptr { mutable, inner: Box::new(inner) }
        }
        ast::Type::RawPtr { mutness, inner, .. } => {
            let mutable = match mutness {
                Mutness::Const => false,
                Mutness::Mut | Mutness::Default => true,
            };
            let inner = resolve_type_in(sym, inner, type_params, errors);
            HType::RawPtr { mutable, inner: Box::new(inner) }
        }
        ast::Type::OwnPtr { mutness, inner, .. } => {
            let mutable = match mutness {
                Mutness::Const => false,
                Mutness::Mut | Mutness::Default => true,
            };
            let inner = resolve_type_in(sym, inner, type_params, errors);
            HType::OwnPtr { mutable, inner: Box::new(inner) }
        }
        ast::Type::Heap { inner, .. } => {
            let inner = resolve_type_in(sym, inner, type_params, errors);
            HType::Heap { inner: Box::new(inner) }
        }
        ast::Type::Array { len, elem, .. } => {
            let elem = resolve_type_in(sym, elem, type_params, errors);
            HType::Array { len: *len, elem: Box::new(elem) }
        }
        ast::Type::Slice { mutness, elem, .. } => {
            let mutable = matches!(mutness, Mutness::Mut);
            let elem = resolve_type_in(sym, elem, type_params, errors);
            HType::Slice { mutable, elem: Box::new(elem) }
        }
        ast::Type::Vec { elem, .. } => {
            let elem = resolve_type_in(sym, elem, type_params, errors);
            HType::Vec { elem: Box::new(elem) }
        }
        ast::Type::Dyn { traits, .. } => {
            HType::Dyn { traits: traits.clone() }
        }
        ast::Type::FnPtr { ret, params, .. } => {
            let r = resolve_type_in(sym, ret, type_params, errors);
            let ps: Vec<HType> = params.iter().map(|p| resolve_type_in(sym, p, type_params, errors)).collect();
            HType::FnPtr { ret: Box::new(r), params: ps }
        }
        ast::Type::AssocPath { base, segment, .. } => {
            // `T::Slot` — the base resolves to (typically) a TyVar; we keep
            // it as an AssocType placeholder that monomorphization resolves
            // by looking up the impl whose receiver pattern unifies with the
            // base's concrete substitution.
            let on = resolve_type_in(sym, base, type_params, errors);
            HType::AssocType { on: Box::new(on), segment: segment.clone(), attr_hint: None }
        }
        ast::Type::Generic { name, args, span } => {
            // Built-in `Rust<T>` from the Maka↔Rust bridge: an opaque heap
            // handle to a Rust value.  Same ABI as `own *mut unit`; the `T`
            // label is carried so Maka can route per-call-site Send / Sync
            // probes back to the sidecar at thread-crossing sites.
            if name == "Rust" && args.len() == 1 {
                let label = match &args[0] {
                    ast::Type::Named(n, _) => n.clone(),
                    other => format!("{:?}", other),
                };
                return HType::RustOpaque(label);
            }
            let resolved_args: Vec<HType> = args.iter().map(|a| resolve_type_in(sym, a, type_params, errors)).collect();
            let key = resolved_args.iter().map(|t| t.key()).collect::<Vec<_>>().join(",");
            // Concrete instantiation already monomorphized → use it.
            if let Some(sid) = sym.struct_instantiations.get(&(name.clone(), key.clone())) {
                return HType::Struct(StructId(*sid));
            }
            if let Some(eid) = sym.enum_instantiations.get(&(name.clone(), key)) {
                return HType::Enum(EnumId(*eid));
            }
            // Template (used inside generic bodies — yields a TyVar-bearing pattern).
            if let Some((id, _)) = sym.struct_by_name(name) {
                HType::Struct(id)
            } else if let Some((id, _)) = sym.enum_by_name(name) {
                HType::Enum(id)
            } else {
                errors.push(SemaError { msg: format!("unknown generic type `{}`", name), span: *span });
                HType::Int
            }
        }
    }
}

impl SymTab {
    pub fn collect(m: &ast::Module) -> Result<SymTab, Vec<SemaError>> {
        let mut errors = Vec::new();
        let mut sym = SymTab::default();

        // Built-in opaque types for concurrency.  Codegen lowers `Thread` to
        // `pthread_t` (via the `__maka_thread_t` typedef in the prologue) and
        // accepts `*Thread` as the spawn/join handle type.
        sym.structs.push(StructInfo {
            name: "Thread".to_string(),
            type_params: Vec::new(),
            template: None,
            fields: Vec::new(),
            is_pub: true,
            module_path: Vec::new(),
            span: maka_lexer::Span::dummy(),
            where_bounds: Vec::new(),
        });

        // Pass 1: enums and struct names
        for (idx, item) in m.items.iter().enumerate() {
            let item_module: Vec<String> = m.item_modules.get(idx).cloned().unwrap_or_default();
            match item {
                ast::Item::Data(d) => {
                    if sym.structs.iter().any(|s| s.name == d.name) {
                        errors.push(SemaError { msg: format!("duplicate data type `{}`", d.name), span: d.span });
                    }
                    // Resolve the where clauses against the data decl's own type params.
                    let where_bounds: Vec<(String, Vec<HType>, Vec<(String, HType)>)> = d.where_clauses.iter().map(|w| {
                        let args: Vec<HType> = w.args.iter()
                            .map(|a| resolve_type_in(&sym, a, &d.type_params, &mut errors))
                            .collect();
                        let bindings: Vec<(String, HType)> = w.assoc_type_bindings.iter()
                            .map(|(n, t)| (n.clone(), resolve_type_in(&sym, t, &d.type_params, &mut errors)))
                            .collect();
                        (w.trait_name.clone(), args, bindings)
                    }).collect();
                    sym.structs.push(StructInfo {
                        name: d.name.clone(),
                        type_params: d.type_params.clone(),
                        template: None,
                        fields: Vec::new(),
                        is_pub: d.is_pub,
                        module_path: item_module.clone(),
                        span: d.span,
                        where_bounds,
                    });
                }
                ast::Item::Enum(e) => {
                    if sym.enums.iter().any(|x| x.name == e.name) {
                        errors.push(SemaError { msg: format!("duplicate enum `{}`", e.name), span: e.span });
                    }
                    // Pass 1: record variant names/tags; fields resolved in Pass 2 once all structs are known.
                    let mut variants = Vec::new();
                    let mut next = 0i64;
                    for v in &e.variants {
                        let tag = v.explicit_value.unwrap_or(next);
                        variants.push(VariantInfo {
                            name: v.name.clone(),
                            tag,
                            fields: Vec::new(),
                            span: v.span,
                        });
                        next = tag + 1;
                    }
                    sym.enums.push(EnumInfo {
                        name: e.name.clone(),
                        type_params: e.type_params.clone(),
                        variants,
                        is_pub: e.is_pub,
                        module_path: item_module.clone(),
                        span: e.span,
                    });
                }
                _ => {}
            }
        }

        // Pass 2a: resolve enum variant fields.
        for item in &m.items {
            if let ast::Item::Enum(e) = item {
                let eid = sym.enum_by_name(&e.name).unwrap().0;
                let tp = e.type_params.clone();
                for (vi, v) in e.variants.iter().enumerate() {
                    let mut fields = Vec::new();
                    for f in &v.fields {
                        let ty = resolve_type_in(&sym, &f.ty, &tp, &mut errors);
                        let mut_payload = matches!(f.mutness, Mutness::Mut);
                        fields.push(FieldInfo {
                            name: f.name.clone(),
                            ty,
                            mut_payload,
                            default: None,
                            is_embed: false,
                            span: f.span,
                        });
                    }
                    sym.enums[eid.0 as usize].variants[vi].fields = fields;
                }
            }
        }

        // Pass 2: resolve struct fields and validate heap-not-in-fields
        for item in &m.items {
            if let ast::Item::Data(d) = item {
                let mut fields = Vec::new();
                for f in &d.fields {
                    let ty = resolve_type_in(&sym, &f.ty, &d.type_params, &mut errors);
                    // Heap fields are forbidden except heap [*]T
                    if let HType::Heap { inner } = &ty {
                        if !matches!(inner.as_ref(), HType::Vec { .. }) {
                            errors.push(SemaError {
                                msg: "`heap` modifier not allowed on struct fields (except `heap [*]T`)".into(),
                                span: f.span,
                            });
                        }
                    }
                    // References inside struct fields need to be initialized — checked at construction.
                    // Slices must be initialized at construction.
                    // Embed fields pass mutability through to their inner struct.
                    let mut_payload = if f.is_embed { true } else { matches!(f.mutness, Mutness::Mut) };
                    // default expression — we defer typechecking of these to use sites for v1 simplicity.
                    let _default = f.default.clone();
                    fields.push(FieldInfo {
                        name: f.name.clone(),
                        ty,
                        mut_payload,
                        default: None,
                        is_embed: f.is_embed,
                        span: f.span,
                    });
                }
                // Resolve struct id and write back
                let id = sym.struct_by_name(&d.name).unwrap().0;
                sym.structs[id.0 as usize].fields = fields;
            }
        }

        // Pass 2b: scan AST for generic struct instantiations and create them.
        let mut struct_inst_requests: Vec<(String, Vec<HType>)> = Vec::new();
        for item in &m.items {
            match item {
                ast::Item::Data(d) => {
                    for f in &d.fields { scan_struct_insts(&sym, &f.ty, &d.type_params, &mut struct_inst_requests, &mut errors); }
                }
                ast::Item::Func(f) => {
                    for p in &f.params { scan_struct_insts(&sym, &p.ty, &f.type_params, &mut struct_inst_requests, &mut errors); }
                    scan_struct_insts(&sym, &f.ret, &f.type_params, &mut struct_inst_requests, &mut errors);
                    scan_block(&sym, &f.body, &f.type_params, &mut struct_inst_requests, &mut errors);
                }
                ast::Item::Logic(l) => {
                    for f in &l.funcs {
                        for p in &f.params { scan_struct_insts(&sym, &p.ty, &f.type_params, &mut struct_inst_requests, &mut errors); }
                        scan_struct_insts(&sym, &f.ret, &f.type_params, &mut struct_inst_requests, &mut errors);
                        scan_block(&sym, &f.body, &f.type_params, &mut struct_inst_requests, &mut errors);
                    }
                }
                ast::Item::Has(h) => {
                    for f in &h.funcs {
                        for p in &f.params { scan_struct_insts(&sym, &p.ty, &f.type_params, &mut struct_inst_requests, &mut errors); }
                        scan_struct_insts(&sym, &f.ret, &f.type_params, &mut struct_inst_requests, &mut errors);
                        scan_block(&sym, &f.body, &f.type_params, &mut struct_inst_requests, &mut errors);
                    }
                }
                ast::Item::Extern(e) => {
                    for p in &e.params { scan_struct_insts(&sym, &p.ty, &[], &mut struct_inst_requests, &mut errors); }
                    scan_struct_insts(&sym, &e.ret, &[], &mut struct_inst_requests, &mut errors);
                }
                _ => {}
            }
        }
        // Instantiate each unique request.
        struct_inst_requests.sort_by(|a, b| (a.0.clone(), a.1.iter().map(|t| t.key()).collect::<String>())
            .cmp(&(b.0.clone(), b.1.iter().map(|t| t.key()).collect::<String>())));
        struct_inst_requests.dedup_by(|a, b| a.0 == b.0 && a.1.iter().map(|t| t.key()).collect::<String>() == b.1.iter().map(|t| t.key()).collect::<String>());
        for (name, args) in &struct_inst_requests {
            let key = args.iter().map(|t| t.key()).collect::<Vec<_>>().join(",");
            if sym.struct_instantiations.contains_key(&(name.clone(), key.clone())) { continue; }
            let Some((tid, tinfo)) = sym.struct_by_name(name) else { continue; };
            let template = tinfo.clone();
            if template.type_params.len() != args.len() { continue; }
            let mangled = format!("{}__{}", name, args.iter().map(|t| t.key()).collect::<Vec<_>>().join("_"));
            let env: std::collections::HashMap<String, HType> = template.type_params.iter().cloned().zip(args.iter().cloned()).collect();
            let new_fields: Vec<FieldInfo> = template.fields.iter().map(|f| {
                // First substitute the template's type vars with the concrete
                // args, then resolve any AssocType placeholders against the
                // registered impls (§10.5 monomorphization-time resolution).
                let subst_ty = f.ty.subst(&env);
                let resolved_ty = resolve_assoc_types_in(&sym, &subst_ty);
                FieldInfo {
                    name: f.name.clone(),
                    ty: resolved_ty,
                    mut_payload: f.mut_payload,
                    default: f.default.clone(),
                    is_embed: f.is_embed,
                    span: f.span,
                }
            }).collect();
            let _ = tid;
            let new_id = StructId(sym.structs.len() as u32);
            let tmpl_is_pub = template.is_pub;
            let tmpl_mod = template.module_path.clone();
            sym.structs.push(StructInfo {
                name: mangled,
                type_params: Vec::new(),
                template: Some(name.clone()),
                fields: new_fields,
                is_pub: tmpl_is_pub,
                module_path: tmpl_mod,
                span: template.span,
                where_bounds: Vec::new(),
            });
            sym.struct_instantiations.insert((name.clone(), key), new_id.0);
        }

        // Same shape, for enums.  `Option<int>`, `Option<string>` etc. each get
        // their own EnumInfo whose variant fields have the type-param substituted.
        for (name, args) in &struct_inst_requests {
            let key = args.iter().map(|t| t.key()).collect::<Vec<_>>().join(",");
            if sym.enum_instantiations.contains_key(&(name.clone(), key.clone())) { continue; }
            let Some((_eid, einfo)) = sym.enum_by_name(name) else { continue; };
            let template = einfo.clone();
            if template.type_params.len() != args.len() { continue; }
            let mangled = format!("{}__{}", name, args.iter().map(|t| t.key()).collect::<Vec<_>>().join("_"));
            let env: std::collections::HashMap<String, HType> = template.type_params.iter().cloned().zip(args.iter().cloned()).collect();
            let new_variants: Vec<VariantInfo> = template.variants.iter().map(|v| VariantInfo {
                name: v.name.clone(),
                tag: v.tag,
                fields: v.fields.iter().map(|f| FieldInfo {
                    name: f.name.clone(),
                    ty: f.ty.subst(&env),
                    mut_payload: f.mut_payload,
                    default: f.default.clone(),
                    is_embed: f.is_embed,
                    span: f.span,
                }).collect(),
                span: v.span,
            }).collect();
            let new_id = EnumId(sym.enums.len() as u32);
            sym.enums.push(EnumInfo {
                name: mangled,
                type_params: Vec::new(),
                variants: new_variants,
                is_pub: template.is_pub,
                module_path: template.module_path.clone(),
                span: template.span,
            });
            sym.enum_instantiations.insert((name.clone(), key), new_id.0);
        }

        // Pass 3: function signatures (top-level `fn`, `extern`, and `logic`-block funcs)
        for (idx, item) in m.items.iter().enumerate() {
            let item_module: Vec<String> = m.item_modules.get(idx).cloned().unwrap_or_default();
            let item_imports: Vec<(Vec<String>, String)> = m.item_imports.get(idx).cloned().unwrap_or_default();
            let item_has_imports: Vec<ast::HasImport> = m.item_has_imports.get(idx).cloned().unwrap_or_default();
            match item {
                ast::Item::Func(f) => {
                    if sym.func_by_name(&f.name).is_some() {
                        errors.push(SemaError { msg: format!("duplicate function `{}`", f.name), span: f.span });
                        continue;
                    }
                    let (param_tys, ret) = resolve_signature(&sym, &f.params, &f.ret, &f.type_params, &mut errors);
                    // Enforce `pub` AND import on any data/enum the signature references.
                    for (i, pty) in param_tys.iter().enumerate() {
                        let psp = f.params.get(i).map(|p| p.span).unwrap_or(f.span);
                        check_type_visibility(&sym, pty, &item_module, &item_imports, psp, &mut errors);
                    }
                    check_type_visibility(&sym, &ret, &item_module, &item_imports, f.span, &mut errors);
                    // Resolve where-clause bounds into `(trait_name, type_args)`.
                    let where_bounds: Vec<(String, Vec<HType>, Vec<(String, HType)>)> = f.where_clauses.iter().map(|w| {
                        let args: Vec<HType> = w.args.iter()
                            .map(|a| resolve_type_in(&sym, a, &f.type_params, &mut errors))
                            .collect();
                        let bindings: Vec<(String, HType)> = w.assoc_type_bindings.iter()
                            .map(|(n, t)| (n.clone(), resolve_type_in(&sym, t, &f.type_params, &mut errors)))
                            .collect();
                        (w.trait_name.clone(), args, bindings)
                    }).collect();
                    let param_names: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                    let fid = FuncId(sym.sigs.len() as u32);
                    sym.sigs.push(FuncSig {
                        name: f.name.clone(),
                        param_tys, param_names, ret,
                        is_extern: false,
                        c_name: f.name.clone(),
                        logic: None,
                        type_params: f.type_params.clone(),
                        is_inline: f.is_inline,
                        is_gate: f.is_gate,
                        is_variadic: false,
                        is_pub: f.is_pub,
                        module_path: item_module.clone(),
                        imports: item_imports.clone(),
                        has_imports: item_has_imports.clone(),
                        where_bounds,
                    });
                    sym.ast_funcs.insert(fid.0, f.clone());
                }
                ast::Item::Extern(e) => {
                    if sym.func_by_name(&e.name).is_some() {
                        errors.push(SemaError { msg: format!("duplicate extern `{}`", e.name), span: e.span });
                        continue;
                    }
                    let ret = resolve_type(&sym, &e.ret, &mut errors);
                    let mut param_tys = Vec::new();
                    let mut param_names = Vec::new();
                    for p in &e.params {
                        param_tys.push(resolve_type(&sym, &p.ty, &mut errors));
                        param_names.push(p.name.clone());
                    }
                    sym.sigs.push(FuncSig {
                        name: e.name.clone(),
                        param_tys, param_names, ret,
                        is_extern: true,
                        c_name: e.c_name.clone(),
                        logic: None,
                        type_params: Vec::new(),
                        is_inline: false,
                        is_gate: e.is_gate,
                        is_variadic: e.is_variadic,
                        is_pub: e.is_pub,
                        module_path: item_module.clone(),
                        imports: item_imports.clone(),
                        has_imports: Vec::new(),
                        where_bounds: Vec::new(),
                    });
                }
                ast::Item::Logic(l) => {
                    if sym.logic_by_name(&l.name).is_some() {
                        errors.push(SemaError { msg: format!("duplicate logic `{}`", l.name), span: l.span });
                        continue;
                    }
                    let mut func_ids = Vec::new();
                    // Count overloads for unique mangling.
                    let mut name_seen: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
                    // Track which receiver types we've already recorded a HasImpl for
                    // (a logic block may declare multiple methods on the same receiver).
                    let mut recorded_receivers: std::collections::HashSet<String> = std::collections::HashSet::new();
                    for f in &l.funcs {
                        let (param_tys, ret) = resolve_signature(&sym, &f.params, &f.ret, &f.type_params, &mut errors);
                        // Record that the first parameter's underlying struct implements
                        // the trait named by this logic block — bridge into both the legacy
                        // `trait_impls` index and the visibility-aware `has_impls` list.
                        if let Some(first_ty) = param_tys.first() {
                            let key = underlying_struct_key(&sym, first_ty);
                            if let Some(k) = key {
                                sym.trait_impls.entry(l.name.clone()).or_default().insert(k.clone());
                                if recorded_receivers.insert(k.clone()) {
                                    sym.has_impls.push(HasImpl {
                                        attr_name: l.name.clone(),
                                        type_key: k,
                                        attr_args: Vec::new(),
                                        is_pub: l.is_pub,
                                        module_path: item_module.clone(),
                                        func_ids: Vec::new(),
                                        assoc_type_defs: Vec::new(),
                                        receiver_tyvars: Vec::new(),
                                        receiver_pattern: first_ty.clone(),
                                    });
                                }
                            }
                        }
                        let param_names: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                        let overload_idx = *name_seen.entry(f.name.clone()).and_modify(|v| *v += 1).or_insert(0);
                        let c_name = if overload_idx == 0 {
                            format!("{}__{}", l.name, f.name)
                        } else {
                            format!("{}__{}_{}", l.name, f.name, overload_idx)
                        };
                        let fid = FuncId(sym.sigs.len() as u32);
                        sym.sigs.push(FuncSig {
                            name: f.name.clone(),
                            param_tys, param_names, ret,
                            is_extern: false,
                            c_name,
                            logic: Some(l.name.clone()),
                            type_params: f.type_params.clone(),
                            is_inline: f.is_inline,
                            is_gate: f.is_gate,
                            is_variadic: false,
                            // `pub logic` exports every method inside it.  Per-method
                            // `pub` on a logic-block method isn't a thing in the
                            // grammar — visibility flows from the block.
                            is_pub: l.is_pub || f.is_pub,
                            module_path: item_module.clone(),
                            imports: item_imports.clone(),
                            has_imports: item_has_imports.clone(),
                            where_bounds: Vec::new(),
                        });
                        sym.ast_funcs.insert(fid.0, f.clone());
                        func_ids.push(fid);
                    }
                    sym.logics.push(LogicInfo {
                        name: l.name.clone(),
                        funcs: func_ids,
                        span: l.span,
                    });
                }
                ast::Item::Global(_) => {
                    // Handled in a dedicated pass below (needs the type-check
                    // helpers since the initializer is an expression).
                }
                ast::Item::Constexpr(c) => {
                    if sym.constexprs.iter().any(|x| x.name == c.name && x.module_path.as_slice() == item_module.as_slice()) {
                        errors.push(SemaError { msg: format!("duplicate constexpr `{}`", c.name), span: c.span });
                        continue;
                    }
                    sym.constexprs.push(ConstexprInfo {
                        name: c.name.clone(),
                        value: c.value,
                        is_pub: c.is_pub,
                        module_path: item_module.clone(),
                        span: c.span,
                    });
                }
                ast::Item::Attr(a) => {
                    if sym.attr_by_name(&a.name).is_some() {
                        errors.push(SemaError { msg: format!("duplicate attr `{}`", a.name), span: a.span });
                        continue;
                    }
                    // Capture each method's shape (raw AST so the `_` placeholder can be
                    // substituted per `has` impl) and whether it carries a default body.
                    let methods: Vec<AttrMethod> = a.funcs.iter().map(|f| AttrMethod {
                        name: f.name.clone(),
                        decl: f.clone(),
                        has_default: !f.body.stmts.is_empty(),
                    }).collect();
                    // Reject duplicate method names within the attr decl.
                    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                    for m in &methods {
                        if !seen.insert(m.name.clone()) {
                            errors.push(SemaError {
                                msg: format!("attr `{}` declares method `{}` twice", a.name, m.name),
                                span: m.decl.span,
                            });
                        }
                    }
                    let assoc_type_decls: Vec<(String, Option<HType>, _)> = a.assoc_types.iter()
                        .map(|d| {
                            let default = d.default.as_ref().map(|t| resolve_type_in(&sym, t, &a.type_params, &mut errors));
                            (d.name.clone(), default, d.span)
                        })
                        .collect();
                    sym.attrs.push(AttrInfo {
                        name: a.name.clone(),
                        type_params: a.type_params.clone(),
                        is_pub: a.is_pub,
                        module_path: item_module.clone(),
                        span: a.span,
                        methods,
                        assoc_type_decls,
                    });
                }
                ast::Item::Has(h) => {
                    // Validate referenced attr exists.
                    let attr_info = match sym.attr_by_name(&h.attr_name) {
                        Some(a) => a.clone(),
                        None => {
                            errors.push(SemaError {
                                msg: format!("`has` references unknown attr `{}`", h.attr_name),
                                span: h.span,
                            });
                            continue;
                        }
                    };
                    // Validate the receiver pattern.  Accepted shapes:
                    //   (a) a known struct or enum name (legacy concrete receiver)
                    //   (b) a primitive name (`int`, `bool`, sized ints, `char`, `string`, `float`)
                    //   (c) a parametric pointer / reference / generic receiver
                    //       (`*T`, `&T`, `own *T`, `raw *T`, `Box<T>` — §10.4).
                    let primitive_names = [
                        "int", "bool", "char", "string", "float", "unit",
                        "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                        "isize", "usize", "f32", "f64",
                    ];
                    let is_struct_or_enum = sym.struct_by_name(&h.type_name).is_some()
                        || sym.enum_by_name(&h.type_name).is_some();
                    let is_primitive = primitive_names.contains(&h.type_name.as_str());
                    let is_parametric = !matches!(&h.receiver, maka_ast::Type::Named(_, _));
                    if !(is_struct_or_enum || is_primitive || is_parametric) {
                        errors.push(SemaError {
                            msg: format!("`has` references unknown type `{}`", h.type_name),
                            span: h.span,
                        });
                        continue;
                    }
                    // Resolve concrete attr args (`Color has Convert<int> { ... }`).
                    // Arity must match the attr's declared `type_params`.
                    let attr_args: Vec<HType> = h.attr_args.iter()
                        .map(|t| resolve_type(&sym, t, &mut errors))
                        .collect();
                    if attr_args.len() != attr_info.type_params.len() {
                        errors.push(SemaError {
                            msg: format!(
                                "`{} has {}` provides {} attr arg(s) but the attr declares {}",
                                h.type_name, h.attr_name, attr_args.len(), attr_info.type_params.len(),
                            ),
                            span: h.span,
                        });
                    }
                    // Resolve the receiver pattern to an HType.  Type-variable
                    // identifiers introduced by the receiver (e.g. `T` in `*T`)
                    // are collected so we know which identifiers in the impl's
                    // assoc-type defs are free variables to be substituted at
                    // call sites.
                    let receiver_tyvars = collect_receiver_tyvars(&sym, &h.receiver);
                    let receiver_pattern = resolve_type_in(&sym, &h.receiver, &receiver_tyvars, &mut errors);
                    // Resolve assoc-type definitions in the impl, with the
                    // receiver's type variables in scope.
                    let mut assoc_type_defs: Vec<(String, HType)> = Vec::new();
                    let mut def_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                    for d in &h.assoc_type_defs {
                        if !def_seen.insert(d.name.clone()) {
                            errors.push(SemaError {
                                msg: format!("`type {} = ...` defined twice in `{} has {}`", d.name, h.type_name, h.attr_name),
                                span: d.span,
                            });
                            continue;
                        }
                        let v = resolve_type_in(&sym, &d.value, &receiver_tyvars, &mut errors);
                        assoc_type_defs.push((d.name.clone(), v));
                    }
                    // Validate: every declared assoc-type in the attr must be
                    // provided by the impl OR have a default; no extras allowed.
                    let decl_names: std::collections::HashSet<String> =
                        attr_info.assoc_type_decls.iter().map(|(n, _, _)| n.clone()).collect();
                    for (n, _) in &assoc_type_defs {
                        if !decl_names.contains(n) {
                            errors.push(SemaError {
                                msg: format!("`{} has {}`: `type {}` not declared by attr `{}`", h.type_name, h.attr_name, n, h.attr_name),
                                span: h.span,
                            });
                        }
                    }
                    for (decl_name, default_ty, decl_sp) in &attr_info.assoc_type_decls {
                        if !def_seen.contains(decl_name) {
                            // No def from impl — inherit default if attr has one.
                            if let Some(d) = default_ty {
                                assoc_type_defs.push((decl_name.clone(), d.clone()));
                            } else {
                                errors.push(SemaError {
                                    msg: format!("`{} has {}` is missing `type {} = ...;` required by attr `{}`", h.type_name, h.attr_name, decl_name, h.attr_name),
                                    span: *decl_sp,
                                });
                            }
                        }
                    }
                    // §10.4 coherence: reject overlap with any prior impl of
                    // the same attr.  Overlap is "some concrete type unifies
                    // with both receiver patterns".  We rely on the
                    // pattern-vs-pattern unification helper.
                    let mut overlapped = false;
                    for prior in sym.has_impls.iter() {
                        if prior.attr_name != h.attr_name { continue; }
                        // Two impls of the same attr conflict only if BOTH the
                        // receiver patterns overlap AND the attr-args match
                        // exactly.  `Foo has Convert<int>` and
                        // `Foo has Convert<string>` are disjoint via attr-args
                        // even though the receivers are identical.
                        if prior.attr_args.len() != attr_args.len() { continue; }
                        if !prior.attr_args.iter().zip(attr_args.iter())
                            .all(|(a, b)| crate::typeck::type_eq(a, b)) { continue; }
                        if patterns_overlap(
                            &prior.receiver_pattern, &prior.receiver_tyvars,
                            &receiver_pattern, &receiver_tyvars,
                        ) {
                            errors.push(SemaError {
                                msg: format!(
                                    "overlapping `has {}` impls for receiver `{}` and `{}` — receivers unify with a common concrete type, which makes method dispatch ambiguous (§10.4)",
                                    h.attr_name, prior.type_key, h.type_name,
                                ),
                                span: h.span,
                            });
                            overlapped = true;
                            break;
                        }
                    }
                    if overlapped { continue; }
                    // Record the impl: `<T: Attr>` bound is satisfied when T == type_name,
                    // subject to visibility filtering at bound-check time.
                    sym.trait_impls.entry(h.attr_name.clone())
                        .or_default()
                        .insert(h.type_name.clone());
                    sym.has_impls.push(HasImpl {
                        attr_name: h.attr_name.clone(),
                        type_key: h.type_name.clone(),
                        attr_args,
                        is_pub: h.is_pub,
                        module_path: item_module.clone(),
                        func_ids: Vec::new(),
                        assoc_type_defs,
                        receiver_tyvars: receiver_tyvars.clone(),
                        receiver_pattern,
                    });
                    let has_impl_idx = sym.has_impls.len() - 1;
                    // Contract-match: every `has` method must correspond to an attr decl;
                    // every attr method must either be implemented or have a default body.
                    let mut impl_index: std::collections::HashMap<String, &ast::FuncDecl> =
                        std::collections::HashMap::new();
                    for f in &h.funcs {
                        if impl_index.insert(f.name.clone(), f).is_some() {
                            errors.push(SemaError {
                                msg: format!("`{}` implemented twice in `{} has {}`", f.name, h.type_name, h.attr_name),
                                span: f.span,
                            });
                        }
                        if !attr_info.methods.iter().any(|m| m.name == f.name) {
                            errors.push(SemaError {
                                msg: format!(
                                    "method `{}` not declared by attr `{}` — `has` blocks may only implement attr-declared methods",
                                    f.name, h.attr_name,
                                ),
                                span: f.span,
                            });
                        }
                    }
                    // Build the final method-decl list: explicit impls first, defaults filled in.
                    let mut resolved_funcs: Vec<(ast::FuncDecl, bool /* is_default */)> = Vec::new();
                    for am in &attr_info.methods {
                        if let Some(user_decl) = impl_index.get(&am.name) {
                            // Shape-check user impl against the attr declaration (after `_`
                            // substitution into the impl type AND substitution of the attr's
                            // generic args).
                            let last_idx = sym.has_impls.len().saturating_sub(1);
                            let attr_args_for_check = sym.has_impls.get(last_idx)
                                .map(|h| h.attr_args.clone())
                                .unwrap_or_default();
                            check_attr_shape_ty(
                                &sym, &am.decl, user_decl,
                                &h.type_name, &h.receiver, &receiver_tyvars,
                                &h.attr_name,
                                &attr_info.type_params, &attr_args_for_check,
                                &mut errors,
                            );
                            resolved_funcs.push(((*user_decl).clone(), false));
                        } else if am.has_default {
                            // Synthesize: clone the attr's decl and rewrite `_` → impl type in
                            // both signature and body (in-type positions).
                            let synth = synthesize_default_ty(&am.decl, &h.receiver);
                            resolved_funcs.push((synth, true));
                        } else {
                            errors.push(SemaError {
                                msg: format!(
                                    "missing impl of `{}` for `{} has {}` (no default body in attr `{}`)",
                                    am.name, h.type_name, h.attr_name, h.attr_name,
                                ),
                                span: h.span,
                            });
                        }
                    }
                    // Register each method as an attr-qualified sig.  `_` in params/ret is
                    // already gone (substituted at AST level above), so normal resolution works.
                    let mut name_seen: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
                    for (f, _is_default) in &resolved_funcs {
                        // Substitute `_` with the receiver's full AST Type tree
                        // (so `&_ self` in a `*T has Foo` impl becomes `&*T self`).
                        // The receiver's own type variables (`T` in `*T`) must
                        // also be added to the method's type_params so they
                        // resolve to TyVars during signature/body resolution.
                        let mut f_subst = substitute_func_placeholders_ty(f, &h.receiver);
                        for v in &receiver_tyvars {
                            if !f_subst.type_params.contains(v) {
                                f_subst.type_params.push(v.clone());
                            }
                        }
                        let (param_tys, ret) = resolve_signature(&sym, &f_subst.params, &f_subst.ret, &f_subst.type_params, &mut errors);
                        let param_names: Vec<String> = f_subst.params.iter().map(|p| p.name.clone()).collect();
                        let overload_idx = *name_seen.entry(f_subst.name.clone()).and_modify(|v| *v += 1).or_insert(0);
                        // Include attr-args in the c_name so distinct impls of the
                        // same method on the same type (e.g. `Cents has Convert<int>`
                        // and `Cents has Convert<string>`) get unique symbols.
                        let attr_args_suffix: String = {
                            let hi = &sym.has_impls[has_impl_idx];
                            if hi.attr_args.is_empty() { String::new() }
                            else {
                                format!("__{}", hi.attr_args.iter().map(|t| t.key()).collect::<Vec<_>>().join("_"))
                            }
                        };
                        let c_name = if overload_idx == 0 {
                            format!("{}{}__{}__{}", h.attr_name, attr_args_suffix, h.type_name, f_subst.name)
                        } else {
                            format!("{}{}__{}__{}_{}", h.attr_name, attr_args_suffix, h.type_name, f_subst.name, overload_idx)
                        };
                        let fid = FuncId(sym.sigs.len() as u32);
                        sym.sigs.push(FuncSig {
                            name: f_subst.name.clone(),
                            param_tys, param_names, ret,
                            is_extern: false,
                            c_name,
                            logic: Some(h.attr_name.clone()),
                            type_params: f_subst.type_params.clone(),
                            is_inline: f_subst.is_inline,
                            is_gate: f_subst.is_gate,
                            is_variadic: false,
                            is_pub: h.is_pub,
                            module_path: item_module.clone(),
                            imports: item_imports.clone(),
                            has_imports: item_has_imports.clone(),
                            where_bounds: Vec::new(),
                        });
                        sym.ast_funcs.insert(fid.0, f_subst);
                        sym.has_impls[has_impl_idx].func_ids.push(fid);
                    }
                }
                _ => {}
            }
        }

        // Pass 4: enforce data-decl `where` bounds at every concrete instantiation.
        // Must run after pass 3 so `trait_impls` (populated by Item::Has / Item::Logic
        // arms) is up to date — the bound check otherwise sees an empty impl table.
        for (req_name, req_args) in &struct_inst_requests {
            let Some((_, tinfo)) = sym.struct_by_name(req_name) else { continue; };
            let template_bounds = tinfo.where_bounds.clone();
            let tspan = tinfo.span;
            if template_bounds.is_empty() { continue; }
            let env: std::collections::HashMap<String, HType> = tinfo.type_params.iter().cloned().zip(req_args.iter().cloned()).collect();
            for (trait_name, type_args, _bindings) in &template_bounds {
                for a in type_args {
                    let concrete = a.subst(&env);
                    let satisfied = underlying_struct_key(&sym, &concrete).as_ref()
                        .and_then(|k| sym.trait_impls.get(trait_name).map(|s| s.contains(k)))
                        .unwrap_or(false);
                    if !satisfied {
                        let pretty = underlying_struct_key(&sym, &concrete)
                            .unwrap_or_else(|| crate::typeck::type_str(&concrete));
                        errors.push(SemaError {
                            msg: format!(
                                "type `{}` does not satisfy `{}` bound at instantiation of `data {}`",
                                pretty, trait_name, req_name,
                            ),
                            span: tspan,
                        });
                    }
                }
            }
        }

        // Pass 5: walk all struct instantiations and re-resolve any AssocType
        // placeholders in their field types.  Struct instantiation happened
        // in Pass 2b — before Pass 3 registered `has` impls — so any
        // `T::Slot` field types were left abstract.  Now that impls are
        // registered, we can resolve them concretely.
        let sym_snapshot = sym.clone();
        for s in sym.structs.iter_mut() {
            for f in s.fields.iter_mut() {
                f.ty = resolve_assoc_types_in(&sym_snapshot, &f.ty);
            }
        }

        if errors.is_empty() { Ok(sym) } else { Err(errors) }
    }
}

/// Walk an AST type looking for `Generic { name, args }` references and record them
/// (with resolved HTypes for the args) so they can be instantiated.
fn scan_struct_insts(
    sym: &SymTab,
    t: &ast::Type,
    type_params: &[String],
    out: &mut Vec<(String, Vec<HType>)>,
    errors: &mut Vec<SemaError>,
) {
    match t {
        ast::Type::Generic { name, args, .. } => {
            // `Rust<T>` is a built-in bridge type (see resolve_type_in); no
            // instantiation needed — it lowers directly to `own *mut unit`.
            if name == "Rust" && args.len() == 1 {
                return;
            }
            for a in args { scan_struct_insts(sym, a, type_params, out, errors); }
            let resolved: Vec<HType> = args.iter().map(|a| resolve_type_in(sym, a, type_params, errors)).collect();
            // Only record if all args are concrete (no TyVars) — a fully-applied instantiation.
            let any_tyvar = resolved.iter().any(|t| has_tyvar(t));
            if !any_tyvar {
                out.push((name.clone(), resolved));
            }
        }
        ast::Type::Ref { inner, .. } | ast::Type::Ptr { inner, .. } | ast::Type::RawPtr { inner, .. } | ast::Type::OwnPtr { inner, .. } | ast::Type::Heap { inner, .. } => {
            scan_struct_insts(sym, inner, type_params, out, errors);
        }
        ast::Type::Array { elem, .. } | ast::Type::Slice { elem, .. } | ast::Type::Vec { elem, .. } => {
            scan_struct_insts(sym, elem, type_params, out, errors);
        }
        _ => {}
    }
}

fn scan_block(sym: &SymTab, b: &ast::Block, tp: &[String], out: &mut Vec<(String, Vec<HType>)>, errors: &mut Vec<SemaError>) {
    for s in &b.stmts {
        scan_stmt(sym, s, tp, out, errors);
    }
}

fn scan_stmt(sym: &SymTab, s: &ast::Stmt, tp: &[String], out: &mut Vec<(String, Vec<HType>)>, errors: &mut Vec<SemaError>) {
    match s {
        ast::Stmt::Let { ty, init, .. } => {
            scan_struct_insts(sym, ty, tp, out, errors);
            scan_struct_insts_expr(sym, init, tp, out, errors);
        }
        ast::Stmt::Assign { place, value, .. } => {
            scan_struct_insts_expr(sym, place, tp, out, errors);
            scan_struct_insts_expr(sym, value, tp, out, errors);
        }
        ast::Stmt::ExprStmt(e, _) => scan_struct_insts_expr(sym, e, tp, out, errors),
        ast::Stmt::Return(Some(e), _) => scan_struct_insts_expr(sym, e, tp, out, errors),
        ast::Stmt::Return(None, _) => {}
        ast::Stmt::If { cond, then_block, else_block, .. } => {
            scan_struct_insts_expr(sym, cond, tp, out, errors);
            scan_block(sym, then_block, tp, out, errors);
            if let Some(b) = else_block { scan_block(sym, b, tp, out, errors); }
        }
        ast::Stmt::While { cond, body, .. } => {
            scan_struct_insts_expr(sym, cond, tp, out, errors);
            scan_block(sym, body, tp, out, errors);
        }
        ast::Stmt::Block(b) | ast::Stmt::Unsafe(b, _) => scan_block(sym, b, tp, out, errors),
        ast::Stmt::ForRange { var_ty, start, end, body, .. } => {
            scan_struct_insts(sym, var_ty, tp, out, errors);
            scan_struct_insts_expr(sym, start, tp, out, errors);
            scan_struct_insts_expr(sym, end, tp, out, errors);
            scan_block(sym, body, tp, out, errors);
        }
        ast::Stmt::ForEach { var_ty, src, body, .. } => {
            scan_struct_insts(sym, var_ty, tp, out, errors);
            scan_struct_insts_expr(sym, src, tp, out, errors);
            scan_block(sym, body, tp, out, errors);
        }
        ast::Stmt::Break(_) | ast::Stmt::Continue(_) => {}
        ast::Stmt::Match { scrutinee, arms, .. } => {
            scan_struct_insts_expr(sym, scrutinee, tp, out, errors);
            for a in arms {
                if let Some(g) = &a.guard { scan_struct_insts_expr(sym, g, tp, out, errors); }
                match &a.body {
                    ast::ArmBody::Expr(e) => scan_struct_insts_expr(sym, e, tp, out, errors),
                    ast::ArmBody::Block(b) => scan_block(sym, b, tp, out, errors),
                }
            }
        }
        ast::Stmt::Yield(e, _) => scan_struct_insts_expr(sym, e, tp, out, errors),
        ast::Stmt::Propagate(Some(e), _) => scan_struct_insts_expr(sym, e, tp, out, errors),
        ast::Stmt::Propagate(None, _) => {}
    }
}

fn scan_struct_insts_expr(sym: &SymTab, e: &ast::Expr, tp: &[String], out: &mut Vec<(String, Vec<HType>)>, errors: &mut Vec<SemaError>) {
    match e {
        ast::Expr::Cast { ty, expr, .. } | ast::Expr::CheckedCast { ty, expr, .. } => {
            scan_struct_insts(sym, ty, tp, out, errors);
            scan_struct_insts_expr(sym, expr, tp, out, errors);
        }
        ast::Expr::Bin { lhs, rhs, .. } => { scan_struct_insts_expr(sym, lhs, tp, out, errors); scan_struct_insts_expr(sym, rhs, tp, out, errors); }
        ast::Expr::Un { expr, .. } => scan_struct_insts_expr(sym, expr, tp, out, errors),
        ast::Expr::Unwrap { expr, .. } => scan_struct_insts_expr(sym, expr, tp, out, errors),
        ast::Expr::Ref { expr, .. } => scan_struct_insts_expr(sym, expr, tp, out, errors),
        ast::Expr::Field { base, .. } => scan_struct_insts_expr(sym, base, tp, out, errors),
        ast::Expr::Index { base, idx, .. } => { scan_struct_insts_expr(sym, base, tp, out, errors); scan_struct_insts_expr(sym, idx, tp, out, errors); }
        ast::Expr::Call { callee, args, .. } => {
            scan_struct_insts_expr(sym, callee, tp, out, errors);
            for a in args { scan_struct_insts_expr(sym, a, tp, out, errors); }
        }
        ast::Expr::Struct { fields, .. } => for (_, fe) in fields { scan_struct_insts_expr(sym, fe, tp, out, errors); },
        ast::Expr::ArrayLit { elems, .. } => for e in elems { scan_struct_insts_expr(sym, e, tp, out, errors); },
        ast::Expr::HeapAlloc { value, .. } => scan_struct_insts_expr(sym, value, tp, out, errors),
        ast::Expr::WallMod { expr, .. } => scan_struct_insts_expr(sym, expr, tp, out, errors),
        ast::Expr::VariantCtor { fields, .. } => for (_, fe) in fields { scan_struct_insts_expr(sym, fe, tp, out, errors); },
        ast::Expr::Match { scrutinee, arms, .. } => {
            scan_struct_insts_expr(sym, scrutinee, tp, out, errors);
            for a in arms {
                if let Some(g) = &a.guard { scan_struct_insts_expr(sym, g, tp, out, errors); }
                match &a.body {
                    ast::ArmBody::Expr(e) => scan_struct_insts_expr(sym, e, tp, out, errors),
                    ast::ArmBody::Block(b) => scan_block(sym, b, tp, out, errors),
                }
            }
        }
        _ => {}
    }
}

fn has_tyvar(t: &HType) -> bool {
    match t {
        HType::TyVar(_) => true,
        HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } | HType::Heap { inner } => has_tyvar(inner),
        HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => has_tyvar(elem),
        _ => false,
    }
}

// ============================================================================
//  attr / has helpers
// ============================================================================

/// Substitute every `_` placeholder occurrence in a FuncDecl's signature AND
/// body with the implementing type name.  Used both for synthesizing default
/// methods inherited from an `attr` decl and for normalizing user-written
/// `has` blocks that use `_` for the receiver.
pub fn substitute_func_placeholders(f: &ast::FuncDecl, impl_ty: &str) -> ast::FuncDecl {
    let mut out = f.clone();
    out.ret = out.ret.subst_placeholder(impl_ty);
    for p in out.params.iter_mut() { p.ty = p.ty.subst_placeholder(impl_ty); }
    substitute_block_placeholders(&mut out.body, impl_ty);
    out
}

/// Type-tree-aware variant of `substitute_func_placeholders`: substitutes the
/// `_` placeholder with the receiver's full AST Type tree (carrying type
/// variables in their parametric positions).  Used by parametric `has` impls
/// (§10.4).
pub fn substitute_func_placeholders_ty(f: &ast::FuncDecl, recv: &ast::Type) -> ast::FuncDecl {
    let mut out = f.clone();
    out.ret = out.ret.subst_placeholder_ty(recv);
    for p in out.params.iter_mut() { p.ty = p.ty.subst_placeholder_ty(recv); }
    substitute_block_placeholders_ty(&mut out.body, recv);
    out
}

fn substitute_block_placeholders_ty(b: &mut ast::Block, recv: &ast::Type) {
    for s in &mut b.stmts { substitute_stmt_placeholders_ty(s, recv); }
}

fn substitute_stmt_placeholders_ty(s: &mut ast::Stmt, recv: &ast::Type) {
    use ast::Stmt::*;
    match s {
        Let { ty, init, .. } => {
            *ty = ty.subst_placeholder_ty(recv);
            substitute_expr_placeholders_ty(init, recv);
        }
        Assign { place, value, .. } => {
            substitute_expr_placeholders_ty(place, recv);
            substitute_expr_placeholders_ty(value, recv);
        }
        Return(opt, _) => { if let Some(e) = opt { substitute_expr_placeholders_ty(e, recv); } }
        ExprStmt(e, _) => substitute_expr_placeholders_ty(e, recv),
        If { cond, then_block, else_block, .. } => {
            substitute_expr_placeholders_ty(cond, recv);
            substitute_block_placeholders_ty(then_block, recv);
            if let Some(eb) = else_block { substitute_block_placeholders_ty(eb, recv); }
        }
        While { cond, body, .. } => {
            substitute_expr_placeholders_ty(cond, recv);
            substitute_block_placeholders_ty(body, recv);
        }
        ForRange { var_ty, start, end, body, .. } => {
            *var_ty = var_ty.subst_placeholder_ty(recv);
            substitute_expr_placeholders_ty(start, recv);
            substitute_expr_placeholders_ty(end, recv);
            substitute_block_placeholders_ty(body, recv);
        }
        ForEach { var_ty, src, body, .. } => {
            *var_ty = var_ty.subst_placeholder_ty(recv);
            substitute_expr_placeholders_ty(src, recv);
            substitute_block_placeholders_ty(body, recv);
        }
        Block(b) | Unsafe(b, _) => substitute_block_placeholders_ty(b, recv),
        Match { scrutinee, arms, .. } => {
            substitute_expr_placeholders_ty(scrutinee, recv);
            for a in arms {
                if let Some(g) = a.guard.as_mut() { substitute_expr_placeholders_ty(g, recv); }
                match &mut a.body {
                    ast::ArmBody::Expr(e) => substitute_expr_placeholders_ty(e, recv),
                    ast::ArmBody::Block(b) => substitute_block_placeholders_ty(b, recv),
                }
            }
        }
        Yield(e, _) => substitute_expr_placeholders_ty(e, recv),
        Propagate(opt, _) => if let Some(e) = opt { substitute_expr_placeholders_ty(e, recv); },
        Break(_) | Continue(_) => {}
    }
}

fn substitute_expr_placeholders_ty(e: &mut ast::Expr, recv: &ast::Type) {
    use ast::Expr::*;
    match e {
        Lit(_, _) | Ident(_, _) => {}
        Bin { lhs, rhs, .. } => {
            substitute_expr_placeholders_ty(lhs, recv);
            substitute_expr_placeholders_ty(rhs, recv);
        }
        Un { expr, .. } | Unwrap { expr, .. } | Ref { expr, .. }
        | HeapAlloc { value: expr, .. } | Free { value: expr, .. } => {
            substitute_expr_placeholders_ty(expr, recv);
        }
        Field { base, .. } => substitute_expr_placeholders_ty(base, recv),
        Index { base, idx, .. } => {
            substitute_expr_placeholders_ty(base, recv);
            substitute_expr_placeholders_ty(idx, recv);
        }
        Call { callee, args, .. } => {
            substitute_expr_placeholders_ty(callee, recv);
            for a in args { substitute_expr_placeholders_ty(a, recv); }
        }
        Cast { expr, ty, .. } | CheckedCast { expr, ty, .. } => {
            substitute_expr_placeholders_ty(expr, recv);
            *ty = ty.subst_placeholder_ty(recv);
        }
        Struct { fields, .. } | VariantCtor { fields, .. } => {
            for (_, fe) in fields.iter_mut() { substitute_expr_placeholders_ty(fe, recv); }
        }
        ArrayLit { elems, .. } => for el in elems { substitute_expr_placeholders_ty(el, recv); },
        Match { scrutinee, arms, .. } => {
            substitute_expr_placeholders_ty(scrutinee, recv);
            for a in arms {
                if let Some(g) = a.guard.as_mut() { substitute_expr_placeholders_ty(g, recv); }
                match &mut a.body {
                    ast::ArmBody::Expr(e) => substitute_expr_placeholders_ty(e, recv),
                    ast::ArmBody::Block(b) => substitute_block_placeholders_ty(b, recv),
                }
            }
        }
        Lambda { ret, params, body, .. } => {
            *ret = ret.subst_placeholder_ty(recv);
            for p in params { p.ty = p.ty.subst_placeholder_ty(recv); }
            match body {
                ast::LambdaBody::Expr(b) => substitute_expr_placeholders_ty(b, recv),
                ast::LambdaBody::Block(b) => substitute_block_placeholders_ty(b, recv),
            }
        }
        WallMod { expr, .. } => substitute_expr_placeholders_ty(expr, recv),
    }
}

fn substitute_block_placeholders(b: &mut ast::Block, impl_ty: &str) {
    for s in &mut b.stmts { substitute_stmt_placeholders(s, impl_ty); }
}

fn substitute_stmt_placeholders(s: &mut ast::Stmt, impl_ty: &str) {
    use ast::Stmt::*;
    match s {
        Let { ty, init, .. } => {
            *ty = ty.subst_placeholder(impl_ty);
            substitute_expr_placeholders(init, impl_ty);
        }
        Assign { place, value, .. } => {
            substitute_expr_placeholders(place, impl_ty);
            substitute_expr_placeholders(value, impl_ty);
        }
        Return(opt, _) => {
            if let Some(e) = opt { substitute_expr_placeholders(e, impl_ty); }
        }
        ExprStmt(e, _) => substitute_expr_placeholders(e, impl_ty),
        If { cond, then_block, else_block, .. } => {
            substitute_expr_placeholders(cond, impl_ty);
            substitute_block_placeholders(then_block, impl_ty);
            if let Some(eb) = else_block { substitute_block_placeholders(eb, impl_ty); }
        }
        While { cond, body, .. } => {
            substitute_expr_placeholders(cond, impl_ty);
            substitute_block_placeholders(body, impl_ty);
        }
        ForRange { var_ty, start, end, body, .. } => {
            *var_ty = var_ty.subst_placeholder(impl_ty);
            substitute_expr_placeholders(start, impl_ty);
            substitute_expr_placeholders(end, impl_ty);
            substitute_block_placeholders(body, impl_ty);
        }
        ForEach { var_ty, src, body, .. } => {
            *var_ty = var_ty.subst_placeholder(impl_ty);
            substitute_expr_placeholders(src, impl_ty);
            substitute_block_placeholders(body, impl_ty);
        }
        Block(b) => substitute_block_placeholders(b, impl_ty),
        Unsafe(b, _) => substitute_block_placeholders(b, impl_ty),
        Match { scrutinee, arms, .. } => {
            substitute_expr_placeholders(scrutinee, impl_ty);
            for a in arms {
                if let Some(g) = a.guard.as_mut() { substitute_expr_placeholders(g, impl_ty); }
                match &mut a.body {
                    ast::ArmBody::Expr(e) => substitute_expr_placeholders(e, impl_ty),
                    ast::ArmBody::Block(b) => substitute_block_placeholders(b, impl_ty),
                }
            }
        }
        Yield(e, _) => substitute_expr_placeholders(e, impl_ty),
        Propagate(opt, _) => if let Some(e) = opt { substitute_expr_placeholders(e, impl_ty); },
        Break(_) | Continue(_) => {}
    }
}

fn substitute_expr_placeholders(e: &mut ast::Expr, impl_ty: &str) {
    use ast::Expr::*;
    match e {
        Lit(_, _) | Ident(_, _) => {}
        Bin { lhs, rhs, .. } => {
            substitute_expr_placeholders(lhs, impl_ty);
            substitute_expr_placeholders(rhs, impl_ty);
        }
        Un { expr, .. } | Unwrap { expr, .. } | Ref { expr, .. }
        | HeapAlloc { value: expr, .. } | Free { value: expr, .. } => {
            substitute_expr_placeholders(expr, impl_ty);
        }
        Field { base, .. } => substitute_expr_placeholders(base, impl_ty),
        Index { base, idx, .. } => {
            substitute_expr_placeholders(base, impl_ty);
            substitute_expr_placeholders(idx, impl_ty);
        }
        Call { callee, args, .. } => {
            substitute_expr_placeholders(callee, impl_ty);
            for a in args { substitute_expr_placeholders(a, impl_ty); }
        }
        Cast { expr, ty, .. } | CheckedCast { expr, ty, .. } => {
            substitute_expr_placeholders(expr, impl_ty);
            *ty = ty.subst_placeholder(impl_ty);
        }
        Struct { fields, .. } | VariantCtor { fields, .. } => {
            for (_, fe) in fields.iter_mut() { substitute_expr_placeholders(fe, impl_ty); }
        }
        ArrayLit { elems, .. } => for el in elems { substitute_expr_placeholders(el, impl_ty); }
        Match { scrutinee, arms, .. } => {
            substitute_expr_placeholders(scrutinee, impl_ty);
            for a in arms {
                if let Some(g) = a.guard.as_mut() { substitute_expr_placeholders(g, impl_ty); }
                match &mut a.body {
                    ast::ArmBody::Expr(e) => substitute_expr_placeholders(e, impl_ty),
                    ast::ArmBody::Block(b) => substitute_block_placeholders(b, impl_ty),
                }
            }
        }
        Lambda { ret, params, body, .. } => {
            *ret = ret.subst_placeholder(impl_ty);
            for p in params { p.ty = p.ty.subst_placeholder(impl_ty); }
            match body {
                ast::LambdaBody::Expr(b) => substitute_expr_placeholders(b, impl_ty),
                ast::LambdaBody::Block(b) => substitute_block_placeholders(b, impl_ty),
            }
        }
        WallMod { expr, .. } => substitute_expr_placeholders(expr, impl_ty),
    }
}

/// Validate that a `has`-block method's signature matches the attr's declaration
/// (after `_` substitution).  Mismatches are reported via `errors`.
pub fn check_attr_shape(
    sym: &SymTab,
    attr_decl: &ast::FuncDecl,
    has_decl: &ast::FuncDecl,
    impl_ty: &str,
    attr_name: &str,
    attr_type_params: &[String],
    attr_args: &[HType],
    errors: &mut Vec<SemaError>,
) {
    check_attr_shape_ty(sym, attr_decl, has_decl, impl_ty, &ast::Type::Named(impl_ty.to_string(), maka_lexer::Span::dummy()), &[], attr_name, attr_type_params, attr_args, errors);
}

/// Type-tree-aware variant: substitute `_` with the receiver's full AST Type
/// tree (so `&_ self` becomes e.g. `&*T self` for a `*T has Foo` impl),
/// and resolve the impl-side signature with the receiver's type variables in
/// scope so they become TyVars (not "unknown type") during the contract check.
pub fn check_attr_shape_ty(
    sym: &SymTab,
    attr_decl: &ast::FuncDecl,
    has_decl: &ast::FuncDecl,
    impl_ty_name: &str,
    receiver: &ast::Type,
    receiver_tyvars: &[String],
    attr_name: &str,
    attr_type_params: &[String],
    attr_args: &[HType],
    errors: &mut Vec<SemaError>,
) {
    let impl_ty = impl_ty_name;
    // Substitute `_` in the attr decl to the receiver Type tree, then resolve
    // with attr type-params AND receiver tyvars in scope.
    let attr_subst = substitute_func_placeholders_ty(attr_decl, receiver);
    let combined: Vec<String> = attr_type_params.iter()
        .chain(attr_subst.type_params.iter())
        .chain(receiver_tyvars.iter())
        .cloned().collect();
    let (a_params_raw, a_ret_raw) = resolve_signature(sym, &attr_subst.params, &attr_subst.ret, &combined, &mut Vec::new());
    let env: std::collections::HashMap<String, HType> = attr_type_params.iter().cloned().zip(attr_args.iter().cloned()).collect();
    let a_params: Vec<HType> = a_params_raw.iter().map(|t| t.subst(&env)).collect();
    let a_ret = a_ret_raw.subst(&env);
    let has_subst = substitute_func_placeholders_ty(has_decl, receiver);
    let combined_has: Vec<String> = has_subst.type_params.iter()
        .chain(receiver_tyvars.iter())
        .cloned().collect();
    let (h_params, h_ret) = resolve_signature(sym, &has_subst.params, &has_subst.ret, &combined_has, &mut Vec::new());

    if a_params.len() != h_params.len() {
        errors.push(SemaError {
            msg: format!(
                "method `{}` for `{}` has {} parameters but attr `{}` declares {}",
                has_decl.name, impl_ty, h_params.len(), attr_name, a_params.len(),
            ),
            span: has_decl.span,
        });
        return;
    }
    for (i, (a, h)) in a_params.iter().zip(h_params.iter()).enumerate() {
        if !htype_eq(a, h) {
            errors.push(SemaError {
                msg: format!(
                    "method `{}` for `{}`: param {} type `{}` does not match attr `{}` declaration `{}`",
                    has_decl.name, impl_ty, i,
                    crate::typeck::type_str(h), attr_name, crate::typeck::type_str(a),
                ),
                span: has_decl.params.get(i).map(|p| p.span).unwrap_or(has_decl.span),
            });
        }
    }
    if !htype_eq(&a_ret, &h_ret) {
        errors.push(SemaError {
            msg: format!(
                "method `{}` for `{}`: return type `{}` does not match attr `{}` declaration `{}`",
                has_decl.name, impl_ty,
                crate::typeck::type_str(&h_ret), attr_name, crate::typeck::type_str(&a_ret),
            ),
            span: has_decl.span,
        });
    }
}

/// Structural equality between two resolved HTypes — same shape, same nominal IDs.
fn htype_eq(a: &HType, b: &HType) -> bool {
    use HType::*;
    match (a, b) {
        (Int, Int) | (Float, Float) | (Bool, Bool) | (Unit, Unit) | (Str, Str) | (Char, Char) => true,
        (SizedInt { signed: s1, bits: b1 }, SizedInt { signed: s2, bits: b2 }) => s1 == s2 && b1 == b2,
        (Struct(a), Struct(b)) => a == b,
        (Enum(a), Enum(b)) => a == b,
        (TyVar(a), TyVar(b)) => a == b,
        (Ref { mutable: ma, inner: ia }, Ref { mutable: mb, inner: ib }) => ma == mb && htype_eq(ia, ib),
        (Ptr { mutable: ma, inner: ia }, Ptr { mutable: mb, inner: ib }) => ma == mb && htype_eq(ia, ib),
        (RawPtr { mutable: ma, inner: ia }, RawPtr { mutable: mb, inner: ib }) => ma == mb && htype_eq(ia, ib),
        (OwnPtr { mutable: ma, inner: ia }, OwnPtr { mutable: mb, inner: ib }) => ma == mb && htype_eq(ia, ib),
        (Heap { inner: ia }, Heap { inner: ib }) => htype_eq(ia, ib),
        (Array { len: la, elem: ea }, Array { len: lb, elem: eb }) => la == lb && htype_eq(ea, eb),
        (Slice { mutable: ma, elem: ea }, Slice { mutable: mb, elem: eb }) => ma == mb && htype_eq(ea, eb),
        (Vec { elem: ea }, Vec { elem: eb }) => htype_eq(ea, eb),
        (FnPtr { ret: ra, params: pa }, FnPtr { ret: rb, params: pb }) => {
            htype_eq(ra, rb) && pa.len() == pb.len() && pa.iter().zip(pb).all(|(x, y)| htype_eq(x, y))
        }
        _ => false,
    }
}

pub fn synthesize_default(attr_decl: &ast::FuncDecl, impl_ty: &str) -> ast::FuncDecl {
    substitute_func_placeholders(attr_decl, impl_ty)
}

pub fn synthesize_default_ty(attr_decl: &ast::FuncDecl, receiver: &ast::Type) -> ast::FuncDecl {
    substitute_func_placeholders_ty(attr_decl, receiver)
}
