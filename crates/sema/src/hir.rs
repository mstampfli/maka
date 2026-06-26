//! HIR: typed, resolved intermediate representation produced by sema.

use maka_lexer::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumId(pub u32);

/// Resolved type. `mutness` for references/pointers/slices is captured in the constructor.
#[derive(Debug, Clone, PartialEq)]
pub enum HType {
    Int,
    /// Default float — IEEE-754 binary64, lowered to C `double`.  Maka source
    /// spells this `float` (or `f64`).  Same ABI as a C `double` — safe to
    /// use at extern boundaries against C functions that take `double`.
    Float,
    /// Sized float — currently only 32-bit (binary32, C `float`).  Maka source
    /// spells this `f32`.  Distinct from `HType::Float` at the ABI level — a
    /// C function declared `float sinf(float)` must be `extern f32 sinf(f32)`
    /// on the Maka side, NOT `extern float sinf(float)`.
    SizedFloat { bits: u8 },
    Bool,
    Char,
    /// Sized integer: signed/unsigned and bit width (8, 16, 32, 64). 0 bits = pointer-sized.
    SizedInt { signed: bool, bits: u8 },
    Unit,
    Struct(StructId),
    Enum(EnumId),
    Ref { mutable: bool, inner: Box<HType> },
    Ptr { mutable: bool, inner: Box<HType> },
    /// Pointer of unknown provenance (typically from C interop).  Identical wire
    /// representation to `Ptr` but every observation (deref, field, index,
    /// null-narrowing) requires being inside an `unsafe { }` block.
    RawPtr { mutable: bool, inner: Box<HType> },
    /// Nullable owning pointer (`own *T`).  Holds either `null` or an alloc'd
    /// value.  Reassigning frees the old value; scope exit frees whatever's
    /// currently held.  Coerces to `*T` (downgrade) and accepts `null`.
    OwnPtr { mutable: bool, inner: Box<HType> },
    /// Heap appears only as a *storage modifier* on bindings/params/returns.
    /// In type form it is also used to mark return/parameter types.
    Heap { inner: Box<HType> },
    Array { len: i64, elem: Box<HType> },
    Slice { mutable: bool, elem: Box<HType> },
    Vec { elem: Box<HType> },
    /// String literal type (lowered to `[]char` essentially); for now its own type.
    Str,
    /// `null` literal before context-driven coercion.
    NullT,
    /// `dyn Trait` (or `dyn (A + B)`) — fat pointer (data, vtable[s]).
    Dyn { traits: Vec<String> },
    /// Function pointer: `RetType(P1, P2, ...)`.
    FnPtr { ret: Box<HType>, params: Vec<HType> },
    /// Unresolved type variable from a generic decl (used during sema's pre-monomorphization phase).
    TyVar(String),
    /// A generic struct/enum **with its arg structure preserved** — used as
    /// the receiver pattern of a parametric `has` impl that references a
    /// generic type with type variables, e.g. `Result<T, E> has Try` stores
    /// `GenericPattern { template_name: "Result", args: [TyVar("T"), TyVar("E")], is_enum: true }`.
    ///
    /// At a concrete call site with `Result<int, MyErr>`, receiver unification
    /// looks up the concrete instantiation's template name + its remembered
    /// `template_args` (stored on StructInfo / EnumInfo at instantiation time)
    /// and unifies arg-by-arg, binding the impl's tyvars.
    ///
    /// Lives **only** at the pattern / receiver layer.  Never reaches codegen.
    GenericPattern { template_name: String, args: Vec<HType>, is_enum: bool },
    /// `T::Slot` — an unresolved associated-type path.  `on` is the receiver
    /// type (typically a TyVar for `T::Slot` inside a generic body; resolved
    /// to a concrete type at monomorphization).  `attr_hint` is `None` unless
    /// the source explicitly qualified the path (reserved for future spec
    /// revisions).  Substitution resolves this to a concrete type at each
    /// instantiation by looking up the impl whose receiver pattern unifies
    /// with `on`.
    AssocType { on: Box<HType>, segment: String, attr_hint: Option<String> },
    /// `Rust<T>` — an opaque handle to a Rust-side heap value.  Layout / ABI
    /// identical to `OwnPtr { mutable: true, inner: Unit }`; the `String`
    /// carries the Rust type name so Maka can route per-call-site
    /// `Send` / `Sync` probes back to the sidecar at thread-crossing
    /// (spawn / transfer / share) sites.
    RustOpaque(String),
}

impl HType {
    pub fn is_heap(&self) -> bool { matches!(self, HType::Heap { .. }) }
    pub fn strip_heap(&self) -> &HType { if let HType::Heap { inner } = self { inner } else { self } }

    /// The owned heap-string type (what `String` and every allocating string
    /// builtin - concat, `read_line`, `format`, `*_to_str` - produce).  Single
    /// source of truth for its internal representation: `own *string`, a mutable
    /// nullable owning pointer to a NUL-terminated buffer, auto-freed at scope
    /// exit.  Lowers to a single `char*` (a string is an array of chars, so
    /// `own *string` is one owned char buffer, never a `char**`).
    pub fn owned_string() -> HType {
        HType::OwnPtr { mutable: true, inner: Box::new(HType::Str) }
    }
    /// True if `self` is the owned heap-string type produced by `owned_string()`.
    pub fn is_owned_string(&self) -> bool {
        matches!(self, HType::OwnPtr { inner, .. } if matches!(**inner, HType::Str))
    }

    /// Replace any TyVar(name) with the matching substitution.
    pub fn subst(&self, env: &std::collections::HashMap<String, HType>) -> HType {
        match self {
            HType::TyVar(n) => env.get(n).cloned().unwrap_or_else(|| self.clone()),
            HType::Ref { mutable, inner } => HType::Ref { mutable: *mutable, inner: Box::new(inner.subst(env)) },
            HType::Ptr { mutable, inner } => HType::Ptr { mutable: *mutable, inner: Box::new(inner.subst(env)) },
            HType::RawPtr { mutable, inner } => HType::RawPtr { mutable: *mutable, inner: Box::new(inner.subst(env)) },
            HType::OwnPtr { mutable, inner } => HType::OwnPtr { mutable: *mutable, inner: Box::new(inner.subst(env)) },
            HType::Heap { inner } => HType::Heap { inner: Box::new(inner.subst(env)) },
            HType::Array { len, elem } => HType::Array { len: *len, elem: Box::new(elem.subst(env)) },
            HType::Slice { mutable, elem } => HType::Slice { mutable: *mutable, elem: Box::new(elem.subst(env)) },
            HType::Vec { elem } => HType::Vec { elem: Box::new(elem.subst(env)) },
            HType::AssocType { on, segment, attr_hint } => HType::AssocType {
                on: Box::new(on.subst(env)),
                segment: segment.clone(),
                attr_hint: attr_hint.clone(),
            },
            HType::GenericPattern { template_name, args, is_enum } => HType::GenericPattern {
                template_name: template_name.clone(),
                args: args.iter().map(|a| a.subst(env)).collect(),
                is_enum: *is_enum,
            },
            HType::FnPtr { ret, params } => HType::FnPtr {
                ret: Box::new(ret.subst(env)),
                params: params.iter().map(|p| p.subst(env)).collect(),
            },
            _ => self.clone(),
        }
    }

    /// Generate a stable key string (used in mangling / dedup).
    pub fn key(&self) -> String {
        match self {
            HType::Int => "int".into(),
            HType::Float => "float".into(),
            HType::SizedFloat { bits } => format!("f{}", bits),
            HType::Bool => "bool".into(),
            HType::Char => "char".into(),
            HType::SizedInt { signed, bits } => format!("{}{}", if *signed {"i"} else {"u"}, bits),
            HType::Unit => "unit".into(),
            HType::Str => "str".into(),
            HType::NullT => "null".into(),
            HType::Struct(i) => format!("S{}", i.0),
            HType::Enum(i) => format!("E{}", i.0),
            HType::Ref { mutable, inner } => format!("R{}{}", if *mutable {"m"} else {"c"}, inner.key()),
            HType::Ptr { mutable, inner } => format!("P{}{}", if *mutable {"m"} else {"c"}, inner.key()),
            HType::RawPtr { mutable, inner } => format!("RP{}{}", if *mutable {"m"} else {"c"}, inner.key()),
            HType::OwnPtr { mutable, inner } => format!("OP{}{}", if *mutable {"m"} else {"c"}, inner.key()),
            HType::Heap { inner } => format!("H{}", inner.key()),
            HType::Array { len, elem } => format!("A{}_{}", len, elem.key()),
            HType::Slice { mutable, elem } => format!("Sl{}{}", if *mutable {"m"} else {"c"}, elem.key()),
            HType::Vec { elem } => format!("V{}", elem.key()),
            HType::Dyn { traits } => format!("D{}", traits.join("+")),
            HType::FnPtr { ret, params } => {
                let mut s = format!("F{}_", ret.key());
                for p in params { s.push_str(&p.key()); s.push('_'); }
                s
            }
            HType::TyVar(n) => format!("'{}", n),
            HType::AssocType { on, segment, .. } => format!("AT{}_{}", on.key(), segment),
            HType::GenericPattern { template_name, args, .. } => {
                let inner: Vec<String> = args.iter().map(|a| a.key()).collect();
                format!("GP{}__{}", template_name, inner.join("_"))
            }
            // Key is shared with `own *mut unit` so monomorphisation, dedup, and
            // type-equality treat the two as the same — the label is purely
            // out-of-band metadata for probe routing.
            HType::RustOpaque(_) => format!("OPm{}", HType::Unit.key()),
        }
    }

    /// Pull the carried Rust type name from a `Rust<T>` opaque, if any.
    /// Returns `None` for any other type — including the normalised
    /// `OwnPtr<Unit>` form that `Rust<T>` shares its layout with.
    pub fn rust_opaque_label(&self) -> Option<&str> {
        if let HType::RustOpaque(s) = self { Some(s.as_str()) } else { None }
    }

    pub fn is_ref_like(&self) -> bool {
        matches!(self, HType::Ref { .. } | HType::Ptr { .. } | HType::Slice { .. })
    }
    pub fn is_pointer(&self) -> bool {
        matches!(self, HType::Ptr { .. })
    }
    pub fn pointer_mutness(&self) -> Option<bool> {
        if let HType::Ptr { mutable, .. } = self { Some(*mutable) } else { None }
    }
    pub fn ref_mutness(&self) -> Option<bool> {
        if let HType::Ref { mutable, .. } = self { Some(*mutable) } else { None }
    }
    pub fn elem(&self) -> Option<&HType> {
        match self {
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } | HType::Heap { inner } => Some(inner),
            HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => Some(elem),
            _ => None,
        }
    }
}

/// Pattern-vs-pattern unification.  Both sides have their own type
/// variables (listed in `lhs_vars` / `rhs_vars`).  Returns `Some(())` iff
/// some concrete type unifies with both patterns simultaneously — i.e. the
/// patterns OVERLAP per §10.4's coherence rule.
pub fn patterns_overlap(
    lhs: &HType,
    lhs_vars: &[String],
    rhs: &HType,
    rhs_vars: &[String],
) -> bool {
    fn go(
        lhs: &HType,
        lvars: &[String],
        rhs: &HType,
        rvars: &[String],
        lenv: &mut std::collections::HashMap<String, HType>,
        renv: &mut std::collections::HashMap<String, HType>,
    ) -> bool {
        // A type variable on either side: bind to the other side.  We track
        // bindings to detect conflicting requirements (`Pair<A, A>` vs
        // `Pair<int, bool>` should NOT overlap because the var A would have
        // to be both int and bool).
        if let HType::TyVar(n) = lhs {
            if lvars.iter().any(|v| v == n) {
                if let Some(prev) = lenv.get(n) {
                    return prev == rhs;
                }
                lenv.insert(n.clone(), rhs.clone());
                return true;
            }
        }
        if let HType::TyVar(n) = rhs {
            if rvars.iter().any(|v| v == n) {
                if let Some(prev) = renv.get(n) {
                    return prev == lhs;
                }
                renv.insert(n.clone(), lhs.clone());
                return true;
            }
        }
        match (lhs, rhs) {
            (HType::Int, HType::Int) | (HType::Bool, HType::Bool)
            | (HType::Char, HType::Char) | (HType::Float, HType::Float)
            | (HType::Str, HType::Str) | (HType::Unit, HType::Unit) => true,
            (HType::SizedInt { signed: ls, bits: lb }, HType::SizedInt { signed: rs, bits: rb })
                => ls == rs && lb == rb,
            (HType::Struct(li), HType::Struct(ri)) => li == ri,
            (HType::Enum(li), HType::Enum(ri)) => li == ri,
            (HType::Ref { mutable: lm, inner: li }, HType::Ref { mutable: rm, inner: ri })
                if lm == rm => go(li, lvars, ri, rvars, lenv, renv),
            (HType::Ptr { mutable: lm, inner: li }, HType::Ptr { mutable: rm, inner: ri })
                if lm == rm => go(li, lvars, ri, rvars, lenv, renv),
            (HType::RawPtr { mutable: lm, inner: li }, HType::RawPtr { mutable: rm, inner: ri })
                if lm == rm => go(li, lvars, ri, rvars, lenv, renv),
            (HType::OwnPtr { mutable: lm, inner: li }, HType::OwnPtr { mutable: rm, inner: ri })
                if lm == rm => go(li, lvars, ri, rvars, lenv, renv),
            (HType::Heap { inner: li }, HType::Heap { inner: ri }) => go(li, lvars, ri, rvars, lenv, renv),
            (HType::Slice { mutable: lm, elem: le }, HType::Slice { mutable: rm, elem: re })
                if lm == rm => go(le, lvars, re, rvars, lenv, renv),
            (HType::Array { len: ll, elem: le }, HType::Array { len: rl, elem: re })
                if ll == rl => go(le, lvars, re, rvars, lenv, renv),
            (HType::Vec { elem: le }, HType::Vec { elem: re }) => go(le, lvars, re, rvars, lenv, renv),
            _ => false,
        }
    }
    let mut lenv = std::collections::HashMap::new();
    let mut renv = std::collections::HashMap::new();
    go(lhs, lhs_vars, rhs, rhs_vars, &mut lenv, &mut renv)
}

/// First-order structural unification of a `has`-receiver PATTERN against a
/// concrete (or partially concrete) HType.  Pattern type variables (listed in
/// `tyvars`) bind to whatever they meet.  Concrete head constructors must
/// match exactly — no implicit subtyping.  Returns the unification env on
/// success.
///
/// Used at:
///   (a) generic call sites — pat = impl's receiver_pattern, actual = the
///       concrete instantiation, to pick the matching impl.
///   (b) associated-type resolution — same.
pub fn receiver_unify(
    pat: &HType,
    actual: &HType,
    tyvars: &[String],
) -> Option<std::collections::HashMap<String, HType>> {
    // Back-compat shim — old callers without a SymTab.  Cannot handle
    // GenericPattern against a concrete Struct/Enum (needs SymTab to
    // recover the concrete's template_args).  New callers should use
    // `receiver_unify_with_sym`.
    receiver_unify_impl(pat, actual, tyvars, None)
}

/// `receiver_unify` that can recover concrete instantiations' template
/// args from the SymTab.  Use this when the actual may be a concrete
/// `Struct(id)`/`Enum(id)` for a generic template.
pub fn receiver_unify_with_sym(
    pat: &HType,
    actual: &HType,
    tyvars: &[String],
    sym: &SymTab,
) -> Option<std::collections::HashMap<String, HType>> {
    receiver_unify_impl(pat, actual, tyvars, Some(sym))
}

fn receiver_unify_impl(
    pat: &HType,
    actual: &HType,
    tyvars: &[String],
    sym: Option<&SymTab>,
) -> Option<std::collections::HashMap<String, HType>> {
    let mut env: std::collections::HashMap<String, HType> = std::collections::HashMap::new();
    fn go(
        pat: &HType,
        actual: &HType,
        tyvars: &[String],
        env: &mut std::collections::HashMap<String, HType>,
        sym: Option<&SymTab>,
    ) -> bool {
        if let HType::TyVar(n) = pat {
            if tyvars.iter().any(|v| v == n) {
                if let Some(prev) = env.get(n) {
                    return prev == actual;
                }
                env.insert(n.clone(), actual.clone());
                return true;
            }
        }
        // §10.4: GenericPattern (impl receiver) vs concrete Struct/Enum
        // (call-site).  Look up the concrete's StructInfo/EnumInfo to read
        // its `template` name + `template_args`, then match arg-by-arg.
        if let (HType::GenericPattern { template_name: pn, args: pargs, is_enum: pe }, sym) = (pat, sym) {
            let Some(sym) = sym else { return false; };
            match actual {
                HType::Struct(sid) => {
                    if *pe { return false; }
                    let info = sym.struct_info(*sid);
                    let Some(t) = info.template.as_ref() else { return false; };
                    if t != pn { return false; }
                    if info.template_args.len() != pargs.len() { return false; }
                    for (pa, ca) in pargs.iter().zip(info.template_args.iter()) {
                        if !go(pa, ca, tyvars, env, Some(sym)) { return false; }
                    }
                    return true;
                }
                HType::Enum(eid) => {
                    if !*pe { return false; }
                    let info = sym.enum_info(*eid);
                    let Some(t) = info.template.as_ref() else { return false; };
                    if t != pn { return false; }
                    if info.template_args.len() != pargs.len() { return false; }
                    for (pa, ca) in pargs.iter().zip(info.template_args.iter()) {
                        if !go(pa, ca, tyvars, env, Some(sym)) { return false; }
                    }
                    return true;
                }
                HType::GenericPattern { template_name: un, args: uargs, is_enum: ue } => {
                    if pn != un || pe != ue || pargs.len() != uargs.len() { return false; }
                    for (pa, ua) in pargs.iter().zip(uargs.iter()) {
                        if !go(pa, ua, tyvars, env, Some(sym)) { return false; }
                    }
                    return true;
                }
                _ => return false,
            }
        }
        match (pat, actual) {
            (HType::Int, HType::Int)
            | (HType::Bool, HType::Bool)
            | (HType::Char, HType::Char)
            | (HType::Float, HType::Float)
            | (HType::Str, HType::Str)
            | (HType::Unit, HType::Unit) => true,
            (HType::SizedInt { signed: ps, bits: pb }, HType::SizedInt { signed: us, bits: ub })
                => ps == us && pb == ub,
            (HType::Struct(pi), HType::Struct(ui)) => pi == ui,
            (HType::Enum(pi), HType::Enum(ui)) => pi == ui,
            (HType::Ref { mutable: pm, inner: pi }, HType::Ref { mutable: um, inner: ui })
                if pm == um => go(pi, ui, tyvars, env, sym),
            (HType::Ptr { mutable: pm, inner: pi }, HType::Ptr { mutable: um, inner: ui })
                if pm == um => go(pi, ui, tyvars, env, sym),
            (HType::RawPtr { mutable: pm, inner: pi }, HType::RawPtr { mutable: um, inner: ui })
                if pm == um => go(pi, ui, tyvars, env, sym),
            (HType::OwnPtr { mutable: pm, inner: pi }, HType::OwnPtr { mutable: um, inner: ui })
                if pm == um => go(pi, ui, tyvars, env, sym),
            (HType::Heap { inner: pi }, HType::Heap { inner: ui }) => go(pi, ui, tyvars, env, sym),
            (HType::Slice { mutable: pm, elem: pe }, HType::Slice { mutable: um, elem: ue })
                if pm == um => go(pe, ue, tyvars, env, sym),
            (HType::Array { len: pl, elem: pe }, HType::Array { len: ul, elem: ue })
                if pl == ul => go(pe, ue, tyvars, env, sym),
            (HType::Vec { elem: pe }, HType::Vec { elem: ue }) => go(pe, ue, tyvars, env, sym),
            _ => false,
        }
    }
    if go(pat, actual, tyvars, &mut env, sym) { Some(env) } else { None }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageClass {
    Stack,
    Heap,
    Static,
    Param,
}

#[derive(Debug, Clone)]
pub struct LocalInfo {
    pub name: String,
    pub ty: HType,
    pub storage: StorageClass,
    /// `mut` on the payload (whether direct writes are allowed)
    pub mut_payload: bool,
    /// Reassignability of the binding itself.
    /// - `*T` bindings: always true.
    /// - Plain mutable values: true.
    /// - References / slices / array handles / heap: false.
    pub reassignable: bool,
    /// `thread_local` modifier: emits `static __thread` in C.
    pub thread_local: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub ty: HType,
    /// `mut` modifier on field type.
    pub mut_payload: bool,
    pub default: Option<HExpr>,
    pub is_embed: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct StructInfo {
    pub name: String,
    /// Generic type parameters declared (`<A, B>`).
    pub type_params: Vec<String>,
    /// For instantiations, the underlying template name (e.g. "Pair") and the concrete args.
    pub template: Option<String>,
    /// For instantiations, the resolved HType args used to build this
    /// monomorphization (e.g. `[Int, Str]` for `Pair<int, string>`).  Empty
    /// for templates and non-generic structs.  Used by receiver_unify so
    /// `GenericPattern { Pair, [TyVar A, TyVar B] }` against a concrete
    /// `Struct(Pair<int, string>_id)` can recover the original args and
    /// bind A=int, B=string.
    pub template_args: Vec<HType>,
    pub fields: Vec<FieldInfo>,
    pub is_pub: bool,
    pub module_path: Vec<String>,
    pub span: Span,
    /// `where` clauses on the data decl — enforced at concrete instantiation time
    /// just like function where-bounds.  Each entry is `(attr_name, type_args)` and
    /// the position-0 substituted type must have a visible `has` impl.
    /// Where-bound: (trait name, args, optional assoc-type bindings).
    /// `assoc_bindings` carries `<T: Foo<Slot = i64>>` constraints; at each
    /// instantiation, the picked impl's `type Slot = R` must type_eq the
    /// bound's value (after substitution).
    pub where_bounds: Vec<(String, Vec<HType>, Vec<(String, HType)>)>,
}

#[derive(Debug, Clone)]
pub struct VariantInfo {
    pub name: String,
    pub tag: i64,
    pub fields: Vec<FieldInfo>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumInfo {
    pub name: String,
    pub type_params: Vec<String>,
    pub variants: Vec<VariantInfo>,
    pub is_pub: bool,
    pub module_path: Vec<String>,
    pub span: Span,
    /// For instantiations, the underlying template name (e.g. "Result").
    /// Empty for templates and non-generic enums.  Mirror of `StructInfo.template`.
    pub template: Option<String>,
    /// For instantiations, the resolved HType args used to build this
    /// monomorphization.  Mirror of `StructInfo.template_args`.
    pub template_args: Vec<HType>,
}

impl EnumInfo {
    pub fn variant_index(&self, name: &str) -> Option<usize> {
        self.variants.iter().position(|v| v.name == name)
    }
    /// True iff every variant is payload-less — a C-style enum.
    pub fn is_simple(&self) -> bool {
        self.variants.iter().all(|v| v.fields.is_empty())
    }
}

#[derive(Debug, Clone)]
pub struct FuncSig {
    pub name: String,
    pub param_tys: Vec<HType>,
    pub param_names: Vec<String>,
    pub ret: HType,
    /// If true, this is an `extern` declaration; do not emit a body, and call by `c_name`.
    pub is_extern: bool,
    pub c_name: String,
    /// If part of a logic block, the logic's name (for namespacing/mangling).
    pub logic: Option<String>,
    /// Generic type parameters declared on this function (`<T, U>`).
    pub type_params: Vec<String>,
    /// True if declared with the `inline` modifier.
    pub is_inline: bool,
    /// True if declared with the `wall` modifier — calls into this function are wall crossings,
    /// so `transfer`/`share` are permitted at the direct call site.
    pub is_gate: bool,
    /// True for variadic externs (`extern int printf(*char fmt, ...);`).  The arity check at
    /// call sites is relaxed to `args.len() >= param_tys.len()`.
    pub is_variadic: bool,
    /// `pub` visibility — if false, callers outside `module_path` get a sema error.
    pub is_pub: bool,
    /// Dotted module path the function was declared in.  Empty for the root module.
    pub module_path: Vec<String>,
    /// `import` declarations visible at this function's source file.  When the function
    /// calls a name from a different module, the (callee.module, callee.name) pair must
    /// appear in this list — otherwise the call is rejected with "not imported".
    pub imports: Vec<(Vec<String>, String)>,
    /// `use Mod.Type.Attr;` declarations active at this function's file — used to
    /// authorize cross-module `has` impls during bound checks at instantiation.
    pub has_imports: Vec<maka_ast::HasImport>,
    /// Resolved `where` clauses: `(trait_name, type_args)`.  Type args are HTypes that
    /// may contain `TyVar`s (substituted at instantiation time).  Each clause asserts
    /// that the concrete substitute of the first TyVar must implement the named trait.
    /// Where-bound: (trait name, args, optional assoc-type bindings).
    /// `assoc_bindings` carries `<T: Foo<Slot = i64>>` constraints; at each
    /// instantiation, the picked impl's `type Slot = R` must type_eq the
    /// bound's value (after substitution).
    pub where_bounds: Vec<(String, Vec<HType>, Vec<(String, HType)>)>,
}

#[derive(Debug, Clone)]
pub struct HFunc {
    pub id: FuncId,
    pub name: String,
    pub params: Vec<LocalId>,
    pub ret: HType,
    pub locals: Vec<LocalInfo>,
    pub body: HBlock,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HBinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
    BitAnd, BitOr, BitXor, Shl, Shr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HUnOp { Neg, Not, BitNot }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HAssignOp { Assign, Add, Sub, Mul, Div, Mod }

#[derive(Debug, Clone, PartialEq)]
pub enum CastKind {
    /// Numeric: int→int, int→float, float→int, float→float
    Numeric,
    /// Same-width sign change (bit-preserving).
    SignChange,
    /// Enum → underlying int.
    EnumToInt,
    /// Char ↔ int.
    CharIntInt,
    /// §3 `int → Enum` — runtime bounds-check against the variant set
    /// (same shape as `arr[i]`); panics on out-of-range and returns the
    /// `Enum` value.  Result HType is the `Enum`, never wrapped in a
    /// pointer.
    IntToEnumChecked,
    /// §6.6 `*int → *Enum` — runtime peek-and-tag-check at the pointee.
    /// On in-range: returns the **same** pointer cast to `*Enum`.  On
    /// out-of-range: returns `null`.  Maka's "pointer is the nullable
    /// carrier" convention — failure rides in the result type, no panic.
    IntPtrToEnumPtrChecked,
    /// implicit (used by codegen when source==target type)
    Identity,
    /// `&T as dyn Trait` / `&mut T as dyn Trait` / `T as dyn Trait`: produces a fat pointer.
    ToDyn { trait_name: String, struct_id: StructId },
    /// Reinterpret cast: `*T` ↔ `*U`, `*T` ↔ `usize`/`isize`. The user takes responsibility.
    /// For `*T ↔ *U` the lifetime pass still propagates dep edges (both sides alias the
    /// same memory).  For `*T → usize → *T` round-trips the integer step breaks the chain,
    /// because dep edges only flow through ref-like Locals.  Codegen emits a plain C cast.
    Reinterpret,
}

#[derive(Debug, Clone)]
pub struct HExpr {
    pub kind: HExprKind,
    pub ty: HType,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HExprKind {
    LitInt(i64),
    LitFloat(f64),
    LitBool(bool),
    LitChar(char),
    LitStr(String),
    LitNull,
    LitUnit,
    /// A typed all-zero value (`(T){0}`): NULL pointers, empty owning composites
    /// (a String/Vec/struct with null buffers).  Used to initialize the synthetic
    /// `__yield` result local of a value-producing match arm so a `yield` nested in
    /// an `if`/`while` can assign it.  Dropping a ZeroInit owning value is a no-op
    /// (it frees only null buffers), so the first `yield`'s free-on-reassign is
    /// safe.  A leaf with no sub-expressions and no moves.
    ZeroInit,
    Local(LocalId),
    /// reference to a top-level enum variant value
    EnumVariant(EnumId, usize),
    Field { base: Box<HExpr>, field: usize },
    Index { base: Box<HExpr>, idx: Box<HExpr> },
    Call { callee: FuncId, args: Vec<HExpr> },
    Bin { op: HBinOp, lhs: Box<HExpr>, rhs: Box<HExpr> },
    Un { op: HUnOp, expr: Box<HExpr> },
    /// `expr!` — unwrap a pointer; `skip_check=true` when narrowed.
    Unwrap { expr: Box<HExpr>, skip_check: bool },
    /// `&expr` or `&mut expr` — produces &T or *T (context decides; here we represent as Ref)
    AddrOfRef { mutable: bool, place: Box<HExpr> },
    /// `expr as T`
    Cast { expr: Box<HExpr>, kind: CastKind, to: HType },
    /// `expr as? T` — produces `*T` with null on failure
    CheckedCast { expr: Box<HExpr>, kind: CastKind, to: HType },
    /// data-struct literal
    Struct { id: StructId, fields: Vec<(usize, HExpr)> },
    /// array literal
    ArrayLit(Vec<HExpr>),
    /// Null-check `p != null` or `p == null`. Used to inform narrowing; stays semantically `!=`/`==`.
    /// (We just emit Bin Eq/Ne — narrowing is recorded on subsequent unwraps inside the branch.)

    /// implicit coerce ptr->const-ptr or refmut->refconst
    DropWrite(Box<HExpr>),
    /// Conversion: an array (`[N]T`) to a slice (`[]T` / `[]mut T`).
    ArrayToSlice { base: Box<HExpr>, len: i64 },
    /// Implicit read-dereference of a `&T` (or `&mut T`) reference to produce the referent value.
    /// Used in numeric and equality contexts where a reference appears as a value.
    DerefRef(Box<HExpr>),
    /// `heap value` — emit `malloc(sizeof(T)); *p = value; p` and produce a fresh heap LID.
    HeapAlloc(Box<HExpr>),
    /// `free value` — bare-word deallocator for `raw *T` (only inside `unsafe { }`).
    /// Codegen lowers to `free((void*)(value))`.
    Free(Box<HExpr>),
    /// Function name used in expression position (function pointer value).
    FnRef(FuncId),
    /// Indirect call through a function pointer value.
    CallIndirect { callee: Box<HExpr>, args: Vec<HExpr> },
    /// Inline-function expansion. The callee is the *template* function id.
    /// At codegen, this is emitted as a statement-expression that splices the body.
    /// `propagate_drops` lists the CALLER's owning locals live (and not moved) at
    /// this call site - the lifetime pass fills it.  A `propagate` inside the inline
    /// body early-returns the caller's C frame, so codegen must free these before
    /// that return or they leak (the inline's own scope-exit drops only cover the
    /// inline's frame, not the caller's).  `loop_jump_drops` is the analogue for a
    /// `break`/`continue` spliced from the inline that targets the CALLER's enclosing
    /// loop: the caller's loop-body owning locals live at this call site, freed
    /// before the jump.
    InlineCall { callee: FuncId, args: Vec<HExpr>, propagate_drops: Vec<LocalId>, loop_jump_drops: Vec<LocalId> },
    /// Closure value: produces a Callable_KEY with the lifted fn pointer + an env struct.
    /// `env_struct` is the StructId of the synthetic env type; `env_values` lists the field-initializers
    /// in declaration order.  `capture_lids` parallels `env_values` and lists the `LocalId`
    /// each capture is bound to *inside* the lifted body — used by the lifetime pass to
    /// forward capture-site non-null facts into the closure's initial state (§6.3).
    Closure { lifted: FuncId, env_struct: StructId, env_values: Vec<HExpr>, capture_lids: Vec<LocalId> },
    /// `transfer x` at a call-site argument — yields the source's value AND marks the source as moved.
    /// The lifetime pass walks this kind and rejects any further use of the moved binding.
    Transfer(Box<HExpr>),
    /// Length of a slice/vector value (used by for-each desugar).
    SliceLen(Box<HExpr>),
    /// Discriminant tag of an enum value: `e.tag` -> int.  For simple enums
    /// the value already is its tag, so this is identity; for tagged enums
    /// codegen reads the `.tag` C struct field.
    EnumTag(Box<HExpr>),
    /// Read or write a module-scope global by id.  Codegen emits the global's
    /// C name verbatim; mutability is gated by `GlobalInfo.is_mut`.
    GlobalRef(GlobalId),
    /// Tagged-enum variant constructor: `Enum.Variant{...}`.
    VariantCtor { enum_id: EnumId, variant: usize, fields: Vec<(usize, HExpr)> },
    /// Match expression. Each arm has an optional variant tag (None = else/literal/wildcard),
    /// per-field local bindings, an optional guard, and a body (which yields a value).
    Match {
        scrutinee: Box<HExpr>,
        arms: Vec<HMatchArm>,
        /// Yielded type (Unit for statement-form).
        result_ty: HType,
    },
}

#[derive(Debug, Clone)]
pub enum HArmKind {
    /// Match a specific variant tag (with optional binding LocalIds for each variant field).
    Variant { enum_id: EnumId, variant: usize, bindings: Vec<Option<LocalId>>, lit_checks: Vec<Option<HExpr>> },
    /// Match a primitive literal.
    Lit(HExpr),
    /// Match a null pointer.
    Null,
    /// Catch-all.
    Else,
}

#[derive(Debug, Clone)]
pub struct HMatchArm {
    pub kind: HArmKind,
    pub guard: Option<HExpr>,
    pub body: HBlock,
    /// For expression-form match, a final value expression; None if statement-form.
    pub value: Option<HExpr>,
    /// If non-None, bind the scrutinee value to this local before evaluating guard/body.
    pub scrut_binding: Option<LocalId>,
}

#[derive(Debug, Clone)]
pub struct HBlock {
    pub stmts: Vec<HStmt>,
    /// Heap locals that need freeing at this block's exit, in **reverse declaration order**.
    pub heap_to_free: Vec<LocalId>,
    /// Pointers whose deps became empty due to LIDs dying at this block's exit; set to NULL.
    pub ptr_nulls: Vec<LocalId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HStmt {
    /// New local; `init` may be a `HExpr` of the declared type.
    Let { local: LocalId, init: HExpr, span: Span },
    /// Assignment to a place.  `drop_old` = free the previous owning value before
    /// the store (SPEC 6.2).  The lifetime pass clears it when the place's owner
    /// was already MOVED out before this reassignment, so codegen does not
    /// double-free the moved-out value.  Defaults to true.
    Assign { op: HAssignOp, place: HExpr, value: HExpr, drop_old: bool, span: Span },
    ExprStmt(HExpr),
    Return { value: Option<HExpr>, /* heap locals to free in *this* scope chain before return */ heap_drops: Vec<LocalId>, span: Span },
    If { cond: HExpr, then_b: HBlock, else_b: Option<HBlock>, span: Span },
    While { cond: HExpr, body: HBlock, span: Span },
    Block(HBlock),
    Unsafe(HBlock, Span),
    /// `heap_drops`: owning locals declared inside the enclosing loop that must
    /// be freed before the jump (break/continue leaves the loop-body scope chain
    /// without running its scope-exit drops, same as `return`).
    Break { heap_drops: Vec<LocalId>, span: Span },
    Continue { heap_drops: Vec<LocalId>, span: Span },
    /// Native C `for (init; cond; step) body` — for-range lowers to this.
    ForC { init: Box<HStmt>, cond: HExpr, step: Box<HStmt>, body: HBlock, span: Span },
    /// `for (T var in src) body` over a slice/array/vec value.
    /// Codegen handles the type-specific length and element access.
    ForEach { var: LocalId, src: HExpr, body: HBlock, span: Span },
    /// `propagate X;` — only valid in `inline` function bodies. At an inline-expansion site
    /// codegens to a real C `return X;` of the *outer* function (exits caller's frame).
    /// `propagate [expr];` — `value` is `None` when the surrounding inline (and
    /// its non-inline caller) returns `unit`.
    Propagate { value: Option<HExpr>, span: Span },
}

/// Global symbol table assembled before per-function type checking.
#[derive(Debug, Clone)]
pub struct LogicInfo {
    pub name: String,
    /// FuncIds that belong to this logic, in declaration order.
    pub funcs: Vec<FuncId>,
    pub span: Span,
}

/// Registered `attr Name { sigs }` declaration.  Stores each method's expected
/// shape (param types and return, in placeholder form using `_` for the impl
/// type) and the optional default body the `has` impl inherits when it doesn't
/// override.  Used for contract-matching + default-body synthesis.
#[derive(Debug, Clone)]
pub struct AttrInfo {
    pub name: String,
    /// Generic type parameters: `attr Convert<U> { ... }` → `["U"]`.
    pub type_params: Vec<String>,
    /// Optional default per type parameter, parallel to `type_params`:
    /// `attr Add<R = _> { ... }`.  Kept as raw AST so `_` (Self) resolves
    /// against each `has` impl's receiver; a `has` impl that omits a trailing
    /// attr type-argument inherits the default.
    pub type_param_defaults: Vec<Option<maka_ast::Type>>,
    pub is_pub: bool,
    pub module_path: Vec<String>,
    pub span: Span,
    pub methods: Vec<AttrMethod>,
    /// Associated-type declarations: `type Slot;` (no default) or
    /// `type Slot = DefaultT;` (with default) lines (§10.5).  Tuples are
    /// (name, optional default HType, span).  Impls without a definition
    /// inherit the default if present, else it's a missing-impl error.
    pub assoc_type_decls: Vec<(String, Option<HType>, Span)>,
}

/// One method signature declared inside an `attr` block.  Param/ret types are
/// stored in raw AST form (with `_` placeholder occurrences intact) so the
/// resolver can substitute the impl type and re-resolve per `has` block.
#[derive(Debug, Clone)]
pub struct AttrMethod {
    pub name: String,
    pub decl: maka_ast::FuncDecl,
    /// `true` when the attr provides a default body (non-empty block).  Missing
    /// methods in a `has` block inherit this default; methods without defaults
    /// must be implemented or it's an error.
    pub has_default: bool,
}

/// One `Type has Attr { ... }` impl record — tracked separately from sigs so
/// bound checks can filter by visibility (file-private vs `pub`) and by explicit
/// `use Mod.Type.Attr;` imports.
#[derive(Debug, Clone)]
pub struct HasImpl {
    pub attr_name: String,
    pub type_key: String,
    /// Concrete attr-args for a generic attr (`Color has Convert<int>` →
    /// `["int"]`).  Empty for non-generic attrs.  Multi-arg `where T has
    /// Attr<U>` bounds compare both the receiver type AND these args.
    pub attr_args: Vec<HType>,
    pub is_pub: bool,
    pub module_path: Vec<String>,
    /// FuncIds of the methods registered for this impl.  Used by the typecheck
    /// pass to find the bodies for `has`-block methods (including defaults
    /// synthesized from the `attr`).
    pub func_ids: Vec<FuncId>,
    /// Associated-type definitions: `type Slot = ConcreteType;` resolved to
    /// HType at registration.  Free type variables from the impl's receiver
    /// pattern (e.g. `T` in `*T has Foo { type Slot = *T; }`) appear here as
    /// `HType::TyVar(name)` and are substituted at the call site using the
    /// unification env from receiver matching.  Order matches `AttrInfo.assoc_type_decls`.
    pub assoc_type_defs: Vec<(String, HType)>,
    /// Type variables introduced by the receiver pattern (e.g. `["T"]` for
    /// `*T has Foo`).  Used by `assoc_type_defs` substitution and by the
    /// receiver-unification machinery in `typeck`.
    pub receiver_tyvars: Vec<String>,
    /// Receiver pattern resolved to HType (with TyVars at parametric positions).
    /// For concrete receivers this is e.g. `HType::Struct(id)`; for `*T` it's
    /// `HType::Ptr { mutable: true, inner: TyVar("T") }`; for `int` it's `HType::Int`.
    pub receiver_pattern: HType,
}

#[derive(Debug, Clone, Default)]
pub struct SymTab {
    pub structs: Vec<StructInfo>,
    pub enums: Vec<EnumInfo>,
    pub sigs: Vec<FuncSig>,
    pub funcs: Vec<HFunc>,
    pub logics: Vec<LogicInfo>,
    /// `attr Name { ... }` declarations, indexed by name for `<T: Attr>` /
    /// `where T has Attr` validity checks.
    pub attrs: Vec<AttrInfo>,
    /// For each FuncId, optionally the source AST (kept for generics and instantiations).
    pub ast_funcs: std::collections::HashMap<u32, maka_ast::FuncDecl>,
    /// Instantiations: (template_fid, concrete arg-type keys) → concrete FuncId.
    pub instantiations: std::collections::HashMap<(u32, String), u32>,
    /// Struct instantiations: (template_name, arg-key) → concrete StructId.
    pub struct_instantiations: std::collections::HashMap<(String, String), u32>,
    /// Enum instantiations: (template_name, arg-key) → concrete EnumId.  Generic
    /// enums (`enum Option<T> { Some { T value }, None }`) are monomorphized just
    /// like generic structs — each concrete `Option<int>`, `Option<string>` etc.
    /// gets its own EnumInfo with substituted variant payloads.
    pub enum_instantiations: std::collections::HashMap<(String, String), u32>,
    /// Trait impl table: trait name → set of HType keys that have an entry in the
    /// `logic Trait { method(Self, ...) }` block.  Each method's first parameter's
    /// underlying struct counts as one implementation of the trait.
    pub trait_impls: std::collections::HashMap<String, std::collections::HashSet<String>>,
    /// Full `Type has Attr` impl records with visibility metadata.  The flat
    /// `trait_impls` map above is a fast-path index; this list is what bound
    /// checks scan when filtering by `pub`/`use` imports.
    pub has_impls: Vec<HasImpl>,
    /// Top-level `pub? constexpr int NAME = ...;` declarations, registered as
    /// importable named-int constants.  In-file uses still resolve via the
    /// parser's fold map; cross-module references in expression position go
    /// through `find_constexpr` here (with the usual pub + import check).
    pub constexprs: Vec<ConstexprInfo>,
    /// Module-scope `pub? mut Type NAME = <literal>;` declarations.
    /// Reads / writes refer to these by `GlobalId` (index into the Vec).
    pub globals: Vec<GlobalInfo>,
    /// `Rust<T>` type names that must satisfy `Send` because they're
    /// transferred or spawned across threads.  Collected by sema and
    /// emitted as `assert_send::<T>()` into the sidecar by the rust-bridge.
    pub send_probes: Vec<String>,
    /// `Rust<T>` type names that must satisfy `Sync` because they're
    /// `share`d across a gate.  Emitted as `assert_sync::<T>()`.
    pub sync_probes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ConstexprInfo {
    pub name: String,
    pub value: i64,
    pub is_pub: bool,
    pub module_path: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct GlobalId(pub u32);

#[derive(Debug, Clone)]
pub struct GlobalInfo {
    pub name: String,
    pub c_name: String,
    pub ty: HType,
    /// The user-supplied initializer expression, captured as HExpr.
    /// Codegen emits its textual form as the C static initializer; the C
    /// compiler enforces "must be a constant expression."
    pub init: HExpr,
    pub is_mut: bool,
    pub is_pub: bool,
    pub module_path: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirModule {
    pub sym: SymTab,
    /// Non-fatal diagnostics surfaced during analysis (e.g. flow-sensitive
    /// auto-null pointer-use warnings). The driver prints these but compilation
    /// proceeds normally.
    pub warnings: Vec<crate::SemaWarning>,
    /// Header names from `cinclude "name.h";` directives.  Codegen emits one
    /// `#include <name.h>` per entry in the prologue.
    pub cincludes: Vec<String>,
    /// Raw C source pasted verbatim into the generated C at module scope,
    /// in source order, from `cblock "...";` directives.
    pub cblocks: Vec<String>,
}

impl SymTab {
    pub fn struct_by_name(&self, n: &str) -> Option<(StructId, &StructInfo)> {
        self.structs.iter().enumerate().find(|(_, s)| s.name == n).map(|(i, s)| (StructId(i as u32), s))
    }
    pub fn enum_by_name(&self, n: &str) -> Option<(EnumId, &EnumInfo)> {
        self.enums.iter().enumerate().find(|(_, e)| e.name == n).map(|(i, e)| (EnumId(i as u32), e))
    }
    pub fn func_by_name(&self, n: &str) -> Option<(FuncId, &FuncSig)> {
        // Match by logic-prefix-free name first (top-level), else by mangled name.
        self.sigs.iter().enumerate()
            .find(|(_, s)| s.logic.is_none() && s.name == n)
            .map(|(i, s)| (FuncId(i as u32), s))
    }
    pub fn func_by_qualified(&self, logic: &str, name: &str) -> Option<(FuncId, &FuncSig)> {
        self.sigs.iter().enumerate()
            .find(|(_, s)| s.logic.as_deref() == Some(logic) && s.name == name)
            .map(|(i, s)| (FuncId(i as u32), s))
    }
    /// All overload candidates for a name (top-level or logic-qualified).
    pub fn funcs_by_qualified(&self, logic: Option<&str>, name: &str) -> Vec<(FuncId, &FuncSig)> {
        self.sigs.iter().enumerate()
            .filter(|(_, s)| s.logic.as_deref() == logic && s.name == name)
            .map(|(i, s)| (FuncId(i as u32), s))
            .collect()
    }
    pub fn logic_by_name(&self, n: &str) -> Option<&LogicInfo> {
        self.logics.iter().find(|l| l.name == n)
    }
    pub fn attr_by_name(&self, n: &str) -> Option<&AttrInfo> {
        self.attrs.iter().find(|a| a.name == n)
    }
    pub fn struct_info(&self, id: StructId) -> &StructInfo { &self.structs[id.0 as usize] }
    pub fn enum_info(&self, id: EnumId) -> &EnumInfo { &self.enums[id.0 as usize] }
    pub fn func_sig(&self, id: FuncId) -> &FuncSig { &self.sigs[id.0 as usize] }
}
