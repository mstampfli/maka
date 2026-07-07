//! Maka AST (spec v1.2).

use maka_lexer::Span;

// ---------- Types ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutness {
    Default,
    Const,
    Mut,
}

#[derive(Debug, Clone)]
pub enum Type {
    /// Named type (`int`, `float`, `bool`, `char`, `unit`, `MyStruct`, ...)
    Named(String, Span),
    /// `T::Slot` — associated-type path.  `base` is a Type (typically a
    /// Named type variable referencing the enclosing generic parameter or
    /// concrete receiver), `segment` is the associated-type name declared
    /// by an attr the base is bound by.  Resolved at monomorphization.
    AssocPath { base: Box<Type>, segment: String, span: Span },
    /// `&Mut? T`
    Ref { mutness: Mutness, inner: Box<Type>, span: Span },
    /// `*Mut? T`
    Ptr { mutness: Mutness, inner: Box<Type>, span: Span },
    /// `raw *Mut? T` — pointer of unknown provenance (e.g. C library origin).
    /// Storage and copy are safe; deref / field / index / narrowing require `unsafe { }`.
    RawPtr { mutness: Mutness, inner: Box<Type>, span: Span },
    /// `own *Mut? T` — nullable owning pointer.  Auto-freed at scope/drop or when
    /// reassigned (the old value is freed before the new one is stored).  Coerces
    /// to `*T` (downgrade), takes `null`.
    OwnPtr { mutness: Mutness, inner: Box<Type>, span: Span },
    /// `heap T` — *storage modifier*, allowed only on binding/param/return sites.
    Heap { inner: Box<Type>, span: Span },
    /// `[N]T`
    Array { len: i64, elem: Box<Type>, span: Span },
    /// `[]T` (read-only slice) or `[]mut T` (writable slice)
    Slice { mutness: Mutness, elem: Box<Type>, span: Span },
    /// `[*]T` — vector payload (only valid inside `heap [*]T`).
    Vec { elem: Box<Type>, span: Span },
    /// `()` — the unit type literal in source.
    Unit(Span),
    /// `dyn Trait` or `dyn (T1 + T2)`, and its locked sibling `some Trait`.
    /// `dyn` (`locked: false`) is a per-value existential (elements can differ);
    /// `some` (`locked: true`) is a per-collection existential locked to ONE hidden
    /// concrete type.  Both carry only trait names; type args ignored for v1.
    Dyn { traits: Vec<String>, locked: bool, span: Span },
    /// Generic type instantiation `Name<T, U>`.
    Generic { name: String, args: Vec<Type>, span: Span },
    /// Function pointer type: `RetType(P1, P2, ...)`.
    FnPtr { ret: Box<Type>, params: Vec<Type>, span: Span },
}

impl Type {
    /// Substitute the `_` placeholder type with `concrete`. Used by `has`-block
    /// processing in sema: the user writes `_ self` in attr/has signatures and the
    /// resolver rewrites those occurrences to the implementing type.
    pub fn subst_placeholder(&self, concrete: &str) -> Type {
        match self {
            Type::Named(n, sp) if n == "_" => Type::Named(concrete.to_string(), *sp),
            Type::Named(_, _) | Type::Unit(_) => self.clone(),
            Type::AssocPath { base, segment, span } => Type::AssocPath {
                base: Box::new(base.subst_placeholder(concrete)),
                segment: segment.clone(),
                span: *span,
            },
            Type::Ref { mutness, inner, span } => Type::Ref {
                mutness: *mutness, inner: Box::new(inner.subst_placeholder(concrete)), span: *span,
            },
            Type::Ptr { mutness, inner, span } => Type::Ptr {
                mutness: *mutness, inner: Box::new(inner.subst_placeholder(concrete)), span: *span,
            },
            Type::RawPtr { mutness, inner, span } => Type::RawPtr {
                mutness: *mutness, inner: Box::new(inner.subst_placeholder(concrete)), span: *span,
            },
            Type::OwnPtr { mutness, inner, span } => Type::OwnPtr {
                mutness: *mutness, inner: Box::new(inner.subst_placeholder(concrete)), span: *span,
            },
            Type::Heap { inner, span } => Type::Heap {
                inner: Box::new(inner.subst_placeholder(concrete)), span: *span,
            },
            Type::Array { len, elem, span } => Type::Array {
                len: *len, elem: Box::new(elem.subst_placeholder(concrete)), span: *span,
            },
            Type::Slice { mutness, elem, span } => Type::Slice {
                mutness: *mutness, elem: Box::new(elem.subst_placeholder(concrete)), span: *span,
            },
            Type::Vec { elem, span } => Type::Vec {
                elem: Box::new(elem.subst_placeholder(concrete)), span: *span,
            },
            Type::Dyn { traits, locked, span } => Type::Dyn { traits: traits.clone(), locked: *locked, span: *span },
            Type::Generic { name, args, span } => Type::Generic {
                name: name.clone(),
                args: args.iter().map(|a| a.subst_placeholder(concrete)).collect(),
                span: *span,
            },
            Type::FnPtr { ret, params, span } => Type::FnPtr {
                ret: Box::new(ret.subst_placeholder(concrete)),
                params: params.iter().map(|p| p.subst_placeholder(concrete)).collect(),
                span: *span,
            },
        }
    }

    /// Type-tree-aware placeholder substitution: replace `_` (and `Self`) with
    /// the receiver's full Type tree.  Used by parametric `has` impls where the
    /// receiver pattern is e.g. `*T` and method signatures spelled with `&_ self`
    /// must expand to `&*T self`.
    pub fn subst_placeholder_ty(&self, recv: &Type) -> Type {
        match self {
            Type::Named(n, _) if n == "_" => recv.clone(),
            Type::Named(_, _) | Type::Unit(_) => self.clone(),
            Type::AssocPath { base, segment, span } => Type::AssocPath {
                base: Box::new(base.subst_placeholder_ty(recv)),
                segment: segment.clone(),
                span: *span,
            },
            Type::Ref { mutness, inner, span } => Type::Ref {
                mutness: *mutness, inner: Box::new(inner.subst_placeholder_ty(recv)), span: *span,
            },
            Type::Ptr { mutness, inner, span } => Type::Ptr {
                mutness: *mutness, inner: Box::new(inner.subst_placeholder_ty(recv)), span: *span,
            },
            Type::RawPtr { mutness, inner, span } => Type::RawPtr {
                mutness: *mutness, inner: Box::new(inner.subst_placeholder_ty(recv)), span: *span,
            },
            Type::OwnPtr { mutness, inner, span } => Type::OwnPtr {
                mutness: *mutness, inner: Box::new(inner.subst_placeholder_ty(recv)), span: *span,
            },
            Type::Heap { inner, span } => Type::Heap {
                inner: Box::new(inner.subst_placeholder_ty(recv)), span: *span,
            },
            Type::Array { len, elem, span } => Type::Array {
                len: *len, elem: Box::new(elem.subst_placeholder_ty(recv)), span: *span,
            },
            Type::Slice { mutness, elem, span } => Type::Slice {
                mutness: *mutness, elem: Box::new(elem.subst_placeholder_ty(recv)), span: *span,
            },
            Type::Vec { elem, span } => Type::Vec {
                elem: Box::new(elem.subst_placeholder_ty(recv)), span: *span,
            },
            Type::Dyn { traits, locked, span } => Type::Dyn { traits: traits.clone(), locked: *locked, span: *span },
            Type::Generic { name, args, span } => Type::Generic {
                name: name.clone(),
                args: args.iter().map(|a| a.subst_placeholder_ty(recv)).collect(),
                span: *span,
            },
            Type::FnPtr { ret, params, span } => Type::FnPtr {
                ret: Box::new(ret.subst_placeholder_ty(recv)),
                params: params.iter().map(|p| p.subst_placeholder_ty(recv)).collect(),
                span: *span,
            },
        }
    }
}

/// One associated-type declaration inside an `attr` block: `type Name;`
/// (signature-only) — the impl must provide a `type Name = ConcreteType;`.
/// With an optional `default`, the impl may omit the definition and the
/// default is used instead.
#[derive(Debug, Clone)]
pub struct AssocTypeDecl {
    pub name: String,
    pub default: Option<Type>,
    pub span: Span,
}

/// One associated-type definition inside a `has` impl: `type Name = T;`
#[derive(Debug, Clone)]
pub struct AssocTypeDef {
    pub name: String,
    pub value: Type,
    pub span: Span,
}

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Named(_, s) | Type::Unit(s) => *s,
            Type::AssocPath { span, .. } => *span,
            Type::Ref { span, .. }
            | Type::Ptr { span, .. }
            | Type::RawPtr { span, .. }
            | Type::OwnPtr { span, .. }
            | Type::Heap { span, .. }
            | Type::Array { span, .. }
            | Type::Slice { span, .. }
            | Type::Vec { span, .. }
            | Type::Dyn { span, .. }
            | Type::Generic { span, .. }
            | Type::FnPtr { span, .. } => *span,
        }
    }
}

// ---------- Expressions ----------

#[derive(Debug, Clone)]
pub enum Lit {
    Int(i64),
    Float(f64),
    Bool(bool),
    Char(char),
    Str(String),
    Null,
    Unit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
    BitAnd, BitOr, BitXor, Shl, Shr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Lit(Lit, Span),
    Ident(String, Span),
    Bin { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr>, span: Span },
    Un { op: UnOp, expr: Box<Expr>, span: Span },
    /// postfix unwrap `p!`
    Unwrap { expr: Box<Expr>, span: Span },
    /// borrow expressions
    Ref { mutness: Mutness, expr: Box<Expr>, span: Span },
    Field { base: Box<Expr>, name: String, span: Span },
    Index { base: Box<Expr>, idx: Box<Expr>, span: Span },
    /// `callee(args)`, optionally with explicit generic type arguments written
    /// turbofish-style: `callee::<T, U>(args)`.  `type_args` is empty for an
    /// ordinary call (type parameters are then inferred from the arguments and
    /// the expected return type).
    Call { callee: Box<Expr>, args: Vec<Expr>, type_args: Vec<Type>, span: Span },
    /// `expr as T`
    Cast { expr: Box<Expr>, ty: Type, span: Span },
    /// `expr as? T`
    CheckedCast { expr: Box<Expr>, ty: Type, span: Span },
    /// struct literal: `T { f = e, g = e }` or `{ f = e }` when type inferred from LHS
    Struct { ty: Option<String>, fields: Vec<(String, Expr)>, span: Span },
    /// array literal `[e, e, ...]`
    ArrayLit { elems: Vec<Expr>, span: Span },
    /// `heap value` — allocate `value` on the heap and produce a `*T` pointer.
    HeapAlloc { value: Box<Expr>, span: Span },
    /// `free value` — deallocate a `raw *T`.  Sema requires the arg to be
    /// `raw *T` AND the call site to be inside an `unsafe { ... }` block.
    /// `deep` (`free deep value`) additionally runs the recursive drop glue on
    /// the pointer's target first, reclaiming an owned graph parked behind a raw
    /// pointer (the FFI-singleton teardown case) instead of leaking it.
    Free { value: Box<Expr>, deep: bool, span: Span },
    /// `Enum.Variant { f = e, ... }` — construct a tagged-enum variant value.
    /// Payload-less variants use `Field` (kept for back-compat with C-style enums).
    VariantCtor { enum_name: String, variant: String, fields: Vec<(String, Expr)>, span: Span },
    /// `match (expr) { arms }` as an expression.
    Match { scrutinee: Box<Expr>, arms: Vec<MatchArm>, span: Span },
    /// `RetType(params) [caps] body` — lambda expression.
    Lambda {
        ret: Type,
        params: Vec<Param>,
        captures: Vec<LambdaCapture>,
        body: LambdaBody,
        span: Span,
    },
    /// `transfer expr` / `share expr` — D11 wall-crossing modifiers at call sites.
    WallMod { mode: WallMode, expr: Box<Expr>, span: Span },
    /// `Attr::method(args)` / `receiver.Attr::method(args)` — attr-qualified call.
    /// Distinct from `Expr::Call { Field { Ident(Attr), method }, args }` because
    /// the `::` form must bypass local-shadowing (so `Stored::run(&Stored)` still
    /// reaches attr `Stored` even when a local named `Stored` is in scope).
    /// `receiver = Some(_)` for the postfix form `r.Attr::method()`.
    AttrCall { attr: String, name: String, receiver: Option<Box<Expr>>, args: Vec<Expr>, span: Span },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallMode { Transfer, Share }

#[derive(Debug, Clone)]
pub struct LambdaCapture {
    pub name: String,
    /// 'v' = by-value, 'r' = by-ref, 'm' = by-mut-ref.
    pub mode: char,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum LambdaBody {
    Expr(Box<Expr>),
    Block(Block),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Lit(_, s)
            | Expr::Ident(_, s)
            | Expr::Bin { span: s, .. }
            | Expr::Un { span: s, .. }
            | Expr::Unwrap { span: s, .. }
            | Expr::Ref { span: s, .. }
            | Expr::Field { span: s, .. }
            | Expr::Index { span: s, .. }
            | Expr::Call { span: s, .. }
            | Expr::AttrCall { span: s, .. }
            | Expr::Cast { span: s, .. }
            | Expr::CheckedCast { span: s, .. }
            | Expr::Struct { span: s, .. }
            | Expr::ArrayLit { span: s, .. }
            | Expr::HeapAlloc { span: s, .. }
            | Expr::Free { span: s, .. }
            | Expr::VariantCtor { span: s, .. }
            | Expr::Match { span: s, .. }
            | Expr::Lambda { span: s, .. }
            | Expr::WallMod { span: s, .. } => *s,
        }
    }
}

// ---------- Statements ----------

#[derive(Debug, Clone)]
pub enum Stmt {
    /// declaration: `[mut|const]? Type? name = expr;`  — but Maka requires a type.
    Let {
        mutness: Mutness,
        ty: Type,
        name: String,
        init: Expr,
        thread_local: bool,
        span: Span,
    },
    /// positional destructuring bind: `([mut] a, [mut] b, ...) = expr;` — binds
    /// each name to the corresponding field (by declaration order) of the struct
    /// `expr` evaluates to (an rblock tuple `__MakaTup`, or any `data`).  Each
    /// `bool` is that binding's `mut`ness.
    LetTuple {
        names: Vec<(bool, String)>,
        init: Expr,
        span: Span,
    },
    Assign {
        op: AssignOp,
        place: Expr,
        value: Expr,
        span: Span,
    },
    ExprStmt(Expr, Span),
    Return(Option<Expr>, Span),
    If {
        cond: Expr,
        then_block: Block,
        else_block: Option<Block>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Block,
        span: Span,
    },
    Block(Block),
    /// `unsafe { ... }`
    Unsafe(Block, Span),
    /// `match (scrutinee) { arm... }` as a statement.
    Match { scrutinee: Expr, arms: Vec<MatchArm>, span: Span },
    /// `yield expr;` inside expression-form blocks (match arms, if blocks, while bodies).
    Yield(Expr, Span),
    /// `propagate expr;` — only valid inside an `inline` function; returns the value from the caller's frame.
    /// `propagate [expr];` — only legal inside an `inline` function.  When the
    /// caller's return type is `unit`, the expression is omitted (`propagate;`).
    Propagate(Option<Expr>, Span),
    /// `for (T name in start..end) { body }`. Inclusive flag = true for `..=`.
    ForRange {
        var_ty: Type,
        var_name: String,
        start: Expr,
        end: Expr,
        inclusive: bool,
        body: Block,
        span: Span,
    },
    /// `for (T name in src) { body }` — iterate a slice/array/vector.
    ForEach {
        var_ty: Type,
        var_name: String,
        src: Expr,
        body: Block,
        span: Span,
    },
    /// `inline for (name in fields(value)) { body }` — a compile-time loop,
    /// unrolled once per field of `value`'s struct type.  Inside the body
    /// `name.name`, `name.value`, `name.index`, and `name.ty` refer to the
    /// current field (`ty` rather than `type`, which is a reserved keyword).
    /// Lowered by sema; never reaches codegen.
    InlineFor {
        var_name: String,
        iter: Expr,
        body: Block,
        span: Span,
    },
    Break(Span),
    Continue(Span),
}

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    /// Tag-only variant match (no destructure): `Eof`.
    Variant { enum_name: Option<String>, variant: String, span: Span },
    /// Variant with destructure: `IntLit{value}`.
    VariantDestructure {
        enum_name: Option<String>,
        variant: String,
        /// (field_name_in_decl, optional rename in pattern, optional literal-check)
        fields: Vec<PatField>,
        span: Span,
    },
    /// Literal: `0`, `'a'`, `true`, `false`.
    Lit(Lit, Span),
    /// `null`
    Null(Span),
    /// `else` — catch-all.
    Else(Span),
    /// Variable binding (used in guards and primitive matches): `x`.
    Ident(String, Span),
    /// Or-pattern: `A | B | C`.
    Or(Vec<Pattern>, Span),
}

#[derive(Debug, Clone)]
pub struct PatField {
    /// Field name in the variant declaration.
    pub field: String,
    /// Optional local binding name (defaults to `field`).
    pub binding: Option<String>,
    /// Optional literal value to check against.
    pub literal: Option<Lit>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ArmBody {
    Expr(Expr),       // single trailing expression
    Block(Block),     // {} block
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: ArmBody,
    pub span: Span,
}

// ---------- Top-level items ----------

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct WhereClause {
    pub trait_name: String,
    pub args: Vec<Type>,
    /// Optional assoc-type bindings (§10.5 "bounds on associated types"):
    /// `<T: Foo<Slot = i64>>` carries `[("Slot", i64)]` here.  Each pair
    /// requires the impl's `type Name = ...` definition (after
    /// substitution) to type_eq the bound's value at instantiation.
    pub assoc_type_bindings: Vec<(String, Type)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: String,
    pub type_params: Vec<String>,           // generic params, e.g. `<T,U>`
    pub params: Vec<Param>,
    pub ret: Type,
    pub body: Block,
    pub is_inline: bool,
    pub is_gate: bool,
    pub is_pub: bool,
    /// `export fn` - emit a stable, unmangled C symbol callable from C/Rust
    /// (implies external linkage; signature must be C-ABI-safe).  Reverse-FFI.
    pub is_export: bool,
    pub where_clauses: Vec<WhereClause>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub mutness: Mutness,
    pub ty: Type,
    pub name: String,
    pub default: Option<Expr>,
    pub is_embed: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct DataDecl {
    pub name: String,
    pub type_params: Vec<String>,
    pub fields: Vec<FieldDecl>,
    pub where_clauses: Vec<WhereClause>,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct VariantDecl {
    pub name: String,
    /// Fields for tagged variants; empty for payload-less variants.
    pub fields: Vec<FieldDecl>,
    /// Optional explicit tag value (only valid for payload-less variants without fields).
    pub explicit_value: Option<i64>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub type_params: Vec<String>,
    pub variants: Vec<VariantDecl>,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternDecl {
    pub name: String,
    /// The C symbol name; defaults to `name`.
    pub c_name: String,
    pub params: Vec<Param>,
    pub ret: Type,
    pub is_gate: bool,
    /// Trailing `...` in the C signature — for FFI to C variadic functions like `printf`.
    pub is_variadic: bool,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Item {
    Func(FuncDecl),
    Data(DataDecl),
    Enum(EnumDecl),
    Extern(ExternDecl),
    /// `attr Name { signatures }` — declares an attribute (trait) with method signatures
    /// and optional default bodies.  The signatures use `_` as a placeholder for the
    /// implementing type.
    Attr(AttrDecl),
    /// `Type has Attr { method bodies }` — provides an implementation of `Attr` for `Type`.
    Has(HasDecl),
    /// `cinclude "header.h";` — emit `#include <header.h>` in the C prologue.
    CInclude(String, Span),
    /// `cblock { ...raw C... }` — paste verbatim into the generated C at module scope.
    CBlock(String, Span),
    /// `clink "flag-or-file";` — a link input for the final C link step.  A value
    /// starting with `-l`/`-L` is a linker flag (library / search path); anything
    /// else is a `.a`/`.o`/`.c` file compiled/linked alongside the generated C.
    /// The in-source counterpart of the CLI `-l`/`-L`/`--link` args, and the C-side
    /// mirror of `rdep`, so a program that uses a C library is self-describing.
    CLink(String, Span),
    /// `pub? constexpr T NAME = expr;` — a named compile-time integer constant.
    /// Always available in the defining file via the parser's pre-scan fold map;
    /// also registered as a symbol so it can be imported and used by name in
    /// expression position from other modules.
    Constexpr(ConstexprDecl),
    /// `pub? mut Type NAME = expr;` — a module-scope mutable global.  Useful for
    /// process-wide state (frame counters, debug flags, RNG seeds, etc.) that
    /// would otherwise be threaded through every function signature.
    Global(GlobalDecl),
    /// `rblock "raw Rust source";` — inline Rust source compiled into a sidecar
    /// crate.  Each `pub fn` becomes a callable Maka function via auto-generated
    /// `extern "C"` shims.  See RUST_INTEROP.md.
    Rblock(String, Span),
    /// `rdep name = "version";` — a Cargo dependency line added to the sidecar
    /// crate's `Cargo.toml`.  Right-hand side is spliced verbatim (`"1.10"` or
    /// `"{ version = \"1\", features = [\"derive\"] }"`).
    Rdep(String, String, Span),
}

#[derive(Debug, Clone)]
pub struct GlobalDecl {
    pub name: String,
    pub ty: Type,
    pub init: Expr,
    pub is_mut: bool,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ConstexprDecl {
    pub name: String,
    pub value: i64,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct AttrDecl {
    pub name: String,
    /// Generic type parameters declared on the attr: `attr Convert<U> { ... }`.
    pub type_params: Vec<String>,
    /// Optional default for each type parameter, parallel to `type_params`:
    /// `attr Add<R = _> { ... }`.  `_` means the implementing type (Self).  A
    /// `has` impl that omits an attr type-argument inherits the default; a
    /// parameter with no default must be supplied explicitly.
    pub type_param_defaults: Vec<Option<Type>>,
    /// Method signatures inside the attr block (may have default bodies in `funcs`).
    pub funcs: Vec<FuncDecl>,
    /// Associated-type declarations: `type Slot;` lines (§10.5).
    pub assoc_types: Vec<AssocTypeDecl>,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HasDecl {
    /// Canonical-form string of the receiver pattern (e.g. "Color", "*T",
    /// "int").  Computed from `receiver` at parse time.  Kept for back-compat
    /// with code that keys impls by string (legacy nominal-receiver path).
    pub type_name: String,
    /// The receiver pattern as written in source.  For a parametric receiver
    /// (`*T has Foo`, `Box<T> has Foo`), this carries the full type tree
    /// including type variables.  For a concrete receiver, it's just the
    /// concrete `Type`.
    pub receiver: Type,
    /// The attribute name (e.g. "Drawable").
    pub attr_name: String,
    /// Concrete type arguments for a generic attr: `Color has Convert<int>`.
    pub attr_args: Vec<Type>,
    /// Method bodies — receiver type must match `receiver`.
    pub funcs: Vec<FuncDecl>,
    /// Associated-type definitions: `type Slot = ConcreteType;` lines (§10.5).
    pub assoc_type_defs: Vec<AssocTypeDef>,
    pub is_pub: bool,
    pub span: Span,
}

/// Canonical short string of a `has`-receiver pattern, used as the
/// back-compat `HasDecl.type_name` field.  For a concrete struct/enum it's
/// the bare name; for a parametric receiver it's a short shape sketch.
pub fn receiver_canonical_name(t: &Type) -> String {
    match t {
        Type::Named(n, _) => n.clone(),
        Type::AssocPath { base, segment, .. } => {
            format!("{}::{}", receiver_canonical_name(base), segment)
        }
        Type::Ptr { inner, mutness, .. } => {
            let pref = match mutness { Mutness::Const => "*const ", Mutness::Mut | Mutness::Default => "*" };
            format!("{}{}", pref, receiver_canonical_name(inner))
        }
        Type::Ref { inner, mutness, .. } => {
            let pref = match mutness { Mutness::Const | Mutness::Default => "&", Mutness::Mut => "&mut " };
            format!("{}{}", pref, receiver_canonical_name(inner))
        }
        Type::RawPtr { inner, mutness, .. } => {
            let pref = match mutness { Mutness::Const => "raw *const ", Mutness::Mut | Mutness::Default => "raw *" };
            format!("{}{}", pref, receiver_canonical_name(inner))
        }
        Type::Generic { name, args, .. } => {
            let inner: Vec<String> = args.iter().map(receiver_canonical_name).collect();
            format!("{}<{}>", name, inner.join(","))
        }
        _ => "<unsupported>".to_string(),
    }
}

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub items: Vec<Item>,
    /// Dotted module path declared via `module a.b.c;` at the top of the file.
    /// `None` when no declaration; treated as the implicit root module ``.
    pub module_path: Option<Vec<String>>,
    /// Per-item module path (filled in by the driver when merging multiple files).
    /// Empty initially; the i-th entry corresponds to `items[i]`.  When two items
    /// have different module paths, the resolver enforces `pub` between them.
    pub item_modules: Vec<Vec<String>>,
    /// Imports per file, parallel to items (i-th item's file's imports).  Each
    /// entry is a list of qualified imports `(module_path, name)` — `name` is
    /// the specific bound name visible at unqualified positions.
    pub item_imports: Vec<Vec<(Vec<String>, String)>>,
    /// Imports declared in this file (only used at parse time; the driver
    /// flattens these into `item_imports`).
    pub imports: Vec<ImportDecl>,
    /// `use ModPath.Type.Attr;` declarations — explicit propagation of a `pub has`
    /// implementation from another module into this file's bound-check scope.
    pub has_imports: Vec<HasImport>,
    /// Per-item has_imports, flattened by the driver across multi-file builds.
    pub item_has_imports: Vec<Vec<HasImport>>,
}

#[derive(Debug, Clone)]
pub struct HasImport {
    pub module_path: Vec<String>,
    pub type_name: String,
    pub attr_name: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ImportDecl {
    /// `import a.b.c;`           → path=[a,b,c],    items=[]   (wildcard: bring all pub from a.b.c)
    /// `import a.b.{x, y};`      → path=[a,b],      items=[x,y]
    /// `import a.b.c as alias;`  → path=[a,b,c],    items=[alias-binding]
    pub path: Vec<String>,
    pub names: Vec<String>,
}
