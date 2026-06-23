# Maka Language Specification - Current Implementation

This document reflects the **actual** state of the Maka compiler as of the most
recent commit. It supersedes all prior `Maka_Spec_V1_*` documents - those drafts
contain features that were proposed and later removed (optionals, `own *T` early
drafts, `wall` keyword, separate `heap` value-form, etc.). When this document
disagrees with an older spec, **this document wins**.

The compiler is a Rust workspace at `~/dev/maka`. Source files use the `.maka`
extension. Compilation pipeline: `lexer → parser → sema → codegen → C → cc`.

---

## 1. Lexical structure

### 1.1 Source layout

A `.maka` source file consists of:

1. An optional `module path.name;` declaration.
2. Zero or more `import path.name;` declarations.
3. Zero or more module-scope `constexpr T NAME = expr;` declarations.
4. Zero or more top-level items (functions, data, enums, externs, logic blocks,
   `cinclude` / `cblock` directives).

### 1.2 Comments

`// line comment` and `/* block comment */`. Block comments do not nest.

### 1.3 Identifiers and literals

- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`.
- Integer literals: decimal (`42`), with optional suffix (`42i32`, `255u8`). `_`
  permitted as a digit separator (`1_000_000`).
- Float literals: `1.5`, `3.14`, with optional suffix (`1.5f32`).
- String literals: `"..."` with `\n \t \r \\ \" \0 \xHH` escapes.
- Char literals: `'A'`.
- `null`, `true`, `false`, `()` (unit).

### 1.4 Keywords

```
mut const constexpr unsafe inline propagate
extern cinclude cblock rblock rdep raw own alloc free
data enum logic attr has where dyn type
if else while for in break continue match yield return
gate transfer share thread_local module import use pub
spawn join
```

The `_` identifier is a reserved placeholder usable only as a type inside
`attr` / `has` blocks (it refers to the implementing type).

Built-in type names that are also reserved: `int` `float` `bool` `char` `unit`
`string` `i8 i16 i32 i64 u8 u16 u32 u64 isize usize` `f32 f64` `Thread`.

Built-in function names (callable like normal functions, recognized by typeck):
`log` `panic` `spawn` `join`.

### 1.5 Operators

Arithmetic: `+ - * / %`. Bitwise: `& | ^ << >>`. Comparison: `== != < <= > >=`.
Logical: `&& ||`. Assignment: `= += -= *= /= %=`. Unary: `- ! &` (and `&mut`).
Postfix: `!` (pointer unwrap). Field: `.`. Index: `[]`. Cast: `as`.
Range: `..` and `..=` (used in `for`). Address-of via `&` and `&mut`.
Type-level path separator: `::` (used only in type expressions, for
associated-type paths like `T::Slot` — see §10.5).

---

## 2. Type system

### 2.1 Primitives

| Maka | C representation |
|---|---|
| `int` | `int64_t` (alias `maka_int`) |
| `float` | `double` (alias `maka_float`) |
| `bool` | `_Bool` (`bool`) |
| `char` / `u8` | `uint8_t` (alias `maka_char`) |
| `unit` | opaque `struct { int dummy; }` (alias `maka_unit`) |
| `string` | `const char*` |
| `i8 i16 i32 i64` | `int8_t … int64_t` |
| `u16 u32 u64` | `uint16_t … uint64_t` |
| `isize` / `usize` | `intptr_t` / `uintptr_t` |
| `f32` | `float` (IEEE-754 binary32, 4 bytes) — distinct ABI from `float`/`f64` |
| `f64` / `float` | `double` (IEEE-754 binary64, 8 bytes); `float` and `f64` are aliases |

`int` and the sized integers (`i32`, `u8`, etc.) are **distinct types**.
Implicit conversion between them is forbidden; use `as` to convert.

`char` and `u8` are the same byte type - they alias.

### 2.2 Pointers and references

Five categories of pointer-like things, each with a distinct contract:

| type | nullable | owns | rebindable | tracked | deref | use case |
|---|---|---|---|---|---|---|
| `own *T` | yes | yes (when non-null) | yes | – | `!` | optional / switchable owner |
| `own &T` | no | yes | (move on assign) | – | auto | guaranteed-owned slot |
| `*T` | yes | no | yes | no | `!` | non-owning view, cursor, FFI shim |
| `&T` / `&mut T` | no | no | no | yes | auto | tracked borrow |
| `raw *T` | yes | no | yes | no | `!` (inside `unsafe`) | unsafe escape, FFI |

`own &T` is the same internal type that older specs called `heap T` - the
binding owns a single heap allocation that is auto-freed at scope exit or
transferred via assignment. It is non-null by construction and accessed without
`!`. The legacy keyword `heap` has been removed; write `own &T` instead.

`*T` is the flexible escape valve: nullable, untracked, freely rebindable.
Linked-list `next` pointers, optional struct fields, downgrades of `own *T` for
read-only views - all `*T`.  `*T` does **not** own; it cannot be `alloc`'d into
directly, and there is no `free()` builtin you can call on it - either the owner
auto-frees it at its scope exit, or you go through an FFI shim for C-allocated
memory.

`raw *T` is **never shareable**, **never auto-derefs**, and its observation
(deref, field access through it, index, narrowing) requires the use site to be
inside an `unsafe { }` block. It is what extern C functions are typically
declared with when the C side controls the pointer's lifetime.

#### Coercions

| from | to | allowed? |
|---|---|---|
| `null` | `*T`, `own *T`, `raw *T` | yes |
| `null` | `own &T` | **no** (`own &T` cannot be null) |
| `&mut T` | `&T` | yes (drop write) |
| `*T mut` | `*const T` | yes (drop write) |
| `*T` | `raw *T` | yes (discarding tracking) |
| `*T` | `own *T` | **no** (cannot claim ownership you don't have) |
| `T` (struct/scalar) | `own &T`, `own *T` | yes - implicit heap-allocation |
| `own &T` | `*T`, `own *T` | yes (downgrade) |
| `own *T` | `*T` | yes (downgrade) |
| `&T` / `&mut T` | `*T` | yes |
| `&T` / `&mut T` / `*T` / `own *T` / `own &T` | `usize`, `int` | yes (reinterpret as address) |
| `*T` ↔ `*U` | yes (reinterpret) |
| `usize` → `*T` | inside `unsafe { }` only |

### 2.3 The `alloc value` expression

`alloc value` heap-allocates `value` and returns an **owning** pointer.  Its
expression type is `own &T` and coerces into `own *T`, so the safe-Maka
destination must be one of those two forms:

- `own *T x = alloc T { ... };` produces `own *T` (nullable owner).
- `own &T x = alloc T { ... };` produces `own &T` (strict non-null owner).

Landing an `alloc` in `*T` or `&T` is a **compile error** — a non-owning
binding would have nothing to auto-free at scope exit, which would leak.
The sema error reads *"`alloc value` must land in an owning slot (`own *T`
or `own &T`) — assigning an allocation to a non-owning `*T` would leak with
no auto-free."*

Inside an `unsafe { ... }` block one extra destination is allowed:

- `raw *T x = alloc T { ... };` produces `raw *T` (manual-memory escape hatch).

`raw *T` opts out of auto-free entirely.  The caller releases it explicitly
with `free p;` (the keyword, §6.5) **inside the same `unsafe` block**.
Outside `unsafe`, an alloc-into-`raw *T` errors with the message above
pointing at the `unsafe` requirement.

`alloc` is no longer a type modifier - writing `alloc T` in a type position is
a compile error directing the user at `own *T` or `own &T`.

`free p;` is a **keyword statement** (bare-word, no parens), valid **only on
`raw *T`** and **only inside `unsafe { ... }`**.  It lowers to a C `free`
call and is the inverse of `alloc → raw *T`.  Outside `unsafe`, or on any
other pointer kind, sema rejects it.

For Maka-managed memory there is no `free`:

- `own *T` and `own &T` auto-free at scope exit.  The free is **recursive**:
  freeing a value also frees the owned pointers it contains (struct fields,
  enum-variant payloads, array elements), all the way down - so a heap tree,
  linked list, or recursive enum AST frees completely when its root drops.
- **Owning temporaries are freed too.** A freshly-owned value consumed inline -
  e.g. `log(format(...))`, `f(a + b)`, or a discarded `_ = alloc ...` - is
  hoisted into a hidden owning binding and freed at scope exit; passing it to an
  `own` parameter instead transfers ownership (no double-free). So owning
  values do not leak whether they are bound to a name or used inline.
- To release early, assign `null` to an `own *T` (the auto-free fires
  immediately, the binding becomes null, and the lifetime pass invalidates
  every `*T` / `&T` aliasing it — see §6.4).
- For C-allocated buffers, declare an FFI shim (`extern "free" unit
  __libc_free(*T p);`) or call `free()` from inside a `cblock`.

### 2.4 Aggregate types

- `data Name { fields }` - C-layout struct with named fields, optional defaults,
  optional `mut` per field.
- `data Name<T, U> { fields }` - generic struct, monomorphized at use sites.
- `enum Name { Variant, Variant{int field, ...} }` - tagged union. Variants
  without payload are tag-only (C-style); with payload use the union layout.
- `[N]T` - fixed-length array (C `T[N]`).
- `[]T` / `[]mut T` - slice (pointer + length).
- `[*]T` - vector payload (only inside `own &[*]T`).
- `Name<T, U>` - generic instantiation. Monomorphized at compile time.
- `RetType(P1, P2)` - function pointer type (also covers closure types via the
  internal fat-callable representation).

### 2.4.1 `embed` - composition with promotion

A field declared `embed` carries an unnamed inner struct whose fields and
methods are promoted onto the outer struct:

```maka
data Base { mut int x = 0; }
data Outer { embed Base b; int label; }

logic Tickable { unit tick(&mut Base self) { self.x = self.x + 1; } }

unit pass_base(&Base x) { log(x.x); }

unit main() {
    mut Outer o = { b = { x = 10 }, label = 7 };
    o.tick();        // promoted method call → ticks o.b.x via embed
    log(o.x);        // 11 (promoted field)
    pass_base(&o);   // upcast: `&Outer` is accepted where `&Base` is wanted
}
```

**Field promotion.** `outer.field` resolves to the embed-nested `outer.b.field`
when no direct field shadows it. Chains of `embed` are transitively searched
in declaration order.

**Method promotion.** A method whose first parameter's underlying type can be
reached from the receiver through an `embed` chain is callable on the receiver
- either as `outer.method(...)` or by passing `&outer` where `&Inner` is
wanted. The receiver expression is rewritten to drill through the embed fields
automatically.

**Ambiguity is a compile error.** When more than one distinct `embed` path
reaches the same field name (or method-receiver type), the access is rejected;
disambiguate by writing the qualified path explicitly (e.g. `outer.a.common`).

### 2.5 Dynamic dispatch

`dyn Trait` and `dyn (T1 + T2)` produce fat pointers (data + vtable). Trait
implementations live in `logic Name { ... }` blocks. Calls dispatched via the
first `dyn` argument.

---

## 3. Expressions

Standard infix and unary operators with conventional precedence. Maka-specific:

- `expr!` - pointer unwrap (deref). Required before field/index access on `*T`,
  `own *T`, and `raw *T`. The compiler must prove the pointer is non-null at
  this site - see §6.
- `&x` / `&mut x` - borrow.
- `alloc value` - heap allocation (see §2.3).
- `expr as Type` — cast.  Whether and how the cast is checked is
  determined by the **type pair**, not by a separate sigil:
  - **Numeric / primitive** (`int → float`, `MyEnum → int`, `int → char`,
    sized ↔ unsized): plain conversion, no runtime check.  `int → char`
    truncates to the low 8 bits (C semantics — same as `(unsigned char)x`).
  - **`int → Enum`**: runtime bounds-checked against the variant count.
    Same shape as `arr[i]` — panics on out-of-range with
    `\`int as <Enum>\`: tag out of range`.  Result is the `Enum` value
    itself (no `*`, no nullable wrapper).
  - **`*T → *U`** (between `data` structs): allowed in safe code iff `U`'s
    field list is a structural prefix of `T`'s — same names, same types,
    same order, identical offsets.  See §6.6.  Otherwise must be inside
    `unsafe { ... }`.
  - **`*int → *Enum`**: runtime peek-and-tag-check at the pointee.  In
    range: yields the same address typed as `*Enum`.  Out of range:
    yields `null`.  Failure rides in the result type — no panic.
    The "pointer is the nullable carrier" convention.
  - **`*Enum → *int`**: unconditional reinterpret.  Every enum variant
    has a valid `int` representation, so no check is required.
  - **`int → *T`**, **`raw *T` observation**, etc.: unchanged per §6.5.
- `transfer x` / `share x` - only at direct argument positions of `gate`
  function calls (see §7).
- `match (expr) { arms }` - pattern matching, can appear as expression or
  statement. Exhaustiveness checked.
- `if (cond) { ... } else { ... }` as an expression - arms use `yield` to
  produce values, exactly like match arms.  `else` is required for the
  expression form; `else if` chains work naturally.  Statement-form `if`
  (without an `else`) is also fine; you only see the value-yielding form
  in expression position.
- `[expr; N]` - array fill-literal.  Replicates `expr` N times (N must be
  a compile-time integer).  Same shape as Rust's array fill.
- Direct `==` / `!=` on enum values of the same enum type.  Simple
  (payload-less) enums compare as plain ints; tagged enums compare via
  the discriminant tag.
- `e.tag` - read the discriminant of an enum value as an `int`.  For simple
  enums this is just the underlying integer; for tagged enums it is the
  variant index.  Useful for switching on a variant without writing a full
  `match` arm and for `format`-style display.
- `format(fmt, ...)` - typed string interpolation builtin.  `fmt` must be a
  string literal containing `{}` placeholders; each `{}` consumes one trailing
  argument.  Result is `own *char` (`String`) and is auto-freed at scope exit.
  Per-argument types are routed through generated to-string helpers:
  `int`/`usize` → `__maka_int_to_str`, `bool` → `__maka_bool_to_str`,
  `float` → `__maka_float_to_str`, `char` → `__maka_char_to_str`, `string` /
  `own *char` pass through unchanged.
- `RetType(params) [captures] { body }` - lambda. Captures named with mode
  `[name]` (by value), `[&name]` (by const ref), `[&mut name]` (by mut ref).
- `EnumName.Variant { field = expr, ... }` - tagged enum constructor.

---

## 4. Statements

```
let_stmt    := [mut|const|thread_local]? Type name = expr;
assign_stmt := place [+=|-=|*=|/=|%=|=] expr;
expr_stmt   := expr;
return_stmt := return [expr];
if_stmt     := if (expr) block [else (if_stmt|block)]
while_stmt  := while (expr) block
for_range   := for (Type name in expr..expr) block       // exclusive
for_range   := for (Type name in expr..=expr) block      // inclusive
for_each    := for (Type name in expr) block             // over slice/array/vec
break, continue
block_stmt  := { stmt* }
unsafe_stmt := unsafe block
match_stmt  := match (expr) { arms }
yield_stmt  := yield expr;                                // inside expr-blocks
propagate_stmt := propagate [expr] ;                      // inside inline fn
```

`Let` requires an explicit type. `mut int x = 5;` for a writable binding;
without `mut`, the binding is immutable.

`thread_local` on a `let` emits `static __thread` in C.

`propagate [expr];` is only valid inside a function marked `inline`. When the
caller invokes such a function, `propagate` returns from the **caller's** frame
(GCC statement-expression trick), enabling early-exit error patterns. The
expression is omitted (`propagate;`) when the caller returns `unit`; supplying
an expression that doesn't match the caller's return type is a compile error,
**including** for `propagate` reached through a chain of inline calls - the
check follows transitively to the outermost non-inline caller.

A `return` statement inside an `inline` function exits the *inline expansion*
only - even when it appears inside a user-written loop in that inline body.
`propagate` is the only way to escape the surrounding non-inline frame.

---

## 5. Items (top level)

```
func_decl    := [pub]? [inline]? [gate]? RetType name [<TyParams>] (params) [where ...] block
extern_decl  := extern [gate]? ["c_link_name"]? RetType name (params [, ...]?);
data_decl    := [pub]? data Name [<TyParams>] [where ...] { field_decl* }
enum_decl    := [pub]? enum Name [<TyParams>] { variant_decl* }
logic_decl   := logic Name { func_decl* }
attr_decl    := [pub]? attr Name { attr_method* }
has_decl     := [pub]? Name has Name { func_decl* }
attr_method  := RetType name (params) [where ...]  ";" | block
use_decl     := use ModPath . Receiver . Attr ;
Receiver     := Type | PrimitiveType | "*" Receiver | "&" Receiver | "&mut" Receiver
              | "own" "*" Receiver | "own" "&" Receiver | "raw" "*" Receiver
              | Ident "<" Receiver ("," Receiver)* ">"
cinclude     := cinclude "header.h";
cblock       := cblock "raw C source";
rblock       := rblock "raw Rust source";
rdep         := rdep NAME = "version";
constexpr    := [pub]? constexpr Type NAME = constant_int_expr;   // named constant
            |  [pub]? constexpr RetType NAME(params) { body }     // compile-time function
global       := [pub]? [mut]? Type NAME = expr;
```

### Compile-time functions (`constexpr fn`)

A `constexpr` function is an ordinary function that may *also* be evaluated at
compile time.  When a call to one appears in a constant position - an array
length `[fib(6)]T`, an array-fill count `[e; fib(6)]`, or a `constexpr NAME =`
initializer - the parser interprets the body and folds the call to an integer.
The same function is emitted as a normal C function, so it remains callable at
run time:

```maka
constexpr int fib(int n) {
    if (n < 2) { return n; }
    return fib(n - 1) + fib(n - 2);
}

constexpr int TABLE = fib(10) + 1;     // folded to 56 at compile time

unit main() {
    [fib(6)]int row = [0; fib(6)];     // length folded to 8
    log(fib(10));                      // 55 - same function, called at run time
}
```

The compile-time interpreter is integer-valued.  It supports `int` / `bool` /
`char` parameters and locals, `let`, assignment (including `+= -= *= /= %=`),
`if`/`else`, `while` (with `break`/`continue`), `return`, recursion, and calls
to other `constexpr` functions, over the full integer/comparison/logical/bitwise
operator set.  Constructs it cannot evaluate (`alloc`, pointers, `match`, string
ops, calls to non-`constexpr` functions) make the fold fail; the use site then
reports that the value is not a compile-time constant.  Runaway recursion or
loops are bounded by a step budget.  Generic `constexpr` functions are not folded
(generics do not cross the compile-time boundary).

### Compile-time reflection (`inline for` over `fields`)

`inline for (f in fields(value)) { body }` is a compile-time loop: the body is
unrolled once per field of `value`'s struct type, and never appears as a loop in
the generated C.  Inside the body the loop variable exposes the current field:

| accessor  | meaning                                              |
|-----------|------------------------------------------------------|
| `f.name`  | the field's name, as a `string` literal              |
| `f.value` | the field itself (`value.<field>`), with its own type |
| `f.index` | the field's zero-based position, as an `int`         |
| `f.ty`    | the field's type rendered as a `string`              |

`ty` is spelled `ty`, not `type` (`type` is a reserved keyword).

Because each iteration is unrolled separately, `f.value` may have a *different
type* per field, which a runtime loop could not express.  This is the mechanism
for writing derive-style code once and having it apply to every `data`:

```maka
data Vec3 { int x; int y; int z; }

int sum<T>(&T v) {
    mut int total = 0;
    inline for (f in fields(v)) { total = total + f.value; }   // unrolled x/y/z
    return total;
}

unit dump<T>(&T v) {
    inline for (f in fields(v)) { log(f.name); log(f.value); }
}
```

The unroll happens during type checking and rides on monomorphization: in a
generic function the fields are only known once a concrete type is bound, so the
body is unrolled (and checked) per instantiation, not on the generic template.
Rules and limits:

- `fields(x)` takes a single argument that must be a plain variable (it is
  re-read once per field, so side effects are not duplicated).
- The argument's type must resolve to a `data` struct (peeling `&`/`own &`);
  a concrete non-struct receiver is an error.
- Only struct fields are reflected; enum-variant reflection is not yet provided.
- Embedded (`embed`) fields are reached as ordinary fields of the outer struct.

A module-scope `Type NAME = expr;` declares a global.  Without `mut` the
global is read-only; with `mut` it is writable from any function in the
module.  Globals are emitted as `static <ctype> __maka_global__NAME = init;`
at file scope in the generated C, so the initializer must fold to a C
constant expression - integer / boolean / character / float literals and
their arithmetic and bitwise combinations are supported; calls and
allocations are not.  Visibility follows the same rules as functions
(`pub` makes the global importable from other modules; immutable globals
read like a name; mutable globals can be read and written through the same
binding).

`pub constexpr` is exportable: another module brings the name in with
`import path.NAME;`, and references in expression position substitute the
integer value at the use site.  Array-size references must still be
in-file constexprs (the parser folds at parse time).

`<TyParams>` accepts both `<T>` and `<T: Attr>` shorthand - the latter desugars
to `where T has Attr`.

`pub` is enforced cross-module - see §11.

`inline` marks a function for caller-frame splicing (statement-expression
expansion). Recursion is forbidden among inline functions. `propagate` only
works inside `inline`.

`gate` marks a function as a synchronization-boundary crossing - see §7.

### 5.1 `cinclude`, `cblock`, `rblock`, `rdep`

Inline Rust via `rblock` and Cargo deps via `rdep` are documented separately
in [`RUST_INTEROP.md`](RUST_INTEROP.md): the driver compiles each rblock as
a sidecar Cargo crate, emits `#[no_mangle] extern "C"` shims per `pub fn`,
and injects matching Maka `extern` declarations into the AST so calls
look like ordinary Maka function calls.  Type marshalling auto-handles
primitives, `&str`, `String`, and `#[repr(C)]` structs; everything else
flows through as an opaque heap handle (`own *mut unit` on the Maka
side, `Box::into_raw` ↔ `Box::from_raw` on the Rust side, auto-dropped).

### 5.2 `cinclude` and `cblock`

```maka
cinclude "math.h";
cinclude "stdio.h";
cblock "static double sq(double x) { return x*x; }";
```

`cinclude "name.h";` emits `#include <name.h>` in the generated C prologue.
`cblock "..."` pastes the contents verbatim at module scope after typedefs
and before any extern function prototypes. The string is treated raw - embedded
braces, semicolons, etc. are fine.

### 5.2 `extern` and variadic FFI

```maka
extern float sqrt(float x);                    // libm
extern gate unit thread_spawn_with_buf(*unit buf, int n);
extern i32 printf(string fmt, ...);            // variadic
```

`extern` declares a C function. `gate` makes calls to it a boundary crossing
(allowing `transfer`/`share` at the call site). Optional string literal
overrides the C link name. Trailing `...` marks a variadic. Variadic call sites
require at least the fixed-arity prefix to match; trailing args type-check
without coercion.

The driver accepts:
- `-l<name>` and `-L<path>` for linker flags;
- `--link path.c|path.o|path.a` for C source files / objects to compile-and-link
  alongside;
- `--link -lname` / `--link -L/path` as alternate forms.

---

## 6. Lifetime and ownership

### 6.1 Move semantics on owning types

Assigning between `own *T` or `own &T` bindings transfers ownership. The source
is invalidated; subsequent use is a compile error.

```maka
own &Node a = alloc Node { value = 1 };
own &Node b = a;                  // move
// log(a.value);                   // ❌ use of moved value
```

Returning an owning value moves it to the caller. Passing it as a `own *T` /
`own &T` argument moves it into the callee.

**Implicit reborrow.** Inside a function with `&mut T` (or `&T`) parameter
`g`, writing `&mut g` (or `&g`) at a call site that wants `&mut T` would
yield `&mut &mut T` - one borrow layer too many.  The compiler peels the
outer borrow automatically, so helper-to-helper chains can write either
`g` (the parameter itself) or `&mut g` (visually explicit) without
worrying about the wrapper layer.

### 6.2 Auto-free at scope exit

When an owning binding goes out of scope without being moved, the compiler
emits `free(binding)` automatically. `own *T` also auto-frees on reassignment
(the previous value is freed before the new one is stored).

### 6.3 Forced handling for `*T` deref

Dereferencing any nullable pointer (`*T`, `own *T`, `raw *T`) requires the
lifetime pass to **prove** the value is non-null at the deref site. There is no
runtime null-check macro. Proof comes from:

- The value is the immediate result of `alloc value`.
- The local is currently inside a narrowing window opened by `if (p != null)`.
- The local appears after a guarding early-exit: `if (p == null) { return; }`.
- The local appears inside a `while (p != null)` body.
- The value is the result of a call to a function whose **interprocedural
  summary** is `NeverNull` — i.e. every return path in the callee is itself
  provably non-null.  Summaries are computed via fixpoint over the lowered
  HIR after every function (including instantiations) is checked, so chains
  like `top` → `mid` → `leaf` converge as long as the leaf is provable.
  Functions whose return type is not a nullable carrier (`&T`, value types)
  are trivially `NeverNull`; functions returning `*T`/`own *T`/`raw *T` must
  have every return expression provably non-null *under flow tracking* —
  the summary pass propagates per-LID `known_nonnull` through Let/Assign,
  honours `if (p != null)` / `if (p == null) return;` narrowing, and joins
  branches at if/else.  So `*T make() { *T p = alloc T; return p; }`
  classifies as `NeverNull` even though the literal return expression is
  just a `Local`.
- The value is captured by a closure from an outer scope where it was
  provably non-null at the capture site.  Captures of by-value or `&`-mode
  locals carry the outer flow fact into the synthesized lifted body so the
  closure's first use of the binding does not need a redundant guard.

Without a proof, the compiler rejects the deref with a message that suggests
the appropriate guard. **There is no `MAKA_UNWRAP` runtime macro** - the
compiler will not insert a panic on null.

### 6.4 Downstream invalidation on owner change

`*T` aliases and `&T` / `&mut T` borrows that depend on an owner's pointee
are invalidated whenever the owner's pointee changes — at scope exit, on
re-assignment, on null-assignment, on move.  The **owner is never restricted**;
all the cost falls on the downstream views.

The dispatch per alias kind:

| alias kind | what happens when its owner mutates / moves / null-assigns |
|---|---|
| `*T` (nullable, untracked) | flow state auto-NULLs the alias; the next deref needs a fresh non-null proof, comparisons-to-null fire the silent-overwrite warning below |
| `&T` / `&mut T` (tracked borrow) | the borrow is **poisoned**; any subsequent use is a hard compile error (`use of poisoned reference X`) |

The auto-NULL for `*T` emits the same flow-sensitive warning the compiler used
to fire for the scope-exit-only case:

```
warning at L:C: pointer `p` was auto-nulled when its pointee went out of scope
and has not been explicitly re-assigned on every code path since; this use
observes that silent overwrite - re-assign `p` yourself before reading it
```

Comparing the pointer to `null` (`p == null` / `p != null`) does **not**
suppress the warning - the comparison reflects the silent overwrite and the
programmer should know. Only an explicit user assignment to the pointer (even
to `null`) clears the warning state.

Borrowing references (`&T`, `&mut T`) outliving their referent are a **hard
compile error** (`poisoned`).

### 6.5 `unsafe { }`

`unsafe { ... }` permits exactly six operations otherwise forbidden:

1. Casting an integer to a pointer (`usize as *T`, `int as *T`).
2. Casting a reference to `raw *T` (`&T as raw *T` — drops borrow tracking).
3. Observing a `raw *T` (deref, field, index, narrowing-based deref).
4. The manual-memory escape-hatch pair, **only meaningful together**:
   - `raw *T x = alloc T { ... };` — allocate into an untracked pointer
     (no auto-free; the binding leaves with no destructor).
   - `free p;` — bare-word keyword statement, lowers to a C `free` call;
     accepts only `raw *T`.
5. Mentioning `*unit` (the untyped opaque pointer) in a let binding,
   function parameter, or return type.  In safe code `*unit` is rejected
   outright — typed handles from the stdlib (`Atomic`, `Mutex`,
   `TlsConn`, etc.) carry the runtime handle.  Inside `unsafe { }` the
   bare `*unit` is allowed for raw FFI plumbing.  `*unit` also remains
   allowed in `extern` declarations and inside `cblock`.
6. Casting between pointer types whose inner types are not structurally
   prefix-compatible (`*Foo → *Bar` where `Bar`'s fields are not a prefix
   of `Foo`'s).  Safe in-prefix casts are documented in §6.6.

Inside `unsafe`, `raw *T` still has to be narrowed (forced-handling — §6.3)
before deref.  `unsafe` does not turn off the lifetime pass; it just unlocks
those six operations.

### 6.6 Pointer-kind conversions and the `&(p!)` pattern

Maka's pointer kinds (`own *T`, `own &T`, `*T`, `&T`, `&mut T`, `raw *T`) are
**lifetime annotations** on top of the same C-level address; the data shape
is identical across kinds.  Converting between them is therefore a no-op at
codegen — only the type-system tag changes.

Cross-type pointer casts (`*T → *U` where `T ≠ U`) follow a separate rule:
in safe code they're allowed iff `U`'s field list is a **structural prefix**
of `T`'s — both must be `data` types, and `U`'s fields must match the first
|U.fields| of `T`'s by name, type, and order (which guarantees identical
C-level offsets).  This makes the common header-extraction pattern safe:

```maka
data NetHeader { u8 version; int length; }
data NetPacket { u8 version; int length; int payload_len; }

unit handle(*NetPacket p) {
    *NetHeader hdr = p as *NetHeader;  // prefix-safe, no unsafe needed
    log(hdr!.length);
}
```

Cross-type casts that don't satisfy the prefix rule (`*Foo → *Bar` between
unrelated layouts, `*float → *int` for bit-pattern punning, etc.) require
`unsafe { }` (§6.5 item 6).  The prefix rule avoids the strict-aliasing
hazard: every access through `*hdr` goes through one of the shared prefix
fields, so the C compiler's TBAA sees identical type access on both sides.

The `*int ↔ *Enum` pair is exempt from the prefix rule because an `Enum`
is just one of the integer's valid tag values.  `*int → *Enum` reads the
pointee, checks it against the variant set, and returns either the same
pointer cast to `*Enum` (in range) or `null` (out of range) — failure is
carried in the nullable result, not as a panic.  `*Enum → *int` is
unconditional: every variant is by construction a valid `int`.  Both
directions are no-ops at the bit level; only the type tag changes.

```maka
enum Move { Idle, Walking = 1, Running = 2 }

unit handle(*int tag) {
    *Move m = tag as *Move;          // null if *tag ∉ {0,1,2}
    if (m == null) { return; }
    log(m! as int);                  // safe: *Enum → int via deref + cast
}
```

The implicit-coercion table is governed by a single principle: **loosening
flags is implicit, tightening a flag requires proof**.

- *owning* can only be dropped (a non-owner cannot become an owner without
  a phantom free-obligation).
- *tracked* can be dropped freely (`&T → *T`, `&mut T → *T`).
- *nullable* can be gained freely (`own &T → own *T`, `own &T → *T`).
- *nullable* can be **dropped only with a non-null proof**, discharged by
  the `!` operator.

The dispatch for conversions to `&T` / `&mut T` (the most subtle case):

| source | how to convert | why |
|---|---|---|
| `own *T`, `*T`, `raw *T` (nullable) | `&T b = &(src!);` — `!` discharges the null obligation, `&` produces the borrow | nullable → non-null is a tightening |
| `own &T` (non-null, owning) | `&T b = src;` — implicit retype | already non-null, just dropping the *owning* flag |
| `&T`, `&mut T` (non-null, tracked) | implicit reborrow / mutability-peel | already in the target shape |

The non-null sources don't need `!` because there's no null obligation to
discharge — that's the unifying rule.  `!` exists **only** to prove
non-null; on a non-nullable source it would be meaningless and is rejected
(`! only valid on nullable pointers`).

The lifetime pass tracks alias relationships through this conversion machinery
so the §6.4 downstream-invalidation rule fires on every kind of view, no
matter which combination of `as`, `!`, or implicit coerce produced it.

**`&` peel on Ref-typed places.**  A `&T` / `&mut T` binding reads as the
pointee value at top level (auto-deref applies — `b.v` accesses through the
ref).  Consistently, `&b` where `b: &T` does **not** stack to `&&T` / `*&T`;
it gives the address `b` already stores — a `&T` (or `*T` in a pointer
context).  This is the inverse direction of auto-deref: the same `b` is
"value at top level" for `.v` access and "address" under `&`.  Fat-pointer
ref kinds (`&dyn Trait`, `&[T]`) keep the no-peel behavior because their
ref value carries extra metadata beyond the bare address; `&m` on those
goes through the existing reborrow/coerce path.

---

## 7. Concurrency

### 7.1 `gate` functions and the `transfer` / `share` discipline

A function declared `gate` is a synchronization-boundary crossing. At every
direct call site, each argument may carry a modifier:

```maka
gate fn worker(int payload, *int data) { /* ... */ }

unit main() {
    int x = 10;
    own *int data = alloc 42;
    worker(share x, transfer data);    // x copied, data ownership moves
}
```

`transfer X`: invalidates `X` in the caller - ownership crosses.

`share X`: the type of `X` must be `Shareable`. Primitives, sync primitives,
and structs whose every field is Shareable are auto-derived as Shareable. `*T`
to mutable data and `raw *T` are NOT Shareable.

For generic structs with associated-type-placeholder fields (e.g.
`data Wrapper<T: Stored> { T::Slot inner; }`, §10.5), Shareability is
evaluated **per concrete instantiation**, not at the generic declaration
site.  `Wrapper<int>` is Shareable iff the resolved `int::Slot` is
Shareable; `Wrapper<*Foo>` is Shareable iff `*Foo::Slot` is.  The
unmonomorphized form has no Shareable verdict.

Recognized Shareable types by name: `Mutex`, `RwLock`, `Spinlock`, `Channel`,
`AtomicI8`–`AtomicI64`, `AtomicU8`–`AtomicU64`, `AtomicBool`, `AtomicPtr`,
`Thread`.

Calls to non-`gate` functions reject `transfer`/`share` annotations.

### 7.2 Three-tier spawn API

Maka exposes three concurrency tiers with the same surface shape, picked
by which keyword you use to spawn.  All three take a `unit()` closure and
return a `*Thread` handle; `join(*Thread)` blocks until the work finishes.

```maka
*Thread t1 = thread(unit() { log(1); });    // OS thread — blocking-safe
*Thread t2 = spawn(unit()  { log(2); });    // fiber — concurrent IO
*Thread t3 = job(unit()    { log(3); });    // work item — parallel compute
join(t1);
join(t2);
join(t3);
```

`thread` is the kernel-thread tier (blocking-safe, ~8 MB stack, true
parallelism).  `spawn` is the fiber tier (ergonomic concurrent IO).
`job` is the work-stealing pool tier (parallel compute fanout).  See
`CONCURRENCY.md` for the full design and decision tree.

**Cross-thread capture rule** — `thread`, `job`, and `spawn_pool` cross
a thread boundary.  Their closure captures are checked at the call site,
analogous to the `transfer` / `share` rule for `gate` (§7.1):

- **By-value capture of an owning type** (`own *T` / `own &T`) is allowed —
  by-value moves ownership into the thread (transfer semantics).
- **By-value capture of a Shareable type** (per the Shareable bound in §7.1)
  is allowed — the value is copied into the thread's env (share semantics).
- **Bare `unit` capture** is allowed — `unit` has no runtime representation,
  so there's nothing to share or transfer.
- **Borrow capture** (`[&x]` / `[&mut x]`) is **rejected** — the borrow's
  lifetime is tied to the caller's scope, but the thread can outlive that
  scope, so the borrow would dangle or race.
- **Non-Shareable non-owning capture** (`*T`, `raw *T`, mutable slices, etc.)
  is **rejected** — the pointee lifetime is the caller's owner's, with the
  same lifetime hazard.
- **`*unit` is rejected** like any other non-Shareable pointer.  The
  stdlib exposes its concurrency primitives as typed opaque-handle wrappers
  (`Atomic`, `Mutex`, `WaitGroup`, `Once`, `RwLock`, `IntChan`, `FloatChan`,
  `ByteChan`, `TlsConn`) which are in the Shareable allowlist by name.
  User code captures the typed handle into a spawn closure; the raw `*unit`
  the wrapper holds never escapes the stdlib.  (Earlier drafts of this spec
  allowed a `*unit` carve-out at spawn boundaries — the carve-out was
  dropped once the stdlib typed-handle migration landed.)

The fiber tier (`spawn`) runs on the same thread as the caller, so
captures are unrestricted — borrows are fine because the fiber's
lifetime is bounded by its caller.

**Implementation status**: all three are currently `pthread_create`-backed.
The real fiber runtime (slab pool + scheduler + epoll reactor) and the
real job runtime (work-stealing pool) replace the backings without
changing the surface.  User code written today against `thread` /
`spawn` / `job` will keep working.

Composition helpers documented in `CONCURRENCY.md`:
  - `join(&[]Handle<T>) -> []T` — homogeneous wait-all
  - `select(&[]Handle<T>) -> T` — homogeneous race, winner cancels losers
  - `par_for` / `par_reduce` / `par_map` — data-parallel over slices

Heterogeneous composition (different return types) lives in user code:
either wrap each spawn body's return in a common `enum` and pass a
homogeneous slice, or spawn each handle separately and await each into
its own typed variable.  No `JoinN<T1, T2, ...>` heterogeneous structs.

All of the above are wired into the runtime as compiler builtins; see
`CONCURRENCY.md` for usage examples and the implementation status table.

`Thread` is a built-in opaque type (recognized by name; backed by `pthread_t`
in the generated C). It is Shareable.

### 7.3 Sync primitives (typed handles)

`Mutex`, `RwLock`, `WaitGroup`, `Once`, `Atomic`, `AtomicBool`, `AtomicPtr`,
the `Chan` family (`IntChan` / `FloatChan` / `ByteChan`), and `TlsConn` are
exposed as typed opaque handles in the stdlib — a `data` declaration
wrapping a single `*unit` field that holds the raw runtime pointer:

```maka
pub data Mutex { *unit h; }
extern "maka_fmutex_new"     *unit __fmutex_new();
extern "maka_fmutex_lock"    unit  __fmutex_lock(*unit m);
extern "maka_fmutex_unlock"  unit  __fmutex_unlock(*unit m);
extern "maka_fmutex_destroy" unit  __fmutex_destroy(*unit m);

pub Mutex mutex_new()              { Mutex m = { h = __fmutex_new() }; return m; }
pub unit  mutex_lock(Mutex m)      { __fmutex_lock(m.h); }
pub unit  mutex_unlock(Mutex m)    { __fmutex_unlock(m.h); }
pub unit  mutex_destroy(Mutex m)   { __fmutex_destroy(m.h); }
```

The wrapper is named-Shareable (the type checker's Shareable allowlist
matches `Mutex`, `RwLock`, `Spinlock`, `Channel`, `Atomic`, `AtomicBool`,
`AtomicPtr`, the sized `AtomicI{8,16,32,64}` / `AtomicU{8,16,32,64}`
variants, `WaitGroup`, `Once`, `IntChan`, `FloatChan`, `ByteChan`,
`TlsConn`, and `Thread` by name).  User code captures the typed handle
into spawn closures by value; the raw `*unit` it holds never
escapes the stdlib.  This replaces the original FFI-style `*unit`-only
sync surface, which is no longer part of the public stdlib API.

### 7.4 Concurrency primitives (the irreducible base)

The atoms Maka can't express in pure Maka itself — CPU atomic instructions,
kernel waits/wakes, syscalls — are exposed as **compiler builtins**.  These
are recognized by name, dispatch to the right C intrinsic in codegen, and
are the lowest layer the rest of the concurrency story is built on.

Everything else in the concurrency stack — `Atomic`, `AtomicBool`,
`AtomicPtr`, `Mutex`, `RwLock`, `WaitGroup`, `Once`, the `*Chan` family
(`IntChan` / `FloatChan` / `ByteChan`), etc. — is **pure Maka source**
built on top of these builtins.  The stdlib reads like Maka, not like
FFI.  These are non-generic typed handles wrapping a single `*unit`
field — see §7.3.

| builtin | signature | C lowering |
|---|---|---|
| `atomic_cas` | `<T: int> atomic_cas(&mut T p, T expected, T new) -> T` | `__atomic_compare_exchange_n(p, &expected, new, false, SEQ_CST, SEQ_CST)`; returns the old value either way |
| `atomic_load` | `<T: int> atomic_load(&T p) -> T` | `__atomic_load_n(p, SEQ_CST)` |
| `atomic_store` | `<T: int> atomic_store(&mut T p, T v)` | `__atomic_store_n(p, v, SEQ_CST)` |
| `atomic_fetch_add` | `<T: int> atomic_fetch_add(&mut T p, T delta) -> T` | `__atomic_fetch_add(p, delta, SEQ_CST)`; returns old value |
| `atomic_fetch_sub` | `<T: int>` ditto | `__atomic_fetch_sub` |
| `atomic_fetch_and` | `<T: int>` ditto | `__atomic_fetch_and` |
| `atomic_fetch_or`  | `<T: int>` ditto | `__atomic_fetch_or` |
| `atomic_fetch_xor` | `<T: int>` ditto | `__atomic_fetch_xor` |
| `atomic_fence` | `atomic_fence(int order)` | `__atomic_thread_fence(order_map(order))` where order: 1=acquire, 2=release, 3=acq_rel, 4=seq_cst |
| `futex_wait` | `futex_wait(&const int addr, int expected) -> int` | Linux: `syscall(SYS_futex, addr, FUTEX_WAIT, expected, ...)`. Windows: `WaitOnAddress`. Darwin: spin-yield fallback. |
| `futex_wake` | `futex_wake(&const int addr, int n) -> int` | Linux: `syscall(SYS_futex, addr, FUTEX_WAKE, n, ...)`. Windows: `WakeByAddress{Single,All}`. Darwin: no-op. |
| `thread_yield` | `thread_yield()` | POSIX: `sched_yield()`. Win: `SwitchToThread`. |
| `syscall` | `syscall(int n, int a1..a6) -> int` | POSIX: `syscall(n, a1..a6)`. Win: returns -1 (errno is left untouched — the Win32 prologue `#define`s it to a function-call macro, not an lvalue). |

All thirteen are recognized by their bare names with no qualifier (same
recognition pattern as `log` / `panic` / `spawn` / `join`).  All accept
`int` and `i8`/`i16`/`i32`/`i64`/`u8`/`u16`/`u32`/`u64` as the `T` for the
atomic family; codegen dispatches on the argument type at the call site.

**Why this small list and not just `atomic_cas`?**  Mathematically `atomic_cas`
alone suffices to derive every other atomic op via CAS-loops.  But on x86
a CAS-loop atomic load is ~30 cycles versus 1 for a native aligned read;
similarly for add/sub/and/or/xor versus `lock xadd`/`lock add`/etc.  The
direct builtins let codegen emit the single-instruction form when one
exists.  The fence is necessary on weak-memory architectures (ARM, RISC-V)
where individual ops aren't sufficient to enforce ordering across multiple
locations.

**No more direct `extern` atomic / pthread helpers in user code.**  The
previous shape (`extern int maka_atomic_load_i64(&int p);` etc.) is
gone — the stdlib now exposes typed handles (`Atomic`, `Mutex`, `RwLock`,
…) as pure Maka wrappers over these builtins.  See §7.3.

**Future perf primitives** (not implemented; added when measurements show
they matter): `atomic_load_relaxed` / `acquire`, `atomic_store_relaxed` /
`release`, and per-order variants of `fetch_*` for tighter memory
orderings than SEQ_CST.

---

## 8. Pattern matching

```maka
match (e) {
    IntLit{value} value,
    Add{left, right} eval(left) + eval(right),
    Mul{left, right} eval(left) * eval(right),
}
```

Each arm is `pattern body,` where `body` is a single expression or a `{}` block.
Variant patterns destructure named fields. Literal patterns match by value
(`42`, `"hello"`, `true`). `_` matches anything. `null` matches null pointers.

Exhaustiveness is checked at compile time for enum scrutinees. A non-exhaustive
match without a wildcard arm is a compile error.

Match is both an expression and a statement. As an expression, all arms must
produce the same type, returned via `yield expr;` (or just the trailing
expression).

Generics are erased before pattern dispatch — match scrutinees always
have concrete monomorphized types.  A binding destructured out of a
generic struct (`Wrapper<T>` per §10.5) receives the **concrete resolved**
type for the current monomorphization, never an abstract `T::Slot`.

---

## 9. Closures and lambdas

```maka
// No-capture lambda
unit() task = unit() { log("hello"); };

// Capturing lambda - env on stack, must NOT escape
int(int by) bump = int(int by) [&mut counter] { counter = counter + by; };

// Heap env (for closures that escape via spawn or return)
own &unit() job = alloc unit() [transfer payload] { use(payload); };
```

Capture modes:
- `[x]` - by value.
- `[&x]` - by const ref.
- `[&mut x]` - by mut ref.
- `[transfer x]` - moves an owning value into the closure env.
- `[share x]` - shareable capture (Shareable types only).

The **lambda-escape rule**: a closure with non-empty captures that escapes the
spawning frame (returned by value, stored in `spawn`, etc.) requires its env to
be heap-allocated. Use `alloc unit() [...] { ... }` (or whatever signature
matches).

**Cross-thread capture restriction**: spawn-tier closures passed to
`thread`, `job`, or `spawn_pool` are further constrained — borrow captures
(`[&x]` / `[&mut x]`) are rejected outright, and non-borrow captures must
be of an owning type or a Shareable type (the stdlib's typed concurrency
handles — `Atomic`, `Mutex`, `WaitGroup`, etc. — are named-Shareable).
See §7.2 for the full rule and rationale.  The fiber tier (`spawn`) keeps
the unrestricted closure semantics — same thread, lifetime bounded by
caller.

Lambdas without captures are lifted to top-level functions at AST level. With
captures, they are compiled to a synthesized env struct + a lifted function;
the closure value is a fat pointer `Callable_<KEY> { code, env }`.

Lambdas inherit the **lexical `unsafe { }` scope** of their call site: a
lambda body written inside `unsafe { }` may use the operations §6.5 unlocks
(e.g. mention `*unit`).  The lambda-lift threads the `unsafe` state through
to the synthetic top-level function so the type checker sees the same
unsafe context the lambda was lexically written in.

---

## 10. Generics

`data Name<T, U> { ... }` and `RetType name<T, U>(params) { ... }`. Type
parameters appear in field types, parameter types, return types.

Generics are **monomorphized** at compile time: every concrete instantiation
gets its own struct/function in the generated C. `Vec<T>`, `Pair<int, string>`
etc. are expanded to distinct C structs.

### 10.1 Attributes: `attr` + `has` (preferred)

```maka
// Declare a contract.  `_` is the placeholder for the implementing type.
attr Show {
    unit show(&_ self);                       // signature-only - required
    string label(&_ self) { return "any"; }   // default body - optional
}

// Implement.  `_` in the impl is rewritten to the receiver type.
data Point { int x; }

Point has Show {
    unit show(&Point self) { log(self.x); }
    // `label` is inherited from the attr default.
}
```

**Contract matching.** Every `has` method must correspond to a method declared
in the `attr`. Every attr method must either be implemented in the `has` block
or have a default body in the attr. Signature mismatches (arity, param types,
return type - compared after `_` is substituted with the implementing type) are
rejected.

**Bound syntax.** Two surfaces are accepted and mean the same thing:

```maka
unit render<T: Show>(&T x) { x.show(); }                 // shorthand
unit render<T>(&T x) where T has Show { x.show(); }      // long form
```

At each generic instantiation, the substituted type must have a visible `has`
impl for the named attr - otherwise the call is rejected.

**Method dispatch.** `x.show()` on a value of type `T` with bound `T: Show`
resolves to the `has` impl chosen by the receiver's concrete type at
instantiation. Postfix method calls also dispatch to attr-namespaced methods
when no top-level function of that name exists.  When the candidate's
first parameter is `&_ self` or `&mut _ self`, the receiver is
auto-borrowed — `x.show()` matches `show(&Foo)` without the user spelling
`(&x).show()`.

**Disambiguation under multiple bounds.** When two attrs in scope both
declare a method with the same name, qualify the call with the attr name
via `::`:

```maka
A::run(&f)        // prefix qualified
f.A::run()        // postfix qualified — auto-borrows the receiver
```

The bare `run(&f)` form errors as `ambiguous call to \`run\`: N candidates`;
the qualified forms filter candidates to a specific attr's impl before
overload resolution.

**Dot-form vs. `::` form.** `Attr.method(args)` (dot) is still accepted
for legacy reasons (it was the original `Logic.fn(args)` qualified-call
spelling).  But the dot form is **shadowed by locals**: when a binding
of the same name is in scope, `name.method(args)` is a postfix call on
the local, not an attr-qualified call.  The `::` form is **never
shadowed** — it always names the attr.  Prefer `::` for new code; the
dot form remains for compatibility with the legacy `logic` block call
style (§10.2).

See `181_attr_qualified_call.maka` (the qualified forms) and
`182_local_shadows_attr.maka` (shadowing rule) for worked examples.
No `T::` prefix — the receiver type is already at the call site.

**Visibility.** `has` impls are file-private by default. To use a `has` impl
in another module:
1. The `has` block must be marked `pub`.
2. The consuming file must declare `use ModulePath.Type.Attr;`.

```maka
// shapes.maka
module shapes;
pub data Point { int x; }
pub attr Show { unit show(&_ self); }
pub Point has Show { unit show(&Point self) { log(self.x); } }

// app.maka
module app;
import shapes.Point;
use shapes.Point.Show;             // explicit propagation

unit go<T: Show>(&T x) { x.show(); }
unit main() { Point p = { x = 42 }; go(&p); }
```

Without the `use`, the bound check fails with a hint naming the exact `use`
declaration needed. There is no implicit propagation - `pub has` impls are
opt-in at every consumer.

### 10.2 `logic` blocks as legacy trait shape

```maka
logic Drawable {
    unit draw(&Color self) { /* ... */ }
}

unit render<T>(&T x) where Drawable<T> { /* ... */ }

render(&color);    // OK: Color implements Drawable (via the logic block)
```

The older `logic Trait { method(&Receiver self) }` pattern is still accepted
and registers an impl just like `has`: the first param's underlying nominal
type becomes the implementer. The new `attr`/`has` form is preferred - it
makes contract and impl explicit, supports default bodies, and is the only
form that contract-matches.

A `logic` block may be marked `pub`: the trait registration and every method
inside it become exported.  Visibility composes with the same `use Mod.Type.Trait;`
machinery as `has` impls - a `pub logic` is reachable cross-module only when
the consumer explicitly opts in.  Per-method `pub` on a logic-block method is
not part of the grammar; visibility flows from the block.

**`logic` is frozen at its current shape.**  The new features from
§10.4 (parametric receivers) and §10.5 (associated types) are exclusive
to `attr`/`has`.  `logic` blocks are subject to the §10.4 coherence
rule against overlapping impls but cannot themselves use generic
receivers or declare associated types.  When the trait extensions
stabilize, `logic` is expected to be deprecated in favor of `attr`/`has`;
no further features will be added to the `logic` form.

### 10.3 Monomorphization

Generics are **monomorphized** at compile time: every concrete instantiation
gets its own struct/function in the generated C. `Vec<T>`, `Pair<int, string>`
etc. are expanded to distinct C structs.

### 10.4 Generic `has` receivers

> **Stabilized 2026-06-18.**  §10.4–10.7 — parametric `has` receivers,
> overlap-rejection coherence, associated types with the placeholder data
> pattern, **default associated types**, and **bounds on associated
> types** (`<T: Foo<Slot = i64>>`) — are all implemented end-to-end and
> exercised by the test suite (`169_has_primitive`, `170_assoc_type_basic`,
> `171_parametric_has`, `172_worked_example_atomic`,
> `173_default_assoc_type`, `174_bounded_assoc_type`, `neg_overlap_impl`).

The receiver of a `has` impl can be **parametric**, not only a concrete named
type.  Four receiver kinds are accepted:

```maka
*T  has Stored { ... }              // any `*T` — non-owning mutable pointer
&T  has Stored { ... }              // any `&T` — borrow
own *T has Stored { ... }           // any `own *T` — owning pointer
Box<T> has Stored { ... }           // any concrete generic struct
int has Stored { ... }              // primitives (`int`/`bool`/`i8..u64`)
```

Inside a generic-receiver `has` block, `T` refers to the bound type
variable.  Methods see the receiver type via the usual `_` placeholder
substitution (e.g. `&_ self` becomes `&*T self` in a `*T has Stored`
block; `&_` is an accepted alias).

**Receiver patterns.** The grammar admits:
- A concrete named type (`Foo`, `Foo<int>`, `int`, `bool`, `i32`, etc.)
- A primitive (`int`, `bool`, sized ints, `char`, `string`, `float`)
- A pointer / reference family with one type-variable inside:
  `*T`, `&T`, `&mut T`, `own *T`, `own &T`, `raw *T`
- A generic struct with one or more type variables:
  `Box<T>`, `Pair<A, B>` — the variables bind the generic positions

Receiver kinds can also nest: `*Box<T> has Stored { ... }` matches `*Box<int>`,
`*Box<string>`, etc., but not `*int`.

**Resolution.** At a generic call site `foo<U>(...)` with bound `U: Stored`
and concrete `U = X`, sema walks the registered `Stored` impls and unifies
`X` against each receiver pattern.  The unique matching impl is the
resolved instance.

**Unification algorithm.**  Unification of a receiver pattern against a
concrete type is **first-order, structural, no implicit subtyping**:
the pattern and the concrete type are walked in parallel; type variables
in the pattern bind to whatever they meet; concrete head constructors
(pointer kind, struct name, primitive) must match exactly; differing
mutness (`*T` vs `*const T` vs `*mut T`) is **not** a match.  The
algorithm bottoms out on primitives (each is its own atom; `int` and
`bool` do not unify).  No backtracking, no occurs check (cycles forbidden
elsewhere — see §10.5).

Implementation note: unification receives HTypes that are **already
monomorphized**.  A receiver pattern `Box<T>` and a concrete `Box<int>`
both arrive at the unifier as a single `StructId` (the monomorphizer
turned `Box<int>` into a distinct struct at instantiation time, and the
pattern's `Box<T>` carries `TyVar(T)` in the type-arg position only
inside the impl's own bookkeeping — by the time unification runs on a
call site, the call site has already produced a concrete `StructId`).
This means the unifier compares at the instance level, not the
template level, and no `Generic<…>`-vs-`Generic<…>` case is needed.

**Coherence — overlap is rejected.** If a new `has` impl's receiver pattern
overlaps with an already-registered impl for the same attr, the new impl
is a compile error.  Two patterns overlap iff there is some concrete type
that simultaneously unifies with both (i.e. an assignment of all type
variables on both sides exists making the patterns structurally equal).
Examples:

- `*Foo has Stored` and `*T has Stored` — `*Foo` unifies with both; overlap, rejected at the second.
- `*T has Stored` and `&T has Stored` — disjoint (different head constructors); OK.
- `*T has Stored` and `*const T has Stored` — disjoint (different mutness); OK.
- `Box<int> has Stored` and `Box<T> has Stored` — `Box<int>` unifies with both; overlap, rejected.
- `Pair<A, B> has Stored` and `Pair<int, B> has Stored` — `Pair<int, int>` unifies with both; overlap, rejected.  Partial specificity in any position still counts.
- `int has Stored` and `bool has Stored` — disjoint primitives; OK.

The rule keeps method dispatch unambiguous without specialization rules.
Users who want concrete-overrides-generic must add a separate trait or
use composition.

**When overlap is diagnosed.**  Overlap is checked **at the registration
site of the second impl**, during the sema resolve pass — not lazily at
the call site.  An impl that conflicts with an imported impl is rejected
in the importing module, with both impl locations included in the error.
There is no "ok in this module, fails at use site" — registering an
overlapping impl is always an error, even if the conflicting impl came
from a `use`'d declaration.

**Visibility.** `pub` and the cross-module `use Mod.Type.Attr;` rule
(§10.1) carry over unchanged.  For a parametric receiver `R has Attr`,
the use form spells `R` verbatim:

```maka
use shapes.*T.Stored;        // imports the `*T has Stored` impl from shapes
use shapes.Box<T>.Stored;    // imports the `Box<T> has Stored` impl
```

The grammar production (§5) is correspondingly relaxed: in `use ModPath . R . Attr ;`,
`R` may be any receiver pattern (concrete `Type`, primitive name, or
parametric form — `*T`, `&T`, `&mut T`, `own *T`, `own &T`, `raw *T`,
or `Name<T1, T2, ...>` with type variables in any subset of positions).

**Receiver placeholder.**  Inside an `attr` declaration or a `has` impl
body, the existing `_` placeholder type (§1.4) refers to the receiver.
In an `attr` it stands for "the implementing type, whatever it is"; in
a `has` impl it is substituted with the impl's receiver pattern
(post-type-variable-binding at monomorphization).  There is no separate
`Self` keyword — `_` covers both jobs.

**Interaction with `logic` blocks.**  `logic` blocks (§10.2) are subject
to the same coherence rule — a `logic` block whose first-parameter
receiver type duplicates an existing `has` (or `logic`) impl of the same
attr is rejected with the same overlap diagnostic.  However, `logic`
blocks **do not** gain the new features: parametric receivers (§10.4) and
associated types (§10.5) are exclusive to the `attr`/`has` form.  See
§10.2 for the deprecation stance.

### 10.5 Associated types on `attr`

An `attr` may declare **associated types** — type-level slots each impl
fills in.  Associated types extend the contract from "the impl provides
these methods" to "the impl provides these methods AND picks these
types."

```maka
attr Stored {
    type Slot;                            // associated type — impl chooses
    unit init(&mut _ self, _ value);
}

*T has Stored {
    type Slot = *T;                       // for a *T receiver, Slot is *T
    unit init(&mut _ self, *T value) { self.inner = value; }
}

own *T has Stored {
    type Slot = OwnCell<T>;               // for own *T, Slot is OwnCell<T>
    unit init(&mut _ self, own *T value) { /* ... */ }
}

data OwnCell<T> { own *T ptr; int drop_flag; }
```

**Multiple associated types.**  An `attr` may declare any number of
associated types, in any order, intermixed with method signatures.
Each `has` impl must provide a `type Name = ...;` for every associated
type the attr declares (modulo the default-assoc-type extension below).

```maka
attr Pair { type Left; type Right; Left first(&_ self); Right second(&_ self); }
data Tagged<L, R> { L l; R r; }
Tagged<L, R> has Pair {
    type Left  = L;
    type Right = R;
    L first (&_ self) { return self.l; }
    R second(&_ self) { return self.r; }
}
```

**Path syntax.** Inside a function or struct definition with a bound
`T: Stored`, the associated type is named `T::Slot`.  This is a *type
expression*: it can appear anywhere a type can.

```maka
// Function with assoc-type return:
T::Slot fetch<T: Stored>(&T self) { ... }

// Generic struct using an assoc type as a field type:
data Wrapper<T: Stored> {
    T::Slot inner;                        // ← the placeholder
}
```

When `Wrapper<*Foo>` is instantiated, sema looks up `*T has Stored`,
substitutes `T = Foo`, reads `type Slot = *T = *Foo`, and emits the
concrete struct `{ *Foo inner; }`.  For `Wrapper<own *Foo>` the resolved
slot is `OwnCell<Foo>`, so the emitted struct is `{ OwnCell<Foo> inner; }`
which expands inline to `{ own *Foo ptr; int drop_flag; }`.

**`data` is still the single source of layout truth.** The slot count
of `Wrapper<T>` is fixed by the `data` declaration: exactly one field
named `inner`.  The associated-type resolution can only choose what
*type* sits in that slot — never whether more slots exist.  The "I want
to inject additional fields per impl" pattern is expressible by having
the resolved `Slot` itself be a struct: that struct's fields become the
extra storage, but they live behind the named slot.  Composition is
preferred over hidden injection; see §10.7 below for the rationale.

**Abstract vs concrete typing.**  An unmonomorphized generic function
with a bound — `fn f<T: Stored>() { let x: T::Slot = ...; }` — type-
checks `T::Slot` **abstractly**: sema verifies the assoc type is
declared in the bound attr and treats the path as an opaque type
parameter throughout the body.  Operations applicable to an abstract
`T::Slot` are only those the attr's method signatures expose (e.g.
`atomic_load_cell(&self)` returning `T::Slot`).  Concrete resolution is
deferred to monomorphization; the resolved type is substituted into the
function body and rechecked against the concrete type's operations.

A struct literal at a concrete instantiation — `Wrapper<*Foo> { inner = ptr }` —
type-checks `ptr` against the **post-resolution** type (i.e. `*Foo`,
the resolved `*T::Slot`), not against the abstract `T::Slot`.  After
monomorphization, every field type is concrete.

**Pattern matching.**  Pattern scrutinees always have concrete
monomorphized types — generics are erased before pattern dispatch.
A binding in `match (w) { Wrapper { inner } => ... }` receives the
**concrete resolved** field type of the matched instantiation (`*Foo`,
not the abstract `T::Slot`).

**No-impl-at-decl is fine.**  Writing `data Wrapper<T: Stored> { T::Slot inner; }`
in a module where **no** `Stored` impls exist yet is not an error.  The
data declaration is checked structurally; the bound is only enforced at
each concrete instantiation site.  An unused generic struct with bounds
is valid in isolation.

**Sizing.**  An unmonomorphized generic struct (`Wrapper<T>` for type
variable `T`) has **no defined size or layout** — it is abstract.  Only
concrete instantiations (`Wrapper<int>`, `Wrapper<*Foo>`) have a size;
that size depends on the resolved `T::Slot`.  `sizeof` (when added) on
the bare `Wrapper<T>` is a compile error; on `Wrapper<X>` it returns the
concrete size.  Code cannot reference an unmonomorphized generic at
runtime — every value of a generic type lives at a concrete instantiation.

**Cyclic assoc-type definitions are forbidden.**  An `has` impl whose
`type Slot = R` body causes `R`'s resolution (directly or through any
chain of struct-field type lookups) to refer back to the same
parameterized struct *with the same parameter substitution* is rejected
at the impl declaration site with the diagnostic
`type Slot = ... creates a cycle involving T`.  Example:

```maka
*T has Stored { type Slot = Wrapper<T>; }     // ← rejected (cycle)
data Wrapper<T: Stored> { T::Slot inner; }    // would infinitely expand
```

Cycles broken by an indirection (`type Slot = *Wrapper<T>` — a pointer
behind which the recursion lives) are permitted; the size and layout are
well-defined because the indirection is finite.

**Disambiguating assoc types under multiple bounds.**  When a generic
parameter has multiple bounds and two of them declare an associated type
with the same name, write `T::AttrName::Name` to pin the lookup to a
specific attr's slot:

```maka
attr A { type Slot; Slot a_get(&_ self); }
attr B { type Slot; Slot b_get(&_ self); }

data Pick<T> where T has A, T has B {
    T::A::Slot a_val;
    T::B::Slot b_val;
}
```

Both `T::A::Slot` and `T::B::Slot` resolve cleanly because each path
names exactly one attr.  Bare `T::Slot` under two bounds that both
declare it is still ambiguous and surfaces as a downstream type-mismatch
diagnostic; rename or qualify.

See `180_qualified_assoc_path.maka` for the worked example.

**Errors and hints (parity with §10.1).**

- *Missing impl.*  `Wrapper<X>` where no `X has Stored` impl is in scope:
  `type \`X\` does not implement \`Stored\` (required for assoc type \`T::Slot\`)`
  plus, if a `pub` impl exists in another module, the hint
  `add \`use Module.X.Stored;\``.
- *Overlapping impls.*  The receiver-overlap diagnostic from §10.4 fires
  before assoc-type resolution.
- *Cycle.*  The cycle diagnostic above, with both impl and struct
  locations.

**Dyn dispatch interaction.**  `dyn Attr` (§2.5) is **not** compatible
with associated-type field placement in v1.  A struct field cannot have
type `dyn Attr::Slot` (the type isn't known statically; the vtable
doesn't carry layout).  `T::Slot` in a struct field requires `T` to be
either a concrete type or a generic bound, both of which monomorphize.
A future revision may add `dyn Attr<Slot = ConcreteType>` ("dyn with
fixed associated types") in the manner of Rust's object-safety rules,
but it is not in v1.

**Default associated types** (stabilized 2026-06-18):
```maka
attr Stored {
    type Slot = int;                      // default — impls may omit `type Slot = ...;`
                                          // and inherit this value
}
int has Stored { /* no type Slot — inherits int from the default */ }
```
See `173_default_assoc_type.maka` for the worked example.

**Bounds on associated types** (`<T: Stored<Slot = i64>>`, stabilized
2026-06-18): restrict the bound to those impls whose `Slot` resolves
exactly to the named type.  Bindings live inside the bound's angle
brackets, mixed with positional attr-args, in any order:

```maka
int  use_int<T: Stored<Slot = int>>(&T x)         { return read(x); }
int  some_fn<T: Convert<int, Slot = string>>(...) { /* ... */ }
```

At each instantiation site, sema looks up the impl whose receiver
pattern unifies with `T`, reads its `type Slot = R`, substitutes the
impl's type variables via the unification env, and rejects the
instantiation if `R` doesn't `type_eq` the bound's named value.  See
`174_bounded_assoc_type.maka` for the worked example.

**Coherence with assoc types.** When two parametric `has` impls overlap
in the receiver pattern, they conflict regardless of which assoc types
they pick — the coherence rule is purely on receivers (§10.4).

**Resolution order.** At each generic instantiation of `foo<T: Stored>`
with concrete `T = X`:
1. Find the unique `has` impl for `Stored` whose receiver unifies with `X`.
2. Read the impl's `type Slot = R` line.
3. Substitute the impl's type variables in `R` from the unification.
4. The resulting concrete type is `X::Slot`.

If step 1 finds no impl or two impls, instantiation is rejected with
a specific error.

Implementation note: assoc-type resolution runs as a **post-pass** in
the resolver — after every `has` impl has been registered.  Struct
instantiation (e.g. materializing `Wrapper<int>`) happens earlier in
the pipeline, while the impls of the attr the wrapper is bound by may
not yet exist; any `T::Slot` field types are left as abstract
`AssocType` placeholders at that point.  Once all impls are
registered, a final pass walks every materialized struct's fields and
substitutes each placeholder with the unifier-resolved concrete type.
Code that lives outside generic bodies never sees an unresolved
`AssocType` — by the time typecheck runs on a concrete instantiation,
the field types are concrete.

### 10.6 Worked example: `AtomicPtr<T>`

```maka
attr AtomicCell {
    type Storage;                                          // impl picks the cell type
    Storage atomic_load_cell(&_ self);
    unit    atomic_store_cell(&mut _ self, Storage v);
    Storage atomic_swap_cell (&mut _ self, Storage v);
}

// Pointer atomics: a typed cell holds a typed pointer.
*T has AtomicCell {
    type Storage = *T;
    *T  atomic_load_cell (&_ self)               { return atomic_load(self); }
    unit atomic_store_cell(&mut _ self, *T v)    { atomic_store(self, v); }
    *T  atomic_swap_cell (&mut _ self, *T v) {
        let old = atomic_load(self);
        atomic_store(self, v);
        return old;
    }
}

// Ints, bools, etc. fall under one primitive impl per type — small surface,
// stable, and users don't need extension.
int has AtomicCell {
    type Storage = int;
    int  atomic_load_cell (&_ self)              { return atomic_load(self); }
    unit atomic_store_cell(&mut _ self, int v)   { atomic_store(self, v); }
    int  atomic_swap_cell (&mut _ self, int v) { /* CAS-loop */ ... }
}

bool has AtomicCell { type Storage = bool; /* ... */ }

// The wrapper:
pub data Atomic<T: AtomicCell> {
    T::Storage cell;                                   // placeholder; concrete type per T
}
```

User code:

```maka
Atomic<*Box> ap = atomic_new(&my_box);                 // Storage = *Box
*Box current = atomic_load_cell(&ap);
atomic_store_cell(&mut ap, &other_box);

Atomic<int>  ai = atomic_new(0);                       // Storage = int
int          n  = atomic_load_cell(&ai);
```

User extension — declare your own atomic-able type:

```maka
data MyHandle { *unit raw; }
MyHandle has AtomicCell {
    type Storage = MyHandle;
    MyHandle atomic_load_cell(&_ self)             { /* ... */ }
    /* ... */
}

Atomic<MyHandle> ah = atomic_new(my_handle);
```

### 10.7 Why field-injection is rejected

A natural-looking alternative to associated types is letting the `has`
impl inject *new* fields into the wrapper struct:

```maka
attr Stored {
    extra_fields { ... }                  // hypothetical — REJECTED
}
*T has Stored {
    extra_fields { int rc; }              // would add rc to Wrapper<*T>
}
```

This is **not** part of Maka.  The "`data` declaration is the complete
layout" invariant is a load-bearing property of the language: a reader
of `data Foo { ... }` is guaranteed that they see every field Foo will
ever have, in every monomorphization, in every compilation unit.  That
invariant unlocks local reasoning about size, layout, FFI compatibility,
and debugger views.

Every motivating case for injected fields (refcounting, drop-flags,
observer slots) is expressible via the `type Slot = SomeBundle<T>`
pattern — the impl declares a struct, packs its bookkeeping fields
there, and that struct becomes the wrapper's named slot.  The wrapper's
shape stays uniform; the bundle's shape is visible at *its* declaration
site.  Locality is preserved.

The cost — one extra named struct per pattern — is small.  The benefit —
no "the struct I read is not the struct I get" surprise — is large.

---

## 11. Modules and visibility

### 11.1 Source-level declaration

```maka
module helper;            // optional, at file top

pub int twice(int n) { return n + n; }
int secret(int n) { return n * 7; }
```

A file may declare its module name with `module path.name;` at the top.
Without it, the file belongs to the implicit root module `<root>`.

### 11.2 Per-item visibility

The `pub` modifier on a top-level item makes it visible across module
boundaries.

### 11.3 Cross-module enforcement (calls and type references)

When function A in module X references a function or type B:

- If B's module == X's module: always allowed.
- If B's module != X's module: **two** checks must pass:
  1. **`pub`**: B must be marked `pub`. Otherwise the error is
     `` `B` is private to module `Y`; mark it `pub` to <call|use> from `X` ``.
  2. **Imported**: B's `(module, name)` must appear in the file's import list.
     Extern declarations are exempt (they link externally, not by Maka module).
     Otherwise the error is
     `` `B` is in module `Y` and must be imported (`import Y.B;`) to call from `X` ``.

The same `pub` rule applies to `data` and `enum` types: referencing a private
`data Secret { ... }` from another module yields
`` data type `Secret` is private to module `helper`; mark it `pub` to use from `caller` ``.

The driver tags every parsed item with its source file's `module` declaration
(or the implicit root if absent) and that file's import list. The resolver
stores the module path on every `FuncSig`/`StructInfo`/`EnumInfo`; the type
checker enforces both rules at call and type-binding sites.

### 11.4 `import` declarations

```maka
import a.b.c;             // bring single name `c` (declared in module a.b)
import a.b.c as alias;    // same, bind under `alias`
import a.b.{x, y, z};     // selective list
import a;                 // bring `a` from the root module
import a.b.*;             // wildcard - bring every `pub` name from `a.b`
```

The wildcard form (`import a.b.*;`) authorises any `pub` item in `a.b`.
Resolution still respects the same `pub` rule per item; the `*` only
means "I commit to importing whatever this module exports" without
listing each name.  Use it sparingly when a module is a curated
re-export hub.

Imports are **enforced**: a cross-module call/type reference must have a
matching import in the caller's file or be in the same module. Without
imports, no cross-module symbols are visible (calls fail with the
"must be imported" diagnostic).

Names of built-in functions (`log`, `panic`, `spawn`, `join`) are
always visible and require no import.

### 11.5 `use ModPath.Type.Attr;` - explicit `has` propagation

```maka
use shapes.Point.Show;
```

`use` declarations propagate a `pub has Type Attr` impl from another module
into the current file's bound-check scope. They follow the same prelude region
as `import` (after `module`, before any items). At least three dotted segments
are required - the last two are always `(Type, Attr)`, everything before is
the module path.

A `use` declaration also authorizes calls to that impl's methods across the
module boundary - you do not need a separate `import` for each method, since
the `use` covers the whole impl.

**Bound visibility for generic types.**  Importing a `pub` generic type
(e.g. `import shapes.Wrapper;` for `pub data Wrapper<T: Stored> { ... }`)
does **not** implicitly bring the bound's attr into scope.  The bound
attr (`Stored` in this case) must be independently visible at the
instantiation site — typically via its own `import` (for the attr name)
and a `use Module.X.Stored;` for the specific impl being relied on at
instantiation.  Without that, instantiation is rejected and the missing
`use` line is suggested in the error hint (per the rule in §10.5).

**Parametric receivers.**  For a `pub` parametric `has` impl
(`pub *T has Stored { ... }` in `shapes`), the use form spells the
receiver verbatim: `use shapes.*T.Stored;`.  See §10.4 for the
full receiver grammar.

---

## 12. C interop summary

What a Maka program can do directly with C:

- Declare extern functions (including variadic).
- Include system headers via `cinclude`.
- Embed raw C with `cblock`.
- Link against installed libraries via `-l` / `-L` flags.
- Reinterpret cast `*T ↔ *U`, `*T ↔ usize`, `&T ↔ usize`.
- Synthesize pointers from integers inside `unsafe { }`.
- Escape lifetime tracking via `raw *T`.

What a Maka program **cannot** do directly:
- Use C macros that expand to non-function syntax - they must be wrapped in a
  cblock helper.
- Use C struct types without declaring an equivalent `data` struct.
- Vararg-call a non-extern function.

### 12.1 ABI at the FFI boundary

Maka's source-level types are **language types**, not target ABI types.
At extern declarations the user must spell out the C-side width
explicitly — codegen lowers the type the user wrote, nothing more.

| C type (target) | What you MUST write in Maka |
|---|---|
| `char`, `signed char`, `unsigned char` | `i8` / `u8` (or `char`) — all map to a 1-byte cell |
| `short` | `i16` |
| `unsigned short` | `u16` |
| `int` | `i32` (NOT `int` — Maka `int` ≡ `i64`) |
| `unsigned` | `u32` |
| `long` (LP64) | `i64` (NOT `int` on its own — same value, but be explicit) |
| `long long` | `i64` |
| `size_t`, `uintptr_t` | `usize` |
| `ssize_t`, `intptr_t` | `isize` |
| `float` | `f32` (NOT `float` — Maka `float` ≡ `f64` ≡ C `double`) |
| `double` | `float` (or `f64`) |
| `void*`, `T*` | `*unit` (in extern decls), `*T` (when typed) |

The two foot-guns:

1. **Maka `int` is always 64-bit.**  Writing `extern int read(int fd, ...)`
   declares `int64_t read(int64_t, …)` on the C side, which mismatches
   libc's `int read(int, …)` and will silently corrupt arguments and
   return values.  Use `i32` for any C function whose signature mentions
   `int`.

2. **Maka `float` is double.**  `float`/`f64`/native lower to C `double`
   (8 bytes); `f32` is the dedicated 4-byte float.  `extern f32 sqrtf(f32 x)`
   correctly calls libc's `float sqrtf(float)`; writing `extern float sqrtf(float x)`
   compiles to `double sqrtf(double)` — ABI mismatch, garbage results.

The sized-int family (`i8`/`i16`/`i32`/`i64`/`u8`/`u16`/`u32`/`u64`) and
the new `f32` ARE ABI-exact — what you write is what gets emitted.
Default `int` / `float` are convenient inside Maka but risky at the
boundary; the safe rule is "use sized types in every `extern` signature
and `cblock` shim."

A worked test lives at `tests/programs/175_float_ffi_abi.maka`.

---

## 13. Code generation

Generated C requires `-std=c11`. Standard headers always included:
`stdio.h, stdlib.h, stdint.h, stdbool.h, string.h, wchar.h`. `_XOPEN_SOURCE
600` is defined. pthread is linked when threading primitives are used.

Heap allocations: `malloc` + assign in a GCC statement-expression. Auto-frees
emit `free(p);` at scope exits and `if (p) free(p);` for `own *T`.

Closure trampolines: `static void __tramp_<name>(void* env, args...) { ... }`
unpacks env and calls the lifted body.

Spawn lowering: `(Thread*)__maka_spawn(closure.code, closure.env)`. The
prologue defines `__maka_spawn`, `__maka_join`, `__maka_thread_entry` as
pthread wrappers.

### 13.1 `--freestanding` (kernel / no-libc target)

`makac --freestanding` strips the libc-using prologue, skips the auto-
include of `stdlib/std.maka`, and routes the allocator + log + panic
hooks to user-provided extern symbols.  Generated C compiles cleanly
under `gcc -ffreestanding -nostdlib`.

**Emitted prologue**: only the headers C requires every freestanding
implementation to provide — `<stdint.h>`, `<stdbool.h>`, `<stddef.h>`,
`<stdarg.h>` (plus user `cinclude` directives).  No `<stdio.h>`,
`<stdlib.h>`, `<pthread.h>`, `<sys/*>`, or any other libc header.

**User-supplied runtime hooks** (declared extern, never defined by codegen):

| Symbol | Lowered from | Notes |
|---|---|---|
| `void* __maka_alloc(size_t sz)` | `alloc T { ... }` | OS-provided heap (bump / slab / buddy / whatever). |
| `void __maka_free(void* p)` | `free p;` and auto-frees | May be a no-op if the kernel leaks. |
| `void __maka_panic(const char* msg)` | `panic`, `maka_check_idx` | Halt, log, hcf — author's call. |
| `void __maka_log_int(int64_t v)` | `log(int)` | Useful for early-boot debug; can be a no-op. |
| `void __maka_log_str(const char* s)` | `log(string)` | Same. |

The codegen prologue declares these as `extern` and `#define`s `malloc` /
`free` / `maka_panic` / `maka_log_*` to forward to them — so existing
emit sites (heap allocation in stmt-expressions, index checks, etc.)
work unchanged with no per-site gating.

**No `int main` shim** — the OS author calls `maka_main()` directly from
their `_start` / boot code.  Codegen emits the user's `unit main()` as
`void maka_main(void)`.

**Atomic builtins** (`atomic_load`, `atomic_cas`, `atomic_fetch_*`,
`atomic_fence`) lower to `__atomic_*` C intrinsics that gcc/clang emit
inline — no libc, no syscalls.  They work unchanged in freestanding mode.

**Platform-dependent builtins** (`futex_wait`, `futex_wake`,
`thread_yield`, `syscall`) assume a host kernel exists, so they should
not be used in freestanding code targeting the kernel level — the OS
*is* the host kernel.  Calling them compiles, but the user-supplied
runtime must define `__maka_futex_*` / `__maka_thread_yield` /
`__maka_syscall` itself.

**Build invocation** (example):
```
makac --freestanding kernel.maka --emit-c -o kernel.c
gcc -ffreestanding -nostdlib -fno-stack-protector -nostartfiles \
    -c kernel.c -o kernel.o
gcc -ffreestanding -nostdlib -c runtime.c -o runtime.o
ld -T linker.ld -o kernel.img kernel.o runtime.o
```

A worked stub runtime + smoke test live at `tests/freestanding/`.

---

## 14. Built-in functions (reserved `FuncId`)

| name | FuncId | signature | notes |
|---|---|---|---|
| `log` | `u32::MAX` | `unit log(T x)` | accepts any single arg; auto-derefs primitive refs; auto-coerces `own *char` to `string` |
| `panic` | `u32::MAX - 2` | `unit panic(string msg)` | prints to stderr, calls `abort()` |
| `spawn` | `u32::MAX - 3` | `*Thread spawn(unit() closure)` | accepts bare or alloc'd closure |
| `join` | `u32::MAX - 4` | `unit join(*Thread t)` | blocks; reclaims handle |
| `+` (concat) | `u32::MAX - 5` (and `_freel/_freer/_freeb`) | `own *char (string, string)` | binop on two `string`s - result is heap-allocated, auto-freed; chained concats use freeing variants so intermediates don't leak |
| `read_line` | `u32::MAX - 6` | `own *char read_line()` | reads one line from stdin (NUL-terminated, no trailing `\n`); returns `null` on EOF |
| `read_int` | `u32::MAX - 7` | `int read_int()` | reads one base-10 integer from stdin; panics on malformed input |
| `__maka_int_to_str` | `u32::MAX - 11` | `own *char (int)` | format-arg converter; never written by user code |
| `__maka_bool_to_str` | `u32::MAX - 12` | `own *char (bool)` | format-arg converter |
| `__maka_float_to_str` | `u32::MAX - 13` | `own *char (float)` | format-arg converter |
| `__maka_char_to_str` | `u32::MAX - 14` | `own *char (char)` | format-arg converter |

---

### 14.0 String types and the stdlib

- **`string`** - a borrowed view of NUL-terminated bytes (`const char*`).
  Stack-only handle, never owns heap.  Literals, slices of buffers, and
  borrowed views of `String` all have this type.  Hardcoded type name.
- **`String`** - a compiler alias for `own *char`: heap-owned, NUL-terminated,
  auto-freed at scope exit.  Returned by constructors (`a + b`, `read_line()`)
  and stored in `String` bindings.  Coerces to `string` for reads, log, and
  function arguments.  Hardcoded type name in the compiler.

Everything else stdlib lives in `stdlib/std.maka` (real Maka source,
embedded into the compiler via `include_str!` at build time) and every
item there - types, enums, functions - is `pub` in `module std;` and
**requires an explicit `import std.Name;`** to use.  Same rule that
governs any cross-module reference: `pub` makes an item importable, not
automatically visible.

Currently provided by `stdlib/std.maka`:

- `Option<T>` - generic tagged option.
- `Result<T, E>` - generic tagged result.
- `str_len(string) -> usize` - byte length of a borrowed string.
- `str_eq(string, string) -> bool` - byte-equal comparison.

Genuine compiler builtins (always in scope, never declared in Maka source):
`log`, `panic`, `spawn`, `join`, `read_line`, `read_int`, `+` on strings,
and `.len` on slices / arrays / vectors.

One ergonomic exception: the `for x in user_iterator` desugaring references
`Option<T>`, so the compiler injects a synthetic `import std.Option;` for
the duration of the lowered body.  Files that iterate user types via
`for ... in ...` do not need to write the import themselves - same model as
`for x in slice` (which doesn't ask the user to import any compiler-
internal helper).

### 14.1 The `main` function

```maka
unit main()                  // program ignores CLI args
unit main([]string args)     // program receives a slice of CLI args
int main()                   // exit code = the return value
int main([]string args)      // both: args + exit code
```

The slice form receives a borrowed view of the OS-level `argv`: `args[0]` is
the program name, `args[1..]` are user-supplied arguments. The slice and its
strings are read-only and live for the program's lifetime; Maka code may not
free them. The OS only delivers C strings - any further parsing (`--port 8080`
→ int) is library/user code on top of `args`.

### 14.2 Slice / array / vector length

`.len` on a value of `[N]T`, `[]T`, or `[*]T` (and `&` / `own &` references to
those) yields the element count as `usize`. The codegen lowers to a constant
for fixed arrays, a struct field read for slices/vectors.

```maka
[3]int arr = [10, 20, 30];
log(arr.len);          // 3
[]int s = arr;
log(s.len);            // 3
```

---

## 15. Driver invocation

```
makac <input.maka>... [-o output] [--emit-c] [--run] [--release|-O0..-O3|-Os] [--link <file|flag>] [-l name] [-L path]
```

- Multiple `.maka` inputs are merged into a single module set; each retains its
  declared module path for `pub` enforcement.
- `--emit-c` writes the generated `.c` instead of (or alongside) compiling it.
- `--run` invokes the compiled binary immediately after build.
- `--release` (alias `-O2`) optimizes the generated C; `-O0` (default), `-O1`,
  `-O3`, `-Os` are also accepted.  The driver always passes `-fwrapv` (Maka
  `int` wraps two's-complement) and `-fno-strict-aliasing`, so optimized builds
  stay well-defined.
- `--link foo.c` compiles `foo.c` alongside the generated C.
- `--link -lname` and `-l name` pass `-l` to the C compiler (after objects, so
  GNU ld resolves symbols correctly).
- `--link -L/path` and `-L /path` add a library search path.

---

## 16. What this spec deliberately omits

The following appear in older drafts or aspirational discussions but are
**not part of the current implementation**:

- Optionals (`?T`, `??T`, etc.) - removed in v1.2 and never returned.
- `await` and `async` - removed; the keywords no longer lex.
- `heap` as a keyword - removed; use `own &T` for the strict-owning binding and
  `alloc value` for the allocation expression.
- `alloc` as a type modifier - `alloc T` in a type position is a parse error.
- The `wall` keyword - renamed to `gate`.
- `move()` as an explicit operator - moves are implicit on assignment.
- Self-hosting - the compiler is written in Rust.

---

## 17. Open and acknowledged gaps

These are real limitations the implementation is honest about:

- **Borrow-escape via field stash is rejected conservatively.** Storing a
  `&T`/`&mut T` value into a struct field whose container is reachable through
  a parameter is rejected, since the compiler can't prove the borrow's source
  outlives the struct.  Lifetime annotations would let this loosen; v1 takes
  the safe-by-default rejection.
- **No qualified-call dispatch syntax for `has` methods.** `Attr.method(x)` is
  not accepted as a forced-dispatch form.  Within a generic with bounds,
  ambiguity is resolved by the surrounding `where T has Attr<U>` clause; but
  outside that context, two identically-named methods on the same type need
  the surrounding where-bound to disambiguate.
- **No auto-borrow on method calls.** `p.method()` requires `p` to match the
  receiver's type exactly; if the method takes `&_ self`, the call site must
  write `(&p).method()`.  No magic `&` insertion at dispatch.

These are tractable to fix; they are not architectural blockers.
