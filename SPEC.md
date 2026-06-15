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
extern cinclude cblock rblock rdep raw own alloc
data enum logic attr has where dyn
if else while for in break continue match yield return
gate transfer share thread_local module import use pub
spawn join
```

The `_` identifier is a reserved placeholder usable only as a type inside
`attr` / `has` blocks (it refers to the implementing type).

Built-in type names that are also reserved: `int` `float` `bool` `char` `unit`
`string` `i8 i16 i32 i64 u8 u16 u32 u64 isize usize` `f32 f64` `Thread`.

Built-in function names (callable like normal functions, recognized by typeck):
`log` `free` `panic` `spawn` `join`.

### 1.5 Operators

Arithmetic: `+ - * / %`. Bitwise: `& | ^ << >>`. Comparison: `== != < <= > >=`.
Logical: `&& ||`. Assignment: `= += -= *= /= %=`. Unary: `- ! &` (and `&mut`).
Postfix: `!` (pointer unwrap). Field: `.`. Index: `[]`. Cast: `as`, `as?`
(checked). Range: `..` and `..=` (used in `for`). Address-of via `&` and `&mut`.

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
| `f32` / `f64` | `double` (both - single precision is not yet distinct) |

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
Linked-list `next` pointers, optional struct fields, function returns where the
caller manages free - all `*T`.

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

`alloc value` heap-allocates `value` and returns an owning pointer. The result
type is **context-typed** by the destination slot:

- `own *T x = alloc T { ... };` produces `own *T`.
- `own &T x = alloc T { ... };` produces `own &T`.
- `*T x = alloc T { ... };` produces `*T` (no ownership tracking; caller must
  `free()` manually).

`alloc` is no longer a type modifier - writing `alloc T` in a type position is
a compile error directing the user at `own *T` or `own &T`.

`free(p)` is the built-in deallocator for `*T`. Calling `free` on an `own *T` or
`own &T` is unnecessary (compiler frees automatically) but not currently
rejected.

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
- `expr as Type` - unchecked cast.
- `expr as? Type` - checked cast (currently for `int → enum` and `int → char`).
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
use_decl     := use ModPath . Type . Attr ;
cinclude     := cinclude "header.h";
cblock       := cblock "raw C source";
rblock       := rblock "raw Rust source";
rdep         := rdep NAME = "version";
constexpr    := [pub]? constexpr Type NAME = constant_int_expr;
global       := [pub]? [mut]? Type NAME = expr;
```

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

Without a proof, the compiler rejects the deref with a message that suggests
the appropriate guard. **There is no `MAKA_UNWRAP` runtime macro** - the
compiler will not insert a panic on null.

### 6.4 Dangling-pointer collapse + warning

If a `*T` aliases a local and that local goes out of scope, the compiler
auto-NULLs the `*T` at scope exit (it would otherwise dangle). When a
subsequent read of `*T` observes that NULL without an explicit re-assignment on
every code path, a flow-sensitive warning fires:

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

`unsafe { ... }` permits exactly two operations otherwise forbidden:

1. Casting an integer to a pointer (`usize as *T`, `int as *T`).
2. Observing a `raw *T` (deref, field, index, narrowing-based deref).

Inside `unsafe`, `raw *T` behaves identically to `*T` - the forced-handling
rule still applies (you still need narrowing to deref). `unsafe` does not turn
off the lifetime pass; it just unlocks two specific operations.

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

**Implementation status**: all three are currently `pthread_create`-backed.
The real fiber runtime (slab pool + scheduler + epoll reactor) and the
real job runtime (work-stealing pool) replace the backings without
changing the surface.  User code written today against `thread` /
`spawn` / `job` will keep working.

Composition helpers documented in `CONCURRENCY.md`:
  - `join(h1, h2, ..., hN)` → `JoinN<T1, ..., TN>` — wait for all
  - `select(h1, h2, ..., hN)` → `SelectN<T1, ..., TN>` — wait for first
  - `par_for` / `par_reduce` / `par_map` — data-parallel over slices

These ship as part of the real runtime; not yet wired in this MVP.

`Thread` is a built-in opaque type (recognized by name; backed by `pthread_t`
in the generated C). It is Shareable.

### 7.3 Sync primitives via FFI

`Mutex`, `RwLock`, `Spinlock`, and `Channel` are exposed through `extern`
declarations of pthread-backed helpers automatically emitted in the codegen
prologue. The user declares:

```maka
extern *unit maka_mutex_new();
extern unit maka_mutex_lock(*unit m);
extern unit maka_mutex_unlock(*unit m);
extern unit maka_mutex_destroy(*unit m);
```

Then uses them as opaque `*unit` handles. The Shareable allowlist matches by
the user-declared struct name when applicable.

### 7.4 Atomics

`extern int maka_atomic_load_i64(&int p);` and the family of
`maka_atomic_fetch_*_i64(&mut int p, int delta)` helpers are emitted in the
prologue and may be declared via `extern`.

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

Lambdas without captures are lifted to top-level functions at AST level. With
captures, they are compiled to a synthesized env struct + a lifted function;
the closure value is a fat pointer `Callable_<KEY> { code, env }`.

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
when no top-level function of that name exists.

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

### 10.3 Monomorphization

Generics are **monomorphized** at compile time: every concrete instantiation
gets its own struct/function in the generated C. `Vec<T>`, `Pair<int, string>`
etc. are expanded to distinct C structs.

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

Names of built-in functions (`log`, `free`, `panic`, `spawn`, `join`) are
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

---

## 14. Built-in functions (reserved `FuncId`)

| name | FuncId | signature | notes |
|---|---|---|---|
| `log` | `u32::MAX` | `unit log(T x)` | accepts any single arg; auto-derefs primitive refs; auto-coerces `own *char` to `string` |
| `free` | `u32::MAX - 1` | `unit free(*T p)` | non-owning `*T` only; rejects `own *T` / `own &T` (would double-free) |
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
`log`, `panic`, `free`, `spawn`, `join`, `read_line`, `read_int`, `+` on
strings, and `.len` on slices / arrays / vectors.

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

`.len` on a value of `[N]T`, `[]T`, or `[*]T` (and `&`/`heap` references to
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
makac <input.maka>... [-o output] [--emit-c] [--run] [--link <file|flag>] [-l name] [-L path]
```

- Multiple `.maka` inputs are merged into a single module set; each retains its
  declared module path for `pub` enforcement.
- `--emit-c` writes the generated `.c` instead of (or alongside) compiling it.
- `--run` invokes the compiled binary immediately after build.
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
