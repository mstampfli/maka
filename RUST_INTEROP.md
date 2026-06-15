# Maka ↔ Rust interop

This document specifies how Maka calls into Rust source written inline in
`.maka` files, including crate dependencies, type marshalling, opaque
handles for arbitrary Rust types, panic safety, and `Send`/`Sync`
enforcement at thread-crossing call sites.

It is companion to `SPEC.md` (the language reference).  Where the two
disagree, `SPEC.md` is authoritative for Maka the language; this
document is authoritative for the Rust bridge.

---

## 1. Surface syntax

Two new top-level items, alongside `cinclude` / `cblock`:

```
rblock_item := rblock "raw Rust source";
rdep_item   := rdep IDENT = STRING_LIT;
```

```maka
rdep serde = "1";
rdep regex = "1.10";

rblock "
    use regex::Regex;

    pub fn count_matches(hay: &str, pat: &str) -> i32 {
        Regex::new(pat).map(|r| r.find_iter(hay).count() as i32).unwrap_or(-1)
    }

    #[repr(C)]
    pub struct Stats { pub total: i32, pub matched: i32 }

    pub fn analyze(text: &str) -> Stats {
        Stats { total: text.len() as i32, matched: count_matches(text, r\"\\d+\") }
    }
";

unit main() {
    log(count_matches(\"hi 42 yo 7\", \"\\d+\"));   // 2
    Stats s = analyze(\"a 1 b 2 c 3\");
    log(s.total);                                // 11
    log(s.matched);                              // 3
}
```

`rblock` takes a single raw string containing Rust source.  Multiple
`rblock` items in the same module are concatenated in source order;
multiple modules each get their own sidecar crate keyed by module path.

`rdep` adds a Cargo dependency line.  The right-hand side is a string
literal containing either a bare version (`"1.10"`) or an inline-table
fragment (`"{ version = \"1\", features = [\"derive\"] }"`) which is
spliced verbatim into the generated `Cargo.toml`.

Each `pub fn` declared in an `rblock` becomes a callable Maka function
in the surrounding module; each `pub struct` (when `#[repr(C)]`)
becomes a Maka `data` type; everything else is reachable as an opaque
`Rust<T>` handle (see §4).

---

## 2. Architecture

`makac` gains a Rust sidecar pipeline that runs after parse and before
sema:

```
makac app.maka
  ├─ parse + collect rblocks/rdeps per module
  ├─ for each module M with rblocks:
  │    sidecar = .maka_cache/rust/<sha256(rblocks + rdeps + rustc_version)>/
  │    if sidecar/.built absent:
  │        emit Cargo.toml from rdeps
  │        emit src/lib.rs from rblocks + auto-shims
  │        cargo build --release --manifest-path sidecar/Cargo.toml
  │        touch sidecar/.built
  │    register sidecar/target/release/libmaka_rust_<M>.a as --link input
  ├─ sig-parse each rblock → extract pub fn / pub struct
  ├─ inject Maka extern decls + data decls + Rust<T> markers into AST
  ├─ sema, codegen as normal
  └─ cc out.c <staticlibs> -o app
```

Cache key is a SHA-256 over the concatenation of `rblock` bodies (raw),
`rdep` lines (sorted), and the output of `rustc --version`.  A miss
re-runs cargo; a hit short-circuits straight to the link step.

---

## 3. The marshalling table

Every parameter and return type of every `pub fn` exposed to Maka is
classified into one of three bands:

### 3.1 Identity types

These pass through unchanged at the C ABI boundary:

| Rust                            | Maka            | Notes                       |
|---------------------------------|-----------------|-----------------------------|
| `i8`, `i16`, `i32`, `i64`       | `i8`..`i64`     | direct                      |
| `u8`, `u16`, `u32`, `u64`       | `u8`..`u64`     | direct                      |
| `isize`, `usize`                | `isize`, `usize`| direct                      |
| `f32`, `f64`                    | `f32`, `f64`    | direct                      |
| `bool`                          | `bool`          | direct (1 byte)             |
| `char`                          | `char`          | 4-byte Unicode scalar       |
| `*const T`, `*mut T`            | `*T`, `*mut T`  | raw pointers, unchecked     |
| `#[repr(C)] struct Foo { ... }` | `data Foo`      | identical layout            |
| `()`                            | `unit`          | direct                      |

`#[repr(C)] pub struct` declarations in rblocks are mirrored as Maka
`data` types in the same module, fields preserving order and type per
the marshalling table.  Maka source can then read and write fields
exactly like any other `data`.

### 3.2 Typed-shim types

These have direct Maka analogues; the shim performs a conversion at
the call boundary:

| Rust                | Maka              | Shim mechanism                                      |
|---------------------|-------------------|-----------------------------------------------------|
| `&str`              | `string`          | `(ptr, len)` via `from_raw_parts` + `from_utf8`     |
| `String`            | `String`          | `leak()` to ptr; Maka frees via `__maka_free_cstr`  |
| `&[T]`              | `[]T`             | `(ptr, len)` slice struct                            |
| `Vec<T>`            | `Vec<T>`          | `(ptr, len, cap)` struct; ownership transfers        |
| `Option<T>`         | `Option<T>`       | tagged struct `{tag: u8, value: T}` (Maka enum)     |
| `Result<T, E>`      | `Result<T, E>`    | tagged struct `{tag: u8, ok: T, err: E}`            |

The inner `T`/`E` of `Vec`, `Option`, `Result` is recursively
classified.  If any inner type is itself unmarshallable, the outer is
unmarshallable too (the shim cannot construct a tagged Maka enum
around an opaque payload without further machinery — that may relax
later).

### 3.3 Opaque handles

Anything not covered by §3.1 or §3.2 is wrapped as `Rust<T>` — an
opaque, owned heap handle.  Maka cannot inspect the bytes; it can only
pass the handle back to Rust functions in the same rblock.

| Rust          | Maka         | Shim mechanism                                       |
|---------------|--------------|------------------------------------------------------|
| `T` (any)     | `Rust<T>`    | `Box::into_raw(Box::new(t))` ↔ `Box::from_raw(ptr)`  |
| `&T`          | `&Rust<T>`   | borrow of the boxed pointer                          |
| `&mut T`      | `&mut Rust<T>`| mutable borrow                                       |

`Rust<T>` participates in Maka's existing `own` ownership tracking.
At scope exit Maka calls the generated `__maka_rust_drop_<mangled_T>`
shim, which executes `drop(Box::from_raw(ptr))` so the Rust destructor
runs.

`&Rust<T>` and `&mut Rust<T>` follow Maka's borrow-checker rules — the
borrow may not outlive the owner, may not escape via field stash, etc.
Across the boundary these are still raw pointers (lifetime info is
lost at the C ABI); Maka's borrow checker enforces the safety locally
without coordination with rustc.

Methods on opaque types (`impl T { pub fn foo(&self, ...) -> ... }`)
are auto-exposed: the Rust sig parser walks `impl` blocks and emits a
corresponding `extern "C"` shim per `pub fn` and a Maka function
signature taking `&Rust<T>` (or `&mut Rust<T>` / `own Rust<T>` for
`&mut self` / `self`).  These become callable as method calls on the
Maka side.

---

## 4. Generated shims

For each user `pub fn` the compiler emits a `#[no_mangle] pub extern
"C" fn __maka_shim_<name>` next to it.  The shim:

1. Receives flattened C-ABI parameters per the marshalling table.
2. Unmarshals each parameter into its native Rust form.
3. Wraps the call to the user function in `std::panic::catch_unwind`.
4. On `Ok(value)`: marshals the return value back to C ABI and
   returns.
5. On `Err(payload)`: writes the panic message to a process-static
   `__MAKA_PANIC_MSG` buffer and calls into the Maka-side
   `__maka_rust_panic(const char *msg)` runtime hook, which calls
   `panic()` (Maka's own).  The Rust side then aborts.

Example: for `pub fn count_matches(hay: &str, pat: &str) -> i32`:

```rust
#[no_mangle]
pub extern "C" fn __maka_shim_count_matches(
    hay_ptr: *const u8, hay_len: usize,
    pat_ptr: *const u8, pat_len: usize,
) -> i32 {
    let r = std::panic::catch_unwind(|| {
        let hay = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(hay_ptr, hay_len)).unwrap()
        };
        let pat = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(pat_ptr, pat_len)).unwrap()
        };
        count_matches(hay, pat)
    });
    match r {
        Ok(v) => v,
        Err(_) => unsafe {
            extern "C" { fn __maka_rust_panic(msg: *const u8) -> !; }
            __maka_rust_panic(b"Rust panic across FFI boundary\0".as_ptr())
        },
    }
}
```

For opaque-returning fns: `Box::into_raw(Box::new(result)) as *mut u8`.
For opaque-taking fns: `unsafe { &*(ptr as *const T) }`.

---

## 5. Auto-droppers for `Rust<T>`

For each opaque type `T` the compiler emits exactly one dropper per
sidecar:

```rust
#[no_mangle]
pub extern "C" fn __maka_rust_drop_<mangled_T>(ptr: *mut u8) {
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr as *mut T) };  // drop on scope exit
    }
}
```

The Maka codegen treats `Rust<T>` like `own *T`: at scope exit it
generates a call to the dropper.  Maka's existing
already-tracks-`own`-pointers infrastructure handles the lifecycle
(reassignment frees the old box, moves transfer ownership, borrows do
not free).

---

## 6. `Send` / `Sync` at thread-crossing sites

Maka's concurrency primitives (`spawn`, `share`, `transfer`) already
enforce `Shareable` / sendable bounds for native Maka types.  For
`Rust<T>` values, Maka delegates the check to rustc.

### 6.1 What ships today

The bridge emits an unconditional `Send` probe for every Rust type that
appears opaquely (as `Rust<T>`, `&T`, or `&mut T`) in any rblock
signature.  The probe lives in a `const _: () = { ... }` block at the
bottom of the generated `lib.rs`:

```rust
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<serde_json::Value>();
    assert_send::<Counter>();
    // ...
};
```

`cargo build` either accepts or rejects.  On rejection, the rustc error
surfaces verbatim (see §7), e.g.:

```
error[E0277]: `Rc<String>` cannot be sent between threads safely
   --> src/lib.rs:33:29
```

The check is over-conservative — it asserts `Send` for every exposed
opaque type whether or not Maka code actually crosses threads with
it.  In practice all common Rust crates expose `Send` types, so this
catches the genuinely dangerous cases (`Rc<T>`, `RefCell<T>` in
positions Maka would later spawn) without false positives on real
ecosystem code.

### 6.2 What's planned

Per-call-site probes (Send for `spawn`/`transfer`, Sync for `share`)
require sema to track `Rust<T>` type names through call-site analysis
and feed them back to the bridge before cargo builds.  The
architecture for that is:

```rust
const _: () = {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<serde_json::Value>();   // emitted at spawn / transfer sites
    assert_sync::<serde_json::Value>();   // emitted at share sites
};
```

with the driver scraping any rustc error and remapping back to the
Maka source line that did the thread-crossing:

```
maka error at app.maka:42: cannot `spawn` with `Rust<Rc<String>>`
  rustc: `Rc<String>` cannot be sent between threads safely
  (Rc is single-threaded; use Arc instead)
```

This loses no functionality — the unconditional check in §6.1 is a
strict superset of the per-call check — but trades better diagnostics
for a future sema pass.

---

## 7. Errors

### 7.1 Rust compile errors

`cargo build` failures are captured and surfaced verbatim — the rustc
error including its source span is printed under a header pointing at
the sidecar directory.  Line-by-line remapping back to the original
`.maka` file is planned but not yet wired (would require carrying
rblock byte offsets through the AST).

### 7.2 Cannot-cross-boundary errors

Caught at Maka's sig-parsing step, before cargo runs.  Examples:

```
maka error: cannot pass `HashMap<String, i32>` across the Rust boundary
  at app.maka:42, in rblock function `parse_config`
  Rust types must be marshallable (primitive, &str, String, &[T], Vec<T>,
  Option<T>, Result<T,E>, raw pointer, #[repr(C)] struct) or referenced as
  an opaque `Rust<T>` handle.  Hint: change the parameter to
  `&HashMap<String, i32>` and the Maka side will see `&Rust<HashMap<...>>`.
```

### 7.3 Panic at runtime

A Rust panic across the shim does not unwind — it's caught and routed
to Maka's `panic()` builtin, which prints the panic message to stderr
and `abort()`s.  No process state is left in a torn-down half-Rust
half-Maka condition.

---

## 8. Build pipeline integration

The driver gains two flags:

- `--rust-profile=release|dev` — selects the cargo profile.  Default
  `release` for `makac app.maka`, `dev` when `--debug` is also passed.
- `--no-rust` — refuse to build sidecars; fail if any rblock is
  present.  Useful for CI environments without Rust installed.

The sidecar lives under `.maka_cache/rust/<sha256>/` rooted at the
project's working directory (or `$MAKA_CACHE` if set).  The driver
detects rustc / cargo at PATH; missing tools produce a clear error
listing the install command (`rustup` / distro package).

Sidecar `Cargo.toml` boilerplate:

```toml
[package]
name = "maka_rust_<module>"
version = "0.0.0"
edition = "2021"

[lib]
crate-type = ["staticlib"]
path = "src/lib.rs"

[profile.release]
panic = "abort"      # rustc panics handled by our catch_unwind, not by Cargo
opt-level = 3
lto = false

[dependencies]
<rdep lines>
```

`panic = "abort"` is deliberate — the catch_unwind wrapper handles
unwinds programmatically, so Cargo's abort-on-panic gives faster
codegen without losing the safety story.

---

## 9. Hard limits

Honest list of things this design does NOT do:

- **Generics across the boundary.**  A Rust `pub fn read<T:
  Deserialize>(...)` is not callable directly from Maka — you must
  monomorphize on the Rust side (`pub fn read_user(...) -> User`).
- **Trait objects.**  `Box<dyn Trait>` survives as an opaque handle but
  Maka cannot call trait methods through it; you write an rblock
  helper that does the dispatch.
- **Lifetimes across the boundary.**  Neither rustc nor Maka can
  verify that a borrow handed to Rust outlives Rust's use.  Same
  caveat as C FFI; Maka's borrow checker enforces locally.
- **`async fn`.**  No shared executor.  Wrap with a blocking shim
  inside the rblock (`fn blocking_x(...) { runtime.block_on(x(...)) }`)
  and expose that.
- **Custom allocators / `#[global_allocator]`.**  The sidecar and the
  Maka-emitted C share libc malloc / free.  An rblock setting its own
  global allocator could double-free Maka-owned strings; not enforced
  yet, document-only.
- **Cargo-feature unification across rblocks.**  Each module gets its
  own sidecar; identical deps in different modules build twice.  Could
  be unified with a workspace later; v1 keeps it simple.
- **First build is slow.**  `cargo build` of a small graph is seconds;
  a graph touching `serde`/`tokio`/`regex` is minutes.  Subsequent
  builds hit the cache and are instant.

---

## 10. Worked examples

### 10.1 Primitive in, primitive out

```maka
rblock "
    pub fn add(a: i32, b: i32) -> i32 { a + b }
";

unit main() { log(add(2, 3)); }   // 5
```

### 10.2 `&str` and `String`

```maka
rblock "
    pub fn shout(s: &str) -> String { s.to_uppercase() }
";

unit main() { log(shout(\"hi\")); }   // HI
```

### 10.3 `#[repr(C)]` struct as a Maka `data`

```maka
rblock "
    #[repr(C)] pub struct V2 { pub x: f64, pub y: f64 }
    pub fn add(a: V2, b: V2) -> V2 { V2 { x: a.x + b.x, y: a.y + b.y } }
";

unit main() {
    V2 a = { x = 1.0, y = 2.0 };
    V2 b = { x = 3.0, y = 4.0 };
    V2 c = add(a, b);
    log(c.x); log(c.y);            // 4.0  6.0
}
```

### 10.4 Opaque handle for a non-marshallable type

```maka
rdep regex = "1";

rblock "
    use regex::Regex;
    pub fn compile(p: &str) -> Regex { Regex::new(p).unwrap() }
    pub fn matches(r: &Regex, s: &str) -> bool { r.is_match(s) }
";

unit main() {
    Rust<Regex> r = compile(\"\\\\d+\");
    log(matches(&r, \"abc 42\"));   // true
    // r is auto-dropped at scope exit
}
```

### 10.5 `Option<T>` round-trip

```maka
rblock "
    pub fn find_digit(s: &str) -> Option<i32> {
        s.chars().find_map(|c| c.to_digit(10).map(|d| d as i32))
    }
";

unit main() {
    match (find_digit(\"abc7def\")) {
        Option.Some{value} log(value),    // 7
        Option.None        log(-1),
    }
}
```

### 10.6 Thread-crossing with Send check

```maka
rblock "
    use std::sync::Arc;
    pub fn make_shared(s: String) -> Arc<String> { Arc::new(s) }
    pub fn read(a: &Arc<String>) -> String { (**a).clone() }
";

unit main() {
    Rust<Arc<String>> shared = make_shared(\"hello\");
    *Thread t = spawn() { log(read(&share shared)); };  // Sync probe inserted
    join(t);
}
```

If the user wrote `Rust<Rc<String>>` instead, the sidecar's
`assert_send` (or `assert_sync` for `share`) probe would fail at
`cargo build`, and the driver remaps the diagnostic to the `spawn`
line.

---

## 11. Implementation phases

The implementation lands in phases corresponding to the task list in
the source tree:

1. **Lexer / AST / parser** — `rblock`, `rdep` tokens + items.
2. **Mini Rust sig parser** — extract `pub fn` / `pub struct` / `impl`.
3. **Sidecar emission** — `Cargo.toml` + `src/lib.rs` with shims.
4. **Cargo orchestration + cache** — content-addressed sidecar cache.
5. **Extern + data injection** — splice generated Maka items into AST.
6. **Opaque `Rust<T>`** — type-system marker, dropper, ownership.
7. **Send/Sync probes** — thread-crossing assertions in sidecar.
8. **Tests** — covering each marshalling band, opaque, panic, Send.

The repository tracks one task per phase under `TaskList`; each is
closed by a self-contained commit.

---

End of spec.
