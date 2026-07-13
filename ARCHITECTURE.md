# Architecture

How the Maka compiler works. Language semantics live in `SPEC.md` (the source of
truth); this file is how the *implementation* is laid out and how a change flows
through it.

## Theory of operation

`makac` is a classic multi-pass compiler that lowers Maka source to **portable C**
and hands the C to a system `cc`. Source becomes tokens, tokens become an AST, the
AST is resolved into a typed HIR (name resolution, type checking, generic
monomorphization, and a lifetime/ownership pass all live here), and the HIR is
emitted as one self-contained C translation unit. Optional inline Rust (`rblock`)
is compiled to a sidecar staticlib and linked in. There is no bytecode and no VM:
every Maka construct has a direct C lowering, and the ownership rules the language
promises are enforced statically in the `sema` crate, never at run time.

## Codemap

One workspace, eleven crates (`crates/*`), three binaries (`makac` = driver,
`maka` = cli, and the LSP server). Each crate owns one phase.

| crate | responsibility (owns X / never does Y) |
|-------|-----------------------------------------|
| `lexer` | source text -> tokens + spans. Owns lexical structure; knows nothing about grammar. |
| `ast` | the surface-syntax data types (`Expr`, `Stmt`, `Type`, `Item`, ...) + AST-level helpers. No logic beyond shape. |
| `parser` | tokens -> AST (recursive descent). Owns grammar; no name/type resolution. |
| `sema` | AST -> typed **HIR**: name resolution, type check, generic monomorphization, and the lifetime/ownership/null-proof pass. The semantic core; owns all correctness rules. |
| `codegen` | HIR -> a C string. Owns C lowering + the runtime prologue; assumes the HIR is already well-typed and concrete (no generics survive to here). |
| `bridge` | the Rust FFI (`rblock`/`rdep`) sidecar: parse inline Rust, mirror types into the AST (pre-sema), build the sidecar crate + `Send`/`Sync` probes (post-sema). See `RUST_INTEROP.md`. |
| `lint` | naming/style checks over the AST (`STYLE_GUIDE.md`); gates CI, changes nothing. |
| `fmt` | layout-only source formatter (tokens -> reprinted source). |
| `driver` | the **`makac`** binary: orchestrates the phases above, then invokes `cc`. Entry point of compilation. |
| `cli` | the **`maka`** binary: a cargo-like project front-end (`maka new/build/run/test/fmt/lint/add`). Std-only; shells out to `makac`. |
| `lsp` | language-server front-end (editor integration) reusing `parser`/`sema`/`lint`/`fmt`. |

Non-crate homes of truth: `stdlib/std.maka` (the standard library, **real Maka
source** embedded into the binary via `include_str!` in `driver` and `lsp`),
`SPEC.md` (language semantics), `tests/programs/*` (the end-to-end suites).

## Dependency DAG and boundaries

Dependencies point strictly downward; there are no cycles.

```
lexer  <- ast <- parser <- lint
   ^       ^  \      \
   |       |   \      +--- sema <- codegen
   |       |    \            \
  fmt     bridge \            \
                  +----------- driver (makac) --> invokes `cc`
                              /  |  |  |  |  |
        (parser sema codegen bridge lint fmt ast lexer)
   lsp --> (parser sema lint fmt bridge ast lexer)
   cli --> (nothing; shells out to the makac binary)
```

- **`sema` is the only crate that decides correctness.** `parser` accepts anything
  grammatical; `codegen` trusts its input. A rejected program is rejected in `sema`.
- **`codegen` never sees a generic.** All monomorphization happens in `sema`; a
  `HType::TyVar`/`GenericPattern` reaching codegen is a bug (codegen skips templates
  by `type_params.is_empty()`).
- **`cli` does not depend on the compiler crates** - it is a thin process wrapper,
  so the project front-end and the compiler version independently.
- **The C boundary:** `codegen` emits C text; `driver` owns the `cc` invocation and
  link line. Generated C is the only artifact that crosses into the C toolchain.

## Invariants (and where each is enforced)

- **No implicit heap; `alloc` only lands in an owning slot** (`own *T`/`own &T`).
  Enforced in `sema/typeck.rs` at the `HeapAlloc` check.
- **A nullable deref (`p!`) needs a compile-time non-null proof** - no runtime null
  check exists. Enforced by the lifetime pass (`sema/lifetime.rs`); the deref is
  simply not emitted without proof.
- **Ownership is move-based; `own *T` auto-nulls on move, `own &T` and owning values
  poison.** Moves funnel through the single choke point `sema/lifetime.rs::mark_moved`
  (SPEC 6.1); the runtime `= NULL` for an auto-null is emitted at the move site by
  `codegen`.
- **A `has Move` / `has Drop` VALUE type is affine too** (move-only, use-after-move
  is an error), and its `drop` runs at scope exit - the stack destructor, no heap
  (SPEC 6.4c). `has Drop` implies `Move` via the prelude supertrait `attr Drop: Move`.
  Affine-ness is decided by `sema/lifetime.rs::nominal_affine_marked` -> `ty_owns_heap`
  (so the existing move-checker and drop pass engage unchanged); the heap-block *free*
  stays gated on `own *`/`heap` pointers, so a bare stack value is drop-glued but never
  freed.
- **`sema` and `codegen` agree on "does T satisfy trait X" by construction** - both
  route through the `has`-impl registry helpers (`resolve::type_impls_trait_visible`,
  `underlying_struct_key`, `has_impl_visible`); `if (T has X)`, `where T has X`, and
  generic bounds use the same predicate. Supertraits (`attr Sub: Super`) are honored by
  all of them via `resolve::attr_has_supertrait` (transitive), plus a Pass-3b closure
  that mirrors `trait_impls[Sub]` into `trait_impls[Super]`.
- **Builtins live in a reserved `FuncId` range** (`u32::MAX - N`, ~1024 slots) and
  generic-instantiation placeholders below `PLACEHOLDER_FID_BASE` (both in
  `sema/lib.rs`); codegen dispatches builtins via `is_builtin_sentinel`. Never mint a
  real `FuncId` in that range.
- **The stdlib is ordinary Maka**, compiled through the same pipeline as user code -
  there is no privileged stdlib path. It is embedded from `stdlib/std.maka`.

## Key flows

**Compilation (the load-bearing path), `driver/src/main.rs::main`:**
```
parse each file (maka_parser)                     -> AST module
  -> rust_bridge::prepare      (mirror rblock types into the AST, pre-sema)
  -> maka_sema::analyze                            -> typed HIR
        = SymTab::collect (resolve.rs)             (names, sigs, has-impls)
          -> instantiation fixpoint (lib.rs)       (monomorphize; per-instantiation
             = TypeChecker::check_func_with_id      typeck.rs body check + lifetime)
          -> lifetime/null-proof pass (lifetime.rs)
  -> rust_bridge::finish       (build sidecar staticlib + Send/Sync probes)
  -> maka_codegen::emit_with_debug                 -> C string
  -> write temp .c, Command::new(cc) ...           -> executable
  -> (--run) execute
```

**Generic instantiation:** a call to `f<T>` records an `InstantiationReq` (with the
concrete args + caller module) in `typeck.rs`; the `analyze` fixpoint in `lib.rs`
drains requests, mangles a concrete `FuncSig`, re-checks the body with `subst`
mapping `T`->concrete, and rewrites placeholder `FuncId`s to the real ones. Generic
bodies are checked ONLY per concrete instantiation (never on the abstract template),
which is why `inline for (fields)` and `if (T has X)` can fold per-monomorphization.

**Rust FFI:** `bridge::prepare` (pre-sema) sig-parses `rblock`s and injects extern
decls + mirrored `#[repr(C)]` data; `bridge::finish` (post-sema) emits the sidecar
`Cargo.toml`/`lib.rs` (with `Send`/`Sync` assertion probes from `sema`), runs
`cargo build`, and returns staticlib paths for `driver` to link. Full spec:
`RUST_INTEROP.md`.

## Entry points

- **`makac`** (compiler): `crates/driver/src/main.rs::main`.
- **`maka`** (project tool): `crates/cli/src/main.rs::main` (shells to `makac`).
- **language server:** `crates/lsp/src/main.rs`.
- **stdlib:** `stdlib/std.maka` (edited as Maka source; recompiled every build).

See `CONTRIBUTING.md` for the build/test dev loop and where new code goes.
