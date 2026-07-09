# Contributing

How to build, test, and change the Maka compiler. For how it is laid out, read
`ARCHITECTURE.md`; for language semantics, `SPEC.md`.

## Dev loop

```sh
cargo build                    # debug build -> target/debug/{makac,maka}
                               # (unused-mut / dead-code warnings are ambient noise;
                               #  a new *error* is not)

# compile + run one program
./target/debug/makac path/to/prog.maka -o /tmp/prog --run

# see the generated C (best debugging tool)
./target/debug/makac path/to/prog.maka --emit-c -o /tmp/prog.c

# the two gates (must both be FAIL-free before a commit)
bash tests/run_all.sh          # positive suite: compile+run, stdout must match .expected
bash tests/run_neg.sh          # negative suite: every neg_*.maka MUST fail to compile
```

`run_all.sh` runs each program with `makac --run 2>&1`, so **a compile WARNING is
captured into stdout and breaks the `.expected` match** - a test that emits a
warning is failing. Both suites build `target/debug/makac` first; a "hang" is
usually just the debug rebuild - confirm with a single `timeout` run of a release
binary. These two suites are the **local** gate - run them before every commit.
CI (`.github/workflows/ci.yml`) covers something different: cross-platform
concurrency smoke tests (custom fiber context-switch + explicit-Pool M:N migration)
on Linux x86_64 + aarch64/qemu, Windows, and macOS - so it does NOT catch a broken
`.expected`; that is on you locally.

Emit temporary binaries to **`/tmp`** (`-o /tmp/foo`), never the repo root - stray
`makac -o NAME` outputs have polluted the root before.

## Where new code goes

Make each change in the smallest crate it can live in (the phases are in
`ARCHITECTURE.md`):

- a new **token** -> `lexer`; a new **syntax shape** -> `ast` (+ `parser` to accept
  it); a **grammar** change -> `parser`.
- **resolution / type check / lifetime / monomorphization** (almost all language
  features) -> `sema` (`resolve.rs` for name+sig collection, `typeck.rs` for the
  body check, `lifetime.rs` for ownership/null-proof, `lib.rs` for the
  instantiation fixpoint).
- **C emission** -> `codegen`. A new `HExprKind`/`HStmt` must be handled in every
  exhaustive match there and in `sema/lifetime.rs` (the compiler lists them for you).
- **stdlib** -> `stdlib/std.maka` (it is real Maka source; add an `import` path).
- **Rust FFI** -> `bridge` (+ `RUST_INTEROP.md`).

Adding a whole language feature typically touches `ast` (shape) -> `parser`
(grammar) -> `sema` (check + lower to HIR) -> `codegen` (emit C), plus a test. A
feature that only changes checking (no new syntax) stays inside `sema`.

## Tests

- Positive: `tests/programs/NN_short_name.maka` + `NN_short_name.expected` - must
  compile, run, and stdout-match. Optional sidecars: `.deps` (extra `.maka` to
  compile alongside), `.link` (C files / `-l` flags), `.stdin` (piped in), `.cgrep`
  (assert patterns present/`!absent` in the emitted C).
- Negative: `tests/programs/neg_NN_short_name.maka` - must FAIL to compile (any
  non-zero exit; the rejection reason is shown but not asserted).
- For ownership / lifetime / drop / codegen changes, also sanity-check under a
  sanitizer: `makac --emit-c` then `cc -fsanitize=address,undefined`, run, and check
  for leaks. Verify the non-happy path, not just the success case.

## Conventions

- Commits: conventional, descriptive (`feat(lang):`, `fix(sema):`, `docs(spec):`);
  small and focused; plain ASCII (no em/en dashes).
- If a change is user-visible, update `SPEC.md` in the SAME commit (the relevant
  numbered section for behavior; §17 for a newly opened/closed gap). If it changes
  structure, a flow, a boundary, or the dev loop, update `ARCHITECTURE.md` /
  `CONTRIBUTING.md` too.
- Run `makac lint src/*.maka` for `.maka` naming/style (`STYLE_GUIDE.md`); it gates CI.

## The `maka` project tool

`maka` (crate `cli`) is the day-to-day front-end and shells out to `makac`:
`maka new NAME`, `maka build [--release]`, `maka run`, `maka test`, `maka fmt`,
`maka lint`, `maka add NAME`. A project is a directory with a `maka.toml`.
`./install.sh` puts both binaries on PATH.
