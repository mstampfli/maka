# maka_driver

The **`makac`** binary: orchestrates the compile pipeline and invokes `cc`.

- **Job:** parse CLI args; parse each input file (`parser`); run
  `bridge::prepare` -> `sema::analyze` -> `bridge::finish` -> `codegen::emit`; write
  the C (or `--emit-c`); invoke the system `cc` with the link line (staticlibs, C
  deps, libm, platform libs); optionally `--run`. Embeds `stdlib/std.maka` via
  `include_str!`. Owns `#line`/module-file mapping for debug info.
- **DAG:** the top of the compiler DAG - depends on every compiler crate (`parser`,
  `sema`, `codegen`, `bridge`, `lint`, `fmt`, `ast`, `lexer`). Nothing depends on it.
- **Never:** contain language semantics - it wires phases together and shells to the
  C toolchain. Correctness belongs in `sema`, lowering in `codegen`. Full flow +
  entry point: `ARCHITECTURE.md`.
