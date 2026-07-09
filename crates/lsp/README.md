# maka_lsp

The language-server front-end (editor integration).

- **Job:** serve editor features (diagnostics, etc.) by reusing the real compiler
  crates - it parses/resolves/type-checks with the same `parser`/`sema` a compile
  uses, so editor feedback matches `makac`. Embeds `stdlib/std.maka` via
  `include_str!` like the driver.
- **DAG:** depends on `parser`, `sema`, `lint`, `fmt`, `bridge`, `ast`, `lexer`.
  Nothing depends on it (a binary front-end).
- **Never:** fork the language rules - it consumes the same `sema` as compilation so
  the editor never disagrees with the compiler.
