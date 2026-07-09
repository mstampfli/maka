# maka_lint

Naming and style checks over the AST. Rules: `STYLE_GUIDE.md`.

- **Job:** parse `.maka` files and flag naming/style violations (`makac lint ...`,
  `maka lint`). Exits non-zero on any issue, so it gates CI.
- **DAG:** depends on `lexer`, `ast`, `parser`. Depended on by `driver`, `lsp`.
- **Never:** change code or block compilation - it is advisory/gating only, entirely
  separate from the compile path (`sema`/`codegen` do not consult it).
