# maka_fmt

A layout-only source formatter.

- **Job:** reprint `.maka` source with canonical layout (`makac fmt` / `maka fmt`),
  driven off the token stream so it reformats whitespace/indentation without needing
  a full parse.
- **DAG:** depends on `lexer` only. Depended on by `driver`, `lsp`.
- **Never:** change program meaning - formatting is purely presentational (tokens in,
  reprinted tokens out); it must round-trip semantics exactly.
