# maka_parser

Tokens -> AST (recursive descent).

- **Job:** own the grammar. Turn a `maka_lexer` token stream into a `maka_ast`
  module. Handles precedence, the `own *T`/`raw *T` pointer forms, generics vs
  comparison disambiguation (speculative parse with position save/restore), and the
  `dyn (A + B)` / `T has X` shapes.
- **DAG:** depends on `lexer`, `ast`. Depended on by `lint`, `driver`, `lsp`.
- **Never:** resolve names, check types, or reject on meaning - only on grammar. A
  program that parses but is nonsensical is `sema`'s to reject. Errors are returned
  as `ParseError` (no global error list), so speculative parsing can backtrack.
