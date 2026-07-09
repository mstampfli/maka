# maka_ast

The surface-syntax data types.

- **Job:** define the AST - `Expr`, `Stmt`, `Type`, `Item`, `FuncDecl`, `DataDecl`,
  `AttrInfo`, etc. - plus small shape helpers (span accessors, `_`-placeholder
  substitution). This is the interface between `parser` (produces it) and `sema`
  (consumes it).
- **DAG:** depends on `lexer` (for `Span`). Depended on by `parser`, `sema`,
  `codegen`, `bridge`, `lint`, `driver`, `lsp`.
- **Never:** contain resolution, type, or lowering logic - only the syntactic shape
  and trivial helpers over it. Adding a new node here means updating every
  exhaustive match that walks `Expr`/`Stmt` across the workspace.
