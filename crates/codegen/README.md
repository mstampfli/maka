# maka_codegen

Typed HIR -> a portable C translation unit (a `String`).

- **Job:** lower every HIR construct to C - structs/enums, the pointer families,
  `dyn`/`some` fat pointers + vtables, closures, the recursive drop glue, the
  runtime prologue, and `#line` directives for debugging. `emit_with_debug()` is the
  entry; `emit_freestanding()` strips the libc prologue.
- **DAG:** depends on `lexer`, `ast`, `sema`. Depended on by `driver`.
- **Never:** decide correctness or see a generic. It trusts the HIR is well-typed
  and fully monomorphized; a `TyVar`/`GenericPattern` or an unresolved
  placeholder `FuncId` reaching here is an upstream (`sema`) bug. A new
  `HExprKind`/`HStmt` must be handled in its exhaustive matches (`emit_expr`,
  `scan_expr`, and the place/mutation walkers).
