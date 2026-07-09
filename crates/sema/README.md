# maka_sema

AST -> typed HIR. The semantic core: every correctness rule lives here.

- **Job:** name resolution, type checking, generic monomorphization, and the
  lifetime/ownership/null-proof pass. `analyze()` is the entry (`lib.rs`).
  - `resolve.rs` - `SymTab::collect`: gather types, enums, function sigs, `has`-impls,
    globals; the trait-satisfaction/visibility predicates.
  - `hir.rs` - the typed IR (`HType`, `HExpr`/`HExprKind`, `HStmt`, `SymTab`,
    `FuncSig`, `HasImpl`, `AttrInfo`).
  - `typeck.rs` - the body check: expression/statement typing, method/overload
    resolution, `dyn`/`some` dispatch, `inline for`/`if (T has X)` folding, and
    recording generic `InstantiationReq`s.
  - `lib.rs` - the monomorphization fixpoint: drain instantiation requests, re-check
    each concrete instantiation, rewrite placeholder `FuncId`s.
  - `lifetime.rs` - move/auto-null/poison, `*T` non-null proof, borrow-escape, drop
    elaboration.
- **DAG:** depends on `lexer`, `ast`. Depended on by `codegen`, `driver`, `lsp`.
- **Never:** emit C, or let a generic escape - the HIR handed downstream is fully
  concrete (no `HType::TyVar`/`GenericPattern` in a real instantiation). Generic
  bodies are checked ONLY per concrete instantiation, never on the abstract template.
- **Invariant home:** the "no implicit heap", "deref needs proof", "own\* auto-nulls /
  own& poisons", and "T-satisfies-X" rules are all enforced here (see ARCHITECTURE.md).
