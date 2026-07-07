# Maka tooling backlog (linter / LSP / CLI)

Standing goal: make the linter and the language server **excellent**. Items below
are queued; check them off as they land.

## Editor / LSP

- [x] **Stdlib symbols resolve.** The embedded stdlib is materialized to a cache
      file and its top-level symbols are indexed, so source-defined names
      (`String`, `Option`, `str_len`, `string_new`, ...) hover and go-to-definition
      into the stdlib source. Compiler builtins with no source (`Vec`, `push`,
      `pop`, `log`, `format`, `thread`/`spawn`/`job`/`join`, `tag`, `fields`, ...)
      resolve via a `BUILTINS` hover/completion table.
- [ ] **Doc comments.** A standard way to document functions (and types) that the
      server shows on hover. Pick a syntax (e.g. `///` doc lines above the decl,
      or `//!` for module docs), have the lexer/parser retain them attached to the
      following item, and render them in hover markdown below the signature.
- [ ] **Signature help.** While typing a call's arguments, show the callee's
      parameter list and highlight the parameter currently being entered
      (`textDocument/signatureHelp`, trigger on `(` and `,`).
- [x] **Enum hover polish.** The "enum X enum" duplicate no longer reproduces.
      Cross-file enum hover now shows the full variant list (with payload field
      names) via a shared `enum_signature`, matching the in-file path, and the
      enum name is go-to-definition navigable.
- [ ] **Enum variants navigable.** Variants (subtypes) should be go-to-definition
      targets and hoverable, and hovering the enum should surface them.
- [ ] **General polish pass.** Make the linter + LSP "perfect": tighten hover
      output for every symbol kind, scope-aware references/rename, semantic
      tokens, quick-fixes for lint findings, etc.

## Language

- [ ] **Explicit generic type arguments at call sites.** Inference from arguments
      is the default and works well (`total(&table)`, `render(&p)` bind `T` from
      the argument's concrete type). But when `T` appears only in the return type
      or nowhere in the parameters (a constructor like `empty<int>()`), inference
      has nothing to work from and there is no escape hatch. Verified today:
      `id<int>(5)` does NOT parse - `<` is read as less-than ("ordering on
      non-numeric types"), exactly the grammar ambiguity to design around. Add an
      explicit form (a `::<>` turbofish, `id::<int>(5)`, avoids the ambiguity)
      as an optional override; keep argument inference as the default.

## CLI

- [x] **`maka` not found in the terminal.** Fixed: `install.sh` builds release
      and installs both `maka` and `makac` to `~/.cargo/bin` (or `$BINDIR`), so
      `maka build/run/test/fmt/lint` work from any project directory. Documented
      in the README.
