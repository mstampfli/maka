# Maka tooling backlog (linter / LSP / CLI)

Standing goal: make the linter and the language server **excellent**. Items below
are queued; check them off as they land.

## Editor / LSP

- [ ] **Stdlib symbols resolve.** Hover and go-to-definition should work for
      stdlib types and functions (Vec, String, Option, push, str_len, ...), not
      just user code. Today stdlib names get hover-only or nothing; index the
      parsed stdlib into the symbol index (with a synthetic URI into the embedded
      source, or a real path) so they resolve and navigate.
- [ ] **Doc comments.** A standard way to document functions (and types) that the
      server shows on hover. Pick a syntax (e.g. `///` doc lines above the decl,
      or `//!` for module docs), have the lexer/parser retain them attached to the
      following item, and render them in hover markdown below the signature.
- [ ] **Signature help.** While typing a call's arguments, show the callee's
      parameter list and highlight the parameter currently being entered
      (`textDocument/signatureHelp`, trigger on `(` and `,`).
- [ ] **Enum hover polish.** Hover on an enum currently prints "enum X enum"
      (duplicated keyword) - fix the formatting. Hover should also list the
      enum's variants.
- [ ] **Enum variants navigable.** Variants (subtypes) should be go-to-definition
      targets and hoverable, and hovering the enum should surface them.
- [ ] **General polish pass.** Make the linter + LSP "perfect": tighten hover
      output for every symbol kind, scope-aware references/rename, semantic
      tokens, quick-fixes for lint findings, etc.

## CLI

- [ ] **`maka` not found in the terminal.** The `maka` front-end binary is not on
      PATH after a normal build. Provide an install path (a `maka install` /
      documented `cargo install --path crates/cli`, or a symlink target) so
      `maka build/run/test/fmt/lint` work from any project directory.
