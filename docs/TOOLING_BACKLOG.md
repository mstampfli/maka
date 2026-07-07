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
- [x] **Doc comments.** `///` lines directly above a declaration are shown on
      hover (below the signature), for user code, cross-file, and the stdlib.
      Implemented tooling-side (they are ordinary comments to the compiler, so no
      lexer/parser/AST change); documented in the STYLE_GUIDE; seeded docs on key
      stdlib items (Option, String, str_len, str_eq, str_dup, read_file,
      int_to_str, string_new/from). Fuller stdlib docs are ongoing.
- [x] **Signature help.** Typing a call's arguments now shows the callee's
      signature and highlights the active parameter (triggered on `(` and `,`).
      Token-based enclosing-call detection (ignores commas/parens in strings and
      comments), prefers the open file's own function over a same-named stdlib
      overload. (Builtins without a `FuncSig`, e.g. `push`, don't show it yet.)
- [x] **Enum hover polish.** The "enum X enum" duplicate no longer reproduces.
      Cross-file enum hover now shows the full variant list (with payload field
      names) via a shared `enum_signature`, matching the in-file path, and the
      enum name is go-to-definition navigable.
- [x] **Enum variants navigable.** Each variant is indexed, so `Color.Red` hovers
      (as `Color.Red`, with payload field names) and goes to its declaration.
      Hovering the enum already lists the variants (`enum_signature`).
- [x] **Semantic tokens.** Token-accurate identifier classification (type /
      function / variable / property / enumMember) that the TextMate grammar
      cannot infer; keywords/strings/numbers/comments stay TextMate-highlighted.
- [ ] **General polish pass.** Remaining: scope-aware references/rename,
      quick-fixes for lint findings, richer field-declaration semantic tokens.

## Language

- [x] **Explicit generic type arguments at call sites.** Added the `::<>`
      turbofish (`identity::<int>(42)`, `empty_vec::<int>()`) - the `::` sidesteps
      the `<`-as-less-than ambiguity. When present the type arguments bind the
      callee's type parameters directly (bypassing inference); arity, non-generic
      misuse, and argument-type mismatches are all diagnosed. Inference stays the
      default. Parser + AST + typeck; tests `439_turbofish`, `neg_137`, `neg_138`.

## Debugger

- [x] **Readable variable names.** Codegen emits each local with its Maka name
      (`a`, `total`), suffixing `_<id>` only on a genuine clash; gdb/lldb/VS Code
      show clean names from DWARF and the generated C is readable. (Globals still
      carry the `__maka_global__` prefix; Maka-aware pretty-printing of
      `Vec`/`String`/enums is a follow-up.)

## CLI

- [x] **`maka` not found in the terminal.** Fixed: `install.sh` builds release
      and installs both `maka` and `makac` to `~/.cargo/bin` (or `$BINDIR`), so
      `maka build/run/test/fmt/lint` work from any project directory. Documented
      in the README.
- [x] **`maka add`.** Now edits source instead of only guiding. Since a `rdep`
      must sit in a module with an `rblock` (sidecars are per-module), `maka add
      NAME [VER]` inserts `rdep NAME = "VER";` beside the project's rblock when
      there is exactly one, scaffolds a compilable FFI module (`src/NAME.maka`
      with module + rdep + rblock skeleton) when there is none, and lists the
      candidates when there are several. Detects an already-present dep.
