# Maka for VS Code

Language support for [Maka](https://github.com/mstampfli/maka):

- **Syntax highlighting** (TextMate grammar) — keywords, types, the five pointer
  flavors, strings/chars/numbers, comments, function names; `rblock "..."` bodies
  highlight as **Rust** and `cblock "..."` bodies as **C**. Works with no server.
- **Language server** (`maka-lsp`) — live **diagnostics** (parse + type errors),
  **hover** types/signatures, **go-to-definition**, **document outline**,
  **completion**, references/rename, and **Format Document** (layout formatter,
  works with format-on-save). It links the compiler crates, so it reflects what
  the compiler actually sees, and it caches project analysis so hover and
  go-to-definition stay fast on repeat lookups.

## Install (GUI, recommended)

Build a self-contained `.vsix` (it bundles the `maka-lsp` server, so there is
nothing else to configure), then install it from the Extensions GUI:

```sh
cd editors/vscode
./package.sh                       # builds maka-lsp --release, bundles it, makes maka-<v>.vsix
```

In VS Code:

1. Open the **Extensions** view (Ctrl+Shift+X).
2. Click the **`...`** (More Actions) menu at the top of the panel.
3. **Install from VSIX...**, and pick `editors/vscode/maka-<version>.vsix`.
4. Reload if prompted.

Open any `.maka` file: highlighting is instant and the language server (bundled)
provides diagnostics, hover, go-to-definition, the outline, completion,
references/rename, formatting, and style lints - no PATH or settings needed.

**Formatting.** Run **Format Document** (Shift+Alt+F), or enable format-on-save
(`"editor.formatOnSave": true`).  It is a layout formatter: it fixes indentation
(4-space, by brace depth), strips trailing whitespace, and normalizes blank
lines, while leaving comments and strings untouched.  It never reflows code and
verifies its own output is token-identical before applying, so it is safe on
save.  Naming conventions stay a lint (with an LSP Rename to fix them
deliberately), not a silent rewrite.  The same formatter is on the CLI as
`maka fmt` / `makac fmt [--check]`.

## Install (developer symlink)

For live iteration on the extension itself:

```sh
cd editors/vscode
npm install                        # fetches vscode-languageclient
ln -s "$PWD" ~/.vscode/extensions/maka
cargo build --release -p maka_lsp  # or set maka.server.path to a debug build
```

Reload VS Code. If no server is bundled in `bin/`, set
`"maka.server.path": "/path/to/maka/target/release/maka-lsp"` or put it on PATH.

Syntax highlighting alone needs no server; set `"maka.server.enabled": false` to
skip it.

## Settings

- `maka.server.path` — path to the `maka-lsp` binary (default: `maka-lsp` on PATH).
- `maka.server.enabled` — run the language server (default: `true`).

`rblock` and `cblock` functions resolve like any other: the server runs the same
Rust-bridge signature extraction the compiler uses (guarded so a file without an
rblock pays nothing), and `cblock` functions resolve through their `extern`
declaration.
