# Maka for VS Code

Language support for [Maka](https://github.com/mstampfli/maka):

- **Syntax highlighting** (TextMate grammar) — keywords, types, the five pointer
  flavors, strings/chars/numbers, comments, function names; `rblock "..."` bodies
  highlight as **Rust** and `cblock "..."` bodies as **C**. Works with no server.
- **Language server** (`maka-lsp`) — live **diagnostics** (parse + type errors),
  **hover** types/signatures, **go-to-definition**, **document outline**, and
  **completion**. It links the compiler crates, so it reflects what the compiler
  actually sees.

## Setup

1. Build the compiler and the server:

   ```sh
   cd /path/to/maka
   cargo build --release          # produces target/release/maka-lsp (and makac, maka)
   ```

2. Make `maka-lsp` findable. Either add `target/release` to your `PATH`, or set
   the path in VS Code settings:

   ```json
   "maka.server.path": "/path/to/maka/target/release/maka-lsp"
   ```

3. Install this extension's client dependency and load it:

   ```sh
   cd editors/vscode
   npm install                    # fetches vscode-languageclient
   ln -s "$PWD" ~/.vscode/extensions/maka
   ```

   Reload VS Code (Ctrl+Shift+P -> "Developer: Reload Window"). Open any `.maka`
   file: highlighting is immediate, and the server features come online once
   `maka-lsp` is found.

Syntax highlighting alone needs no `npm install` and no server. If you only want
highlighting, set `"maka.server.enabled": false`.

## Package a .vsix (to share / install elsewhere)

```sh
npm install -g @vscode/vsce
cd editors/vscode
npm install
vsce package                      # produces maka-0.2.0.vsix
code --install-extension maka-0.2.0.vsix
```

(The server binary is not bundled; install `maka-lsp` separately and point
`maka.server.path` at it.)

## Settings

- `maka.server.path` — path to the `maka-lsp` binary (default: `maka-lsp` on PATH).
- `maka.server.enabled` — run the language server (default: `true`).

`rblock` and `cblock` functions resolve like any other: the server runs the same
Rust-bridge signature extraction the compiler uses (guarded so a file without an
rblock pays nothing), and `cblock` functions resolve through their `extern`
declaration.
