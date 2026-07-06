# Maka for VS Code

Syntax highlighting for the [Maka](https://github.com/mstampfli/maka) language:
keywords, types, the five pointer flavors, strings/chars/numbers, comments, and
function names. `rblock "..."` bodies are highlighted as **Rust** and
`cblock "..."` bodies as **C**.

This is a grammar-only extension (TextMate). Semantic features (hover types,
go-to-definition, diagnostics) will come from a language server later.

## Install (local)

Symlink or copy this directory into your VS Code extensions folder, then reload:

```sh
ln -s "$PWD" ~/.vscode/extensions/maka
# or: cp -r editors/vscode ~/.vscode/extensions/maka
```

Reload the window (Ctrl+Shift+P -> "Developer: Reload Window"). Any `.maka` file
now highlights.

## Package a .vsix (to share / install elsewhere)

```sh
npm install -g @vscode/vsce
cd editors/vscode
vsce package                 # produces maka-0.1.0.vsix
code --install-extension maka-0.1.0.vsix
```

## What it highlights

- Control flow (`if`/`else`/`while`/`for`/`match`/`return`/`propagate`/`unsafe`)
- Declarations (`data`/`enum`/`attr`/`has`/`logic`/`module`/`import`/`use`)
- Modifiers (`mut`/`const`/`pub`/`own`/`raw`/`inline`/`gate`/`export`)
- Memory (`alloc`/`free`/`transfer`/`share`/`as`) and FFI
  (`cinclude`/`cblock`/`clink`/`rblock`/`rdep`)
- Primitive and sized types (`int`/`float`/`string`/`unit`/`i32`/`usize`/...) and
  PascalCase user types
- Strings (with escapes), char literals, integer/float/hex numbers, `true`/`false`/`null`
- Function names (identifier before `(`)
- Embedded Rust (rblock) and C (cblock)
