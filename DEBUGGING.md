# Debugging Maka

Maka compiles to C and then to a native binary, and the compiler emits
`#line N "file.maka"` directives plus `-g`, so the binary's DWARF debug info
references **Maka source**, not the generated C. That means `gdb`, `lldb`, and
VS Code's debugger let you set breakpoints in `.maka` files, step through Maka
source, and read a Maka backtrace - with no extra setup.

Every build already includes debug info (`makac`, `maka build`, `maka run`).

## gdb

```sh
maka build                       # or: makac app.maka -o app
gdb ./target/app                 # or ./app
(gdb) break app.maka:12          # breakpoint on a Maka line
(gdb) run
Breakpoint 1, tick (s_0=...) at app.maka:12
12          s.frame = s.frame + 1;
(gdb) next                       # step over, in Maka source
(gdb) backtrace                  # frames show app.maka:LINE
(gdb) continue
```

Break by function name works too: `break tick`.

## lldb

```sh
lldb ./target/app
(lldb) breakpoint set --file app.maka --line 12
(lldb) run
(lldb) next
(lldb) bt
```

## VS Code

Install a native-debug extension - **CodeLLDB** (`vadimcn.vscode-lldb`) or
**C/C++** (`ms-vscode.cpptools`) - then add a `.vscode/launch.json`. Set
breakpoints directly in `.maka` files.

CodeLLDB:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug Maka",
      "program": "${workspaceFolder}/target/${workspaceFolderBasename}",
      "args": [],
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

C/C++ (cpptools, gdb):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "cppdbg",
      "request": "launch",
      "name": "Debug Maka",
      "program": "${workspaceFolder}/target/${workspaceFolderBasename}",
      "cwd": "${workspaceFolder}",
      "MIMode": "gdb"
    }
  ]
}
```

A `maka build` puts the binary at `target/<project-name>`; point `program` at it.

## What works

Source-level breakpoints, stepping (`next`/`step`), a Maka backtrace, and
reading variables by their **Maka names**: a parameter `a`, a local `total`
show as `a`, `total` (not `a_0`, `total_3`). The compiler emits each local with
its readable Maka name, only falling back to a unique `name_<id>` spelling when
that name would clash (a shadowed name, or a function/type/runtime symbol of the
same name), so gdb, lldb, and the VS Code variables view all read clean names
straight from DWARF.

Remaining rough edges: a module-scope **global** `g` still shows as
`__maka_global__g` (the prefix keeps it distinct from C globals), and there is
no Maka-aware pretty-printing of `Vec`/`String`/enums yet - both are follow-ups.

## Optimized builds

`--release` (`-O2`) is still built with `-g`, but the optimizer reorders and
elides code, so stepping is jumpy and some locals read `<optimized out>`. Debug
at the default `-O0` for a faithful stepping experience.
