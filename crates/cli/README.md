# maka_cli

The **`maka`** binary: a cargo-like project front-end.

- **Job:** scaffold and drive Maka projects - `maka new/init`, `build`, `run`,
  `test`, `fmt`, `lint`, `add`. A project is a directory with a `maka.toml`. It
  locates `makac` (sibling binary, else PATH) and shells out to it.
- **DAG:** std-only - depends on NO compiler crate. Nothing depends on it.
- **Never:** link against the compiler crates or reimplement compilation. Decoupling
  is deliberate: the project tool and the compiler version independently, and `maka`
  is just process orchestration over `makac`.
