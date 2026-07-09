# maka_bridge

The Rust FFI (`rblock` / `rdep`) sidecar pipeline. Authoritative spec:
`RUST_INTEROP.md`.

- **Job:** compile inline Rust into a per-module sidecar Cargo staticlib and marshal
  values across the C ABI. Two phases:
  - `prepare()` (pre-sema): sig-parse each `rblock` with `syn`, inject `extern`
    decls + mirrored `#[repr(C)]` `data` types (and `Rust<T>` opaque markers) into
    the AST so `sema` can resolve calls into Rust.
  - `finish()` (post-sema): emit the sidecar `Cargo.toml`/`lib.rs` (with the
    `Send`/`Sync` assertion probes `sema` collected), run `cargo build`, return the
    staticlib + harvested `-L`/`-l` link flags.
- **DAG:** depends on `lexer`, `ast`. Depended on by `driver`, `lsp`.
- **Never:** run the C compiler or the final link - it returns artifact paths and
  lets `driver` own the link line. Content-addressed cache under `.maka_cache/rust/`.
