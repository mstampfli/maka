//! `makac` — compile a Maka source file to C and (optionally) invoke a C compiler.

mod rust_bridge;
mod lint;

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // The parser (recursive descent) and the typeck/codegen tree walks recurse
    // once per nesting level, so deeply nested input (e.g. a long chain of binary
    // operators or many nested parentheses) can overflow the default 8 MiB main
    // stack and abort with no diagnostic.  Run the whole compile on a thread with
    // a large stack so realistic deep input compiles instead of crashing.  Any
    // `process::exit` inside `run` still exits the whole process; a panic in `run`
    // is re-raised here, preserving the existing failure behavior.
    let worker = std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(run)
        .expect("failed to spawn compiler worker thread");
    if worker.join().is_err() {
        std::process::exit(101);
    }
}

fn run() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: makac <input.maka>... [-o output] [--emit-c] [--run]");
        eprintln!("       makac lint <file.maka> ...   check style / naming conventions");
        std::process::exit(2);
    }
    // `makac lint FILE...` — style checker (STYLE_GUIDE.md); no compilation.
    if args[1] == "lint" {
        std::process::exit(lint::run(&args[2..]));
    }
    let mut inputs: Vec<String> = Vec::new();
    // .c / .o / .a source files that should be compiled+linked alongside our generated C.
    let mut link_c: Vec<String> = Vec::new();
    // Linker flags (`-lname`, `-L/path`) — passed to the C compiler after all objects so the
    // GNU linker resolves the symbols in the right order.
    let mut link_flags: Vec<String> = Vec::new();
    let mut output: Option<String> = None;
    let mut emit_c = false;
    let mut run = false;
    // Optimization level for the generated C.  Default O0 (fast builds, easy to
    // debug); `--release` / `-O2` turns on optimization.
    let mut opt_level = String::from("-O0");
    let mut rust_profile: Option<String> = None;
    let mut no_rust = false;
    // Freestanding mode — strip the libc-using codegen prologue, skip the
    // auto-include of stdlib/std.maka, and lower alloc/free/panic/log to
    // user-provided extern symbols (__maka_alloc, __maka_free, __maka_panic,
    // __maka_log).  Targets a no-libc / no-OS environment (kernels, boot
    // images).  See "Freestanding mode" in SPEC.
    let mut freestanding = false;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-o" => { i += 1; output = Some(args[i].clone()); }
            "--emit-c" => { emit_c = true; }
            "--run" => { run = true; }
            "--release" => { opt_level = String::from("-O2"); }
            "-O0" | "-O1" | "-O2" | "-O3" | "-Os" => { opt_level = a.clone(); }
            "--no-rust" => { no_rust = true; }
            "--freestanding" => { freestanding = true; no_rust = true; }
            "--rust-profile" => { i += 1; rust_profile = Some(args[i].clone()); }
            s if s.starts_with("--rust-profile=") => {
                rust_profile = Some(s.trim_start_matches("--rust-profile=").to_string());
            }
            "--link" => {
                i += 1;
                let v = &args[i];
                if v.starts_with("-l") || v.starts_with("-L") { link_flags.push(v.clone()); }
                else { link_c.push(v.clone()); }
            }
            s if s.starts_with("-l") || s.starts_with("-L") => link_flags.push(s.to_string()),
            _ => { inputs.push(a.clone()); }
        }
        i += 1;
    }
    if inputs.is_empty() {
        eprintln!("error: no input files");
        std::process::exit(2);
    }
    let first_input = inputs[0].clone();

    // Lex + parse each file and merge.
    let mut merged = maka_ast::Module {
        items: Vec::new(),
        module_path: None,
        item_modules: Vec::new(),
        item_imports: Vec::new(),
        imports: Vec::new(),
        has_imports: Vec::new(),
        item_has_imports: Vec::new(),
    };
    // Standard library: parsed at every build from `stdlib/std.maka`.  The
    // file is embedded into the compiler binary at build time via
    // `include_str!` so deployments stay single-binary, but it's a real Maka
    // source file in the repo that anyone can read or edit.
    //
    // All items in `stdlib/std.maka` declare `module std;` and require an
    // explicit `import std.Name;` to use - this is regular module visibility,
    // not magic prelude.
    let std_src = include_str!("../../../stdlib/std.maka");
    if !freestanding {
    if let Ok(m) = maka_parser::parse(std_src) {
        let path: Vec<String> = m.module_path.clone().unwrap_or_default();
        let flat_imports: Vec<(Vec<String>, String)> = m.imports.iter()
            .flat_map(|imp| imp.names.iter().map(|n| (imp.path.clone(), n.clone())))
            .collect();
        let file_has_imports = m.has_imports.clone();
        for _ in &m.items {
            merged.item_modules.push(path.clone());
            merged.item_imports.push(flat_imports.clone());
            merged.item_has_imports.push(file_has_imports.clone());
        }
        merged.items.extend(m.items);
    }
    }
    for f in &inputs {
        let src = std::fs::read_to_string(f).unwrap_or_else(|e| {
            eprintln!("cannot read {}: {}", f, e); std::process::exit(1);
        });
        match maka_parser::parse(&src) {
            Ok(m) => {
                // Tag every item with this file's module path AND this file's imports.
                let path = m.module_path.unwrap_or_default();
                // Flatten ImportDecl list into a Vec<(module_path, name)> the
                // visibility checker can scan in O(n).
                let flat_imports: Vec<(Vec<String>, String)> = m.imports.iter()
                    .flat_map(|imp| imp.names.iter().map(|n| (imp.path.clone(), n.clone())))
                    .collect();
                let file_has_imports = m.has_imports.clone();
                for _ in &m.items {
                    merged.item_modules.push(path.clone());
                    merged.item_imports.push(flat_imports.clone());
                    merged.item_has_imports.push(file_has_imports.clone());
                }
                merged.items.extend(m.items);
            }
            Err(e) => { eprintln!("{}: {}", f, e); std::process::exit(1); }
        }
    }
    let mut module = merged;

    // Rust interop bridge: build sidecar crates from rblocks, inject extern
    // decls + staticlib paths.  Skipped when no rblock/rdep is present, so
    // builds with zero rust interop pay zero cost.
    let bridge_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let bridge_opts = rust_bridge::BridgeOptions {
        no_rust,
        profile: rust_profile.unwrap_or_else(|| "release".into()),
    };
    // Phase 1: parse rblocks, inject extern decls + mirrored data types into
    // the AST so sema can resolve names and collect Send/Sync probes.
    let prep = match rust_bridge::prepare(&module, &bridge_root, &bridge_opts) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("rust bridge: {}", e);
            std::process::exit(1);
        }
    };
    for (mod_path, item) in prep.injected.clone() {
        module.items.push(item);
        module.item_modules.push(mod_path);
        module.item_imports.push(Vec::new());
        module.item_has_imports.push(Vec::new());
    }

    // Sema
    let hir = match maka_sema::analyze(&module) {
        Ok(h) => h,
        Err(errs) => {
            for e in errs { eprintln!("{}", e); }
            std::process::exit(1);
        }
    };

    // In-source `clink "...";` directives feed the final link line, same split as
    // the `--link` CLI arg: `-l`/`-L` are linker flags, anything else is a source/
    // object/archive to compile and link alongside the generated C.
    for f in &hir.clinks {
        if f.starts_with("-l") || f.starts_with("-L") { link_flags.push(f.clone()); }
        else { link_c.push(f.clone()); }
    }

    // Phase 2: now that sema has surfaced per-call-site Send/Sync probes,
    // build the sidecar crates and add their staticlibs to the C link line.
    match rust_bridge::finish(prep, &hir.sym.send_probes, &hir.sym.sync_probes, &bridge_opts) {
        Ok((libs, flags)) => {
            // Staticlibs go on link_c (emitted first); the -l/-L flags their build
            // scripts requested go on link_flags (emitted after), so a `-lFoo` that
            // resolves an undefined symbol in the sidecar lands to its right, where
            // GNU ld's left-to-right resolution needs it.
            for lib in libs {
                link_c.push(lib);
            }
            for f in flags {
                if !link_flags.contains(&f) { link_flags.push(f); }
            }
        }
        Err(e) => {
            eprintln!("rust bridge: {}", e);
            std::process::exit(1);
        }
    }
    // Non-fatal diagnostics: print to stderr, but don't fail the build.
    for w in &hir.warnings {
        eprintln!("{}", w);
    }

    // Codegen
    let c_code = if freestanding {
        maka_codegen::emit_freestanding(&hir)
    } else {
        maka_codegen::emit(&hir)
    };

    let stem = PathBuf::from(&first_input).file_stem().unwrap().to_string_lossy().to_string();
    let out_c = output.clone().unwrap_or_else(|| format!("{}.c", stem));
    let out_bin = output.clone().unwrap_or_else(|| stem.clone());

    if emit_c {
        ensure_parent_dir(&out_c);
        if let Err(e) = std::fs::write(&out_c, &c_code) {
            eprintln!("cannot write `{}`: {}", out_c, e);
            std::process::exit(1);
        }
        if !run {
            return;
        }
    }

    // Write to temp + invoke cc
    let tmp = format!("/tmp/{}.c", stem);
    std::fs::write(&tmp, &c_code).expect("write C tmp");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let mut cc_args: Vec<String> = vec!["-std=c11".into(), opt_level.clone(),
        // Define integer overflow as two's-complement wrap (Maka `int` is i64
        // and is expected to wrap, e.g. hashes/RNGs), and disable strict
        // aliasing since the runtime reinterprets memory through several pointer
        // types.  Both are no-ops at -O0 but make -O2 builds well-defined.
        "-fwrapv".into(), "-fno-strict-aliasing".into(),
        "-Wno-error".into(), "-Wno-int-conversion".into(),
        "-Wno-incompatible-pointer-types".into(),
        "-Wno-return-type".into(),
        "-Wno-discarded-qualifiers".into(),
        "-w".into(),     // suppress all remaining warnings (lambda trampolines, etc.)
        tmp.clone()];
    for c in &link_c { cc_args.push(c.clone()); }
    // Library flags come AFTER object inputs so ld can resolve symbols in scan order.
    for f in &link_flags { cc_args.push(f.clone()); }
    // libm: floating-point modulo lowers to fmod, which at low optimization is a
    // real libm call rather than an inlined builtin.  Harmless when unused.
    cc_args.push("-lm".into());
    // On Windows the runtime pulls in Winsock (socket/htonl/WSAPoll), the
    // multimedia timer (timeBeginPeriod), the WaitOnAddress futex API, and
    // pthreads (winpthread also supplies clock_gettime/nanosleep), so link them
    // by default - the same role `-lm` plays elsewhere.  `--link -l<name>` stays
    // additive.  Gated on TARGETING Windows (host, or a MAKA_SIDECAR_TARGET
    // override), so `CC=x86_64-w64-mingw32-gcc MAKA_SIDECAR_TARGET=...gnu` can
    // cross-build a Windows binary from another host.
    let targeting_windows = cfg!(windows)
        || rust_bridge::sidecar_target_triple().map(|t| t.contains("windows")).unwrap_or(false);
    if targeting_windows {
        cc_args.push("-lws2_32".into());
        cc_args.push("-lwinmm".into());
        cc_args.push("-lsynchronization".into());
        cc_args.push("-lpthread".into());
        // Rust std (pulled in by any rblock sidecar staticlib) needs these Win32
        // system libs; a staticlib doesn't carry its own native-lib deps, so the
        // final link must supply them.  Harmless for pure-C programs.  Placed
        // after the staticlibs so the linker resolves std's references in order.
        for lib in ["-ladvapi32", "-luserenv", "-lbcrypt", "-lntdll", "-lkernel32",
                    "-lole32", "-loleaut32", "-lshell32"] {
            cc_args.push(lib.into());
        }
    }
    cc_args.push("-o".into());
    ensure_parent_dir(&out_bin);
    cc_args.push(out_bin.clone());
    let status = Command::new(&cc)
        .args(&cc_args)
        .status()
        .expect("cc failed to start");
    if !status.success() {
        eprintln!("cc failed");
        std::process::exit(1);
    }

    if run {
        // A bare filename (no directory component, not absolute) must be launched
        // as `./name` so it resolves in the cwd rather than via a PATH lookup.  An
        // absolute path (`/x`, `C:\x`, `\\?\x`) or one that already has a directory
        // component is used as-is: prefixing `./` there produces a malformed path
        // (e.g. `./C:\...`), which Windows rejects with ERROR_INVALID_NAME (123).
        let p = std::path::Path::new(&out_bin);
        let exec: PathBuf = if p.is_absolute()
            || p.parent().map_or(false, |par| !par.as_os_str().is_empty())
        {
            p.to_path_buf()
        } else {
            std::path::Path::new(".").join(p)
        };
        let st = match Command::new(&exec).status() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cannot run `{}`: {}", exec.display(), e);
                std::process::exit(1);
            }
        };
        // A child killed by a signal (e.g. a Rust panic aborting across the C ABI)
        // has code()==None; report it as 128+signal like a shell, not a masked 0.
        #[cfg(unix)]
        let code = {
            use std::os::unix::process::ExitStatusExt;
            st.code().unwrap_or_else(|| 128 + st.signal().unwrap_or(0))
        };
        #[cfg(not(unix))]
        let code = st.code().unwrap_or(1);
        std::process::exit(code);
    }
}

/// Create the parent directory of an output path if it does not exist, so
/// `-o build/out.c` / `-o dist/app` works without a manual `mkdir` (and fails
/// with a clear message instead of a raw `Os { code: 2/3 }` panic).
fn ensure_parent_dir(path: &str) {
    if let Some(parent) = PathBuf::from(path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("cannot create output directory `{}`: {}", parent.display(), e);
                std::process::exit(1);
            }
        }
    }
}
