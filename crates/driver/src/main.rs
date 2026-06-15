//! `makac` — compile a Maka source file to C and (optionally) invoke a C compiler.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: makac <input.maka>... [-o output] [--emit-c] [--run]");
        std::process::exit(2);
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
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-o" => { i += 1; output = Some(args[i].clone()); }
            "--emit-c" => { emit_c = true; }
            "--run" => { run = true; }
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
    // Prelude: types and functions every program can use without importing.
    //   - `Option<T>` — generic tagged option.
    //   - `String` — owned heap text (compiler alias for `own *char`).
    //   - `str_*` — common string operations, implemented as externs to libc.
    let prelude_src = "\
        pub enum Option<T> { Some { T value }, None }\n\
        pub enum Result<T, E> { Ok { T value }, Err { E err } }\n\
        extern \"strlen\" usize __str_len(string s);\n\
        pub usize str_len(string s) { return __str_len(s); }\n\
        extern \"strcmp\" i32 __str_cmp(string a, string b);\n\
        pub bool str_eq(string a, string b) { return __str_cmp(a, b) == 0; }\n\
        ";
    if let Ok(m) = maka_parser::parse(prelude_src) {
        for _ in &m.items {
            merged.item_modules.push(Vec::new());
            merged.item_imports.push(Vec::new());
            merged.item_has_imports.push(Vec::new());
        }
        merged.items.extend(m.items);
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
    let module = merged;

    // Sema
    let hir = match maka_sema::analyze(&module) {
        Ok(h) => h,
        Err(errs) => {
            for e in errs { eprintln!("{}", e); }
            std::process::exit(1);
        }
    };
    // Non-fatal diagnostics: print to stderr, but don't fail the build.
    for w in &hir.warnings {
        eprintln!("{}", w);
    }

    // Codegen
    let c_code = maka_codegen::emit(&hir);

    let stem = PathBuf::from(&first_input).file_stem().unwrap().to_string_lossy().to_string();
    let out_c = output.clone().unwrap_or_else(|| format!("{}.c", stem));
    let out_bin = output.clone().unwrap_or_else(|| stem.clone());

    if emit_c {
        std::fs::write(&out_c, &c_code).expect("write C");
        if !run {
            return;
        }
    }

    // Write to temp + invoke cc
    let tmp = format!("/tmp/{}.c", stem);
    std::fs::write(&tmp, &c_code).expect("write C tmp");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let mut cc_args: Vec<String> = vec!["-std=c11".into(), "-O0".into(),
        "-Wno-error".into(), "-Wno-int-conversion".into(),
        "-Wno-incompatible-pointer-types".into(),
        "-Wno-return-type".into(),
        "-Wno-discarded-qualifiers".into(),
        "-w".into(),     // suppress all remaining warnings (lambda trampolines, etc.)
        tmp.clone()];
    for c in &link_c { cc_args.push(c.clone()); }
    // Library flags come AFTER object inputs so ld can resolve symbols in scan order.
    for f in &link_flags { cc_args.push(f.clone()); }
    cc_args.push("-o".into());
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
        let exec = if out_bin.starts_with('/') || out_bin.starts_with("./") { out_bin.clone() } else { format!("./{}", out_bin) };
        let st = Command::new(&exec).status().expect("run failed");
        std::process::exit(st.code().unwrap_or(0));
    }
}
