//! `maka` — the Maka project front-end: a small, cargo-like wrapper around the
//! `makac` compiler.  Std-only; it locates `makac` (next to itself, else on
//! PATH) and shells out.
//!
//!   maka new NAME        scaffold a new project in ./NAME
//!   maka init            scaffold in the current directory
//!   maka build [--release]        compile src/*.maka to target/<name>
//!   maka run   [--release] [-- args...]   build, then run with args
//!   maka test  [--release]        compile+run tests/*.maka, diff vs .expected
//!   maka lint                     run `makac lint` over src/*.maka
//!   maka fmt [--check] [FILE...]  reformat (layout) src/*.maka in place
//!   maka add NAME [VERSION]       guidance for adding a dependency
//!
//! A project is a directory with a `maka.toml`:
//!
//!   [package]
//!   name = "myapp"
//!   version = "0.1.0"
//!
//!   [build]
//!   entry = "src/main.maka"     # optional; default is src/main.maka

use std::path::{Path, PathBuf};
use std::process::{exit, Command};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let rest: Vec<String> = args.iter().skip(2).cloned().collect();
    let code = match cmd {
        "new" => cmd_new(&rest),
        "init" => cmd_init(&rest),
        "build" => cmd_build(&rest),
        "run" => cmd_run(&rest),
        "test" => cmd_test(&rest),
        "lint" => cmd_lint(&rest),
        "fmt" => cmd_fmt(&rest),
        "add" => cmd_add(&rest),
        "help" | "--help" | "-h" => {
            print_help();
            0
        }
        "version" | "--version" | "-V" => {
            println!("maka {}", env!("CARGO_PKG_VERSION"));
            0
        }
        other => {
            eprintln!("maka: unknown command `{}`\n", other);
            print_help();
            2
        }
    };
    exit(code);
}

fn print_help() {
    eprint!(
        "\
maka - the Maka project tool

usage:
  maka new <name>              create a new project in ./<name>
  maka init                    create a project in the current directory
  maka build [--release]       compile src/*.maka to target/<name>
  maka run   [--release] [-- <args>]   build, then run (args pass to the program)
  maka test  [--release]       compile+run tests/*.maka, diff stdout vs .expected
  maka lint                    check style/naming with `makac lint` over src/
  maka fmt [--check] [file...] reformat (layout) src/ in place, or the given files
  maka add <name> [version]    how to add a Rust crate / C library dependency
  maka version                 print the version
"
    );
}

// ---------------------------------------------------------------- manifest

struct Manifest {
    name: String,
    entry: String,
}

/// Walk up from the current directory to the project root (the dir with a
/// `maka.toml`), returning (root, manifest).
fn load_manifest() -> Result<(PathBuf, Manifest), String> {
    let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        let man = dir.join("maka.toml");
        if man.is_file() {
            let text = std::fs::read_to_string(&man).map_err(|e| e.to_string())?;
            let name = toml_value(&text, "name")
                .unwrap_or_else(|| dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "app".into()));
            let entry = toml_value(&text, "entry").unwrap_or_else(|| "src/main.maka".to_string());
            return Ok((dir, Manifest { name, entry }));
        }
        if !dir.pop() {
            return Err("no `maka.toml` found in this or any parent directory (run `maka new`/`maka init`)".to_string());
        }
    }
}

/// Extract `key = "value"` from a small maka.toml (first match wins).  A minimal
/// scan, not a full TOML parser: the manifest has a handful of string keys.
fn toml_value(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let v = rest.trim().trim_matches('"');
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------- makac

/// Locate the `makac` compiler: prefer a sibling of this `maka` executable
/// (so a dev build in target/debug finds its own makac), else fall back to PATH.
fn makac() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sib = dir.join(if cfg!(windows) { "makac.exe" } else { "makac" });
            if sib.is_file() {
                return sib;
            }
        }
    }
    PathBuf::from("makac")
}

/// Every `.maka` file under `dir` (recursively), sorted for a stable order.
fn maka_sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_maka(dir, &mut out);
    out.sort();
    out
}
fn collect_maka(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_maka(&p, out);
        } else if p.extension().map_or(false, |x| x == "maka") {
            out.push(p);
        }
    }
}

// ---------------------------------------------------------------- commands

fn cmd_new(args: &[String]) -> i32 {
    let Some(name) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("maka new: expected a project name");
        return 2;
    };
    let root = PathBuf::from(name);
    if root.exists() {
        eprintln!("maka new: `{}` already exists", name);
        return 1;
    }
    match scaffold(&root, name) {
        Ok(()) => {
            println!("created project `{}`", name);
            println!("  cd {} && maka run", name);
            0
        }
        Err(e) => {
            eprintln!("maka new: {}", e);
            1
        }
    }
}

fn cmd_init(_args: &[String]) -> i32 {
    let root = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("maka init: {}", e);
            return 1;
        }
    };
    if root.join("maka.toml").exists() {
        eprintln!("maka init: `maka.toml` already exists here");
        return 1;
    }
    let name = root
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "app".into());
    match scaffold(&root, &name) {
        Ok(()) => {
            println!("initialized project `{}`", name);
            0
        }
        Err(e) => {
            eprintln!("maka init: {}", e);
            1
        }
    }
}

fn scaffold(root: &Path, name: &str) -> Result<(), String> {
    let w = |rel: &str, contents: &str| -> Result<(), String> {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&p, contents).map_err(|e| e.to_string())
    };
    w(
        "maka.toml",
        &format!("[package]\nname = \"{}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.maka\"\n", name),
    )?;
    w(
        "src/main.maka",
        "unit main() {\n    log(\"hello from maka\");\n}\n",
    )?;
    w(".gitignore", "/target\n/.maka_cache\n")?;
    Ok(())
}

/// Compile the project's sources to `target/<name>`.  Returns the binary path.
fn build(release: bool) -> Result<(PathBuf, PathBuf), String> {
    let (root, man) = load_manifest()?;
    let src_dir = root.join("src");
    if !src_dir.is_dir() {
        return Err("no `src/` directory in the project".to_string());
    }
    let sources = maka_sources(&src_dir);
    if sources.is_empty() {
        return Err("no `.maka` sources under `src/`".to_string());
    }
    let target = root.join("target");
    std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
    let out = target.join(&man.name);

    let mut cmd = Command::new(makac());
    cmd.current_dir(&root);
    for s in &sources {
        cmd.arg(s);
    }
    cmd.arg("-o").arg(&out);
    if release {
        cmd.arg("--release");
    }
    let status = cmd.status().map_err(|e| format!("failed to run makac: {}", e))?;
    if !status.success() {
        return Err("compilation failed".to_string());
    }
    Ok((root, out))
}

fn cmd_build(args: &[String]) -> i32 {
    let release = args.iter().any(|a| a == "--release");
    match build(release) {
        Ok((_, out)) => {
            println!("built {}", out.display());
            0
        }
        Err(e) => {
            eprintln!("maka build: {}", e);
            1
        }
    }
}

fn cmd_run(args: &[String]) -> i32 {
    let release = args.iter().take_while(|a| *a != "--").any(|a| a == "--release");
    // Everything after `--` is forwarded to the program.
    let prog_args: Vec<&String> = match args.iter().position(|a| a == "--") {
        Some(i) => args[i + 1..].iter().collect(),
        None => Vec::new(),
    };
    let out = match build(release) {
        Ok((_, out)) => out,
        Err(e) => {
            eprintln!("maka run: {}", e);
            return 1;
        }
    };
    match Command::new(&out).args(&prog_args).status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("maka run: failed to launch {}: {}", out.display(), e);
            1
        }
    }
}

fn cmd_test(args: &[String]) -> i32 {
    let release = args.iter().any(|a| a == "--release");
    let (root, _man) = match load_manifest() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("maka test: {}", e);
            return 1;
        }
    };
    let tdir = root.join("tests");
    if !tdir.is_dir() {
        eprintln!("maka test: no `tests/` directory");
        return 0;
    }
    let mut tests = maka_sources(&tdir);
    tests.retain(|p| p.extension().map_or(false, |x| x == "maka"));
    tests.sort();
    if tests.is_empty() {
        eprintln!("maka test: no `.maka` tests under `tests/`");
        return 0;
    }
    let bindir = root.join("target").join("test");
    let _ = std::fs::create_dir_all(&bindir);
    let mut failed = 0;
    for t in &tests {
        let stem = t.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let bin = bindir.join(&stem);
        let mut cmd = Command::new(makac());
        cmd.current_dir(&root).arg(t).arg("-o").arg(&bin).arg("--run");
        if release {
            cmd.arg("--release");
        }
        let output = match cmd.output() {
            Ok(o) => o,
            Err(e) => {
                println!("FAIL {}: {}", stem, e);
                failed += 1;
                continue;
            }
        };
        let got = String::from_utf8_lossy(&output.stdout);
        if !output.status.success() {
            println!("FAIL {} (exit {})", stem, output.status.code().unwrap_or(-1));
            failed += 1;
            continue;
        }
        let exp_path = t.with_extension("expected");
        if let Ok(expected) = std::fs::read_to_string(&exp_path) {
            if got.trim_end_matches('\n') == expected.trim_end_matches('\n') {
                println!("ok   {}", stem);
            } else {
                println!("FAIL {} (output mismatch)", stem);
                failed += 1;
            }
        } else {
            // No .expected: passing means it compiled and ran with exit 0.
            println!("ok   {} (ran, no .expected to diff)", stem);
        }
    }
    if failed == 0 {
        println!("test result: ok ({} passed)", tests.len());
        0
    } else {
        println!("test result: {} failed of {}", failed, tests.len());
        1
    }
}

fn cmd_lint(_args: &[String]) -> i32 {
    let (root, _man) = match load_manifest() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("maka lint: {}", e);
            return 1;
        }
    };
    let sources = maka_sources(&root.join("src"));
    if sources.is_empty() {
        eprintln!("maka lint: no `.maka` sources under `src/`");
        return 0;
    }
    let mut cmd = Command::new(makac());
    cmd.current_dir(&root).arg("lint");
    for s in &sources {
        cmd.arg(s);
    }
    match cmd.status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("maka lint: failed to run makac: {}", e);
            1
        }
    }
}

fn cmd_fmt(args: &[String]) -> i32 {
    // Split flags from explicit file arguments.  With no files given, format the
    // project's `src/*.maka`; with files, format exactly those (so `maka fmt a.maka`
    // works outside a project too).
    let mut flags: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    for a in args {
        if a.starts_with('-') {
            flags.push(a.clone());
        } else {
            files.push(a.clone());
        }
    }

    let mut cmd = Command::new(makac());
    cmd.arg("fmt");
    for f in &flags {
        cmd.arg(f);
    }

    if files.is_empty() {
        let (root, _man) = match load_manifest() {
            Ok(x) => x,
            Err(e) => {
                eprintln!("maka fmt: {} (or pass files explicitly)", e);
                return 1;
            }
        };
        let sources = maka_sources(&root.join("src"));
        if sources.is_empty() {
            eprintln!("maka fmt: no `.maka` sources under `src/`");
            return 0;
        }
        cmd.current_dir(&root);
        for s in &sources {
            cmd.arg(s);
        }
    } else {
        for f in &files {
            cmd.arg(f);
        }
    }

    match cmd.status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("maka fmt: failed to run makac: {}", e);
            1
        }
    }
}

fn cmd_add(args: &[String]) -> i32 {
    let name = args.first().map(String::as_str).unwrap_or("<name>");
    let ver = args.get(1).map(String::as_str).unwrap_or("1");
    // Maka declares dependencies IN SOURCE, next to the code that uses them, so
    // the sidecar build sees exactly what each module needs.  Guide, don't guess.
    eprintln!(
        "\
maka dependencies live in source, beside the code that uses them:

  Rust crate `{name}` - inside the `rblock` that uses it:

      rdep {name} = \"{ver}\";
      rblock \"
          use {name}::...;
          pub fn ... {{ ... }}
      \";

  C library - a link directive:

      clink \"-l{name}\";          // a linker flag
      clink \"vendor/{name}.c\";   // or a source/object/archive to link in

See RUST_INTEROP.md and SPEC 5.2. A manifest-driven `add` that generates this
plumbing is planned; for now add the two lines above to your module."
    );
    0
}
