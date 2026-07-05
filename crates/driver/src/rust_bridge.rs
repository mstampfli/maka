//! Rust interop bridge.
//!
//! For every Maka module that contains an `rblock`, this module:
//!   1. Concatenates its rblocks into a single sidecar crate (`.maka_cache/rust/<hash>/`).
//!   2. Sig-parses each `pub fn` to discover its parameter and return types.
//!   3. Emits a `#[no_mangle] extern "C"` shim per function that marshals between
//!      C-ABI and native Rust types (`&str`, `String`, `Vec<T>`, etc.), and a
//!      generic opaque-pointer fallback for any type without a typed shim.
//!   4. Runs `cargo build --release` to produce a `staticlib`.
//!   5. Injects matching Maka `extern` declarations back into the AST so the
//!      surrounding Maka code can call the shim by its Maka name.
//!   6. Returns the produced staticlib paths so the driver can link them.
//!
//! See `RUST_INTEROP.md` for the full design.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use maka_ast::{
    DataDecl, ExternDecl, FieldDecl, Item, Module, Mutness, Param, Type,
};
use maka_lexer::Span;

/// One result of processing the rust bridge for a Maka build.
pub struct BridgeOutput {
    /// Items to splice into the module, paired with the module path each item
    /// belongs to (so the driver can keep `item_modules` in sync).
    pub injected: Vec<(Vec<String>, Item)>,
    /// Static library paths to pass to the C linker.
    pub staticlibs: Vec<String>,
}

/// Driver-tuned options for the bridge.
pub struct BridgeOptions {
    /// Refuse to build any sidecar; error out if an rblock is present.
    pub no_rust: bool,
    /// Cargo profile name to build under (`release` / `dev`).
    pub profile: String,
}

/// Per-module state captured during `prepare` and consumed by `finish`.
pub struct BridgePrep {
    project_root: PathBuf,
    rustc_version_bytes: Vec<u8>,
    modules: Vec<PreparedModule>,
    /// Items the driver should splice into the Maka AST before running sema
    /// (extern decls, mirrored data, container struct data, etc.).
    pub injected: Vec<(Vec<String>, Item)>,
}

struct PreparedModule {
    module_path: Vec<String>,
    surface: RustSurface,
    combined_rust: String,
    rdeps: Vec<(String, String)>,
}

/// Phase 1 — parse + inject.  Walks the merged Maka module, sig-parses each
/// module's rblocks, and returns the items to splice into the AST along with
/// a `BridgePrep` token to feed into `finish` once sema has produced its
/// `Send`/`Sync` probe lists.
pub fn prepare(module: &Module, project_root: &Path, opts: &BridgeOptions) -> Result<BridgePrep, String> {
    let mut per_module: HashMap<Vec<String>, ModuleBundle> = HashMap::new();
    for (idx, item) in module.items.iter().enumerate() {
        let mp = module.item_modules.get(idx).cloned().unwrap_or_default();
        let bundle = per_module.entry(mp).or_default();
        match item {
            Item::Rblock(src, _) => bundle.rblocks.push(src.clone()),
            Item::Rdep(name, ver, _) => bundle.rdeps.push((name.clone(), ver.clone())),
            _ => {}
        }
    }

    let mut prepped: Vec<PreparedModule> = Vec::new();
    let mut injected: Vec<(Vec<String>, Item)> = Vec::new();

    for (module_path, bundle) in &per_module {
        if bundle.rblocks.is_empty() {
            if !bundle.rdeps.is_empty() {
                return Err(format!(
                    "`rdep` declared without any `rblock` in module `{}`",
                    if module_path.is_empty() { "<root>".to_string() } else { module_path.join(".") }
                ));
            }
            continue;
        }
        if opts.no_rust {
            return Err(format!(
                "`--no-rust` is set but module `{}` contains rblock(s); pass without `--no-rust` to build them",
                if module_path.is_empty() { "<root>".to_string() } else { module_path.join(".") }
            ));
        }

        let combined_rust = bundle.rblocks.join("\n\n// ----- rblock boundary -----\n\n");
        let surface = parse_rust_surface(&combined_rust)
            .map_err(|e| format!("rust signature parse error: {}", e))?;

        // Inject mirrored Maka data decls for each #[repr(C)] struct, all
        // container instantiations, and one extern per free fn / impl method.
        for s in &surface.structs {
            if !s.repr_c {
                continue;
            }
            injected.push((module_path.clone(), Item::Data(build_data_decl(s))));
        }
        let containers = collect_container_insts(&surface);
        for d in build_container_data_decls(&containers) {
            injected.push((module_path.clone(), Item::Data(d)));
        }
        for f in &surface.fns {
            let extern_decl = build_extern_decl(f);
            injected.push((module_path.clone(), Item::Extern(extern_decl)));
        }

        prepped.push(PreparedModule {
            module_path: module_path.clone(),
            surface,
            combined_rust,
            rdeps: bundle.rdeps.clone(),
        });
    }

    let rustc_v = Command::new("rustc")
        .arg("--version")
        .output()
        .map_err(|e| format!("`rustc --version` failed (is rustc installed?): {}", e))?;

    Ok(BridgePrep {
        project_root: project_root.to_path_buf(),
        rustc_version_bytes: rustc_v.stdout,
        modules: prepped,
        injected,
    })
}

/// Phase 2 — build sidecar crates with per-call-site `Send` / `Sync` probes
/// supplied by sema.  Returns staticlib paths to feed the C linker.
pub fn finish(
    prep: BridgePrep,
    send_probes: &[String],
    sync_probes: &[String],
    opts: &BridgeOptions,
) -> Result<Vec<String>, String> {
    let mut staticlibs: Vec<String> = Vec::new();
    for pm in &prep.modules {
        // Filter the global probe lists down to the types this module's
        // sidecar can actually see in scope — anything else would cause a
        // gratuitous rustc error rather than an honest one.
        let known: std::collections::HashSet<String> = pm
            .surface
            .structs
            .iter()
            .map(|s| s.name.clone())
            .chain(pm.surface.fns.iter().flat_map(|f| {
                let mut v = Vec::new();
                for p in &f.params { collect_opaque_names(&p.ty, &mut v); }
                collect_opaque_names(&f.ret, &mut v);
                v
            }))
            .collect();
        let local_send: Vec<&String> = send_probes.iter().filter(|t| known.contains(*t)).collect();
        let local_sync: Vec<&String> = sync_probes.iter().filter(|t| known.contains(*t)).collect();

        // Cache key now folds in the probes — adding or removing a thread
        // crossing invalidates the cache, so the next build re-verifies.
        let mut hasher = DefaultHasher::new();
        pm.combined_rust.hash(&mut hasher);
        for (k, v) in &pm.rdeps {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        prep.rustc_version_bytes.hash(&mut hasher);
        // Bridge codegen version: bump when the generated shim/drop-shim/container
        // code changes, so a stale cached sidecar is regenerated.
        "bridge-codegen-v2-rust-drop-shims".hash(&mut hasher);
        for p in &local_send { ("send", p).hash(&mut hasher); }
        for p in &local_sync { ("sync", p).hash(&mut hasher); }
        let hash = format!("{:016x}", hasher.finish());

        let crate_name = format!("{}_{}", sidecar_crate_name(&pm.module_path), &hash[..12]);
        let sidecar_dir = prep.project_root.join(".maka_cache").join("rust").join(&hash);
        let shared_target_root = prep
            .project_root
            .join(".maka_cache")
            .join("rust")
            .join("_shared_target");
        let profile_dir = if opts.profile == "dev" { "debug" } else { "release" };
        let staticlib_path = shared_target_root
            .join(profile_dir)
            .join(format!("lib{}.a", crate_name));
        let built_marker = sidecar_dir.join(".built");

        if !built_marker.exists() || !staticlib_path.exists() {
            build_sidecar_with_probes(
                &sidecar_dir,
                &crate_name,
                &pm.combined_rust,
                &pm.surface,
                &pm.rdeps,
                &opts.profile,
                &local_send,
                &local_sync,
            )?;
        }

        staticlibs.push(staticlib_path.to_string_lossy().to_string());
    }
    Ok(staticlibs)
}

/// Pull opaque-type label names out of a `RustType`, recursing into
/// containers.  Used to compute the set of Rust type names visible to a
/// given module's sidecar (for probe filtering).
fn collect_opaque_names(ty: &RustType, out: &mut Vec<String>) {
    match ty {
        RustType::Opaque(n) | RustType::RefOpaque(n) | RustType::RefMutOpaque(n) => out.push(n.clone()),
        RustType::OptionOf(inner) | RustType::VecOf(inner) => collect_opaque_names(inner, out),
        RustType::ResultOf(ok, err) => {
            collect_opaque_names(ok, out);
            collect_opaque_names(err, out);
        }
        _ => {}
    }
}

/// Back-compat top-level entry: one-shot parse + build with NO per-call-site
/// probes.  Kept so that callers that don't have access to sema's probe
/// output (e.g. dev tooling) still work.  The two-phase `prepare` / `finish`
/// path is preferred and is what the production driver uses.
pub fn process(module: &Module, project_root: &Path, opts: &BridgeOptions) -> Result<BridgeOutput, String> {
    let prep = prepare(module, project_root, opts)?;
    let injected = prep.injected.clone();
    let staticlibs = finish(prep, &[], &[], opts)?;
    Ok(BridgeOutput { injected, staticlibs })
}

#[derive(Default)]
struct ModuleBundle {
    rblocks: Vec<String>,
    rdeps: Vec<(String, String)>,
}

fn sidecar_crate_name(module_path: &[String]) -> String {
    if module_path.is_empty() {
        "maka_rust_root".to_string()
    } else {
        format!("maka_rust_{}", module_path.join("_"))
    }
}

// ------------------------------------------------------------------------
// Sidecar emission

/// Convenience wrapper: invoke `build_sidecar_with_probes` with empty probe
/// lists.  Used by the legacy one-shot `process` entry point.
fn build_sidecar(
    dir: &Path,
    crate_name: &str,
    rust_src: &str,
    surface: &RustSurface,
    rdeps: &[(String, String)],
    profile: &str,
) -> Result<(), String> {
    let no_probes: Vec<&String> = Vec::new();
    build_sidecar_with_probes(
        dir,
        crate_name,
        rust_src,
        surface,
        rdeps,
        profile,
        &no_probes,
        &no_probes,
    )
}

fn build_sidecar_with_probes(
    dir: &Path,
    crate_name: &str,
    rust_src: &str,
    surface: &RustSurface,
    rdeps: &[(String, String)],
    profile: &str,
    extra_send: &[&String],
    extra_sync: &[&String],
) -> Result<(), String> {
    std::fs::create_dir_all(dir.join("src")).map_err(|e| e.to_string())?;

    // Cargo.toml.  `libc` is unconditionally added so shims can use the same
    // allocator Maka frees with (`libc::malloc` / `libc::free`), avoiding the
    // cross-allocator UB risk when the Rust global allocator differs from
    // libc's malloc family.
    let mut cargo_toml = format!(
        "[workspace]\n\n\
         [package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n\
         [lib]\ncrate-type = [\"staticlib\"]\npath = \"src/lib.rs\"\n\n\
         [profile.release]\npanic = \"abort\"\nopt-level = 3\nlto = false\n\n\
         [dependencies]\nlibc = \"0.2\"\n",
        crate_name
    );
    for (n, v) in rdeps {
        cargo_toml.push_str(&format!("{} = {}\n", n, format_rdep_rhs(v)));
    }
    std::fs::write(dir.join("Cargo.toml"), &cargo_toml).map_err(|e| e.to_string())?;

    // lib.rs = user source + shims + Send probes
    let mut lib_rs = String::new();
    lib_rs.push_str("// Auto-generated by makac.  Do not hand-edit.\n");
    lib_rs.push_str("#![allow(non_snake_case, unused_imports, dead_code, unused_unsafe, unused_parens, path_statements, unused_variables, unused_assignments, clippy::all)]\n\n");
    lib_rs.push_str(rust_src);
    lib_rs.push_str("\n\n// ===== auto-generated maka shims =====\n\n");
    lib_rs.push_str("use std::os::raw::c_char;\n");
    lib_rs.push_str("use std::ffi::CStr;\n\n");
    // Collect every Option<T>, Result<T,E>, Vec<T> instantiation used in any
    // signature; emit a single `#[repr(C)]` struct definition per unique
    // shape into the prologue so all shims can refer to it by name.
    let containers = collect_container_insts(surface);
    if !containers.is_empty() {
        lib_rs.push_str("// auto-generated typed-container ABI structs\n");
        for c in &containers {
            lib_rs.push_str(&emit_container_struct(c));
        }
        lib_rs.push('\n');
    }
    for f in &surface.fns {
        lib_rs.push_str(&emit_shim(f));
        lib_rs.push('\n');
    }
    // Drop shims for opaque `Rust<T>` handles: Maka hands the boxed pointer back
    // here at scope exit so the Rust value is dropped (otherwise it leaks).
    let mut opaque_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for f in &surface.fns {
        let mut v = Vec::new();
        for p in &f.params { collect_opaque_names(&p.ty, &mut v); }
        collect_opaque_names(&f.ret, &mut v);
        for n in v { opaque_names.insert(n); }
    }
    if !opaque_names.is_empty() {
        lib_rs.push_str("\n// auto-generated Rust<T> drop shims\n");
        for n in &opaque_names {
            // The fn NAME must be a sanitised identifier (a raw `Vec<String>`
            // is invalid Rust in a fn name and, for nested generics, a syntax
            // error); the real type is kept only in the `*mut <type>` cast.  The
            // sanitised name matches the Maka-side `Rust<sanitise(name)>` label,
            // so codegen's `__maka_drop_<label>` call resolves this definition.
            lib_rs.push_str(&format!(
                "#[no_mangle]\npub extern \"C\" fn __maka_drop_{mangled}(p: *mut u8) {{ if !p.is_null() {{ unsafe {{ drop(Box::from_raw(p as *mut {real})); }} }} }}\n",
                mangled = sanitise(n), real = n));
        }
    }
    // Per-call-site Send / Sync probes routed from sema.  A `Send` probe
    // is emitted for every `Rust<T>` that's `transfer`'d, spawn-captured,
    // or otherwise carried across a thread boundary.  `Sync` probes
    // come from `share` sites.  rustc rejects with a precise message
    // ("Rc<T> cannot be sent between threads safely") on violation.
    lib_rs.push_str("\n// ===== auto-generated thread-crossing probes =====\n");
    lib_rs.push_str("const _: () = {\n");
    lib_rs.push_str("    const fn assert_send<T: Send>() {}\n");
    lib_rs.push_str("    const fn assert_sync<T: Sync>() {}\n");
    let mut send_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for ty in extra_send { send_set.insert((*ty).clone()); }
    for ty in &send_set {
        lib_rs.push_str(&format!("    assert_send::<{}>();\n", ty));
    }
    let mut sync_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for ty in extra_sync { sync_set.insert((*ty).clone()); }
    for ty in &sync_set {
        lib_rs.push_str(&format!("    assert_sync::<{}>();\n", ty));
    }
    lib_rs.push_str("};\n");
    std::fs::write(dir.join("src/lib.rs"), &lib_rs).map_err(|e| e.to_string())?;

    // Share one compiled-deps cache across all sidecars so libc / serde etc.
    // build once per Maka workspace, not once per module-hash.
    let shared_target = dir
        .parent()
        .map(|p| p.join("_shared_target"))
        .unwrap_or_else(|| dir.join("target"));
    let verbose = std::env::var("MAKA_RUST_VERBOSE").ok().as_deref() == Some("1");
    if verbose {
        eprintln!("makac: building rust sidecar at {}", dir.display());
    }
    let mut cargo_args: Vec<&str> = vec!["build"];
    if profile != "dev" {
        cargo_args.push("--release");
    }
    if !verbose {
        cargo_args.push("--quiet");
    }
    let mut cmd = Command::new("cargo");
    cmd.current_dir(dir)
        .env("CARGO_TARGET_DIR", &shared_target)
        .args(&cargo_args);
    // Capture cargo's output; only surface it on failure so tests that diff
    // combined stdout+stderr aren't polluted by sidecar progress lines.
    let out = cmd
        .output()
        .map_err(|e| format!("cargo invocation failed: {}", e))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        return Err(format!(
            "cargo build failed for rblock sidecar at {}\n--- stderr ---\n{}\n--- stdout ---\n{}",
            dir.display(),
            stderr,
            stdout
        ));
    }
    if verbose {
        std::io::Write::write_all(&mut std::io::stderr(), &out.stderr).ok();
    }
    std::fs::write(dir.join(".built"), "").ok();
    Ok(())
}

fn format_rdep_rhs(v: &str) -> String {
    // If the RHS is already a TOML inline table or array, splice it verbatim.
    // Otherwise treat it as a bare version string and quote it.
    let t = v.trim();
    if t.starts_with('{') || t.starts_with('[') {
        t.to_string()
    } else {
        format!("\"{}\"", t.replace('"', "\\\""))
    }
}

// ------------------------------------------------------------------------
// Shim emission

fn emit_shim(f: &RustFn) -> String {
    let mut out = String::new();
    out.push_str(&format!("#[no_mangle]\npub extern \"C\" fn __maka_shim_{}(", f.name));

    // Shim signature parameters (C-ABI flattened)
    let mut shim_params: Vec<String> = Vec::new();
    for p in &f.params {
        shim_params.push(format!("{}: {}", p.name, rust_marshal_in_ty(&p.ty)));
    }
    out.push_str(&shim_params.join(", "));
    out.push_str(") -> ");
    out.push_str(&rust_marshal_out_ty(&f.ret));
    out.push_str(" {\n");

    out.push_str("    let __r = std::panic::catch_unwind(|| {\n");

    // Unmarshal each parameter
    let mut unmarshal: Vec<String> = Vec::new();
    let mut call_args: Vec<String> = Vec::new();
    for p in &f.params {
        let (decl, expr) = unmarshal_in(&p.name, &p.ty);
        if !decl.is_empty() {
            unmarshal.push(decl);
        }
        call_args.push(expr);
    }
    for u in &unmarshal {
        out.push_str("        ");
        out.push_str(u);
        out.push('\n');
    }

    // Call user function (free fn name OR `Type::method` for impl methods).
    out.push_str(&format!("        let __v = {}({});\n", f.rust_call, call_args.join(", ")));
    // Marshal return
    out.push_str(&format!("        {}\n", marshal_out("__v", &f.ret)));
    out.push_str("    });\n");

    // Panic handling
    out.push_str("    match __r {\n");
    out.push_str("        Ok(v) => v,\n");
    out.push_str(&format!(
        "        Err(_) => {{ eprintln!(\"rust panic in shim '{}'\"); std::process::abort(); }}\n",
        f.name
    ));
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

/// Rust type that appears in the shim's C-ABI signature for an INCOMING value.
/// All Rust integer widths are normalised to `i64` at the boundary so the
/// Maka side can use its default `int` type (= int64) without ceremony.
/// Floats are normalised to `f64` (= Maka `float`).
fn rust_marshal_in_ty(ty: &RustType) -> String {
    match ty {
        RustType::Prim(s) => prim_abi_ty(s).to_string(),
        RustType::Bool => "bool".to_string(),
        RustType::Unit => "()".to_string(),
        RustType::StrSlice => "*const c_char".to_string(),
        RustType::OwnedString => "*const c_char".to_string(),
        RustType::ReprC(name) => name.clone(),
        RustType::RefReprC(name) => format!("*const {}", name),
        RustType::RefMutReprC(name) => format!("*mut {}", name),
        RustType::Opaque(_) => "*mut u8".to_string(),
        RustType::RefOpaque(_) => "*const u8".to_string(),
        RustType::RefMutOpaque(_) => "*mut u8".to_string(),
        RustType::RawConstPtr => "*const u8".to_string(),
        RustType::RawMutPtr => "*mut u8".to_string(),
        RustType::OptionOf(inner) => format!("__MakaOpt_{}", sanitise(&rust_name_of(inner))),
        RustType::ResultOf(ok, err) => format!(
            "__MakaRes_{}_{}",
            sanitise(&rust_name_of(ok)),
            sanitise(&rust_name_of(err)),
        ),
        RustType::VecOf(inner) => format!("__MakaVec_{}", sanitise(&rust_name_of(inner))),
        RustType::CFnPtr { params, ret } => format!("extern \"C\" fn({}) -> {}",
            params.iter().map(rust_marshal_in_ty).collect::<Vec<_>>().join(", "), rust_marshal_out_ty(ret)),
    }
}

/// Rust type that appears in the shim's C-ABI return type.
fn rust_marshal_out_ty(ty: &RustType) -> String {
    match ty {
        RustType::Prim(s) => prim_abi_ty(s).to_string(),
        RustType::Bool => "bool".to_string(),
        RustType::Unit => "()".to_string(),
        RustType::StrSlice => "*mut c_char".to_string(),
        RustType::OwnedString => "*mut c_char".to_string(),
        RustType::ReprC(name) => name.clone(),
        RustType::RefReprC(name) => format!("*const {}", name),
        RustType::RefMutReprC(name) => format!("*mut {}", name),
        RustType::Opaque(_) => "*mut u8".to_string(),
        RustType::RefOpaque(_) => "*const u8".to_string(),
        RustType::RefMutOpaque(_) => "*mut u8".to_string(),
        RustType::RawConstPtr => "*const u8".to_string(),
        RustType::RawMutPtr => "*mut u8".to_string(),
        RustType::OptionOf(inner) => format!("__MakaOpt_{}", sanitise(&rust_name_of(inner))),
        RustType::ResultOf(ok, err) => format!(
            "__MakaRes_{}_{}",
            sanitise(&rust_name_of(ok)),
            sanitise(&rust_name_of(err)),
        ),
        RustType::VecOf(inner) => format!("__MakaVec_{}", sanitise(&rust_name_of(inner))),
        RustType::CFnPtr { params, ret } => format!("extern \"C\" fn({}) -> {}",
            params.iter().map(rust_marshal_in_ty).collect::<Vec<_>>().join(", "), rust_marshal_out_ty(ret)),
    }
}

fn prim_abi_ty(s: &str) -> &'static str {
    match s {
        "f32" | "f64" => "f64",
        _ => "i64",
    }
}

/// Returns (binding statement, expression to pass to the user fn).  When the
/// binding is empty, the expression can be inlined directly.
fn unmarshal_in(name: &str, ty: &RustType) -> (String, String) {
    match ty {
        RustType::Prim(p) => {
            // Convert from the normalised C-ABI width (i64 / f64) back to the user's
            // declared primitive.
            ("".into(), format!("({} as {})", name, p))
        }
        RustType::Bool | RustType::Unit | RustType::RawConstPtr | RustType::RawMutPtr => {
            ("".into(), name.to_string())
        }
        RustType::StrSlice => (
            format!(
                "let {n} = unsafe {{ CStr::from_ptr({n}).to_str().unwrap_or(\"\") }};",
                n = name
            ),
            name.to_string(),
        ),
        RustType::OwnedString => (
            // A `String` param is passed BY VALUE (ownership transfers): Maka mints a
            // libc-malloc'd copy and moves it in.  Copy it into a Rust String, then
            // FREE the transferred buffer - otherwise it leaks every call.
            format!(
                "let {n} = unsafe {{ let __v = CStr::from_ptr({n}).to_str().unwrap_or(\"\").to_string(); libc::free({n} as *mut libc::c_void); __v }};",
                n = name
            ),
            name.to_string(),
        ),
        RustType::Opaque(t) => (
            format!("let {n} = unsafe {{ *Box::from_raw({n} as *mut {t}) }};", n = name, t = t),
            name.to_string(),
        ),
        RustType::RefOpaque(t) => (
            format!("let {n} = unsafe {{ &*({n} as *const {t}) }};", n = name, t = t),
            name.to_string(),
        ),
        RustType::RefMutOpaque(t) => (
            format!("let {n} = unsafe {{ &mut *({n} as *mut {t}) }};", n = name, t = t),
            name.to_string(),
        ),
        RustType::ReprC(_) => ("".into(), name.to_string()),
        RustType::RefReprC(_) => (
            format!("let {n} = unsafe {{ &*{n} }};", n = name),
            name.to_string(),
        ),
        RustType::RefMutReprC(_) => (
            format!("let {n} = unsafe {{ &mut *{n} }};", n = name),
            name.to_string(),
        ),
        // Inbound `Option<T>` / `Result<T,E>`: reconstruct from the tagged
        // struct's tag + payload fields.
        RustType::OptionOf(inner) => {
            let inner_ty = rust_name_of(inner);
            (
                format!(
                    "let {n} = if {n}.tag == 0 {{ Some({n}.value as {ty}) }} else {{ None }};",
                    n = name,
                    ty = inner_ty
                ),
                name.to_string(),
            )
        }
        RustType::ResultOf(ok, err) => {
            let ok_ty = rust_name_of(ok);
            let err_ty = rust_name_of(err);
            (
                format!(
                    "let {n} = if {n}.tag == 0 {{ Ok({n}.ok as {ok}) }} else {{ Err({n}.err as {err}) }};",
                    n = name,
                    ok = ok_ty,
                    err = err_ty
                ),
                name.to_string(),
            )
        }
        // Inbound `Vec<T>`: copy from the user-provided libc-malloc'd buffer
        // into a fresh Rust Vec so allocator boundaries stay clean.
        RustType::VecOf(inner) => {
            let inner_ty = rust_name_of(inner);
            (
                // A `Vec<T>` param transfers ownership: copy the elements into a Rust
                // Vec, then FREE the transferred libc buffer (else it leaks each call).
                format!(
                    "let {n} = unsafe {{ let __v = if {n}.ptr.is_null() {{ Vec::new() }} else {{ std::slice::from_raw_parts({n}.ptr as *const {ty}, {n}.len).to_vec() }}; if !{n}.ptr.is_null() {{ libc::free({n}.ptr as *mut libc::c_void); }} __v }};",
                    n = name,
                    ty = inner_ty
                ),
                name.to_string(),
            )
        }
        // A C callback (bare fn pointer) crosses the ABI unchanged.
        RustType::CFnPtr { .. } => ("".into(), name.to_string()),
    }
}

fn marshal_out(value: &str, ty: &RustType) -> String {
    match ty {
        RustType::Prim(p) => format!("({} as {})", value, prim_abi_ty(p)),
        RustType::Bool | RustType::RawConstPtr | RustType::RawMutPtr => {
            format!("{}", value)
        }
        RustType::Unit => format!("{{ {}; () }}", value),
        RustType::StrSlice | RustType::OwnedString => {
            // Allocate via libc::malloc so Maka's libc::free is safe.
            format!(
                "{{ let s = {v}.to_string(); let bytes = s.into_bytes(); let len = bytes.len(); \
                 let buf = unsafe {{ libc::malloc(len + 1) as *mut u8 }}; \
                 if buf.is_null() {{ std::ptr::null_mut() }} else {{ \
                   unsafe {{ std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, len); *buf.add(len) = 0; }} \
                   buf as *mut c_char }} }}",
                v = value
            )
        }
        RustType::Opaque(_) => {
            format!("Box::into_raw(Box::new({})) as *mut u8", value)
        }
        RustType::RefOpaque(_) | RustType::RefMutOpaque(_) => {
            // Can't return references through the C ABI safely.  Treat as opaque pointer.
            format!("({} as *const _) as *mut u8", value)
        }
        RustType::ReprC(_) => format!("{}", value),
        // A bare C fn pointer crosses the ABI unchanged.
        RustType::CFnPtr { .. } => format!("{}", value),
        RustType::RefReprC(name) => format!("({} as *const {})", value, name),
        RustType::RefMutReprC(name) => format!("({} as *const {} as *mut {})", value, name, name),
        RustType::OptionOf(inner) => {
            let inner_ty = rust_name_of(inner);
            let lbl = sanitise(&inner_ty);
            // Default-zero the payload on None so the Maka side reads a
            // deterministic value when `tag == 1`.
            format!(
                "match {v} {{ Some(x) => __MakaOpt_{lbl} {{ tag: 0, value: x as {ty} }}, None => __MakaOpt_{lbl} {{ tag: 1, value: 0 as {ty} }}, }}",
                v = value,
                lbl = lbl,
                ty = inner_ty
            )
        }
        RustType::ResultOf(ok, err) => {
            let ok_ty = rust_name_of(ok);
            let err_ty = rust_name_of(err);
            let ok_lbl = sanitise(&ok_ty);
            let err_lbl = sanitise(&err_ty);
            format!(
                "match {v} {{ Ok(x) => __MakaRes_{ol}_{el} {{ tag: 0, ok: x as {okty}, err: 0 as {errty} }}, Err(e) => __MakaRes_{ol}_{el} {{ tag: 1, ok: 0 as {okty}, err: e as {errty} }}, }}",
                v = value,
                ol = ok_lbl,
                el = err_lbl,
                okty = ok_ty,
                errty = err_ty
            )
        }
        RustType::VecOf(inner) => {
            let inner_ty = rust_name_of(inner);
            let lbl = sanitise(&inner_ty);
            // Copy from Rust's allocator into a libc::malloc'd buffer so
            // Maka can free the bytes with `free(v.ptr)` after use.
            format!(
                "{{ let __v: Vec<{ty}> = {v}; let __len = __v.len(); \
                 if __len == 0 {{ __MakaVec_{lbl} {{ ptr: std::ptr::null_mut(), len: 0, cap: 0 }} }} else {{ \
                 let __bytes = __len * std::mem::size_of::<{ty}>(); \
                 let __ptr = unsafe {{ libc::malloc(__bytes) as *mut {ty} }}; \
                 if !__ptr.is_null() {{ unsafe {{ std::ptr::copy_nonoverlapping(__v.as_ptr(), __ptr, __len); }} }} \
                 __MakaVec_{lbl} {{ ptr: __ptr, len: __len, cap: __len }} }} }}",
                v = value,
                lbl = lbl,
                ty = inner_ty
            )
        }
    }
}

// ------------------------------------------------------------------------
// Maka extern injection

fn build_extern_decl(f: &RustFn) -> ExternDecl {
    let sp = Span { start: 0, end: 0, line: 0, col: 0 };
    let params: Vec<Param> = f
        .params
        .iter()
        .map(|p| Param {
            name: p.name.clone(),
            ty: rust_to_maka_ty(&p.ty, sp, false),
            span: sp,
        })
        .collect();
    let ret = rust_to_maka_ty(&f.ret, sp, true);
    ExternDecl {
        name: f.name.clone(),
        c_name: format!("__maka_shim_{}", f.name),
        params,
        ret,
        is_gate: false,
        is_variadic: false,
        is_pub: false,
        span: sp,
    }
}

/// Maka type for a `#[repr(C)]` struct FIELD.  A param/return primitive is
/// normalised to Maka `int`/`float` (i64/f64) at the ABI boundary and narrowed
/// back inside the shim, which is ergonomic and harmless.  But a by-value struct
/// field is memcpy'd across the ABI with NO per-field conversion, so its Maka
/// mirror MUST have the exact Rust width, or the two structs disagree on size and
/// on the struct-return ABI class and silently corrupt (e.g. `i32` -> Maka `int`
/// makes a 2-field struct 16 bytes on the Maka side vs 8 on the Rust side).
/// Every Rust primitive name (i8/i16/i32/i64/u8/u16/u32/u64/isize/usize/f32/f64)
/// is also a valid Maka sized-type name with the matching C width, so a primitive
/// field maps to the same-named sized type; other field kinds are pointer-width
/// or a nested faithful mirror and defer to the ordinary mapping.
fn rust_to_maka_field_ty(ty: &RustType, sp: Span) -> Type {
    match ty {
        RustType::Prim(s) => Type::Named(s.clone(), sp),
        _ => rust_to_maka_ty(ty, sp, false),
    }
}

fn rust_to_maka_ty(ty: &RustType, sp: Span, is_return: bool) -> Type {
    match ty {
        RustType::Prim(s) => Type::Named(rust_prim_to_maka(s).to_string(), sp),
        RustType::Bool => Type::Named("bool".to_string(), sp),
        RustType::Unit => Type::Unit(sp),
        // `&str` is position-sensitive.  As a PARAM it is a borrowed view: the
        // shim reads it via CStr, no allocation, so Maka `string` (a view) is
        // right.  As a RETURN the shim ALWAYS mallocs an owned copy (see
        // marshal_out, shared with OwnedString), so it must surface as an owned
        // `own *const string` that Maka frees at scope exit - otherwise every
        // `&str`-returning call leaks the copy.
        RustType::StrSlice if is_return => Type::OwnPtr {
            mutness: Mutness::Const,
            inner: Box::new(Type::Named("string".to_string(), sp)),
            span: sp,
        },
        RustType::StrSlice => Type::Named("string".to_string(), sp),
        // Returning a Rust `String` is owned heap → Maka `own *string` (= `String`),
        // a single owned `char*`.  Receiving one: Maka caller passes `string`, shim
        // copies into an owned String.  The codegen path for return values handles
        // ownership via the `own` modifier.
        RustType::OwnedString => Type::OwnPtr {
            mutness: Mutness::Const,
            inner: Box::new(Type::Named("string".to_string(), sp)),
            span: sp,
        },
        // Owned: surfaces as `Rust<T>` (= `own *mut unit`).  This is the form
        // Maka users see when binding a return value.  The label must be the
        // FULL sanitised type name (not the generics-stripped `opaque_label`):
        // it is the identifier the auto-drop shim is keyed on, and stripping
        // `Vec<String>` / `Vec<Vec<i64>>` both to `Vec` collides them AND fails
        // to match the shim, so the generated drop function never links.  Both
        // sides now sanitise the SAME `name`, so the names agree.
        RustType::Opaque(name) => Type::Generic {
            name: "Rust".to_string(),
            args: vec![Type::Named(sanitise(name), sp)],
            span: sp,
        },
        // Borrowed: surfaces as a raw `*const unit` / `*mut unit` on the Maka
        // extern.  Maka coerces `own *mut unit` → `*mut unit` at call sites so
        // the user can write `tick(c)` rather than ceremony — and passing a
        // raw pointer doesn't trip the move check, matching Rust's borrow.
        RustType::RefOpaque(_) => Type::Ptr {
            mutness: Mutness::Const,
            inner: Box::new(Type::Named("unit".to_string(), sp)),
            span: sp,
        },
        RustType::RefMutOpaque(_) => Type::Ptr {
            mutness: Mutness::Mut,
            inner: Box::new(Type::Named("unit".to_string(), sp)),
            span: sp,
        },
        RustType::RawConstPtr => Type::Ptr {
            mutness: Mutness::Const,
            inner: Box::new(Type::Named("unit".to_string(), sp)),
            span: sp,
        },
        RustType::RawMutPtr => Type::Ptr {
            mutness: Mutness::Mut,
            inner: Box::new(Type::Named("unit".to_string(), sp)),
            span: sp,
        },
        // A C callback `R (*)(A, B)` maps to a Maka fn-pointer type; a named Maka
        // function is passed for it (its direct C address matches the C signature).
        RustType::CFnPtr { params, ret } => Type::FnPtr {
            ret: Box::new(rust_to_maka_ty(ret, sp, true)),
            params: params.iter().map(|p| rust_to_maka_ty(p, sp, false)).collect(),
            span: sp,
        },
        RustType::ReprC(name) => Type::Named(name.clone(), sp),
        RustType::RefReprC(name) => Type::Ref {
            mutness: Mutness::Const,
            inner: Box::new(Type::Named(name.clone(), sp)),
            span: sp,
        },
        RustType::RefMutReprC(name) => Type::Ref {
            mutness: Mutness::Mut,
            inner: Box::new(Type::Named(name.clone(), sp)),
            span: sp,
        },
        RustType::OptionOf(inner) => Type::Named(
            format!("__MakaOpt_{}", sanitise(&rust_name_of(inner))),
            sp,
        ),
        RustType::ResultOf(ok, err) => Type::Named(
            format!(
                "__MakaRes_{}_{}",
                sanitise(&rust_name_of(ok)),
                sanitise(&rust_name_of(err))
            ),
            sp,
        ),
        RustType::VecOf(inner) => Type::Named(
            format!("__MakaVec_{}", sanitise(&rust_name_of(inner))),
            sp,
        ),
    }
}

/// Rust type names (which may be Vec<T>, HashMap<String,i32>, etc.) need to
/// become Maka identifiers for the `Rust<...>` phantom argument.  We sanitise:
/// drop generic args, strip path qualifiers.  This is purely a label for the
/// Maka surface — the actual data is opaque.
fn opaque_label(rust_name: &str) -> String {
    // Take up to the first `<` (generic args), then take just the last `::` segment.
    let base = rust_name.split_once('<').map(|(a, _)| a).unwrap_or(rust_name);
    let tail = base.rsplit_once("::").map(|(_, t)| t).unwrap_or(base);
    let cleaned: String = tail.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    if cleaned.is_empty() { "T".to_string() } else { cleaned }
}

fn rust_prim_to_maka(s: &str) -> &'static str {
    // Rust int widths normalise to Maka `int` (= int64) at the boundary, with
    // narrowing inside the shim.  Floats normalise to Maka `float` (= f64).
    // This keeps Maka call sites ergonomic — int literals work without casts.
    match s {
        "f32" | "f64" => "float",
        _ => "int",
    }
}

// ------------------------------------------------------------------------
// Rust signature extraction via `syn`.
//
// We invoke the real Rust parser and walk it for the items we care about:
//   - `pub fn`            → free Maka extern function
//   - `pub struct`        → if `#[repr(C)]`, mirrored as a Maka `data` decl
//                           (fields readable/writable from Maka source)
//   - `impl T { pub fn }` → method exposed as `T_method` free function with
//                           the receiver as first argument
// Everything else is passed through to rustc verbatim and ignored on the
// Maka side (macros, traits, type aliases, doc comments, lifetimes).

#[derive(Debug, Default, Clone)]
struct RustSurface {
    fns: Vec<RustFn>,
    structs: Vec<RustStruct>,
}

#[derive(Debug, Clone)]
struct RustFn {
    /// Mangled name for the bridge: free fns use the bare name; methods use
    /// `TypeName_methodName`.  This is what the Maka extern is called by.
    name: String,
    /// The Rust expression `<TypeName>::<methodName>` (free fns: just `name`).
    rust_call: String,
    params: Vec<RustParam>,
    ret: RustType,
}

#[derive(Debug, Clone)]
struct RustParam {
    name: String,
    ty: RustType,
}

#[derive(Debug, Clone)]
struct RustStruct {
    name: String,
    repr_c: bool,
    fields: Vec<RustStructField>,
}

#[derive(Debug, Clone)]
struct RustStructField {
    name: String,
    ty: RustType,
}

#[derive(Debug, Clone)]
enum RustType {
    Prim(String),         // i32, u64, f32, f64, isize, usize ...
    Bool,
    Unit,
    StrSlice,             // &str
    OwnedString,          // String
    /// A `#[repr(C)] pub struct` we mirror on the Maka side; passes by value
    /// across the C ABI.  Carries the type name so the shim writes
    /// `param: <name>` and the Maka extern declares the matching data type.
    ReprC(String),
    /// `&T` where `T` is a `#[repr(C)]` struct — passes as `*const T`,
    /// unmarshals into a `&T` reference inside the shim.
    RefReprC(String),
    /// `&mut T` where `T` is `#[repr(C)]`.
    RefMutReprC(String),
    Opaque(String),       // any owned named type (Foo, Vec<T>, HashMap<...>)
    RefOpaque(String),    // &T  (T not specially marshalled)
    RefMutOpaque(String), // &mut T
    /// `Option<T>` where `T` is a primitive — mirrored as a C-ABI tagged
    /// struct on both sides.  Maka users read `.tag` (0 = Some, 1 = None)
    /// and `.value`.
    OptionOf(Box<RustType>),
    /// `Result<T, E>` where `T` and `E` are primitives — same mechanism as
    /// `OptionOf`: `.tag` (0 = Ok, 1 = Err), `.value`, `.err`.
    ResultOf(Box<RustType>, Box<RustType>),
    /// `Vec<T>` where `T` is a primitive — `(ptr, len, cap)` struct.  The
    /// shim copies into a `libc::malloc`'d buffer so Maka can free
    /// element memory with its standard `free()`.
    VecOf(Box<RustType>),
    RawConstPtr,          // *const T
    RawMutPtr,            // *mut T
    /// `extern "C" fn(A, B) -> R` — a bare C function pointer (callback).
    /// Mirrors to a Maka fn-pointer type; a named Maka function is passed for it.
    CFnPtr { params: Vec<RustType>, ret: Box<RustType> },
}

/// A field type whose Maka mirror has the SAME layout as the Rust type, so it is
/// safe inside a #[repr(C)] struct.  Owned-heap and opaque-by-value types are NOT
/// identity - they mirror to a differently-sized pointer/struct and corrupt the
/// C ABI.
fn is_repr_c_identity(ty: &RustType) -> bool {
    !matches!(ty,
        RustType::OwnedString | RustType::StrSlice | RustType::VecOf(_)
        | RustType::OptionOf(_) | RustType::ResultOf(..) | RustType::Opaque(_))
}

/// Walk Rust source and build the full surface we expose to Maka.
fn parse_rust_surface(src: &str) -> Result<RustSurface, String> {
    let file: syn::File =
        syn::parse_file(src).map_err(|e| format!("syn parse error: {}", e))?;

    // Pass 1: gather all pub structs + their repr(C) flag and fields.
    // We need the set of #[repr(C)] names BEFORE typing fn signatures so the
    // type classifier can choose ReprC over Opaque for those names.
    let mut structs: Vec<RustStruct> = Vec::new();
    for item in &file.items {
        if let syn::Item::Struct(s) = item {
            if !is_pub(&s.vis) {
                continue;
            }
            let repr_c = has_repr_c(&s.attrs);
            let mut fields: Vec<RustStructField> = Vec::new();
            if let syn::Fields::Named(named) = &s.fields {
                for f in &named.named {
                    let fname = f
                        .ident
                        .as_ref()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "_".to_string());
                    let fty = map_syn_type(&f.ty, &Default::default())?;
                    fields.push(RustStructField { name: fname, ty: fty });
                }
            }
            structs.push(RustStruct {
                name: s.ident.to_string(),
                repr_c,
                fields,
            });
        }
    }

    let known_repr_c: std::collections::HashSet<String> = structs
        .iter()
        .filter(|s| s.repr_c)
        .map(|s| s.name.clone())
        .collect();

    // Re-classify struct field types now that we know which other structs are
    // repr(C). (Self-referential or forward references resolve here.)
    for s in &mut structs {
        for f in &mut s.fields {
            f.ty = reclassify(&f.ty, &known_repr_c);
        }
        // A #[repr(C)] struct is mirrored field-for-field into a C struct.  An
        // owned-heap or opaque field (String / &str / Vec / Option / Result / an
        // opaque by-value type) has a DIFFERENT size and layout in the Maka
        // mirror (e.g. Rust `String` is 24 bytes but mirrors to a bare `char*`),
        // so the two structs disagree on size and struct-return ABI class - the
        // shim writes through an sret pointer the caller never supplied -> a wild
        // write / SEGV.  Reject such fields.
        if s.repr_c {
            for f in &s.fields {
                if !is_repr_c_identity(&f.ty) {
                    return Err(format!(
                        "#[repr(C)] struct `{}`: field `{}` has type `{}`, which is not a C-identity type. Owned/opaque fields (String, &str, Vec, Option, Result, or an opaque struct by value) have a different layout in the Maka mirror and corrupt the C ABI (size + struct-return mismatch). Use primitive, bool, raw-pointer, reference, or other #[repr(C)]-struct fields; return owned data separately or behind an opaque handle.",
                        s.name, f.name, rust_name_of(&f.ty)
                    ));
                }
            }
        }
    }

    // Pass 2: free fns and impl-block methods.
    let mut fns: Vec<RustFn> = Vec::new();
    for item in &file.items {
        match item {
            syn::Item::Fn(f) => {
                if let Some(rfn) = lower_free_fn(f, &known_repr_c)? {
                    fns.push(rfn);
                }
            }
            syn::Item::Impl(im) => {
                if im.trait_.is_some() {
                    continue; // skip trait impls — those are reachable through the type, not the trait
                }
                // Get the impl-target type name (`impl Foo { ... }` → "Foo").
                let recv_name = match &*im.self_ty {
                    syn::Type::Path(tp) => tp
                        .path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string()),
                    _ => None,
                };
                let Some(recv_name) = recv_name else { continue };
                for sub in &im.items {
                    if let syn::ImplItem::Fn(m) = sub {
                        if let Some(rfn) =
                            lower_impl_method(&recv_name, m, &known_repr_c)?
                        {
                            fns.push(rfn);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(RustSurface { fns, structs })
}

fn lower_free_fn(
    f: &syn::ItemFn,
    known_repr_c: &std::collections::HashSet<String>,
) -> Result<Option<RustFn>, String> {
    if !is_pub(&f.vis) {
        return Ok(None);
    }
    if !f.sig.generics.params.is_empty() {
        return Ok(None);
    }
    if f.sig.abi.is_some() {
        return Ok(None);
    }
    let name = f.sig.ident.to_string();
    let mut params: Vec<RustParam> = Vec::new();
    for arg in &f.sig.inputs {
        match arg {
            syn::FnArg::Receiver(_) => {
                return Err(format!(
                    "`{}`: free functions cannot take `self`",
                    name
                ));
            }
            syn::FnArg::Typed(pt) => {
                let pname = pat_to_name(&pt.pat).unwrap_or_else(|| "_arg".to_string());
                let pty = map_syn_type(&pt.ty, known_repr_c)?;
                params.push(RustParam { name: pname, ty: pty });
            }
        }
    }
    let ret = match &f.sig.output {
        syn::ReturnType::Default => RustType::Unit,
        syn::ReturnType::Type(_, t) => map_syn_type(t, known_repr_c)?,
    };
    Ok(Some(RustFn {
        rust_call: name.clone(),
        name,
        params,
        ret,
    }))
}

fn lower_impl_method(
    recv: &str,
    m: &syn::ImplItemFn,
    known_repr_c: &std::collections::HashSet<String>,
) -> Result<Option<RustFn>, String> {
    if !is_pub(&m.vis) {
        return Ok(None);
    }
    if !m.sig.generics.params.is_empty() {
        return Ok(None);
    }
    let method = m.sig.ident.to_string();
    // Mangled extern name: `Type_method`.
    let mangled = format!("{}_{}", recv, method);
    let mut params: Vec<RustParam> = Vec::new();
    let mut receiver_form: Option<&'static str> = None;
    for arg in &m.sig.inputs {
        match arg {
            syn::FnArg::Receiver(r) => {
                // `self` / `&self` / `&mut self`.  Map the implicit type to
                // an explicit parameter of the receiver type.
                let ty = if r.reference.is_some() {
                    if r.mutability.is_some() {
                        receiver_form = Some("&mut");
                        RustType::RefMutOpaque(recv.to_string())
                    } else {
                        receiver_form = Some("&");
                        RustType::RefOpaque(recv.to_string())
                    }
                } else {
                    receiver_form = Some("owned");
                    RustType::Opaque(recv.to_string())
                };
                params.push(RustParam {
                    name: "self_".to_string(),
                    ty: classify_named(ty, known_repr_c),
                });
            }
            syn::FnArg::Typed(pt) => {
                let pname = pat_to_name(&pt.pat).unwrap_or_else(|| "_arg".to_string());
                let pty = map_syn_type(&pt.ty, known_repr_c)?;
                params.push(RustParam { name: pname, ty: pty });
            }
        }
    }
    let _ = receiver_form;
    let ret = match &m.sig.output {
        syn::ReturnType::Default => RustType::Unit,
        syn::ReturnType::Type(_, t) => map_syn_type(t, known_repr_c)?,
    };
    let rust_call = format!("<{} as ::std::default::Default>::default; {}::{}", recv, recv, method);
    // The leading `<T as Default>::default;` is a parser-friendly no-op to keep
    // rustc honest about T being in scope, then the real call uses `T::method`.
    // Simpler: just emit the path.  Replace with the clean form:
    let rust_call = format!("{}::{}", recv, method);
    Ok(Some(RustFn {
        name: mangled,
        rust_call,
        params,
        ret,
    }))
}

/// If `ty` is currently Opaque/RefOpaque/RefMutOpaque on a name that's
/// known repr(C), promote it.
fn classify_named(ty: RustType, known_repr_c: &std::collections::HashSet<String>) -> RustType {
    match ty {
        RustType::Opaque(n) if known_repr_c.contains(&n) => RustType::ReprC(n),
        RustType::RefOpaque(n) if known_repr_c.contains(&n) => RustType::RefReprC(n),
        RustType::RefMutOpaque(n) if known_repr_c.contains(&n) => RustType::RefMutReprC(n),
        other => other,
    }
}

fn reclassify(ty: &RustType, known_repr_c: &std::collections::HashSet<String>) -> RustType {
    match ty {
        RustType::Opaque(n) if known_repr_c.contains(n) => RustType::ReprC(n.clone()),
        RustType::RefOpaque(n) if known_repr_c.contains(n) => RustType::RefReprC(n.clone()),
        RustType::RefMutOpaque(n) if known_repr_c.contains(n) => RustType::RefMutReprC(n.clone()),
        other => other.clone(),
    }
}

fn has_repr_c(attrs: &[syn::Attribute]) -> bool {
    for a in attrs {
        if a.path().is_ident("repr") {
            let mut found = false;
            let _ = a.parse_nested_meta(|m| {
                if m.path.is_ident("C") {
                    found = true;
                }
                Ok(())
            });
            if found {
                return true;
            }
        }
    }
    false
}

fn is_pub(v: &syn::Visibility) -> bool {
    matches!(v, syn::Visibility::Public(_))
}

fn pat_to_name(p: &syn::Pat) -> Option<String> {
    match p {
        syn::Pat::Ident(id) => Some(id.ident.to_string()),
        syn::Pat::Wild(_) => Some("_".to_string()),
        _ => None,
    }
}

/// Classify a `syn::Type` into our marshalling bands.  `known_repr_c` lifts
/// any `Foo` named in that set into `ReprC(Foo)` so the shim passes the
/// struct by value rather than treating it as an opaque pointer.
fn map_syn_type(
    t: &syn::Type,
    known_repr_c: &std::collections::HashSet<String>,
) -> Result<RustType, String> {
    match t {
        syn::Type::Tuple(tt) if tt.elems.is_empty() => Ok(RustType::Unit),
        // `extern "C" fn(A, B) -> R` — a bare C callback the Maka side supplies a
        // function for.  Recursively map the argument and return types.
        syn::Type::BareFn(bf) => {
            let mut params = Vec::new();
            for inp in &bf.inputs { params.push(map_syn_type(&inp.ty, known_repr_c)?); }
            let ret = match &bf.output {
                syn::ReturnType::Default => RustType::Unit,
                syn::ReturnType::Type(_, rt) => map_syn_type(rt, known_repr_c)?,
            };
            Ok(RustType::CFnPtr { params, ret: Box::new(ret) })
        }
        syn::Type::Path(tp) => {
            let last = tp
                .path
                .segments
                .last()
                .ok_or_else(|| "empty type path".to_string())?;
            let ident = last.ident.to_string();
            match ident.as_str() {
                "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "usize" | "f32" | "f64" => Ok(RustType::Prim(ident)),
                "bool" => Ok(RustType::Bool),
                "String" => Ok(RustType::OwnedString),
                "Option" => {
                    let arg = first_generic_arg(&last.arguments)
                        .ok_or_else(|| "Option needs one type argument".to_string())?;
                    let inner = map_syn_type(arg, known_repr_c)?;
                    if marshallable_in_container(&inner) {
                        Ok(RustType::OptionOf(Box::new(inner)))
                    } else {
                        // Fall back to opaque if the inner type isn't a scalar
                        // we can pack into a C-ABI tagged struct.
                        let mut s = String::new();
                        write_syn_type(t, &mut s);
                        Ok(RustType::Opaque(s))
                    }
                }
                "Result" => {
                    let (a, b) = first_two_generic_args(&last.arguments)
                        .ok_or_else(|| "Result needs two type arguments".to_string())?;
                    let ok = map_syn_type(a, known_repr_c)?;
                    let err = map_syn_type(b, known_repr_c)?;
                    if marshallable_in_container(&ok) && marshallable_in_container(&err) {
                        Ok(RustType::ResultOf(Box::new(ok), Box::new(err)))
                    } else {
                        let mut s = String::new();
                        write_syn_type(t, &mut s);
                        Ok(RustType::Opaque(s))
                    }
                }
                "Vec" => {
                    let arg = first_generic_arg(&last.arguments)
                        .ok_or_else(|| "Vec needs one type argument".to_string())?;
                    let inner = map_syn_type(arg, known_repr_c)?;
                    if marshallable_in_container(&inner) {
                        Ok(RustType::VecOf(Box::new(inner)))
                    } else {
                        let mut s = String::new();
                        write_syn_type(t, &mut s);
                        Ok(RustType::Opaque(s))
                    }
                }
                _ if known_repr_c.contains(&ident) => Ok(RustType::ReprC(ident)),
                _ => {
                    let mut s = String::new();
                    write_syn_type(t, &mut s);
                    Ok(RustType::Opaque(s))
                }
            }
        }
        syn::Type::Reference(r) => {
            let inner = &*r.elem;
            if let syn::Type::Path(tp) = inner {
                if let Some(seg) = tp.path.segments.last() {
                    if seg.ident == "str" {
                        return Ok(RustType::StrSlice);
                    }
                    let n = seg.ident.to_string();
                    if known_repr_c.contains(&n) {
                        return Ok(if r.mutability.is_some() {
                            RustType::RefMutReprC(n)
                        } else {
                            RustType::RefReprC(n)
                        });
                    }
                }
            }
            let mut s = String::new();
            write_syn_type(inner, &mut s);
            if r.mutability.is_some() {
                Ok(RustType::RefMutOpaque(s))
            } else {
                Ok(RustType::RefOpaque(s))
            }
        }
        syn::Type::Ptr(p) => {
            if p.mutability.is_some() {
                Ok(RustType::RawMutPtr)
            } else {
                Ok(RustType::RawConstPtr)
            }
        }
        _ => {
            let mut s = String::new();
            write_syn_type(t, &mut s);
            Ok(RustType::Opaque(s))
        }
    }
}

fn first_generic_arg(args: &syn::PathArguments) -> Option<&syn::Type> {
    if let syn::PathArguments::AngleBracketed(ab) = args {
        for a in &ab.args {
            if let syn::GenericArgument::Type(t) = a {
                return Some(t);
            }
        }
    }
    None
}

fn first_two_generic_args(args: &syn::PathArguments) -> Option<(&syn::Type, &syn::Type)> {
    if let syn::PathArguments::AngleBracketed(ab) = args {
        let mut tys: Vec<&syn::Type> = ab
            .args
            .iter()
            .filter_map(|a| if let syn::GenericArgument::Type(t) = a { Some(t) } else { None })
            .collect();
        if tys.len() >= 2 {
            return Some((tys.remove(0), tys.remove(0)));
        }
    }
    None
}

/// Which types are valid as the inner element of a typed container
/// (`Option<T>`, `Result<T, E>`, `Vec<T>`).  Restricted to scalars for now;
/// later stages can allow `#[repr(C)]` structs and even nested containers.
fn marshallable_in_container(ty: &RustType) -> bool {
    matches!(
        ty,
        RustType::Prim(_) | RustType::Bool | RustType::ReprC(_)
    )
}

fn write_syn_type(t: &syn::Type, out: &mut String) {
    match t {
        syn::Type::Tuple(tt) if tt.elems.is_empty() => out.push_str("()"),
        syn::Type::Path(tp) => {
            for (i, seg) in tp.path.segments.iter().enumerate() {
                if i > 0 || tp.path.leading_colon.is_some() {
                    out.push_str("::");
                }
                out.push_str(&seg.ident.to_string());
                if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments {
                    out.push('<');
                    for (j, a) in ab.args.iter().enumerate() {
                        if j > 0 {
                            out.push_str(", ");
                        }
                        match a {
                            syn::GenericArgument::Type(ty) => write_syn_type(ty, out),
                            syn::GenericArgument::Lifetime(lt) => {
                                out.push('\'');
                                out.push_str(&lt.ident.to_string());
                            }
                            _ => out.push('_'),
                        }
                    }
                    out.push('>');
                }
            }
        }
        syn::Type::Reference(r) => {
            out.push('&');
            if r.mutability.is_some() {
                out.push_str("mut ");
            }
            write_syn_type(&r.elem, out);
        }
        syn::Type::Ptr(p) => {
            if p.mutability.is_some() {
                out.push_str("*mut ");
            } else {
                out.push_str("*const ");
            }
            write_syn_type(&p.elem, out);
        }
        syn::Type::Tuple(tt) => {
            out.push('(');
            for (i, e) in tt.elems.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_syn_type(e, out);
            }
            out.push(')');
        }
        syn::Type::Slice(s) => {
            out.push('[');
            write_syn_type(&s.elem, out);
            out.push(']');
        }
        syn::Type::Array(a) => {
            // The length is load-bearing: it is part of the type's SIZE (so the
            // generated `Box::<[T; N]>::from_raw` drop shim needs the real N to
            // compile) AND part of the mangled shim name (dropping it collides
            // `[u8; 4]` with `[u8; 8]`).  Render the const-length expression
            // verbatim rather than erasing it to `_`.
            out.push('[');
            write_syn_type(&a.elem, out);
            out.push_str("; ");
            let len = &a.len;
            out.push_str(&quote::quote!(#len).to_string().replace(' ', ""));
            out.push(']');
        }
        _ => out.push('_'),
    }
}

/// A unique typed-container shape used somewhere in the surface.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
enum ContainerInst {
    /// `Option<T>` for a particular concrete T.
    Option(String),
    /// `Result<T, E>`.
    Result(String, String),
    /// `Vec<T>`.
    Vec(String),
}

/// Walk the surface and collect every container shape we need to define a
/// matching `#[repr(C)]` struct for.  Each shape is generated exactly once,
/// keyed by its element type(s).
fn collect_container_insts(surface: &RustSurface) -> Vec<ContainerInst> {
    let mut seen = std::collections::BTreeSet::new();
    for f in &surface.fns {
        for p in &f.params {
            push_container(&p.ty, &mut seen);
        }
        push_container(&f.ret, &mut seen);
    }
    seen.into_iter().collect()
}

fn push_container(ty: &RustType, out: &mut std::collections::BTreeSet<ContainerInst>) {
    match ty {
        RustType::OptionOf(inner) => {
            out.insert(ContainerInst::Option(rust_name_of(inner)));
            push_container(inner, out);
        }
        RustType::ResultOf(a, b) => {
            out.insert(ContainerInst::Result(rust_name_of(a), rust_name_of(b)));
            push_container(a, out);
            push_container(b, out);
        }
        RustType::VecOf(inner) => {
            out.insert(ContainerInst::Vec(rust_name_of(inner)));
            push_container(inner, out);
        }
        _ => {}
    }
}

/// Stringified Rust type (for use inside generated Rust code).
fn rust_name_of(ty: &RustType) -> String {
    match ty {
        RustType::Prim(s) => s.clone(),
        RustType::Bool => "bool".to_string(),
        RustType::Unit => "()".to_string(),
        RustType::StrSlice => "&str".to_string(),
        RustType::OwnedString => "String".to_string(),
        RustType::ReprC(n) => n.clone(),
        _ => "u64".to_string(),
    }
}

/// Maka-side name component for a container instantiation.  Sanitised so
/// it's a valid Maka identifier (e.g. `int`, `bool`, `V2`).
fn maka_label_of(ty: &RustType) -> String {
    match ty {
        RustType::Prim(_) => "int".to_string(),
        RustType::Bool => "bool".to_string(),
        RustType::ReprC(n) => n.clone(),
        _ => "opaque".to_string(),
    }
}

/// Emit the `#[repr(C)]` Rust struct definition for one container shape.
/// Same layout is mirrored on the Maka side via `inject_container_data`.
fn emit_container_struct(c: &ContainerInst) -> String {
    match c {
        ContainerInst::Option(t) => format!(
            "#[repr(C)] pub struct __MakaOpt_{lbl} {{ pub tag: i64, pub value: {t}, }}\n",
            t = t,
            lbl = sanitise(t),
        ),
        ContainerInst::Result(ok, err) => format!(
            "#[repr(C)] pub struct __MakaRes_{ok_lbl}_{err_lbl} {{ pub tag: i64, pub ok: {ok}, pub err: {err}, }}\n",
            ok = ok,
            err = err,
            ok_lbl = sanitise(ok),
            err_lbl = sanitise(err),
        ),
        ContainerInst::Vec(t) => format!(
            "#[repr(C)] pub struct __MakaVec_{lbl} {{ pub ptr: *mut {t}, pub len: usize, pub cap: usize, }}\n",
            t = t,
            lbl = sanitise(t),
        ),
    }
}

/// Make a Rust type name safe to use as a struct-name suffix.
fn sanitise(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Build Maka `data` decls that mirror each container struct.  Field
/// layouts match the Rust `#[repr(C)]` definitions exactly so values
/// pass by value across the C ABI.
fn build_container_data_decls(insts: &[ContainerInst]) -> Vec<DataDecl> {
    let sp = Span { start: 0, end: 0, line: 0, col: 0 };
    let mut out = Vec::new();
    for c in insts {
        let (name, fields) = match c {
            ContainerInst::Option(t) => {
                let elem_ty = rust_name_to_maka_ty(t, sp);
                (
                    format!("__MakaOpt_{}", sanitise(t)),
                    vec![
                        ("tag".to_string(), Type::Named("int".to_string(), sp)),
                        ("value".to_string(), elem_ty),
                    ],
                )
            }
            ContainerInst::Result(ok, err) => {
                let ok_ty = rust_name_to_maka_ty(ok, sp);
                let err_ty = rust_name_to_maka_ty(err, sp);
                (
                    format!("__MakaRes_{}_{}", sanitise(ok), sanitise(err)),
                    vec![
                        ("tag".to_string(), Type::Named("int".to_string(), sp)),
                        ("ok".to_string(), ok_ty),
                        ("err".to_string(), err_ty),
                    ],
                )
            }
            ContainerInst::Vec(t) => {
                let elem_ty = rust_name_to_maka_ty(t, sp);
                (
                    format!("__MakaVec_{}", sanitise(t)),
                    vec![
                        (
                            // OWNING pointer: makes __MakaVec a move-tracked owning
                            // value.  The shim libc::mallocs the buffer (return) and
                            // libc::frees it when the Vec is passed by value into Rust
                            // (param).  So a by-value Vec is a MOVE - reuse/aliasing is
                            // rejected (was a use-after-free + double-free), and an
                            // un-consumed returned Vec is auto-freed at scope exit
                            // (was a leak).  free() matches libc::malloc.
                            "ptr".to_string(),
                            Type::OwnPtr {
                                mutness: Mutness::Mut,
                                inner: Box::new(elem_ty),
                                span: sp,
                            },
                        ),
                        ("len".to_string(), Type::Named("usize".to_string(), sp)),
                        ("cap".to_string(), Type::Named("usize".to_string(), sp)),
                    ],
                )
            }
        };
        let fields = fields
            .into_iter()
            .map(|(n, ty)| FieldDecl {
                mutness: Mutness::Mut,
                ty,
                name: n,
                default: None,
                is_embed: false,
                span: sp,
            })
            .collect();
        out.push(DataDecl {
            name,
            type_params: Vec::new(),
            fields,
            where_clauses: Vec::new(),
            is_pub: true,
            span: sp,
        });
    }
    out
}

/// Map a Rust scalar / repr-C name to its Maka type for `data` field decls.
fn rust_name_to_maka_ty(name: &str, sp: Span) -> Type {
    match name {
        "f32" | "f64" => Type::Named("float".to_string(), sp),
        "bool" => Type::Named("bool".to_string(), sp),
        _ => Type::Named("int".to_string(), sp),
    }
}

/// Collect the unique Rust type names that appear opaquely in any of the
/// surface's signatures.  These are the candidates for `Send` assertions.
fn collect_send_probes(surface: &RustSurface) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for f in &surface.fns {
        for p in &f.params {
            collect_one(&p.ty, &mut seen);
        }
        collect_one(&f.ret, &mut seen);
    }
    seen.into_iter().collect()
}

fn collect_one(ty: &RustType, out: &mut std::collections::BTreeSet<String>) {
    match ty {
        RustType::Opaque(n) | RustType::RefOpaque(n) | RustType::RefMutOpaque(n) => {
            out.insert(n.clone());
        }
        _ => {}
    }
}

// ------------------------------------------------------------------------
// Maka data decl injection (mirrors a `#[repr(C)] pub struct`).

fn build_data_decl(s: &RustStruct) -> DataDecl {
    let sp = Span { start: 0, end: 0, line: 0, col: 0 };
    let fields: Vec<FieldDecl> = s
        .fields
        .iter()
        .map(|f| FieldDecl {
            mutness: Mutness::Mut,
            ty: rust_to_maka_field_ty(&f.ty, sp),
            name: f.name.clone(),
            default: None,
            is_embed: false,
            span: sp,
        })
        .collect();
    DataDecl {
        name: s.name.clone(),
        type_params: Vec::new(),
        fields,
        where_clauses: Vec::new(),
        is_pub: true,
        span: sp,
    }
}


