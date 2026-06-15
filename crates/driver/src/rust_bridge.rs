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
use std::path::Path;
use std::process::Command;

use maka_ast::{ExternDecl, Item, Module, Mutness, Param, Type};
use maka_lexer::Span;

/// One result of processing the rust bridge for a Maka build.
pub struct BridgeOutput {
    /// Items to splice into the module, paired with the module path each item
    /// belongs to (so the driver can keep `item_modules` in sync).
    pub injected: Vec<(Vec<String>, Item)>,
    /// Static library paths to pass to the C linker.
    pub staticlibs: Vec<String>,
}

/// Top-level entry — walk the merged Maka module, build any needed sidecar
/// crates, and return the externs + staticlibs to splice into the build.
pub fn process(module: &Module, project_root: &Path) -> Result<BridgeOutput, String> {
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

    let mut injected: Vec<(Vec<String>, Item)> = Vec::new();
    let mut staticlibs: Vec<String> = Vec::new();

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

        let combined_rust = bundle.rblocks.join("\n\n// ----- rblock boundary -----\n\n");
        let fns = parse_rust_pub_fns(&combined_rust)
            .map_err(|e| format!("rust signature parse error: {}", e))?;

        // Cache key
        let mut hasher = DefaultHasher::new();
        combined_rust.hash(&mut hasher);
        for (k, v) in &bundle.rdeps {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        let rustc_v = Command::new("rustc")
            .arg("--version")
            .output()
            .map_err(|e| format!("`rustc --version` failed (is rustc installed?): {}", e))?;
        rustc_v.stdout.hash(&mut hasher);
        let hash = format!("{:016x}", hasher.finish());

        // Crate name embeds the hash so a shared `CARGO_TARGET_DIR` produces a
        // distinct staticlib per sidecar (no overwriting across modules/tests).
        let crate_name = format!("{}_{}", sidecar_crate_name(module_path), &hash[..12]);
        let sidecar_dir = project_root.join(".maka_cache").join("rust").join(&hash);
        let shared_target_root = project_root.join(".maka_cache").join("rust").join("_shared_target");
        let staticlib_path = shared_target_root
            .join("release")
            .join(format!("lib{}.a", crate_name));
        let built_marker = sidecar_dir.join(".built");

        if !built_marker.exists() || !staticlib_path.exists() {
            build_sidecar(&sidecar_dir, &crate_name, &combined_rust, &fns, &bundle.rdeps)?;
        }

        staticlibs.push(staticlib_path.to_string_lossy().to_string());

        for f in &fns {
            let extern_decl = build_extern_decl(f);
            injected.push((module_path.clone(), Item::Extern(extern_decl)));
        }
    }

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

fn build_sidecar(
    dir: &Path,
    crate_name: &str,
    rust_src: &str,
    fns: &[RustFn],
    rdeps: &[(String, String)],
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

    // lib.rs = user source + shims
    let mut lib_rs = String::new();
    lib_rs.push_str("// Auto-generated by makac.  Do not hand-edit.\n");
    lib_rs.push_str("#![allow(non_snake_case, unused_imports, dead_code, unused_unsafe, unused_parens, path_statements, unused_variables, unused_assignments, clippy::all)]\n\n");
    lib_rs.push_str(rust_src);
    lib_rs.push_str("\n\n// ===== auto-generated maka shims =====\n\n");
    lib_rs.push_str("use std::os::raw::c_char;\n");
    lib_rs.push_str("use std::ffi::CStr;\n\n");
    for f in fns {
        lib_rs.push_str(&emit_shim(f));
        lib_rs.push('\n');
    }
    std::fs::write(dir.join("src/lib.rs"), &lib_rs).map_err(|e| e.to_string())?;

    eprintln!("makac: building rust sidecar at {}", dir.display());
    // Share one compiled-deps cache across all sidecars so libc / serde etc.
    // build once per Maka workspace, not once per module-hash.
    let shared_target = dir
        .parent()
        .map(|p| p.join("_shared_target"))
        .unwrap_or_else(|| dir.join("target"));
    let status = Command::new("cargo")
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", &shared_target)
        .args(["build", "--release"])
        .status()
        .map_err(|e| format!("cargo invocation failed: {}", e))?;
    if !status.success() {
        return Err(format!(
            "cargo build failed for rblock sidecar at {} (see output above)",
            dir.display()
        ));
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

    // Call user function
    out.push_str(&format!("        let __v = {}({});\n", f.name, call_args.join(", ")));
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
        RustType::Opaque(_) => "*mut u8".to_string(),
        RustType::RefOpaque(_) => "*const u8".to_string(),
        RustType::RefMutOpaque(_) => "*mut u8".to_string(),
        RustType::RawConstPtr => "*const u8".to_string(),
        RustType::RawMutPtr => "*mut u8".to_string(),
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
        RustType::Opaque(_) => "*mut u8".to_string(),
        RustType::RefOpaque(_) => "*const u8".to_string(),
        RustType::RefMutOpaque(_) => "*mut u8".to_string(),
        RustType::RawConstPtr => "*const u8".to_string(),
        RustType::RawMutPtr => "*mut u8".to_string(),
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
            format!(
                "let {n} = unsafe {{ CStr::from_ptr({n}).to_str().unwrap_or(\"\").to_string() }};",
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
            ty: rust_to_maka_ty(&p.ty, sp),
            span: sp,
        })
        .collect();
    let ret = rust_to_maka_ty(&f.ret, sp);
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

fn rust_to_maka_ty(ty: &RustType, sp: Span) -> Type {
    match ty {
        RustType::Prim(s) => Type::Named(rust_prim_to_maka(s).to_string(), sp),
        RustType::Bool => Type::Named("bool".to_string(), sp),
        RustType::Unit => Type::Unit(sp),
        RustType::StrSlice => Type::Named("string".to_string(), sp),
        // Returning a Rust `String` is owned heap → Maka `own *char` (= `String`).
        // Receiving one: Maka caller passes `string`, shim copies into owned String.
        // We declare both directions as `string` in the Maka extern; the codegen
        // path for return values handles ownership via the `own` modifier.
        RustType::OwnedString => Type::OwnPtr {
            mutness: Mutness::Const,
            inner: Box::new(Type::Named("char".to_string(), sp)),
            span: sp,
        },
        RustType::Opaque(_) => Type::OwnPtr {
            mutness: Mutness::Mut,
            inner: Box::new(Type::Named("unit".to_string(), sp)),
            span: sp,
        },
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
    }
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
// We invoke the real Rust parser, then walk it for the items we care about:
// `pub fn` (free functions), `pub struct` (for future `#[repr(C)]` mirroring),
// and `impl T { pub fn ... }` (auto-exposed methods, later stage).  Anything
// else in the Rust source — macros, traits, type aliases, doc comments — is
// passed through to rustc verbatim and ignored on the Maka side.

#[derive(Debug, Clone)]
struct RustFn {
    name: String,
    params: Vec<RustParam>,
    ret: RustType,
}

#[derive(Debug, Clone)]
struct RustParam {
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
    Opaque(String),       // any owned named type (Foo, Vec<T>, HashMap<...>)
    RefOpaque(String),    // &T  (T not specially marshalled)
    RefMutOpaque(String), // &mut T
    RawConstPtr,          // *const T
    RawMutPtr,            // *mut T
}

/// Parse Rust source and extract every visible top-level `pub fn`.
/// Generic functions are skipped (callers must monomorphise on the Rust side
/// and re-expose).  `impl` blocks are not walked here — Stage 3 lifts methods.
fn parse_rust_pub_fns(src: &str) -> Result<Vec<RustFn>, String> {
    let file: syn::File =
        syn::parse_file(src).map_err(|e| format!("syn parse error: {}", e))?;
    let mut out: Vec<RustFn> = Vec::new();
    for item in &file.items {
        if let syn::Item::Fn(f) = item {
            if !is_pub(&f.vis) {
                continue;
            }
            if !f.sig.generics.params.is_empty() {
                continue; // generic function — must be monomorphised by the rblock author
            }
            // Skip already-extern-C functions; they're directly callable.
            if f.sig.abi.is_some() {
                continue;
            }
            let name = f.sig.ident.to_string();
            let mut params: Vec<RustParam> = Vec::new();
            for arg in &f.sig.inputs {
                match arg {
                    syn::FnArg::Receiver(_) => {
                        // `self` argument — only legal in impl, which we don't walk yet.
                        return Err(format!(
                            "`{}`: free functions with `self` are not supported",
                            name
                        ));
                    }
                    syn::FnArg::Typed(pt) => {
                        let pname = pat_to_name(&pt.pat).unwrap_or_else(|| "_arg".to_string());
                        let pty = map_syn_type(&pt.ty)?;
                        params.push(RustParam {
                            name: pname,
                            ty: pty,
                        });
                    }
                }
            }
            let ret = match &f.sig.output {
                syn::ReturnType::Default => RustType::Unit,
                syn::ReturnType::Type(_, t) => map_syn_type(t)?,
            };
            out.push(RustFn { name, params, ret });
        }
    }
    Ok(out)
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

/// Classify a syn::Type into our marshalling bands.
fn map_syn_type(t: &syn::Type) -> Result<RustType, String> {
    match t {
        syn::Type::Tuple(tt) if tt.elems.is_empty() => Ok(RustType::Unit),
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
                _ => {
                    // Reconstruct a printable form for the opaque payload.
                    let mut s = String::new();
                    write_syn_type(t, &mut s);
                    Ok(RustType::Opaque(s))
                }
            }
        }
        syn::Type::Reference(r) => {
            let inner = &*r.elem;
            // `&str`
            if let syn::Type::Path(tp) = inner {
                if let Some(seg) = tp.path.segments.last() {
                    if seg.ident == "str" {
                        return Ok(RustType::StrSlice);
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

/// Reconstruct a textual form of a syn::Type for use in generated shim source.
/// `quote!()` would give us tokens but we don't want to depend on `quote`.
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
            out.push('[');
            write_syn_type(&a.elem, out);
            out.push_str("; _]");
        }
        _ => out.push('_'),
    }
}
