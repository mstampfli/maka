//! `maka-lsp` — a Language Server for Maka.  It links the compiler crates and
//! works from the real parsed AST + typed HIR + symbol table, so hover types,
//! go-to-definition, and completion reflect what the compiler actually sees.
//!
//! Features: diagnostics (parse + sema), hover (type of the symbol under the
//! cursor), go-to-definition, document symbols (outline), and completion.
//!
//! `rblock` functions resolve too: the same Rust bridge the compiler uses
//! (`maka_bridge::prepare`) extracts their signatures into extern decls before
//! analysis.  `cblock` functions resolve via their `extern` declaration as usual.

use dashmap::DashMap;
use maka_ast::{EnumDecl, Item, Module};
use maka_lexer::{Span, TokKind};
use maka_sema::hir::{HType, HirModule, SymTab};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

/// The Maka standard library, embedded like the compiler does, so name
/// resolution (Vec, String, Option, ...) works during analysis.
const STDLIB: &str = include_str!("../../../stdlib/std.maka");

/// The parsed stdlib, built exactly once for the process (it never changes), so
/// a cache-miss re-analysis clones it instead of re-lexing the whole file.
fn stdlib_module() -> Module {
    static CELL: OnceLock<Module> = OnceLock::new();
    CELL.get_or_init(|| maka_parser::parse(STDLIB).unwrap_or_default()).clone()
}

/// Parse a source string, memoized by its content hash.  A cache-miss project
/// re-analysis parses many files, but only the file the user just edited has a
/// new hash, so every unchanged file (and the double-parse of the open buffer)
/// is a clone, not a re-parse.  Bounded so a long editing session (each keystroke
/// mints a fresh key) can never grow without limit.
fn parse_cached(text: &str) -> std::result::Result<Module, String> {
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    const CAP: usize = 512;
    static CACHE: OnceLock<Mutex<HashMap<u64, Module>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    let key = h.finish();
    if let Some(m) = cache.lock().unwrap().get(&key) {
        return Ok(m.clone());
    }
    let parsed = maka_parser::parse(text)?;
    let mut g = cache.lock().unwrap();
    if g.len() >= CAP {
        g.clear(); // crude bound: drop everything and re-warm; correctness unaffected
    }
    g.insert(key, parsed.clone());
    Ok(parsed)
}

/// A cross-file top-level definition, for go-to-definition and hover of symbols
/// declared in another project file.
#[derive(Clone)]
struct SymDef {
    name: String,
    uri: Url,
    range: Range,
    detail: String,
}

/// Result of analyzing a document in the context of its whole project.
struct Analysis {
    /// Parse of JUST the open document (user spans), for the outline + locals.
    user_ast: Option<Module>,
    /// Diagnostics for the open document.
    diagnostics: Vec<Diagnostic>,
    /// The typed HIR from the project-wide analysis (None if it failed hard).
    hir: Option<HirModule>,
    /// Every top-level definition across every project file (name -> file+range).
    symbol_index: Vec<SymDef>,
}

struct Backend {
    client: Client,
    docs: DashMap<Url, String>,
    /// Monotonic snapshot version: bumped on every open/change/close.  A cached
    /// `Analysis` is valid only while it matches, so any edit to any buffer (which
    /// can affect any file cross-file) invalidates every cached analysis at once.
    version: AtomicU64,
    /// Per-document cached analysis, tagged with the version it was computed at.
    /// This is the fix for repeated hover / go-to-definition being slow: without
    /// it, *every* request re-parsed the stdlib and the whole project and re-ran
    /// sema on a fresh thread.  Now that work runs once per edit; every subsequent
    /// lookup at the same state is a clone of the shared `Arc<Analysis>`.
    cache: DashMap<Url, (u64, Arc<Analysis>)>,
}

// ---------------------------------------------------------------- analysis

/// Parse + merge stdlib + analyze, on a large-stack thread (the parser/sema
/// recurse per nesting level) with panics caught, so a compiler bug can never
/// take the server down.
fn to_path(uri: &Url) -> Option<std::path::PathBuf> {
    uri.to_file_path().ok()
}
fn path_uri(p: &std::path::Path) -> Option<Url> {
    Url::from_file_path(p).ok()
}

/// The project root: the nearest ancestor directory holding a `maka.toml`, or
/// `None` (so we never scan an unbounded tree from a loose file that happens to
/// live high in the filesystem).
fn find_root(file: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut cur = file.parent();
    let mut hops = 0;
    while let Some(d) = cur {
        if d.join("maka.toml").is_file() {
            return Some(d.to_path_buf());
        }
        hops += 1;
        if hops > 40 {
            break;
        }
        cur = d.parent();
    }
    None
}

const MAX_PROJECT_FILES: usize = 2000;

/// Every `.maka` file under `dir` (skipping hidden / build dirs), recursively,
/// bounded by depth and a file cap so it can never walk away.
fn gather_maka(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>, depth: u32) {
    if out.len() >= MAX_PROJECT_FILES || depth > 12 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        if out.len() >= MAX_PROJECT_FILES {
            return;
        }
        let p = e.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with('.') {
            continue; // .git, .maka_cache, dotfiles
        }
        if p.is_dir() {
            if matches!(name, "target" | "node_modules") {
                continue;
            }
            gather_maka(&p, out, depth + 1);
        } else if p.extension().map_or(false, |x| x == "maka") {
            out.push(p);
        }
    }
}

/// Render a source-level `ast::Type` as written (before resolution), so a
/// generic signature shows its type parameters (`T`, `Vec<T>`) rather than a
/// monomorphized instance's concrete types.
fn ast_type_str(t: &maka_ast::Type) -> String {
    use maka_ast::Type::*;
    let m = |mn: &maka_ast::Mutness| if matches!(mn, maka_ast::Mutness::Mut) { "mut " } else { "" };
    match t {
        Named(n, _) => n.clone(),
        AssocPath { base, segment, .. } => format!("{}::{}", ast_type_str(base), segment),
        Ref { mutness, inner, .. } => format!("&{}{}", m(mutness), ast_type_str(inner)),
        Ptr { mutness, inner, .. } => format!("*{}{}", m(mutness), ast_type_str(inner)),
        RawPtr { mutness, inner, .. } => format!("raw *{}{}", m(mutness), ast_type_str(inner)),
        OwnPtr { mutness, inner, .. } => format!("own *{}{}", m(mutness), ast_type_str(inner)),
        Heap { inner, .. } => format!("own &{}", ast_type_str(inner)),
        Array { len, elem, .. } => format!("[{}]{}", len, ast_type_str(elem)),
        Slice { mutness, elem, .. } => format!("[]{}{}", m(mutness), ast_type_str(elem)),
        Vec { elem, .. } => format!("[*]{}", ast_type_str(elem)),
        Unit(_) => "unit".to_string(),
        Dyn { traits, locked, .. } => {
            format!("{} {}", if *locked { "some" } else { "dyn" }, traits.join(" + "))
        }
        Generic { name, args, .. } => {
            format!("{}<{}>", name, args.iter().map(ast_type_str).collect::<std::vec::Vec<_>>().join(", "))
        }
        FnPtr { ret, params, .. } => {
            format!("{}({})", ast_type_str(ret), params.iter().map(ast_type_str).collect::<std::vec::Vec<_>>().join(", "))
        }
    }
}

/// The full source signature of a function declaration, with type parameters:
/// `T identity<T>(T x)`.
fn func_signature_ast(f: &maka_ast::FuncDecl) -> String {
    let tp = if f.type_params.is_empty() {
        String::new()
    } else {
        format!("<{}>", f.type_params.join(", "))
    };
    let params = f
        .params
        .iter()
        .map(|p| format!("{} {}", ast_type_str(&p.ty), p.name))
        .collect::<std::vec::Vec<_>>()
        .join(", ");
    format!("{} {}{}({})", ast_type_str(&f.ret), f.name, tp, params)
}

/// Render an enum for hover: a multi-line body listing each variant with its
/// payload field names, if any (matching the `data` hover style).  Shared by the
/// in-file resolver and the cross-file symbol index so both show the same detail.
fn enum_signature(e: &EnumDecl) -> String {
    let vs = e
        .variants
        .iter()
        .map(|v| {
            if v.fields.is_empty() {
                format!("    {},", v.name)
            } else {
                let fs = v.fields.iter().map(|f| f.name.clone()).collect::<Vec<_>>().join(", ");
                format!("    {} {{ {} }},", v.name, fs)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("enum {} {{\n{}\n}}", e.name, vs)
}

/// The stdlib's top-level symbols, indexed once against a materialized copy of
/// the embedded source so hover and go-to-definition work for `Vec`, `String`,
/// `Option`, `push`, `str_len`, ... just like user code.  The embedded stdlib is
/// written to a stable cache file (the real source isn't guaranteed on disk at
/// runtime), and definitions point into it.  Compiler-internal `__` names are
/// skipped.
fn stdlib_symbols() -> &'static Vec<SymDef> {
    static CELL: OnceLock<Vec<SymDef>> = OnceLock::new();
    CELL.get_or_init(|| {
        let path = std::env::temp_dir().join("maka-lsp").join("std.maka");
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, STDLIB);
        let Some(uri) = path_uri(&path) else { return Vec::new() };
        let mut out = Vec::new();
        if let Ok(m) = maka_parser::parse(STDLIB) {
            collect_symbols(&m, &uri, &mut out);
        }
        out.retain(|d| !d.name.starts_with("__"));
        out
    })
}

/// Index every top-level definition in a module for cross-file go-to-def/hover.
fn collect_symbols(m: &Module, uri: &Url, out: &mut Vec<SymDef>) {
    let mut add = |name: &str, span: Span, detail: String| {
        out.push(SymDef { name: name.to_string(), uri: uri.clone(), range: span_to_range(span), detail });
    };
    for it in &m.items {
        match it {
            Item::Func(f) => add(&f.name, f.span, func_signature_ast(f)),
            Item::Data(d) => add(&d.name, d.span, format!("data {}", d.name)),
            Item::Enum(e) => {
                add(&e.name, e.span, enum_signature(e));
                // Index each variant too, so `Color.Red` hovers and goes to the
                // variant declaration (first match when a name like `None` recurs).
                for v in &e.variants {
                    let detail = if v.fields.is_empty() {
                        format!("{}.{}", e.name, v.name)
                    } else {
                        let fs = v.fields.iter().map(|f| f.name.clone()).collect::<Vec<_>>().join(", ");
                        format!("{}.{} {{ {} }}", e.name, v.name, fs)
                    };
                    add(&v.name, v.span, detail);
                }
            }
            Item::Attr(a) => add(&a.name, a.span, format!("attr {}", a.name)),
            Item::Global(g) => {
                add(&g.name, g.span, format!("{}{}", if g.is_mut { "mut " } else { "" }, g.name))
            }
            Item::Constexpr(c) => add(&c.name, c.span, format!("constexpr {} = {}", c.name, c.value)),
            _ => {}
        }
    }
}

/// Analyze a document in the context of its WHOLE project: parse the stdlib plus
/// every `.maka` file under the project root (open buffers override disk, so
/// unsaved edits are reflected; unopened files are read from disk), merge exactly
/// as the compiler does, run the rblock bridge, and analyze.  Cross-file types
/// therefore resolve and diagnostics match the build; every definition is indexed
/// for go-to-def.  Runs on a large-stack thread with panics caught.
fn analyze_doc(this_path: Option<std::path::PathBuf>, this_text: String, open: std::collections::HashMap<std::path::PathBuf, String>) -> Analysis {
    let empty = || Analysis { user_ast: None, diagnostics: Vec::new(), hir: None, symbol_index: Vec::new() };
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| analyze_inner(this_path, this_text, open)))
                .unwrap_or_else(|_| Analysis { user_ast: None, diagnostics: Vec::new(), hir: None, symbol_index: Vec::new() })
        });
    match handle {
        Ok(h) => h.join().unwrap_or_else(|_| empty()),
        Err(_) => empty(),
    }
}

fn analyze_inner(this_path: Option<std::path::PathBuf>, this_text: String, open: std::collections::HashMap<std::path::PathBuf, String>) -> Analysis {
    let user_ast = parse_cached(&this_text).ok();

    // Discover project files.  With a `maka.toml` root, every `.maka` under it
    // (bounded); otherwise ONLY the open file's own directory (one readdir, no
    // recursion), so a loose multi-file layout still resolves without ever
    // scanning an unbounded tree.
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = &this_path {
        match find_root(p) {
            Some(root) => gather_maka(&root, &mut files, 0),
            None => {
                if let Some(dir) = p.parent() {
                    if let Ok(rd) = std::fs::read_dir(dir) {
                        for e in rd.flatten() {
                            let q = e.path();
                            if q.is_file() && q.extension().map_or(false, |x| x == "maka") {
                                files.push(q);
                            }
                        }
                    }
                }
            }
        }
        if !files.iter().any(|f| f == p) {
            files.push(p.clone());
        }
    }

    let mut merged = Module::default();
    let mut symbol_index: Vec<SymDef> = Vec::new();
    let mut diagnostics = Vec::new();

    let append = |merged: &mut Module, m: Module| {
        let path = m.module_path.clone().unwrap_or_default();
        let flat: Vec<(Vec<String>, String)> = m
            .imports
            .iter()
            .flat_map(|imp| imp.names.iter().map(|n| (imp.path.clone(), n.clone())))
            .collect();
        let hi = m.has_imports.clone();
        for (idx, _) in m.items.iter().enumerate() {
            // Braced `module Y { ... }` items carry `Y`; the rest the file path.
            merged.item_modules.push(m.item_modules.get(idx).cloned().unwrap_or_else(|| path.clone()));
            merged.item_imports.push(flat.clone());
            merged.item_has_imports.push(hi.clone());
        }
        merged.items.extend(m.items);
    };

    append(&mut merged, stdlib_module());

    for f in &files {
        // Prefer an open buffer (unsaved edits) over the on-disk file.
        let text = match open.get(f) {
            Some(t) => t.clone(),
            None => match std::fs::read_to_string(f) {
                Ok(t) => t,
                Err(_) => continue,
            },
        };
        let is_open_file = this_path.as_ref() == Some(f);
        match parse_cached(&text) {
            Ok(m) => {
                if let Some(uri) = path_uri(f) {
                    collect_symbols(&m, &uri, &mut symbol_index);
                }
                append(&mut merged, m);
            }
            Err(msg) => {
                if is_open_file {
                    diagnostics.push(parse_error_diagnostic(&msg));
                    return Analysis { user_ast, diagnostics, hir: None, symbol_index };
                }
                // A parse error in another file: skip it (can't merge), no diag here.
            }
        }
    }
    // If the open file is not on disk yet (never saved), still analyze its buffer.
    if this_path.is_none() {
        match parse_cached(&this_text) {
            Ok(m) => append(&mut merged, m),
            Err(msg) => {
                diagnostics.push(parse_error_diagnostic(&msg));
                return Analysis { user_ast, diagnostics, hir: None, symbol_index };
            }
        }
    }

    // rblock phase-1 signature extraction (same code the compiler uses).
    if merged.items.iter().any(|it| matches!(it, Item::Rblock(_, _))) {
        let opts = maka_bridge::BridgeOptions { no_rust: false, profile: "dev".into() };
        let root = this_path
            .as_ref()
            .and_then(|p| find_root(p))
            .or_else(|| this_path.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf())))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()));
        if let Ok(prep) = maka_bridge::prepare(&merged, &root, &opts) {
            for (mp, item) in prep.injected {
                merged.items.push(item);
                merged.item_modules.push(mp);
                merged.item_imports.push(Vec::new());
                merged.item_has_imports.push(Vec::new());
            }
        }
    }

    // The analysis spans errors across all files, but only the open document's are
    // ours to publish.  Without file-tagged spans we approximate: keep those whose
    // line is within the open document.  A clean project yields none, so cross-file
    // false errors disappear.
    let this_lines = this_text.lines().count() as u32 + 1;
    let hir = match maka_sema::analyze(&merged) {
        Ok(h) => {
            for w in &h.warnings {
                if w.span.line <= this_lines {
                    diagnostics.push(diag(w.span, DiagnosticSeverity::WARNING, w.msg.clone()));
                }
            }
            Some(h)
        }
        Err(errs) => {
            for e in errs {
                if e.span.line <= this_lines {
                    diagnostics.push(diag(e.span, DiagnosticSeverity::ERROR, e.msg));
                }
            }
            None
        }
    };

    // Style/naming lints for the open file (INFORMATION severity, distinct from
    // compiler diagnostics), from the same crate `makac lint` uses.
    if let Some(m) = &user_ast {
        for f in maka_lint::lint_module_findings(m) {
            // Underline the offending NAME (full width), not the declaration's
            // type/keyword that the raw finding points at.
            let (line, col, width) = maka_lint::locate(&this_text, &f);
            let start = Position { line: line.saturating_sub(1), character: col.saturating_sub(1) };
            diagnostics.push(Diagnostic {
                range: Range { start, end: Position { line: start.line, character: start.character + width.max(1) } },
                severity: Some(DiagnosticSeverity::INFORMATION),
                source: Some("maka-lint".into()),
                message: format!("{} [{}]", f.msg, f.rule),
                ..Default::default()
            });
        }
    }

    // Append the stdlib's symbols last, so a project symbol of the same name
    // still wins the first-match lookup while stdlib names (Vec, String, push,
    // ...) resolve and navigate into the materialized stdlib source.
    symbol_index.extend(stdlib_symbols().iter().cloned());

    Analysis { user_ast, diagnostics, hir, symbol_index }
}

// ---------------------------------------------------------------- positions

/// Refine a declaration position to the range of the NAME on that line (searched
/// from the declaration start), so go-to-definition - and its Ctrl-hover source
/// preview - land on the identifier rather than the type/keyword before it.
/// Falls back to a name-width range at the given position.
fn name_range(text: &str, line1: u32, col1: u32, name: &str) -> Range {
    let l0 = line1.saturating_sub(1);
    let fallback = Range {
        start: Position { line: l0, character: col1.saturating_sub(1) },
        end: Position { line: l0, character: col1.saturating_sub(1) + name.len().max(1) as u32 },
    };
    if name.is_empty() {
        return fallback;
    }
    let Some(line) = text.lines().nth(l0 as usize) else {
        return fallback;
    };
    let bytes = line.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut i = (col1.saturating_sub(1) as usize).min(line.len());
    while let Some(rel) = line.get(i..).and_then(|s| s.find(name)) {
        let s = i + rel;
        let e = s + name.len();
        let before = s == 0 || !is_word(bytes[s - 1]);
        let after = e >= bytes.len() || !is_word(bytes[e]);
        if before && after {
            return Range {
                start: Position { line: l0, character: s as u32 },
                end: Position { line: l0, character: e as u32 },
            };
        }
        i = e.max(s + 1);
    }
    fallback
}

fn span_to_range(sp: Span) -> Range {
    let line = sp.line.saturating_sub(1);
    let col = sp.col.saturating_sub(1);
    // Width from the byte length of the span (base-independent); clamp so a
    // multi-line span does not paint an absurd range on one line.
    let width = (sp.end.saturating_sub(sp.start) as u32).clamp(1, 200);
    Range {
        start: Position { line, character: col },
        end: Position { line, character: col + width },
    }
}

fn diag(sp: Span, severity: DiagnosticSeverity, message: String) -> Diagnostic {
    Diagnostic {
        range: span_to_range(sp),
        severity: Some(severity),
        source: Some("maka".into()),
        message,
        ..Default::default()
    }
}

/// Parse errors carry `... at LINE:COL: ...` in their text; recover a position.
fn parse_error_diagnostic(msg: &str) -> Diagnostic {
    let (mut line, mut col) = (0u32, 0u32);
    if let Some(at) = msg.find(" at ") {
        let rest = &msg[at + 4..];
        if let Some((l, c)) = rest.split_once(':') {
            if let Ok(l) = l.trim().parse::<u32>() {
                let c = c.split(':').next().unwrap_or("1").trim().parse::<u32>().unwrap_or(1);
                line = l.saturating_sub(1);
                col = c.saturating_sub(1);
            }
        }
    }
    Diagnostic {
        range: Range {
            start: Position { line, character: col },
            end: Position { line, character: col + 1 },
        },
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("maka".into()),
        message: msg.to_string(),
        ..Default::default()
    }
}

/// The identifier ([A-Za-z0-9_]) surrounding a 0-based (line, character), if any.
fn word_at(text: &str, pos: Position) -> Option<String> {
    let line = text.lines().nth(pos.line as usize)?;
    let bytes = line.as_bytes();
    let ch = pos.character as usize;
    if ch > bytes.len() {
        return None;
    }
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut start = ch.min(bytes.len());
    while start > 0 && is_word(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = ch;
    while end < bytes.len() && is_word(bytes[end]) {
        end += 1;
    }
    if start == end {
        return None;
    }
    Some(line[start..end].to_string())
}

/// Every word-boundaried occurrence of `name` in the document.  Name-based
/// (does not model scope/shadowing, and may match inside comments/strings), but
/// useful for references / highlight / rename of a distinctively-named symbol.
fn occurrences(text: &str, name: &str) -> Vec<Range> {
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let bytes = line.as_bytes();
        let mut i = 0;
        while let Some(rel) = line[i..].find(name) {
            let start = i + rel;
            let end = start + name.len();
            let before_ok = start == 0 || !is_word(bytes[start - 1]);
            let after_ok = end >= bytes.len() || !is_word(bytes[end]);
            if before_ok && after_ok {
                out.push(Range {
                    start: Position { line: lineno as u32, character: start as u32 },
                    end: Position { line: lineno as u32, character: end as u32 },
                });
            }
            i = end.max(start + 1);
        }
    }
    out
}

/// A `///` doc comment: the run of contiguous `///` lines immediately above the
/// declaration on `line` (1-based), stripped of the `///` prefix and one space,
/// joined as markdown.  Any blank or non-doc line ends the run, so only comments
/// attached to the declaration are shown.  These are ordinary comments to the
/// compiler; the convention is purely a tooling one.
fn doc_above(src: &str, line: u32) -> Option<String> {
    if line == 0 {
        return None;
    }
    let lines: Vec<&str> = src.split('\n').collect();
    let mut idx = (line as usize).min(lines.len()).saturating_sub(1); // 0-based decl line
    let mut docs: Vec<String> = Vec::new();
    while idx > 0 {
        idx -= 1;
        let t = lines[idx].trim_start();
        if let Some(rest) = t.strip_prefix("///") {
            docs.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
        } else {
            break; // a blank line or code ends the doc block
        }
    }
    if docs.is_empty() {
        return None;
    }
    docs.reverse();
    Some(docs.join("\n"))
}

/// Byte offset of a 0-based (line, character) position in `text` (character is
/// treated as a UTF-16 offset, exact for the ASCII that Maka identifiers use).
fn pos_to_byte(text: &str, pos: Position) -> usize {
    let mut byte = 0usize;
    for (i, line) in text.split('\n').enumerate() {
        if i as u32 == pos.line {
            let mut col = 0u32;
            for (bi, ch) in line.char_indices() {
                if col >= pos.character {
                    return byte + bi;
                }
                col += ch.len_utf16() as u32;
            }
            return byte + line.len();
        }
        byte += line.len() + 1; // + '\n'
    }
    byte.min(text.len())
}

/// For signature help: given the cursor, find the innermost enclosing call and
/// which argument the cursor is in.  Token-based (so commas/parens inside strings
/// and comments do not fool it): scan the tokens before the cursor backward,
/// tracking bracket depth, to the unmatched `(` whose preceding token is the
/// callee name; count depth-0 commas after it for the active parameter.
fn enclosing_call(text: &str, pos: Position) -> Option<(String, u32)> {
    let cursor = pos_to_byte(text, pos);
    let tokens = maka_lexer::Lexer::new(text).tokenize().ok()?;
    let toks: Vec<&maka_lexer::Token> = tokens.iter().filter(|t| t.span.start < cursor).collect();
    let mut depth = 0i32;
    let mut commas = 0u32;
    let mut i = toks.len();
    while i > 0 {
        i -= 1;
        match &toks[i].kind {
            TokKind::RParen | TokKind::RBracket | TokKind::RBrace => depth += 1,
            TokKind::LParen if depth == 0 => {
                // The enclosing open paren: the callee is the token before it, or,
                // for a turbofish call `name::<...>(`, the ident before the
                // balanced `<...>` and its `::`.
                let mut ci = i;
                if ci > 0 && matches!(toks[ci - 1].kind, TokKind::Gt) {
                    let mut d = 0i32;
                    let mut k = ci - 1;
                    loop {
                        match toks[k].kind {
                            TokKind::Gt => d += 1,
                            TokKind::Lt => d -= 1,
                            _ => {}
                        }
                        if d == 0 || k == 0 {
                            break;
                        }
                        k -= 1;
                    }
                    // toks[k] is the opening `<`; a turbofish has `::` before it.
                    if d == 0 && k >= 2 && matches!(toks[k - 1].kind, TokKind::ColonColon) {
                        ci = k - 1;
                    } else {
                        return None;
                    }
                }
                if ci > 0 {
                    if let TokKind::Ident(name) = &toks[ci - 1].kind {
                        return Some((name.clone(), commas));
                    }
                }
                return None;
            }
            // An enclosing `[`/`{` at depth 0 means we are in an array literal or a
            // block, not a call's argument list.
            TokKind::LBracket | TokKind::LBrace if depth == 0 => return None,
            TokKind::LParen | TokKind::LBracket | TokKind::LBrace => depth -= 1,
            TokKind::Comma if depth == 0 => commas += 1,
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------- lookups

/// Resolve a name to (hover-markdown, optional definition-span).  Definitions
/// are resolved against the USER document (its spans are correct and it isn't
/// polluted by same-named stdlib symbols); a name found only in the stdlib gets
/// hover but no navigable definition.  `line` is the 0-based cursor line.
fn resolve(user: &Module, hir: &HirModule, name: &str, line: u32) -> Option<(String, Option<Span>)> {
    let sym = &hir.sym;
    // The nearest USER function starting at or before the cursor - its HIR twin
    // (matched by start line) carries the typed locals/params.
    let enclosing_line = user
        .items
        .iter()
        .filter_map(|it| match it {
            Item::Func(f) if f.span.line <= line + 1 => Some(f.span.line),
            _ => None,
        })
        .max();
    if let Some(fl) = enclosing_line {
        if let Some(hf) = sym.funcs.iter().find(|f| f.span.line == fl) {
            if let Some(l) = hf.locals.iter().find(|l| l.name == name) {
                let kw = if l.mut_payload { "mut " } else { "" };
                return Some((code_md(&format!("{}{} {}", kw, render_type(&l.ty, sym), name)), Some(l.span)));
            }
        }
    }
    // A user top-level declaration.  Prefer the HIR twin at the same start line
    // for typed detail; fall back to the bare kind.
    for it in &user.items {
        match it {
            Item::Func(f) if f.name == name => {
                // Render the declared signature from the AST, so a generic
                // function shows its type parameters and `T`-typed params
                // (`T identity<T>(T x)`) rather than a monomorphized instance.
                return Some((code_md(&func_signature_ast(f)), Some(f.span)));
            }
            Item::Data(d) if d.name == name => {
                let md = sym
                    .structs
                    .iter()
                    .find(|s| s.name == name && s.span.line == d.span.line)
                    .map(|s| {
                        let fields = s
                            .fields
                            .iter()
                            .map(|fl| format!("    {} {};", render_type(&fl.ty, sym), fl.name))
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("data {} {{\n{}\n}}", name, fields)
                    })
                    .unwrap_or_else(|| format!("data {}", name));
                return Some((code_md(&md), Some(d.span)));
            }
            Item::Enum(e) if e.name == name => {
                return Some((code_md(&enum_signature(e)), Some(e.span)));
            }
            Item::Global(g) if g.name == name => {
                let kw = if g.is_mut { "mut " } else { "" };
                let ty = sym
                    .globals
                    .iter()
                    .find(|gi| gi.name == name)
                    .map(|gi| render_type(&gi.ty, sym))
                    .unwrap_or_default();
                return Some((code_md(&format!("{}{} {}", kw, ty, name).trim().to_string()), Some(g.span)));
            }
            Item::Constexpr(c) if c.name == name => {
                return Some((code_md(&format!("constexpr int {} = {}", name, c.value)), Some(c.span)));
            }
            _ => {}
        }
    }
    // A stdlib / builtin function: hover only (its definition is not in this file).
    if let Some(sig) = sym.sigs.iter().find(|s| s.name == name && !s.name.starts_with("__")) {
        return Some((code_md(&fn_signature(sig, sym)), None));
    }
    None
}

/// Peel references / pointers to the struct they ultimately point at.
fn struct_id_of(t: &HType) -> Option<maka_sema::hir::StructId> {
    match t {
        HType::Struct(id) => Some(*id),
        HType::Ref { inner, .. }
        | HType::Ptr { inner, .. }
        | HType::OwnPtr { inner, .. }
        | HType::RawPtr { inner, .. }
        | HType::Heap { inner } => struct_id_of(inner),
        _ => None,
    }
}

/// The struct a base identifier resolves to (a local in the enclosing function,
/// or a global), for field-access hover.
fn base_struct_id(user: &Module, hir: &HirModule, base: &str, line: u32) -> Option<maka_sema::hir::StructId> {
    let sym = &hir.sym;
    let enclosing = user
        .items
        .iter()
        .filter_map(|it| match it {
            Item::Func(f) if f.span.line <= line + 1 => Some(f.span.line),
            _ => None,
        })
        .max();
    if let Some(fl) = enclosing {
        if let Some(hf) = sym.funcs.iter().find(|f| f.span.line == fl) {
            if let Some(l) = hf.locals.iter().find(|l| l.name == base) {
                return struct_id_of(&l.ty);
            }
        }
    }
    sym.globals.iter().find(|g| g.name == base).and_then(|g| struct_id_of(&g.ty))
}

/// Hover for a struct/enum field: a field access `base.name` (resolve the base's
/// type and look up the field, so it works for stdlib types too), or a field
/// declaration inside a user `data`/`enum`.  Returns (markdown, def span).
fn resolve_field(
    user: &Module,
    hir: &HirModule,
    text: &str,
    pos: Position,
    name: &str,
) -> Option<(String, Option<Span>)> {
    let sym = &hir.sym;
    // Field access `base.name`: is the hovered word preceded (past spaces) by `.`?
    if let Some(line) = text.lines().nth(pos.line as usize) {
        let bytes = line.as_bytes();
        let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        let mut start = (pos.character as usize).min(bytes.len());
        while start > 0 && is_word(bytes[start - 1]) {
            start -= 1;
        }
        let mut d = start;
        while d > 0 && bytes[d - 1] == b' ' {
            d -= 1;
        }
        if d > 0 && bytes[d - 1] == b'.' {
            let mut e = d - 1;
            while e > 0 && bytes[e - 1] == b' ' {
                e -= 1;
            }
            let mut bs = e;
            while bs > 0 && is_word(bytes[bs - 1]) {
                bs -= 1;
            }
            let base = &line[bs..e];
            if let Some(sid) = base_struct_id(user, hir, base, pos.line) {
                let sinfo = sym.struct_info(sid);
                if let Some(fl) = sinfo.fields.iter().find(|fl| fl.name == name) {
                    let kw = if fl.mut_payload { "mut " } else { "" };
                    let md = format!("{}{} {}", kw, render_type(&fl.ty, sym), name);
                    return Some((code_md(&md), Some(fl.span)));
                }
            }
        }
    }
    // Field declaration inside a user `data` or `enum` variant.
    for it in &user.items {
        match it {
            Item::Data(d) => {
                for fl in &d.fields {
                    if fl.name == name && fl.span.line == pos.line + 1 {
                        let kw = if matches!(fl.mutness, maka_ast::Mutness::Mut) { "mut " } else { "" };
                        return Some((code_md(&format!("{}{} {}", kw, ast_type_str(&fl.ty), name)), Some(fl.span)));
                    }
                }
            }
            Item::Enum(e) => {
                for v in &e.variants {
                    for fl in &v.fields {
                        if fl.name == name && fl.span.line == pos.line + 1 {
                            return Some((code_md(&format!("{} {}", ast_type_str(&fl.ty), name)), Some(fl.span)));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn fn_signature(sig: &maka_sema::hir::FuncSig, sym: &SymTab) -> String {
    let params = sig
        .param_names
        .iter()
        .zip(sig.param_tys.iter())
        .map(|(n, t)| format!("{} {}", render_type(t, sym), n))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} {}({})", render_type(&sig.ret, sym), sig.name, params)
}

/// Render a type for display, resolving struct/enum ids to their names (which
/// `maka_sema::type_str` cannot, since it has no symbol table) and recursing
/// through pointers/containers; scalars fall back to `type_str`.
fn render_type(t: &HType, sym: &SymTab) -> String {
    let generic = |base: &str, args: &[HType]| {
        if args.is_empty() {
            base.to_string()
        } else {
            let a = args.iter().map(|x| render_type(x, sym)).collect::<std::vec::Vec<_>>().join(", ");
            format!("{}<{}>", base, a)
        }
    };
    match t {
        HType::Struct(id) => {
            let s = sym.struct_info(*id);
            generic(s.template.as_deref().unwrap_or(&s.name), &s.template_args)
        }
        HType::Enum(id) => {
            let e = sym.enum_info(*id);
            generic(e.template.as_deref().unwrap_or(&e.name), &e.template_args)
        }
        HType::Ref { mutable, inner } => format!("&{}{}", if *mutable { "mut " } else { "" }, render_type(inner, sym)),
        HType::Ptr { mutable, inner } => format!("*{}{}", if *mutable { "mut " } else { "" }, render_type(inner, sym)),
        HType::RawPtr { mutable, inner } => format!("raw *{}{}", if *mutable { "mut " } else { "" }, render_type(inner, sym)),
        HType::OwnPtr { inner, .. } => format!("own *{}", render_type(inner, sym)),
        HType::Heap { inner } => format!("own &{}", render_type(inner, sym)),
        HType::Vec { elem } => format!("Vec<{}>", render_type(elem, sym)),
        HType::Slice { mutable, elem } => format!("[]{}{}", if *mutable { "mut " } else { "" }, render_type(elem, sym)),
        HType::Array { len, elem } => format!("[{}]{}", len, render_type(elem, sym)),
        _ => maka_sema::type_str(t),
    }
}

fn code_md(code: &str) -> String {
    format!("```maka\n{}\n```", code)
}

// ---------------------------------------------------------------- symbols

fn document_symbols(m: &Module) -> Vec<DocumentSymbol> {
    let mut out = Vec::new();
    let sym = |name: String, kind: SymbolKind, sp: Span, children: Vec<DocumentSymbol>| {
        let range = span_to_range(sp);
        #[allow(deprecated)]
        DocumentSymbol {
            name,
            detail: None,
            kind,
            tags: None,
            deprecated: None,
            range,
            selection_range: range,
            children: if children.is_empty() { None } else { Some(children) },
        }
    };
    for it in &m.items {
        match it {
            Item::Func(f) => out.push(sym(f.name.clone(), SymbolKind::FUNCTION, f.span, vec![])),
            Item::Data(d) => {
                let fields = d
                    .fields
                    .iter()
                    .map(|fl| sym(fl.name.clone(), SymbolKind::FIELD, fl.span, vec![]))
                    .collect();
                out.push(sym(d.name.clone(), SymbolKind::STRUCT, d.span, fields));
            }
            Item::Enum(e) => {
                let variants = e
                    .variants
                    .iter()
                    .map(|v| sym(v.name.clone(), SymbolKind::ENUM_MEMBER, v.span, vec![]))
                    .collect();
                out.push(sym(e.name.clone(), SymbolKind::ENUM, e.span, variants));
            }
            Item::Attr(a) => out.push(sym(a.name.clone(), SymbolKind::INTERFACE, a.span, vec![])),
            Item::Global(g) => out.push(sym(g.name.clone(), SymbolKind::VARIABLE, g.span, vec![])),
            Item::Constexpr(c) => out.push(sym(c.name.clone(), SymbolKind::CONSTANT, c.span, vec![])),
            _ => {}
        }
    }
    out
}

const KEYWORDS: &[&str] = &[
    "if", "else", "while", "for", "in", "break", "continue", "match", "yield", "return",
    "propagate", "unsafe", "data", "enum", "attr", "has", "module", "import", "use",
    "type", "embed", "where", "dyn", "some", "mut", "const", "own", "raw", "pub", "inline",
    "gate", "export", "constexpr", "thread_local", "extern", "alloc", "free", "transfer",
    "share", "as", "cinclude", "cblock", "clink", "rblock", "rdep", "true", "false", "null",
];
const PRIMS: &[&str] = &[
    "int", "float", "bool", "char", "string", "unit", "i8", "i16", "i32", "i64", "isize",
    "u8", "u16", "u32", "u64", "usize", "f32", "f64",
];

/// Compiler builtins - special-cased types and functions with no source location
/// to navigate to.  Hover shows a short description and they appear in completion,
/// so they resolve like the source-defined stdlib.  `(name, is_type, doc)`.
const BUILTINS: &[(&str, bool, &str)] = &[
    ("Vec", true, "Vec<T> - a growable, heap-owned vector. Construct with the `[]` literal; grow with `push`. Frees its buffer (and owning elements) on drop."),
    ("push", false, "push(vec, x) - append `x` to a `Vec<T>`, growing the buffer as needed."),
    ("pop", false, "pop(vec) - remove the last element of a `Vec<T>` and return it as `Option<T>` (`None` if empty)."),
    ("len", false, "v.len - the element count of a `Vec`, array, or slice (a field access)."),
    ("length", false, "s.length() - the character length of a `String` (excludes the trailing NUL)."),
    ("log", false, "log(x) - print a value followed by a newline to stdout."),
    ("panic", false, "panic(msg) - abort the program with a message."),
    ("format", false, "format(fmt, args...) - build an owned `string` from a `{}`-style template."),
    ("tag", false, "value.tag - the discriminant of an enum value, as `int` (the variant index for tagged enums)."),
    ("fields", false, "fields(value) - a compile-time list of a struct's fields, for `inline for (f in fields(v))`."),
    ("alloc", true, "alloc value - the sole heap allocator; the result must land in an `own *T` / `own &T` slot."),
    ("thread", false, "thread(unit() {...}) - spawn a true OS thread (blocking-safe, parallel); returns `*Thread`."),
    ("spawn", false, "spawn(unit() {...}) - spawn a cooperative fiber (concurrent IO on one OS thread); returns `*Thread`."),
    ("job", false, "job(unit() {...}) - run on the work-stealing pool (parallel compute); returns `*Thread`."),
    ("spawn_pool", false, "spawn_pool(unit() {...}) - spawn a fiber onto a background pool; returns `*Thread`."),
    ("join", false, "join(t) - block until the thread/fiber/job finishes; consumes the handle."),
    ("try_join", false, "try_join(t) - non-blocking join; returns whether the handle had finished."),
    ("detach", false, "detach(t) - let a thread run to completion without being joined."),
    ("cancel", false, "cancel(t) - request cancellation of a fiber/job."),
    ("select", false, "select(a, b, ...) - wait for the first of several fibers to complete."),
];

// Semantic-token classification.  We only classify IDENTIFIERS - which the
// TextMate grammar cannot tell apart - and leave keywords / strings / numbers /
// comments to it.  Indices must match `semantic_legend`.
const ST_TYPE: u32 = 0;
const ST_FUNCTION: u32 = 1;
const ST_VARIABLE: u32 = 2;
const ST_PROPERTY: u32 = 3;
const ST_ENUM_MEMBER: u32 = 4;

fn semantic_legend() -> SemanticTokensLegend {
    SemanticTokensLegend {
        token_types: vec![
            SemanticTokenType::TYPE,
            SemanticTokenType::FUNCTION,
            SemanticTokenType::VARIABLE,
            SemanticTokenType::PROPERTY,
            SemanticTokenType::ENUM_MEMBER,
        ],
        token_modifiers: vec![],
    }
}

/// From the index of a `<`, is it a balanced `<...>` immediately followed by
/// `(`?  Distinguishes a generic function `f<T>(` / `f<T>()` from a `<`
/// comparison, so the name can be classified as a function.  Bounded so a stray
/// `<` never scans far.
fn generic_then_paren(tokens: &[maka_lexer::Token], lt: usize) -> bool {
    let mut d = 0i32;
    let end = (lt + 64).min(tokens.len());
    for k in lt..end {
        match tokens[k].kind {
            TokKind::Lt => d += 1,
            TokKind::Gt => {
                d -= 1;
                if d == 0 {
                    return matches!(tokens.get(k + 1).map(|t| &t.kind), Some(TokKind::LParen));
                }
            }
            // A terminator or an unbalanced bracket means this was not a
            // type-argument list.
            TokKind::Semicolon | TokKind::LBrace | TokKind::RBrace | TokKind::RParen => return false,
            _ => {}
        }
    }
    false
}

/// Classify every identifier token into a semantic type, delta-encoded per the
/// LSP protocol.  A member access (`.name`) is a property, or an enum member if
/// PascalCase; a primitive name or any PascalCase name is a type; a name directly
/// before `(` is a function; anything else is a variable.
fn semantic_tokens(text: &str) -> Vec<SemanticToken> {
    let Ok(tokens) = maka_lexer::Lexer::new(text).tokenize() else {
        return Vec::new();
    };
    let mut data = Vec::new();
    let mut prev_line = 0u32;
    let mut prev_start = 0u32;
    for i in 0..tokens.len() {
        let TokKind::Ident(name) = &tokens[i].kind else { continue };
        let after_dot = i > 0 && matches!(tokens[i - 1].kind, TokKind::Dot);
        let next = tokens.get(i + 1).map(|t| &t.kind);
        // A function call/decl: `name(`, a turbofish/qualified `name::`, or a
        // generic `name<...>(` (a balanced `<...>` immediately followed by `(`,
        // which distinguishes a generic function decl/call from a `<` comparison).
        let is_func = matches!(next, Some(TokKind::LParen) | Some(TokKind::ColonColon))
            || (matches!(next, Some(TokKind::Lt)) && generic_then_paren(&tokens, i + 1));
        let is_pascal = name.chars().next().map_or(false, |c| c.is_ascii_uppercase());
        let ttype = if after_dot {
            if is_pascal { ST_ENUM_MEMBER } else { ST_PROPERTY }
        } else if PRIMS.contains(&name.as_str()) {
            ST_TYPE
        } else if is_func {
            ST_FUNCTION
        } else if is_pascal {
            ST_TYPE
        } else {
            ST_VARIABLE
        };
        let sp = tokens[i].span;
        let line = sp.line.saturating_sub(1);
        let start = sp.col.saturating_sub(1);
        let length = sp.end.saturating_sub(sp.start) as u32;
        if length == 0 {
            continue;
        }
        let delta_line = line.saturating_sub(prev_line);
        let delta_start = if delta_line == 0 { start.saturating_sub(prev_start) } else { start };
        data.push(SemanticToken {
            delta_line,
            delta_start,
            length,
            token_type: ttype,
            token_modifiers_bitset: 0,
        });
        prev_line = line;
        prev_start = start;
    }
    data
}

// ---------------------------------------------------------------- server

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: Some(ServerInfo { name: "maka-lsp".into(), version: None }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                completion_provider: Some(CompletionOptions::default()),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".into(), ",".into()]),
                    retrigger_characters: Some(vec![",".into()]),
                    work_done_progress_options: Default::default(),
                }),
                references_provider: Some(OneOf::Left(true)),
                document_highlight_provider: Some(OneOf::Left(true)),
                rename_provider: Some(OneOf::Left(true)),
                document_formatting_provider: Some(OneOf::Left(true)),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                        legend: semantic_legend(),
                        full: Some(SemanticTokensFullOptions::Bool(true)),
                        range: Some(false),
                        work_done_progress_options: Default::default(),
                    }),
                ),
                ..Default::default()
            },
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "maka-lsp ready")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, p: DidOpenTextDocumentParams) {
        let uri = p.text_document.uri.clone();
        self.docs.insert(uri.clone(), p.text_document.text);
        self.bump_version();
        self.publish(uri).await;
    }

    async fn did_change(&self, mut p: DidChangeTextDocumentParams) {
        if let Some(change) = p.content_changes.pop() {
            let uri = p.text_document.uri.clone();
            self.docs.insert(uri.clone(), change.text);
            self.bump_version();
            self.publish(uri).await;
        }
    }

    async fn did_close(&self, p: DidCloseTextDocumentParams) {
        self.docs.remove(&p.text_document.uri);
        self.cache.remove(&p.text_document.uri);
        self.bump_version();
    }

    async fn hover(&self, p: HoverParams) -> Result<Option<Hover>> {
        let pos = p.text_document_position_params.position;
        let uri = p.text_document_position_params.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        let Some(name) = word_at(&text, pos) else {
            return Ok(None);
        };
        // Primitives / keywords: a short note without a full analysis.
        if PRIMS.contains(&name.as_str()) {
            return Ok(Some(hover_md(format!("{} - built-in type", name))));
        }
        let a = self.analyze(&uri);
        // Determine the hover markdown and where the definition lives, so a `///`
        // doc comment written above it can be appended.
        let mut base: Option<String> = None;
        let mut def: Option<(Url, u32)> = None; // (file, 1-based decl line)
        if let (Some(hir), Some(user)) = (&a.hir, &a.user_ast) {
            if let Some((md, sp)) = resolve(user, hir, &name, pos.line) {
                base = Some(md);
                if let Some(sp) = sp {
                    def = Some((uri.clone(), sp.line));
                }
            }
            // A struct/enum field (access `base.field` or a `data`/`enum` field decl).
            if base.is_none() {
                if let Some((md, sp)) = resolve_field(user, hir, &text, pos, &name) {
                    base = Some(md);
                    if let Some(sp) = sp {
                        def = Some((uri.clone(), sp.line));
                    }
                }
            }
        }
        if base.is_none() {
            // A definition in another project file or the stdlib.
            if let Some(d) = a.symbol_index.iter().find(|d| d.name == name) {
                base = Some(code_md(&d.detail));
                def = Some((d.uri.clone(), d.range.start.line + 1));
            }
        }
        let Some(mut md) = base else {
            // A compiler builtin (Vec, push, thread, ...): a description.
            if let Some((_, _, doc)) = BUILTINS.iter().find(|(n, _, _)| *n == name.as_str()) {
                return Ok(Some(hover_md(doc.to_string())));
            }
            return Ok(None);
        };
        // A resolved-but-unlocated symbol (e.g. a stdlib function reached via its
        // signature) may still have a doc in the indexed source; find its location.
        if def.is_none() {
            if let Some(d) = a.symbol_index.iter().find(|d| d.name == name) {
                def = Some((d.uri.clone(), d.range.start.line + 1));
            }
        }
        if let Some((u, line)) = def {
            if let Some(src) = self.source_of(&u) {
                if let Some(doc) = doc_above(&src, line) {
                    md = format!("{}\n\n---\n{}", md, doc);
                }
            }
        }
        Ok(Some(hover_md(md)))
    }

    async fn goto_definition(
        &self,
        p: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let pos = p.text_document_position_params.position;
        let uri = p.text_document_position_params.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        let Some(name) = word_at(&text, pos) else {
            return Ok(None);
        };
        let a = self.analyze(&uri);
        // A local / parameter, a declaration, or a field in THIS file.
        if let (Some(hir), Some(user)) = (&a.hir, &a.user_ast) {
            let hit = resolve(user, hir, &name, pos.line)
                .and_then(|(_, sp)| sp)
                .or_else(|| resolve_field(user, hir, &text, pos, &name).and_then(|(_, sp)| sp));
            if let Some(sp) = hit {
                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri: uri.clone(),
                    range: name_range(&text, sp.line, sp.col, &name),
                })));
            }
        }
        // A top-level definition in another project file (jump to that file);
        // refine the target to the name using that file's source.
        if let Some(d) = a.symbol_index.iter().find(|d| d.name == name) {
            let range = self
                .source_of(&d.uri)
                .map(|src| name_range(&src, d.range.start.line + 1, d.range.start.character + 1, &name))
                .unwrap_or(d.range);
            return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                uri: d.uri.clone(),
                range,
            })));
        }
        Ok(None)
    }

    async fn document_symbol(
        &self,
        p: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = p.text_document.uri;
        if !self.docs.contains_key(&uri) {
            return Ok(None);
        }
        let a = self.analyze(&uri);
        if let Some(m) = &a.user_ast {
            return Ok(Some(DocumentSymbolResponse::Nested(document_symbols(m))));
        }
        Ok(None)
    }

    /// Semantic tokens: token-accurate identifier classification (types /
    /// functions / variables / members) that the TextMate grammar cannot infer.
    async fn semantic_tokens_full(
        &self,
        p: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = p.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: semantic_tokens(&text),
        })))
    }

    /// Signature help: while typing a call's arguments, show the callee's
    /// signature and highlight the parameter being entered.
    async fn signature_help(&self, p: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let pos = p.text_document_position_params.position;
        let uri = p.text_document_position_params.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        let Some((name, active)) = enclosing_call(&text, pos) else {
            return Ok(None);
        };
        let a = self.analyze(&uri);
        let Some(hir) = &a.hir else { return Ok(None) };
        let sym = &hir.sym;
        let mut found: Option<(String, Vec<String>)> = None;
        // Prefer the open file's own function (matched to its HIR twin by start
        // line), so a same-named stdlib overload does not shadow it.
        if let Some(user) = &a.user_ast {
            if let Some(f) = user.items.iter().find_map(|it| match it {
                Item::Func(f) if f.name == name => Some(f),
                _ => None,
            }) {
                // Render from the AST so a generic function shows its type
                // parameters and `T`-typed params, consistent with hover.
                let params: Vec<String> =
                    f.params.iter().map(|p| format!("{} {}", ast_type_str(&p.ty), p.name)).collect();
                found = Some((func_signature_ast(f), params));
            }
        }
        // Otherwise any signature of that name (stdlib / builtin extern).
        if found.is_none() {
            if let Some(s) = sym.sigs.iter().find(|s| s.name == name && !s.name.starts_with("__")) {
                let params: Vec<String> = s
                    .param_names
                    .iter()
                    .zip(s.param_tys.iter())
                    .map(|(n, t)| format!("{} {}", render_type(t, sym), n))
                    .collect();
                let label = format!("{} {}({})", render_type(&s.ret, sym), s.name, params.join(", "));
                found = Some((label, params));
            }
        }
        let Some((label, params)) = found else { return Ok(None) };
        Ok(Some(SignatureHelp {
            signatures: vec![signature_from(label, params, active)],
            active_signature: Some(0),
            active_parameter: Some(active),
        }))
    }

    async fn completion(&self, p: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = p.text_document_position.text_document.uri;
        let has_doc = self.docs.contains_key(&uri);
        let mut items: Vec<CompletionItem> = Vec::new();
        let mut add = |label: &str, kind: CompletionItemKind, detail: Option<String>| {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(kind),
                detail,
                ..Default::default()
            });
        };
        for k in KEYWORDS {
            add(k, CompletionItemKind::KEYWORD, None);
        }
        for t in PRIMS {
            add(t, CompletionItemKind::STRUCT, Some("built-in type".into()));
        }
        for (name, is_type, _doc) in BUILTINS {
            let kind = if *is_type { CompletionItemKind::STRUCT } else { CompletionItemKind::FUNCTION };
            add(name, kind, Some("built-in".into()));
        }
        if has_doc {
            let a = self.analyze(&uri);
            if let Some(hir) = &a.hir {
                for f in &hir.sym.funcs {
                    if f.name.starts_with("__") {
                        continue;
                    }
                    let detail =
                        hir.sym.sigs.iter().find(|s| s.name == f.name).map(|s| fn_signature(s, &hir.sym));
                    add(&f.name, CompletionItemKind::FUNCTION, detail);
                }
                for s in hir.sym.structs.iter().filter(|s| s.template.is_none()) {
                    add(&s.name, CompletionItemKind::STRUCT, None);
                }
                for e in hir.sym.enums.iter().filter(|e| e.template.is_none()) {
                    add(&e.name, CompletionItemKind::ENUM, None);
                }
                for g in &hir.sym.globals {
                    add(&g.name, CompletionItemKind::VARIABLE, Some(render_type(&g.ty, &hir.sym)));
                }
            }
        }
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn references(&self, p: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let pos = p.text_document_position.position;
        let uri = p.text_document_position.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        let Some(name) = word_at(&text, pos) else {
            return Ok(None);
        };
        let locs = occurrences(&text, &name)
            .into_iter()
            .map(|range| Location { uri: uri.clone(), range })
            .collect();
        Ok(Some(locs))
    }

    async fn document_highlight(
        &self,
        p: DocumentHighlightParams,
    ) -> Result<Option<Vec<DocumentHighlight>>> {
        let pos = p.text_document_position_params.position;
        let uri = p.text_document_position_params.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        let Some(name) = word_at(&text, pos) else {
            return Ok(None);
        };
        let hl = occurrences(&text, &name)
            .into_iter()
            .map(|range| DocumentHighlight { range, kind: Some(DocumentHighlightKind::TEXT) })
            .collect();
        Ok(Some(hl))
    }

    async fn rename(&self, p: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let pos = p.text_document_position.position;
        let uri = p.text_document_position.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        let Some(name) = word_at(&text, pos) else {
            return Ok(None);
        };
        let edits: Vec<TextEdit> = occurrences(&text, &name)
            .into_iter()
            .map(|range| TextEdit { range, new_text: p.new_name.clone() })
            .collect();
        let mut changes = std::collections::HashMap::new();
        changes.insert(uri, edits);
        Ok(Some(WorkspaceEdit { changes: Some(changes), ..Default::default() }))
    }

    /// Format Document / format-on-save: layout-only, comment- and
    /// string-preserving (see `maka_fmt`).  Emits a single whole-document edit;
    /// `format_checked` refuses (and we emit no edit) if formatting would ever
    /// change the token stream, so this is safe to run on every save.
    async fn formatting(&self, p: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let uri = p.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        match maka_fmt::format_checked(&text) {
            Ok(out) if out != text => Ok(Some(vec![TextEdit {
                range: Range { start: Position { line: 0, character: 0 }, end: doc_end(&text) },
                new_text: out,
            }])),
            _ => Ok(None), // already formatted, or a safety-check refusal: no edit
        }
    }
}

/// The end position of a document (line = newline count, character = UTF-16
/// length of the final line), so a whole-document replacement range is exact.
fn doc_end(text: &str) -> Position {
    let mut line = 0u32;
    let mut col = 0u32;
    for ch in text.chars() {
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += ch.len_utf16() as u32;
        }
    }
    Position { line, character: col }
}

impl Backend {
    /// Analyze the document at `uri` in the context of its whole project,
    /// snapshotting every OTHER open buffer so unsaved edits across files are
    /// reflected.  Cached per document at the current snapshot version: a repeat
    /// call at the same state (the common case for hover / definition / outline)
    /// returns the shared result without re-analyzing.
    fn analyze(&self, uri: &Url) -> Arc<Analysis> {
        let ver = self.version.load(Ordering::Acquire);
        if let Some(hit) = self.cache.get(uri) {
            if hit.0 == ver {
                return hit.1.clone();
            }
        }
        let this_text = self.docs.get(uri).map(|t| t.clone()).unwrap_or_default();
        let this_path = to_path(uri);
        let mut open: std::collections::HashMap<std::path::PathBuf, String> =
            std::collections::HashMap::new();
        for e in self.docs.iter() {
            if let Some(p) = to_path(e.key()) {
                open.insert(p, e.value().clone());
            }
        }
        let a = Arc::new(analyze_doc(this_path, this_text, open));
        self.cache.insert(uri.clone(), (ver, a.clone()));
        a
    }

    /// Invalidate all cached analyses (any edit can change any file cross-file).
    fn bump_version(&self) {
        self.version.fetch_add(1, Ordering::AcqRel);
    }

    /// The source text of `uri`: the open buffer if any, else read from disk (so
    /// doc comments resolve for cross-file and stdlib definitions too).
    fn source_of(&self, uri: &Url) -> Option<String> {
        if let Some(t) = self.docs.get(uri) {
            return Some(t.clone());
        }
        to_path(uri).and_then(|p| std::fs::read_to_string(p).ok())
    }

    async fn publish(&self, uri: Url) {
        let a = self.analyze(&uri);
        self.client.publish_diagnostics(uri, a.diagnostics.clone(), None).await;
    }
}

fn hover_md(md: String) -> Hover {
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: md,
        }),
        range: None,
    }
}

/// Build a `SignatureInformation` from a rendered signature label and its
/// per-parameter labels, highlighting the active one.
fn signature_from(label: String, params: Vec<String>, active: u32) -> SignatureInformation {
    let parameters = params
        .iter()
        .map(|p| ParameterInformation {
            label: ParameterLabel::Simple(p.clone()),
            documentation: None,
        })
        .collect::<Vec<_>>();
    let active = active.min(params.len().saturating_sub(1) as u32);
    SignatureInformation {
        label,
        documentation: None,
        parameters: Some(parameters),
        active_parameter: Some(active),
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(|client| Backend {
        client,
        docs: DashMap::new(),
        version: AtomicU64::new(0),
        cache: DashMap::new(),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
