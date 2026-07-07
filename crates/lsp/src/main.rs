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
use maka_ast::{Item, Module};
use maka_lexer::Span;
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

/// Index every top-level definition in a module for cross-file go-to-def/hover.
fn collect_symbols(m: &Module, uri: &Url, out: &mut Vec<SymDef>) {
    let mut add = |name: &str, span: Span, detail: String| {
        out.push(SymDef { name: name.to_string(), uri: uri.clone(), range: span_to_range(span), detail });
    };
    for it in &m.items {
        match it {
            Item::Func(f) => add(&f.name, f.span, format!("{}(...)", f.name)),
            Item::Data(d) => add(&d.name, d.span, format!("data {}", d.name)),
            Item::Enum(e) => add(&e.name, e.span, format!("enum {}", e.name)),
            Item::Attr(a) => add(&a.name, a.span, format!("attr {}", a.name)),
            Item::Logic(l) => add(&l.name, l.span, format!("logic {}", l.name)),
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
        for _ in &m.items {
            merged.item_modules.push(path.clone());
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
            let start = Position { line: f.line.saturating_sub(1), character: f.col.saturating_sub(1) };
            diagnostics.push(Diagnostic {
                range: Range { start, end: Position { line: start.line, character: start.character + 1 } },
                severity: Some(DiagnosticSeverity::INFORMATION),
                source: Some("maka-lint".into()),
                message: format!("{} [{}]", f.msg, f.rule),
                ..Default::default()
            });
        }
    }

    Analysis { user_ast, diagnostics, hir, symbol_index }
}

// ---------------------------------------------------------------- positions

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
                // Build the signature from the HIR twin matched by start line, so
                // a same-named stdlib function can't shadow it.
                let md = sym
                    .funcs
                    .iter()
                    .find(|hf| hf.name == name && hf.span.line == f.span.line)
                    .map(|hf| hfunc_signature(hf, sym))
                    .unwrap_or_else(|| format!("{}(...)", name));
                return Some((code_md(&md), Some(f.span)));
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
                let vs = e.variants.iter().map(|v| v.name.clone()).collect::<Vec<_>>().join(", ");
                return Some((code_md(&format!("enum {} {{ {} }}", name, vs)), Some(e.span)));
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

fn hfunc_signature(hf: &maka_sema::hir::HFunc, sym: &SymTab) -> String {
    let params = hf
        .params
        .iter()
        .map(|lid| {
            let l = &hf.locals[lid.0 as usize];
            format!("{} {}", render_type(&l.ty, sym), l.name)
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} {}({})", render_type(&hf.ret, sym), hf.name, params)
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
            Item::Logic(l) => out.push(sym(l.name.clone(), SymbolKind::INTERFACE, l.span, vec![])),
            Item::Global(g) => out.push(sym(g.name.clone(), SymbolKind::VARIABLE, g.span, vec![])),
            Item::Constexpr(c) => out.push(sym(c.name.clone(), SymbolKind::CONSTANT, c.span, vec![])),
            _ => {}
        }
    }
    out
}

const KEYWORDS: &[&str] = &[
    "if", "else", "while", "for", "in", "break", "continue", "match", "yield", "return",
    "propagate", "unsafe", "data", "enum", "attr", "has", "logic", "module", "import", "use",
    "type", "embed", "where", "dyn", "some", "mut", "const", "own", "raw", "pub", "inline",
    "gate", "export", "constexpr", "thread_local", "extern", "alloc", "free", "transfer",
    "share", "as", "cinclude", "cblock", "clink", "rblock", "rdep", "true", "false", "null",
];
const PRIMS: &[&str] = &[
    "int", "float", "bool", "char", "string", "unit", "i8", "i16", "i32", "i64", "isize",
    "u8", "u16", "u32", "u64", "usize", "f32", "f64",
];

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
                references_provider: Some(OneOf::Left(true)),
                document_highlight_provider: Some(OneOf::Left(true)),
                rename_provider: Some(OneOf::Left(true)),
                document_formatting_provider: Some(OneOf::Left(true)),
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
            return Ok(Some(hover_md(format!("{} — built-in type", name))));
        }
        let a = self.analyze(&uri);
        // Open-file local/param/decl (typed via the HIR).
        if let (Some(hir), Some(user)) = (&a.hir, &a.user_ast) {
            if let Some((md, _)) = resolve(user, hir, &name, pos.line) {
                return Ok(Some(hover_md(md)));
            }
        }
        // A definition in another project file.
        if let Some(d) = a.symbol_index.iter().find(|d| d.name == name) {
            return Ok(Some(hover_md(code_md(&d.detail))));
        }
        Ok(None)
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
        // A local / parameter, or a declaration in THIS file.
        if let (Some(hir), Some(user)) = (&a.hir, &a.user_ast) {
            if let Some((_, Some(sp))) = resolve(user, hir, &name, pos.line) {
                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri: uri.clone(),
                    range: span_to_range(sp),
                })));
            }
        }
        // A top-level definition in another project file (jump to that file).
        if let Some(d) = a.symbol_index.iter().find(|d| d.name == name) {
            return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                uri: d.uri.clone(),
                range: d.range,
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
