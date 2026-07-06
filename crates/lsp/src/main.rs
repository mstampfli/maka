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
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

/// The Maka standard library, embedded like the compiler does, so name
/// resolution (Vec, String, Option, ...) works during analysis.
const STDLIB: &str = include_str!("../../../stdlib/std.maka");

/// Result of analyzing one document's text.
struct Analysis {
    /// Parse of JUST this document (user spans), for the outline.
    user_ast: Option<Module>,
    /// Compiler diagnostics for this document.
    diagnostics: Vec<Diagnostic>,
    /// The typed HIR from the stdlib-merged analysis (None if it failed hard).
    hir: Option<HirModule>,
}

struct Backend {
    client: Client,
    docs: DashMap<Url, String>,
}

// ---------------------------------------------------------------- analysis

/// Parse + merge stdlib + analyze, on a large-stack thread (the parser/sema
/// recurse per nesting level) with panics caught, so a compiler bug can never
/// take the server down.
fn analyze_text(text: &str) -> Analysis {
    let text = text.to_string();
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| analyze_inner(&text)))
                .unwrap_or_else(|_| Analysis { user_ast: None, diagnostics: Vec::new(), hir: None })
        })
        .expect("spawn analysis thread");
    handle.join().unwrap_or(Analysis { user_ast: None, diagnostics: Vec::new(), hir: None })
}

fn analyze_inner(text: &str) -> Analysis {
    // Parse of the document alone, for the outline + accurate user spans.
    let user_ast = maka_parser::parse(text).ok();

    // Build the merged module: stdlib first (module `std`), then the user file,
    // carrying each item's module path + imports exactly as the driver does.
    let mut merged = Module::default();
    let mut push_all = |m: Module| {
        let path = m.module_path.clone().unwrap_or_default();
        let flat: Vec<(Vec<String>, String)> = m
            .imports
            .iter()
            .flat_map(|imp| imp.names.iter().map(|n| (imp.path.clone(), n.clone())))
            .collect();
        let has_imports = m.has_imports.clone();
        for _ in &m.items {
            merged.item_modules.push(path.clone());
            merged.item_imports.push(flat.clone());
            merged.item_has_imports.push(has_imports.clone());
        }
        merged.items.extend(m.items);
    };
    if let Ok(std_m) = maka_parser::parse(STDLIB) {
        push_all(std_m);
    }

    let mut diagnostics = Vec::new();
    match maka_parser::parse(text) {
        Ok(user_m) => push_all(user_m),
        Err(msg) => {
            diagnostics.push(parse_error_diagnostic(&msg));
            return Analysis { user_ast, diagnostics, hir: None };
        }
    }

    // If the file uses `rblock`, run the Rust bridge's phase-1 signature
    // extraction (the same code the compiler uses) so calls into rblock `pub fn`s
    // resolve.  Guarded on an rblock actually being present, since `prepare`
    // spawns `rustc --version`; a failure (no rustc, malformed rblock) just
    // leaves those calls unresolved rather than breaking the rest.
    if merged.items.iter().any(|it| matches!(it, Item::Rblock(_, _))) {
        let opts = maka_bridge::BridgeOptions { no_rust: false, profile: "dev".into() };
        let root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        if let Ok(prep) = maka_bridge::prepare(&merged, &root, &opts) {
            for (mp, item) in prep.injected {
                merged.items.push(item);
                merged.item_modules.push(mp);
                merged.item_imports.push(Vec::new());
                merged.item_has_imports.push(Vec::new());
            }
        }
    }

    let hir = match maka_sema::analyze(&merged) {
        Ok(h) => {
            for w in &h.warnings {
                diagnostics.push(diag(w.span, DiagnosticSeverity::WARNING, w.msg.clone()));
            }
            Some(h)
        }
        Err(errs) => {
            for e in errs {
                diagnostics.push(diag(e.span, DiagnosticSeverity::ERROR, e.msg));
            }
            None
        }
    };

    // Style/naming lints (STYLE_GUIDE.md), reported as INFORMATION so they read
    // as suggestions distinct from compiler errors/warnings.  Same crate the
    // `makac lint` CLI uses, so the editor and the CLI agree.
    if let Some(m) = &user_ast {
        for f in maka_lint::lint_module_findings(m) {
            let start = Position {
                line: f.line.saturating_sub(1),
                character: f.col.saturating_sub(1),
            };
            diagnostics.push(Diagnostic {
                range: Range { start, end: Position { line: start.line, character: start.character + 1 } },
                severity: Some(DiagnosticSeverity::INFORMATION),
                source: Some("maka-lint".into()),
                message: format!("{} [{}]", f.msg, f.rule),
                ..Default::default()
            });
        }
    }

    Analysis { user_ast, diagnostics, hir }
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
        self.docs.insert(uri.clone(), p.text_document.text.clone());
        self.publish(uri, p.text_document.text).await;
    }

    async fn did_change(&self, mut p: DidChangeTextDocumentParams) {
        if let Some(change) = p.content_changes.pop() {
            let uri = p.text_document.uri.clone();
            self.docs.insert(uri.clone(), change.text.clone());
            self.publish(uri, change.text).await;
        }
    }

    async fn did_close(&self, p: DidCloseTextDocumentParams) {
        self.docs.remove(&p.text_document.uri);
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
        let a = analyze_text(&text);
        if let (Some(hir), Some(user)) = (&a.hir, &a.user_ast) {
            if let Some((md, _)) = resolve(user, hir, &name, pos.line) {
                return Ok(Some(hover_md(md)));
            }
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
        let a = analyze_text(&text);
        if let (Some(hir), Some(user)) = (&a.hir, &a.user_ast) {
            if let Some((_, Some(sp))) = resolve(user, hir, &name, pos.line) {
                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri,
                    range: span_to_range(sp),
                })));
            }
        }
        Ok(None)
    }

    async fn document_symbol(
        &self,
        p: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = p.text_document.uri;
        let Some(text) = self.docs.get(&uri).map(|t| t.clone()) else {
            return Ok(None);
        };
        let a = analyze_text(&text);
        if let Some(m) = &a.user_ast {
            return Ok(Some(DocumentSymbolResponse::Nested(document_symbols(m))));
        }
        Ok(None)
    }

    async fn completion(&self, p: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = p.text_document_position.text_document.uri;
        let text = self.docs.get(&uri).map(|t| t.clone());
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
        if let Some(text) = text {
            let a = analyze_text(&text);
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
}

impl Backend {
    async fn publish(&self, uri: Url, text: String) {
        let a = analyze_text(&text);
        self.client.publish_diagnostics(uri, a.diagnostics, None).await;
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
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
