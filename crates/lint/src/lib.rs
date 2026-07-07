//! `makac lint FILE.maka ...` — a syntactic style checker that enforces the
//! naming conventions in STYLE_GUIDE.md.  It parses each file (no sema needed)
//! and reports declarations whose casing does not match their kind:
//!
//!   type / trait / enum-variant  -> PascalCase
//!   function / method / var / field / param / module segment -> snake_case
//!   module-level constant (immutable global, constexpr) -> SCREAMING_SNAKE_CASE
//!   mutable global -> snake_case
//!
//! Names beginning with `_` (compiler-internal, or the `_` discard) are skipped,
//! as are `extern` symbols (they mirror foreign C names Maka does not own).
//!
//! Exit status is non-zero when any issue is found, so it can gate CI.

use maka_ast::*;
use maka_lexer::Span;

pub struct Finding {
    pub line: u32,
    pub col: u32,
    pub rule: &'static str,
    pub msg: String,
    /// The offending identifier itself, so a consumer with the source can anchor
    /// the underline on the NAME rather than the declaration start (which is the
    /// type or keyword the name follows).  See [`locate`].
    pub name: String,
}

/// Refine a finding to the exact (line, col, width) of the offending NAME, using
/// the source text.  The AST anchors a declaration at its start (the type or the
/// keyword), so a naive underline lands on the type of `Point badName` rather
/// than on `badName`; here we scan the finding's line from the declaration start
/// for the first whole-word occurrence of the name and point there, spanning the
/// whole identifier.  All columns are 1-based bytes (matching `Span`); falls back
/// to the finding's own position (width 1) if the name is not found on the line.
pub fn locate(src: &str, f: &Finding) -> (u32, u32, u32) {
    if f.name.is_empty() {
        return (f.line, f.col, 1);
    }
    let fallback = (f.line, f.col, 1);
    let Some(line) = src.lines().nth(f.line.saturating_sub(1) as usize) else {
        return fallback;
    };
    let bytes = line.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let start = (f.col.saturating_sub(1) as usize).min(line.len());
    let mut i = start;
    while let Some(rel) = line.get(i..).and_then(|s| s.find(f.name.as_str())) {
        let s = i + rel;
        let e = s + f.name.len();
        let before_ok = s == 0 || !is_word(bytes[s - 1]);
        let after_ok = e >= bytes.len() || !is_word(bytes[e]);
        if before_ok && after_ok {
            return (f.line, s as u32 + 1, f.name.len() as u32);
        }
        i = e.max(s + 1);
    }
    fallback
}

/// Lint every path; returns a process exit code (0 = all clean).
pub fn run(paths: &[String]) -> i32 {
    if paths.is_empty() {
        eprintln!("usage: makac lint <file.maka> [more.maka ...]");
        return 2;
    }
    let mut issues = 0usize;
    for path in paths {
        let src = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("lint: cannot read {}: {}", path, e);
                issues += 1;
                continue;
            }
        };
        let module = match maka_parser::parse(&src) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{}: parse error: {}", path, e);
                issues += 1;
                continue;
            }
        };
        let mut found = Vec::new();
        lint_module(&module, &mut found);
        found.sort_by(|a, b| (a.line, a.col).cmp(&(b.line, b.col)));
        for f in &found {
            // Point the report at the name itself, not the declaration's type.
            let (line, col, _w) = locate(&src, f);
            println!("{}:{}:{}: {}  [{}]", path, line, col, f.msg, f.rule);
        }
        issues += found.len();
    }
    if issues == 0 {
        eprintln!("lint: clean");
        0
    } else {
        eprintln!("lint: {} issue(s)", issues);
        1
    }
}

// ------------------------------------------------------------------ predicates

fn first_is(s: &str, f: impl Fn(char) -> bool) -> bool {
    s.chars().next().map_or(false, f)
}
/// PascalCase: starts uppercase, no underscores (acronyms like `TcpConn` pass).
fn is_pascal(s: &str) -> bool {
    first_is(s, |c| c.is_ascii_uppercase()) && !s.contains('_')
}
/// snake_case: starts lowercase, only lowercase / digit / underscore.
fn is_snake(s: &str) -> bool {
    first_is(s, |c| c.is_ascii_lowercase())
        && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}
/// SCREAMING_SNAKE_CASE: starts uppercase, only uppercase / digit / underscore.
fn is_screaming(s: &str) -> bool {
    first_is(s, |c| c.is_ascii_uppercase())
        && s.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}
/// Compiler-internal / discard names are not the user's to style.
fn skip(name: &str) -> bool {
    name.starts_with('_')
}

fn check(name: &str, span: Span, kind: NameKind, out: &mut Vec<Finding>) {
    if skip(name) {
        return;
    }
    let (ok, want, rule): (bool, &str, &'static str) = match kind {
        NameKind::Type => (is_pascal(name), "PascalCase", "naming/type"),
        NameKind::Variant => (is_pascal(name), "PascalCase", "naming/variant"),
        NameKind::Fn => (is_snake(name), "snake_case", "naming/fn"),
        // A binding is snake_case, or SCREAMING_SNAKE_CASE for a local/global
        // constant (a single uppercase letter like `N` qualifies as the latter).
        // This still rejects camelCase / PascalCase, the actual mistakes.
        NameKind::Var => (
            is_snake(name) || is_screaming(name),
            "snake_case (or SCREAMING_SNAKE_CASE for a constant)",
            "naming/var",
        ),
        NameKind::Field => (is_snake(name), "snake_case", "naming/field"),
        NameKind::Module => (is_snake(name), "snake_case", "naming/module"),
        NameKind::Const => (is_screaming(name), "SCREAMING_SNAKE_CASE", "naming/const"),
    };
    if !ok {
        out.push(Finding {
            line: span.line,
            col: span.col,
            rule,
            msg: format!("{} `{}` should be {}", kind.label(), name, want),
            name: name.to_string(),
        });
    }
}

#[derive(Clone, Copy)]
enum NameKind {
    Type,
    Variant,
    Fn,
    Var,
    Field,
    Module,
    Const,
}
impl NameKind {
    fn label(self) -> &'static str {
        match self {
            NameKind::Type => "type",
            NameKind::Variant => "enum variant",
            NameKind::Fn => "function",
            NameKind::Var => "binding",
            NameKind::Field => "field",
            NameKind::Module => "module segment",
            NameKind::Const => "constant",
        }
    }
}

// ------------------------------------------------------------------ walk

/// Collect style/naming findings for a parsed module (used by both the CLI and
/// the language server).
pub fn lint_module_findings(m: &Module) -> Vec<Finding> {
    let mut out = Vec::new();
    lint_module(m, &mut out);
    out
}

fn lint_module(m: &Module, out: &mut Vec<Finding>) {
    if let Some(path) = &m.module_path {
        // A `module a.b.c;` path: every segment is snake_case.  The declaration
        // has no per-segment span, so anchor at the module's first item (or 1:1).
        let span = m.items.first().map(item_span).unwrap_or_else(Span::dummy);
        for seg in path {
            check(seg, span, NameKind::Module, out);
        }
    }
    for it in &m.items {
        lint_item(it, out);
    }
}

fn lint_item(it: &Item, out: &mut Vec<Finding>) {
    match it {
        Item::Data(d) => {
            check(&d.name, d.span, NameKind::Type, out);
            for f in &d.fields {
                check(&f.name, f.span, NameKind::Field, out);
            }
        }
        Item::Enum(e) => {
            check(&e.name, e.span, NameKind::Type, out);
            for v in &e.variants {
                check(&v.name, v.span, NameKind::Variant, out);
                for f in &v.fields {
                    check(&f.name, f.span, NameKind::Field, out);
                }
            }
        }
        Item::Attr(a) => {
            check(&a.name, a.span, NameKind::Type, out);
            for f in &a.funcs {
                lint_func(f, out);
            }
        }
        Item::Logic(l) => {
            check(&l.name, l.span, NameKind::Type, out);
            for f in &l.funcs {
                lint_func(f, out);
            }
        }
        Item::Has(h) => {
            // The receiver/attr names are references, not declarations, so only
            // the method bodies are the impl's own to style.
            for f in &h.funcs {
                lint_func(f, out);
            }
        }
        Item::Func(f) => lint_func(f, out),
        Item::Global(g) => {
            // An immutable global is a constant (SCREAMING_SNAKE_CASE); a mutable
            // one is process state named like an ordinary binding (snake_case).
            let kind = if g.is_mut { NameKind::Var } else { NameKind::Const };
            check(&g.name, g.span, kind, out);
        }
        Item::Constexpr(c) => check(&c.name, c.span, NameKind::Const, out),
        // `extern` mirrors a foreign C symbol; the rest carry no user-named decls.
        _ => {}
    }
}

fn lint_func(f: &FuncDecl, out: &mut Vec<Finding>) {
    check(&f.name, f.span, NameKind::Fn, out);
    for p in &f.params {
        // `self` is the reserved receiver name; skip it.
        if p.name != "self" {
            check(&p.name, p.span, NameKind::Var, out);
        }
    }
    lint_block(&f.body, out);
}

fn lint_block(b: &Block, out: &mut Vec<Finding>) {
    for s in &b.stmts {
        lint_stmt(s, out);
    }
}

fn lint_stmt(s: &Stmt, out: &mut Vec<Finding>) {
    match s {
        Stmt::Let { name, span, .. } => check(name, *span, NameKind::Var, out),
        Stmt::LetTuple { names, span, .. } => {
            for (_, n) in names {
                check(n, *span, NameKind::Var, out);
            }
        }
        Stmt::If { then_block, else_block, .. } => {
            lint_block(then_block, out);
            if let Some(b) = else_block {
                lint_block(b, out);
            }
        }
        Stmt::While { body, .. } => lint_block(body, out),
        Stmt::ForRange { var_name, body, span, .. } => {
            check(var_name, *span, NameKind::Var, out);
            lint_block(body, out);
        }
        Stmt::ForEach { var_name, body, span, .. } => {
            check(var_name, *span, NameKind::Var, out);
            lint_block(body, out);
        }
        Stmt::InlineFor { body, .. } => lint_block(body, out),
        Stmt::Block(b) => lint_block(b, out),
        Stmt::Unsafe(b, _) => lint_block(b, out),
        Stmt::Match { arms, .. } => {
            for a in arms {
                if let ArmBody::Block(b) = &a.body {
                    lint_block(b, out);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn findings(src: &str) -> Vec<Finding> {
        lint_module_findings(&maka_parser::parse(src).expect("parse"))
    }

    #[test]
    fn locate_points_at_name_not_type() {
        // `Point badLocal = ...`: the finding anchors at the decl start (`Point`),
        // but `locate` must move it onto `badLocal` with the name's width.
        let src = "unit f() {\n    Point badLocal = zero();\n}\n";
        let fs = findings(src);
        let bad = fs.iter().find(|f| f.name == "badLocal").expect("flagged");
        let (line, col, width) = locate(src, bad);
        assert_eq!(line, 2);
        // 1-based byte col of `badLocal` in "    Point badLocal = ...".
        assert_eq!(col, "    Point ".len() as u32 + 1);
        assert_eq!(width, "badLocal".len() as u32);
    }

    #[test]
    fn locate_handles_function_and_keyword_decls() {
        let src = "int BadFunc() {\n    return 0;\n}\ndata badType { int x; }\n";
        let fs = findings(src);
        let f = fs.iter().find(|x| x.name == "BadFunc").unwrap();
        assert_eq!(locate(src, f), (1, "int ".len() as u32 + 1, 7)); // on BadFunc, not `int`
        let d = fs.iter().find(|x| x.name == "badType").unwrap();
        assert_eq!(locate(src, d), (4, "data ".len() as u32 + 1, 7)); // on badType, not `data`
    }

    #[test]
    fn locate_falls_back_when_name_absent() {
        // A finding whose name isn't on the given line degrades to its own pos.
        let f = Finding { line: 9, col: 3, rule: "naming/var", msg: String::new(), name: "ghost".into() };
        assert_eq!(locate("only one line\n", &f), (9, 3, 1));
    }

    #[test]
    fn does_not_flag_conforming_names() {
        let src = "int add(int a, int b) {\n    int sum = a + b;\n    return sum;\n}\n";
        assert!(findings(src).is_empty(), "clean code should have no findings");
    }
}

fn item_span(it: &Item) -> Span {
    match it {
        Item::Data(d) => d.span,
        Item::Enum(e) => e.span,
        Item::Attr(a) => a.span,
        Item::Logic(l) => l.span,
        Item::Has(h) => h.span,
        Item::Func(f) => f.span,
        Item::Global(g) => g.span,
        Item::Constexpr(c) => c.span,
        Item::Extern(e) => e.span,
        Item::CInclude(_, s)
        | Item::CBlock(_, s)
        | Item::CLink(_, s)
        | Item::Rblock(_, s)
        | Item::Rdep(_, _, s) => *s,
    }
}
