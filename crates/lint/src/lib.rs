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
            println!("{}:{}:{}: {}  [{}]", path, f.line, f.col, f.msg, f.rule);
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
