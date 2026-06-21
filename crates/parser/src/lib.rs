//! Maka recursive-descent parser (spec v1.2).

use maka_ast::*;
use maka_lexer::{Span, TokKind, Token};
use std::fmt;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub msg: String,
    pub span: Span,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at {}: {}", self.span, self.msg)
    }
}

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
    constexprs: std::collections::HashMap<String, i64>,
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Self { toks, pos: 0, constexprs: std::collections::HashMap::new() }
    }

    pub fn parse_module(mut self) -> Result<Module, ParseError> {
        // Optional `module path.name;`
        let mut module_path: Option<Vec<String>> = None;
        if self.eat(&TokKind::Module) {
            module_path = Some(self.parse_dotted_path()?);
            self.expect(&TokKind::Semicolon, "`;`")?;
        }
        // Optional `import path.name [as alias];` declarations.  Captured into the
        // module's import list so the visibility checker can use them.
        let mut imports: Vec<ImportDecl> = Vec::new();
        while self.at(&TokKind::Import) {
            self.bump();
            // Parse a dotted path; the last segment may be a single name or a `{ ... }`
            // selective list.  Examples:
            //   import a.b.c;                  → path=[a,b], names=[c]   (single)
            //   import a.b.c as x;             → path=[a,b], names=[x]   (aliased; we keep alias as the bound name)
            //   import a.b.{x, y};             → path=[a,b], names=[x,y] (selective list)
            //   import a;                      → path=[],    names=[a]   (whole module by name)
            let path0 = self.parse_dotted_path()?;
            let mut path = path0.clone();
            let mut names: Vec<String> = Vec::new();
            // `.{x, y}` selective list — consume the dot first, then the brace.
            if self.at(&TokKind::Dot) && matches!(self.peek_at(1), TokKind::LBrace) {
                self.bump(); // .
            }
            // `.*` wildcard — bring every `pub` item from the named module into
            // scope.  The resolver expands the `*` into the concrete names at
            // visibility-check time.
            if self.at(&TokKind::Dot) && matches!(self.peek_at(1), TokKind::Star) {
                self.bump(); // .
                self.bump(); // *
                names.push("*".to_string());
                self.expect(&TokKind::Semicolon, "`;`")?;
                imports.push(ImportDecl { path, names });
                continue;
            }
            if self.eat(&TokKind::LBrace) {
                if !self.at(&TokKind::RBrace) {
                    loop {
                        let (n, _) = self.expect_ident("imported name")?;
                        names.push(n);
                        if !self.eat(&TokKind::Comma) { break; }
                        if self.at(&TokKind::RBrace) { break; }
                    }
                }
                self.expect(&TokKind::RBrace, "`}`")?;
            } else {
                // path's last segment is the imported name.  Optional `as alias`.
                let last = path.pop().unwrap_or_default();
                let mut bound = last;
                if let TokKind::Ident(i) = self.peek() {
                    if i == "as" {
                        self.bump();
                        let (alias, _) = self.expect_ident("import alias")?;
                        bound = alias;
                    }
                }
                names.push(bound);
            }
            self.expect(&TokKind::Semicolon, "`;`")?;
            imports.push(ImportDecl { path, names });
        }
        // `use ModPath.Type.Attr;` declarations — explicit propagation of `pub has` impls
        // from another module. Allowed in the same prelude region as `import`.
        let mut has_imports: Vec<HasImport> = Vec::new();
        while self.at(&TokKind::Use) {
            let kw = self.bump();
            let path = self.parse_dotted_path()?;
            if path.len() < 3 {
                return Err(ParseError {
                    msg: "`use` requires at least `Module.Type.Attr` (one module segment + Type + Attr)".into(),
                    span: kw.span,
                });
            }
            let attr_name = path[path.len() - 1].clone();
            let type_name = path[path.len() - 2].clone();
            let module_path: Vec<String> = path[..path.len() - 2].to_vec();
            self.expect(&TokKind::Semicolon, "`;`")?;
            has_imports.push(HasImport { module_path, type_name, attr_name, span: kw.span });
        }
        // Pre-pass: scan tokens from current position for top-level `constexpr T NAME = INT;` decls.
        // Bracket-depth-aware so we only pick up module-scope ones, not anything nested.
        self.prescan_constexprs();
        let mut items = Vec::new();
        while !self.at(&TokKind::Eof) {
            // Optional `pub` modifier on top-level items.
            let is_pub = self.eat(&TokKind::Pub);
            // Module-scope constexpr decls.  The pre-scan already captured the
            // value into our fold map (so array sizes and in-file folds work);
            // we additionally emit an `Item::Constexpr` so the resolver can
            // register it as a cross-module-importable symbol.
            if self.at(&TokKind::Constexpr) {
                if let Some(decl) = self.parse_constexpr_decl(is_pub)? {
                    items.push(Item::Constexpr(decl));
                }
                continue;
            }
            let mut item = self.parse_item()?;
            // Stamp the `pub` flag onto whatever decl came back.
            match &mut item {
                Item::Func(f)   => f.is_pub = is_pub,
                Item::Data(d)   => d.is_pub = is_pub,
                Item::Enum(e)   => e.is_pub = is_pub,
                Item::Extern(e) => e.is_pub = is_pub,
                Item::Logic(l)  => l.is_pub = is_pub,
                Item::Attr(a)   => a.is_pub = is_pub,
                Item::Has(h)    => h.is_pub = is_pub,
                Item::Global(g) => g.is_pub = is_pub,
                _ => {}
            }
            items.push(item);
        }
        Ok(Module {
            items, module_path,
            item_modules: Vec::new(),
            item_imports: Vec::new(),
            imports,
            has_imports,
            item_has_imports: Vec::new(),
        })
    }

    // Pre-scan: walk forward from `self.pos`, find every module-scope
    // `constexpr T NAME = <int-expr>;` and record its folded value.
    // Stays at the same position (saves and restores).
    fn prescan_constexprs(&mut self) {
        let save = self.pos;
        let mut depth = 0i32;
        while self.pos < self.toks.len() && !matches!(self.toks[self.pos].kind, TokKind::Eof) {
            match &self.toks[self.pos].kind {
                TokKind::LBrace | TokKind::LParen | TokKind::LBracket => { depth += 1; self.pos += 1; }
                TokKind::RBrace | TokKind::RParen | TokKind::RBracket => { depth -= 1; self.pos += 1; }
                TokKind::Constexpr if depth == 0 => {
                    self.pos += 1;
                    // Skip type tokens until we see an Ident followed by `=`.
                    while self.pos < self.toks.len() {
                        if let TokKind::Ident(name) = &self.toks[self.pos].kind {
                            let nm = name.clone();
                            if matches!(self.peek_at(1), TokKind::Eq) {
                                self.pos += 2; // past name and `=`
                                if let Some(v) = self.try_fold_int() {
                                    self.constexprs.insert(nm, v);
                                }
                                // Skip to `;`.
                                while self.pos < self.toks.len()
                                    && !matches!(self.toks[self.pos].kind, TokKind::Semicolon | TokKind::Eof)
                                { self.pos += 1; }
                                break;
                            }
                        }
                        if matches!(self.toks[self.pos].kind, TokKind::Semicolon | TokKind::Eof) { break; }
                        self.pos += 1;
                    }
                }
                _ => self.pos += 1,
            }
        }
        self.pos = save;
    }

    // Parse a module-scope `constexpr T NAME = expr;` declaration and return
    // a ConstexprDecl (already-folded value).  The pre-scan path computed the
    // value; here we just walk the tokens to extract the name and reach `;`.
    fn parse_constexpr_decl(&mut self, is_pub: bool) -> Result<Option<ConstexprDecl>, ParseError> {
        let kw_span = self.peek_span();
        self.bump();                          // `constexpr`
        // Skip the type tokens until we find the name (Ident followed by `=`).
        let mut name: Option<String> = None;
        while self.pos < self.toks.len() && !matches!(self.peek(), TokKind::Semicolon | TokKind::Eof) {
            if let TokKind::Ident(n) = self.peek().clone() {
                if matches!(self.peek_at(1), TokKind::Eq) {
                    name = Some(n);
                    break;
                }
            }
            self.bump();
        }
        // Walk to the semicolon - the value was already folded during pre-scan
        // and lives in `self.constexprs`.
        while !matches!(self.peek(), TokKind::Semicolon | TokKind::Eof) { self.bump(); }
        self.expect(&TokKind::Semicolon, "`;`")?;
        if let Some(n) = name {
            if let Some(&v) = self.constexprs.get(&n) {
                return Ok(Some(ConstexprDecl { name: n, value: v, is_pub, span: kw_span }));
            }
        }
        Ok(None)
    }

    // Try to fold a constexpr-int expression starting at self.pos: literals, idents (looked up
    // in self.constexprs), and `+ - *` binops with normal left-to-right precedence (no parens here).
    // Stops at `;` or `]` and advances self.pos to that delimiter. Returns Some(value) on success.
    fn try_fold_int(&mut self) -> Option<i64> {
        self.fold_addsub()
    }
    fn fold_addsub(&mut self) -> Option<i64> {
        let mut left = self.fold_muldiv()?;
        loop {
            match self.peek() {
                TokKind::Plus => { self.bump(); let r = self.fold_muldiv()?; left = left.wrapping_add(r); }
                TokKind::Minus => { self.bump(); let r = self.fold_muldiv()?; left = left.wrapping_sub(r); }
                _ => return Some(left),
            }
        }
    }
    fn fold_muldiv(&mut self) -> Option<i64> {
        let mut left = self.fold_atom()?;
        loop {
            match self.peek() {
                TokKind::Star => { self.bump(); let r = self.fold_atom()?; left = left.wrapping_mul(r); }
                TokKind::Slash => { self.bump(); let r = self.fold_atom()?; if r == 0 { return None; } left /= r; }
                _ => return Some(left),
            }
        }
    }
    fn fold_atom(&mut self) -> Option<i64> {
        match self.peek().clone() {
            TokKind::Int(n) => { self.bump(); Some(n) }
            TokKind::Minus => { self.bump(); let n = self.fold_atom()?; Some(-n) }
            TokKind::LParen => { self.bump(); let v = self.fold_addsub()?; if !self.eat(&TokKind::RParen) { return None; } Some(v) }
            TokKind::Ident(name) => { self.bump(); self.constexprs.get(&name).copied() }
            _ => None,
        }
    }

    fn parse_dotted_path(&mut self) -> Result<Vec<String>, ParseError> {
        let mut parts = Vec::new();
        let (first, _) = self.expect_ident("module path component")?;
        parts.push(first);
        while self.at(&TokKind::Dot) && matches!(self.peek_at(1), TokKind::Ident(_)) {
            self.bump();
            let (next, _) = self.expect_ident("module path component")?;
            parts.push(next);
        }
        Ok(parts)
    }

    // -------- helpers --------
    fn peek(&self) -> &TokKind { &self.toks[self.pos].kind }
    fn peek_at(&self, k: usize) -> &TokKind {
        if self.pos + k < self.toks.len() { &self.toks[self.pos + k].kind } else { &TokKind::Eof }
    }
    fn peek_span(&self) -> Span { self.toks[self.pos].span }
    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if !matches!(t.kind, TokKind::Eof) { self.pos += 1; }
        t
    }
    fn at(&self, k: &TokKind) -> bool { std::mem::discriminant(self.peek()) == std::mem::discriminant(k) }
    fn eat(&mut self, k: &TokKind) -> bool {
        if self.at(k) { self.bump(); true } else { false }
    }
    fn expect(&mut self, k: &TokKind, what: &str) -> Result<Token, ParseError> {
        if self.at(k) { Ok(self.bump()) } else {
            // Nested generics close with `>>` which the lexer combines into ShrOp.
            // When the parser is asking for a Gt, accept ShrOp by consuming the
            // first `>` and rewriting the remainder into a Gt token in place.
            if matches!(k, TokKind::Gt) && matches!(self.peek(), TokKind::ShrOp) {
                let tok = self.toks[self.pos].clone();
                self.toks[self.pos] = Token { kind: TokKind::Gt, span: tok.span };
                return Ok(Token { kind: TokKind::Gt, span: tok.span });
            }
            Err(ParseError { msg: format!("expected {}, got {:?}", what, self.peek()), span: self.peek_span() })
        }
    }
    fn expect_ident(&mut self, what: &str) -> Result<(String, Span), ParseError> {
        let t = self.toks[self.pos].clone();
        if let TokKind::Ident(s) = t.kind {
            self.pos += 1;
            Ok((s, t.span))
        } else {
            Err(ParseError { msg: format!("expected {}, got {:?}", what, self.peek()), span: t.span })
        }
    }

    // -------- items --------
    fn parse_item(&mut self) -> Result<Item, ParseError> {
        let kind = self.peek().clone();
        match &kind {
            TokKind::Data => Ok(Item::Data(self.parse_data()?)),
            TokKind::Enum => Ok(Item::Enum(self.parse_enum()?)),
            TokKind::Extern => Ok(Item::Extern(self.parse_extern()?)),
            TokKind::Cinclude => self.parse_cinclude(),
            TokKind::Cblock => self.parse_cblock(),
            TokKind::Rblock => self.parse_rblock(),
            TokKind::Rdep => self.parse_rdep(),
            TokKind::Logic => Ok(Item::Logic(self.parse_logic()?)),
            TokKind::Attr => Ok(Item::Attr(self.parse_attr()?)),
            TokKind::Inline => Ok(Item::Func(self.parse_func()?)),
            TokKind::Gate => Ok(Item::Func(self.parse_func()?)),
            // Module-scope `mut Type NAME = expr;` — a mutable global.
            TokKind::Mut => Ok(Item::Global(self.parse_global(true)?)),
            // `Type has Attr { ... }` — detect by looking two tokens ahead for `Has`.
            TokKind::Ident(_) if matches!(self.peek_at(1), TokKind::Has) => {
                Ok(Item::Has(self.parse_has()?))
            }
            // Parametric `has`: `*T has Foo { ... }`, `&T has Foo { ... }`,
            // `own *T has Foo { ... }`, `raw *T has Foo { ... }`, `Box<T> has Foo`.
            // Disambiguates from globals/funcs (which also start with a type)
            // by speculatively parsing a type and checking if the next token is
            // `has`.  No heap allocation for the common (non-has) case — we
            // restore self.pos.
            _ if self.looks_like_parametric_has_item() => {
                Ok(Item::Has(self.parse_has()?))
            }
            _ => {
                // Disambiguate `Type NAME = expr;` (immutable global) from
                // `Type NAME(...) { ... }` (function). Both start with a type;
                // the deciding tokens are after the identifier that follows it.
                let save = self.pos;
                let is_global = (|| -> bool {
                    if self.parse_type().is_err() { return false; }
                    if !matches!(self.peek(), TokKind::Ident(_)) { return false; }
                    matches!(self.peek_at(1), TokKind::Eq)
                })();
                self.pos = save;
                if is_global {
                    Ok(Item::Global(self.parse_global(false)?))
                } else {
                    Ok(Item::Func(self.parse_func()?))
                }
            }
        }
    }

    /// Parse a module-scope `[mut|const]? Type NAME = expr;` global declaration.
    /// `is_mut` is true when the leading `mut` keyword has been seen and is to
    /// be consumed here; otherwise we're at an immutable / `const`-prefixed form.
    fn parse_global(&mut self, is_mut: bool) -> Result<GlobalDecl, ParseError> {
        let span = self.peek_span();
        if is_mut { self.expect(&TokKind::Mut, "`mut`")?; }
        let ty = self.parse_type()?;
        let (name, _) = self.expect_ident("global name")?;
        self.expect(&TokKind::Eq, "`=`")?;
        let init = self.parse_expr()?;
        self.expect(&TokKind::Semicolon, "`;`")?;
        Ok(GlobalDecl { name, ty, init, is_mut, is_pub: false, span })
    }

    fn parse_attr(&mut self) -> Result<AttrDecl, ParseError> {
        let kw = self.expect(&TokKind::Attr, "`attr`")?;
        let (name, _) = self.expect_ident("attribute name")?;
        // Optional generic params: `attr Convert<U> { ... }`.
        let type_params = self.parse_type_params()?;
        self.expect(&TokKind::LBrace, "`{`")?;
        let mut funcs = Vec::new();
        let mut assoc_types: Vec<AssocTypeDecl> = Vec::new();
        while !self.at(&TokKind::RBrace) {
            // `type Name;` or `type Name = DefaultType;` — associated-type
            // declaration (§10.5).  With a default, the impl may omit the
            // definition and the default is used; without one, the impl
            // MUST provide `type Name = ConcreteType;`.
            if self.at(&TokKind::Type) {
                let sp = self.peek_span();
                self.bump();
                let (n, _) = self.expect_ident("associated-type name")?;
                let default = if self.eat(&TokKind::Eq) {
                    Some(self.parse_type()?)
                } else { None };
                self.expect(&TokKind::Semicolon, "`;`")?;
                assoc_types.push(AssocTypeDecl { name: n, default, span: sp });
            } else {
                funcs.push(self.parse_attr_method()?);
            }
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok(AttrDecl { name, type_params, funcs, assoc_types, is_pub: false, span: kw.span })
    }

    /// Parse one method declaration inside an `attr` block.  Accepts either:
    ///   `RetType name(<params>) [where ...];`            → signature-only (no default)
    ///   `RetType name(<params>) [where ...] { body }`    → signature with default body
    fn parse_attr_method(&mut self) -> Result<FuncDecl, ParseError> {
        let start = self.peek_span();
        let mut is_inline = false;
        let mut is_gate = false;
        loop {
            if self.eat(&TokKind::Inline) { is_inline = true; }
            else if self.eat(&TokKind::Gate) { is_gate = true; }
            else { break; }
        }
        let ret = self.parse_type()?;
        let (name, _) = self.expect_ident("method name")?;
        let (type_params, inline_bounds) = self.parse_type_params_with_bounds()?;
        self.expect(&TokKind::LParen, "`(`")?;
        let mut params = Vec::new();
        if !self.at(&TokKind::RParen) {
            loop {
                let pstart = self.peek_span();
                let ty = self.parse_type()?;
                let (pname, _) = self.expect_ident("parameter name")?;
                params.push(Param { name: pname, ty, span: pstart });
                if !self.eat(&TokKind::Comma) { break; }
            }
        }
        self.expect(&TokKind::RParen, "`)`")?;
        let mut where_clauses = self.parse_where_clauses()?;
        for b in inline_bounds { where_clauses.push(b); }
        // Either `;` (no default body) or `{ ... }` (default body).
        let body = if self.eat(&TokKind::Semicolon) {
            // Signature-only: synthesize an empty block as the "no body" placeholder.
            // The resolver detects empty bodies as "no default" via `Block.stmts.is_empty()`.
            Block { stmts: Vec::new(), span: start }
        } else {
            self.parse_block()?
        };
        Ok(FuncDecl {
            name, type_params, params, ret, body,
            is_inline, is_gate, is_pub: false, where_clauses, span: start,
        })
    }

    fn parse_has(&mut self) -> Result<HasDecl, ParseError> {
        let start = self.peek_span();
        // Receiver can be a full Type (concrete name, primitive, or parametric
        // pattern like `*T`, `Box<T>`).  The legacy single-ident path is a
        // special case of parse_type.
        let receiver = self.parse_type()?;
        let type_name = receiver_canonical_name(&receiver);
        self.expect(&TokKind::Has, "`has`")?;
        let (attr_name, _) = self.expect_ident("attribute name")?;
        // Concrete attr args: `Color has Convert<int> { ... }`.
        let mut attr_args: Vec<Type> = Vec::new();
        if self.eat(&TokKind::Lt) {
            loop {
                attr_args.push(self.parse_type()?);
                if !self.eat(&TokKind::Comma) { break; }
            }
            self.expect(&TokKind::Gt, "`>`")?;
        }
        self.expect(&TokKind::LBrace, "`{`")?;
        let mut funcs = Vec::new();
        let mut assoc_type_defs: Vec<AssocTypeDef> = Vec::new();
        while !self.at(&TokKind::RBrace) {
            // `type Name = ConcreteType;` — associated-type definition (§10.5).
            if self.at(&TokKind::Type) {
                let sp = self.peek_span();
                self.bump();
                let (n, _) = self.expect_ident("associated-type name")?;
                self.expect(&TokKind::Eq, "`=`")?;
                let value = self.parse_type()?;
                self.expect(&TokKind::Semicolon, "`;`")?;
                assoc_type_defs.push(AssocTypeDef { name: n, value, span: sp });
            } else {
                funcs.push(self.parse_func()?);
            }
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok(HasDecl { type_name, receiver, attr_name, attr_args, funcs, assoc_type_defs, is_pub: false, span: start })
    }

    /// Speculative lookahead: do the next tokens read as `<Type> has`?  Used
    /// to detect parametric `has`-items at the item-dispatch level without
    /// consuming the type.  Restores `self.pos` afterwards.
    fn looks_like_parametric_has_item(&mut self) -> bool {
        // Candidates: leading token is `*`, `&`, `own`, `raw`, or an Ident
        // whose type expression has further qualification (`Box<T>`).  Plain
        // `Ident HAS` is handled by the dedicated arm above; that path
        // short-circuits before this one runs.
        if !matches!(
            self.peek(),
            TokKind::Star | TokKind::Amp | TokKind::Own | TokKind::Raw | TokKind::Ident(_)
        ) {
            return false;
        }
        let save = self.pos;
        let ok = self.parse_type().is_ok() && matches!(self.peek(), TokKind::Has);
        self.pos = save;
        ok
    }

    fn parse_logic(&mut self) -> Result<LogicDecl, ParseError> {
        let kw = self.expect(&TokKind::Logic, "`logic`")?;
        let (name, _) = self.expect_ident("logic name")?;
        self.expect(&TokKind::LBrace, "`{`")?;
        let mut funcs = Vec::new();
        while !self.at(&TokKind::RBrace) {
            funcs.push(self.parse_func()?);
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok(LogicDecl { name, funcs, is_pub: false, span: kw.span })
    }

    fn parse_type_params(&mut self) -> Result<Vec<String>, ParseError> {
        let (names, _) = self.parse_type_params_with_bounds()?;
        Ok(names)
    }

    /// Parse `<T, U: Trait>` and return both the names AND any inline bound shorthands
    /// converted to `WhereClause` form.  Callers that already pass `where_clauses`
    /// downstream should merge these.
    fn parse_type_params_with_bounds(&mut self) -> Result<(Vec<String>, Vec<WhereClause>), ParseError> {
        let mut out = Vec::new();
        let mut bounds = Vec::new();
        if !self.eat(&TokKind::Lt) { return Ok((out, bounds)); }
        loop {
            let start = self.peek_span();
            let (n, _) = self.expect_ident("type parameter")?;
            // `<T: Attr>` shorthand → equivalent to `where T has Attr`.
            // Extended: `<T: Attr<Slot = i64>>` adds an assoc-type binding.
            if self.eat(&TokKind::Colon) {
                let (trait_name, _) = self.expect_ident("attribute name")?;
                let mut args: Vec<Type> = Vec::new();
                let mut bindings: Vec<(String, Type)> = Vec::new();
                if self.eat(&TokKind::Lt) {
                    loop {
                        // Bound entries can be either `Type` (positional attr-arg)
                        // or `Name = Type` (assoc-type binding).  Distinguish by
                        // peeking for `Ident =`.
                        if matches!((self.peek().clone(), self.peek_at(1).clone()), (TokKind::Ident(_), TokKind::Eq)) {
                            let (bn, _) = self.expect_ident("assoc-type name")?;
                            self.expect(&TokKind::Eq, "`=`")?;
                            let bv = self.parse_type()?;
                            bindings.push((bn, bv));
                        } else {
                            args.push(self.parse_type()?);
                        }
                        if !self.eat(&TokKind::Comma) { break; }
                    }
                    self.expect(&TokKind::Gt, "`>`")?;
                }
                let mut all_args = vec![Type::Named(n.clone(), start)];
                all_args.extend(args);
                bounds.push(WhereClause { trait_name, args: all_args, assoc_type_bindings: bindings, span: start });
            }
            out.push(n);
            if !self.eat(&TokKind::Comma) { break; }
        }
        self.expect(&TokKind::Gt, "`>`")?;
        Ok((out, bounds))
    }

    fn parse_where_clauses(&mut self) -> Result<Vec<WhereClause>, ParseError> {
        let mut out = Vec::new();
        if !self.eat(&TokKind::Where) { return Ok(out); }
        loop {
            let start = self.peek_span();
            // Two accepted forms:
            //   `T has Attr`            — preferred new spelling
            //   `Attr<T>`               — legacy form, kept for backward compat
            // Detection: peek for `Ident HAS` to pick the new form.
            if matches!((self.peek().clone(), self.peek_at(1).clone()), (TokKind::Ident(_), TokKind::Has)) {
                let (type_var, ty_span) = self.expect_ident("type parameter")?;
                self.expect(&TokKind::Has, "`has`")?;
                let (attr_name, _) = self.expect_ident("attribute name")?;
                let mut args: Vec<Type> = vec![Type::Named(type_var, ty_span)];
                let mut bindings: Vec<(String, Type)> = Vec::new();
                if self.eat(&TokKind::Lt) {
                    loop {
                        if matches!((self.peek().clone(), self.peek_at(1).clone()), (TokKind::Ident(_), TokKind::Eq)) {
                            let (bn, _) = self.expect_ident("assoc-type name")?;
                            self.expect(&TokKind::Eq, "`=`")?;
                            let bv = self.parse_type()?;
                            bindings.push((bn, bv));
                        } else {
                            args.push(self.parse_type()?);
                        }
                        if !self.eat(&TokKind::Comma) { break; }
                    }
                    self.expect(&TokKind::Gt, "`>`")?;
                }
                out.push(WhereClause { trait_name: attr_name, args, assoc_type_bindings: bindings, span: start });
            } else {
                let (trait_name, _) = self.expect_ident("trait name")?;
                let mut args = Vec::new();
                let mut bindings: Vec<(String, Type)> = Vec::new();
                if self.eat(&TokKind::Lt) {
                    loop {
                        if matches!((self.peek().clone(), self.peek_at(1).clone()), (TokKind::Ident(_), TokKind::Eq)) {
                            let (bn, _) = self.expect_ident("assoc-type name")?;
                            self.expect(&TokKind::Eq, "`=`")?;
                            let bv = self.parse_type()?;
                            bindings.push((bn, bv));
                        } else {
                            args.push(self.parse_type()?);
                        }
                        if !self.eat(&TokKind::Comma) { break; }
                    }
                    self.expect(&TokKind::Gt, "`>`")?;
                }
                out.push(WhereClause { trait_name, args, assoc_type_bindings: bindings, span: start });
            }
            if !self.eat(&TokKind::Comma) { break; }
        }
        Ok(out)
    }

    fn parse_cinclude(&mut self) -> Result<Item, ParseError> {
        let kw = self.expect(&TokKind::Cinclude, "`cinclude`")?;
        let header = if let TokKind::StrLit(_) = self.peek() {
            if let TokKind::StrLit(s) = self.bump().kind { s } else { unreachable!() }
        } else {
            return Err(ParseError { msg: "expected string literal after `cinclude`".into(), span: self.peek_span() });
        };
        self.expect(&TokKind::Semicolon, "`;`")?;
        Ok(Item::CInclude(header, kw.span))
    }

    fn parse_cblock(&mut self) -> Result<Item, ParseError> {
        let kw = self.expect(&TokKind::Cblock, "`cblock`")?;
        // The body is a single string literal containing the raw C source.  Using a string
        // literal keeps the lexer simple — embedded `}` is just a normal character.
        let body = if let TokKind::StrLit(_) = self.peek() {
            if let TokKind::StrLit(s) = self.bump().kind { s } else { unreachable!() }
        } else {
            return Err(ParseError { msg: "expected string literal after `cblock`".into(), span: self.peek_span() });
        };
        self.expect(&TokKind::Semicolon, "`;`")?;
        Ok(Item::CBlock(body, kw.span))
    }

    fn parse_rblock(&mut self) -> Result<Item, ParseError> {
        let kw = self.expect(&TokKind::Rblock, "`rblock`")?;
        let body = if let TokKind::StrLit(_) = self.peek() {
            if let TokKind::StrLit(s) = self.bump().kind { s } else { unreachable!() }
        } else {
            return Err(ParseError { msg: "expected string literal after `rblock`".into(), span: self.peek_span() });
        };
        self.expect(&TokKind::Semicolon, "`;`")?;
        Ok(Item::Rblock(body, kw.span))
    }

    fn parse_rdep(&mut self) -> Result<Item, ParseError> {
        let kw = self.expect(&TokKind::Rdep, "`rdep`")?;
        let (name, _) = self.expect_ident("crate name")?;
        self.expect(&TokKind::Eq, "`=`")?;
        let version = if let TokKind::StrLit(_) = self.peek() {
            if let TokKind::StrLit(s) = self.bump().kind { s } else { unreachable!() }
        } else {
            return Err(ParseError { msg: "expected string literal after `=` in `rdep`".into(), span: self.peek_span() });
        };
        self.expect(&TokKind::Semicolon, "`;`")?;
        Ok(Item::Rdep(name, version, kw.span))
    }

    fn parse_extern(&mut self) -> Result<ExternDecl, ParseError> {
        let kw = self.expect(&TokKind::Extern, "`extern`")?;
        let is_gate = self.eat(&TokKind::Gate);
        // Optional C link name as string literal: `extern "puts" ...`
        let c_name = if let TokKind::StrLit(_) = self.peek() {
            if let TokKind::StrLit(s) = self.bump().kind { Some(s) } else { None }
        } else { None };
        let ret = self.parse_type()?;
        let (name, _) = self.expect_ident("extern function name")?;
        self.expect(&TokKind::LParen, "`(`")?;
        let mut params = Vec::new();
        let mut is_variadic = false;
        if !self.at(&TokKind::RParen) {
            loop {
                // Trailing `...` marks the extern as variadic — must come after at least one named param,
                // matching C's rule that variadics require a fixed-arity prefix.
                if self.at(&TokKind::DotDot) && matches!(self.peek_at(1), TokKind::Dot) {
                    self.bump(); self.bump();
                    is_variadic = true;
                    break;
                }
                let pstart = self.peek_span();
                let ty = self.parse_type()?;
                let (pname, _) = self.expect_ident("parameter name")?;
                params.push(Param { name: pname, ty, span: pstart });
                if !self.eat(&TokKind::Comma) { break; }
            }
        }
        self.expect(&TokKind::RParen, "`)`")?;
        self.expect(&TokKind::Semicolon, "`;`")?;
        let c_name = c_name.unwrap_or_else(|| name.clone());
        Ok(ExternDecl { name, c_name, params, ret, is_gate, is_variadic, is_pub: false, span: kw.span })
    }

    fn parse_data(&mut self) -> Result<DataDecl, ParseError> {
        let kw = self.expect(&TokKind::Data, "`data`")?;
        let (name, _) = self.expect_ident("struct name")?;
        let type_params = self.parse_type_params()?;
        let where_clauses = self.parse_where_clauses()?;
        self.expect(&TokKind::LBrace, "`{`")?;
        let mut fields = Vec::new();
        while !self.at(&TokKind::RBrace) {
            fields.push(self.parse_field()?);
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok(DataDecl { name, type_params, fields, where_clauses, is_pub: false, span: kw.span })
    }

    fn parse_field(&mut self) -> Result<FieldDecl, ParseError> {
        let start = self.peek_span();
        let is_embed = self.eat(&TokKind::Embed);
        let mutness = self.parse_mutness();
        let ty = self.parse_type()?;
        let (name, _) = self.expect_ident("field name")?;
        let default = if self.eat(&TokKind::Eq) {
            Some(self.parse_expr()?)
        } else { None };
        self.expect(&TokKind::Semicolon, "`;`")?;
        Ok(FieldDecl { mutness, ty, name, default, is_embed, span: start })
    }

    fn parse_enum(&mut self) -> Result<EnumDecl, ParseError> {
        let kw = self.expect(&TokKind::Enum, "`enum`")?;
        let (name, _) = self.expect_ident("enum name")?;
        let type_params = self.parse_type_params()?;
        self.expect(&TokKind::LBrace, "`{`")?;
        let mut variants = Vec::new();
        while !self.at(&TokKind::RBrace) {
            let vstart = self.peek_span();
            let (vname, _) = self.expect_ident("variant name")?;
            let mut explicit_value = None;
            let mut fields: Vec<FieldDecl> = Vec::new();
            if self.eat(&TokKind::Eq) {
                if let TokKind::Int(n) = *self.peek() { self.bump(); explicit_value = Some(n); }
                else { return Err(ParseError { msg: "expected integer".into(), span: self.peek_span() }); }
            } else if self.eat(&TokKind::LBrace) {
                // Variant payload: `{ Type name, Type name }`
                while !self.at(&TokKind::RBrace) {
                    let fstart = self.peek_span();
                    let mutness = self.parse_mutness();
                    let ty = self.parse_type()?;
                    let (fname, _) = self.expect_ident("field name")?;
                    let default = if self.eat(&TokKind::Eq) { Some(self.parse_expr()?) } else { None };
                    fields.push(FieldDecl {
                        mutness, ty, name: fname, default,
                        is_embed: false, span: fstart,
                    });
                    if !self.eat(&TokKind::Comma) { break; }
                    if self.at(&TokKind::RBrace) { break; }
                }
                self.expect(&TokKind::RBrace, "`}` to close variant fields")?;
            }
            variants.push(VariantDecl { name: vname, fields, explicit_value, span: vstart });
            if !self.eat(&TokKind::Comma) { break; }
            if self.at(&TokKind::RBrace) { break; }
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok(EnumDecl { name, type_params, variants, is_pub: false, span: kw.span })
    }

    fn parse_func(&mut self) -> Result<FuncDecl, ParseError> {
        let start = self.peek_span();
        let mut is_inline = false;
        let mut is_gate = false;
        loop {
            if self.eat(&TokKind::Inline) { is_inline = true; }
            else if self.eat(&TokKind::Gate) { is_gate = true; }
            else { break; }
        }
        let ret = self.parse_type()?;
        let (name, _) = self.expect_ident("function name")?;
        let (type_params, inline_bounds) = self.parse_type_params_with_bounds()?;
        self.expect(&TokKind::LParen, "`(`")?;
        let mut params = Vec::new();
        if !self.at(&TokKind::RParen) {
            loop {
                let pstart = self.peek_span();
                let ty = self.parse_type()?;
                let (pname, _) = self.expect_ident("parameter name")?;
                params.push(Param { name: pname, ty, span: pstart });
                if !self.eat(&TokKind::Comma) { break; }
            }
        }
        self.expect(&TokKind::RParen, "`)`")?;
        let mut where_clauses = self.parse_where_clauses()?;
        // Inline `<T: Attr>` shorthand contributes additional bounds.
        for b in inline_bounds { where_clauses.push(b); }
        let body = self.parse_block()?;
        Ok(FuncDecl { name, type_params, params, ret, body, is_inline, is_gate, is_pub: false, where_clauses, span: start })
    }

    // -------- types --------
    fn parse_mutness(&mut self) -> Mutness {
        // thread_local is recognised but currently behaves as a regular binding.
        let _ = self.eat(&TokKind::ThreadLocal);
        if self.eat(&TokKind::Mut) { Mutness::Mut }
        else if self.eat(&TokKind::Const) || self.eat(&TokKind::Constexpr) { Mutness::Const }
        else { Mutness::Default }
    }

    pub fn parse_type(&mut self) -> Result<Type, ParseError> {
        let base = self.parse_type_base()?;
        // Function pointer postfix: `BaseType(P1, P2, ...)` — used for D8.
        if self.at(&TokKind::LParen) {
            // Peek: if the inside looks like a type list (vs. e.g. unit type or call), consume.
            // For type position, `(...)` after a non-call expression is the fn-ptr form.
            let span = self.peek_span();
            // Heuristic: only treat as fn-ptr if we're not at end of decl.
            self.bump(); // (
            let mut params = Vec::new();
            if !self.at(&TokKind::RParen) {
                loop {
                    params.push(self.parse_type()?);
                    if !self.eat(&TokKind::Comma) { break; }
                }
            }
            self.expect(&TokKind::RParen, "`)`")?;
            return Ok(Type::FnPtr { ret: Box::new(base), params, span });
        }
        Ok(base)
    }

    fn parse_type_base(&mut self) -> Result<Type, ParseError> {
        let start = self.peek_span();
        // `alloc` is no longer a type modifier — it is the allocation expression only.
        // Use `own *T` (nullable owning) or `own &T` (strict owning) in type position.
        if self.at(&TokKind::Alloc) {
            return Err(ParseError {
                msg: "`alloc` is not a type modifier; use `own *T` or `own &T` for owning slots".into(),
                span: start,
            });
        }
        // `raw *T` — provenance-unknown pointer; deref/field/index requires `unsafe { }`.
        if self.eat(&TokKind::Raw) {
            self.expect(&TokKind::Star, "`*` after `raw`")?;
            let mutness = self.parse_mutness();
            let inner = self.parse_type()?;
            return Ok(Type::RawPtr { mutness, inner: Box::new(inner), span: start });
        }
        // `own *T` (nullable owning pointer) or `own &T` (strict owning, alias for `heap T`).
        if self.eat(&TokKind::Own) {
            if self.eat(&TokKind::Star) {
                let mutness = self.parse_mutness();
                let inner = self.parse_type()?;
                return Ok(Type::OwnPtr { mutness, inner: Box::new(inner), span: start });
            }
            if self.eat(&TokKind::Amp) {
                let _mutness = self.parse_mutness();   // borrow mutability is irrelevant for the owning view
                let inner = self.parse_type()?;
                // `own &T` is the strict, non-null owning form — same semantics as the
                // legacy `heap T` value-form.
                return Ok(Type::Heap { inner: Box::new(inner), span: start });
            }
            return Err(ParseError { msg: "expected `*` or `&` after `own`".into(), span: self.peek_span() });
        }
        // `dyn Trait` or `dyn (Trait + Trait + ...)`
        if self.eat(&TokKind::Dyn) {
            let mut traits = Vec::new();
            if self.eat(&TokKind::LParen) {
                loop {
                    let (n, _) = self.expect_ident("trait name")?;
                    traits.push(n);
                    if !self.eat(&TokKind::Plus) { break; }
                }
                self.expect(&TokKind::RParen, "`)`")?;
            } else {
                let (n, _) = self.expect_ident("trait name")?;
                traits.push(n);
            }
            return Ok(Type::Dyn { traits, span: start });
        }
        // mut/const without sigil — applies to the *next* type's payload mutness (handled in normalisation)
        // but allowed at top: e.g. `mut int x` means mut binding of int.
        // Parsing strategy: if we see `mut`/`const` at type position, only when followed by a primitive/named type we just push mutness into the binding (handled in Stmt::Let parsing). At this level, types do not start with mut/const.

        match self.peek() {
            TokKind::Amp => {
                self.bump();
                let mutness = self.parse_mutness();
                let inner = self.parse_type()?;
                Ok(Type::Ref { mutness, inner: Box::new(inner), span: start })
            }
            TokKind::Star => {
                self.bump();
                let mutness = self.parse_mutness();
                let inner = self.parse_type()?;
                Ok(Type::Ptr { mutness, inner: Box::new(inner), span: start })
            }
            TokKind::LBracket => {
                self.bump();
                // forms: `[]T`, `[]mut T`, `[N]T`, `[*]T`
                if self.eat(&TokKind::RBracket) {
                    let mutness = if self.eat(&TokKind::Mut) { Mutness::Mut } else { Mutness::Default };
                    let elem = self.parse_type()?;
                    return Ok(Type::Slice { mutness, elem: Box::new(elem), span: start });
                }
                if self.eat(&TokKind::Star) {
                    self.expect(&TokKind::RBracket, "`]`")?;
                    let elem = self.parse_type()?;
                    return Ok(Type::Vec { elem: Box::new(elem), span: start });
                }
                // `[N]T` where N is a literal or a constexpr expression.
                let arr_span = self.peek_span();
                let n = self.try_fold_int()
                    .ok_or_else(|| ParseError { msg: "array length must be an integer literal or a `constexpr` expression".into(), span: arr_span })?;
                self.expect(&TokKind::RBracket, "`]`")?;
                let elem = self.parse_type()?;
                return Ok(Type::Array { len: n, elem: Box::new(elem), span: start });
            }
            TokKind::LParen => {
                self.bump();
                // unit type `()`
                self.expect(&TokKind::RParen, "`)`")?;
                Ok(Type::Unit(start))
            }
            TokKind::Unit => {
                self.bump();
                Ok(Type::Named("unit".to_string(), start))
            }
            // `_` placeholder — only meaningful inside `attr`/`has` blocks; the
            // resolver substitutes it with the implementing type. Outside that
            // context the resolver rejects the unresolved sentinel.
            TokKind::Underscore => {
                self.bump();
                let mut base = Type::Named("_".to_string(), start);
                // Allow trailing `::Seg` for assoc-type paths on the
                // placeholder (e.g. `_::Ok`, `_::Err`).
                while self.at(&TokKind::ColonColon) {
                    self.bump();
                    let (seg, sseg) = self.expect_ident("associated-type name")?;
                    base = Type::AssocPath { base: Box::new(base), segment: seg, span: sseg };
                }
                Ok(base)
            }
            TokKind::Ident(_) => {
                let (name, sp) = self.expect_ident("type name")?;
                let mut head = if self.at(&TokKind::Lt) {
                    // Generic type instantiation `Name<T, U>`. Only accept if the inside parses cleanly.
                    let save = self.pos;
                    self.bump(); // <
                    let mut args = Vec::new();
                    let ok = loop {
                        match self.parse_type() {
                            Ok(t) => args.push(t),
                            Err(_) => break false,
                        }
                        if self.eat(&TokKind::Comma) { continue; }
                        if self.at(&TokKind::Gt) { break true; }
                        break false;
                    };
                    if ok && self.eat(&TokKind::Gt) {
                        Type::Generic { name, args, span: sp }
                    } else {
                        self.pos = save;
                        Type::Named(name, sp)
                    }
                } else {
                    Type::Named(name, sp)
                };
                // Trailing `::Seg::Seg2...` — associated-type path.
                while self.at(&TokKind::ColonColon) {
                    self.bump();
                    let (seg, sseg) = self.expect_ident("associated-type name")?;
                    head = Type::AssocPath { base: Box::new(head), segment: seg, span: sseg };
                }
                Ok(head)
            }
            other => Err(ParseError { msg: format!("expected type, got {:?}", other), span: self.peek_span() }),
        }
    }

    // -------- blocks/stmts --------
    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let lb = self.expect(&TokKind::LBrace, "`{`")?;
        let mut stmts = Vec::new();
        while !self.at(&TokKind::RBrace) && !self.at(&TokKind::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok(Block { stmts, span: lb.span })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek_span();
        match self.peek() {
            TokKind::If => self.parse_if(),
            TokKind::While => self.parse_while(),
            TokKind::For => self.parse_for(),
            TokKind::Break => { self.bump(); self.expect(&TokKind::Semicolon, "`;`")?; Ok(Stmt::Break(start)) }
            TokKind::Continue => { self.bump(); self.expect(&TokKind::Semicolon, "`;`")?; Ok(Stmt::Continue(start)) }
            TokKind::Match => {
                self.bump();
                let (scrut, arms) = self.parse_match_after_kw()?;
                // Optional trailing `;` — match-as-statement reads more
                // naturally with one (matches every other statement form's
                // terminator).  The grammar accepts either shape.
                let _ = self.eat(&TokKind::Semicolon);
                Ok(Stmt::Match { scrutinee: scrut, arms, span: start })
            }
            TokKind::Yield => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(&TokKind::Semicolon, "`;`")?;
                Ok(Stmt::Yield(e, start))
            }
            TokKind::Propagate => {
                self.bump();
                let e = if self.at(&TokKind::Semicolon) { None } else { Some(self.parse_expr()?) };
                self.expect(&TokKind::Semicolon, "`;`")?;
                Ok(Stmt::Propagate(e, start))
            }
            TokKind::Return => {
                self.bump();
                let e = if self.at(&TokKind::Semicolon) { None } else { Some(self.parse_expr()?) };
                self.expect(&TokKind::Semicolon, "`;`")?;
                Ok(Stmt::Return(e, start))
            }
            TokKind::LBrace => Ok(Stmt::Block(self.parse_block()?)),
            TokKind::Unsafe => {
                self.bump();
                let b = self.parse_block()?;
                Ok(Stmt::Unsafe(b, start))
            }
            _ => {
                // Could be a declaration (`Type name = expr;` possibly preceded by `mut`/`const`)
                // or an assignment/expression statement.
                if self.looks_like_decl() {
                    self.parse_let()
                } else {
                    self.parse_assign_or_expr()
                }
            }
        }
    }

    fn looks_like_decl(&self) -> bool {
        // Heuristic: starts with `mut`/`const`/`heap`/`&`/`*`/`[`, OR an identifier
        // that's followed by something that begins a type-continuation leading to an ident `=`.
        // We do a small look-ahead: try to "skim" a type then check we land on an ident `=`.
        let mut p = self.pos;
        let t = self.toks.clone();
        let kinds: Vec<&TokKind> = t.iter().map(|x| &x.kind).collect();
        // skim modifiers
        loop {
            match kinds.get(p) {
                Some(TokKind::Mut) | Some(TokKind::Const) | Some(TokKind::Constexpr) | Some(TokKind::ThreadLocal) => { p += 1; continue; }
                _ => break,
            }
        }
        // optional heap
        // `alloc` is no longer a type modifier — skip if present, but parse_type will reject.
        // skim a type
        let after = Self::skim_type(&kinds, p);
        if let Some(np) = after {
            // expect ident then '=' or ';'
            if let Some(TokKind::Ident(_)) = kinds.get(np) {
                if matches!(kinds.get(np + 1), Some(TokKind::Eq) | Some(TokKind::Semicolon)) {
                    return true;
                }
            }
        }
        false
    }

    fn skim_type(kinds: &[&TokKind], mut p: usize) -> Option<usize> {
        let after_base = Self::skim_type_base(kinds, p)?;
        // Function pointer postfix: `BaseType(P1, P2, ...)`.
        if matches!(kinds.get(after_base), Some(TokKind::LParen)) {
            let mut q = after_base + 1;
            let mut depth = 1;
            while q < kinds.len() && depth > 0 {
                match kinds.get(q) {
                    Some(TokKind::LParen) => depth += 1,
                    Some(TokKind::RParen) => depth -= 1,
                    _ => {}
                }
                q += 1;
            }
            if depth == 0 { return Some(q); }
        }
        let _ = p;
        Some(after_base)
    }

    fn skim_type_base(kinds: &[&TokKind], mut p: usize) -> Option<usize> {
        // recursive skim of the type grammar
        // heap? raw?
        // `alloc` is no longer a type modifier — skip if present, but parse_type will reject.
        if matches!(kinds.get(p), Some(TokKind::Raw)) {
            p += 1;
            // `raw *T` — the next token must be `*`, then a mutness opt, then a type base.
            if !matches!(kinds.get(p), Some(TokKind::Star)) { return None; }
            p += 1;
            if matches!(kinds.get(p), Some(TokKind::Mut) | Some(TokKind::Const)) { p += 1; }
            return Self::skim_type(kinds, p);
        }
        if matches!(kinds.get(p), Some(TokKind::Own)) {
            p += 1;
            if !matches!(kinds.get(p), Some(TokKind::Star) | Some(TokKind::Amp)) { return None; }
            p += 1;
            if matches!(kinds.get(p), Some(TokKind::Mut) | Some(TokKind::Const)) { p += 1; }
            return Self::skim_type(kinds, p);
        }
        match kinds.get(p)? {
            TokKind::Amp => {
                p += 1;
                if matches!(kinds.get(p), Some(TokKind::Mut) | Some(TokKind::Const)) { p += 1; }
                return Self::skim_type(kinds, p);
            }
            TokKind::Star => {
                p += 1;
                if matches!(kinds.get(p), Some(TokKind::Mut) | Some(TokKind::Const)) { p += 1; }
                return Self::skim_type(kinds, p);
            }
            TokKind::LBracket => {
                p += 1;
                match kinds.get(p)? {
                    TokKind::RBracket => {
                        p += 1;
                        if matches!(kinds.get(p), Some(TokKind::Mut)) { p += 1; }
                        return Self::skim_type(kinds, p);
                    }
                    TokKind::Star => {
                        p += 1;
                        if !matches!(kinds.get(p), Some(TokKind::RBracket)) { return None; }
                        p += 1;
                        return Self::skim_type(kinds, p);
                    }
                    TokKind::Int(_) | TokKind::Ident(_) | TokKind::Minus | TokKind::LParen => {
                        // Skim a constexpr-int expression: ints, idents, `+ - * / ( )` only.
                        while let Some(t) = kinds.get(p) {
                            match t {
                                TokKind::Int(_) | TokKind::Ident(_) | TokKind::Plus | TokKind::Minus
                                | TokKind::Star | TokKind::Slash | TokKind::LParen | TokKind::RParen => p += 1,
                                _ => break,
                            }
                        }
                        if !matches!(kinds.get(p), Some(TokKind::RBracket)) { return None; }
                        p += 1;
                        return Self::skim_type(kinds, p);
                    }
                    _ => return None,
                }
            }
            TokKind::LParen => {
                p += 1;
                if matches!(kinds.get(p), Some(TokKind::RParen)) { return Some(p + 1); }
                return None;
            }
            TokKind::Ident(_) | TokKind::Unit => {
                p += 1;
                // Optionally followed by `<T,...>` for generic instantiation.
                if matches!(kinds.get(p), Some(TokKind::Lt)) {
                    let mut q = p + 1;
                    let mut depth = 1;
                    // Skim until matching `>` at depth 0 (basic).
                    while q < kinds.len() && depth > 0 {
                        match kinds.get(q) {
                            Some(TokKind::Lt) => depth += 1,
                            Some(TokKind::Gt) => depth -= 1,
                            _ => {}
                        }
                        q += 1;
                        if depth == 0 { break; }
                    }
                    if depth == 0 {
                        return Some(q);
                    }
                }
                Some(p)
            }
            TokKind::Dyn => {
                p += 1;
                if matches!(kinds.get(p), Some(TokKind::LParen)) {
                    let mut q = p + 1;
                    let mut depth = 1;
                    while q < kinds.len() && depth > 0 {
                        match kinds.get(q) {
                            Some(TokKind::LParen) => depth += 1,
                            Some(TokKind::RParen) => depth -= 1,
                            _ => {}
                        }
                        q += 1;
                    }
                    Some(q)
                } else if matches!(kinds.get(p), Some(TokKind::Ident(_))) {
                    Some(p + 1)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn parse_let(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek_span();
        let thread_local = self.eat(&TokKind::ThreadLocal);
        let mutness = self.parse_mutness();
        let ty = self.parse_type()?;
        let (name, _) = self.expect_ident("variable name")?;
        self.expect(&TokKind::Eq, "`=`")?;
        let init = self.parse_expr()?;
        self.expect(&TokKind::Semicolon, "`;`")?;
        Ok(Stmt::Let { mutness, ty, name, init, thread_local, span: start })
    }

    fn parse_if(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek_span();
        self.expect(&TokKind::If, "`if`")?;
        self.expect(&TokKind::LParen, "`(`")?;
        let cond = self.parse_expr()?;
        self.expect(&TokKind::RParen, "`)`")?;
        let then_block = self.parse_block()?;
        let else_block = if self.eat(&TokKind::Else) {
            if self.at(&TokKind::If) {
                // turn `else if` into nested block of single if-stmt
                let inner = self.parse_if()?;
                Some(Block { stmts: vec![inner], span: start })
            } else {
                Some(self.parse_block()?)
            }
        } else { None };
        Ok(Stmt::If { cond, then_block, else_block, span: start })
    }

    fn parse_for(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek_span();
        self.expect(&TokKind::For, "`for`")?;
        self.expect(&TokKind::LParen, "`(`")?;
        let var_ty = self.parse_type()?;
        let (var_name, _) = self.expect_ident("loop variable")?;
        self.expect(&TokKind::In, "`in`")?;
        let s = self.parse_or()?;
        // Range form: `a..b` / `a..=b`. Otherwise treat as iteration over slice/array.
        let kind_inclusive = if self.eat(&TokKind::DotDotEq) { Some(true) }
            else if self.eat(&TokKind::DotDot) { Some(false) }
            else { None };
        let result = match kind_inclusive {
            Some(inclusive) => {
                let e = self.parse_or()?;
                self.expect(&TokKind::RParen, "`)`")?;
                let body = self.parse_block()?;
                Stmt::ForRange {
                    var_ty, var_name,
                    start: s, end: e, inclusive,
                    body, span: start,
                }
            }
            None => {
                self.expect(&TokKind::RParen, "`)`")?;
                let body = self.parse_block()?;
                Stmt::ForEach { var_ty, var_name, src: s, body, span: start }
            }
        };
        Ok(result)
    }

    fn parse_while(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek_span();
        self.expect(&TokKind::While, "`while`")?;
        self.expect(&TokKind::LParen, "`(`")?;
        let cond = self.parse_expr()?;
        self.expect(&TokKind::RParen, "`)`")?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body, span: start })
    }

    fn parse_assign_or_expr(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek_span();
        // `_ = expr;` — explicit discard pattern.
        if matches!(self.peek(), TokKind::Underscore) {
            // peek ahead to detect `_ =`
            if matches!(self.peek_at(1), TokKind::Eq) {
                self.bump(); // _
                self.bump(); // =
                let value = self.parse_expr()?;
                self.expect(&TokKind::Semicolon, "`;`")?;
                // Emit as ExprStmt with semantic "this is discarded".
                // We represent this as a Stmt::ExprStmt wrapped in a marker. For simplicity:
                // build a synthetic Ident("_") on the lhs is not desired; we just emit an ExprStmt
                // with a flag, but since AST lacks a flag we use an Assign with a synthetic Underscore lvalue.
                let place = Expr::Ident("_".into(), start);
                return Ok(Stmt::Assign { op: AssignOp::Assign, place, value, span: start });
            }
        }
        let lhs = self.parse_expr()?;
        let op = match self.peek() {
            TokKind::Eq => Some(AssignOp::Assign),
            TokKind::PlusEq => Some(AssignOp::AddAssign),
            TokKind::MinusEq => Some(AssignOp::SubAssign),
            TokKind::StarEq => Some(AssignOp::MulAssign),
            TokKind::SlashEq => Some(AssignOp::DivAssign),
            TokKind::PercentEq => Some(AssignOp::ModAssign),
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let rhs = self.parse_expr()?;
            self.expect(&TokKind::Semicolon, "`;`")?;
            return Ok(Stmt::Assign { op, place: lhs, value: rhs, span: start });
        }
        self.expect(&TokKind::Semicolon, "`;`")?;
        Ok(Stmt::ExprStmt(lhs, start))
    }

    // -------- expressions --------
    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and()?;
        while self.eat(&TokKind::OrOr) {
            let rhs = self.parse_and()?;
            let span = lhs.span();
            lhs = Expr::Bin { op: BinOp::Or, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_eq()?;
        while self.eat(&TokKind::AndAnd) {
            let rhs = self.parse_eq()?;
            let span = lhs.span();
            lhs = Expr::Bin { op: BinOp::And, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    fn parse_eq(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitor()?;
        loop {
            let op = match self.peek() {
                TokKind::EqEq => BinOp::Eq,
                TokKind::NotEq => BinOp::Ne,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_bitor()?;
            let span = lhs.span();
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    fn parse_bitor(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitxor()?;
        while self.eat(&TokKind::Pipe) {
            let rhs = self.parse_bitxor()?;
            let span = lhs.span();
            lhs = Expr::Bin { op: BinOp::BitOr, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    fn parse_bitxor(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitand()?;
        while self.eat(&TokKind::Caret) {
            let rhs = self.parse_bitand()?;
            let span = lhs.span();
            lhs = Expr::Bin { op: BinOp::BitXor, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    fn parse_bitand(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_rel()?;
        // Binary `&` is bit-and ONLY between two expressions. To avoid clashing with `&x`
        // (prefix reference), we only treat `&` as bit-and when the next token isn't followed
        // by a clear unary/primary start. Easier: just accept it here; prefix `&` is handled in parse_unary.
        while self.at(&TokKind::Amp) && !self.peek_at_unary_after_amp() {
            self.bump();
            let rhs = self.parse_rel()?;
            let span = lhs.span();
            lhs = Expr::Bin { op: BinOp::BitAnd, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    /// Heuristic: when we see `&` and the next token is a typical operand starter, treat as bit-and.
    fn peek_at_unary_after_amp(&self) -> bool {
        // If after `&` is `mut`/`const`, it's almost certainly a `&mut`/`&const` reference, not bit-and.
        matches!(self.peek_at(1), TokKind::Mut | TokKind::Const)
    }
    fn parse_rel(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_shift()?;
        loop {
            let op = match self.peek() {
                TokKind::Lt => BinOp::Lt,
                TokKind::Gt => BinOp::Gt,
                TokKind::LtEq => BinOp::Le,
                TokKind::GtEq => BinOp::Ge,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_shift()?;
            let span = lhs.span();
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    fn parse_shift(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek() {
                TokKind::ShlOp => BinOp::Shl,
                TokKind::ShrOp => BinOp::Shr,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_add()?;
            let span = lhs.span();
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                TokKind::Plus => BinOp::Add,
                TokKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_mul()?;
            let span = lhs.span();
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }
    fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                TokKind::Star => BinOp::Mul,
                TokKind::Slash => BinOp::Div,
                TokKind::Percent => BinOp::Mod,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_unary()?;
            let span = lhs.span();
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek_span();
        match self.peek() {
            TokKind::Alloc => {
                self.bump();
                // `alloc value` — heap-allocates `value` and produces an owning pointer.
                // Result type is context-typed: `own *T`, `own &T`, or (inside
                // `unsafe { }`) `raw *T` per target.
                let v = self.parse_unary()?;
                Ok(Expr::HeapAlloc { value: Box::new(v), span: start })
            }
            TokKind::Free => {
                self.bump();
                // `free value` — bare-word deallocator for `raw *T`.  Sema
                // checks the operand is `raw *T` AND the call site is inside
                // an `unsafe { }` block.
                let v = self.parse_unary()?;
                Ok(Expr::Free { value: Box::new(v), span: start })
            }
            TokKind::Transfer | TokKind::Share => {
                let mode = if matches!(self.peek(), TokKind::Transfer) {
                    WallMode::Transfer
                } else { WallMode::Share };
                self.bump();
                let e = self.parse_unary()?;
                Ok(Expr::WallMod { mode, expr: Box::new(e), span: start })
            }
            TokKind::Minus => {
                self.bump();
                let e = self.parse_unary()?;
                Ok(Expr::Un { op: UnOp::Neg, expr: Box::new(e), span: start })
            }
            TokKind::Bang => {
                self.bump();
                let e = self.parse_unary()?;
                Ok(Expr::Un { op: UnOp::Not, expr: Box::new(e), span: start })
            }
            TokKind::Amp => {
                self.bump();
                let mutness = if self.eat(&TokKind::Mut) { Mutness::Mut } else { Mutness::Default };
                // Parse inner without absorbing trailing `as`, so `&mut p as T` becomes `(&mut p) as T`.
                let e = self.parse_postfix_inner(false)?;
                let mut wrapped = Expr::Ref { mutness, expr: Box::new(e), span: start };
                loop {
                    match self.peek() {
                        TokKind::As => {
                            let s = self.peek_span();
                            self.bump();
                            let ty = self.parse_type()?;
                            wrapped = Expr::Cast { expr: Box::new(wrapped), ty, span: s };
                        }
                        _ => break,
                    }
                }
                Ok(wrapped)
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        self.parse_postfix_inner(true)
    }

    /// `allow_as`: if false, do not absorb trailing `as`/`as?` (caller handles them).
    fn parse_postfix_inner(&mut self, allow_as: bool) -> Result<Expr, ParseError> {
        let mut e = self.parse_primary()?;
        loop {
            match self.peek() {
                TokKind::Dot => {
                    let span = self.peek_span();
                    self.bump();
                    let (name, _) = self.expect_ident("field name")?;
                    // `Enum.Variant {fields}` — tagged-variant constructor.
                    if let Expr::Ident(enum_name, _) = &e {
                        if self.at(&TokKind::LBrace) && Self::is_struct_lit_context(self) {
                            self.expect(&TokKind::LBrace, "`{`")?;
                            let mut fields = Vec::new();
                            if !self.at(&TokKind::RBrace) {
                                loop {
                                    let (fname, _) = self.expect_ident("field name")?;
                                    self.expect(&TokKind::Eq, "`=`")?;
                                    let val = self.parse_expr()?;
                                    fields.push((fname, val));
                                    if !self.eat(&TokKind::Comma) { break; }
                                    if self.at(&TokKind::RBrace) { break; }
                                }
                            }
                            self.expect(&TokKind::RBrace, "`}`")?;
                            e = Expr::VariantCtor {
                                enum_name: enum_name.clone(),
                                variant: name,
                                fields,
                                span,
                            };
                            continue;
                        }
                    }
                    // `receiver.Attr::method(args)` — attr-qualified postfix
                    // call (§10.5).  Lower to `Call { callee: Attr.method,
                    // args: receiver :: args }` so the existing qualified-call
                    // path handles dispatch and the receiver flows through as
                    // arg 0 (with postfix auto-borrow).
                    if self.at(&TokKind::ColonColon) {
                        let save = self.pos;
                        self.bump();
                        if let TokKind::Ident(meth) = self.peek().clone() {
                            let meth_span = self.peek_span();
                            self.bump();
                            if self.at(&TokKind::LParen) {
                                self.bump();
                                let mut call_args = vec![e];
                                if !self.at(&TokKind::RParen) {
                                    loop {
                                        call_args.push(self.parse_expr()?);
                                        if !self.eat(&TokKind::Comma) { break; }
                                    }
                                }
                                self.expect(&TokKind::RParen, "`)`")?;
                                let callee = Expr::Field {
                                    base: Box::new(Expr::Ident(name, span)),
                                    name: meth,
                                    span: meth_span,
                                };
                                e = Expr::Call { callee: Box::new(callee), args: call_args, span };
                                continue;
                            }
                        }
                        // Not a qualified-method-call shape; rewind.
                        self.pos = save;
                    }
                    e = Expr::Field { base: Box::new(e), name, span };
                }
                TokKind::LBracket => {
                    let span = self.peek_span();
                    self.bump();
                    let idx = self.parse_expr()?;
                    self.expect(&TokKind::RBracket, "`]`")?;
                    e = Expr::Index { base: Box::new(e), idx: Box::new(idx), span };
                }
                TokKind::LParen => {
                    let span = self.peek_span();
                    self.bump();
                    let mut args = Vec::new();
                    if !self.at(&TokKind::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if !self.eat(&TokKind::Comma) { break; }
                        }
                    }
                    self.expect(&TokKind::RParen, "`)`")?;
                    e = Expr::Call { callee: Box::new(e), args, span };
                }
                TokKind::Bang => {
                    let span = self.peek_span();
                    self.bump();
                    e = Expr::Unwrap { expr: Box::new(e), span };
                }
                TokKind::As if allow_as => {
                    let span = self.peek_span();
                    self.bump();
                    let ty = self.parse_type()?;
                    e = Expr::Cast { expr: Box::new(e), ty, span };
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let span = self.peek_span();
        // Speculatively parse a lambda: `RetType(Type name, ...) [caps]? body`.
        if let Some(lam) = self.try_parse_lambda()? {
            return Ok(lam);
        }
        match self.peek().clone() {
            TokKind::Int(n) => { self.bump(); Ok(Expr::Lit(Lit::Int(n), span)) }
            TokKind::Float(f) => { self.bump(); Ok(Expr::Lit(Lit::Float(f), span)) }
            TokKind::StrLit(s) => { self.bump(); Ok(Expr::Lit(Lit::Str(s), span)) }
            TokKind::CharLit(c) => { self.bump(); Ok(Expr::Lit(Lit::Char(c), span)) }
            TokKind::True => { self.bump(); Ok(Expr::Lit(Lit::Bool(true), span)) }
            TokKind::False => { self.bump(); Ok(Expr::Lit(Lit::Bool(false), span)) }
            TokKind::Null => { self.bump(); Ok(Expr::Lit(Lit::Null, span)) }
            TokKind::LParen => {
                self.bump();
                if self.eat(&TokKind::RParen) {
                    return Ok(Expr::Lit(Lit::Unit, span));
                }
                let e = self.parse_expr()?;
                self.expect(&TokKind::RParen, "`)`")?;
                Ok(e)
            }
            TokKind::LBracket => {
                self.bump();
                let mut elems = Vec::new();
                if !self.at(&TokKind::RBracket) {
                    let first = self.parse_expr()?;
                    // `[expr; N]` fill-literal: parse the count and replicate
                    // the first element N times.  Count must be a compile-time
                    // integer (literal or constexpr).
                    if self.eat(&TokKind::Semicolon) {
                        let count_span = self.peek_span();
                        let count = self.try_fold_int().ok_or_else(|| ParseError {
                            msg: "expected a constant integer length after `;` in fill-literal".into(),
                            span: count_span,
                        })?;
                        self.expect(&TokKind::RBracket, "`]`")?;
                        if count < 0 {
                            return Err(ParseError { msg: "array fill-length must be non-negative".into(), span: count_span });
                        }
                        let mut filled = Vec::with_capacity(count as usize);
                        for _ in 0..count { filled.push(first.clone()); }
                        return Ok(Expr::ArrayLit { elems: filled, span });
                    }
                    elems.push(first);
                    while self.eat(&TokKind::Comma) {
                        if self.at(&TokKind::RBracket) { break; }
                        elems.push(self.parse_expr()?);
                    }
                }
                self.expect(&TokKind::RBracket, "`]`")?;
                Ok(Expr::ArrayLit { elems, span })
            }
            TokKind::Match => {
                self.bump();
                let (scrut, arms) = self.parse_match_after_kw()?;
                Ok(Expr::Match { scrutinee: Box::new(scrut), arms, span })
            }
            // `if (cond) { ... } else { ... }` as a value-yielding expression.
            // Desugars to `match (cond) { true { ... }, else { ... } }` so all
            // the existing match-arm typeck + codegen plumbing applies.  Arms
            // produce values via `yield` exactly like a match.  `else if` chains
            // are accepted by recursively parsing the right-hand side.
            TokKind::If => Ok(self.parse_if_expr()?),
            TokKind::LBrace => {
                // struct literal with inferred type (e.g. in `Slot s = { id = 1 };`)
                Ok(self.parse_struct_lit(None, span)?)
            }
            TokKind::Ident(name) => {
                self.bump();
                // If immediately followed by `{` then this is a struct literal with named type
                if self.at(&TokKind::LBrace) && Self::is_struct_lit_context(self) {
                    return Ok(self.parse_struct_lit(Some(name), span)?);
                }
                // `Attr::method(...)` — attr-qualified prefix call, synonym for
                // the existing `Attr.method(...)` shape (§10.5).  Lower to the
                // same Field node so the qualified-call path in `check_call`
                // picks it up unchanged.
                if self.at(&TokKind::ColonColon) {
                    let save = self.pos;
                    self.bump();
                    if let TokKind::Ident(seg) = self.peek().clone() {
                        let seg_span = self.peek_span();
                        self.bump();
                        if self.at(&TokKind::LParen) {
                            return Ok(Expr::Field {
                                base: Box::new(Expr::Ident(name, span)),
                                name: seg,
                                span: seg_span,
                            });
                        }
                    }
                    // Not a qualified call shape — rewind so the type parser
                    // (or whoever) gets a clean look.
                    self.pos = save;
                }
                // Module-scope constexpr substitution: if this ident names a constexpr and it's
                // used in a plain value position (not a call, indexing, field, generic, struct lit),
                // replace it with its folded integer literal.
                if let Some(&v) = self.constexprs.get(&name) {
                    if !matches!(self.peek(), TokKind::LParen | TokKind::LBracket | TokKind::Dot | TokKind::Lt | TokKind::Bang) {
                        return Ok(Expr::Lit(Lit::Int(v), span));
                    }
                }
                Ok(Expr::Ident(name, span))
            }
            other => Err(ParseError { msg: format!("expected expression, got {:?}", other), span }),
        }
    }

    fn is_struct_lit_context(&self) -> bool {
        // After `Name`, a `{` begins a struct literal if its body looks like field-init list
        // (i.e. `ident = expr` or `}` for empty).
        if !matches!(self.peek(), TokKind::LBrace) { return false; }
        // peek next non-`{` token
        let k1 = self.peek_at(1);
        if matches!(k1, TokKind::RBrace) { return true; }
        if let TokKind::Ident(_) = k1 {
            if matches!(self.peek_at(2), TokKind::Eq) { return true; }
        }
        false
    }

    /// After consuming `match`, parse `(scrut) { arms }` and return both.
    fn parse_match_after_kw(&mut self) -> Result<(Expr, Vec<MatchArm>), ParseError> {
        self.expect(&TokKind::LParen, "`(`")?;
        let scrut = self.parse_expr()?;
        self.expect(&TokKind::RParen, "`)`")?;
        self.expect(&TokKind::LBrace, "`{`")?;
        let mut arms = Vec::new();
        while !self.at(&TokKind::RBrace) {
            let astart = self.peek_span();
            let pat = self.parse_pattern()?;
            let guard = if self.at(&TokKind::If) {
                self.bump();
                Some(self.parse_expr()?)
            } else { None };
            // Body: either `{ block }` or a single expression
            let body = if self.at(&TokKind::LBrace) {
                ArmBody::Block(self.parse_block()?)
            } else {
                ArmBody::Expr(self.parse_expr()?)
            };
            arms.push(MatchArm { pattern: pat, guard, body, span: astart });
            // Separator: comma optional after block-bodied arm; required otherwise.
            self.eat(&TokKind::Comma);
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok((scrut, arms))
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        // Or-patterns use `|`, but our lexer doesn't have that single token.
        // We accept only single patterns for now.
        self.parse_pattern_atom()
    }

    fn parse_pattern_atom(&mut self) -> Result<Pattern, ParseError> {
        let start = self.peek_span();
        match self.peek() {
            TokKind::Null => { self.bump(); Ok(Pattern::Null(start)) }
            TokKind::Else => { self.bump(); Ok(Pattern::Else(start)) }
            TokKind::Int(n) => { let n = *n; self.bump(); Ok(Pattern::Lit(Lit::Int(n), start)) }
            TokKind::Float(f) => { let f = *f; self.bump(); Ok(Pattern::Lit(Lit::Float(f), start)) }
            TokKind::CharLit(c) => { let c = *c; self.bump(); Ok(Pattern::Lit(Lit::Char(c), start)) }
            TokKind::True => { self.bump(); Ok(Pattern::Lit(Lit::Bool(true), start)) }
            TokKind::False => { self.bump(); Ok(Pattern::Lit(Lit::Bool(false), start)) }
            TokKind::Minus => {
                self.bump();
                if let TokKind::Int(n) = *self.peek() { self.bump(); Ok(Pattern::Lit(Lit::Int(-n), start)) }
                else { Err(ParseError { msg: "expected integer after `-`".into(), span: self.peek_span() }) }
            }
            TokKind::Ident(_) => {
                let (name, _) = self.expect_ident("pattern")?;
                let mut enum_name: Option<String> = None;
                let mut variant = name.clone();
                if self.eat(&TokKind::Dot) {
                    let (vn, _) = self.expect_ident("variant name")?;
                    enum_name = Some(name);
                    variant = vn;
                }
                // Disambiguate `Variant{field, ...}` (destructure pattern) from
                // `Variant { stmt; ... }` (block-form match-arm body).  Peek past
                // the `{` - if it's empty, a stmt-keyword, or `Ident (` / `Ident .`
                // (call / method), it's a body block; let the match-arm parser
                // own the `{` and parse the block.  Otherwise it's a destructure.
                if self.at(&TokKind::LBrace) {
                    let looks_like_destructure = match (self.peek_at(1), self.peek_at(2)) {
                        (TokKind::RBrace, _) => false,
                        (TokKind::If, _) | (TokKind::While, _) | (TokKind::For, _)
                        | (TokKind::Return, _) | (TokKind::Mut, _)
                        | (TokKind::Unsafe, _) | (TokKind::Match, _) | (TokKind::Break, _)
                        | (TokKind::Continue, _) | (TokKind::Yield, _) | (TokKind::Propagate, _)
                        | (TokKind::Underscore, _) => false,
                        (TokKind::Ident(_), TokKind::LParen) => false,
                        (TokKind::Ident(_), TokKind::Dot)    => false,
                        // `Ident = ...` is an assignment statement — always a
                        // block body.  (Literal-match destructure `{ field = N }`
                        // is unused in practice; we tilt toward the common case.)
                        (TokKind::Ident(_), TokKind::Eq)     => false,
                        // `mut Type name = ...` and `Type name = ...` are let
                        // statements that need a block body.
                        (TokKind::Ident(_), TokKind::Ident(_)) => false,
                        _ => true,
                    };
                    if looks_like_destructure {
                        self.bump();
                        let mut fields = Vec::new();
                        while !self.at(&TokKind::RBrace) {
                            let fstart = self.peek_span();
                            let (fname, _) = self.expect_ident("field name")?;
                            let mut binding = None;
                            let mut literal = None;
                            if self.eat(&TokKind::Eq) {
                                match self.peek() {
                                    TokKind::Int(n) => { let n = *n; self.bump(); literal = Some(Lit::Int(n)); }
                                    TokKind::True => { self.bump(); literal = Some(Lit::Bool(true)); }
                                    TokKind::False => { self.bump(); literal = Some(Lit::Bool(false)); }
                                    TokKind::Ident(_) => {
                                        let (rename, _) = self.expect_ident("rename")?;
                                        binding = Some(rename);
                                    }
                                    other => return Err(ParseError {
                                        msg: format!("expected rename or literal, got {:?}", other),
                                        span: self.peek_span(),
                                    }),
                                }
                            }
                            fields.push(PatField { field: fname, binding, literal, span: fstart });
                            if !self.eat(&TokKind::Comma) { break; }
                            if self.at(&TokKind::RBrace) { break; }
                        }
                        self.expect(&TokKind::RBrace, "`}`")?;
                        return Ok(Pattern::VariantDestructure { enum_name, variant, fields, span: start });
                    }
                }
                if enum_name.is_some() {
                    Ok(Pattern::Variant { enum_name, variant, span: start })
                } else {
                    // bare identifier — treated as variant if it matches an enum variant at sema time,
                    // otherwise as a variable binding (used in guards).
                    Ok(Pattern::Ident(variant, start))
                }
            }
            other => Err(ParseError {
                msg: format!("expected pattern, got {:?}", other),
                span: start,
            }),
        }
    }

    /// Speculatively parse a lambda expression starting at the current position.
    /// Returns `Ok(Some(lambda))` if a lambda was parsed, `Ok(None)` to indicate no lambda here.
    /// Parse an `if (cond) { ... } else { ... }` expression and desugar it
    /// into a bool-match.  Else is required for expression-form ifs.
    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek_span();
        self.expect(&TokKind::If, "`if`")?;
        self.expect(&TokKind::LParen, "`(`")?;
        let cond = self.parse_expr()?;
        self.expect(&TokKind::RParen, "`)`")?;
        let then_blk = self.parse_block()?;
        self.expect(&TokKind::Else, "`else` (if-expression requires an else branch)")?;
        let else_body = if self.at(&TokKind::If) {
            let nested = self.parse_if_expr()?;
            ArmBody::Block(Block {
                stmts: vec![Stmt::Yield(nested, start)],
                span: start,
            })
        } else {
            ArmBody::Block(self.parse_block()?)
        };
        Ok(Expr::Match {
            scrutinee: Box::new(cond),
            arms: vec![
                MatchArm {
                    pattern: Pattern::Lit(Lit::Bool(true), start),
                    guard: None,
                    body: ArmBody::Block(then_blk),
                    span: start,
                },
                MatchArm {
                    pattern: Pattern::Else(start),
                    guard: None,
                    body: else_body,
                    span: start,
                },
            ],
            span: start,
        })
    }

    fn try_parse_lambda(&mut self) -> Result<Option<Expr>, ParseError> {
        let save = self.pos;
        let start = self.peek_span();
        // Try parsing a *base type* (a single type token without `(...)` postfix),
        // then `(Type name, ...)`, then optional `[caps]`, then body.
        let ret_ty = match self.parse_type_base() {
            Ok(t) => t,
            Err(_) => { self.pos = save; return Ok(None); }
        };
        if !self.at(&TokKind::LParen) { self.pos = save; return Ok(None); }
        self.bump(); // (
        // Try parsing a parameter list: `Type name, ...`. If we don't see `Type Ident` pattern, abort.
        let mut params: Vec<Param> = Vec::new();
        let mut first = true;
        if !self.at(&TokKind::RParen) {
            loop {
                let p_start = self.peek_span();
                let p_ty = match self.parse_type_base() {
                    Ok(t) => t,
                    Err(_) => { self.pos = save; return Ok(None); }
                };
                // Must be followed by an identifier.
                let p_name = match self.peek().clone() {
                    TokKind::Ident(n) => { self.bump(); n },
                    _ => { self.pos = save; return Ok(None); }
                };
                params.push(Param { name: p_name, ty: p_ty, span: p_start });
                if self.eat(&TokKind::Comma) { first = false; continue; }
                if self.at(&TokKind::RParen) { break; }
                self.pos = save; return Ok(None);
            }
        }
        let _ = first;
        if !self.eat(&TokKind::RParen) { self.pos = save; return Ok(None); }
        // Optional capture list `[name, &name, &mut name, ...]`.
        let mut captures: Vec<LambdaCapture> = Vec::new();
        if self.eat(&TokKind::LBracket) {
            if !self.at(&TokKind::RBracket) {
                loop {
                    let cap_start = self.peek_span();
                    let mode = if self.eat(&TokKind::Amp) {
                        if self.eat(&TokKind::Mut) { 'm' } else { 'r' }
                    } else { 'v' };
                    let cap_name = match self.peek().clone() {
                        TokKind::Ident(n) => { self.bump(); n },
                        _ => { self.pos = save; return Ok(None); }
                    };
                    captures.push(LambdaCapture { name: cap_name, mode, span: cap_start });
                    if !self.eat(&TokKind::Comma) { break; }
                    if self.at(&TokKind::RBracket) { break; }
                }
            }
            if !self.eat(&TokKind::RBracket) { self.pos = save; return Ok(None); }
        }
        // Body: either `{ block }` or a single expression.
        let body = if self.at(&TokKind::LBrace) {
            let b = match self.parse_block() {
                Ok(b) => b,
                Err(_) => { self.pos = save; return Ok(None); }
            };
            LambdaBody::Block(b)
        } else {
            let e = match self.parse_expr() {
                Ok(e) => e,
                Err(_) => { self.pos = save; return Ok(None); }
            };
            LambdaBody::Expr(Box::new(e))
        };
        Ok(Some(Expr::Lambda {
            ret: ret_ty,
            params,
            captures,
            body,
            span: start,
        }))
    }

    fn parse_struct_lit(&mut self, ty: Option<String>, span: Span) -> Result<Expr, ParseError> {
        self.expect(&TokKind::LBrace, "`{`")?;
        let mut fields = Vec::new();
        if !self.at(&TokKind::RBrace) {
            loop {
                let (fname, _) = self.expect_ident("field name")?;
                self.expect(&TokKind::Eq, "`=`")?;
                let val = self.parse_expr()?;
                fields.push((fname, val));
                if !self.eat(&TokKind::Comma) { break; }
                if self.at(&TokKind::RBrace) { break; } // trailing comma
            }
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok(Expr::Struct { ty, fields, span })
    }
}

pub fn parse(src: &str) -> Result<Module, String> {
    let toks = maka_lexer::Lexer::new(src).tokenize().map_err(|e| e.to_string())?;
    Parser::new(toks).parse_module().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_main() {
        let m = parse("unit main() { }").unwrap();
        assert_eq!(m.items.len(), 1);
    }
    #[test]
    fn let_and_assign() {
        let _ = parse("unit main() { mut int x = 1; x = 2; }").unwrap();
    }
    #[test]
    fn pointer_decl() {
        let _ = parse("unit main() { *int p = null; }").unwrap();
    }
    #[test]
    fn heap_decl() {
        let _ = parse("unit main() { heap int n = 42; }").unwrap();
    }
    #[test]
    fn struct_lit() {
        let m = parse("data V { int x = 0; } unit main() { V v = { x = 10 }; }").unwrap();
        assert_eq!(m.items.len(), 2);
    }
    #[test]
    fn casts() {
        let _ = parse("unit main() { int n = 1; float f = n as float; *int p = null; }").unwrap();
    }
    #[test]
    fn if_while_return() {
        let _ = parse("int main() { mut int i = 0; while (i < 10) { i = i + 1; } return i; }").unwrap();
    }
}
