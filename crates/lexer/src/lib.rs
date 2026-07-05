//! Maka lexer (spec v1.2).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: u32,
    pub col: u32,
}

impl Span {
    pub fn new(start: usize, end: usize, line: u32, col: u32) -> Self {
        Self { start, end, line, col }
    }
    pub fn dummy() -> Self {
        Self { start: 0, end: 0, line: 0, col: 0 }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokKind {
    // literals
    Int(i64),
    Float(f64),
    StrLit(String),
    CharLit(char),
    True,
    False,
    Null,
    Ident(String),

    // keywords
    Mut,
    Const,
    Alloc,
    Free,
    Unsafe,
    Extern,
    Cinclude,
    Cblock,
    Rblock,
    Rdep,
    Raw,
    Own,
    Logic,
    Attr,
    Has,
    Where,
    Dyn,
    Some,
    Export,
    Match,
    Yield,
    Constexpr,
    Inline,
    Gate,
    Propagate,
    For,
    In,
    Break,
    Continue,
    DotDot,
    DotDotEq,
    Transfer,
    Share,
    ThreadLocal,
    Module,
    Import,
    Use,
    Pub,
    Data,
    Enum,
    Embed,
    If,
    Else,
    While,
    Return,
    As,        // `as` — the cast operator (target's nullability decides whether the cast can fail)
    Unit,      // the keyword `unit`
    Type,      // the keyword `type` — used in `attr { type Name; }` and `has { type Name = T; }`
    ColonColon,// `::` — path separator for `T::Slot` assoc-type paths
    // primitive type names left as plain Idents handled by parser/types

    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Dot,
    Colon,
    Underscore,

    // operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,         // & (also bit-and via overload)
    Pipe,        // | (bit-or)
    Caret,       // ^ (bit-xor)
    Tilde,       // ~ (unary bit-not)
    ShlOp,       // <<
    ShrOp,       // >>
    Bang,        // ! (unary or postfix unwrap)
    AndAnd,
    OrOr,
    Eq,          // =
    EqEq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,

    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct LexError {
    pub msg: String,
    pub span: Span,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lex error at {}: {}", self.span, self.msg)
    }
}

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src: src.as_bytes(), pos: 0, line: 1, col: 1 }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, LexError> {
        let mut out = Vec::new();
        loop {
            self.skip_trivia()?;
            if self.eof() {
                out.push(Token { kind: TokKind::Eof, span: self.here(0) });
                return Ok(out);
            }
            let tok = self.next_token()?;
            out.push(tok);
        }
    }

    fn eof(&self) -> bool { self.pos >= self.src.len() }

    fn peek(&self, o: usize) -> u8 {
        if self.pos + o < self.src.len() { self.src[self.pos + o] } else { 0 }
    }

    fn bump(&mut self) -> u8 {
        let c = self.src[self.pos];
        self.pos += 1;
        if c == b'\n' { self.line += 1; self.col = 1; } else { self.col += 1; }
        c
    }

    fn here(&self, len: usize) -> Span {
        Span::new(self.pos.saturating_sub(len), self.pos, self.line, self.col)
    }

    fn span_from(&self, start: usize, sline: u32, scol: u32) -> Span {
        Span::new(start, self.pos, sline, scol)
    }

    fn skip_trivia(&mut self) -> Result<(), LexError> {
        loop {
            while !self.eof() && matches!(self.peek(0), b' ' | b'\t' | b'\r' | b'\n') {
                self.bump();
            }
            if self.peek(0) == b'/' && self.peek(1) == b'/' {
                while !self.eof() && self.peek(0) != b'\n' { self.bump(); }
                continue;
            }
            if self.peek(0) == b'/' && self.peek(1) == b'*' {
                self.bump(); self.bump();
                let mut depth = 1;
                while !self.eof() && depth > 0 {
                    if self.peek(0) == b'/' && self.peek(1) == b'*' {
                        self.bump(); self.bump(); depth += 1;
                    } else if self.peek(0) == b'*' && self.peek(1) == b'/' {
                        self.bump(); self.bump(); depth -= 1;
                    } else {
                        self.bump();
                    }
                }
                if depth != 0 {
                    return Err(LexError { msg: "unterminated block comment".into(), span: self.here(0) });
                }
                continue;
            }
            break;
        }
        Ok(())
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        let sline = self.line;
        let scol = self.col;
        let c = self.peek(0);

        // identifier / keyword
        if is_ident_start(c) {
            while is_ident_cont(self.peek(0)) { self.bump(); }
            let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
            let kind = match text {
                "mut" => TokKind::Mut,
                "const" => TokKind::Const,
                "alloc" => TokKind::Alloc,
                "free" => TokKind::Free,
                "unsafe" => TokKind::Unsafe,
                "extern" => TokKind::Extern,
                "logic" => TokKind::Logic,
                "attr" => TokKind::Attr,
                "has" => TokKind::Has,
                "where" => TokKind::Where,
                "dyn" => TokKind::Dyn,
                "some" => TokKind::Some,
                "export" => TokKind::Export,
                "match" => TokKind::Match,
                "yield" => TokKind::Yield,
                "constexpr" => TokKind::Constexpr,
                "inline" => TokKind::Inline,
                "gate" => TokKind::Gate,
                "propagate" => TokKind::Propagate,
                "for" => TokKind::For,
                "in" => TokKind::In,
                "break" => TokKind::Break,
                "continue" => TokKind::Continue,
                "transfer" => TokKind::Transfer,
                "share" => TokKind::Share,
                "thread_local" => TokKind::ThreadLocal,
                "module" => TokKind::Module,
                "cinclude" => TokKind::Cinclude,
                "cblock" => TokKind::Cblock,
                "rblock" => TokKind::Rblock,
                "rdep" => TokKind::Rdep,
                "raw" => TokKind::Raw,
                "own" => TokKind::Own,
                "import" => TokKind::Import,
                "use" => TokKind::Use,
                "pub" => TokKind::Pub,
                "data" => TokKind::Data,
                "enum" => TokKind::Enum,
                "embed" => TokKind::Embed,
                "if" => TokKind::If,
                "else" => TokKind::Else,
                "while" => TokKind::While,
                "return" => TokKind::Return,
                "true" => TokKind::True,
                "false" => TokKind::False,
                "null" => TokKind::Null,
                "unit" => TokKind::Unit,
                "type" => TokKind::Type,
                "as" => TokKind::As,
                "_" => TokKind::Underscore,
                _ => TokKind::Ident(text.to_string()),
            };
            return Ok(Token { kind, span: self.span_from(start, sline, scol) });
        }

        // number
        if c.is_ascii_digit() {
            return self.number(start, sline, scol);
        }

        // char literal
        if c == b'\'' {
            return self.char_lit(start, sline, scol);
        }

        // string literal
        if c == b'"' {
            return self.str_lit(start, sline, scol);
        }

        // punctuation/operators
        let two = (c, self.peek(1));
        let kind = match two {
            (b'=', b'=') => { self.bump(); self.bump(); TokKind::EqEq }
            (b'!', b'=') => { self.bump(); self.bump(); TokKind::NotEq }
            (b'<', b'=') => { self.bump(); self.bump(); TokKind::LtEq }
            (b'>', b'=') => { self.bump(); self.bump(); TokKind::GtEq }
            (b'&', b'&') => { self.bump(); self.bump(); TokKind::AndAnd }
            (b'|', b'|') => { self.bump(); self.bump(); TokKind::OrOr }
            (b'+', b'=') => { self.bump(); self.bump(); TokKind::PlusEq }
            (b'-', b'=') => { self.bump(); self.bump(); TokKind::MinusEq }
            (b'*', b'=') => { self.bump(); self.bump(); TokKind::StarEq }
            (b'/', b'=') => { self.bump(); self.bump(); TokKind::SlashEq }
            (b'%', b'=') => { self.bump(); self.bump(); TokKind::PercentEq }
            (b':', b':') => { self.bump(); self.bump(); TokKind::ColonColon }
            (b'.', b'.') => {
                self.bump(); self.bump();
                if self.peek(0) == b'=' { self.bump(); TokKind::DotDotEq } else { TokKind::DotDot }
            }
            (b'<', b'<') => { self.bump(); self.bump(); TokKind::ShlOp }
            (b'>', b'>') => { self.bump(); self.bump(); TokKind::ShrOp }
            _ => {
                let one = self.bump();
                match one {
                    b'+' => TokKind::Plus,
                    b'-' => TokKind::Minus,
                    b'*' => TokKind::Star,
                    b'/' => TokKind::Slash,
                    b'%' => TokKind::Percent,
                    b'&' => TokKind::Amp,
                    b'|' => TokKind::Pipe,
                    b'^' => TokKind::Caret,
                    b'~' => TokKind::Tilde,
                    b'!' => TokKind::Bang,
                    b'=' => TokKind::Eq,
                    b'<' => TokKind::Lt,
                    b'>' => TokKind::Gt,
                    b'(' => TokKind::LParen,
                    b')' => TokKind::RParen,
                    b'{' => TokKind::LBrace,
                    b'}' => TokKind::RBrace,
                    b'[' => TokKind::LBracket,
                    b']' => TokKind::RBracket,
                    b',' => TokKind::Comma,
                    b';' => TokKind::Semicolon,
                    b'.' => TokKind::Dot,
                    b':' => TokKind::Colon,
                    b'?' => return Err(LexError {
                        msg: "'?' is not part of the language; nullability lives in the type (`*T`), not in a sigil. For a cast that can fail, write `expr as *Type` — the result is nullable and you get `null` on failure.".into(),
                        span: self.span_from(start, sline, scol),
                    }),
                    other => return Err(LexError {
                        msg: format!("unexpected character {:?}", other as char),
                        span: self.span_from(start, sline, scol),
                    }),
                }
            }
        };
        Ok(Token { kind, span: self.span_from(start, sline, scol) })
    }

    fn number(&mut self, start: usize, sline: u32, scol: u32) -> Result<Token, LexError> {
        let first = self.peek(0);
        // hex / bin / oct
        if first == b'0' && matches!(self.peek(1), b'x' | b'X' | b'b' | b'B' | b'o' | b'O') {
            self.bump();
            let base_ch = self.bump();
            let radix: u32 = match base_ch {
                b'x' | b'X' => 16,
                b'b' | b'B' => 2,
                _ => 8,
            };
            let digs = self.pos;
            while is_radix_digit(self.peek(0), radix) || self.peek(0) == b'_' { self.bump(); }
            let raw: String = self.src[digs..self.pos].iter()
                .filter(|&&c| c != b'_').map(|&b| b as char).collect();
            if raw.is_empty() {
                return Err(LexError { msg: "missing digits in integer literal".into(), span: self.span_from(start, sline, scol) });
            }
            // Parse the magnitude as u64 (the `-` sign is a separate token) and
            // bit-reinterpret into the i64 token, so the full unsigned range is
            // accepted (e.g. 0xFFFFFFFFFFFFFFFF for a u64 mask).  Values fitting
            // i64 are unchanged; only > u64::MAX is rejected.
            let v = u64::from_str_radix(&raw, radix).map(|u| u as i64)
                .map_err(|e| LexError { msg: e.to_string(), span: self.span_from(start, sline, scol) })?;
            // optional width suffix
            self.eat_width_suffix();
            return Ok(Token { kind: TokKind::Int(v), span: self.span_from(start, sline, scol) });
        }

        while self.peek(0).is_ascii_digit() || self.peek(0) == b'_' { self.bump(); }
        let mut is_float = false;
        if self.peek(0) == b'.' && self.peek(1).is_ascii_digit() {
            is_float = true;
            self.bump();
            while self.peek(0).is_ascii_digit() || self.peek(0) == b'_' { self.bump(); }
        }
        if matches!(self.peek(0), b'e' | b'E') {
            is_float = true;
            self.bump();
            if matches!(self.peek(0), b'+' | b'-') { self.bump(); }
            while self.peek(0).is_ascii_digit() { self.bump(); }
        }
        // Capture text up to here BEFORE the optional width suffix.
        let end_of_number = self.pos;
        let suffix_was_float = self.eat_width_suffix();
        let txt: String = self.src[start..end_of_number].iter()
            .filter(|&&c| c != b'_').map(|&b| b as char).collect();
        let kind = if is_float || suffix_was_float {
            TokKind::Float(txt.parse::<f64>().map_err(|e| LexError { msg: e.to_string(), span: self.span_from(start, sline, scol) })?)
        } else {
            TokKind::Int(txt.parse::<u64>().map(|u| u as i64).map_err(|e| LexError { msg: e.to_string(), span: self.span_from(start, sline, scol) })?)
        };
        Ok(Token { kind, span: self.span_from(start, sline, scol) })
    }

    /// Consume an optional integer/float width suffix (`i8`..`u64`, `usize`, `isize`, `f32`, `f64`).
    /// Returns true if it was a float suffix.
    fn eat_width_suffix(&mut self) -> bool {
        let p = self.pos;
        // Match against a known suffix.
        let suffixes_f: &[&[u8]] = &[b"f32", b"f64"];
        let suffixes_i: &[&[u8]] = &[
            b"i8", b"i16", b"i32", b"i64", b"isize",
            b"u8", b"u16", b"u32", b"u64", b"usize",
        ];
        // Try longest first to avoid `i16` matching `i1`.
        for s in suffixes_f {
            if self.src.get(p..p + s.len()) == Some(*s)
                && !self.src.get(p + s.len()).map_or(false, |c| is_ident_cont(*c)) {
                for _ in 0..s.len() { self.bump(); }
                return true;
            }
        }
        for s in suffixes_i {
            if self.src.get(p..p + s.len()) == Some(*s)
                && !self.src.get(p + s.len()).map_or(false, |c| is_ident_cont(*c)) {
                for _ in 0..s.len() { self.bump(); }
                return false;
            }
        }
        false
    }

    fn char_lit(&mut self, start: usize, sline: u32, scol: u32) -> Result<Token, LexError> {
        self.bump(); // opening quote
        let c = match self.peek(0) {
            b'\\' => {
                self.bump();
                let esc = self.bump();
                match esc {
                    b'n' => '\n', b't' => '\t', b'r' => '\r', b'\\' => '\\',
                    b'\'' => '\'', b'"' => '"', b'0' => '\0',
                    b'x' => {
                        let h1 = self.bump();
                        let h2 = self.bump();
                        let hex_byte = |b: u8| (b as char).to_digit(16);
                        match (hex_byte(h1), hex_byte(h2)) {
                            (Some(a), Some(b)) => ((a * 16 + b) as u8) as char,
                            _ => return Err(LexError { msg: "invalid \\xNN escape".into(), span: self.span_from(start, sline, scol) }),
                        }
                    }
                    other => return Err(LexError {
                        msg: format!("unknown escape '\\{}'", other as char),
                        span: self.span_from(start, sline, scol),
                    }),
                }
            }
            0 => return Err(LexError { msg: "unterminated char literal".into(), span: self.span_from(start, sline, scol) }),
            _ => self.bump() as char,
        };
        if self.peek(0) != b'\'' {
            return Err(LexError { msg: "expected closing '".into(), span: self.span_from(start, sline, scol) });
        }
        self.bump();
        Ok(Token { kind: TokKind::CharLit(c), span: self.span_from(start, sline, scol) })
    }

    fn str_lit(&mut self, start: usize, sline: u32, scol: u32) -> Result<Token, LexError> {
        self.bump(); // opening "
        let mut s = String::new();
        loop {
            if self.eof() {
                return Err(LexError { msg: "unterminated string literal".into(), span: self.span_from(start, sline, scol) });
            }
            let c = self.peek(0);
            if c == b'"' { self.bump(); break; }
            if c == b'\\' {
                self.bump();
                let esc = self.bump();
                let ch = match esc {
                    b'n' => '\n', b't' => '\t', b'r' => '\r', b'\\' => '\\',
                    b'\'' => '\'', b'"' => '"', b'0' => '\0',
                    b'x' => {
                        let h1 = self.bump();
                        let h2 = self.bump();
                        let hex_byte = |b: u8| (b as char).to_digit(16);
                        match (hex_byte(h1), hex_byte(h2)) {
                            (Some(a), Some(b)) => ((a * 16 + b) as u8) as char,
                            _ => return Err(LexError { msg: "invalid \\xNN escape".into(), span: self.span_from(start, sline, scol) }),
                        }
                    }
                    other => return Err(LexError {
                        msg: format!("unknown escape '\\{}'", other as char),
                        span: self.span_from(start, sline, scol),
                    }),
                };
                s.push(ch);
            } else {
                s.push(self.bump() as char);
            }
        }
        Ok(Token { kind: TokKind::StrLit(s), span: self.span_from(start, sline, scol) })
    }
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}
fn is_ident_cont(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}
fn is_radix_digit(c: u8, radix: u32) -> bool {
    (c as char).to_digit(radix).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lex(s: &str) -> Vec<TokKind> {
        Lexer::new(s).tokenize().unwrap().into_iter().map(|t| t.kind).collect()
    }
    #[test]
    fn basics() {
        let t = lex("mut int x = 42;");
        assert_eq!(t[0], TokKind::Mut);
        assert!(matches!(t[1], TokKind::Ident(ref s) if s == "int"));
        assert!(matches!(t[3], TokKind::Eq));
        assert!(matches!(t[4], TokKind::Int(42)));
    }
    #[test]
    fn pointer_unwrap() {
        let t = lex("p! = null");
        assert!(matches!(t[1], TokKind::Bang));
        assert!(matches!(t[3], TokKind::Null));
    }
    #[test]
    fn as_question_rejected() {
        // `as?` no longer exists — `?` is rejected everywhere.  Fallible
        // casts spell themselves as `expr as *T` (target nullability says it).
        assert!(Lexer::new("x as? T").tokenize().is_err());
    }
    #[test]
    fn question_rejected() {
        assert!(Lexer::new("x?").tokenize().is_err());
    }
    #[test]
    fn floats() {
        let t = lex("3.14 1e9 2.5e-3");
        assert!(matches!(t[0], TokKind::Float(_)));
        assert!(matches!(t[1], TokKind::Float(_)));
        assert!(matches!(t[2], TokKind::Float(_)));
    }
    #[test]
    fn strings() {
        let t = lex(r#""hello\n""#);
        if let TokKind::StrLit(s) = &t[0] { assert_eq!(s, "hello\n"); } else { panic!() }
    }
}
