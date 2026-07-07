//! The Maka layout formatter.
//!
//! This is a **re-indenter**, not a reflower: it never moves tokens between
//! lines and never touches the text inside strings or comments.  It only
//!
//!   * re-indents each line to `4 * bracket_depth` spaces (dedenting lines that
//!     begin with `}` / `)` / `]`),
//!   * strips trailing whitespace on code lines,
//!   * collapses runs of blank lines to a single blank, dropping leading and
//!     trailing blank lines, and
//!   * ensures the file ends with exactly one newline.
//!
//! Because Maka is not whitespace-sensitive (statements end with `;`, blocks are
//! `{ }`), re-indentation is semantics-preserving by construction.  Lines that
//! begin inside a multi-line block comment or a multi-line string are emitted
//! **verbatim**, so comment art and string contents are never disturbed.  The
//! transform is idempotent: `format(format(x)) == format(x)`.
//!
//! `format_checked` additionally re-lexes the input and the output and refuses
//! the result if the token stream changed — a belt-and-braces guard so that
//! format-on-save can never silently corrupt a file, even given a formatter bug.

use maka_lexer::{Lexer, TokKind};

/// State of the byte scanner as it crosses a line boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum St {
    Normal,
    Block(u32), // inside a (possibly nested) /* */ block comment, depth >= 1
    Str,        // inside a "..." string literal
    Chr,        // inside a '...' char literal
}

/// Per-line facts recovered by the scanner.
#[derive(Clone, Copy, Debug)]
struct LineInfo {
    /// Bracket depth (`{([` minus `)]}`, counted only outside strings/comments)
    /// at the first byte of the line.
    depth: i32,
    /// The line begins inside a block comment or a string literal, so its text
    /// must be preserved verbatim (re-indenting would corrupt comment art or,
    /// worse, string data).
    verbatim: bool,
}

/// Scan the source once, producing a `LineInfo` per line (line `k` corresponds
/// to `src.split('\n').nth(k)`).  Comments and string/char literals follow the
/// exact rules of the compiler lexer (nested `/* */`, `\`-escapes in strings and
/// chars), so brackets inside them never affect the depth.
fn scan(src: &[u8]) -> Vec<LineInfo> {
    let mut lines = Vec::new();
    let mut st = St::Normal;
    let mut depth: i32 = 0;
    // Line 0 starts at depth 0 in Normal state.
    lines.push(LineInfo { depth, verbatim: false });
    let mut i = 0usize;
    let n = src.len();
    // Push the record for the line beginning just after a '\n'.  The scanner's
    // current state `st` at the newline IS that next line's opening state, so a
    // line inside a block comment or an unterminated string is marked verbatim.
    macro_rules! newline {
        () => {{
            lines.push(LineInfo {
                depth,
                verbatim: matches!(st, St::Block(_) | St::Str | St::Chr),
            });
        }};
    }
    while i < n {
        let c = src[i];
        let c1 = if i + 1 < n { src[i + 1] } else { 0 };
        match st {
            St::Normal => match c {
                b'/' if c1 == b'/' => {
                    // Line comment: skip to end of line (do not enter a state
                    // that survives the newline).
                    i += 2;
                    while i < n && src[i] != b'\n' {
                        i += 1;
                    }
                }
                b'/' if c1 == b'*' => {
                    st = St::Block(1);
                    i += 2;
                }
                b'"' => {
                    st = St::Str;
                    i += 1;
                }
                b'\'' => {
                    st = St::Chr;
                    i += 1;
                }
                b'{' | b'(' | b'[' => {
                    depth += 1;
                    i += 1;
                }
                b'}' | b')' | b']' => {
                    depth -= 1;
                    i += 1;
                }
                b'\n' => {
                    i += 1;
                    newline!();
                }
                _ => i += 1,
            },
            St::Block(d) => {
                if c == b'/' && c1 == b'*' {
                    st = St::Block(d + 1);
                    i += 2;
                } else if c == b'*' && c1 == b'/' {
                    st = if d == 1 { St::Normal } else { St::Block(d - 1) };
                    i += 2;
                } else if c == b'\n' {
                    i += 1;
                    newline!();
                } else {
                    i += 1;
                }
            }
            St::Str => {
                if c == b'\\' {
                    i += 2; // skip the escaped byte (covers \" and \\)
                } else if c == b'"' {
                    st = St::Normal;
                    i += 1;
                } else if c == b'\n' {
                    i += 1;
                    newline!();
                } else {
                    i += 1;
                }
            }
            St::Chr => {
                if c == b'\\' {
                    i += 2;
                } else if c == b'\'' {
                    st = St::Normal;
                    i += 1;
                } else if c == b'\n' {
                    i += 1;
                    newline!();
                } else {
                    i += 1;
                }
            }
        }
    }
    lines
}

const INDENT: &str = "    ";

/// Format `src` as layout-only (see the module docs).  Infallible: on any input
/// it returns a re-indented, blank-normalized string.
pub fn format(src: &str) -> String {
    let infos = scan(src.as_bytes());
    let raw_lines: Vec<&str> = src.split('\n').collect();

    // Build the output lines, remembering which ones are verbatim so the
    // blank-collapse pass does not treat an empty comment/string line as a
    // droppable blank.
    let mut built: Vec<(String, bool)> = Vec::with_capacity(raw_lines.len());
    for (idx, raw) in raw_lines.iter().enumerate() {
        let info = infos.get(idx).copied().unwrap_or(LineInfo { depth: 0, verbatim: false });
        // Does this line *end* inside a string/block comment?  Equivalent to the
        // next line beginning verbatim.  If so, its trailing bytes are literal
        // data and must not be trimmed.
        let ends_verbatim = infos.get(idx + 1).map_or(false, |n| n.verbatim);

        if info.verbatim {
            // Wholly inside a block comment or multi-line string: untouched.
            built.push(((*raw).to_string(), true));
            continue;
        }

        let content = if ends_verbatim { raw.trim_start() } else { raw.trim() };
        if content.is_empty() {
            built.push((String::new(), false));
            continue;
        }

        // Lines beginning with closers dedent by the number of leading closers
        // (`}`, `})`, `}}` all handled).
        let mut lead_closers = 0i32;
        for &b in content.as_bytes() {
            if b == b'}' || b == b')' || b == b']' {
                lead_closers += 1;
            } else {
                break;
            }
        }
        let indent = (info.depth - lead_closers).max(0) as usize;
        let mut line = String::with_capacity(indent * INDENT.len() + content.len());
        for _ in 0..indent {
            line.push_str(INDENT);
        }
        line.push_str(content);
        built.push((line, false));
    }

    // Collapse blank runs to a single blank; drop leading and trailing blanks.
    // Verbatim lines count as content (they reset any pending blank), so blank
    // lines inside a block comment survive.
    let mut out: Vec<String> = Vec::with_capacity(built.len());
    let mut pending_blank = false;
    let mut seen_content = false;
    for (line, verbatim) in built {
        let is_blank = !verbatim && line.is_empty();
        if is_blank {
            if seen_content {
                pending_blank = true;
            }
            continue;
        }
        if pending_blank {
            out.push(String::new());
            pending_blank = false;
        }
        out.push(line);
        seen_content = true;
    }

    let mut s = out.join("\n");
    if !s.is_empty() {
        s.push('\n');
    }
    s
}

/// Format, then verify the result lexes to the *same* token stream as the input.
/// Returns `Err` (and the original is left untouched by callers) if the token
/// streams differ or either fails to lex — a hard guarantee that formatting is
/// token-preserving, safe enough to run on every save.
pub fn format_checked(src: &str) -> std::result::Result<String, String> {
    let out = format(src);
    // Fast path: nothing changed.
    if out == src {
        return Ok(out);
    }
    let before = lex_kinds(src).map_err(|e| format!("input does not lex: {}", e))?;
    let after = lex_kinds(&out).map_err(|e| format!("formatter produced un-lexable output: {}", e))?;
    if before != after {
        return Err(format!(
            "formatter changed the token stream ({} tokens -> {}); refusing to apply",
            before.len(),
            after.len()
        ));
    }
    Ok(out)
}

/// The token *kinds* of `src`, for equivalence checking.  Comments and
/// whitespace are not tokens, so two layouts of the same code compare equal.
fn lex_kinds(src: &str) -> std::result::Result<Vec<TokKind>, String> {
    Lexer::new(src)
        .tokenize()
        .map(|toks| toks.into_iter().map(|t| t.kind).collect())
        .map_err(|e| e.to_string())
}

/// Command-line entry point for `makac fmt` / `maka fmt`.
///
/// `run(["--check", FILE...])` reports which files are not formatted (exit 1 if
/// any) without writing.  `run([FILE...])` rewrites each file in place, printing
/// the ones it changed.  A file whose format would change the token stream (a
/// formatter bug) is reported and left untouched; it never corrupts a file.
pub fn run(args: &[String]) -> i32 {
    let mut check = false;
    let mut files: Vec<&String> = Vec::new();
    for a in args {
        match a.as_str() {
            "--check" | "-c" => check = true,
            "--help" | "-h" => {
                eprintln!("usage: makac fmt [--check] <file.maka>...");
                return 0;
            }
            s if s.starts_with('-') => {
                eprintln!("maka fmt: unknown flag `{}`", s);
                return 2;
            }
            _ => files.push(a),
        }
    }
    if files.is_empty() {
        eprintln!("usage: makac fmt [--check] <file.maka>...");
        return 2;
    }

    let mut had_error = false; // a file failed to read / format
    let mut unformatted = 0usize; // files that (would) change

    for f in files {
        let src = match std::fs::read_to_string(f) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("maka fmt: {}: {}", f, e);
                had_error = true;
                continue;
            }
        };
        let out = match format_checked(&src) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("maka fmt: {}: {}", f, e);
                had_error = true;
                continue;
            }
        };
        if out == src {
            continue;
        }
        unformatted += 1;
        if check {
            println!("would reformat {}", f);
        } else if let Err(e) = std::fs::write(f, &out) {
            eprintln!("maka fmt: {}: {}", f, e);
            had_error = true;
        } else {
            println!("formatted {}", f);
        }
    }

    if had_error {
        return 1;
    }
    if check && unformatted > 0 {
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idem(src: &str) -> String {
        let once = format(src);
        assert_eq!(once, format(&once), "not idempotent:\n{}", once);
        once
    }

    #[test]
    fn reindents_blocks() {
        let src = "unit main() {\nlog(1);\n     if (x) {\n  y();\n}\n}\n";
        let want = "unit main() {\n    log(1);\n    if (x) {\n        y();\n    }\n}\n";
        assert_eq!(idem(src), want);
    }

    #[test]
    fn multiline_call_and_struct() {
        let src = "foo(\na,\nb\n);\nPoint p = {\nx = 1,\ny = 2,\n};\n";
        let want = "foo(\n    a,\n    b\n);\nPoint p = {\n    x = 1,\n    y = 2,\n};\n";
        assert_eq!(idem(src), want);
    }

    #[test]
    fn braces_in_strings_and_comments_do_not_count() {
        let src = "unit f() {\nlog(\"}{)(\");\n// } { ) (\ny();\n}\n";
        let want = "unit f() {\n    log(\"}{)(\");\n    // } { ) (\n    y();\n}\n";
        assert_eq!(idem(src), want);
    }

    #[test]
    fn block_comment_interior_is_verbatim() {
        // The middle lines of a /* */ keep their original indentation exactly,
        // including a leading `}` that is only text.
        let src = "unit f() {\n/*\n   } not code\n     aligned art\n*/\nx();\n}\n";
        let out = idem(src);
        assert!(out.contains("\n   } not code\n"), "interior changed:\n{}", out);
        assert!(out.contains("\n     aligned art\n"), "interior changed:\n{}", out);
        // Code after the comment is re-indented to depth 1.
        assert!(out.contains("\n    x();\n"), "post-comment not reindented:\n{}", out);
    }

    #[test]
    fn nested_block_comment() {
        let src = "unit f() {\n/* outer /* inner */ still */\nx();\n}\n";
        let out = idem(src);
        assert!(out.contains("\n    x();\n"), "depth wrong after nested comment:\n{}", out);
    }

    #[test]
    fn multiline_string_body_untouched() {
        // A literal newline inside a string: the continuation line must not be
        // re-indented, or the string's contents would change.
        let src = "unit f() {\nstring s = \"line1\n   line2 keeps    spaces\";\nx();\n}\n";
        let out = format(src);
        assert!(out.contains("\n   line2 keeps    spaces\";\n"), "string body changed:\n{}", out);
    }

    #[test]
    fn collapses_blank_runs_and_trims_edges() {
        let src = "\n\nunit a() {}\n\n\n\nunit b() {}\n\n\n";
        let want = "unit a() {}\n\nunit b() {}\n";
        assert_eq!(idem(src), want);
    }

    #[test]
    fn multiple_leading_closers() {
        let src = "unit f() {\nwhile (x) {\ng(a(\nb\n));\n}\n}\n";
        // The `));` line has two leading closers and dedents accordingly.
        let out = idem(src);
        assert!(out.contains("\n        ));\n"), "double-closer indent wrong:\n{}", out);
    }

    #[test]
    fn checked_matches_and_is_token_preserving() {
        let src = "unit main(){\nint x=1;\n log( x );\n}\n";
        let out = format_checked(src).expect("should be token-preserving");
        assert_eq!(out, format(src));
    }

    #[test]
    fn already_formatted_is_stable() {
        let src = "unit main() {\n    log(1);\n}\n";
        assert_eq!(format(src), src);
    }
}
