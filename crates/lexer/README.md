# maka_lexer

Source text -> a token stream with spans.

- **Job:** tokenize `.maka` source; own the lexical grammar (keywords, literals,
  operators, comments) and `Span` (byte offsets for diagnostics and `#line`).
- **DAG:** leaf crate - depends on nothing. Every other crate depends on it.
- **Never:** know about grammar or meaning. A token is shape only; whether a
  sequence of tokens is valid is the parser's problem.
