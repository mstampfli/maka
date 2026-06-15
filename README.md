# Maka

A systems language with explicit ownership, structural traits via `attr`/`has`,
and inline functions that can propagate errors through the caller's frame.
Compiles to portable C.

Maka is opinionated about a small set of things:

- **Pointers tell you what they are.** Five pointer kinds, each with a single
  job: `&T`/`&mut T` for tracked borrows, `*T` for nullable non-owning, `own
  *T` / `own &T` for nullable / strict heap owners, `raw *T` for everything
  the compiler can't see (C interop, fixed addresses, unsafe code).
- **`alloc value` is the only allocator.** It must land in an owning slot.
- **Forced handling.** You can't deref a nullable pointer without proof of
  non-null at the deref site - narrowing windows, early-exit guards, or
  immediate-from-`alloc`. No `MAKA_UNWRAP` macro, no runtime null check.
- **`attr` + `has` instead of traits + impl.** An `attr` declares the
  contract (with optional default bodies); `Type has Attr { ... }` provides
  the implementation. `_` is the placeholder for the implementing type.
- **Inline functions splice into the caller's frame.** `inline fn f() {
  propagate X; }` returns `X` from *the outer function*, not from `f`. Useful
  for shaped early-exit. Recursion among inline functions is forbidden.

## Hello, Maka

```maka
unit main() {
    log("hello, world");
}
```

```sh
makac hello.maka -o hello --run
```

## A taste

```maka
attr Show {
    string label(&_ self) { return "any"; }   // default body
}

data Point { mut int x; mut int y; }

Point has Show {
    string label(&Point self) { return "point"; }
}

unit render<T: Show>(&T x) {
    log(x.label());
}

unit main() {
    Point p = { x = 3, y = 4 };
    render(&p);                                  // "point"
}
```

Ownership through an `inline` callee:

```maka
data Box { int v; }

inline int take(own *Box b) {
    if (b == null) { return -1; }
    return b!.v;     // forced-handling proof comes from the guard above
}

unit main() {
    own *Box b = alloc Box { v = 42 };
    log(take(b));    // 42 - ownership moves into `take`, freed at splice exit
}
```

User iterators:

```maka
data Counter { mut int cur; int end; }

logic CounterIter {
    Option<int> next(&mut Counter self) {
        if (self.cur >= self.end) { return Option.None; }
        int v = self.cur;
        self.cur = self.cur + 1;
        return Option.Some{ value = v };
    }
}

unit main() {
    Counter c = { cur = 10, end = 14 };
    for (int x in c) { log(x); }   // 10 11 12 13
}
```

## Project layout

```
crates/
  lexer/        tokens, span, source
  ast/          surface syntax
  parser/       tokens -> AST
  sema/         HIR, resolution, type check, lifetime/move
  codegen/      C emission
  driver/       CLI: makac
stdlib/
  std.maka      real Maka source for the stdlib (Option, Result, str_*),
                embedded into the compiler at build time via include_str!
tests/
  programs/     end-to-end .maka tests with .expected output
  run_all.sh    positive suite
  run_neg.sh    negative suite (every neg_*.maka must fail to compile)
```

The stdlib lives in `module std;`. Types like `Option<T>` and `Result<T, E>`
are accessible cross-module without import (same rule as any `pub data`);
functions like `str_len` and `str_eq` need an explicit
`import std.{str_len, str_eq};`.

## Building

```sh
cargo build --release
./target/release/makac hello.maka -o hello --run
```

## Driver flags

```
makac <input.maka>... [-o output] [--emit-c] [--run]
                      [--link <file|flag>] [-l name] [-L path]
```

- Multiple `.maka` inputs are merged; each retains its declared module path
  for `pub` enforcement.
- `--emit-c` writes the generated C alongside or instead of compiling.
- `--run` invokes the compiled binary after build.
- `--link foo.c` compiles a C source alongside the generated code.
- `-l name` / `-L /path` pass through to the C linker.

## Spec

The current language spec lives at `SPEC.md` and is the
authoritative description of every feature: lexical structure, types,
expressions, statements, items, ownership rules, concurrency, pattern
matching, closures, generics, modules, C interop, codegen lowering, built-in
functions, driver invocation, and what the implementation deliberately omits.

## Tests

```sh
bash tests/run_all.sh        # positive suite, expects stdout match
bash tests/run_neg.sh        # negative suite, every neg_*.maka must reject
```

## Status

Early but real. Positive suite passes 95 programs; negative suite passes 40.
The language is implemented in Rust as a workspace; the generated C builds
with any reasonably recent gcc/clang.

What's intentionally *not* in v1, but is on the road:

- Auto-borrow at method calls (today: `(&p).method()` when the method's
  receiver is `&_ self`, since dispatch matches receiver type exactly).
- `Attr.method(x)` qualified-call form.
- A `format(...)` for typed string interpolation.
- File I/O beyond stdin/stdout.

## License

MIT.
