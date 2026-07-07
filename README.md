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
- **Compile-time evaluation.** A `constexpr` function folds to a value in
  constant positions (array sizes, `constexpr` initializers) and is still a
  normal function at run time. `inline for (f in fields(v))` unrolls once per
  struct field, so derive-style code (print, sum, serialize) is written once.

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

Compile-time evaluation and reflection:

```maka
// Folded at compile time; also a normal function at run time.
constexpr int fib(int n) {
    if (n < 2) { return n; }
    return fib(n - 1) + fib(n - 2);
}

data Vec3 { int x; int y; int z; }

// `inline for` unrolls once per field; `f.value` has each field's own type,
// so one body covers every struct.
int sum<T>(&T v) {
    mut int total = 0;
    inline for (f in fields(v)) { total = total + f.value; }
    return total;
}

unit main() {
    [fib(6)]int row = [0; fib(6)];     // length folded to 8 at compile time
    log(row.len);                      // 8
    Vec3 p = { x = 7, y = 8, z = 9 };
    log(sum(&p));                      // 24
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
  std.maka      real Maka source for the stdlib (Option, Result, str_*,
                Vec<T>, HashMap<V>, Atomic<T>, file I/O, concurrency),
                embedded into the compiler at build time via include_str!
tests/
  programs/     end-to-end .maka tests with .expected output
  run_all.sh    positive suite
  run_neg.sh    negative suite (every neg_*.maka must fail to compile)
```

The stdlib lives in `module std;`. Every cross-module item - types, enums,
functions, attrs - requires an explicit `import`.  Programs that touch
stdlib write `import std.Option;`, `import std.Result;`,
`import std.{str_len, str_eq};` etc.  The error message at every unimported
cross-module reference names the exact import line to add.

## Building

```sh
cargo build --release
./target/release/makac hello.maka -o hello --run
```

## Installing (put `maka` and `makac` on PATH)

```sh
./install.sh                    # builds release, installs both to ~/.cargo/bin
BINDIR=~/.local/bin ./install.sh   # or choose the directory
```

`maka` locates `makac` as a sibling, so both go in the same directory. After
installing, use the project tool from any project:

```sh
maka new hello && cd hello
maka run                        # build + run src/main.maka
maka fmt                        # format src/*.maka
maka lint                       # check naming/style
```

The idiomatic Cargo alternative is `cargo install --path crates/driver` and
`cargo install --path crates/cli`.

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

Early but real. Positive suite passes 202 programs; negative suite passes 57.
The language is implemented in Rust as a workspace; the generated C builds
cleanly with modern gcc/clang (including gcc 14, which treats implicit
declarations and void value-returns as hard errors), and the whole test corpus
is LeakSanitizer-clean.  The stdlib has generic `Vec<T>` / `HashMap<V>` /
`Atomic<T>`, whole-file I/O, and a fiber/thread concurrency stack.

What's intentionally *not* in v1, but is on the road:

- Auto-borrow at method calls (today: `(&p).method()` when the method's
  receiver is `&_ self`, since dispatch matches receiver type exactly).
- `Attr.method(x)` qualified-call form.

## License

MIT.
