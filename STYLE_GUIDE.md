# Maka style guide and best practices

The conventions the Maka compiler, standard library, and tests follow, and that
`makac lint` enforces. When in doubt, match the surrounding code; when the code
and this guide disagree, the guide wins and the code is a bug.

This guide is normative for naming and layout (the linter checks these) and
advisory for idioms (the "prefer" sections). SPEC.md is the authoritative
language reference; this document is about *how to write* Maka well, not *what
the language is*.

---

## 1. Naming

Case carries meaning in Maka: it tells you at a glance whether a name is a type
or a value. Keep that signal intact.

| Kind                        | Convention             | Examples                          |
|-----------------------------|------------------------|-----------------------------------|
| Type (`data`/`enum`)        | `PascalCase`           | `Point`, `JsonValue`, `HashMap`   |
| Trait (`attr`/`logic`)      | `PascalCase`           | `Show`, `Weigh`, `StringOps`      |
| Enum variant                | `PascalCase`           | `Some`, `None`, `Circle`, `JNull` |
| Function / method           | `snake_case`           | `string_from`, `push_str`, `len`  |
| Local variable / parameter  | `snake_case`           | `head`, `next_frame`, `total`     |
| Module-level constant        | `SCREAMING_SNAKE_CASE` | `MAX_DEPTH`, `RAY_BUDGET`         |
| Module name                 | `snake_case`           | `shapes`, `net`, `json`           |
| Type parameter              | single uppercase       | `T`, `U`, `V`, `A`, `B`           |

- **Primitives are lowercase** (`int`, `float`, `bool`, `char`, `unit`,
  `string`) and are keywords, not user types. Note `string` (the primitive
  `[N]char` value) versus `String` (the growable stdlib type) differ only by
  case; this is deliberate but easy to mistype, so lean on `String` for anything
  you build up and `&string` for read-only views.
- A short math-style single uppercase letter (`N`, `M`, `P`) is an accepted
  constant name where a longer one would only add noise (an array size, a loop
  bound). Anything longer than one letter must be `SCREAMING_SNAKE_CASE`.
- No leading-underscore names in normal code: `__name` is reserved for
  compiler-generated symbols (`__MakaTup`, `__destr`, `__env`). Use `_` alone
  only as the discard target (`_ = expr;`).
- Do not encode the type in the name (`p_ptr`, `str_name`); the type already
  says it. Name for the role (`owner`, `head`, `view`).

## 2. Layout and formatting

Run `maka fmt` (or `makac fmt`, or Format Document in the editor) to apply the
indentation and blank-line rules below automatically. It is a layout-only
formatter: it re-indents, strips trailing whitespace, and collapses blank runs,
but it never reflows your code and never touches comments or strings, so it is
safe to run on save. `maka fmt --check` reports unformatted files without
writing (use it in CI). Naming (section 1) is deliberately *not* auto-fixed by
the formatter, since a rename can be cross-file and semantic; `makac lint` flags
naming and the editor's Rename applies a fix you choose.

- **Four spaces** per indent level. Never tabs.
- **Opening brace on the same line** as the construct it belongs to, one space
  before it (K&R):

  ```maka
  int add(int a, int b) {
      return a + b;
  }

  if (x != null) {
      log(x!.v);
  } else {
      log(0);
  }
  ```

  Not on its own line (no Allman). This applies to functions, `if`/`else`,
  `while`, `for`, `match`, `data`/`enum`/`attr`/`has`/`logic` bodies, and
  `unsafe`/`rblock` blocks.
- One statement per line; one declaration per line.
- A trailing comment is two spaces off the code (`x = 1;  // why`). A
  full-line comment sits at the indentation of the code it describes.
- Keep lines reasonable (~100 columns). Break a long call after `(` and align
  arguments, or bind intermediates to named locals.
- `match` arms: `Pattern body,` one per line; a block body still ends with `,`.
- Plain ASCII only in source, comments, and identifiers. No smart quotes, no
  em/en dashes.

## 3. Types and ownership

Every binding states who owns the value. Pick the weakest pointer that does the
job; reach for `raw` only at an FFI boundary.

| Type       | Owns? | Nullable? | Use it for                                    |
|------------|-------|-----------|-----------------------------------------------|
| `T` value  | -     | -         | the default: pass and return by value         |
| `&T`/`&mut T` | no | no        | a scoped borrow (read, or in-place mutate)    |
| `own *T`   | yes   | yes       | an optional heap owner (auto-freed)           |
| `own &T`   | yes   | no        | a required heap owner, incl. a struct field   |
| `*T`       | no    | yes       | a nullable non-owning view                    |
| `raw *T`   | no    | yes       | the FFI / manual-memory escape hatch only     |

- **Default to values.** Maka arrays and structs are values; pass them plainly
  and let the compiler pass large ones by reference under the hood. Reach for a
  pointer only when you need sharing, nullability, or heap ownership.
- **Borrow to read or mutate in place**; own to keep. A function that only
  reads takes `&T`; one that mutates takes `&mut T`; one that stores the value
  takes it by value or as `own *T`/`own &T`.
- **`own &T` for a required owner, `own *T` for an optional one.** A struct
  field that must always hold a heap value is `own &T`; a slot that can be empty
  is `own *T`.
- **`raw *T` is a last resort.** It opts out of ownership and null tracking and
  every access needs `unsafe`. Confine it to FFI plumbing and the FFI-singleton
  pattern; never use it to dodge a borrow checker complaint in safe code.
- Prefer `Vec<T>` over the low-level `own &[*]T` for a growable array, and
  `String` over hand-rolled `own *char` for growable text.

## 4. Null safety

- **Guard, then deref.** A `*T`/`own *T`/`raw *T` deref (`p!`) needs a
  compile-time non-null proof:

  ```maka
  if (p != null) {
      log(p!.v);
  }
  ```

  or an early exit:

  ```maka
  if (p == null) { return; }
  log(p!.v);
  ```

- The guard narrows a **place**, not just a bare local, for OWNING places:
  `if (xs[0] != null) { xs[0]!.v }` and `if (s.p != null) { s.p!.v }` both work.
  A call, a write to the place or its container, or moving the element out drops
  the narrowing (rebind to a local across a call if you need it to survive one).
- Do not reach for `raw` to skip a null proof. If you cannot prove non-null,
  the value can be null, and the proof is the point.
- A builder that always allocates (`format`, `+` on strings, `str_dup`,
  `string_from`) returns a known-non-null owner and derefs without a guard;
  `read_file`/`read_line` return null on failure, so guard them.

## 5. Errors and optional values

- Use `Option<T>` for "maybe absent" and `Result<T, E>` for "succeeded or
  failed with a reason". Read them by `match` or by their `.tag`/payload
  fields; tag 0 is `Some`/`Ok`.
- Inside an `inline` helper, `propagate X` returns `X` from the *caller*, which
  is the idiom for early-out error plumbing. Keep `propagate` in small inline
  helpers, not deep call chains.
- Do not signal errors by returning a null `*T` from safe code where an
  `Option`/`Result` would say more; reserve null for genuinely optional
  pointers.

## 6. Data, enums, and traits

- Give a `data` field a default when there is an obvious one
  (`mut Order order = Order.Idle;`); it keeps struct literals short.
- Model a closed set of shapes as an `enum` and consume it with an exhaustive
  `match`; the compiler enforces exhaustiveness, so a new variant surfaces every
  site that must handle it.
- **Prefer `attr` + `has`** for traits: declare the contract in `attr`,
  implement per type in `has`, and call by method name (`x.method()`, receiver
  auto-borrowed). Reach for `logic` only when you need qualified dispatch
  `Trait.method(&x)` on a `dyn`/`some` value whose concrete type is hidden.
- Cross-module trait use needs both `pub has` on the impl and `use
  Mod.Type.Attr;` in the consumer. Export the impl deliberately, not by reflex.

## 7. Memory management

- Owning values free themselves at scope exit, recursively (owned fields, enum
  payloads, array/`Vec` elements). Write the ownership correctly and leaks take
  care of themselves; do not add manual frees for Maka-managed memory.
- To release an `own *T` early, assign `null` to it; there is no `free` for
  owned memory.
- For the FFI-singleton pattern (`mut raw *mut World` reset on reboot/load),
  reclaim the whole graph with `free deep p;` inside `unsafe` - it runs the same
  recursive drop scope-exit uses. A plain `free p;` only frees the top
  allocation and leaks the nested owners.
- Read a tuple/struct return by destructuring: `(q, r) = divmod(x);` rather than
  `t.f0`/`t.f1`.

## 8. FFI (C and Rust)

Pick the narrowest mechanism for the job:

- **Call out to Rust/C**: an `rblock pub fn` or an `extern` C declaration -
  value in, value out.
- **Let foreign code call back into Maka**: an `export fn` (a stable unmangled C
  symbol).
- **Hand a callback to a foreign API**: pass a Maka function directly to an
  `extern "C" fn(...)` parameter (bare fn or non-capturing closure); for a
  stateful closure use the `fn_code(f)` + `fn_env(f)` pair.
- **Depend on a crate**: `rdep name = "1";`, or a local one with
  `rdep name = "{ path = \"vendor/x\" }";` (relative to the invocation dir).
- **Link a C lib in source**: `clink "-lfoo";` or `clink "vendor/x.c";`.
- A `cblock`/`extern` definition must match Maka's EMITTED C type names exactly
  (`int` -> `maka_int`, `string` param -> `const char*`, `raw *mut unit` ->
  `maka_unit*`); see SPEC 5.3. Do not guess `void*`/`char*`.
- Keep `unsafe` blocks small and specific: one operation, with the invariant
  that makes it safe stated in a comment right there.

## 9. Concurrency

- Choose the tier by the work: `thread` for blocking or truly parallel work,
  `spawn` (fiber) for cooperative IO concurrency, `job` for parallel compute.
- Share state through typed primitives (`Mutex`, `Atomic<T>`, the `Chan`
  family), never a bare `mut` global written from a thread - the compiler
  rejects the latter.
- Every handle is joined, detached, or cancelled; an abandoned one auto-detaches
  at scope exit, but say what you mean.

## 10. Modules and files

- One module per file, declared at the top (`module shapes;`). File name
  matches the module (`shapes.maka`).
- Export the minimum: `pub` only what other modules genuinely need. An
  unexported helper gets internal linkage and inlines freely.
- `import a.b.Name;` for a single name, `import a.b.{x, y};` for a few; reserve
  wildcard `import a.b.*;` for a curated re-export hub.
- Group `import`/`use` near the top even though the language allows them
  interleaved.

## 11. Tests

- One behavior per test. `tests/programs/NN_short_name.maka` plus a matching
  `.expected` for a program that must compile, run, and match stdout;
  `neg_NN_short_name.maka` for one that must be rejected.
- A negative test asserts the compiler REJECTS a specific misuse; give it a
  comment saying what must be rejected and why.
- After a language or stdlib change, both `tests/run_all.sh` and
  `tests/run_neg.sh` must be green, and a memory/lifetime change should also be
  checked leak-clean (a sanitizer build over a 1000x loop for the definitive
  answer).

## 12. Anti-patterns

- Reaching for `raw *T` to silence a borrow or null error in safe code. Fix the
  ownership instead.
- Deep call chains that thread a nullable pointer without narrowing; guard once
  and bind a non-null local.
- Manual `free` of Maka-managed memory (double-free); let scope exit drop it.
- A `logic` trait where `attr`+`has` would do, forcing qualified dispatch on
  callers for no reason.
- Encoding type or ownership in a name (`ptr_`, `_owned`) instead of the type.
- Duplicated field access (`t.f0`, `t.f1`) where a destructuring bind reads
  clearer.
- Silent truncation or a swallowed error path; surface it or handle it.
