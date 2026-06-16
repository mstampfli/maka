# Concurrency in Maka

Maka exposes **three concurrency tiers**, each picking a different
weight class.  All three follow the same surface shape:

    HandleType<T> h = <tier>(T() { body });   // spawn, get a handle
    T value = join(h);                          // wait, get value

Three builtin spawners (`thread`, `spawn`, `job`), one waiting operation
(`join`), plus a small set of composition helpers (`select`, `par_for`,
`par_reduce`, `par_map`).  No `async` keyword, no `await` keyword, no
`.await` postfix, no Future trait at the user level, no Pin, no
function coloring, no borrows-across-await rule.  Every concurrency
tier is a real function with a real stack — code looks like blocking
code, gets free concurrency.

`SPEC.md` §7 is the canonical short reference.  This document is the
deep dive.

---

## 1. Tier overview

| | `thread` | `spawn` (fiber) | `job` |
|-|---|---|---|
| Body runs on | a new OS thread | a userspace fiber | a worker-pool thread |
| Memory per | ~8 MB stack (kernel) | ~10 KB resident slab, VM-reserved 1 MB | ~32 B work item |
| Spawn cost | ~10–50 μs | ~100 ns (warm pool) | ~10 ns (warm pool) |
| Can pause? | yes, anywhere | yes, anywhere | no (run-to-completion) |
| Yields on IO | kernel blocks the thread | scheduler-yielded transparently | n/a (no IO) |
| Parallel? | yes (own kernel thread) | no (one scheduler thread) | yes (worker pool) |
| Cancel by drop? | flag-poll (janky) | clean (panic at next yield) | between-job boundary only |
| Use for | blocking C, isolated heavy work | concurrent IO with ergonomic code | parallel compute fanout |

The decision tree:

  - Need a blocking C library?  Use **`thread`**.
  - Need many concurrent IO operations and code that reads like
    blocking code?  Use **`spawn`**.
  - Need to fan a CPU-bound computation across cores?  Use **`job`**
    (or, more commonly, `par_for` / `par_reduce`).

---

## 2. Surface

### 2.1 Spawning

All three take a closure (a `T()` callable) and return a typed handle:

```maka
Thread<int> a = thread(int() { return cpu_heavy(); });
Fiber<int>  b = spawn(int()  { return read_file("path"); });
Job<int>    c = job(int()    { return crunch(matrix); });
```

The closure body is the work.  It runs to completion (returning `T`)
or panics.  At spawn time, the handle is returned synchronously
(before the body has finished).

### 2.2 Awaiting / joining

Get the handle's value by calling `join(h)` on it:

```maka
int va = join(a);       // pause this caller until thread a finishes; get its value
int vb = join(b);       // same shape for fibers
int vc = join(c);       // same shape for jobs
```

`join(h)` is the one waiting operation — no `.await` postfix, no
`await` keyword, no separate name per tier.

Dropping a handle without awaiting it **cancels the underlying work**
at the next safe point:

  - For fibers and jobs, the cancellation is clean: at the next yield
    point (fiber) or between-job boundary (job), a panic is injected,
    the body unwinds (running normal drops), and resources are freed.
  - For threads, the cancellation sets a flag the thread polls at
    well-known points (next IO call etc.).  Not all blocking C calls
    can be cancelled mid-flight — that's the OS's behavior, not ours.

### 2.3 Composition — homogeneous-only

`join` and `select` take a slice of same-typed handles.  No heterogeneous
overloads, no auto-generated `JoinN` / `SelectN` structs, no per-arity
field naming — `join` returns `[]T`, `select` returns `T`.

```maka
// Wait for all — returns []T in spawn order.
[]int counts = join([
    spawn(int() { return fetch_count(1); }),
    spawn(int() { return fetch_count(2); }),
    spawn(int() { return fetch_count(3); }),
]);
log(counts[0]);
log(counts[1]);
log(counts.len);

// Race — return the first finisher's value; cancel the rest.
int winner = select([
    spawn(int() { return read_mirror(1); }),
    spawn(int() { return read_mirror(2); }),
    spawn(int() { return read_mirror(3); }),
]);
```

Signatures:

```maka
pub []T join<T>(&[]Handle<T> handles);
pub T   select<T>(&[]Handle<T> handles);
```

**Heterogeneous case — user wraps the returns in a common type.**  This is
the answer to "but what if my two parallel operations return different
types?":

```maka
// "Race a read against a timeout" — both branches return the same enum.
enum Event { Got { int value }, Timeout }

Event r = select([
    spawn(Event() { return Event.Got { value = read_int(&conn) }; }),
    spawn(Event() { sleep(Duration.secs(5)); return Event.Timeout; }),
]);
match (r) {
    Got{value} log(value),
    Timeout    log("timed out"),
};
```

**Spawn-and-await each separately** is the other pattern.  For "fetch
profile AND settings in parallel" you don't need a join primitive — just
spawn both and await each:

```maka
Fiber<Profile>   hp = spawn(Profile()   { return fetch_profile(); });
Fiber<[]Setting> hs = spawn([]Setting() { return fetch_settings(); });

Profile      p = join(hp);       // hp finishes; hs runs concurrently
[]Setting    s = join(hs);       // collect hs's result (probably already done)
// Total time: max(profile_time, settings_time), not the sum.
```

You get two named, typed variables — no wrapper struct, no `.v0` / `.v1`
positional access, no anonymous bundling.  This is cleaner than any
heterogeneous join primitive could be.

### 2.4 Data-parallel helpers (jobs under the hood)

```maka
pub unit par_for<T>(&[]T items, unit(&T) f);
pub U    par_reduce<T, U>(&[]T items, U init, U(U, &T) combine);
pub []U  par_map<T, U>(&[]T items, U(&T) f);
```

These chunk the slice into N chunks (one per CPU core), spawn a job
per chunk, each job processes its chunk linearly.  Per-item overhead
is the function-call cost; per-chunk overhead is one job spawn (~10
ns) amortised over millions of items.

`par_reduce`'s `combine` must be associative (the runtime combines
partial results in any order).

---

## 3. Tier 1 — `thread`

### 3.1 Semantics

```maka
Thread<T> thread<T>(T() body);
```

Calls `pthread_create` (POSIX) or `CreateThread` (Windows) immediately.
The new thread runs `body()` on a fresh ~8 MB kernel stack, in parallel
with whatever else is happening.  The kernel scheduler manages it like
any other process thread.

Threads are **truly parallel** (no shared scheduler), **truly
preemptive** (the kernel preempts the thread on a regular interval),
and **blocking-safe** (a blocking syscall blocks only this thread, not
the rest of the program).

### 3.2 When to use

  - Blocking C libraries that can't yield to a scheduler (libcurl,
    libsqlite, libpq, libffi-based wrappers).
  - Long-running CPU-bound work that should run independently of the
    fiber scheduler.
  - Anything that calls into code you don't control and can't audit
    for yielding behavior.

### 3.3 When NOT to use

  - For lots of concurrent IO operations.  8 MB × 10k = 80 GB; you'll
    OOM.  Use `spawn` (fiber).
  - For tiny compute fanout.  Spawn cost dwarfs the work.  Use `job`.

### 3.4 Implementation

The MVP backs `thread` with `pthread_create`/`pthread_join`.  Same
infrastructure Maka has had since v1.  Future: rename internally to
distinguish from the fiber path; no semantic change.

---

## 4. Tier 2 — `spawn` (fiber)

### 4.1 Semantics

```maka
Fiber<T> spawn<T>(T() body);
```

The runtime acquires a fiber slab (a stable memory region holding the
fiber's stack), initialises the fiber's saved-register state to enter
`body`, and queues the fiber for the scheduler.  `spawn` returns a
handle.

The fiber runs on the **scheduler thread**.  By default there is one
scheduler thread per Maka process.  When the fiber executes a
**yielding IO operation** (`read`, `write`, `sleep`, `accept`, etc.),
the runtime saves the fiber's CPU state into its slab, marks the
fiber as "waiting on fd X" with the reactor, and switches to the next
ready fiber.

When the reactor sees the awaited fd become ready (via `epoll_wait`,
`kqueue`, or `IOCP`), it marks the fiber as runnable and the
scheduler eventually resumes it.

### 4.2 Why borrows-across-yield work

The fiber's slab does NOT move while the fiber is suspended.  The
slab is at a stable virtual address, allocated from a pool.  When the
fiber's `read()` yields, its registers and stack pointer are saved
INTO the slab, the slab itself stays at its same address.  When the
fiber resumes, the stack pointer is restored, and locals on the
fiber's stack are right where they were.

No state-machine struct.  No moves.  No self-references.  No Pin.

A `&local` borrow on the fiber's stack is just a pointer into the
slab's stack region — stays valid while the slab does, which is the
whole fiber lifetime.

### 4.3 Memory model

  - Each fiber gets **1 MB of virtual address space** reserved up
    front (`mmap` with `PROT_NONE` guard page at the bottom, then a
    growable committed region).
  - **Resident memory** is just the pages the fiber actually touches.
    A typical IO handler uses 1–4 KB of stack at peak.  10k fibers ≈
    ~100 MB resident.
  - Slabs are **pooled**: when a fiber finishes, its slab is returned
    to the free list, not `munmap`'d.  Next `spawn` pulls from the
    pool — no `mmap` cost on the warm path.

The 1 MB VM reservation is essentially free on 64-bit systems (128 TB
address space).  Resident memory is bounded by the actual peak working
set across all live fibers.

### 4.4 Scheduler

Single-threaded by default.  One OS thread (the "runtime thread")
hosts the scheduler.  All fibers run on this thread, cooperatively
yielding at IO suspension points.  Mutations within a fiber don't
need synchronisation because no two fibers run simultaneously on the
same thread.

Work-stealing across N OS threads (one per CPU core) is a future
addition.  For Maka's targets (games + IO-bound servers + tools), the
single-threaded model is typically what you want — pin the scheduler
to one core, push CPU work onto `thread` or `job` for parallelism.

### 4.5 Reactor

Wraps `epoll_create1`/`epoll_ctl`/`epoll_wait` (Linux), `kqueue`
(BSD/macOS), and `IOCP` (Windows).  Per-fiber state in the reactor
maps fd → fiber handle.  On readiness event, the corresponding fiber
is moved to the scheduler's ready queue.

### 4.6 IO surface

Standard IO functions are blocking from the fiber's perspective:

```maka
pub int read(&mut TcpStream s, &mut []u8 buf);
pub int write(&mut TcpStream s, &[]u8 buf);
pub unit sleep(Duration d);
pub TcpStream accept(&TcpListener l);
```

Each internally checks "would I block?" — if yes, registers with the
reactor and yields.  Code looks like ordinary blocking code; the
fiber yield is invisible.

### 4.7 Cancellation

Dropping a `Fiber<T>` handle:
  1. Marks the fiber as "cancelled" in its task control structure.
  2. At the fiber's next yield (i.e., next IO call), the runtime
     injects a panic instead of waiting.
  3. The panic unwinds the fiber's stack, running every drop on
     locals as it goes.
  4. When unwinding reaches the fiber's entry frame, the slab returns
     to the pool.

Clean, deterministic, follows Maka's existing drop semantics.

### 4.8 Implementation status

Real fiber implementation requires per-architecture context-switch
assembly (~30 LOC each for x86_64, aarch64, riscv64), a scheduler
(~200 LOC of Maka), a reactor (~400 LOC), and a slab pool (~100
LOC).  Total ~700 LOC of Maka + ~90 LOC of asm.

**For now, `spawn` is backed by `pthread_create`/`pthread_join`** —
the same backing as `thread`.  This is incorrect for the high-
concurrency case (no IO yielding, no slab pool) but lets the surface
ship and lets user code be written today.  Future work replaces the
backing without changing the surface.

---

## 5. Tier 3 — `job`

### 5.1 Semantics

```maka
Job<T> job<T>(T() body);
```

Pushes a work-item onto a **work-stealing queue**.  The work-item is
a closure value (function pointer + captured environment), typically
~32 bytes.  Some worker thread eventually picks it up and runs `body()`
to completion.  Returns a handle the caller can await.

Jobs are **run-to-completion**.  They cannot pause.  They cannot do
IO that would block.  They are pure compute.  This is what makes
them so cheap — no stack of their own, no scheduler context for them,
no fiber slab.

### 5.2 Worker pool

At runtime start, the job pool spawns N worker threads — typically
`num_cpus()`.  Each worker owns a **lock-free deque** (Cilk-style):

  - Workers push to their own deque from the top.
  - Workers pop from their own deque from the top (LIFO; cache-warm).
  - Idle workers steal from another worker's deque from the bottom
    (FIFO; load-balancing).

This is the classical work-stealing pattern.  Throughput scales
linearly with cores; load balances automatically across imbalanced
workloads.

### 5.3 Sync to the calling fiber

A `Job<T>` handle's `join` either:
  - returns immediately if the job has already completed,
  - or parks the calling fiber on the job's completion latch.

When the job finishes, its worker signals the latch, unparking the
caller.  Sync overhead: ~one atomic CAS + one fiber wakeup.

### 5.4 par_for / par_reduce / par_map

```maka
pub unit par_for<T>(&[]T items, unit(&T) f);
pub U    par_reduce<T, U>(&[]T items, U init, U(U, &T) combine);
pub []U  par_map<T, U>(&[]T items, U(&T) f);
```

These chunk the slice.  Default chunk size auto-tunes to maximise
work-stealing balance — start with `len / (cores * 4)` and let
workers split further when idle.

For a billion-element slice on 16 cores, that's ~64 chunks initially,
spawning ~64 jobs.  Each job processes ~16 million items linearly.
Per-item overhead is the function call.  Per-chunk overhead is one
job spawn (~10 ns) amortised across millions of items — invisible.

### 5.5 Implementation status

For MVP, `job` is also backed by `pthread_create` (one thread per
job).  This is wildly suboptimal (defeats the whole point — 10 μs per
job).  Real implementation needs the work-stealing pool, which is
another ~300 LOC.  Surface is shippable today, performance gets the
pool later.

---

## 6. Handles and `await` / `join`

### 6.1 Handle types

```maka
pub data Thread<T>;    // runtime-defined opaque
pub data Fiber<T>;     // runtime-defined opaque
pub data Job<T>;       // runtime-defined opaque
```

All three are **builtin parametric types** registered by the sema
crate (same as `Thread` already is in current Maka).  The runtime
internally distinguishes them.

All three implement an attribute `attr Awaitable<T>` — purely an
internal sema concept; the user never spells `.await` anywhere:

```maka
pub attr Awaitable<T> {
    T __wait(&mut _ self);     // internal; users call `join(h)` instead
}
```

Each handle type implements `__wait` differently:
  - `Thread<T>::__wait` calls `pthread_join`.
  - `Fiber<T>::__wait` parks the calling fiber on the target fiber's
    completion latch, returns the stored result.
  - `Job<T>::__wait` parks the calling fiber on the job's completion
    latch.

### 6.2 `join(h)` — single-handle wait

```maka
pub inline T join<T, H>(H h) where H has Awaitable<T>;
```

Dispatches to the handle's `__wait`.  Single way to wait — no
`.await` postfix anywhere in the user surface.

### 6.3 `join(&[]Handle<T>)` — homogeneous wait-all

```maka
pub []T join<T, H>(&[]H handles) where H has Awaitable<T>;
```

Takes a slice of same-typed handles.  Returns `[]T` with results in
spawn order.  Implementation: await each handle in sequence; total
wait is `max(t_i)` because the children run concurrently.

Heterogeneous case: not supported.  Wrap the differing return types in
a common `enum` (one variant per branch) so all spawn bodies return
that enum; the slice is then homogeneous.  Or — more often the right
choice — spawn each separately and await each into its own typed
variable (no slice, just N independent named values).

### 6.4 `select(&[]Handle<T>)` — homogeneous race / first-wins

```maka
pub T select<T, H>(&[]H handles) where H has Awaitable<T>;
```

Takes a slice of same-typed handles.  Returns the first finisher's
value; all losing handles are dropped (= cancelled).  Same wrap-in-
enum pattern handles the heterogeneous use case explicitly.

Implementation: register a completion notifier on every handle
pointing to the calling fiber.  On first wake, the caller retrieves
the winner's value, drops the rest.

---

## 7. Cross-tier composition

Tiers compose freely.  A fiber can spawn threads, jobs, or other
fibers.  A thread can spawn fibers (the fibers run on a scheduler
that lives on that thread).  A job cannot pause, so it cannot await
anything — but it can FIRE-AND-FORGET other tiers without waiting.

Typical patterns:

```maka
// IO server with parallel request processing.
unit server() {
    TcpListener l = listen("0.0.0.0:8080");
    while (true) {
        TcpStream conn = accept(&l);
        spawn() { handle_request(conn); };   // fiber per connection
    }
}

unit handle_request(TcpStream conn) {
    [4096]u8 buf;
    int n = read(&mut conn, &mut buf);       // fiber yields
    HttpReq req = parse(&buf, n);

    // CPU-bound transformation on a worker thread — let the fiber yield
    // until the thread completes.
    Thread<Response> bg = thread(Response() { return compute(req); });
    Response resp = join(bg);

    // Concurrent IO: fetch from upstream while preparing response.
    // Different return types → spawn each, join each into typed var.
    Fiber<Bytes> hf = spawn(Bytes() { return fetch_upstream(req); });
    Fiber<unit>  hp = spawn(unit()  { return prepare_response(); });
    Bytes body = join(hf);
    join(hp);

    write(&mut conn, body);
}
```

```maka
// Frame-of-work game-engine pattern.
unit frame() {
    // CPU fanout across all entities.
    par_for(&entities, unit(&Entity e) { update(e); });
    par_for(&particles, unit(&Particle p) { advance(p); });

    // Heavy compute on its own thread so it doesn't starve the frame.
    Thread<Image> renderer = thread(Image() { return render_scene(); });

    // Continue frame work while renderer runs.
    update_ui();

    // Pick up the rendered image.
    Image img = join(renderer);
    present(img);
}
```

---

## 8. What does NOT exist

- ❌ `async` keyword.
- ❌ `await` as a separate language keyword (it's a method on `Awaitable<T>`).
- ❌ State-machine codegen pass.
- ❌ Pin, Unpin, `pin!`, `Box::pin`.
- ❌ Function coloring.
- ❌ "Borrows can't cross await" — borrows work everywhere; the
  fiber's stack is stable.
- ❌ Tuples.  No `JoinN` / `SelectN` heterogeneous structs either —
  homogeneous-only `join` / `select` returning `[]T` / `T`; the
  heterogeneous case lives in user-written enums or separate
  spawn-and-await pairs.
- ❌ `?` for error propagation in async (use `.must()` from `std.err`
  same as everywhere else).
- ❌ `CancellationToken` — drop the handle.
- ❌ A future-vs-handle distinction.  Everything that can be awaited
  is a handle.

---

## 9. Status

| Area | Status |
|---|---|
| Surface (`thread`, `spawn`, `job` keywords + handle types) | **Implemented** |
| `join(h)` for single handle | **Implemented** |
| `join(&[]*Thread)` — homogeneous wait-all over a slice | **Implemented** |
| `select(&[]*Thread)` — homogeneous race; losers cancelled | **Implemented** |
| `par_for_range(start, end, body)` — chunk integer range across job pool | **Implemented** |
| `par_reduce_int(start, end, init, combine)` — parallel fold | **Implemented** |
| `sleep_ms` / `sleep_us` runtime calls | **Implemented** |
| `yield_now()` — cooperative yield from a fiber | **Implemented** |
| `thread()` — pthread with default ~8 MB stack | **Implemented** |
| `spawn()` — real cooperative fiber (ucontext + scheduler) | **Implemented** |
| `job()` — N-worker pthread pool with Chase-Lev work-stealing deques | **Implemented** |
| Cooperative scheduler with ready queue + sleep wheel | **Implemented** |
| Fiber-aware `sleep_ms`: yields instead of blocking when in a fiber | **Implemented** |
| Fiber-aware `join`: drives scheduler so other fibers progress | **Implemented** |
| Fiber-aware `select`: cancels losers by walking ready/sleep queues | **Implemented** |
| Epoll reactor + `wait_fd` / `read_async` / `write_async` | **Implemented** (Linux; kqueue/IOCP ports pending) |
| Work-stealing per-worker deques for the job pool | **Implemented** — Chase-Lev lock-free deques per worker |
| Slab pool with VM-reserve + page-on-touch for fibers | **Implemented** — `mmap PROT_NONE 1 MB` + `mprotect 64 KB` top page |
| `par_map_int` and reductions over typed slices | **Implemented** — `par_map_int` returns a fresh `[]int`; `par_reduce_int` folds |
| `par_for_each(slice, body)` — chunked iteration over a `[]int` | **Implemented** |
| `par_filter_int(slice, pred)` — parallel keep-where, returns filtered `[]int` | **Implemented** |
| `par_scan_int(slice, combine)` — two-pass associative prefix scan | **Implemented** |
| `par_map_int(slice, fn)` / `par_reduce_int(slice, init, combine)` — slice overloads | **Implemented** |
| `cancel(*Thread)` — user-callable cancellation across all three tiers | **Implemented** |
| `try_join(*Thread) -> bool` — non-blocking poll | **Implemented** |
| `join_timeout(*Thread, ms) -> bool` — bounded join | **Implemented** |
| `select_timeout(slice, ms) -> int` — race with deadline | **Implemented** |
| `wait_fd_timeout(fd, events, ms) -> bool` — bounded IO wait inside a fiber | **Implemented** |
| `detach(*Thread)` — opts out of join; runtime auto-reaps the handle | **Implemented** |
| Multi-fiber `wait_fd` on the same fd — per-fd registration table | **Implemented** |
| `EPOLLERR` / `EPOLLHUP` wake the waiter regardless of registered interest | **Implemented** |
| Fiber-aware `Mutex`/`WaitGroup`/`Once` — park via swapcontext, never block the worker | **Implemented** |
| `Atomic<i64>` — load/store/add/cas wrapper over `_Atomic int64_t` | **Implemented** |
| TCP helpers: `tcp_listen`, `tcp_accept_async`, `tcp_connect_v4`, `close_fd` | **Implemented** (Linux) |
| Closure env auto-freed on fiber/thread/job completion (no leak per spawn) | **Implemented** |
| kqueue backend (macOS/BSD) | Pending |
| IOCP backend (Windows) | Pending |
| `poll()` fallback reactor for non-epoll kernels | Pending |
| Cross-thread fiber migration (load-balance fibers across worker schedulers) | Pending — job pool covers parallelism today |
| Generic `par_map<T, U>` / `par_reduce<T>` for non-int element types | Pending |

The **surface is final** and user code written against it today will
continue to work as the runtime improves.  Performance will improve
substantially when the real fiber and job runtimes land; correctness
of the surface won't change.

---

End of spec.
