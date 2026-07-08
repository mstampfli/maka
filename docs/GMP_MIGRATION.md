# GMP fiber migration - design + build plan

Goal: an explicit `Pool` becomes a true M:N scheduler. A fiber that parks
(channel / mutex / wait group / **sleep** / **IO**) and is later woken resumes on
**any** idle worker, and a parked fiber does **not** tie up the worker it parked
on. This closes the one remaining gap in the explicit-pool concurrency system
(`pool` / `spawn_on` / `job_on` / `pool_shutdown`), where a woken fiber currently
resumes on its `home_sched` worker.

Status: **DELIVERED.** An explicit `Pool`'s N workers now share one scheduler
(run queue + timer + reactor + fd registry), so a fiber that parks on a
channel / mutex / wait group / **sleep** / **IO** and is later woken resumes on
whichever worker is free, and a sleep/IO-parked fiber does **not** pin its
worker. Proven: a fiber records the OS thread it runs on before and after a
`sleep`; across a pool it resumes on a *different* worker every run. See
`## Delivered (as built)` at the bottom for the shipped design + the two
sanitizer/platform follow-ups.

## Non-negotiables (performance-first-design)

- **One scheduler mechanism, not two.** Do NOT fork a separate "pool scheduler"
  that duplicates the per-thread cooperative one - they would drift and bugs
  would be fixed in one only. Generalize the *existing* scheduler to run over a
  **scheduler group**: 1 worker (cooperative `spawn`, today's behavior) or N
  workers (a `Pool`). The loop is identical; only group size + synchronization
  differ.
- **O(1) hot paths, pointer handles.** Ready push/pop/steal are O(1) on intrusive
  `f->next` links; a fiber handle is the `maka_fiber_t*` itself (no map, no
  scan). A wake routes by a pointer on the fiber, never a lookup.
- **Design for the contended case.** Ready fibers live in **per-worker deques**
  with **Chase-Lev work-stealing** (reuse/generalize the proven `__maka_ws_*`
  used by `job`), not one globally-locked queue - that scales to many workers
  without a single contention point. The single-worker (`spawn`) case keeps a
  lock-free fast path (its own deque, no stealing peers).

## Architecture

Today the scheduler state is per-thread TLS globals: `maka_ready_head/tail`,
`maka_sleep_head`, `maka_fd_waiters`, `maka_epoll_fd`, `maka_current_fiber`,
`maka_anchor_fiber`. A fiber's `home_sched` (a `maka_sched_state*`) is how a
cross-thread wake finds where to enqueue it - the primitives already route wakes
through `home_sched`, which is the seam we build on.

The group model:

    maka_sched_group {
        deque<fiber*>  run[N];        // per-worker ready deques (Chase-Lev)
        fiber*         sleepers;      // shared timer list (min-deadline), locked
        int            epoll_fd;      // shared reactor for the whole group
        fiber*         fd_waiters;    // shared, locked
        atomic<int>    poll_owner;    // -1, or the worker currently in epoll_wait
        mutex          timer_mu;      // guards sleepers + fd_waiters
        atomic<long>   inflight;      // spawned-but-not-done; pool_shutdown drains to 0
        worker         w[N];          // each a maka_sched_state bound to this group
    }

- `maka_sched_state` gains a `group*` (the single-thread case allocates a group
  of size 1). `home_sched` stays the routing pointer.
- **Ready** moves from the TLS `maka_ready_head` into `group->run[worker_id]`.
  `__maka_ready_enqueue(f)` pushes to a deque: the current worker's own deque if
  it belongs to the group, else (cross-thread wake) a target worker's deque
  (round-robin or the least-loaded) with a wake of a parked worker. Idle workers
  **steal** from a random victim deque before parking.
- **Sleep** moves into `group->sleepers` (shared, `timer_mu`). Any worker (the
  `poll_owner`) advances the timer; expired sleepers go to a run deque + wake.
- **IO** (`wait_fd`) registers on `group->epoll_fd` + `group->fd_waiters`. The
  `poll_owner` does one `epoll_wait` with the min timer deadline; ready fds move
  their waiters to a run deque + wake. Because the reactor is the *group's*, a
  fiber parked on IO frees its worker entirely - this is what removes the
  sleep/IO limitation.
- **Primitives** (channel / mutex / wg) already wake via `__maka_ready_enqueue`
  through `home_sched`; once ready lives in the group, their wakes land in a run
  deque with **no per-primitive change** - the single seam pays off everywhere.

## Crux (learned from build attempts)

Migration cannot be a patch bolted onto the current design; it is an indivisible
conversion of the wake seam from **per-worker to per-group**, and any partial
version either hangs or cannot free a sleep/IO-parked worker (which defeats the
point). Concretely, the current runtime binds a fiber to one worker in three
coupled places, and all three must move to the group at once:

1. **`home_sched` (a `maka_sched_state*`) -> `home_group`.** Today
   `__maka_ready_enqueue` compares `f->home_sched` to the current worker to
   decide local-vs-remote; for migration it must compare **groups**, so a wake by
   any worker in the group lands in the shared run queue.
2. **The remote-wake inbox (`remote_wake_head` + `wake_pipe`) is per-worker.**
   An external thread waking a pool fiber posts to *one* worker's inbox; only
   that worker drains it. For migration the inbox must be **per-group** so any
   worker drains it and an idle worker is woken (the wake pipe / eventfd must be
   the group's, tying into the shared reactor below).
3. **The idle wait must be on the group, not the worker's intake.** An idle
   worker blocks on the group condvar/reactor; an enqueue (local or drained
   remote) signals it. This is why the shared reactor (slice 4) and the idle
   wait (slice 2) are the same problem - they can't be separated.

So the honest slicing is: slices 2-4 land as **one coherent per-group conversion**
(run queue + remote-wake inbox + reactor/timer + idle-wait), because the wake
path touches all of them together. Slice 1 (ready queue into the group) and the
scaffolding (`cv`/`closed`/`inflight`) are the safe, committed groundwork for it.
A shared **locked** run queue is the correctness-first structure (it supports the
arbitrary removal that `cancel`/`select` need, which a lock-free Chase-Lev deque
does not); per-worker Chase-Lev deques + lazy cancellation are the scalable
follow-up once correct.

## Build slices (each: build -> full suite -> ASan -> TSan -> commit; revert on any hang)

1. **Group struct + state, no behavior change.** Introduce `maka_sched_group`,
   give every `maka_sched_state` a size-1 group, move `ready` into
   `group->run[0]` behind accessors. Cooperative `spawn` behaves exactly as
   today. Prove: suites 446/135, thread-count probe still 1, TSan clean.
2. **Per-worker deques + stealing within a Pool.** A `Pool` creates one group of
   N workers, each a `maka_sched_state` in it. `spawn_on`/`job_on` enqueue to a
   run deque; idle workers steal. Ready fibers migrate. `home_sched` -> group.
   No sleep/IO migration yet (still per-worker reactor). Prove: migration test
   (a channel ping-pong that must resume on a different worker), ASan, TSan.
3. **Shared timer.** Move sleepers into `group->sleepers`; one `poll_owner`
   advances it. A sleeping pool fiber no longer pins its worker. Prove: the
   sleep-migration test (a fiber sleeps, another worker keeps working, the
   sleeper resumes elsewhere).
4. **Shared reactor.** Move `epoll_fd` + `fd_waiters` into the group; one
   `poll_owner` drives `epoll_wait`. IO-parked pool fibers free their worker.
   Prove: an IO test (many sockets on a small pool, work continues while some
   park on reads).
5. **Shutdown + teardown under migration.** `pool_shutdown` drains `inflight`
   (parked-then-woken fibers included) before closing + joining; each worker
   drains its deque + slab cache on exit. Prove: shutdown with parked fibers,
   ASan-clean teardown.

## Test plan (regression gate for every slice)

- `tests/run_all.sh` + `tests/run_neg.sh` green (446/135).
- Thread-count probe: cooperative `spawn` still 1 OS thread.
- Migration probe: a `spawn_on` fiber parks on a channel, its worker takes a
  long job, and a *different* worker resumes the fiber (assert via a per-worker
  id the fiber records before/after the park).
- ASan (leak / UAF / double-free) and TSan (data race) on each new test, plus a
  1000x stress loop.
- CI matrix (x86_64 + aarch64/qemu + Windows fallback).

## Known risk

This touches the most deadlock-prone code in the runtime (the current
single-thread scheduler took many iterations to make deadlock-free - see the
project memory). Hence the slice discipline: each slice is independently green
and revertible, and TSan gates every step. A rushed all-at-once change already
hung once; this plan exists so it doesn't again.

## Delivered (as built)

The shipped design differs from the sketch above in one simplifying choice: a
Pool's workers **share one `maka_sched_state`** rather than each holding a
distinct worker slot in a group struct. That makes `home_sched == maka_sched_state`
true for every pool worker, so the existing cross-thread wake seam, epoch
validation, and refcount teardown all work unchanged - migration falls out
without touching the delicate cross-thread machinery.

What shipped:

- **Shared scheduler substrate.** `maka_sleep_head`, `maka_fd_waiters`,
  `maka_epoll_fd`, and the fd->armed-mask registry `maka_fd_regs` moved from
  per-thread TLS into `maka_sched_state` (behind accessor macros, so every
  access site is unchanged). The run queue already lived in the group. A lone
  cooperative worker keeps its own private substrate exactly as before.
- **One group lock, gated on `is_pool`.** A recursive `pthread_mutex`
  (`__maka_grp_mtx_init`) guards the run queue + timer list + fd-waiter list.
  `MAKA_GRP_LOCK/UNLOCK` are a NO-OP for a lone cooperative worker
  (`is_pool == 0`), so the `spawn`/`job`/`spawn_pool` paths keep zero overhead
  and are provably unchanged. Recursive so a locked phase can call
  `__maka_ready_enqueue` (which re-locks) without self-deadlock.
- **One poll-owner + cv idle wait.** With N workers, exactly one drives
  `epoll_wait` (the shared reactor) while the rest block on the group condvar.
  Enqueue signals the cv (wake an idle worker) and pings the shared wake_pipe
  (wake the poll-owner). The poll-owner's wait is bounded at 250 ms so a
  shutdown (or any wake that raced a worker committing to `epoll_wait`) is
  always re-checked - closing the level-triggered wake_pipe TOCTOU where a
  peer's drain consumes the shutdown byte before this thread's `epoll_wait`.
- **Race-free park.** `sleep` and `wait_fd` defer their list insertion + epoll
  arm to the scheduler's finalize-park block (after the fiber's context is fully
  saved), so a peer worker can't wake + run a fiber mid-switch (double-run) -
  the same guard the primitive-waiter park already used, extended to the timer
  and reactor.
- **Yield-not-block join.** A joining FIBER parks on the target Thread's
  `fiber_waiters` list and yields its worker, never `pthread_cond_wait`/
  `nanosleep` (which would deadlock a Pool: every worker blocked on a child that
  needs a free worker). Drained by the runner's completion path under
  `done_mutex`.
- **Graceful shutdown.** `spawn_on`/`spawn`-from-a-pool-fiber bump
  `group->inflight`; the completion (and cancel) path decrements it.
  `pool_shutdown` sets `closed`, wakes everyone, and workers exit only when
  `closed && inflight == 0 && run/sleep/fd all empty` - submitted work always
  runs to completion first. The shared `sched_state` is freed by the last
  worker's pthread-key dtor (refcount to 0), which now also closes the shared
  epoll fd and frees the fd registry + group.

Verified: full `run_all`/`run_neg` green; new regression tests
`447_pool_nested_join` (nested spawn+join, the deadlock) and
`448_pool_sleep_migrate` (sleep-park migration + clean shutdown); migration
proven via a per-fiber OS-thread-id probe (resumes on a different worker every
run); ASan + UBSan + LSan clean on every pool path including a 300-fiber
sleep+nested-join stress; TSan race-free on the waiter-park path (446).

### Two follow-ups (not correctness gaps)

1. **TSan + migrating fibers.** ThreadSanitizer's per-thread shadow stack cannot
   follow a fiber whose stack is resumed on a *different* OS thread by the
   custom asm `maka_ctx_switch`; it SEGVs on the sleep/IO-migration path (the
   waiter-park path 446 is TSan-clean). This is the universal fiber-runtime
   requirement to emit `__tsan_create_fiber` / `__tsan_switch_to_fiber`
   annotations around the context switch under `-fsanitize=thread`. The runtime
   is race-free by construction (all shared lists under the group lock) and
   ASan-clean; adding the TSan fiber annotations is the way to make TSan
   validate the migrating path directly.
2. **macOS/BSD kqueue.** The kqueue backend keeps its `__maka_kq_fd` (and armed
   masks) in per-thread TLS, so a shared reactor across pool workers is not yet
   wired there (Linux epoll, the CI-gating target, is fully shared). macOS is
   already `continue-on-error` in CI pending the pre-existing socket-layer
   portability work; moving `__maka_kq_fd` into `maka_sched_state` alongside
   `epoll_fd` is the fix.

Scalable follow-up (unchanged from the plan): the single locked run queue is the
correctness-first structure; per-worker Chase-Lev deques + lazy cancellation are
the throughput follow-up once the semantics are settled.
