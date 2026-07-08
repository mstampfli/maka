# GMP fiber migration — design + build plan

Goal: an explicit `Pool` becomes a true M:N scheduler. A fiber that parks
(channel / mutex / wait group / **sleep** / **IO**) and is later woken resumes on
**any** idle worker, and a parked fiber does **not** tie up the worker it parked
on. This closes the one remaining gap in the explicit-pool concurrency system
(`pool` / `spawn_on` / `job_on` / `pool_shutdown`), where a woken fiber currently
resumes on its `home_sched` worker.

Status: **designed, not yet built.** The current pool distributes *new* fibers
across workers via a shared intake and pins them on pickup; `job` already
work-steals (Chase-Lev). This document is the plan for migrating *parked* fibers.

## Non-negotiables (performance-first-design)

- **One scheduler mechanism, not two.** Do NOT fork a separate "pool scheduler"
  that duplicates the per-thread cooperative one — they would drift and bugs
  would be fixed in one only. Generalize the *existing* scheduler to run over a
  **scheduler group**: 1 worker (cooperative `spawn`, today's behavior) or N
  workers (a `Pool`). The loop is identical; only group size + synchronization
  differ.
- **O(1) hot paths, pointer handles.** Ready push/pop/steal are O(1) on intrusive
  `f->next` links; a fiber handle is the `maka_fiber_t*` itself (no map, no
  scan). A wake routes by a pointer on the fiber, never a lookup.
- **Design for the contended case.** Ready fibers live in **per-worker deques**
  with **Chase-Lev work-stealing** (reuse/generalize the proven `__maka_ws_*`
  used by `job`), not one globally-locked queue — that scales to many workers
  without a single contention point. The single-worker (`spawn`) case keeps a
  lock-free fast path (its own deque, no stealing peers).

## Architecture

Today the scheduler state is per-thread TLS globals: `maka_ready_head/tail`,
`maka_sleep_head`, `maka_fd_waiters`, `maka_epoll_fd`, `maka_current_fiber`,
`maka_anchor_fiber`. A fiber's `home_sched` (a `maka_sched_state*`) is how a
cross-thread wake finds where to enqueue it — the primitives already route wakes
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
  fiber parked on IO frees its worker entirely — this is what removes the
  sleep/IO limitation.
- **Primitives** (channel / mutex / wg) already wake via `__maka_ready_enqueue`
  through `home_sched`; once ready lives in the group, their wakes land in a run
  deque with **no per-primitive change** — the single seam pays off everywhere.

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
single-thread scheduler took many iterations to make deadlock-free — see the
project memory). Hence the slice discipline: each slice is independently green
and revertible, and TSan gates every step. A rushed all-at-once change already
hung once; this plan exists so it doesn't again.
