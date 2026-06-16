//! Maka → C codegen. Produces a portable C99/C11 translation of an HIR module.
//!
//! Heap T values are represented as `T*` allocated with `malloc` (single owner);
//! freed at scope exit unless moved. Pointer unwrap (`p!`) emits a runtime null check
//! when not narrowed.

use maka_sema::*;

pub fn emit(m: &HirModule) -> String {
    let mut cx = Cx::new(&m.sym, &m.cincludes, &m.cblocks);
    cx.emit_module();
    cx.out
}

struct Cx<'a> {
    sym: &'a SymTab,
    out: String,
    indent: usize,
    /// User C interop directives, forwarded verbatim by sema.
    module_cincludes: &'a [String],
    module_cblocks: &'a [String],
    /// Map of (slice/vec elem type-key) → emitted typedef name.
    slice_types: std::collections::BTreeSet<String>,
    vec_types: std::collections::BTreeSet<String>,
    /// Dyn instantiations seen: (Trait, StructId) — generate vtables for these.
    dyn_insts: std::collections::BTreeSet<(String, u32)>,
    /// Traits seen — generate Dyn_Trait struct typedef for these.
    dyn_traits: std::collections::BTreeSet<String>,
    /// Fat-callable signatures seen — emit `Callable_KEY` typedef + raw-fn-ptr typedef.
    callable_sigs: std::collections::BTreeMap<String, (HType, Vec<HType>)>,
    /// Functions whose names appear as values — emit trampolines.
    fn_trampolines: std::collections::BTreeSet<u32>,
    /// Lifted-lambda function ids that need closure trampolines (env cast + call).
    closure_trampolines: std::collections::BTreeSet<u32>,
    /// Per-emission counter for inline expansions: each statement-expression gets
    /// a unique tag so labels and locals never collide across multiple call sites
    /// of the same inline within the same C function.
    inline_call_seq: u32,
}

impl<'a> Cx<'a> {
    fn new(sym: &'a SymTab, cincludes: &'a [String], cblocks: &'a [String]) -> Self {
        Self {
            sym, out: String::new(), indent: 0,
            module_cincludes: cincludes,
            module_cblocks: cblocks,
            slice_types: Default::default(), vec_types: Default::default(),
            dyn_insts: Default::default(), dyn_traits: Default::default(),
            callable_sigs: Default::default(),
            fn_trampolines: Default::default(),
            closure_trampolines: Default::default(),
            inline_call_seq: 0,
        }
    }

    fn w(&mut self, s: &str) { self.out.push_str(s); }
    fn wl(&mut self, s: &str) {
        for _ in 0..self.indent { self.out.push_str("    "); }
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn open(&mut self) { self.indent += 1; }
    fn close(&mut self) { if self.indent > 0 { self.indent -= 1; } }

    fn emit_module(&mut self) {
        self.emit_prologue();
        self.emit_structs_and_enums();

        // First pass: scan for slice/vec types used in signatures and function bodies so we can typedef them.
        let funcs = self.sym.funcs.clone();
        for f in &funcs {
            self.scan_func(f);
        }
        self.emit_slice_typedefs();
        self.emit_vec_typedefs();
        self.emit_dyn_typedefs();
        self.emit_callable_typedefs();

        // Raw user C from `cblock "...";` directives — pasted verbatim at module scope.
        let blocks: Vec<String> = self.module_cblocks.to_vec();
        for b in &blocks {
            self.w(b);
            self.w("\n");
        }

        // Extern signatures (FFI)
        for s in &self.sym.sigs.clone() {
            if !s.is_extern { continue; }
            let ret = self.c_ret_type(&s.ret);
            let mut params = String::new();
            if s.param_tys.is_empty() && !s.is_variadic { params.push_str("void"); }
            else {
                let parts: Vec<String> = s.param_tys.iter().enumerate().map(|(i, t)| {
                    format!("{} {}", self.c_type(t), s.param_names.get(i).cloned().unwrap_or_else(|| format!("p{}", i)))
                }).collect();
                params.push_str(&parts.join(", "));
                if s.is_variadic {
                    if !s.param_tys.is_empty() { params.push_str(", "); }
                    params.push_str("...");
                }
            }
            self.wl(&format!("extern {} {}({});", ret, s.c_name, params));
        }

        // Forward decls (non-extern)
        for f in &funcs {
            if self.sym.func_sig(f.id).is_inline { continue; }
            let sig = self.func_signature(f);
            self.wl(&format!("{};", sig));
        }
        self.wl("");

        // Vtable instances (need function forward decls to be visible).
        self.emit_dyn_vtable_instances();
        // Trampolines for functions whose names are used as fat-callable values.
        self.emit_trampolines(&funcs);
        // Closure trampolines for capturing lambdas.
        self.emit_closure_trampolines();

        // Module-scope globals.  Emit each as a C static so the symbol is
        // file-local (matches Maka's "private unless pub" rule at the link
        // layer too).  Init expressions are emitted verbatim; the C compiler
        // enforces "must be a constant expression."
        for g in &self.sym.globals.clone() {
            let cty = self.c_type(&g.ty);
            // Use a dummy HFunc for emit_expr since globals have no locals.
            let init_str = self.emit_global_init(&g.init);
            self.wl(&format!("static {} {} = {};", cty, g.c_name, init_str));
        }

        // Function bodies
        for f in &funcs {
            self.emit_func(f);
        }
        // TCP socket helper bodies — emitted last so that the sys/socket.h
        // include is positioned AFTER all user code.  Otherwise socket.h's
        // pollution (e.g. the bare `accept` symbol) clobbers user functions
        // sharing those names.
        self.emit_socket_helpers();

        // Synthesize a C `int main` that calls Maka `main`.  Two surface shapes:
        //
        //   unit main()                  → ignore argc/argv
        //   unit main([]string args)     → build a `Slice_str` from (argc, argv)
        //
        // The slice form receives a borrowed view of argv; Maka code may not free
        // it.  argv[0] is the program name, matching every other language.
        let user_main = funcs.iter().find(|f| f.name == "main");
        match user_main {
            Some(f) => {
                let sig = self.sym.func_sig(f.id);
                let takes_args = sig.param_tys.len() == 1
                    && matches!(&sig.param_tys[0], HType::Slice { elem, .. } if matches!(elem.as_ref(), HType::Str));
                if takes_args {
                    let call_returning_int = matches!(f.ret, HType::Int);
                    if call_returning_int {
                        self.wl("int main(int argc, char** argv) { Slice_str args = { .ptr = (const char**)argv, .len = (maka_int)argc }; return (int)maka_main(args); }");
                    } else {
                        self.wl("int main(int argc, char** argv) { Slice_str args = { .ptr = (const char**)argv, .len = (maka_int)argc }; maka_main(args); return 0; }");
                    }
                } else {
                    match f.ret {
                        HType::Int => self.wl("int main(int argc, char** argv) { (void)argc; (void)argv; return (int)maka_main(); }"),
                        _ => self.wl("int main(int argc, char** argv) { (void)argc; (void)argv; maka_main(); return 0; }"),
                    }
                }
            }
            None => self.wl("int main(void) { return 0; }"),
        }
    }

    fn emit_prologue(&mut self) {
        self.w("// generated by makac\n");
        self.w("#define _XOPEN_SOURCE 600\n");
        self.w("#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <stdbool.h>\n#include <string.h>\n#include <wchar.h>\n");
        // User-requested system headers from `cinclude "name.h";` directives.
        let extras: Vec<String> = self.module_cincludes.to_vec();
        for h in &extras { self.w(&format!("#include <{}>\n", h)); }
        self.w("typedef int64_t maka_int;\ntypedef double maka_float;\ntypedef uint8_t maka_char;\n");
        self.w("typedef struct { int dummy; } maka_unit;\n");
        self.w("static maka_unit MAKA_UNIT = {0};\n");
        self.w("static void maka_panic(const char* msg) { fprintf(stderr, \"panic: %s\\n\", msg); abort(); }\n");
        // No MAKA_UNWRAP: the sema pass proves every `*T` deref is non-null,
        // so codegen emits a raw `(*p)` and never needs a runtime null check.
        self.w("static inline maka_int maka_check_idx(maka_int i, maka_int len, const char* msg){ if(i<0||i>=len) maka_panic(msg); return i; }\n");
        // log helpers
        self.w("static void maka_log_int(maka_int v) { printf(\"%lld\\n\", (long long)v); }\n");
        self.w("static void maka_log_float(maka_float v) { printf(\"%g\\n\", v); }\n");
        self.w("static void maka_log_bool(bool v) { puts(v?\"true\":\"false\"); }\n");
        self.w("static void maka_log_char(maka_char v) { putchar((int)v); putchar('\\n'); }\n");
        self.w("static void maka_log_str(const char* v) { puts(v); }\n");
        self.w("static void maka_log_ptr(const void* v) { if (v) printf(\"<ptr %p>\\n\", v); else puts(\"null\"); }\n");
        // String concat (`a + b` on two `string`s) — returns a malloc'd NUL-terminated
        // buffer.  Owned by the caller; auto-freed at scope-exit because the result
        // type is `own *char`.  Variants `_freel` / `_freer` / `_freeb` accept owning
        // intermediates and free them after the copy, so chained `a + b + c` doesn't
        // leak the inner result.
        self.w("static char* __maka_str_concat(const char* a, const char* b) { size_t la=a?strlen(a):0, lb=b?strlen(b):0; char* r=(char*)malloc(la+lb+1); if(a)memcpy(r,a,la); if(b)memcpy(r+la,b,lb); r[la+lb]=0; return r; }\n");
        self.w("static char* __maka_str_concat_freel(char* a, const char* b) { char* r = __maka_str_concat(a, b); free(a); return r; }\n");
        self.w("static char* __maka_str_concat_freer(const char* a, char* b) { char* r = __maka_str_concat(a, b); free(b); return r; }\n");
        self.w("static char* __maka_str_concat_freeb(char* a, char* b) { char* r = __maka_str_concat(a, b); free(a); free(b); return r; }\n");
        // `format(fmt, ...)` placeholder converters.  Each returns a malloc'd
        // string (caller owns) except `bool_to_str` which returns a static
        // literal pointer (no free needed).  The concat helpers above handle
        // the cleanup of the intermediate `_to_str` results when the result
        // is chained through `+`.
        self.w("static char* __maka_int_to_str(maka_int n)   { char buf[32]; snprintf(buf, sizeof(buf), \"%lld\", (long long)n); size_t L=strlen(buf); char* r=(char*)malloc(L+1); memcpy(r,buf,L+1); return r; }\n");
        self.w("static const char* __maka_bool_to_str(bool b) { return b ? \"true\" : \"false\"; }\n");
        self.w("static char* __maka_float_to_str(maka_float v){ char buf[40]; snprintf(buf, sizeof(buf), \"%g\", v);             size_t L=strlen(buf); char* r=(char*)malloc(L+1); memcpy(r,buf,L+1); return r; }\n");
        self.w("static char* __maka_char_to_str(maka_char c)  { char* r=(char*)malloc(2); r[0]=(char)c; r[1]=0; return r; }\n");
        // stdin readers: `read_line()` returns one heap line without the trailing
        // newline (NULL on EOF); `read_int()` reads a base-10 integer (panics on
        // malformed input).
        self.w("static char* __maka_read_line(void) { char* buf=NULL; size_t cap=0; ssize_t n=getline(&buf,&cap,stdin); if(n<0){ free(buf); return NULL; } if(n>0 && buf[n-1]=='\\n') buf[n-1]=0; return buf; }\n");
        self.w("static maka_int __maka_read_int(void) { long long v=0; if(scanf(\"%lld\", &v)!=1) maka_panic(\"read_int: malformed input\"); return (maka_int)v; }\n");
        // D11 concurrency primitives (defined here so the linker can resolve `extern` refs from Maka).
        self.w("maka_int maka_atomic_load_i64(maka_int* p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }\n");
        self.w("void maka_atomic_store_i64(maka_int* p, maka_int v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }\n");
        self.w("maka_int maka_atomic_fetch_add_i64(maka_int* p, maka_int d) { return __atomic_fetch_add(p, d, __ATOMIC_SEQ_CST); }\n");
        self.w("maka_int maka_atomic_fetch_sub_i64(maka_int* p, maka_int d) { return __atomic_fetch_sub(p, d, __ATOMIC_SEQ_CST); }\n");
        self.w("void maka_fence(maka_int ord) { (void)ord; __atomic_thread_fence(__ATOMIC_SEQ_CST); }\n");
        // Mutex via pthread: opaque pointer-sized handle, exposed via maka_unit* to match Maka's `*unit`.
        self.w("#include <pthread.h>\n");
        self.w("#include <stdatomic.h>\n");
        self.w("maka_unit* maka_mutex_new(void) { pthread_mutex_t* m = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t)); pthread_mutex_init(m, NULL); return (maka_unit*)m; }\n");
        self.w("void maka_mutex_lock(maka_unit* m) { pthread_mutex_lock((pthread_mutex_t*)m); }\n");
        self.w("void maka_mutex_unlock(maka_unit* m) { pthread_mutex_unlock((pthread_mutex_t*)m); }\n");
        self.w("void maka_mutex_destroy(maka_unit* m) { pthread_mutex_destroy((pthread_mutex_t*)m); free(m); }\n");
        // RwLock via pthread_rwlock_t.
        self.w("maka_unit* maka_rwlock_new(void) { pthread_rwlock_t* r = (pthread_rwlock_t*)malloc(sizeof(pthread_rwlock_t)); pthread_rwlock_init(r, NULL); return (maka_unit*)r; }\n");
        self.w("void maka_rwlock_read_lock(maka_unit* r) { pthread_rwlock_rdlock((pthread_rwlock_t*)r); }\n");
        self.w("void maka_rwlock_write_lock(maka_unit* r) { pthread_rwlock_wrlock((pthread_rwlock_t*)r); }\n");
        self.w("void maka_rwlock_unlock(maka_unit* r) { pthread_rwlock_unlock((pthread_rwlock_t*)r); }\n");
        self.w("void maka_rwlock_destroy(maka_unit* r) { pthread_rwlock_destroy((pthread_rwlock_t*)r); free(r); }\n");
        // Spinlock via pthread_spinlock_t (process-private).
        self.w("maka_unit* maka_spinlock_new(void) { pthread_spinlock_t* s = (pthread_spinlock_t*)malloc(sizeof(pthread_spinlock_t)); pthread_spin_init(s, PTHREAD_PROCESS_PRIVATE); return (maka_unit*)s; }\n");
        self.w("void maka_spinlock_lock(maka_unit* s) { pthread_spin_lock((pthread_spinlock_t*)s); }\n");
        self.w("void maka_spinlock_unlock(maka_unit* s) { pthread_spin_unlock((pthread_spinlock_t*)s); }\n");
        self.w("void maka_spinlock_destroy(maka_unit* s) { pthread_spin_destroy((pthread_spinlock_t*)s); free(s); }\n");
        // Channel: a simple unbounded queue of int64_t protected by a mutex + condvar.
        self.w("typedef struct maka_chan_node_t { maka_int v; struct maka_chan_node_t* next; } maka_chan_node_t;\n");
        self.w("typedef struct { pthread_mutex_t m; pthread_cond_t c; maka_chan_node_t* head; maka_chan_node_t* tail; maka_int count; } maka_channel_t;\n");
        self.w("maka_unit* maka_channel_new(void) { maka_channel_t* ch = (maka_channel_t*)malloc(sizeof(maka_channel_t)); pthread_mutex_init(&ch->m, NULL); pthread_cond_init(&ch->c, NULL); ch->head = NULL; ch->tail = NULL; ch->count = 0; return (maka_unit*)ch; }\n");
        self.w("void maka_channel_send(maka_unit* p, maka_int v) { maka_channel_t* ch = (maka_channel_t*)p; maka_chan_node_t* n = (maka_chan_node_t*)malloc(sizeof(maka_chan_node_t)); n->v = v; n->next = NULL; pthread_mutex_lock(&ch->m); if (ch->tail) ch->tail->next = n; else ch->head = n; ch->tail = n; ch->count++; pthread_cond_signal(&ch->c); pthread_mutex_unlock(&ch->m); }\n");
        self.w("maka_int maka_channel_recv(maka_unit* p) { maka_channel_t* ch = (maka_channel_t*)p; pthread_mutex_lock(&ch->m); while (!ch->head) pthread_cond_wait(&ch->c, &ch->m); maka_chan_node_t* n = ch->head; ch->head = n->next; if (!ch->head) ch->tail = NULL; ch->count--; pthread_mutex_unlock(&ch->m); maka_int v = n->v; free(n); return v; }\n");
        self.w("void maka_channel_destroy(maka_unit* p) { maka_channel_t* ch = (maka_channel_t*)p; while (ch->head) { maka_chan_node_t* n = ch->head; ch->head = n->next; free(n); } pthread_mutex_destroy(&ch->m); pthread_cond_destroy(&ch->c); free(ch); }\n");
        // Byte-channel: variable-size items (per-channel item_size).  Users
        // wrap this with typed sugar in their own modules — e.g.
        //   data PointChan { *unit handle }
        //   pt_chan_send(c, p) { chan_bytes_send(c.handle, &p as raw *unit); }
        // The runtime memcpy()s `item_size` bytes per send/recv.  Unbounded
        // capacity for v1.
        self.w("typedef struct maka_bnode_t {\n");
        self.w("    struct maka_bnode_t* next;\n");
        self.w("    char data[];\n");
        self.w("} maka_bnode_t;\n");
        self.w("typedef struct {\n");
        self.w("    int item_size;\n");
        self.w("    pthread_mutex_t m;\n");
        self.w("    pthread_cond_t  c;\n");
        self.w("    maka_bnode_t*   head;\n");
        self.w("    maka_bnode_t*   tail;\n");
        self.w("    int count;\n");
        self.w("    int closed;\n");
        self.w("} maka_bchan_t;\n");
        self.w("maka_unit* maka_chan_bytes_new(int64_t item_size) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)calloc(1, sizeof(maka_bchan_t));\n");
        self.w("    c->item_size = (int)item_size;\n");
        self.w("    pthread_mutex_init(&c->m, NULL);\n");
        self.w("    pthread_cond_init(&c->c, NULL);\n");
        self.w("    return (maka_unit*)c;\n");
        self.w("}\n");
        self.w("void maka_chan_bytes_send(maka_unit* p, maka_unit* src) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)p;\n");
        self.w("    maka_bnode_t* n = (maka_bnode_t*)malloc(sizeof(maka_bnode_t) + (size_t)c->item_size);\n");
        self.w("    memcpy(n->data, (void*)src, (size_t)c->item_size);\n");
        self.w("    n->next = NULL;\n");
        self.w("    pthread_mutex_lock(&c->m);\n");
        self.w("    if (c->tail) c->tail->next = n; else c->head = n;\n");
        self.w("    c->tail = n; c->count++;\n");
        self.w("    pthread_cond_signal(&c->c);\n");
        self.w("    pthread_mutex_unlock(&c->m);\n");
        self.w("}\n");
        // recv is defined later (after the scheduler) so it can yield via
        // swapcontext when called from the anchor with other fiber work
        // ready.  Forward declaration so other early code can reference it.
        self.w("void maka_chan_bytes_recv(maka_unit* p, maka_unit* dst);\n");
        self.w("int64_t maka_chan_bytes_count(maka_unit* p) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)p;\n");
        self.w("    pthread_mutex_lock(&c->m);\n");
        self.w("    int64_t v = c->count;\n");
        self.w("    pthread_mutex_unlock(&c->m);\n");
        self.w("    return v;\n");
        self.w("}\n");
        self.w("void maka_chan_bytes_close(maka_unit* p) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)p;\n");
        self.w("    pthread_mutex_lock(&c->m);\n");
        self.w("    c->closed = 1;\n");
        self.w("    pthread_cond_broadcast(&c->c);\n");
        self.w("    pthread_mutex_unlock(&c->m);\n");
        self.w("}\n");
        self.w("void maka_chan_bytes_destroy(maka_unit* p) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)p;\n");
        self.w("    while (c->head) { maka_bnode_t* n = c->head; c->head = n->next; free(n); }\n");
        self.w("    pthread_mutex_destroy(&c->m); pthread_cond_destroy(&c->c);\n");
        self.w("    free(c);\n");
        self.w("}\n");
        // Atomic<i64> — the only sync primitive whose body has no scheduler
        // dependency, so it lives here.  The fiber-aware Mutex / WaitGroup /
        // Once primitives are emitted after the scheduler is defined (they
        // need maka_fiber_t and __maka_ready_enqueue in scope).
        self.w("maka_unit* maka_atomic_i64_new(int64_t v) {\n");
        self.w("    _Atomic int64_t* a = (_Atomic int64_t*)malloc(sizeof(_Atomic int64_t));\n");
        self.w("    atomic_init(a, v);\n");
        self.w("    return (maka_unit*)a;\n");
        self.w("}\n");
        self.w("int64_t maka_atomic_i64_load(maka_unit* a) { return atomic_load((_Atomic int64_t*)a); }\n");
        self.w("void maka_atomic_i64_store(maka_unit* a, int64_t v) { atomic_store((_Atomic int64_t*)a, v); }\n");
        self.w("int64_t maka_atomic_i64_add(maka_unit* a, int64_t d) { return atomic_fetch_add((_Atomic int64_t*)a, d); }\n");
        self.w("int64_t maka_atomic_i64_cas(maka_unit* a, int64_t expected, int64_t desired) {\n");
        self.w("    int64_t e = expected;\n");
        self.w("    return atomic_compare_exchange_strong((_Atomic int64_t*)a, &e, desired) ? 1 : 0;\n");
        self.w("}\n");
        self.w("void maka_atomic_i64_destroy(maka_unit* a) { free(a); }\n");
        // Forward-declare the int slice type used by par_map_int.  If the
        // user's program references `[]int` elsewhere, `emit_slice_typedefs`
        // will skip re-emitting since `out.contains(...)` is true.
        self.w("typedef struct Slice_maka_int { maka_int* ptr; maka_int len; } Slice_maka_int;\n");
        self.w("typedef struct Slice_maka_float { maka_float* ptr; maka_int len; } Slice_maka_float;\n");
        self.slice_types.insert("maka_int".to_string());
        self.slice_types.insert("maka_float".to_string());
        // ====================================================================
        // CONCURRENCY RUNTIME
        // ====================================================================
        //
        // Three tiers exposed at the Maka surface (thread / spawn / job), each
        // returning the same `Thread*` handle and waited via `join`.  All three
        // are pthread-backed today; the surface is final and supports a future
        // swap to ucontext+epoll fibers / lock-free job pool without changing
        // user code.
        //
        // Differentiation per tier:
        //   thread(): full ~8 MB pthread stack          (blocking-safe)
        //   spawn():  smaller ~64 KB stack              ("fiber" — cheap)
        //   job():    worker pool (one pthread per CPU) (no per-job stack alloc)
        //
        // Handle structure carries an atomic done flag, a result slot (int64
        // type-erased payload), and a mutex/condvar so join() blocks cleanly.
        self.w("#include <unistd.h>\n#include <time.h>\n#include <errno.h>\n#include <signal.h>\n");
        self.w("typedef struct Thread {\n");
        self.w("    pthread_t       handle;\n");
        self.w("    pthread_mutex_t done_mutex;\n");
        self.w("    pthread_cond_t  done_cond;\n");
        self.w("    int             done_flag;       /* set to 1 when work finishes */\n");
        self.w("    int64_t         result;          /* type-erased return value */\n");
        self.w("    int             is_job;          /* 1 for job-pool work item */\n");
        self.w("    int             is_fiber;        /* 1 for cooperative fiber */\n");
        self.w("    _Atomic int     detached;        /* 1 if user opted out of join */\n");
        self.w("} Thread;\n");
        self.w("typedef struct { void* code; void* env; } __maka_closure_fat;\n");
        self.w("typedef struct __maka_handle_args_s { void* code; void* env; Thread* h; } __maka_handle_args_t;\n");
        // Entry for thread/spawn: run the closure body, set done flag, broadcast.
        self.w("static void* __maka_handle_entry(void* arg) {\n");
        self.w("    __maka_handle_args_t* a = (__maka_handle_args_t*)arg;\n");
        self.w("    void (*code)(void*) = (void (*)(void*))a->code;\n");
        self.w("    code(a->env);\n");
        self.w("    pthread_mutex_lock(&a->h->done_mutex);\n");
        self.w("    a->h->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&a->h->done_cond);\n");
        self.w("    pthread_mutex_unlock(&a->h->done_mutex);\n");
        self.w("    free(a->env);\n");
        self.w("    free(a);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        // thread() — kernel thread tier, default ~8 MB stack.
        self.w("maka_unit* __maka_spawn_thread(void* code, void* env) {\n");
        self.w("    Thread* t = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("    pthread_mutex_init(&t->done_mutex, NULL);\n");
        self.w("    pthread_cond_init(&t->done_cond, NULL);\n");
        self.w("    __maka_handle_args_t* a = (__maka_handle_args_t*)malloc(sizeof(__maka_handle_args_t));\n");
        self.w("    a->code = code; a->env = env; a->h = t;\n");
        self.w("    pthread_create(&t->handle, NULL, __maka_handle_entry, a);\n");
        self.w("    return (maka_unit*)t;\n");
        self.w("}\n");
        // ====================================================================
        // COOPERATIVE FIBER RUNTIME (ucontext-based)
        // ====================================================================
        //
        // spawn() creates a userspace fiber with its own ucontext + stack.
        // Fibers run on the calling pthread (a "scheduler thread") under a
        // round-robin scheduler.  They cooperate by yielding at IO/sleep/
        // explicit yield points.
        //
        // Key model:
        //   - Each pthread has its own thread-local scheduler + ready queue.
        //   - main starts as if it were already "running" a fiber (the
        //     "anchor fiber" lazily created on first spawn).
        //   - spawn(closure) creates a new ucontext, queues it; returns handle.
        //   - join(handle) enters the scheduler: scheduler runs ready fibers
        //     until handle is done, polls thread/job handles meanwhile, then
        //     returns to the caller.
        //   - Within a fiber, sleep_ms / yield_now switch to the scheduler;
        //     the scheduler picks the next ready fiber.
        //   - Outside a fiber (main not currently in scheduler, OS threads,
        //     job-pool workers), sleep_ms blocks via nanosleep as before.
        self.w("#include <ucontext.h>\n");
        self.w("#include <sys/mman.h>\n");
        self.w("#define MAKA_FIBER_STACK_SIZE (64 * 1024)\n");
        self.w("#define MAKA_FIBER_SLAB_RESERVE (1024 * 1024) /* 1 MB VM per fiber */\n");
        self.w("#define MAKA_FIBER_GUARD_PAGE 4096\n");
        // Per-thread free-list of slabs.  Reusing a slab avoids the mmap/munmap
        // cost on every spawn — same trick goroutines use.
        self.w("typedef struct maka_slab_s {\n");
        self.w("    void* base;                  /* mmap base (guard page at top) */\n");
        self.w("    void* stack_top;             /* start of usable stack region */\n");
        self.w("    struct maka_slab_s* next;    /* free-list link */\n");
        self.w("} maka_slab_t;\n");
        self.w("static __thread maka_slab_t* maka_slab_pool = NULL;\n");
        self.w("static maka_slab_t* __maka_slab_alloc(void) {\n");
        self.w("    if (maka_slab_pool) {\n");
        self.w("        maka_slab_t* s = maka_slab_pool;\n");
        self.w("        maka_slab_pool = s->next; s->next = NULL;\n");
        self.w("        return s;\n");
        self.w("    }\n");
        self.w("    /* mmap PROT_NONE for the full VM range, then mprotect the usable\n");
        self.w("       region read/write.  Bottom page stays PROT_NONE as the stack\n");
        self.w("       overflow guard. */\n");
        self.w("    void* base = mmap(NULL, MAKA_FIBER_SLAB_RESERVE, PROT_NONE,\n");
        self.w("                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);\n");
        self.w("    if (base == MAP_FAILED) return NULL;\n");
        self.w("    /* Commit only the top MAKA_FIBER_STACK_SIZE bytes; leave the bottom\n");
        self.w("       MAKA_FIBER_SLAB_RESERVE - MAKA_FIBER_STACK_SIZE as PROT_NONE so a\n");
        self.w("       stack-overflowing fiber segfaults cleanly instead of trampling\n");
        self.w("       another fiber's slab. */\n");
        self.w("    void* commit_start = (char*)base + MAKA_FIBER_SLAB_RESERVE - MAKA_FIBER_STACK_SIZE;\n");
        self.w("    if (mprotect(commit_start, MAKA_FIBER_STACK_SIZE, PROT_READ | PROT_WRITE) != 0) {\n");
        self.w("        munmap(base, MAKA_FIBER_SLAB_RESERVE);\n");
        self.w("        return NULL;\n");
        self.w("    }\n");
        self.w("    maka_slab_t* s = (maka_slab_t*)malloc(sizeof(maka_slab_t));\n");
        self.w("    s->base = base;\n");
        self.w("    s->stack_top = commit_start;\n");
        self.w("    s->next = NULL;\n");
        self.w("    return s;\n");
        self.w("}\n");
        self.w("static void __maka_slab_free(maka_slab_t* s) {\n");
        self.w("    /* Return to the pool — never munmap during the program's lifetime.\n");
        self.w("       This is what makes spawn cheap in the steady state. */\n");
        self.w("    s->next = maka_slab_pool;\n");
        self.w("    maka_slab_pool = s;\n");
        self.w("}\n");
        self.w("typedef struct maka_fiber_s {\n");
        self.w("    ucontext_t ctx;\n");
        self.w("    maka_slab_t* slab;    /* mmap'd 1 MB slab; stack lives at the top */\n");
        self.w("    int   state;          /* 0=ready 1=running 2=blocked-fiber 3=sleep 4=done */\n");
        self.w("    void  (*entry_code)(void*);\n");
        self.w("    void* entry_env;\n");
        self.w("    Thread* completion;   /* the Thread handle returned to user */\n");
        self.w("    int64_t wake_at_ns;\n");
        self.w("    int   waiting_fd;     /* fd if blocked on IO; -1 otherwise */\n");
        self.w("    int   waiting_events; /* EPOLLIN/EPOLLOUT bits this waiter is interested in */\n");
        self.w("    int64_t wait_deadline_ns; /* 0 = no deadline; otherwise wake-by-this time */\n");
        self.w("    int   wait_timed_out; /* set by scheduler when deadline expires */\n");
        self.w("    struct maka_fiber_s* next;        /* ready / sleep / fd-wait queue link */\n");
        self.w("    struct maka_fiber_s* waiters;     /* fibers blocked on this fiber */\n");
        self.w("    struct maka_fiber_s* next_waiter; /* waiter list link */\n");
        self.w("} maka_fiber_t;\n");
        self.w("static __thread ucontext_t maka_sched_ctx;\n");
        self.w("static __thread maka_fiber_t* maka_current_fiber = NULL;\n");
        self.w("static __thread maka_fiber_t* maka_ready_head = NULL;\n");
        self.w("static __thread maka_fiber_t* maka_ready_tail = NULL;\n");
        self.w("static __thread maka_fiber_t* maka_sleep_head = NULL;\n");
        self.w("static __thread int maka_sched_inited = 0;\n");
        // Blocking-syscall watchdog: each scheduler updates `last_tick_ns`
        // at the top of every loop iteration.  If MAKA_WATCHDOG_MS is set in
        // the env, a global watchdog thread periodically checks every
        // registered scheduler — if its tick hasn't advanced past the
        // threshold AND it has pending work, we warn on stderr.  The
        // assumption is that the fiber is stuck in a blocking syscall.
        self.w("typedef struct maka_sched_tick_s {\n");
        self.w("    _Atomic int64_t last_tick_ns;\n");
        self.w("    _Atomic int     has_work;\n");
        self.w("    _Atomic int     warned;\n");
        self.w("    struct maka_sched_tick_s* next;\n");
        self.w("} maka_sched_tick_t;\n");
        self.w("static pthread_mutex_t __maka_ticks_mu = PTHREAD_MUTEX_INITIALIZER;\n");
        self.w("static maka_sched_tick_t* __maka_ticks_head = NULL;\n");
        self.w("static __thread maka_sched_tick_t* __maka_my_tick = NULL;\n");
        self.w("static _Atomic int __maka_watchdog_started = 0;\n");
        self.w("static int64_t __maka_watchdog_threshold_ns = 0;\n");
        self.w("static __thread maka_fiber_t* maka_join_target = NULL;\n");
        self.w("static __thread maka_fiber_t* maka_anchor_fiber = NULL;\n");
        self.w("static __thread int maka_anchor_wake_on_finish = 0;\n");
        self.w("static __thread int maka_epoll_fd = -1;\n");
        self.w("static __thread int64_t maka_anchor_deadline_ns = 0; /* 0 = none; otherwise scheduler caps its timeout so anchor wakes by this */\n");
        // Reactor backend selection.  Three backends:
        //   * Linux         — epoll(7) used directly.
        //   * macOS / *BSD  — kqueue(2), via a shim exposing the epoll API.
        //   * Any other POSIX — poll(2) fallback (slower; rebuilds pollfd[]
        //                       from maka_fd_regs each scheduler tick).
        // Whichever backend is selected, the rest of the reactor code is
        // backend-agnostic: it speaks the epoll API and the shim translates.
        self.w("#include <poll.h>\n");
        self.w("#ifdef __linux__\n");
        self.w("#include <sys/epoll.h>\n");
        self.w("#define MAKA_USE_EPOLL 1\n");
        self.w("#define MAKA_USE_KQUEUE 0\n");
        self.w("#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)\n");
        self.w("#include <sys/event.h>\n");
        self.w("#include <sys/time.h>\n");
        self.w("#define MAKA_USE_EPOLL 0\n");
        self.w("#define MAKA_USE_KQUEUE 1\n");
        self.w("#define EPOLLIN  0x001\n");
        self.w("#define EPOLLOUT 0x004\n");
        self.w("#define EPOLLERR 0x008\n");
        self.w("#define EPOLLHUP 0x010\n");
        self.w("#define EPOLLONESHOT 0\n");
        self.w("#define EPOLL_CTL_ADD 1\n");
        self.w("#define EPOLL_CTL_MOD 2\n");
        self.w("#define EPOLL_CTL_DEL 3\n");
        self.w("#define EPOLL_CLOEXEC 0\n");
        self.w("typedef struct { int events; union { int fd; void* ptr; } data; } maka_epoll_event_t;\n");
        self.w("#define epoll_event maka_epoll_event_t\n");
        // kqueue: we open the kq fd lazily and translate epoll_ctl into
        // kevent ADD/DELETE filters.  EVFILT_READ and EVFILT_WRITE are added
        // independently per fd; EPOLL_CTL_DEL removes both.
        self.w("static int __maka_kq_fd = -1;\n");
        self.w("static inline int epoll_create1(int flags) {\n");
        self.w("    (void)flags;\n");
        self.w("    if (__maka_kq_fd < 0) __maka_kq_fd = kqueue();\n");
        self.w("    return __maka_kq_fd;\n");
        self.w("}\n");
        self.w("static inline int epoll_ctl(int ep, int op, int fd, struct epoll_event* e) {\n");
        self.w("    (void)ep;\n");
        self.w("    if (__maka_kq_fd < 0) __maka_kq_fd = kqueue();\n");
        self.w("    struct kevent changes[2]; int n = 0;\n");
        self.w("    int flags = (op == EPOLL_CTL_DEL) ? EV_DELETE : EV_ADD;\n");
        self.w("    int want_in  = e && (e->events & EPOLLIN);\n");
        self.w("    int want_out = e && (e->events & EPOLLOUT);\n");
        self.w("    if (op == EPOLL_CTL_DEL || want_in) {\n");
        self.w("        EV_SET(&changes[n], fd, EVFILT_READ, flags, 0, 0, NULL); n++;\n");
        self.w("    }\n");
        self.w("    if (op == EPOLL_CTL_DEL || want_out) {\n");
        self.w("        EV_SET(&changes[n], fd, EVFILT_WRITE, flags, 0, 0, NULL); n++;\n");
        self.w("    }\n");
        self.w("    return kevent(__maka_kq_fd, changes, n, NULL, 0, NULL);\n");
        self.w("}\n");
        self.w("static inline int __maka_epoll_wait_kq(struct epoll_event* evs, int max, int timeout_ms) {\n");
        self.w("    if (__maka_kq_fd < 0) __maka_kq_fd = kqueue();\n");
        self.w("    struct kevent kevs[32]; if (max > 32) max = 32;\n");
        self.w("    struct timespec ts; struct timespec* pts = NULL;\n");
        self.w("    if (timeout_ms >= 0) { ts.tv_sec = timeout_ms/1000; ts.tv_nsec = (timeout_ms%1000)*1000000; pts = &ts; }\n");
        self.w("    int n = kevent(__maka_kq_fd, NULL, 0, kevs, max, pts);\n");
        self.w("    int out = 0;\n");
        self.w("    for (int i = 0; i < n; i++) {\n");
        self.w("        evs[out].events = 0;\n");
        self.w("        if (kevs[i].filter == EVFILT_READ)  evs[out].events |= EPOLLIN;\n");
        self.w("        if (kevs[i].filter == EVFILT_WRITE) evs[out].events |= EPOLLOUT;\n");
        self.w("        if (kevs[i].flags & EV_EOF)         evs[out].events |= EPOLLHUP;\n");
        self.w("        if (kevs[i].flags & EV_ERROR)       evs[out].events |= EPOLLERR;\n");
        self.w("        evs[out].data.fd = (int)kevs[i].ident;\n");
        self.w("        out++;\n");
        self.w("    }\n");
        self.w("    return out;\n");
        self.w("}\n");
        self.w("#define epoll_wait(ep, evs, max, t) __maka_epoll_wait_kq((evs), (max), (t))\n");
        self.w("#else\n");
        self.w("#define MAKA_USE_EPOLL 0\n");
        self.w("#define MAKA_USE_KQUEUE 0\n");
        self.w("#define EPOLLIN  POLLIN\n");
        self.w("#define EPOLLOUT POLLOUT\n");
        self.w("#define EPOLLERR POLLERR\n");
        self.w("#define EPOLLHUP POLLHUP\n");
        self.w("#define EPOLLONESHOT 0\n");
        self.w("#define EPOLL_CTL_ADD 1\n");
        self.w("#define EPOLL_CTL_MOD 2\n");
        self.w("#define EPOLL_CTL_DEL 3\n");
        self.w("#define EPOLL_CLOEXEC 0\n");
        self.w("typedef struct { int events; union { int fd; void* ptr; } data; } maka_epoll_event_t;\n");
        self.w("#define epoll_event maka_epoll_event_t\n");
        self.w("static inline int epoll_create1(int flags) { (void)flags; return 0; }\n");
        self.w("static inline int epoll_ctl(int ep, int op, int fd, struct epoll_event* e) {\n");
        self.w("    (void)ep; (void)op; (void)fd; (void)e; return 0;\n");
        self.w("}\n");
        self.w("static int __maka_epoll_wait_poll(struct epoll_event* evs, int max, int timeout_ms);\n");
        self.w("#define epoll_wait(ep, evs, max, t) __maka_epoll_wait_poll((evs), (max), (t))\n");
        self.w("#endif\n");
        // Per-fd reactor registration so that multiple fibers can wait on the
        // same fd without overwriting each other's epoll entry.  Each fd
        // tracks its currently-armed event mask; the scheduler re-computes
        // the mask on every wake and DELs the fd when the last waiter goes.
        self.w("typedef struct maka_fd_reg_s {\n");
        self.w("    int fd;\n");
        self.w("    int events_mask;        /* epoll bits currently armed */\n");
        self.w("    struct maka_fd_reg_s* next;\n");
        self.w("} maka_fd_reg_t;\n");
        self.w("static __thread maka_fd_reg_t* maka_fd_regs = NULL;\n");
        self.w("static maka_fd_reg_t* __maka_fd_reg_get(int fd) {\n");
        self.w("    for (maka_fd_reg_t* r = maka_fd_regs; r; r = r->next) if (r->fd == fd) return r;\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static maka_fd_reg_t* __maka_fd_reg_ensure(int fd) {\n");
        self.w("    maka_fd_reg_t* r = __maka_fd_reg_get(fd);\n");
        self.w("    if (r) return r;\n");
        self.w("    r = (maka_fd_reg_t*)calloc(1, sizeof(maka_fd_reg_t));\n");
        self.w("    r->fd = fd; r->events_mask = 0; r->next = maka_fd_regs; maka_fd_regs = r;\n");
        self.w("    return r;\n");
        self.w("}\n");
        self.w("static void __maka_fd_recompute(int fd);     /* fwd decl */\n");
        self.w("static void __maka_fd_arm(int fd, int events_mask);  /* fwd decl */\n");
        // poll()-backend epoll_wait shim: walks maka_fd_regs to build the
        // pollfd[] each call, then maps revents back into epoll-style events.
        self.w("#if !MAKA_USE_EPOLL\n");
        self.w("static int __maka_epoll_wait_poll(struct epoll_event* evs, int max, int timeout_ms) {\n");
        self.w("    /* Count registered fds. */\n");
        self.w("    int n = 0;\n");
        self.w("    for (maka_fd_reg_t* r = maka_fd_regs; r; r = r->next) n++;\n");
        self.w("    if (n == 0) { if (timeout_ms > 0) { struct timespec ts = { timeout_ms/1000, (timeout_ms%1000)*1000000 }; nanosleep(&ts, NULL); } return 0; }\n");
        self.w("    struct pollfd* pfds = (struct pollfd*)calloc((size_t)n, sizeof(struct pollfd));\n");
        self.w("    int* fds = (int*)calloc((size_t)n, sizeof(int));\n");
        self.w("    int i = 0;\n");
        self.w("    for (maka_fd_reg_t* r = maka_fd_regs; r; r = r->next, i++) {\n");
        self.w("        pfds[i].fd = r->fd;\n");
        self.w("        pfds[i].events = (short)(r->events_mask & (POLLIN | POLLOUT));\n");
        self.w("        fds[i] = r->fd;\n");
        self.w("    }\n");
        self.w("    int rv = poll(pfds, (nfds_t)n, timeout_ms);\n");
        self.w("    int out = 0;\n");
        self.w("    if (rv > 0) {\n");
        self.w("        for (int j = 0; j < n && out < max; j++) {\n");
        self.w("            if (pfds[j].revents) {\n");
        self.w("                evs[out].events = pfds[j].revents;\n");
        self.w("                evs[out].data.fd = fds[j];\n");
        self.w("                out++;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    free(pfds); free(fds);\n");
        self.w("    return out;\n");
        self.w("}\n");
        self.w("#endif\n");
        self.w("static void __maka_fd_reg_drop(int fd) {\n");
        self.w("    maka_fd_reg_t** prev = &maka_fd_regs;\n");
        self.w("    while (*prev) {\n");
        self.w("        if ((*prev)->fd == fd) {\n");
        self.w("            maka_fd_reg_t* r = *prev; *prev = r->next; free(r); return;\n");
        self.w("        }\n");
        self.w("        prev = &(*prev)->next;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static __thread maka_fiber_t* maka_fd_waiters = NULL;\n");
        self.w("#define MAKA_EV_READ  1\n");
        self.w("#define MAKA_EV_WRITE 2\n");
        self.w("static __thread char maka_sched_stack[256 * 1024];\n");
        self.w("static void __maka_ready_enqueue(maka_fiber_t* f) {\n");
        self.w("    f->state = 0; f->next = NULL;\n");
        self.w("    if (maka_ready_tail) { maka_ready_tail->next = f; maka_ready_tail = f; }\n");
        self.w("    else { maka_ready_head = maka_ready_tail = f; }\n");
        self.w("}\n");
        self.w("static maka_fiber_t* __maka_ready_dequeue(void) {\n");
        self.w("    maka_fiber_t* f = maka_ready_head;\n");
        self.w("    if (!f) return NULL;\n");
        self.w("    maka_ready_head = f->next;\n");
        self.w("    if (!maka_ready_head) maka_ready_tail = NULL;\n");
        self.w("    f->next = NULL;\n");
        self.w("    return f;\n");
        self.w("}\n");
        self.w("static int64_t __maka_now_ns(void) {\n");
        self.w("    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);\n");
        self.w("    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;\n");
        self.w("}\n");
        // Watchdog thread: scans the registered scheduler ticks every
        // (threshold / 2) milliseconds and warns about any tick whose
        // `has_work` flag is set but whose `last_tick_ns` is older than
        // the threshold.  Warn at most once per quiet period.
        self.w("static void* __maka_watchdog_loop(void* arg) {\n");
        self.w("    (void)arg;\n");
        self.w("    int64_t threshold = __maka_watchdog_threshold_ns;\n");
        self.w("    int64_t sleep_ms = (threshold / 1000000) / 2;\n");
        self.w("    if (sleep_ms < 100) sleep_ms = 100;\n");
        self.w("    while (1) {\n");
        self.w("        struct timespec ts = { sleep_ms / 1000, (sleep_ms % 1000) * 1000000L };\n");
        self.w("        nanosleep(&ts, NULL);\n");
        self.w("        int64_t now = __maka_now_ns();\n");
        self.w("        pthread_mutex_lock(&__maka_ticks_mu);\n");
        self.w("        for (maka_sched_tick_t* t = __maka_ticks_head; t; t = t->next) {\n");
        self.w("            if (!atomic_load(&t->has_work)) { atomic_store(&t->warned, 0); continue; }\n");
        self.w("            int64_t last = atomic_load(&t->last_tick_ns);\n");
        self.w("            if (now - last > threshold && !atomic_load(&t->warned)) {\n");
        self.w("                fprintf(stderr, \"maka: scheduler stuck — fibers haven't yielded for %.2fs (likely a blocking syscall inside a fiber)\\n\", (double)(now - last) / 1e9);\n");
        self.w("                atomic_store(&t->warned, 1);\n");
        self.w("            }\n");
        self.w("            if (now - last <= threshold && atomic_load(&t->warned)) atomic_store(&t->warned, 0);\n");
        self.w("        }\n");
        self.w("        pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("    }\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static void __maka_watchdog_register(void) {\n");
        self.w("    if (__maka_watchdog_threshold_ns == 0) {\n");
        self.w("        const char* env = getenv(\"MAKA_WATCHDOG_MS\");\n");
        self.w("        int64_t ms = env ? atoll(env) : 0;\n");
        self.w("        if (ms <= 0) { __maka_watchdog_threshold_ns = -1; return; }\n");
        self.w("        __maka_watchdog_threshold_ns = ms * 1000000LL;\n");
        self.w("    }\n");
        self.w("    if (__maka_watchdog_threshold_ns < 0) return;\n");
        self.w("    if (__maka_my_tick) return;\n");
        self.w("    __maka_my_tick = (maka_sched_tick_t*)calloc(1, sizeof(maka_sched_tick_t));\n");
        self.w("    atomic_init(&__maka_my_tick->last_tick_ns, __maka_now_ns());\n");
        self.w("    pthread_mutex_lock(&__maka_ticks_mu);\n");
        self.w("    __maka_my_tick->next = __maka_ticks_head;\n");
        self.w("    __maka_ticks_head = __maka_my_tick;\n");
        self.w("    pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("    int expected = 0;\n");
        self.w("    if (atomic_compare_exchange_strong(&__maka_watchdog_started, &expected, 1)) {\n");
        self.w("        pthread_t w; pthread_create(&w, NULL, __maka_watchdog_loop, NULL); pthread_detach(w);\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static void __maka_scheduler_loop(void) {\n");
        self.w("    while (1) {\n");
        self.w("        int64_t now = __maka_now_ns();\n");
        self.w("        if (__maka_my_tick) {\n");
        self.w("            atomic_store(&__maka_my_tick->last_tick_ns, now);\n");
        self.w("            atomic_store(&__maka_my_tick->has_work, (maka_ready_head || maka_sleep_head || maka_fd_waiters) ? 1 : 0);\n");
        self.w("        }\n");
        self.w("        /* Move expired sleepers to ready */\n");
        self.w("        maka_fiber_t** prev = &maka_sleep_head;\n");
        self.w("        while (*prev) {\n");
        self.w("            maka_fiber_t* sf = *prev;\n");
        self.w("            if (sf->wake_at_ns <= now) {\n");
        self.w("                *prev = sf->next; sf->next = NULL;\n");
        self.w("                __maka_ready_enqueue(sf);\n");
        self.w("            } else { prev = &sf->next; }\n");
        self.w("        }\n");
        self.w("        /* Is the awaited target done? */\n");
        self.w("        if (maka_join_target) {\n");
        self.w("            Thread* tgt = maka_join_target->completion;\n");
        self.w("            int done = 0;\n");
        self.w("            pthread_mutex_lock(&tgt->done_mutex);\n");
        self.w("            done = tgt->done_flag;\n");
        self.w("            pthread_mutex_unlock(&tgt->done_mutex);\n");
        self.w("            if (done) {\n");
        self.w("                maka_fiber_t* anchor = maka_anchor_fiber;\n");
        self.w("                maka_join_target = NULL;\n");
        self.w("                maka_current_fiber = anchor;\n");
        self.w("                swapcontext(&maka_sched_ctx, &anchor->ctx);\n");
        self.w("                continue;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        /* Pick a ready fiber */\n");
        self.w("        maka_fiber_t* f = __maka_ready_dequeue();\n");
        self.w("        if (f) {\n");
        self.w("            maka_current_fiber = f;\n");
        self.w("            f->state = 1;\n");
        self.w("            swapcontext(&maka_sched_ctx, &f->ctx);\n");
        self.w("            maka_current_fiber = NULL;\n");
        self.w("            if (f->state == 4) {\n");
        self.w("                /* Fiber finished: mark completion + wake waiters. */\n");
        self.w("                Thread* fcompl = f->completion;\n");
        self.w("                pthread_mutex_lock(&fcompl->done_mutex);\n");
        self.w("                fcompl->done_flag = 1;\n");
        self.w("                pthread_cond_broadcast(&fcompl->done_cond);\n");
        self.w("                pthread_mutex_unlock(&fcompl->done_mutex);\n");
        self.w("                while (f->waiters) {\n");
        self.w("                    maka_fiber_t* w = f->waiters;\n");
        self.w("                    f->waiters = w->next_waiter; w->next_waiter = NULL;\n");
        self.w("                    __maka_ready_enqueue(w);\n");
        self.w("                }\n");
        self.w("                if (f != maka_anchor_fiber) { free(f->entry_env); __maka_slab_free(f->slab); free(f); }\n");
        self.w("                /* If the Thread handle was detached, reap it now — no joiner will. */\n");
        self.w("                if (atomic_load(&fcompl->detached)) {\n");
        self.w("                    pthread_mutex_destroy(&fcompl->done_mutex);\n");
        self.w("                    pthread_cond_destroy(&fcompl->done_cond);\n");
        self.w("                    free(fcompl);\n");
        self.w("                }\n");
        self.w("                /* If anchor is parked in a select loop, return so it can re-poll. */\n");
        self.w("                if (maka_anchor_wake_on_finish) {\n");
        self.w("                    maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                    swapcontext(&maka_sched_ctx, &maka_anchor_fiber->ctx);\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        /* No ready fibers — wait for sleepers, fd events, or fall out. */\n");
        self.w("        int have_sleepers = (maka_sleep_head != NULL);\n");
        self.w("        int have_fd_waiters = (maka_fd_waiters != NULL);\n");
        self.w("        if (have_sleepers || have_fd_waiters) {\n");
        self.w("            int64_t timeout_ms = -1;\n");
        self.w("            int64_t now_ns = __maka_now_ns();\n");
        self.w("            if (have_sleepers) {\n");
        self.w("                int64_t min_wake = maka_sleep_head->wake_at_ns;\n");
        self.w("                for (maka_fiber_t* s = maka_sleep_head->next; s; s = s->next) {\n");
        self.w("                    if (s->wake_at_ns < min_wake) min_wake = s->wake_at_ns;\n");
        self.w("                }\n");
        self.w("                int64_t delta = min_wake - now_ns;\n");
        self.w("                if (delta <= 0) { continue; /* wake immediately */ }\n");
        self.w("                timeout_ms = delta / 1000000LL;\n");
        self.w("                if (timeout_ms < 1) timeout_ms = 1;\n");
        self.w("            }\n");
        self.w("            /* Honor wait_fd_timeout deadlines. */\n");
        self.w("            for (maka_fiber_t* f = maka_fd_waiters; f; f = f->next) {\n");
        self.w("                if (f->wait_deadline_ns == 0) continue;\n");
        self.w("                int64_t delta = f->wait_deadline_ns - now_ns;\n");
        self.w("                if (delta <= 0) { timeout_ms = 0; break; }\n");
        self.w("                int64_t dms = delta / 1000000LL;\n");
        self.w("                if (dms < 1) dms = 1;\n");
        self.w("                if (timeout_ms < 0 || dms < timeout_ms) timeout_ms = dms;\n");
        self.w("            }\n");
        self.w("            /* Honor anchor deadline (set by join_timeout / select_timeout). */\n");
        self.w("            if (maka_anchor_deadline_ns != 0) {\n");
        self.w("                int64_t delta = maka_anchor_deadline_ns - now_ns;\n");
        self.w("                if (delta <= 0) { timeout_ms = 0; }\n");
        self.w("                else {\n");
        self.w("                    int64_t dms = delta / 1000000LL;\n");
        self.w("                    if (dms < 1) dms = 1;\n");
        self.w("                    if (timeout_ms < 0 || dms < timeout_ms) timeout_ms = dms;\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("            if (have_fd_waiters && maka_epoll_fd >= 0) {\n");
        self.w("                struct epoll_event evs[32];\n");
        self.w("                int n = epoll_wait(maka_epoll_fd, evs, 32, (int)timeout_ms);\n");
        self.w("                for (int i = 0; i < n; i++) {\n");
        self.w("                    int evfd = evs[i].data.fd;\n");
        self.w("                    int err_hup = evs[i].events & (EPOLLERR | EPOLLHUP);\n");
        self.w("                    /* Wake every fiber waiting on this fd whose interest\n");
        self.w("                       intersects the fired events, plus all of them on err/hup. */\n");
        self.w("                    maka_fiber_t** prev2 = &maka_fd_waiters;\n");
        self.w("                    while (*prev2) {\n");
        self.w("                        maka_fiber_t* w = *prev2;\n");
        self.w("                        if (w->waiting_fd == evfd && (err_hup || (evs[i].events & w->waiting_events))) {\n");
        self.w("                            *prev2 = w->next; w->next = NULL;\n");
        self.w("                            w->waiting_fd = -1; w->waiting_events = 0;\n");
        self.w("                            w->wait_deadline_ns = 0; w->wait_timed_out = 0;\n");
        self.w("                            __maka_ready_enqueue(w);\n");
        self.w("                        } else {\n");
        self.w("                            prev2 = &(*prev2)->next;\n");
        self.w("                        }\n");
        self.w("                    }\n");
        self.w("                    /* Re-arm with the remaining interest, or drop. */\n");
        self.w("                    __maka_fd_recompute(evfd);\n");
        self.w("                }\n");
        self.w("                /* After epoll, reap timed-out fd waiters. */\n");
        self.w("                int64_t now2 = __maka_now_ns();\n");
        self.w("                maka_fiber_t** prev3 = &maka_fd_waiters;\n");
        self.w("                int timed_out_fds[32]; int n_timed_out = 0;\n");
        self.w("                while (*prev3) {\n");
        self.w("                    maka_fiber_t* f = *prev3;\n");
        self.w("                    if (f->wait_deadline_ns != 0 && f->wait_deadline_ns <= now2) {\n");
        self.w("                        int tfd = f->waiting_fd;\n");
        self.w("                        *prev3 = f->next; f->next = NULL;\n");
        self.w("                        f->waiting_fd = -1; f->waiting_events = 0;\n");
        self.w("                        f->wait_deadline_ns = 0;\n");
        self.w("                        f->wait_timed_out = 1;\n");
        self.w("                        __maka_ready_enqueue(f);\n");
        self.w("                        /* Record fd so we can recompute the per-fd mask\n");
        self.w("                           after the walk (avoid mutating the registry while\n");
        self.w("                           the per-waiter walk is still in flight). */\n");
        self.w("                        if (n_timed_out < 32 && tfd >= 0) timed_out_fds[n_timed_out++] = tfd;\n");
        self.w("                    } else {\n");
        self.w("                        prev3 = &(*prev3)->next;\n");
        self.w("                    }\n");
        self.w("                }\n");
        self.w("                for (int k = 0; k < n_timed_out; k++) __maka_fd_recompute(timed_out_fds[k]);\n");
        self.w("            } else if (have_sleepers) {\n");
        self.w("                struct timespec ts = { timeout_ms / 1000, (timeout_ms % 1000) * 1000000 };\n");
        self.w("                nanosleep(&ts, NULL);\n");
        self.w("            }\n");
        self.w("            /* Anchor deadline reached?  Hand control back so the\n");
        self.w("               caller's timeout primitive can finish. */\n");
        self.w("            if (maka_anchor_deadline_ns != 0 && __maka_now_ns() >= maka_anchor_deadline_ns && maka_anchor_fiber) {\n");
        self.w("                maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                swapcontext(&maka_sched_ctx, &maka_anchor_fiber->ctx);\n");
        self.w("            }\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        /* Nothing more to do: poll the awaited target with a small sleep. */\n");
        self.w("        if (maka_join_target) {\n");
        self.w("            struct timespec ts = { 0, 200000 /* 200us */ };\n");
        self.w("            nanosleep(&ts, NULL);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        /* No work, no joiner: hand back to anchor. */\n");
        self.w("        if (maka_anchor_fiber) {\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("            swapcontext(&maka_sched_ctx, &maka_anchor_fiber->ctx);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static void __maka_sched_init(void) {\n");
        self.w("    if (maka_sched_inited) return;\n");
        self.w("    maka_sched_inited = 1;\n");
        self.w("    __maka_watchdog_register();\n");
        self.w("    /* anchor represents the calling (main or pthread) context */\n");
        self.w("    maka_anchor_fiber = (maka_fiber_t*)calloc(1, sizeof(maka_fiber_t));\n");
        self.w("    maka_anchor_fiber->state = 1;\n");
        self.w("    maka_current_fiber = maka_anchor_fiber;\n");
        self.w("    getcontext(&maka_sched_ctx);\n");
        self.w("    maka_sched_ctx.uc_stack.ss_sp = maka_sched_stack;\n");
        self.w("    maka_sched_ctx.uc_stack.ss_size = sizeof(maka_sched_stack);\n");
        self.w("    maka_sched_ctx.uc_link = NULL;\n");
        self.w("    makecontext(&maka_sched_ctx, __maka_scheduler_loop, 0);\n");
        self.w("}\n");
        self.w("static void __maka_fiber_entry(void) {\n");
        self.w("    maka_fiber_t* f = maka_current_fiber;\n");
        self.w("    f->entry_code(f->entry_env);\n");
        self.w("    f->state = 4;\n");
        self.w("    swapcontext(&f->ctx, &maka_sched_ctx);\n");
        self.w("}\n");
        // spawn(): create a userspace cooperative fiber.
        self.w("maka_unit* __maka_spawn_fiber(void* code, void* env) {\n");
        self.w("    __maka_sched_init();\n");
        self.w("    Thread* t = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("    t->is_fiber = 1;\n");
        self.w("    pthread_mutex_init(&t->done_mutex, NULL);\n");
        self.w("    pthread_cond_init(&t->done_cond, NULL);\n");
        self.w("    maka_fiber_t* f = (maka_fiber_t*)calloc(1, sizeof(maka_fiber_t));\n");
        self.w("    f->slab = __maka_slab_alloc();\n");
        self.w("    f->entry_code = (void(*)(void*))code;\n");
        self.w("    f->entry_env = env;\n");
        self.w("    f->completion = t;\n");
        self.w("    f->state = 0;\n");
        self.w("    f->waiting_fd = -1;\n");
        self.w("    f->waiting_events = 0;\n");
        self.w("    f->wait_deadline_ns = 0;\n");
        self.w("    f->wait_timed_out = 0;\n");
        self.w("    getcontext(&f->ctx);\n");
        self.w("    f->ctx.uc_stack.ss_sp = f->slab->stack_top;\n");
        self.w("    f->ctx.uc_stack.ss_size = MAKA_FIBER_STACK_SIZE;\n");
        self.w("    f->ctx.uc_link = &maka_sched_ctx;\n");
        self.w("    makecontext(&f->ctx, __maka_fiber_entry, 0);\n");
        self.w("    __maka_ready_enqueue(f);\n");
        self.w("    return (maka_unit*)t;\n");
        self.w("}\n");
        // ====================================================================
        // Cross-thread fiber pool — `spawn_pool(closure)` pushes the fiber
        // onto a global MPMC queue; N background worker threads drain it.
        // Each worker runs its own per-thread fiber scheduler, so spawned
        // fibers fan out across CPU cores without changing `spawn()`'s
        // single-threaded semantics.
        // ====================================================================
        self.w("typedef struct {\n");
        self.w("    maka_fiber_t* head; maka_fiber_t* tail;\n");
        self.w("    pthread_mutex_t lock; pthread_cond_t cond;\n");
        self.w("    int closed;\n");
        self.w("} maka_pool_q_t;\n");
        self.w("static maka_pool_q_t __maka_pool_q;\n");
        self.w("static _Atomic int __maka_pool_inited = 0;\n");
        self.w("static int __maka_pool_n_workers = 0;\n");
        self.w("static void __maka_pool_q_push(maka_fiber_t* f) {\n");
        self.w("    pthread_mutex_lock(&__maka_pool_q.lock);\n");
        self.w("    f->next = NULL;\n");
        self.w("    if (__maka_pool_q.tail) __maka_pool_q.tail->next = f;\n");
        self.w("    else __maka_pool_q.head = f;\n");
        self.w("    __maka_pool_q.tail = f;\n");
        self.w("    pthread_cond_signal(&__maka_pool_q.cond);\n");
        self.w("    pthread_mutex_unlock(&__maka_pool_q.lock);\n");
        self.w("}\n");
        self.w("static maka_fiber_t* __maka_pool_q_pop_timed(int ms) {\n");
        self.w("    pthread_mutex_lock(&__maka_pool_q.lock);\n");
        self.w("    while (!__maka_pool_q.head && !__maka_pool_q.closed) {\n");
        self.w("        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);\n");
        self.w("        ts.tv_sec += ms / 1000;\n");
        self.w("        ts.tv_nsec += (ms % 1000) * 1000000L;\n");
        self.w("        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }\n");
        self.w("        int r = pthread_cond_timedwait(&__maka_pool_q.cond, &__maka_pool_q.lock, &ts);\n");
        self.w("        if (r != 0) break;\n");
        self.w("    }\n");
        self.w("    maka_fiber_t* f = __maka_pool_q.head;\n");
        self.w("    if (f) {\n");
        self.w("        __maka_pool_q.head = f->next;\n");
        self.w("        if (!__maka_pool_q.head) __maka_pool_q.tail = NULL;\n");
        self.w("        f->next = NULL;\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&__maka_pool_q.lock);\n");
        self.w("    return f;\n");
        self.w("}\n");
        self.w("static void* __maka_pool_worker(void* arg) {\n");
        self.w("    (void)arg;\n");
        self.w("    __maka_sched_init();\n");
        self.w("    while (1) {\n");
        self.w("        maka_fiber_t* f = __maka_pool_q_pop_timed(500);\n");
        self.w("        if (!f) {\n");
        self.w("            pthread_mutex_lock(&__maka_pool_q.lock);\n");
        self.w("            int closed = __maka_pool_q.closed;\n");
        self.w("            pthread_mutex_unlock(&__maka_pool_q.lock);\n");
        self.w("            if (closed) break;\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        __maka_ready_enqueue(f);\n");
        self.w("        /* Drive scheduler until our local queue is empty + nothing waiting. */\n");
        self.w("        swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("        maka_current_fiber = maka_anchor_fiber;\n");
        self.w("    }\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static void __maka_pool_init(void) {\n");
        self.w("    int expected = 0;\n");
        self.w("    if (!atomic_compare_exchange_strong(&__maka_pool_inited, &expected, 1)) return;\n");
        self.w("    pthread_mutex_init(&__maka_pool_q.lock, NULL);\n");
        self.w("    pthread_cond_init(&__maka_pool_q.cond, NULL);\n");
        self.w("    long n = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (n < 2) n = 2;\n");
        self.w("    if (n > 16) n = 16;\n");
        self.w("    __maka_pool_n_workers = (int)n;\n");
        self.w("    for (int i = 0; i < __maka_pool_n_workers; i++) {\n");
        self.w("        pthread_t w; pthread_create(&w, NULL, __maka_pool_worker, NULL); pthread_detach(w);\n");
        self.w("    }\n");
        self.w("}\n");
        // spawn_pool(): spawn a fiber that runs on the background pool.
        self.w("maka_unit* __maka_spawn_pool(void* code, void* env) {\n");
        self.w("    __maka_pool_init();\n");
        self.w("    Thread* t = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("    t->is_fiber = 1;\n");
        self.w("    pthread_mutex_init(&t->done_mutex, NULL);\n");
        self.w("    pthread_cond_init(&t->done_cond, NULL);\n");
        self.w("    maka_fiber_t* f = (maka_fiber_t*)calloc(1, sizeof(maka_fiber_t));\n");
        self.w("    f->slab = __maka_slab_alloc();\n");
        self.w("    f->entry_code = (void(*)(void*))code;\n");
        self.w("    f->entry_env = env;\n");
        self.w("    f->completion = t;\n");
        self.w("    f->state = 0;\n");
        self.w("    f->waiting_fd = -1; f->waiting_events = 0;\n");
        self.w("    f->wait_deadline_ns = 0; f->wait_timed_out = 0;\n");
        self.w("    getcontext(&f->ctx);\n");
        self.w("    f->ctx.uc_stack.ss_sp = f->slab->stack_top;\n");
        self.w("    f->ctx.uc_stack.ss_size = MAKA_FIBER_STACK_SIZE;\n");
        self.w("    f->ctx.uc_link = NULL;\n");
        self.w("    makecontext(&f->ctx, __maka_fiber_entry, 0);\n");
        self.w("    __maka_pool_q_push(f);\n");
        self.w("    return (maka_unit*)t;\n");
        self.w("}\n");
        // Cooperative sleep / yield primitives.
        self.w("void __maka_yield_now(void) {\n");
        self.w("    if (!maka_current_fiber || maka_current_fiber == maka_anchor_fiber) return;\n");
        self.w("    maka_fiber_t* me = maka_current_fiber;\n");
        self.w("    __maka_ready_enqueue(me);\n");
        self.w("    swapcontext(&me->ctx, &maka_sched_ctx);\n");
        self.w("}\n");
        self.w("static void __maka_sleep_fiber(int64_t nanos) {\n");
        self.w("    maka_fiber_t* me = maka_current_fiber;\n");
        self.w("    me->wake_at_ns = __maka_now_ns() + nanos;\n");
        self.w("    me->state = 3;\n");
        self.w("    me->next = maka_sleep_head;\n");
        self.w("    maka_sleep_head = me;\n");
        self.w("    swapcontext(&me->ctx, &maka_sched_ctx);\n");
        self.w("}\n");
        // ====================================================================
        // EPOLL REACTOR — IO primitives that yield through the scheduler.
        // ====================================================================
        // wait_fd(fd, events) parks the fiber until fd becomes ready for the
        // requested events.  Outside a fiber it falls back to poll() so the
        // same call works from any context.
        // Arm or re-arm `fd` in epoll with `events_mask` (bitwise OR of all
        // waiters' requests).  Drops the registration when the mask is 0.
        self.w("static void __maka_fd_arm(int fd, int events_mask) {\n");
        self.w("    if (maka_epoll_fd < 0) { maka_epoll_fd = epoll_create1(EPOLL_CLOEXEC); }\n");
        self.w("    if (events_mask == 0) {\n");
        self.w("        epoll_ctl(maka_epoll_fd, EPOLL_CTL_DEL, fd, NULL);\n");
        self.w("        __maka_fd_reg_drop(fd);\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("    maka_fd_reg_t* r = __maka_fd_reg_ensure(fd);\n");
        self.w("    if (r->events_mask == events_mask) return;\n");
        self.w("    struct epoll_event ev; memset(&ev, 0, sizeof(ev));\n");
        self.w("    ev.events = events_mask | EPOLLERR | EPOLLHUP;\n");
        self.w("    ev.data.fd = fd;\n");
        self.w("    if (epoll_ctl(maka_epoll_fd, EPOLL_CTL_MOD, fd, &ev) != 0) {\n");
        self.w("        if (errno == ENOENT) epoll_ctl(maka_epoll_fd, EPOLL_CTL_ADD, fd, &ev);\n");
        self.w("    }\n");
        self.w("    r->events_mask = events_mask;\n");
        self.w("}\n");
        // Recompute the union event mask for `fd` from the remaining waiters
        // on it and update epoll accordingly.
        self.w("static void __maka_fd_recompute(int fd) {\n");
        self.w("    int mask = 0;\n");
        self.w("    for (maka_fiber_t* w = maka_fd_waiters; w; w = w->next) {\n");
        self.w("        if (w->waiting_fd == fd) mask |= w->waiting_events;\n");
        self.w("    }\n");
        self.w("    __maka_fd_arm(fd, mask);\n");
        self.w("}\n");
        // wait_fd: park `me` waiting for `events` on `fd`.  Safe with multiple
        // fibers parked on the same fd — each gets its own waiter entry and
        // the per-fd registered mask is the union of all waiters' requests.
        self.w("void __maka_wait_fd(int64_t fd, int64_t events) {\n");
        self.w("    if (!maka_current_fiber || maka_current_fiber == maka_anchor_fiber) {\n");
        self.w("        struct pollfd pfd; pfd.fd = (int)fd; pfd.events = 0; pfd.revents = 0;\n");
        self.w("        if (events & MAKA_EV_READ)  pfd.events |= POLLIN;\n");
        self.w("        if (events & MAKA_EV_WRITE) pfd.events |= POLLOUT;\n");
        self.w("        poll(&pfd, 1, -1);\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("    int ep_events = 0;\n");
        self.w("    if (events & MAKA_EV_READ)  ep_events |= EPOLLIN;\n");
        self.w("    if (events & MAKA_EV_WRITE) ep_events |= EPOLLOUT;\n");
        self.w("    maka_current_fiber->waiting_fd = (int)fd;\n");
        self.w("    maka_current_fiber->waiting_events = ep_events;\n");
        self.w("    maka_current_fiber->state = 5;\n");
        self.w("    maka_current_fiber->next = maka_fd_waiters;\n");
        self.w("    maka_fd_waiters = maka_current_fiber;\n");
        self.w("    __maka_fd_recompute((int)fd);\n");
        self.w("    swapcontext(&maka_current_fiber->ctx, &maka_sched_ctx);\n");
        self.w("}\n");
        // Wall-clock helper used by every timeout primitive below.
        self.w("static int64_t __maka_now_ms(void) {\n");
        self.w("    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);\n");
        self.w("    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;\n");
        self.w("}\n");
        // wait_fd_timeout: yields up to `ms` milliseconds, returns 1 on event,
        // 0 on timeout.  Outside a fiber, falls back to a bounded poll().
        self.w("int64_t __maka_wait_fd_timeout(int64_t fd, int64_t events, int64_t ms) {\n");
        self.w("    if (!maka_current_fiber || maka_current_fiber == maka_anchor_fiber) {\n");
        self.w("        struct pollfd pfd; pfd.fd = (int)fd; pfd.events = 0; pfd.revents = 0;\n");
        self.w("        if (events & MAKA_EV_READ)  pfd.events |= POLLIN;\n");
        self.w("        if (events & MAKA_EV_WRITE) pfd.events |= POLLOUT;\n");
        self.w("        int r = poll(&pfd, 1, (int)ms);\n");
        self.w("        return r > 0 ? 1 : 0;\n");
        self.w("    }\n");
        self.w("    int ep_events = 0;\n");
        self.w("    if (events & MAKA_EV_READ)  ep_events |= EPOLLIN;\n");
        self.w("    if (events & MAKA_EV_WRITE) ep_events |= EPOLLOUT;\n");
        self.w("    maka_current_fiber->waiting_fd = (int)fd;\n");
        self.w("    maka_current_fiber->waiting_events = ep_events;\n");
        self.w("    maka_current_fiber->state = 5;\n");
        self.w("    maka_current_fiber->wait_deadline_ns = __maka_now_ns() + ms * 1000000LL;\n");
        self.w("    maka_current_fiber->wait_timed_out = 0;\n");
        self.w("    maka_current_fiber->next = maka_fd_waiters;\n");
        self.w("    maka_fd_waiters = maka_current_fiber;\n");
        self.w("    __maka_fd_recompute((int)fd);\n");
        self.w("    swapcontext(&maka_current_fiber->ctx, &maka_sched_ctx);\n");
        self.w("    /* Scheduler resumes us either on fd event or on deadline. */\n");
        self.w("    return maka_current_fiber->wait_timed_out ? 0 : 1;\n");
        self.w("}\n");
        self.w("#include <fcntl.h>\n");
        // TCP helpers — defined in an inner block so the sys/socket.h /
        // netinet/in.h symbol pollution (especially the bare `accept`
        // function name) doesn't escape into user code.  We define the
        // runtime functions inside `extern "C"`-equivalent scope and
        // reference them by their __maka_ names from the rest of the file.
        self.w("/* TCP runtime — scoped includes to avoid name clashes. */\n");
        self.w("static inline int64_t __maka_tcp_listen(int64_t port, int64_t backlog);\n");
        self.w("static inline int64_t __maka_udp_open(int64_t port);\n");
        self.w("static inline int64_t __maka_udp_send_v4(int64_t fd, int64_t a, int64_t b, int64_t c, int64_t d, int64_t port, maka_unit* buf, int64_t len);\n");
        self.w("static inline int64_t __maka_udp_recv_async(int64_t fd, maka_unit* buf, int64_t cap);\n");
        self.w("static inline int64_t __maka_signalfd_open(int64_t signum);\n");
        self.w("static inline int64_t __maka_signalfd_recv(int64_t fd);\n");
        self.w("static inline int64_t __maka_timerfd_create(int64_t initial_ns, int64_t interval_ns);\n");
        self.w("static inline int64_t __maka_timerfd_recv(int64_t fd);\n");
        self.w("static inline int64_t __maka_eventfd_create(int64_t initial);\n");
        self.w("static inline int64_t __maka_eventfd_signal(int64_t fd, int64_t n);\n");
        self.w("static inline int64_t __maka_eventfd_recv(int64_t fd);\n");
        self.w("static inline int64_t __maka_inotify_open(void);\n");
        self.w("static inline int64_t __maka_inotify_add(int64_t fd, const char* path, int64_t mask);\n");
        self.w("static inline int64_t __maka_inotify_recv(int64_t fd);\n");
        self.w("static inline int64_t __maka_tcp_accept_async(int64_t listen_fd);\n");
        self.w("static inline int64_t __maka_tcp_connect_v4(int64_t a, int64_t b, int64_t c, int64_t d, int64_t port);\n");
        self.w("static inline int64_t __maka_close_fd(int64_t fd);\n");
        self.w("static inline int64_t __maka_dns_resolve_v4(const char* host);\n");
        self.w("static inline int64_t __maka_file_open(const char* path, int64_t flags, int64_t mode);\n");
        self.w("static inline int64_t __maka_file_read_async(int64_t fd, maka_unit* buf, int64_t cap, int64_t offset);\n");
        self.w("static inline int64_t __maka_file_write_async(int64_t fd, maka_unit* buf, int64_t len, int64_t offset);\n");
        self.w("static inline int64_t __maka_unix_listen(const char* path, int64_t backlog);\n");
        self.w("static inline int64_t __maka_unix_connect(const char* path);\n");
        self.w("static inline int64_t __maka_http_parse(const char* buf, int64_t len);\n");
        self.w("static inline int64_t __maka_http_method_off_g(void);\n");
        self.w("static inline int64_t __maka_http_method_len_g(void);\n");
        self.w("static inline int64_t __maka_http_path_off_g(void);\n");
        self.w("static inline int64_t __maka_http_path_len_g(void);\n");
        self.w("static inline int64_t __maka_http_body_off_g(void);\n");
        self.w("static inline int64_t __maka_http_content_length_g(void);\n");
        self.w("static inline int64_t __maka_pipe_create(void);\n");
        self.w("static inline int64_t __maka_pipe_write_fd(void);\n");
        self.w("static inline maka_unit* __maka_tls_client_new(int64_t fd, const char* hostname);\n");
        self.w("static inline int64_t __maka_tls_handshake(maka_unit* p);\n");
        self.w("static inline int64_t __maka_tls_read(maka_unit* p, maka_unit* buf, int64_t cap);\n");
        self.w("static inline int64_t __maka_tls_write(maka_unit* p, maka_unit* buf, int64_t len);\n");
        self.w("static inline void __maka_tls_close(maka_unit* p);\n");
        self.w("int64_t __maka_set_nonblock(int64_t fd) {\n");
        self.w("    int flags = fcntl((int)fd, F_GETFL, 0);\n");
        self.w("    if (flags < 0) return -1;\n");
        self.w("    return fcntl((int)fd, F_SETFL, flags | O_NONBLOCK) < 0 ? -1 : 0;\n");
        self.w("}\n");
        self.w("int64_t __maka_read_async(int64_t fd, maka_unit* buf, int64_t cap) {\n");
        self.w("    while (1) {\n");
        self.w("        ssize_t n = read((int)fd, (void*)buf, (size_t)cap);\n");
        self.w("        if (n >= 0) return (int64_t)n;\n");
        self.w("        if (errno == EAGAIN || errno == EWOULDBLOCK) {\n");
        self.w("            __maka_wait_fd(fd, MAKA_EV_READ); continue;\n");
        self.w("        }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("int64_t __maka_write_async(int64_t fd, maka_unit* buf, int64_t len) {\n");
        self.w("    int64_t w = 0;\n");
        self.w("    while (w < len) {\n");
        self.w("        ssize_t n = write((int)fd, (const char*)(const void*)buf + w, (size_t)(len - w));\n");
        self.w("        if (n >= 0) { w += n; continue; }\n");
        self.w("        if (errno == EAGAIN || errno == EWOULDBLOCK) {\n");
        self.w("            __maka_wait_fd(fd, MAKA_EV_WRITE); continue;\n");
        self.w("        }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("    return w;\n");
        self.w("}\n");
        // ====================================================================
        // JOB POOL — Chase-Lev work-stealing deques (one per worker thread).
        // ====================================================================
        //
        // Each worker has its own lock-free deque.  The owner pushes new jobs
        // to the bottom (top of stack); pops LIFO from the bottom for cache
        // locality on recently-spawned work.  Other workers ("thieves") steal
        // FIFO from the top of the deque.  Atomic operations on top/bottom
        // keep races correct without locks.
        //
        // Reference: Chase & Lev, "Dynamic Circular Work-Stealing Deque", 2005.
        // We use a fixed-size circular buffer (no growth — overflow falls back
        // to a direct pthread spawn, which preserves correctness with a one-
        // off slowdown).
        self.w("#include <stdatomic.h>\n");
        self.w("#define MAKA_WS_CAP 8192\n");
        self.w("typedef struct { void* code; void* env; Thread* h; } __maka_job_entry_t;\n");
        self.w("typedef struct {\n");
        self.w("    _Atomic int64_t top;\n");
        self.w("    _Atomic int64_t bottom;\n");
        self.w("    __maka_job_entry_t buf[MAKA_WS_CAP];\n");
        self.w("} __maka_ws_deque_t;\n");
        self.w("static long __maka_n_workers = 0;\n");
        self.w("static __maka_ws_deque_t* __maka_ws_deques = NULL;\n");
        self.w("static __thread int __maka_ws_worker_id = -1;\n");
        self.w("static __thread unsigned int __maka_ws_rng = 0;\n");
        self.w("static unsigned int __maka_ws_rand(void) {\n");
        self.w("    /* xorshift32 — fast, no global state contention */\n");
        self.w("    unsigned int x = __maka_ws_rng ? __maka_ws_rng : (unsigned int)(uintptr_t)&__maka_ws_rng;\n");
        self.w("    x ^= x << 13; x ^= x >> 17; x ^= x << 5;\n");
        self.w("    __maka_ws_rng = x;\n");
        self.w("    return x;\n");
        self.w("}\n");
        self.w("static int __maka_ws_push(__maka_ws_deque_t* dq, __maka_job_entry_t item) {\n");
        self.w("    int64_t b = atomic_load_explicit(&dq->bottom, memory_order_relaxed);\n");
        self.w("    int64_t t = atomic_load_explicit(&dq->top, memory_order_acquire);\n");
        self.w("    if (b - t >= MAKA_WS_CAP) return 0; /* full */\n");
        self.w("    dq->buf[b % MAKA_WS_CAP] = item;\n");
        self.w("    atomic_thread_fence(memory_order_release);\n");
        self.w("    atomic_store_explicit(&dq->bottom, b + 1, memory_order_relaxed);\n");
        self.w("    return 1;\n");
        self.w("}\n");
        self.w("static int __maka_ws_pop(__maka_ws_deque_t* dq, __maka_job_entry_t* out) {\n");
        self.w("    int64_t b = atomic_load_explicit(&dq->bottom, memory_order_relaxed) - 1;\n");
        self.w("    atomic_store_explicit(&dq->bottom, b, memory_order_relaxed);\n");
        self.w("    atomic_thread_fence(memory_order_seq_cst);\n");
        self.w("    int64_t t = atomic_load_explicit(&dq->top, memory_order_relaxed);\n");
        self.w("    if (t > b) {\n");
        self.w("        atomic_store_explicit(&dq->bottom, b + 1, memory_order_relaxed);\n");
        self.w("        return 0;\n");
        self.w("    }\n");
        self.w("    *out = dq->buf[b % MAKA_WS_CAP];\n");
        self.w("    if (t == b) {\n");
        self.w("        /* Last item — race with a stealer. */\n");
        self.w("        int64_t expected = t;\n");
        self.w("        int won = atomic_compare_exchange_strong_explicit(\n");
        self.w("            &dq->top, &expected, t + 1,\n");
        self.w("            memory_order_seq_cst, memory_order_relaxed);\n");
        self.w("        atomic_store_explicit(&dq->bottom, b + 1, memory_order_relaxed);\n");
        self.w("        return won;\n");
        self.w("    }\n");
        self.w("    return 1;\n");
        self.w("}\n");
        self.w("static int __maka_ws_steal(__maka_ws_deque_t* dq, __maka_job_entry_t* out) {\n");
        self.w("    int64_t t = atomic_load_explicit(&dq->top, memory_order_acquire);\n");
        self.w("    atomic_thread_fence(memory_order_seq_cst);\n");
        self.w("    int64_t b = atomic_load_explicit(&dq->bottom, memory_order_acquire);\n");
        self.w("    if (t >= b) return 0; /* empty */\n");
        self.w("    *out = dq->buf[t % MAKA_WS_CAP];\n");
        self.w("    int64_t expected = t;\n");
        self.w("    if (!atomic_compare_exchange_strong_explicit(\n");
        self.w("            &dq->top, &expected, t + 1,\n");
        self.w("            memory_order_seq_cst, memory_order_relaxed)) {\n");
        self.w("        return 0; /* raced with someone */\n");
        self.w("    }\n");
        self.w("    return 1;\n");
        self.w("}\n");
        self.w("static void __maka_ws_run(__maka_job_entry_t* item) {\n");
        self.w("    ((void(*)(void*))item->code)(item->env);\n");
        self.w("    pthread_mutex_lock(&item->h->done_mutex);\n");
        self.w("    item->h->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&item->h->done_cond);\n");
        self.w("    pthread_mutex_unlock(&item->h->done_mutex);\n");
        self.w("    free(item->env);\n");
        self.w("}\n");
        self.w("static void* __maka_ws_worker(void* arg) {\n");
        self.w("    int id = (int)(intptr_t)arg;\n");
        self.w("    __maka_ws_worker_id = id;\n");
        self.w("    __maka_ws_deque_t* mine = &__maka_ws_deques[id];\n");
        self.w("    int idle_iters = 0;\n");
        self.w("    while (1) {\n");
        self.w("        __maka_job_entry_t item;\n");
        self.w("        if (__maka_ws_pop(mine, &item)) {\n");
        self.w("            __maka_ws_run(&item);\n");
        self.w("            idle_iters = 0; continue;\n");
        self.w("        }\n");
        self.w("        /* Try to steal from a random victim. */\n");
        self.w("        if (__maka_n_workers > 1) {\n");
        self.w("            int v = (int)(__maka_ws_rand() % (unsigned int)__maka_n_workers);\n");
        self.w("            if (v == id) v = (v + 1) % __maka_n_workers;\n");
        self.w("            if (__maka_ws_steal(&__maka_ws_deques[v], &item)) {\n");
        self.w("                __maka_ws_run(&item);\n");
        self.w("                idle_iters = 0; continue;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        idle_iters++;\n");
        self.w("        if (idle_iters < 100) {\n");
        self.w("            /* Spin briefly — fresh work often arrives soon. */\n");
        self.w("            for (volatile int s = 0; s < 32; s++) {}\n");
        self.w("        } else {\n");
        self.w("            /* Longer idle — sleep instead of burning CPU. */\n");
        self.w("            struct timespec ts = { 0, 200000 /* 200us */ };\n");
        self.w("            nanosleep(&ts, NULL);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static int __maka_job_pool_inited = 0;\n");
        self.w("static void __maka_job_pool_init(void) {\n");
        self.w("    if (__maka_job_pool_inited) return;\n");
        self.w("    __maka_job_pool_inited = 1;\n");
        self.w("    long n = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (n < 1) n = 1;\n");
        self.w("    if (n > 64) n = 64;\n");
        self.w("    __maka_n_workers = n;\n");
        self.w("    __maka_ws_deques = (__maka_ws_deque_t*)calloc((size_t)n, sizeof(__maka_ws_deque_t));\n");
        self.w("    for (long i = 0; i < n; i++) {\n");
        self.w("        atomic_init(&__maka_ws_deques[i].top, 0);\n");
        self.w("        atomic_init(&__maka_ws_deques[i].bottom, 0);\n");
        self.w("    }\n");
        self.w("    for (long i = 0; i < n; i++) {\n");
        self.w("        pthread_t w;\n");
        self.w("        pthread_create(&w, NULL, __maka_ws_worker, (void*)(intptr_t)i);\n");
        self.w("        pthread_detach(w);\n");
        self.w("    }\n");
        self.w("}\n");
        // job() — push to a worker's deque (round-robin from non-worker callers,
        // own-deque from worker callers).
        self.w("static __thread int __maka_job_rr = 0;\n");
        self.w("maka_unit* __maka_spawn_job(void* code, void* env) {\n");
        self.w("    __maka_job_pool_init();\n");
        self.w("    Thread* t = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("    pthread_mutex_init(&t->done_mutex, NULL);\n");
        self.w("    pthread_cond_init(&t->done_cond, NULL);\n");
        self.w("    t->is_job = 1;\n");
        self.w("    __maka_job_entry_t item = { code, env, t };\n");
        self.w("    if (__maka_ws_worker_id >= 0) {\n");
        self.w("        /* Caller is a worker — push to own deque (LIFO, cache-warm). */\n");
        self.w("        if (__maka_ws_push(&__maka_ws_deques[__maka_ws_worker_id], item)) {\n");
        self.w("            return (maka_unit*)t;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    /* Non-worker caller, or own deque full: round-robin push. */\n");
        self.w("    for (long try_count = 0; try_count < __maka_n_workers; try_count++) {\n");
        self.w("        int target = (__maka_job_rr + (int)try_count) % (int)__maka_n_workers;\n");
        self.w("        if (__maka_ws_push(&__maka_ws_deques[target], item)) {\n");
        self.w("            __maka_job_rr = (target + 1) % (int)__maka_n_workers;\n");
        self.w("            return (maka_unit*)t;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    /* All deques full: fall back to a dedicated pthread. */\n");
        self.w("    __maka_handle_args_t* a = (__maka_handle_args_t*)malloc(sizeof(__maka_handle_args_t));\n");
        self.w("    a->code = code; a->env = env; a->h = t;\n");
        self.w("    pthread_create(&t->handle, NULL, __maka_handle_entry, a);\n");
        self.w("    return (maka_unit*)t;\n");
        self.w("}\n");
        // Back-compat: `__maka_spawn` is still emitted, aliased to `spawn` (fiber).
        // Existing tests that build `spawn(closure)` and previously hit the
        // pthread path now hit the smaller-stack fiber path — same observable
        // behavior, less RAM.
        self.w("maka_unit* __maka_spawn(void* code, void* env) { return __maka_spawn_fiber(code, env); }\n");
        // ====================================================================
        // JOIN — block on a single handle and reclaim its result.
        // ====================================================================
        self.w("int64_t __maka_join_result(maka_unit* h) {\n");
        self.w("    Thread* t = (Thread*)h;\n");
        self.w("    /* Fast path: already done. */\n");
        self.w("    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("    int done = t->done_flag;\n");
        self.w("    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("    if (!done) {\n");
        self.w("        /* If the scheduler has ready/sleeping/fd-waiting fibers, drive\n");
        self.w("           it instead of pthread-cond-waiting (which would freeze it). */\n");
        self.w("        if (maka_sched_inited && (maka_ready_head || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            /* Walk ready/sleep queues to find the fiber whose completion is t. */\n");
        self.w("            maka_fiber_t* target = NULL;\n");
        self.w("            for (maka_fiber_t* f = maka_ready_head; f; f = f->next) {\n");
        self.w("                if (f->completion == t) { target = f; break; }\n");
        self.w("            }\n");
        self.w("            if (!target) {\n");
        self.w("                for (maka_fiber_t* f = maka_sleep_head; f; f = f->next) {\n");
        self.w("                    if (f->completion == t) { target = f; break; }\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("            if (!target) {\n");
        self.w("                for (maka_fiber_t* f = maka_fd_waiters; f; f = f->next) {\n");
        self.w("                    if (f->completion == t) { target = f; break; }\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("            if (target) {\n");
        self.w("                maka_join_target = target;\n");
        self.w("                /* Switch into the scheduler; it'll swap back to us when\n");
        self.w("                   target finishes. */\n");
        self.w("                swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("                maka_join_target = NULL;\n");
        self.w("                maka_current_fiber = maka_anchor_fiber;\n");
        self.w("            } else {\n");
        self.w("                /* Not a fiber we own — it's a pthread thread or a job. Drive\n");
        self.w("                   scheduler ALSO so any waiting fibers can still progress\n");
        self.w("                   while we periodically poll the foreign handle. */\n");
        self.w("                while (1) {\n");
        self.w("                    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("                    int d = t->done_flag;\n");
        self.w("                    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("                    if (d) break;\n");
        self.w("                    if (maka_ready_head || maka_sleep_head || maka_fd_waiters) {\n");
        self.w("                        maka_join_target = NULL;\n");
        self.w("                        swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("                        maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                    } else {\n");
        self.w("                        struct timespec ts = { 0, 500000 /* 0.5ms */ };\n");
        self.w("                        nanosleep(&ts, NULL);\n");
        self.w("                    }\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("        } else {\n");
        self.w("            /* No scheduler activity: classic cond_wait. */\n");
        self.w("            pthread_mutex_lock(&t->done_mutex);\n");
        self.w("            while (!t->done_flag) {\n");
        self.w("                pthread_cond_wait(&t->done_cond, &t->done_mutex);\n");
        self.w("            }\n");
        self.w("            pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    int64_t r = t->result;\n");
        self.w("    /* Only pthread-backed handles need pthread_join. */\n");
        self.w("    if (!t->is_job && !t->is_fiber) { pthread_join(t->handle, NULL); }\n");
        self.w("    pthread_mutex_destroy(&t->done_mutex);\n");
        self.w("    pthread_cond_destroy(&t->done_cond);\n");
        self.w("    free(t);\n");
        self.w("    return r;\n");
        self.w("}\n");
        self.w("void __maka_join(maka_unit* h) { (void)__maka_join_result(h); }\n");
        // detach(*Thread) — caller opts out of join.  If the fiber/thread has
        // already finished, reap now; otherwise mark detached so scheduler
        // auto-reaps on natural completion.  Calling join on a detached
        // handle is undefined behavior (handle may already be freed).
        self.w("void __maka_detach(maka_unit* h) {\n");
        self.w("    Thread* t = (Thread*)h;\n");
        self.w("    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("    int done = t->done_flag;\n");
        self.w("    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("    if (done) {\n");
        self.w("        if (!t->is_job && !t->is_fiber) { pthread_join(t->handle, NULL); }\n");
        self.w("        pthread_mutex_destroy(&t->done_mutex);\n");
        self.w("        pthread_cond_destroy(&t->done_cond);\n");
        self.w("        free(t);\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("    atomic_store(&t->detached, 1);\n");
        // For thread tier, also pthread_detach so the OS thread reaps itself.
        self.w("    if (!t->is_job && !t->is_fiber) pthread_detach(t->handle);\n");
        self.w("}\n");
        // ====================================================================
        // try_join(h, &out) -> 1 if done (and writes result), 0 if still running.
        // Never blocks, never reaps a non-done handle.  When done, reclaims
        // the handle exactly once (same as join), so a try_join+true MUST
        // NOT be followed by another join on the same handle.
        // ====================================================================
        self.w("int64_t __maka_try_join(maka_unit* h, int64_t* out_result) {\n");
        self.w("    Thread* t = (Thread*)h;\n");
        self.w("    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("    int done = t->done_flag;\n");
        self.w("    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("    if (!done) return 0;\n");
        self.w("    if (out_result) *out_result = t->result;\n");
        self.w("    if (!t->is_job && !t->is_fiber) { pthread_join(t->handle, NULL); }\n");
        self.w("    pthread_mutex_destroy(&t->done_mutex);\n");
        self.w("    pthread_cond_destroy(&t->done_cond);\n");
        self.w("    free(t);\n");
        self.w("    return 1;\n");
        self.w("}\n");
        // ====================================================================
        // join_timeout(h, ms, &out) -> 1 if done within deadline, 0 on timeout.
        // Deadline driving cooperates with the fiber scheduler: yields to it
        // while the timeout has not expired.
        // ====================================================================
        self.w("int64_t __maka_join_timeout(maka_unit* h, int64_t ms, int64_t* out_result) {\n");
        self.w("    int64_t deadline_ms = __maka_now_ms() + ms;\n");
        self.w("    int64_t deadline_ns = __maka_now_ns() + ms * 1000000LL;\n");
        self.w("    Thread* t = (Thread*)h;\n");
        self.w("    int64_t prev_anchor_deadline = maka_anchor_deadline_ns;\n");
        self.w("    maka_anchor_deadline_ns = deadline_ns;\n");
        self.w("    while (1) {\n");
        self.w("        pthread_mutex_lock(&t->done_mutex);\n");
        self.w("        int d = t->done_flag;\n");
        self.w("        pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("        if (d) {\n");
        self.w("            maka_anchor_deadline_ns = prev_anchor_deadline;\n");
        self.w("            if (out_result) *out_result = t->result;\n");
        self.w("            if (!t->is_job && !t->is_fiber) { pthread_join(t->handle, NULL); }\n");
        self.w("            pthread_mutex_destroy(&t->done_mutex);\n");
        self.w("            pthread_cond_destroy(&t->done_cond);\n");
        self.w("            free(t);\n");
        self.w("            return 1;\n");
        self.w("        }\n");
        self.w("        int64_t now = __maka_now_ms();\n");
        self.w("        if (now >= deadline_ms) {\n");
        self.w("            maka_anchor_deadline_ns = prev_anchor_deadline;\n");
        self.w("            return 0;\n");
        self.w("        }\n");
        self.w("        if (maka_sched_inited && (maka_ready_head || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            int64_t rem_ms = deadline_ms - now;\n");
        self.w("            if (rem_ms > 5) rem_ms = 5;\n");
        self.w("            struct timespec ts = { 0, rem_ms * 1000000LL };\n");
        self.w("            nanosleep(&ts, NULL);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        // ====================================================================
        // cancel(h) — user-callable cancellation.  Behavior by tier:
        //   fiber: walk ready / sleep / fd-waiters; remove without resuming;
        //          free slab + handle.
        //   thread: pthread_cancel + pthread_join, then free handle.
        //   job: not supported — jobs run to completion, this is a no-op
        //        with done_flag flip so subsequent join doesn't hang.  In
        //        practice users shouldn't expect job cancellation.
        // ====================================================================
        self.w("void __maka_cancel(maka_unit* h) {\n");
        self.w("    Thread* t = (Thread*)h;\n");
        self.w("    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("    int already_done = t->done_flag;\n");
        self.w("    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("    if (already_done) { (void)__maka_join_result(h); return; }\n");
        self.w("    if (t->is_fiber) {\n");
        self.w("        maka_fiber_t** prev = &maka_ready_head;\n");
        self.w("        maka_fiber_t* found = NULL;\n");
        self.w("        while (*prev) {\n");
        self.w("            if ((*prev)->completion == t) {\n");
        self.w("                found = *prev; *prev = found->next;\n");
        self.w("                if (maka_ready_tail == found) maka_ready_tail = NULL;\n");
        self.w("                break;\n");
        self.w("            }\n");
        self.w("            prev = &(*prev)->next;\n");
        self.w("        }\n");
        self.w("        if (!found) {\n");
        self.w("            prev = &maka_sleep_head;\n");
        self.w("            while (*prev) {\n");
        self.w("                if ((*prev)->completion == t) { found = *prev; *prev = found->next; break; }\n");
        self.w("                prev = &(*prev)->next;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        int cancelled_fd = -1;\n");
        self.w("        if (!found) {\n");
        self.w("            prev = &maka_fd_waiters;\n");
        self.w("            while (*prev) {\n");
        self.w("                if ((*prev)->completion == t) {\n");
        self.w("                    found = *prev; *prev = found->next;\n");
        self.w("                    cancelled_fd = found->waiting_fd;\n");
        self.w("                    break;\n");
        self.w("                }\n");
        self.w("                prev = &(*prev)->next;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        if (found) { free(found->entry_env); __maka_slab_free(found->slab); free(found); }\n");
        self.w("        if (cancelled_fd >= 0) __maka_fd_recompute(cancelled_fd);\n");
        self.w("    } else if (!t->is_job) {\n");
        self.w("        pthread_cancel(t->handle);\n");
        self.w("        pthread_join(t->handle, NULL);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_destroy(&t->done_mutex);\n");
        self.w("    pthread_cond_destroy(&t->done_cond);\n");
        self.w("    free(t);\n");
        self.w("}\n");
        // ====================================================================
        // join(&[]Handle) -> [N] of results.  Handles must all be Thread*.
        // Sequentially joins each (since they're already running concurrently).
        // Caller is responsible for the result-slice memory.
        // ====================================================================
        self.w("void __maka_join_all_i64(maka_unit** handles, int64_t n, int64_t* out) {\n");
        self.w("    for (int64_t i = 0; i < n; i++) {\n");
        self.w("        int64_t r = __maka_join_result(handles[i]);\n");
        self.w("        if (out) out[i] = r;\n");
        self.w("    }\n");
        self.w("}\n");
        // ====================================================================
        // select(&[]Handle) -> first ready, cancel the rest.
        // Implemented by polling done flags + cancelling losers via pthread_cancel.
        // ====================================================================
        self.w("static int __maka_find_winner(maka_unit** handles, int64_t n) {\n");
        self.w("    for (int64_t i = 0; i < n; i++) {\n");
        self.w("        Thread* t = (Thread*)handles[i];\n");
        self.w("        pthread_mutex_lock(&t->done_mutex);\n");
        self.w("        int done = t->done_flag;\n");
        self.w("        pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("        if (done) return (int)i;\n");
        self.w("    }\n");
        self.w("    return -1;\n");
        self.w("}\n");
        self.w("int64_t __maka_select_first_i64(maka_unit** handles, int64_t n, int64_t* out_index) {\n");
        self.w("    while (1) {\n");
        self.w("        int i = __maka_find_winner(handles, n);\n");
        self.w("        if (i >= 0) {\n");
        self.w("            Thread* t = (Thread*)handles[i];\n");
        self.w("            int64_t r = t->result;\n");
        self.w("            if (out_index) *out_index = (int64_t)i;\n");
        self.w("            /* Cancel losers and reap.  Fibers: walk the scheduler queues and\n");
        self.w("               remove without resuming.  pthread threads: pthread_cancel +\n");
        self.w("               pthread_join.  Jobs: just wait — cancellation isn't supported. */\n");
        self.w("            for (int64_t j = 0; j < n; j++) {\n");
        self.w("                if (j == i) continue;\n");
        self.w("                Thread* l = (Thread*)handles[j];\n");
        self.w("                if (l->is_fiber) {\n");
        self.w("                    /* Find and remove the fiber from ready/sleep. */\n");
        self.w("                    maka_fiber_t** prev = &maka_ready_head;\n");
        self.w("                    maka_fiber_t* found = NULL;\n");
        self.w("                    while (*prev) {\n");
        self.w("                        if ((*prev)->completion == l) {\n");
        self.w("                            found = *prev; *prev = found->next;\n");
        self.w("                            if (maka_ready_tail == found) maka_ready_tail = NULL;\n");
        self.w("                            break;\n");
        self.w("                        }\n");
        self.w("                        prev = &(*prev)->next;\n");
        self.w("                    }\n");
        self.w("                    if (!found) {\n");
        self.w("                        prev = &maka_sleep_head;\n");
        self.w("                        while (*prev) {\n");
        self.w("                            if ((*prev)->completion == l) {\n");
        self.w("                                found = *prev; *prev = found->next;\n");
        self.w("                                break;\n");
        self.w("                            }\n");
        self.w("                            prev = &(*prev)->next;\n");
        self.w("                        }\n");
        self.w("                    }\n");
        self.w("                    int loser_fd = -1;\n");
        self.w("                    if (!found) {\n");
        self.w("                        /* Parked in wait_fd — also remove from epoll. */\n");
        self.w("                        prev = &maka_fd_waiters;\n");
        self.w("                        while (*prev) {\n");
        self.w("                            if ((*prev)->completion == l) {\n");
        self.w("                                found = *prev; *prev = found->next;\n");
        self.w("                                loser_fd = found->waiting_fd;\n");
        self.w("                                break;\n");
        self.w("                            }\n");
        self.w("                            prev = &(*prev)->next;\n");
        self.w("                        }\n");
        self.w("                    }\n");
        self.w("                    if (found) { __maka_slab_free(found->slab); free(found); }\n");
        self.w("                    if (loser_fd >= 0) __maka_fd_recompute(loser_fd);\n");
        self.w("                    /* Mark handle done so resource cleanup proceeds. */\n");
        self.w("                    pthread_mutex_lock(&l->done_mutex);\n");
        self.w("                    l->done_flag = 1;\n");
        self.w("                    pthread_mutex_unlock(&l->done_mutex);\n");
        self.w("                    pthread_mutex_destroy(&l->done_mutex);\n");
        self.w("                    pthread_cond_destroy(&l->done_cond);\n");
        self.w("                    free(l);\n");
        self.w("                } else if (!l->is_job) {\n");
        self.w("                    pthread_cancel(l->handle);\n");
        self.w("                    pthread_join(l->handle, NULL);\n");
        self.w("                    pthread_mutex_destroy(&l->done_mutex);\n");
        self.w("                    pthread_cond_destroy(&l->done_cond);\n");
        self.w("                    free(l);\n");
        self.w("                } else {\n");
        self.w("                    (void)__maka_join_result(handles[j]);\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("            /* Reap the winner. */\n");
        self.w("            (void)__maka_join_result(handles[i]);\n");
        self.w("            return r;\n");
        self.w("        }\n");
        self.w("        /* No winner yet — drive scheduler so fibers can progress. */\n");
        self.w("        if (maka_sched_inited && (maka_ready_head || maka_sleep_head)) {\n");
        self.w("            maka_anchor_wake_on_finish = 1;\n");
        self.w("            swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_anchor_wake_on_finish = 0;\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            struct timespec ts = { 0, 500000 };\n");
        self.w("            nanosleep(&ts, NULL);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        // ====================================================================
        // select_timeout(handles, n, ms) -> winner result, or -1 on timeout.
        // out_index gets the winning index, or -1 on timeout.  Losers are
        // cancelled identically to plain select().
        // ====================================================================
        self.w("int64_t __maka_select_timeout_i64(maka_unit** handles, int64_t n, int64_t ms, int64_t* out_index) {\n");
        self.w("    int64_t deadline_ms = __maka_now_ms() + ms;\n");
        self.w("    int64_t deadline_ns = __maka_now_ns() + ms * 1000000LL;\n");
        self.w("    int64_t prev_anchor_deadline = maka_anchor_deadline_ns;\n");
        self.w("    maka_anchor_deadline_ns = deadline_ns;\n");
        self.w("    while (1) {\n");
        self.w("        int i = __maka_find_winner(handles, n);\n");
        self.w("        if (i >= 0) {\n");
        self.w("            maka_anchor_deadline_ns = prev_anchor_deadline;\n");
        self.w("            Thread* t = (Thread*)handles[i];\n");
        self.w("            int64_t r = t->result;\n");
        self.w("            if (out_index) *out_index = (int64_t)i;\n");
        self.w("            for (int64_t j = 0; j < n; j++) {\n");
        self.w("                if (j == i) continue;\n");
        self.w("                __maka_cancel(handles[j]);\n");
        self.w("            }\n");
        self.w("            (void)__maka_join_result(handles[i]);\n");
        self.w("            return r;\n");
        self.w("        }\n");
        self.w("        if (__maka_now_ms() >= deadline_ms) {\n");
        self.w("            maka_anchor_deadline_ns = prev_anchor_deadline;\n");
        self.w("            if (out_index) *out_index = -1;\n");
        self.w("            return -1;\n");
        self.w("        }\n");
        self.w("        if (maka_sched_inited && (maka_ready_head || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            maka_anchor_wake_on_finish = 1;\n");
        self.w("            swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_anchor_wake_on_finish = 0;\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            int64_t rem_ms = deadline_ms - __maka_now_ms();\n");
        self.w("            if (rem_ms > 5) rem_ms = 5;\n");
        self.w("            struct timespec ts = { 0, rem_ms * 1000000LL };\n");
        self.w("            nanosleep(&ts, NULL);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        // ====================================================================
        // SLEEP — sleeps the current thread/fiber for the requested duration.
        // Maka exposes this via `sleep_ms(int)` in stdlib.async.
        // ====================================================================
        self.w("void __maka_sleep_ns(int64_t nanos) {\n");
        self.w("    /* Inside a cooperative fiber (not the anchor), yield with timer. */\n");
        self.w("    if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("        __maka_sleep_fiber(nanos);\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("    /* On the anchor: if other fibers exist, drive the scheduler\n");
        self.w("       in short bursts instead of blocking outright.  This lets\n");
        self.w("       fibers progress while the caller is `sleeping`. */\n");
        self.w("    int64_t deadline = __maka_now_ns() + nanos;\n");
        self.w("    int64_t prev_anchor_deadline = maka_anchor_deadline_ns;\n");
        self.w("    maka_anchor_deadline_ns = deadline;\n");
        self.w("    while (1) {\n");
        self.w("        int64_t now = __maka_now_ns();\n");
        self.w("        if (now >= deadline) { maka_anchor_deadline_ns = prev_anchor_deadline; return; }\n");
        self.w("        if (maka_sched_inited && (maka_ready_head || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            int64_t rem = deadline - now;\n");
        self.w("            struct timespec ts;\n");
        self.w("            ts.tv_sec = rem / 1000000000LL;\n");
        self.w("            ts.tv_nsec = rem % 1000000000LL;\n");
        self.w("            nanosleep(&ts, NULL);\n");
        self.w("            maka_anchor_deadline_ns = prev_anchor_deadline;\n");
        self.w("            return;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        // ====================================================================
        // par_for_range — chunks an integer range across the job pool.
        // Each chunk-job runs `code(env, i)` for every i in [chunk_start,
        // chunk_end).  The signature accepts a unit(int) closure passed as
        // (code, env) — the same fat-callable shape `spawn` already uses.
        // ====================================================================
        self.w("typedef struct {\n");
        self.w("    int64_t start, end;\n");
        self.w("    void (*code)(void*, int64_t);\n");
        self.w("    void* env;\n");
        self.w("    Thread* completion;\n");
        self.w("} __maka_par_chunk_t;\n");
        self.w("static void* __maka_par_chunk_entry(void* arg) {\n");
        self.w("    __maka_par_chunk_t* c = (__maka_par_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->start; i < c->end; i++) {\n");
        self.w("        c->code(c->env, i);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    free(c);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        // par_reduce_int: divide range into chunks, each chunk-job folds its
        // sub-range into a partial result, then combine partials into the final
        // accumulator (in caller after barrier).
        self.w("typedef struct {\n");
        self.w("    int64_t start, end;\n");
        self.w("    int64_t (*combine)(void*, int64_t, int64_t);\n");
        self.w("    void* env;\n");
        self.w("    int64_t partial;\n");
        self.w("    Thread* completion;\n");
        self.w("} __maka_reduce_chunk_t;\n");
        self.w("static void* __maka_reduce_chunk_entry(void* arg) {\n");
        self.w("    __maka_reduce_chunk_t* c = (__maka_reduce_chunk_t*)arg;\n");
        self.w("    int64_t acc = 0;\n");
        self.w("    for (int64_t i = c->start; i < c->end; i++) {\n");
        self.w("        acc = c->combine(c->env, acc, i);\n");
        self.w("    }\n");
        self.w("    c->partial = acc;\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("int64_t __maka_par_reduce_int(int64_t start, int64_t end, int64_t init, void* code, void* env) {\n");
        self.w("    if (start >= end) return init;\n");
        self.w("    int64_t total = end - start;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1;\n");
        self.w("    if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > total) chunks = total;\n");
        self.w("    int64_t per = (total + chunks - 1) / chunks;\n");
        self.w("    Thread** handles = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_reduce_chunk_t** chunks_arr = (__maka_reduce_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL);\n");
        self.w("        pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_reduce_chunk_t* ch = (__maka_reduce_chunk_t*)malloc(sizeof(__maka_reduce_chunk_t));\n");
        self.w("        ch->start = start + c * per;\n");
        self.w("        ch->end = ch->start + per; if (ch->end > end) ch->end = end;\n");
        self.w("        ch->combine = (int64_t(*)(void*, int64_t, int64_t))code;\n");
        self.w("        ch->env = env;\n");
        self.w("        ch->partial = 0;\n");
        self.w("        ch->completion = th;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_reduce_chunk_entry, ch);\n");
        self.w("        handles[c] = th; chunks_arr[c] = ch;\n");
        self.w("    }\n");
        self.w("    int64_t acc = init;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        pthread_join(handles[c]->handle, NULL);\n");
        self.w("        /* Combine partial using the same combine function with two\n");
        self.w("           int args -- works for sum/max/min where the function\n");
        self.w("           is associative on integers regardless of the\n");
        self.w("           \"index\" position. */\n");
        self.w("        acc = chunks_arr[c]->combine(env, acc, chunks_arr[c]->partial);\n");
        self.w("        pthread_mutex_destroy(&handles[c]->done_mutex);\n");
        self.w("        pthread_cond_destroy(&handles[c]->done_cond);\n");
        self.w("        free(handles[c]);\n");
        self.w("        free(chunks_arr[c]);\n");
        self.w("    }\n");
        self.w("    free(handles); free(chunks_arr);\n");
        self.w("    return acc;\n");
        self.w("}\n");
        // par_map_int: chunked parallel map of f(i) for i in [start, end).
        // Returns a freshly-allocated Slice_maka_int of length (end-start).
        // The caller owns the buffer (it's a leaked slice — Maka's slice type
        // doesn't carry an owner).  For typical use the caller stashes it
        // somewhere and uses it; advanced users can free via `free(s.ptr)`.
        self.w("typedef struct {\n");
        self.w("    int64_t start, end;\n");
        self.w("    int64_t (*fn)(void*, int64_t);\n");
        self.w("    void* env;\n");
        self.w("    int64_t* out;        /* shared output buffer */\n");
        self.w("    Thread* completion;\n");
        self.w("} __maka_map_chunk_t;\n");
        self.w("static void* __maka_map_chunk_entry(void* arg) {\n");
        self.w("    __maka_map_chunk_t* c = (__maka_map_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->start; i < c->end; i++) {\n");
        self.w("        c->out[i] = c->fn(c->env, i);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    free(c);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("Slice_maka_int __maka_par_map_int(int64_t start, int64_t end, void* code, void* env) {\n");
        self.w("    if (start >= end) {\n");
        self.w("        Slice_maka_int empty = { .ptr = NULL, .len = 0 };\n");
        self.w("        return empty;\n");
        self.w("    }\n");
        self.w("    int64_t total = end - start;\n");
        self.w("    int64_t* out = (int64_t*)malloc(sizeof(int64_t) * (size_t)total);\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1;\n");
        self.w("    if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > total) chunks = total;\n");
        self.w("    int64_t per = (total + chunks - 1) / chunks;\n");
        self.w("    Thread** handles = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL);\n");
        self.w("        pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_map_chunk_t* ch = (__maka_map_chunk_t*)malloc(sizeof(__maka_map_chunk_t));\n");
        self.w("        ch->start = start + c * per;\n");
        self.w("        ch->end = ch->start + per; if (ch->end > end) ch->end = end;\n");
        self.w("        ch->fn = (int64_t(*)(void*, int64_t))code;\n");
        self.w("        ch->env = env;\n");
        self.w("        ch->out = out - start; /* offset so ch->out[i] writes the right slot */\n");
        self.w("        ch->completion = th;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_map_chunk_entry, ch);\n");
        self.w("        handles[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        pthread_join(handles[c]->handle, NULL);\n");
        self.w("        pthread_mutex_destroy(&handles[c]->done_mutex);\n");
        self.w("        pthread_cond_destroy(&handles[c]->done_cond);\n");
        self.w("        free(handles[c]);\n");
        self.w("    }\n");
        self.w("    free(handles);\n");
        self.w("    Slice_maka_int res = { .ptr = out, .len = total };\n");
        self.w("    return res;\n");
        self.w("}\n");
        self.w("void __maka_par_for_range(int64_t start, int64_t end, void* code, void* env) {\n");
        self.w("    if (start >= end) return;\n");
        self.w("    int64_t total = end - start;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1;\n");
        self.w("    if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > total) chunks = total;\n");
        self.w("    int64_t per = (total + chunks - 1) / chunks;\n");
        self.w("    Thread** handles = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL);\n");
        self.w("        pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_par_chunk_t* ch = (__maka_par_chunk_t*)malloc(sizeof(__maka_par_chunk_t));\n");
        self.w("        ch->start = start + c * per;\n");
        self.w("        ch->end = ch->start + per; if (ch->end > end) ch->end = end;\n");
        self.w("        ch->code = (void(*)(void*, int64_t))code;\n");
        self.w("        ch->env = env;\n");
        self.w("        ch->completion = th;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_chunk_entry, ch);\n");
        self.w("        handles[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        (void)__maka_join_result((maka_unit*)handles[c]);\n");
        self.w("    }\n");
        self.w("    free(handles);\n");
        self.w("}\n");
        // ====================================================================
        // Fiber-aware sync primitives (Mutex, WaitGroup, Once).
        // Emitted AFTER the fiber/scheduler infrastructure because these need
        // maka_fiber_t fields and __maka_ready_enqueue in scope.
        // ====================================================================
        // Fiber-aware mutex.
        self.w("typedef struct {\n");
        self.w("    _Atomic int locked;\n");
        self.w("    pthread_mutex_t kw_mu;\n");
        self.w("    pthread_cond_t  kw_cv;\n");
        self.w("    maka_fiber_t*   fiber_waiters;\n");
        self.w("} maka_fmutex_t;\n");
        self.w("maka_unit* maka_fmutex_new(void) {\n");
        self.w("    maka_fmutex_t* m = (maka_fmutex_t*)calloc(1, sizeof(maka_fmutex_t));\n");
        self.w("    atomic_init(&m->locked, 0);\n");
        self.w("    pthread_mutex_init(&m->kw_mu, NULL);\n");
        self.w("    pthread_cond_init(&m->kw_cv, NULL);\n");
        self.w("    return (maka_unit*)m;\n");
        self.w("}\n");
        self.w("void maka_fmutex_lock(maka_unit* p) {\n");
        self.w("    maka_fmutex_t* m = (maka_fmutex_t*)p;\n");
        self.w("    while (1) {\n");
        self.w("        int expected = 0;\n");
        self.w("        if (atomic_compare_exchange_strong(&m->locked, &expected, 1)) return;\n");
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("            maka_fiber_t* me = maka_current_fiber;\n");
        self.w("            me->next_waiter = m->fiber_waiters;\n");
        self.w("            m->fiber_waiters = me;\n");
        self.w("            me->state = 2;\n");
        self.w("            swapcontext(&me->ctx, &maka_sched_ctx);\n");
        self.w("        } else if (maka_sched_inited && (maka_ready_head || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            pthread_mutex_lock(&m->kw_mu);\n");
        self.w("            while (atomic_load(&m->locked) != 0) pthread_cond_wait(&m->kw_cv, &m->kw_mu);\n");
        self.w("            pthread_mutex_unlock(&m->kw_mu);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("void maka_fmutex_unlock(maka_unit* p) {\n");
        self.w("    maka_fmutex_t* m = (maka_fmutex_t*)p;\n");
        self.w("    atomic_store(&m->locked, 0);\n");
        self.w("    if (m->fiber_waiters) {\n");
        self.w("        maka_fiber_t* w = m->fiber_waiters; m->fiber_waiters = w->next_waiter; w->next_waiter = NULL;\n");
        self.w("        __maka_ready_enqueue(w);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&m->kw_mu);\n");
        self.w("    pthread_cond_signal(&m->kw_cv);\n");
        self.w("    pthread_mutex_unlock(&m->kw_mu);\n");
        self.w("}\n");
        self.w("void maka_fmutex_destroy(maka_unit* p) {\n");
        self.w("    maka_fmutex_t* m = (maka_fmutex_t*)p;\n");
        self.w("    pthread_mutex_destroy(&m->kw_mu); pthread_cond_destroy(&m->kw_cv);\n");
        self.w("    free(m);\n");
        self.w("}\n");
        // WaitGroup.
        self.w("typedef struct {\n");
        self.w("    _Atomic int64_t count;\n");
        self.w("    pthread_mutex_t kw_mu;\n");
        self.w("    pthread_cond_t  kw_cv;\n");
        self.w("    maka_fiber_t*   fiber_waiters;\n");
        self.w("} maka_wg_t;\n");
        self.w("maka_unit* maka_wg_new(void) {\n");
        self.w("    maka_wg_t* w = (maka_wg_t*)calloc(1, sizeof(maka_wg_t));\n");
        self.w("    atomic_init(&w->count, 0);\n");
        self.w("    pthread_mutex_init(&w->kw_mu, NULL);\n");
        self.w("    pthread_cond_init(&w->kw_cv, NULL);\n");
        self.w("    return (maka_unit*)w;\n");
        self.w("}\n");
        self.w("void maka_wg_add(maka_unit* p, int64_t n) {\n");
        self.w("    atomic_fetch_add(&((maka_wg_t*)p)->count, n);\n");
        self.w("}\n");
        self.w("void maka_wg_done(maka_unit* p) {\n");
        self.w("    maka_wg_t* w = (maka_wg_t*)p;\n");
        self.w("    int64_t prev = atomic_fetch_sub(&w->count, 1);\n");
        self.w("    if (prev <= 1) {\n");
        self.w("        while (w->fiber_waiters) {\n");
        self.w("            maka_fiber_t* f = w->fiber_waiters; w->fiber_waiters = f->next_waiter; f->next_waiter = NULL;\n");
        self.w("            __maka_ready_enqueue(f);\n");
        self.w("        }\n");
        self.w("        pthread_mutex_lock(&w->kw_mu);\n");
        self.w("        pthread_cond_broadcast(&w->kw_cv);\n");
        self.w("        pthread_mutex_unlock(&w->kw_mu);\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("void maka_wg_wait(maka_unit* p) {\n");
        self.w("    maka_wg_t* w = (maka_wg_t*)p;\n");
        self.w("    while (atomic_load(&w->count) > 0) {\n");
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("            maka_fiber_t* me = maka_current_fiber;\n");
        self.w("            me->next_waiter = w->fiber_waiters; w->fiber_waiters = me;\n");
        self.w("            me->state = 2;\n");
        self.w("            swapcontext(&me->ctx, &maka_sched_ctx);\n");
        self.w("        } else if (maka_sched_inited && (maka_ready_head || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            /* On the anchor: drive the scheduler so fibers can complete\n");
        self.w("               and call wg_done.  pthread_cond_wait would freeze them. */\n");
        self.w("            swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            pthread_mutex_lock(&w->kw_mu);\n");
        self.w("            while (atomic_load(&w->count) > 0) pthread_cond_wait(&w->kw_cv, &w->kw_mu);\n");
        self.w("            pthread_mutex_unlock(&w->kw_mu);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("void maka_wg_destroy(maka_unit* p) {\n");
        self.w("    maka_wg_t* w = (maka_wg_t*)p;\n");
        self.w("    pthread_mutex_destroy(&w->kw_mu); pthread_cond_destroy(&w->kw_cv);\n");
        self.w("    free(w);\n");
        self.w("}\n");
        // Once.
        self.w("typedef struct {\n");
        self.w("    _Atomic int state;\n");
        self.w("    pthread_mutex_t mu;\n");
        self.w("    pthread_cond_t  cv;\n");
        self.w("} maka_once_t;\n");
        self.w("maka_unit* maka_once_new(void) {\n");
        self.w("    maka_once_t* o = (maka_once_t*)calloc(1, sizeof(maka_once_t));\n");
        self.w("    atomic_init(&o->state, 0);\n");
        self.w("    pthread_mutex_init(&o->mu, NULL);\n");
        self.w("    pthread_cond_init(&o->cv, NULL);\n");
        self.w("    return (maka_unit*)o;\n");
        self.w("}\n");
        self.w("void maka_once_do(maka_unit* p, void* code, void* env) {\n");
        self.w("    maka_once_t* o = (maka_once_t*)p;\n");
        self.w("    int expected = 0;\n");
        self.w("    if (atomic_compare_exchange_strong(&o->state, &expected, 1)) {\n");
        self.w("        ((void(*)(void*))code)(env);\n");
        self.w("        pthread_mutex_lock(&o->mu);\n");
        self.w("        atomic_store(&o->state, 2);\n");
        self.w("        pthread_cond_broadcast(&o->cv);\n");
        self.w("        pthread_mutex_unlock(&o->mu);\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&o->mu);\n");
        self.w("    while (atomic_load(&o->state) != 2) pthread_cond_wait(&o->cv, &o->mu);\n");
        self.w("    pthread_mutex_unlock(&o->mu);\n");
        self.w("}\n");
        self.w("void maka_once_destroy(maka_unit* p) {\n");
        self.w("    maka_once_t* o = (maka_once_t*)p;\n");
        self.w("    pthread_mutex_destroy(&o->mu); pthread_cond_destroy(&o->cv);\n");
        self.w("    free(o);\n");
        self.w("}\n");
        // Fiber-aware byte-channel recv.  On the anchor with pending fiber
        // work, drive the scheduler in short bursts so other fibers can run
        // (and possibly post to the channel) instead of blocking the worker.
        self.w("void maka_chan_bytes_recv(maka_unit* p, maka_unit* dst) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)p;\n");
        self.w("    while (1) {\n");
        self.w("        pthread_mutex_lock(&c->m);\n");
        self.w("        maka_bnode_t* n = c->head;\n");
        self.w("        if (n) {\n");
        self.w("            c->head = n->next; if (!c->head) c->tail = NULL;\n");
        self.w("            c->count--;\n");
        self.w("            memcpy((void*)dst, n->data, (size_t)c->item_size);\n");
        self.w("            pthread_mutex_unlock(&c->m);\n");
        self.w("            free(n);\n");
        self.w("            return;\n");
        self.w("        }\n");
        self.w("        if (c->closed) {\n");
        self.w("            memset((void*)dst, 0, (size_t)c->item_size);\n");
        self.w("            pthread_mutex_unlock(&c->m);\n");
        self.w("            return;\n");
        self.w("        }\n");
        self.w("        pthread_mutex_unlock(&c->m);\n");
        self.w("        if (maka_sched_inited && maka_current_fiber == maka_anchor_fiber\n");
        self.w("            && (maka_ready_head || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            swapcontext(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        pthread_mutex_lock(&c->m);\n");
        self.w("        while (!c->head && !c->closed) pthread_cond_wait(&c->c, &c->m);\n");
        self.w("        pthread_mutex_unlock(&c->m);\n");
        self.w("    }\n");
        self.w("}\n");
        // ====================================================================
        // Slice-based data-parallel primitives (par_for_each / par_filter /
        // par_scan / par_map_slice / par_reduce_slice).  Each works over a
        // `[]int` slice of length `n` rather than an integer range.
        // ====================================================================
        self.w("typedef struct {\n");
        self.w("    int64_t i_start, i_end;\n");
        self.w("    int64_t* in_ptr;\n");
        self.w("    void* env;\n");
        self.w("    Thread* completion;\n");
        self.w("    union { void (*body)(void*, int64_t);\n");
        self.w("            int64_t (*fn)(void*, int64_t);\n");
        self.w("            int (*pred)(void*, int64_t);\n");
        self.w("            int64_t (*combine)(void*, int64_t, int64_t); } code;\n");
        self.w("    int64_t init;     /* reduce/scan seed */\n");
        self.w("    int64_t out_acc;  /* reduce result, scan offset */\n");
        self.w("    int64_t* out_ptr; /* map/scan/filter output */\n");
        self.w("    int64_t  out_len; /* filter: count written */\n");
        self.w("} __maka_slice_chunk_t;\n");

        // par_for_each: body(env, elem) for each elem
        self.w("static void* __maka_par_each_entry(void* arg) {\n");
        self.w("    __maka_slice_chunk_t* c = (__maka_slice_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) c->code.body(c->env, c->in_ptr[i]);\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    free(c); return NULL;\n");
        self.w("}\n");
        self.w("void __maka_par_for_each_i64(Slice_maka_int s, void* code, void* env) {\n");
        self.w("    if (s.len <= 0) return;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.body = (void(*)(void*, int64_t))code;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_each_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) (void)__maka_join_result((maka_unit*)hs[c]);\n");
        self.w("    free(hs);\n");
        self.w("}\n");

        // par_map_int over slice: body(env, elem) -> int.  Output[i] = fn(in[i]).
        self.w("static void* __maka_par_map_slice_entry(void* arg) {\n");
        self.w("    __maka_slice_chunk_t* c = (__maka_slice_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) c->out_ptr[i] = c->code.fn(c->env, c->in_ptr[i]);\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    free(c); return NULL;\n");
        self.w("}\n");
        self.w("Slice_maka_int __maka_par_map_int_slice(Slice_maka_int s, void* code, void* env) {\n");
        self.w("    Slice_maka_int empty = { .ptr = NULL, .len = 0 };\n");
        self.w("    if (s.len <= 0) return empty;\n");
        self.w("    int64_t* out = (int64_t*)malloc(sizeof(int64_t) * (size_t)s.len);\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = out; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.fn = (int64_t(*)(void*, int64_t))code;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_map_slice_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) (void)__maka_join_result((maka_unit*)hs[c]);\n");
        self.w("    free(hs);\n");
        self.w("    Slice_maka_int res = { .ptr = out, .len = s.len };\n");
        self.w("    return res;\n");
        self.w("}\n");

        // par_reduce over slice: per-chunk fold then sequential merge of partials.
        self.w("static void* __maka_par_reduce_slice_entry(void* arg) {\n");
        self.w("    __maka_slice_chunk_t* c = (__maka_slice_chunk_t*)arg;\n");
        self.w("    int64_t acc = c->init;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) acc = c->code.combine(c->env, acc, c->in_ptr[i]);\n");
        self.w("    c->out_acc = acc;\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("int64_t __maka_par_reduce_int_slice(Slice_maka_int s, int64_t init, void* code, void* env) {\n");
        self.w("    if (s.len <= 0) return init;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_slice_chunk_t** chs = (__maka_slice_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->env = env; ch->completion = th; ch->init = init;\n");
        self.w("        ch->code.combine = (int64_t(*)(void*, int64_t, int64_t))code;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_reduce_slice_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    int64_t acc = init;\n");
        self.w("    int64_t (*combine)(void*, int64_t, int64_t) = (int64_t(*)(void*, int64_t, int64_t))code;\n");
        self.w("    int merged_first = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        pthread_join(hs[c]->handle, NULL);\n");
        self.w("        if (!merged_first) { acc = chs[c]->out_acc; merged_first = 1; }\n");
        self.w("        else { acc = combine(env, acc, chs[c]->out_acc); }\n");
        self.w("        pthread_mutex_destroy(&hs[c]->done_mutex);\n");
        self.w("        pthread_cond_destroy(&hs[c]->done_cond);\n");
        self.w("        free(hs[c]); free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(hs); free(chs);\n");
        self.w("    return acc;\n");
        self.w("}\n");

        // par_filter_int: 2-pass parallel filter.  Pass 1 marks predicate
        // result in a flag array; pass 2 prefix-sums-then-scatters into output.
        // For simplicity in v1, do this sequentially in chunks then concat.
        self.w("static void* __maka_par_filter_entry(void* arg) {\n");
        self.w("    __maka_slice_chunk_t* c = (__maka_slice_chunk_t*)arg;\n");
        self.w("    int64_t w = 0;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) {\n");
        self.w("        int64_t v = c->in_ptr[i];\n");
        self.w("        if (c->code.pred(c->env, v)) c->out_ptr[w++] = v;\n");
        self.w("    }\n");
        self.w("    c->out_len = w;\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("Slice_maka_int __maka_par_filter_int(Slice_maka_int s, void* code, void* env) {\n");
        self.w("    Slice_maka_int empty = { .ptr = NULL, .len = 0 };\n");
        self.w("    if (s.len <= 0) return empty;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    int64_t* tmp = (int64_t*)malloc(sizeof(int64_t) * (size_t)s.len);\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_slice_chunk_t** chs = (__maka_slice_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = tmp + ch->i_start;\n");
        self.w("        ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.pred = (int(*)(void*, int64_t))code;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_filter_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    int64_t total = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        pthread_join(hs[c]->handle, NULL);\n");
        self.w("        total += chs[c]->out_len;\n");
        self.w("    }\n");
        self.w("    int64_t* out = (int64_t*)malloc(sizeof(int64_t) * (size_t)(total > 0 ? total : 1));\n");
        self.w("    int64_t w = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        memcpy(out + w, tmp + chs[c]->i_start, sizeof(int64_t) * (size_t)chs[c]->out_len);\n");
        self.w("        w += chs[c]->out_len;\n");
        self.w("        pthread_mutex_destroy(&hs[c]->done_mutex);\n");
        self.w("        pthread_cond_destroy(&hs[c]->done_cond);\n");
        self.w("        free(hs[c]); free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(hs); free(chs); free(tmp);\n");
        self.w("    Slice_maka_int res = { .ptr = out, .len = total };\n");
        self.w("    return res;\n");
        self.w("}\n");

        // ---- Float-slice parallel ops (symmetric to int versions) ----------
        self.w("typedef struct {\n");
        self.w("    int64_t i_start, i_end;\n");
        self.w("    double* in_ptr; double* out_ptr;\n");
        self.w("    void* env;\n");
        self.w("    Thread* completion;\n");
        self.w("    union { void (*body)(void*, double);\n");
        self.w("            double (*fn)(void*, double);\n");
        self.w("            double (*combine)(void*, double, double); } code;\n");
        self.w("    double init;\n");
        self.w("    double out_acc;\n");
        self.w("} __maka_fslice_chunk_t;\n");

        self.w("static void* __maka_par_each_f_entry(void* arg) {\n");
        self.w("    __maka_fslice_chunk_t* c = (__maka_fslice_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) c->code.body(c->env, c->in_ptr[i]);\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    free(c); return NULL;\n");
        self.w("}\n");
        self.w("void __maka_par_for_each_f64(Slice_maka_float s, void* code, void* env) {\n");
        self.w("    if (s.len <= 0) return;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_fslice_chunk_t* ch = (__maka_fslice_chunk_t*)calloc(1, sizeof(__maka_fslice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.body = (void(*)(void*, double))code;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_each_f_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) (void)__maka_join_result((maka_unit*)hs[c]);\n");
        self.w("    free(hs);\n");
        self.w("}\n");

        self.w("static void* __maka_par_map_f_entry(void* arg) {\n");
        self.w("    __maka_fslice_chunk_t* c = (__maka_fslice_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) c->out_ptr[i] = c->code.fn(c->env, c->in_ptr[i]);\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    free(c); return NULL;\n");
        self.w("}\n");
        self.w("Slice_maka_float __maka_par_map_float(Slice_maka_float s, void* code, void* env) {\n");
        self.w("    Slice_maka_float empty = { .ptr = NULL, .len = 0 };\n");
        self.w("    if (s.len <= 0) return empty;\n");
        self.w("    double* out = (double*)malloc(sizeof(double) * (size_t)s.len);\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_fslice_chunk_t* ch = (__maka_fslice_chunk_t*)calloc(1, sizeof(__maka_fslice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = out; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.fn = (double(*)(void*, double))code;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_map_f_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) (void)__maka_join_result((maka_unit*)hs[c]);\n");
        self.w("    free(hs);\n");
        self.w("    Slice_maka_float res = { .ptr = out, .len = s.len };\n");
        self.w("    return res;\n");
        self.w("}\n");

        self.w("static void* __maka_par_reduce_f_entry(void* arg) {\n");
        self.w("    __maka_fslice_chunk_t* c = (__maka_fslice_chunk_t*)arg;\n");
        self.w("    double acc = c->init;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) acc = c->code.combine(c->env, acc, c->in_ptr[i]);\n");
        self.w("    c->out_acc = acc;\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("double __maka_par_reduce_float(Slice_maka_float s, double init, void* code, void* env) {\n");
        self.w("    if (s.len <= 0) return init;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_fslice_chunk_t** chs = (__maka_fslice_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_fslice_chunk_t* ch = (__maka_fslice_chunk_t*)calloc(1, sizeof(__maka_fslice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->env = env; ch->completion = th; ch->init = init;\n");
        self.w("        ch->code.combine = (double(*)(void*, double, double))code;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_reduce_f_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    double acc = init;\n");
        self.w("    double (*combine)(void*, double, double) = (double(*)(void*, double, double))code;\n");
        self.w("    int merged_first = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        pthread_join(hs[c]->handle, NULL);\n");
        self.w("        if (!merged_first) { acc = chs[c]->out_acc; merged_first = 1; }\n");
        self.w("        else { acc = combine(env, acc, chs[c]->out_acc); }\n");
        self.w("        pthread_mutex_destroy(&hs[c]->done_mutex);\n");
        self.w("        pthread_cond_destroy(&hs[c]->done_cond);\n");
        self.w("        free(hs[c]); free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(hs); free(chs);\n");
        self.w("    return acc;\n");
        self.w("}\n");

        // Generic par_map_bytes — input slice of opaque items (in_item_size
        // bytes each), output slice of (out_item_size each).  The body
        // takes (env, in_ptr, out_ptr).  Users wrap to build typed par_map
        // for arbitrary T → U.
        self.w("typedef struct {\n");
        self.w("    int64_t i_start, i_end;\n");
        self.w("    char* in_ptr;\n");
        self.w("    char* out_ptr;\n");
        self.w("    int64_t in_sz;\n");
        self.w("    int64_t out_sz;\n");
        self.w("    void (*body)(void*, void*, void*);\n");
        self.w("    void* env;\n");
        self.w("    Thread* completion;\n");
        self.w("} __maka_bytes_chunk_t;\n");
        self.w("static void* __maka_par_map_bytes_entry(void* arg) {\n");
        self.w("    __maka_bytes_chunk_t* c = (__maka_bytes_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) {\n");
        self.w("        c->body(c->env, (void*)(c->in_ptr + i * c->in_sz), (void*)(c->out_ptr + i * c->out_sz));\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    free(c); return NULL;\n");
        self.w("}\n");
        // Returns a malloc'd buffer of (n * out_item_size) bytes; caller is
        // responsible for freeing (via free() since malloc was used).
        self.w("void* maka_par_map_bytes(void* in_ptr, int64_t n, int64_t in_sz, int64_t out_sz, void* code, void* env) {\n");
        self.w("    if (n <= 0) return NULL;\n");
        self.w("    char* out = (char*)malloc((size_t)(n * out_sz));\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > n) chunks = n;\n");
        self.w("    int64_t per = (n + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_bytes_chunk_t* ch = (__maka_bytes_chunk_t*)calloc(1, sizeof(__maka_bytes_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > n) ch->i_end = n;\n");
        self.w("        ch->in_ptr = (char*)in_ptr;\n");
        self.w("        ch->out_ptr = out;\n");
        self.w("        ch->in_sz = in_sz; ch->out_sz = out_sz;\n");
        self.w("        ch->env = env; ch->completion = th;\n");
        self.w("        ch->body = (void(*)(void*, void*, void*))code;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_map_bytes_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) (void)__maka_join_result((maka_unit*)hs[c]);\n");
        self.w("    free(hs);\n");
        self.w("    return (void*)out;\n");
        self.w("}\n");
        // par_scan_int: 2-pass parallel inclusive scan with associative combine.
        // Pass 1: each chunk computes local prefix into the output slice.
        // Pass 2: cross-chunk offsets are added in via combine.
        self.w("static void* __maka_par_scan_local_entry(void* arg) {\n");
        self.w("    __maka_slice_chunk_t* c = (__maka_slice_chunk_t*)arg;\n");
        self.w("    if (c->i_start < c->i_end) {\n");
        self.w("        int64_t acc = c->in_ptr[c->i_start];\n");
        self.w("        c->out_ptr[c->i_start] = acc;\n");
        self.w("        for (int64_t i = c->i_start + 1; i < c->i_end; i++) {\n");
        self.w("            acc = c->code.combine(c->env, acc, c->in_ptr[i]);\n");
        self.w("            c->out_ptr[i] = acc;\n");
        self.w("        }\n");
        self.w("        c->out_acc = acc;\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static void* __maka_par_scan_offset_entry(void* arg) {\n");
        self.w("    __maka_slice_chunk_t* c = (__maka_slice_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) {\n");
        self.w("        c->out_ptr[i] = c->code.combine(c->env, c->init, c->out_ptr[i]);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("Slice_maka_int __maka_par_scan_int(Slice_maka_int s, void* code, void* env) {\n");
        self.w("    Slice_maka_int empty = { .ptr = NULL, .len = 0 };\n");
        self.w("    if (s.len <= 0) return empty;\n");
        self.w("    int64_t* out = (int64_t*)malloc(sizeof(int64_t) * (size_t)s.len);\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_slice_chunk_t** chs = (__maka_slice_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    int64_t (*combine)(void*, int64_t, int64_t) = (int64_t(*)(void*, int64_t, int64_t))code;\n");
        self.w("    /* pass 1: per-chunk local prefix */\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = out; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.combine = combine;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_scan_local_entry, ch);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    int64_t* offsets = (int64_t*)malloc(sizeof(int64_t) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        pthread_join(hs[c]->handle, NULL);\n");
        self.w("        offsets[c] = chs[c]->out_acc;\n");
        self.w("        pthread_mutex_destroy(&hs[c]->done_mutex);\n");
        self.w("        pthread_cond_destroy(&hs[c]->done_cond);\n");
        self.w("        free(hs[c]);\n");
        self.w("    }\n");
        self.w("    /* pass 2: apply running offset across chunks (chunk 0 stays). */\n");
        self.w("    int64_t running = offsets[0];\n");
        self.w("    for (int64_t c = 1; c < chunks; c++) {\n");
        self.w("        Thread* th = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("        pthread_mutex_init(&th->done_mutex, NULL); pthread_cond_init(&th->done_cond, NULL);\n");
        self.w("        chs[c]->completion = th; chs[c]->init = running;\n");
        self.w("        pthread_create(&th->handle, NULL, __maka_par_scan_offset_entry, chs[c]);\n");
        self.w("        hs[c] = th;\n");
        self.w("        running = combine(env, running, offsets[c]);\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 1; c < chunks; c++) {\n");
        self.w("        pthread_join(hs[c]->handle, NULL);\n");
        self.w("        pthread_mutex_destroy(&hs[c]->done_mutex);\n");
        self.w("        pthread_cond_destroy(&hs[c]->done_cond);\n");
        self.w("        free(hs[c]); free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(chs[0]);\n");
        self.w("    free(hs); free(chs); free(offsets);\n");
        self.w("    Slice_maka_int res = { .ptr = out, .len = s.len };\n");
        self.w("    return res;\n");
        self.w("}\n");
        self.w("\n");
    }

    fn emit_structs_and_enums(&mut self) {
        // Forward decls of structs (in declaration order). Skip generic templates
        // and the built-in `Thread` (declared in the prologue with pthread fields).
        for s in &self.sym.structs {
            if !s.type_params.is_empty() { continue; }
            if s.name == "Thread" { continue; }
            self.wl(&format!("typedef struct {0} {0};", c_ident(&s.name)));
        }
        for e in &self.sym.enums {
            if e.is_simple() {
                // C-style enum: typedef as integer + constants.
                self.wl(&format!("typedef maka_int {0};", c_ident(&e.name)));
                for v in &e.variants {
                    self.wl(&format!("static const {0} {0}__{1} = {2};", c_ident(&e.name), c_ident(&v.name), v.tag));
                }
            } else {
                // Tagged enum: forward decl now, full def after struct typedefs.
                self.wl(&format!("typedef struct {0} {0};", c_ident(&e.name)));
                for v in &e.variants {
                    self.wl(&format!("#define {0}__{1}_TAG {2}", c_ident(&e.name), c_ident(&v.name), v.tag));
                }
            }
        }

        // Now full struct definitions: collect slice/vec types referenced in fields too.
        let structs = self.sym.structs.clone();
        for s in &structs {
            for f in &s.fields {
                self.note_type(&f.ty);
            }
        }
        // Emit slice/vec typedefs needed by structs early.
        self.emit_slice_typedefs();
        self.emit_vec_typedefs();

        for s in &structs {
            if !s.type_params.is_empty() { continue; } // skip templates
            if s.name == "Thread" { continue; }        // built-in, declared in prologue
            self.wl(&format!("struct {} {{", c_ident(&s.name)));
            self.open();
            for f in &s.fields {
                // `c_decl` correctly places the field name in the middle for
                // array-of-T fields (`T name[N]`) and uses the standard
                // `T name` form for everything else.  Plain `c_type()` would
                // emit `T[N] name`, which isn't valid C.
                let decl = self.c_decl(&f.ty, &c_ident(&f.name));
                self.wl(&format!("{};", decl));
            }
            self.close();
            self.wl("};");
        }

        // Tagged enum struct definitions: tag + union of variant payloads.
        let enums = self.sym.enums.clone();
        for e in &enums {
            if e.is_simple() { continue; }
            // First, emit per-variant payload structs (named EnumName_VariantName).
            for v in &e.variants {
                if v.fields.is_empty() { continue; }
                self.wl(&format!("typedef struct {{"));
                self.open();
                for f in &v.fields {
                    let ty = self.c_type(&f.ty);
                    self.wl(&format!("{} {};", ty, c_ident(&f.name)));
                }
                self.close();
                self.wl(&format!("}} {0}_{1}_Payload;", c_ident(&e.name), c_ident(&v.name)));
            }
            // Then the enum struct itself.
            self.wl(&format!("struct {0} {{", c_ident(&e.name)));
            self.open();
            self.wl("maka_int tag;");
            // Union of all non-empty payloads.
            let has_payload = e.variants.iter().any(|v| !v.fields.is_empty());
            if has_payload {
                self.wl("union {");
                self.open();
                for v in &e.variants {
                    if v.fields.is_empty() { continue; }
                    self.wl(&format!("{0}_{1}_Payload {1};", c_ident(&e.name), c_ident(&v.name)));
                }
                self.close();
                self.wl("} payload;");
            }
            self.close();
            self.wl("};");
        }
        self.wl("");
    }

    fn scan_func(&mut self, f: &HFunc) {
        for l in &f.locals { self.note_type(&l.ty); }
        self.note_type(&f.ret);
        // walk body to record types in expressions/types
        self.scan_block(&f.body);
    }

    fn scan_block(&mut self, b: &HBlock) {
        for s in &b.stmts { self.scan_stmt(s); }
    }
    fn scan_stmt(&mut self, s: &HStmt) {
        match s {
            HStmt::Let { init, .. } => { self.note_type(&init.ty); self.scan_expr(init); }
            HStmt::Assign { place, value, .. } => { self.scan_expr(place); self.scan_expr(value); }
            HStmt::ExprStmt(e) => self.scan_expr(e),
            HStmt::Return { value, .. } => if let Some(e) = value { self.scan_expr(e); }
            HStmt::If { cond, then_b, else_b, .. } => { self.scan_expr(cond); self.scan_block(then_b); if let Some(b) = else_b { self.scan_block(b); } }
            HStmt::While { cond, body, .. } => { self.scan_expr(cond); self.scan_block(body); }
            HStmt::Block(b) | HStmt::Unsafe(b, _) => self.scan_block(b),
            HStmt::Break(_) | HStmt::Continue(_) => {}
            HStmt::ForC { init, cond, step, body, .. } => {
                self.scan_stmt(init);
                self.scan_expr(cond);
                self.scan_stmt(step);
                self.scan_block(body);
            }
            HStmt::ForEach { src, body, .. } => {
                self.scan_expr(src);
                self.scan_block(body);
            }
            HStmt::Propagate { value: Some(v), .. } => self.scan_expr(v),
            HStmt::Propagate { value: None, .. } => {}
        }
    }
    fn scan_expr(&mut self, e: &HExpr) {
        self.note_type(&e.ty);
        match &e.kind {
            HExprKind::Bin { lhs, rhs, .. } => { self.scan_expr(lhs); self.scan_expr(rhs); }
            HExprKind::Un { expr, .. } => self.scan_expr(expr),
            HExprKind::Unwrap { expr, .. } => self.scan_expr(expr),
            HExprKind::AddrOfRef { place, .. } => self.scan_expr(place),
            HExprKind::Field { base, .. } => self.scan_expr(base),
            HExprKind::Index { base, idx } => { self.scan_expr(base); self.scan_expr(idx); }
            HExprKind::Call { args, .. } => for a in args { self.scan_expr(a); },
            HExprKind::Cast { expr, kind, .. } => {
                if let CastKind::ToDyn { trait_name, struct_id } = kind {
                    self.dyn_traits.insert(trait_name.clone());
                    self.dyn_insts.insert((trait_name.clone(), struct_id.0));
                }
                self.scan_expr(expr);
            }
            HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) => self.scan_expr(expr),
            HExprKind::ArrayToSlice { base, .. } => self.scan_expr(base),
            HExprKind::DerefRef(inner) => self.scan_expr(inner),
            HExprKind::HeapAlloc(inner) => self.scan_expr(inner),
            HExprKind::CallIndirect { callee, args } => {
                self.scan_expr(callee);
                for a in args { self.scan_expr(a); }
            }
            HExprKind::InlineCall { args, .. } => {
                for a in args { self.scan_expr(a); }
            }
            HExprKind::Closure { lifted, env_values, .. } => {
                self.closure_trampolines.insert(lifted.0);
                for v in env_values { self.scan_expr(v); }
            }
            HExprKind::Transfer(inner) => self.scan_expr(inner),
            HExprKind::SliceLen(inner) | HExprKind::EnumTag(inner) => self.scan_expr(inner),
            HExprKind::FnRef(fid) => { self.fn_trampolines.insert(fid.0); }
            HExprKind::VariantCtor { fields, .. } => for (_, fe) in fields { self.scan_expr(fe); },
            HExprKind::Match { scrutinee, arms, .. } => {
                self.scan_expr(scrutinee);
                for a in arms {
                    if let Some(g) = &a.guard { self.scan_expr(g); }
                    for s in &a.body.stmts { self.scan_stmt(s); }
                    if let Some(v) = &a.value { self.scan_expr(v); }
                }
            }
            HExprKind::Struct { fields, .. } => for (_, fe) in fields { self.scan_expr(fe); },
            HExprKind::ArrayLit(es) => for e in es { self.scan_expr(e); },
            _ => {}
        }
    }

    fn note_type(&mut self, t: &HType) {
        match t {
            HType::Slice { elem, .. } => { self.slice_types.insert(self.type_key(elem)); self.note_type(elem); }
            HType::Vec { elem } => { self.vec_types.insert(self.type_key(elem)); self.note_type(elem); }
            HType::Heap { inner } => self.note_type(inner),
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } => self.note_type(inner),
            HType::Array { elem, .. } => self.note_type(elem),
            HType::Dyn { traits } => for t in traits { self.dyn_traits.insert(t.clone()); },
            HType::FnPtr { ret, params } => {
                let key = fn_sig_key(ret, params);
                self.callable_sigs.insert(key, ((**ret).clone(), params.clone()));
                self.note_type(ret);
                for p in params { self.note_type(p); }
            }
            _ => {}
        }
    }

    fn emit_dyn_typedefs(&mut self) {
        // Typedef structs only (Movement_vtbl + Dyn_Movement). Vtable instances are emitted
        // separately AFTER function forward declarations (via emit_dyn_vtable_instances).
        let traits: Vec<String> = self.dyn_traits.iter().cloned().collect();
        for tn in &traits {
            let Some(linfo) = self.sym.logic_by_name(tn) else { continue; };
            // Take *unique* function names declared in the logic.
            let mut name_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for fid in &linfo.funcs {
                let s = self.sym.func_sig(*fid);
                name_set.insert(s.name.clone());
            }
            // Emit vtable struct.
            self.wl(&format!("typedef struct {0}_vtbl {{", c_ident(tn)));
            self.open();
            for n in &name_set {
                // For each declared name, find one signature to derive the function-pointer shape.
                let sig = linfo.funcs.iter().filter_map(|fid| {
                    let s = self.sym.func_sig(*fid);
                    if s.name == *n { Some(s.clone()) } else { None }
                }).next();
                let Some(sig) = sig else { continue; };
                // The receiver becomes void* (erased).
                let ret = self.c_ret_type(&sig.ret);
                let mut parts: Vec<String> = Vec::new();
                for (i, p) in sig.param_tys.iter().enumerate() {
                    if i == 0 {
                        parts.push("void*".into());
                    } else {
                        parts.push(self.c_type(p));
                    }
                }
                let pstr = if parts.is_empty() { "void".to_string() } else { parts.join(", ") };
                self.wl(&format!("{} (*{})({});", ret, c_ident(n), pstr));
            }
            self.close();
            self.wl(&format!("}} {0}_vtbl;", c_ident(tn)));

            // Dyn struct.
            self.wl(&format!("typedef struct {{ void* data; const {0}_vtbl* vtbl; }} Dyn_{0};", c_ident(tn)));
        }
    }

    fn emit_callable_typedefs(&mut self) {
        // For each fat-callable signature, emit:
        //   typedef RetType (*fn_KEY_raw)(void*, ParamTypes);
        //   typedef struct { fn_KEY_raw code; void* env; } Callable_KEY;
        let sigs: Vec<(String, HType, Vec<HType>)> = self.callable_sigs.iter()
            .map(|(k, (r, ps))| (k.clone(), r.clone(), ps.clone())).collect();
        for (key, ret, params) in &sigs {
            let ret_c = self.c_ret_type(ret);
            let mut parts: Vec<String> = vec!["void*".to_string()];
            for p in params { parts.push(self.c_type(p)); }
            let plist = parts.join(", ");
            self.wl(&format!("typedef {} (*fn_{}_raw)({});", ret_c, key, plist));
            self.wl(&format!("typedef struct {{ fn_{0}_raw code; void* env; }} Callable_{0};", key));
        }
    }

    fn emit_trampolines(&mut self, funcs: &[HFunc]) {
        // For each function whose name is used as a value, emit a trampoline that ignores env.
        let trams: Vec<u32> = self.fn_trampolines.iter().copied().collect();
        for fid_n in trams {
            // Skip if the FuncSig doesn't exist (placeholder ids in inline contexts).
            if (fid_n as usize) >= self.sym.sigs.len() { continue; }
            let sig = self.sym.func_sig(FuncId(fid_n)).clone();
            // Inline functions get inlined at call sites; trampolines aren't useful for them.
            if sig.is_inline { continue; }
            // Skip extern: their type signatures may differ; trampolines for them defer to
            // the user-declared signature.
            let target_name = if sig.is_extern { sig.c_name.clone() }
                else if sig.name == "main" && sig.logic.is_none() { "maka_main".to_string() }
                else { c_ident(&sig.c_name) };
            let ret_c = self.c_ret_type(&sig.ret);
            let mut decl_parts: Vec<String> = vec!["void* __env".to_string()];
            let mut call_args: Vec<String> = Vec::new();
            for (i, p) in sig.param_tys.iter().enumerate() {
                let pn = format!("__a{}", i);
                decl_parts.push(self.c_decl(p, &pn));
                call_args.push(pn);
            }
            let plist = decl_parts.join(", ");
            let tram_name = format!("__tramp_{}", target_name);
            self.wl(&format!("static {} {}({}) {{", ret_c, tram_name, plist));
            self.open();
            self.wl("(void)__env;");
            if matches!(sig.ret, HType::Unit) {
                if !funcs.is_empty() {
                    self.wl(&format!("{}({});", target_name, call_args.join(", ")));
                } else {
                    self.wl(&format!("(void){};", target_name));
                }
                self.wl("return MAKA_UNIT;");
            } else {
                self.wl(&format!("return {}({});", target_name, call_args.join(", ")));
            }
            self.close();
            self.wl("}");
        }
    }

    fn emit_closure_trampolines(&mut self) {
        // For each lifted lambda function that needs a closure trampoline, emit:
        //   RetType lambdaN_closure_trampoline(void* env, ParamTypes args) {
        //     LambdaEnvN* e = (LambdaEnvN*)env;
        //     return lambdaN_body(e, args);
        //   }
        let ids: Vec<u32> = self.closure_trampolines.iter().copied().collect();
        for fid_n in ids {
            if (fid_n as usize) >= self.sym.sigs.len() { continue; }
            let sig = self.sym.func_sig(FuncId(fid_n)).clone();
            // The first param is the env ref; the rest are lambda params.
            if sig.param_tys.is_empty() { continue; }
            let env_ty = sig.param_tys[0].clone();
            let lambda_params: Vec<HType> = sig.param_tys.iter().skip(1).cloned().collect();
            let env_struct = match env_ty {
                HType::Ref { inner, .. } => match *inner {
                    HType::Struct(id) => id,
                    _ => continue,
                },
                _ => continue,
            };
            let env_struct_name = c_ident(&self.sym.struct_info(env_struct).name);
            let key = fn_sig_key(&sig.ret, &lambda_params);
            let ret_c = self.c_ret_type(&sig.ret);
            let lifted_c = c_ident(&sig.c_name);
            let mut decl_parts = vec!["void* __env".to_string()];
            let mut call_args = vec!["__es".to_string()];
            for (i, p) in lambda_params.iter().enumerate() {
                let pn = format!("__a{}", i);
                decl_parts.push(self.c_decl(p, &pn));
                call_args.push(pn);
            }
            let plist = decl_parts.join(", ");
            self.wl(&format!("static {} {}_closure_trampoline({}) {{", ret_c, lifted_c, plist));
            self.open();
            self.wl(&format!("{}* __es = ({}*)__env;", env_struct_name, env_struct_name));
            if matches!(sig.ret, HType::Unit) {
                self.wl(&format!("{}({});", lifted_c, call_args.join(", ")));
                self.wl("return MAKA_UNIT;");
            } else {
                self.wl(&format!("return {}({});", lifted_c, call_args.join(", ")));
            }
            self.close();
            self.wl("}");
            // The signature key for the callable produced.
            let _ = key;
        }
    }

    fn emit_dyn_vtable_instances(&mut self) {
        let insts: Vec<(String, u32)> = self.dyn_insts.iter().cloned().collect();
        for (tn, sid_n) in &insts {
            let sid = StructId(*sid_n);
            let sname = self.sym.struct_info(sid).name.clone();
            let Some(linfo) = self.sym.logic_by_name(tn) else { continue; };
            // For each distinct function name, find the matching overload for this struct.
            let mut name_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for fid in &linfo.funcs {
                let s = self.sym.func_sig(*fid);
                name_set.insert(s.name.clone());
            }
            self.wl(&format!("static const {0}_vtbl {0}_vtbl_for_{1} = {{", c_ident(tn), c_ident(&sname)));
            self.open();
            for n in &name_set {
                // Find the FuncSig for this T.
                let chosen = linfo.funcs.iter().find_map(|fid| {
                    let s = self.sym.func_sig(*fid);
                    if s.name != *n { return None; }
                    if s.param_tys.is_empty() { return None; }
                    if let HType::Ref { inner, .. } = &s.param_tys[0] {
                        if let HType::Struct(id) = inner.as_ref() {
                            if *id == sid { return Some(s.c_name.clone()); }
                        }
                    }
                    None
                });
                if let Some(cn) = chosen {
                    self.wl(&format!(".{} = (void*){},", c_ident(n), cn));
                }
            }
            self.close();
            self.wl("};");
        }
    }

    fn emit_slice_typedefs(&mut self) {
        let mut emitted = Vec::new();
        let keys: Vec<String> = self.slice_types.iter().cloned().collect();
        for key in keys {
            let name = format!("Slice_{}", key);
            if !self.out.contains(&format!("typedef struct {} ", name)) {
                self.wl(&format!("typedef struct {0} {{ {1}* ptr; maka_int len; }} {0};", name, self.c_type_from_key(&key)));
                emitted.push(name);
            }
        }
    }
    fn emit_vec_typedefs(&mut self) {
        let keys: Vec<String> = self.vec_types.iter().cloned().collect();
        for key in keys {
            let name = format!("Vec_{}", key);
            if !self.out.contains(&format!("typedef struct {} ", name)) {
                self.wl(&format!("typedef struct {0} {{ {1}* data; maka_int len; maka_int cap; }} {0};", name, self.c_type_from_key(&key)));
            }
        }
    }

    fn type_key(&self, t: &HType) -> String {
        match t {
            HType::Int => "maka_int".into(),
            HType::SizedInt { signed, bits: 0 } => if *signed { "intptr_t".into() } else { "uintptr_t".into() },
            HType::SizedInt { signed, bits } => format!("{}int{}_t", if *signed {""} else {"u"}, bits),
            HType::Float => "maka_float".into(),
            HType::Bool => "bool".into(),
            HType::Char => "maka_char".into(),
            HType::Unit => "maka_unit".into(),
            HType::Struct(id) => c_ident(&self.sym.struct_info(*id).name),
            HType::Enum(id) => c_ident(&self.sym.enum_info(*id).name),
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } => format!("p_{}", self.type_key(inner)),
            HType::Heap { inner } => format!("h_{}", self.type_key(inner)),
            HType::Array { len, elem } => format!("a{}_{}", len, self.type_key(elem)),
            HType::Slice { elem, .. } => format!("Slice_{}", self.type_key(elem)),
            HType::Vec { elem } => format!("Vec_{}", self.type_key(elem)),
            HType::Str => "str".into(),
            HType::NullT => "nullptr_t".into(),
            HType::Dyn { traits } => format!("Dyn_{}", traits.join("_")),
            HType::FnPtr { ret, params } => {
                let mut s = format!("fn_{}_", self.type_key(ret));
                for p in params { s.push_str(&self.type_key(p)); s.push('_'); }
                s
            }
            HType::TyVar(n) => format!("T_{}", n),
            // `Rust<T>` shares the C layout of `own *mut unit` (= `void*`).
            HType::RustOpaque(_) => "p_maka_unit".into(),
        }
    }

    fn c_type_from_key(&self, k: &str) -> String {
        // Most keys are already valid C type names (struct names, primitives like
        // `maka_int`), but a few Maka-level keys need translation.
        // Pointer keys are encoded as `p_<inner>` / `pm_<inner>` etc. for the
        // hash; in C they need to land as `<inner>*`, recursively.
        if let Some(rest) = k.strip_prefix("p_") {
            return format!("{}*", self.c_type_from_key(rest));
        }
        if let Some(rest) = k.strip_prefix("pm_") {
            return format!("{}*", self.c_type_from_key(rest));
        }
        if let Some(rest) = k.strip_prefix("pc_") {
            return format!("const {}*", self.c_type_from_key(rest));
        }
        match k {
            "str" => "const char*".into(),
            other => other.to_string(),
        }
    }

    fn c_type(&self, t: &HType) -> String {
        match t {
            HType::Int => "maka_int".into(),
            HType::SizedInt { signed, bits: 0 } => if *signed { "intptr_t".into() } else { "uintptr_t".into() },
            HType::SizedInt { signed, bits } => format!("{}int{}_t", if *signed {""} else {"u"}, bits),
            HType::Float => "maka_float".into(),
            HType::Bool => "bool".into(),
            HType::Char => "maka_char".into(),
            HType::Unit => "maka_unit".into(),
            HType::Str => "const char*".into(),
            HType::NullT => "void*".into(),
            HType::Struct(id) => c_ident(&self.sym.struct_info(*id).name),
            HType::Enum(id) => c_ident(&self.sym.enum_info(*id).name),
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } => {
                // Refs/pointers to dyn collapse to the dyn value itself (fat pointer already
                // contains an erased pointer to the underlying data).
                if matches!(inner.as_ref(), HType::Dyn { .. }) {
                    return self.c_type(inner);
                }
                format!("{}*", self.c_type(inner))
            }
            HType::Heap { inner } => match inner.as_ref() {
                // heap [*]T is just the Vec struct (no extra indirection)
                HType::Vec { .. } => self.c_type(inner),
                _ => format!("{}*", self.c_type(inner)),
            },
            HType::Array { len, elem } => {
                // arrays are only valid in fixed-size contexts; use raw element type and
                // append [N] at the declaration site. For type-name use a pointer-decay.
                format!("{}[{}]", self.c_type(elem), len)
            }
            HType::Slice { elem, .. } => format!("Slice_{}", self.type_key(elem)),
            HType::Vec { elem } => format!("Vec_{}", self.type_key(elem)),
            HType::Dyn { traits } => format!("Dyn_{}", c_ident(&traits[0])),
            HType::FnPtr { ret, params } => {
                format!("Callable_{}", fn_sig_key(ret, params))
            }
            HType::TyVar(_) => "void*".into(), // erased; not expected at codegen
            // `Rust<T>` is `own *mut unit` at the C layer (= `void*`).
            HType::RustOpaque(_) => "void*".into(),
        }
    }

    /// Like c_type but suitable for declaring a variable named `name` with that type.
    fn c_decl(&self, t: &HType, name: &str) -> String {
        match t {
            HType::Array { len, elem } => format!("{} {}[{}]", self.c_type(elem), name, len),
            HType::FnPtr { ret, params } => {
                format!("Callable_{} {}", fn_sig_key(ret, params), name)
            }
            _ => format!("{} {}", self.c_type(t), name),
        }
    }

    fn func_signature(&self, f: &HFunc) -> String {
        let ret = self.c_ret_type(&f.ret);
        let sig = self.sym.func_sig(f.id);
        let mangled = if sig.name == "main" && sig.logic.is_none() {
            "maka_main".to_string()
        } else {
            c_ident(&sig.c_name)
        };
        let mut out = format!("{} {}(", ret, mangled);
        if f.params.is_empty() { out.push_str("void"); }
        else {
            let parts: Vec<String> = f.params.iter().map(|id| {
                let li = &f.locals[id.0 as usize];
                self.c_decl(&li.ty.strip_heap().clone(), &local_name(*id, &li.name))
            }).collect();
            // For heap params we still use T* (heap-storage). Use c_decl on the heap-wrapped type instead.
            let parts2: Vec<String> = f.params.iter().map(|id| {
                let li = &f.locals[id.0 as usize];
                self.c_decl(&li.ty, &local_name(*id, &li.name))
            }).collect();
            let _ = parts;
            out.push_str(&parts2.join(", "));
        }
        out.push_str(")");
        out
    }

    fn c_ret_type(&self, t: &HType) -> String {
        match t {
            HType::Unit => "void".into(),
            _ => self.c_type(t),
        }
    }

    fn emit_socket_helpers(&mut self) {
        // Emitted AFTER all user code, and uses manual forward decls + the
        // syscall(2) for `accept` so neither sys/socket.h's symbol pollution
        // nor a clash with a user function named `accept` blows up the build.
        self.w("/* ---- TCP socket runtime ---- */\n");
        self.w("#include <sys/syscall.h>\n");
        // Manual forward decls avoid pulling sys/socket.h's declaration of
        // `accept` into scope (which would conflict with user-defined funcs
        // named `accept`, see test 114).  All socket calls go through these
        // shims; `accept` is invoked via syscall(2) inside the helper below.
        self.w("typedef unsigned int __maka_socklen_t;\n");
        self.w("struct sockaddr_in { unsigned short sin_family; unsigned short sin_port; struct { unsigned int s_addr; } sin_addr; unsigned char sin_zero[8]; };\n");
        self.w("struct sockaddr { unsigned short sa_family; char sa_data[14]; };\n");
        self.w("extern int socket(int, int, int);\n");
        self.w("extern int bind  (int, const struct sockaddr*, __maka_socklen_t);\n");
        self.w("extern int listen(int, int);\n");
        self.w("extern int connect(int, const struct sockaddr*, __maka_socklen_t);\n");
        self.w("extern int setsockopt(int, int, int, const void*, __maka_socklen_t);\n");
        self.w("extern int getsockopt(int, int, int, void*, __maka_socklen_t*);\n");
        self.w("extern long sendto(int, const void*, unsigned long, int, const struct sockaddr*, __maka_socklen_t);\n");
        self.w("extern long recvfrom(int, void*, unsigned long, int, struct sockaddr*, __maka_socklen_t*);\n");
        self.w("extern unsigned short htons(unsigned short);\n");
        self.w("extern unsigned int   htonl(unsigned int);\n");
        self.w("#define __MAKA_AF_INET     2\n");
        self.w("#define __MAKA_SOCK_STREAM 1\n");
        self.w("#define __MAKA_INADDR_ANY  0u\n");
        self.w("#define __MAKA_SOL_SOCKET  1\n");
        self.w("#define __MAKA_SO_REUSEADDR 2\n");
        self.w("#define __MAKA_SO_ERROR    4\n");
        self.w("static inline int64_t __maka_tcp_listen(int64_t port, int64_t backlog) {\n");
        self.w("    int s = socket(__MAKA_AF_INET, __MAKA_SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    int one = 1;\n");
        self.w("    setsockopt(s, __MAKA_SOL_SOCKET, __MAKA_SO_REUSEADDR, &one, sizeof(one));\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa));\n");
        self.w("    sa.sin_family = __MAKA_AF_INET;\n");
        self.w("    sa.sin_addr.s_addr = htonl(__MAKA_INADDR_ANY);\n");
        self.w("    sa.sin_port = htons((unsigned short)port);\n");
        self.w("    if (bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) { close(s); return -1; }\n");
        self.w("    if (listen(s, (int)backlog) != 0) { close(s); return -1; }\n");
        self.w("    int flags = fcntl(s, F_GETFL, 0);\n");
        self.w("    fcntl(s, F_SETFL, flags | O_NONBLOCK);\n");
        self.w("    return s;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_tcp_accept_async(int64_t listen_fd) {\n");
        self.w("    while (1) {\n");
        // Call accept via direct syscall to avoid clashing with a user-named
        // Maka function called `accept` (see tests/programs/114_*).
        self.w("        int c = (int)syscall(SYS_accept, (int)listen_fd, (void*)0, (void*)0);\n");
        self.w("        if (c >= 0) {\n");
        self.w("            int flags = fcntl(c, F_GETFL, 0);\n");
        self.w("            fcntl(c, F_SETFL, flags | O_NONBLOCK);\n");
        self.w("            return c;\n");
        self.w("        }\n");
        self.w("        if (errno == EAGAIN || errno == EWOULDBLOCK) {\n");
        self.w("            __maka_wait_fd(listen_fd, MAKA_EV_READ); continue;\n");
        self.w("        }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_tcp_connect_v4(int64_t a, int64_t b, int64_t c, int64_t d, int64_t port) {\n");
        self.w("    int s = socket(__MAKA_AF_INET, __MAKA_SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    int flags = fcntl(s, F_GETFL, 0);\n");
        self.w("    fcntl(s, F_SETFL, flags | O_NONBLOCK);\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa));\n");
        self.w("    sa.sin_family = __MAKA_AF_INET;\n");
        self.w("    sa.sin_port = htons((unsigned short)port);\n");
        self.w("    unsigned char octets[4] = {(unsigned char)a, (unsigned char)b, (unsigned char)c, (unsigned char)d};\n");
        self.w("    memcpy(&sa.sin_addr.s_addr, octets, 4);\n");
        self.w("    int r = connect(s, (struct sockaddr*)&sa, sizeof(sa));\n");
        self.w("    if (r == 0) return s;\n");
        self.w("    if (errno == EINPROGRESS) {\n");
        self.w("        __maka_wait_fd(s, MAKA_EV_WRITE);\n");
        self.w("        int err = 0; __maka_socklen_t elen = sizeof(err);\n");
        self.w("        getsockopt(s, __MAKA_SOL_SOCKET, __MAKA_SO_ERROR, &err, &elen);\n");
        self.w("        if (err == 0) return s;\n");
        self.w("        close(s); return -1;\n");
        self.w("    }\n");
        self.w("    close(s); return -1;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_close_fd(int64_t fd) { return close((int)fd); }\n");
        // HTTP request line parser — given a request buffer, returns the
        // method, path, and Content-Length as offset+len in pieces.
        // Caller exposes them via separate getters (Maka can't return tuples).
        self.w("static __thread int __maka_http_method_off = -1;\n");
        self.w("static __thread int __maka_http_method_len = 0;\n");
        self.w("static __thread int __maka_http_path_off = -1;\n");
        self.w("static __thread int __maka_http_path_len = 0;\n");
        self.w("static __thread int __maka_http_body_off = -1;\n");
        self.w("static __thread int __maka_http_content_length = -1;\n");
        self.w("static inline int64_t __maka_http_parse(const char* buf, int64_t len) {\n");
        self.w("    __maka_http_method_off = -1; __maka_http_method_len = 0;\n");
        self.w("    __maka_http_path_off   = -1; __maka_http_path_len = 0;\n");
        self.w("    __maka_http_body_off   = -1; __maka_http_content_length = -1;\n");
        self.w("    if (len < 4) return -1;\n");
        // method
        self.w("    int i = 0;\n");
        self.w("    __maka_http_method_off = 0;\n");
        self.w("    while (i < len && buf[i] != ' ') i++;\n");
        self.w("    if (i >= len) return -1;\n");
        self.w("    __maka_http_method_len = i;\n");
        self.w("    i++;\n");
        // path
        self.w("    __maka_http_path_off = i;\n");
        self.w("    while (i < len && buf[i] != ' ') i++;\n");
        self.w("    if (i >= len) return -1;\n");
        self.w("    __maka_http_path_len = i - __maka_http_path_off;\n");
        // skip to end of request line
        self.w("    while (i < len && buf[i] != '\\n') i++;\n");
        self.w("    if (i >= len) return -1;\n");
        self.w("    i++;\n");
        // headers — scan for Content-Length and the empty line that ends the headers.
        self.w("    while (i < len) {\n");
        self.w("        if (buf[i] == '\\r' || buf[i] == '\\n') {\n");
        self.w("            int j = i; if (buf[j] == '\\r' && j + 1 < len && buf[j+1] == '\\n') j += 2; else j++;\n");
        self.w("            __maka_http_body_off = j;\n");
        self.w("            return 0;\n");
        self.w("        }\n");
        // Case-insensitive compare with "Content-Length:"
        self.w("        if (i + 15 < len) {\n");
        self.w("            const char* cl = \"Content-Length:\";\n");
        self.w("            int match = 1;\n");
        self.w("            for (int k = 0; k < 15; k++) {\n");
        self.w("                char a = buf[i + k]; char b = cl[k];\n");
        self.w("                if (a >= 'A' && a <= 'Z') a += 32;\n");
        self.w("                if (b >= 'A' && b <= 'Z') b += 32;\n");
        self.w("                if (a != b) { match = 0; break; }\n");
        self.w("            }\n");
        self.w("            if (match) {\n");
        self.w("                int p = i + 15;\n");
        self.w("                while (p < len && (buf[p] == ' ' || buf[p] == '\\t')) p++;\n");
        self.w("                int v = 0;\n");
        self.w("                while (p < len && buf[p] >= '0' && buf[p] <= '9') { v = v * 10 + (buf[p] - '0'); p++; }\n");
        self.w("                __maka_http_content_length = v;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        while (i < len && buf[i] != '\\n') i++;\n");
        self.w("        if (i < len) i++;\n");
        self.w("    }\n");
        self.w("    return -1;  /* headers incomplete */\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_http_method_off_g(void) { return __maka_http_method_off; }\n");
        self.w("static inline int64_t __maka_http_method_len_g(void) { return __maka_http_method_len; }\n");
        self.w("static inline int64_t __maka_http_path_off_g  (void) { return __maka_http_path_off; }\n");
        self.w("static inline int64_t __maka_http_path_len_g  (void) { return __maka_http_path_len; }\n");
        self.w("static inline int64_t __maka_http_body_off_g  (void) { return __maka_http_body_off; }\n");
        self.w("static inline int64_t __maka_http_content_length_g(void) { return __maka_http_content_length; }\n");
        // pipe_create: opens a non-blocking pipe pair and returns the read
        // fd; the write fd is stashed and retrievable via pipe_write_fd.
        // Per-thread stash — keeps the simple "make a pipe, send it through
        // to a fiber" pattern from needing a cblock helper.
        self.w("extern int pipe(int*);\n");
        self.w("static __thread int __maka_last_pipe_wfd = -1;\n");
        self.w("static inline int64_t __maka_pipe_create(void) {\n");
        self.w("    int fds[2];\n");
        self.w("    if (pipe(fds) != 0) return -1;\n");
        self.w("    int f0 = fcntl(fds[0], F_GETFL, 0); fcntl(fds[0], F_SETFL, f0 | O_NONBLOCK);\n");
        self.w("    int f1 = fcntl(fds[1], F_GETFL, 0); fcntl(fds[1], F_SETFL, f1 | O_NONBLOCK);\n");
        self.w("    __maka_last_pipe_wfd = fds[1];\n");
        self.w("    return fds[0];\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_pipe_write_fd(void) { return __maka_last_pipe_wfd; }\n");
        // ---- TLS (OpenSSL) ----------------------------------------------
        // Conditional on -DMAKA_TLS at compile time + -lssl -lcrypto link.
        // Without that, the helpers return -1 / NULL so user programs at
        // least compile.  When MAKA_TLS is defined, full OpenSSL handshake
        // + reactor-aware read/write that yields on WANT_READ/WANT_WRITE.
        self.w("#ifdef MAKA_TLS\n");
        self.w("#include <openssl/ssl.h>\n");
        self.w("#include <openssl/err.h>\n");
        self.w("static _Atomic int __maka_tls_inited = 0;\n");
        self.w("static SSL_CTX* __maka_tls_ctx = NULL;\n");
        self.w("static void __maka_tls_init_once(void) {\n");
        self.w("    int e = 0;\n");
        self.w("    if (!atomic_compare_exchange_strong(&__maka_tls_inited, &e, 1)) return;\n");
        self.w("    SSL_library_init();\n");
        self.w("    SSL_load_error_strings();\n");
        self.w("    OpenSSL_add_all_algorithms();\n");
        self.w("    __maka_tls_ctx = SSL_CTX_new(TLS_client_method());\n");
        self.w("    if (__maka_tls_ctx) SSL_CTX_set_default_verify_paths(__maka_tls_ctx);\n");
        self.w("}\n");
        self.w("static inline maka_unit* __maka_tls_client_new(int64_t fd, const char* hostname) {\n");
        self.w("    __maka_tls_init_once();\n");
        self.w("    if (!__maka_tls_ctx) return NULL;\n");
        self.w("    SSL* s = SSL_new(__maka_tls_ctx);\n");
        self.w("    if (!s) return NULL;\n");
        self.w("    SSL_set_fd(s, (int)fd);\n");
        self.w("    SSL_set_tlsext_host_name(s, hostname);\n");
        self.w("    return (maka_unit*)s;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_tls_handshake(maka_unit* p) {\n");
        self.w("    SSL* s = (SSL*)p;\n");
        self.w("    while (1) {\n");
        self.w("        int r = SSL_connect(s);\n");
        self.w("        if (r == 1) return 0;\n");
        self.w("        int e = SSL_get_error(s, r);\n");
        self.w("        int fd = SSL_get_fd(s);\n");
        self.w("        if (e == SSL_ERROR_WANT_READ)  { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        if (e == SSL_ERROR_WANT_WRITE) { __maka_wait_fd(fd, MAKA_EV_WRITE); continue; }\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_tls_read(maka_unit* p, maka_unit* buf, int64_t cap) {\n");
        self.w("    SSL* s = (SSL*)p;\n");
        self.w("    while (1) {\n");
        self.w("        int r = SSL_read(s, (void*)buf, (int)cap);\n");
        self.w("        if (r > 0) return (int64_t)r;\n");
        self.w("        int e = SSL_get_error(s, r);\n");
        self.w("        int fd = SSL_get_fd(s);\n");
        self.w("        if (e == SSL_ERROR_WANT_READ)  { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        if (e == SSL_ERROR_WANT_WRITE) { __maka_wait_fd(fd, MAKA_EV_WRITE); continue; }\n");
        self.w("        if (e == SSL_ERROR_ZERO_RETURN) return 0;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_tls_write(maka_unit* p, maka_unit* buf, int64_t len) {\n");
        self.w("    SSL* s = (SSL*)p;\n");
        self.w("    int64_t w = 0;\n");
        self.w("    while (w < len) {\n");
        self.w("        int r = SSL_write(s, (const char*)(const void*)buf + w, (int)(len - w));\n");
        self.w("        if (r > 0) { w += r; continue; }\n");
        self.w("        int e = SSL_get_error(s, r);\n");
        self.w("        int fd = SSL_get_fd(s);\n");
        self.w("        if (e == SSL_ERROR_WANT_READ)  { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        if (e == SSL_ERROR_WANT_WRITE) { __maka_wait_fd(fd, MAKA_EV_WRITE); continue; }\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("    return w;\n");
        self.w("}\n");
        self.w("static inline void __maka_tls_close(maka_unit* p) {\n");
        self.w("    SSL* s = (SSL*)p;\n");
        self.w("    SSL_shutdown(s);\n");
        self.w("    SSL_free(s);\n");
        self.w("}\n");
        self.w("#else\n");
        self.w("static inline maka_unit* __maka_tls_client_new(int64_t fd, const char* hostname) { (void)fd; (void)hostname; return NULL; }\n");
        self.w("static inline int64_t __maka_tls_handshake(maka_unit* p) { (void)p; return -1; }\n");
        self.w("static inline int64_t __maka_tls_read(maka_unit* p, maka_unit* buf, int64_t cap) { (void)p; (void)buf; (void)cap; return -1; }\n");
        self.w("static inline int64_t __maka_tls_write(maka_unit* p, maka_unit* buf, int64_t len) { (void)p; (void)buf; (void)len; return -1; }\n");
        self.w("static inline void __maka_tls_close(maka_unit* p) { (void)p; }\n");
        self.w("#endif\n");
        // Unix domain sockets — bind/connect by path.  Use the same socket
        // forward decls + syscall machinery as the TCP helpers; AF_UNIX = 1.
        self.w("struct __maka_sockaddr_un { unsigned short sun_family; char sun_path[108]; };\n");
        self.w("#define __MAKA_AF_UNIX 1\n");
        self.w("static inline int64_t __maka_unix_listen(const char* path, int64_t backlog) {\n");
        self.w("    int s = socket(__MAKA_AF_UNIX, __MAKA_SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    struct __maka_sockaddr_un sa; memset(&sa, 0, sizeof(sa));\n");
        self.w("    sa.sun_family = __MAKA_AF_UNIX;\n");
        self.w("    size_t pl = strlen(path); if (pl >= sizeof(sa.sun_path)) pl = sizeof(sa.sun_path) - 1;\n");
        self.w("    memcpy(sa.sun_path, path, pl);\n");
        self.w("    unlink(path);  /* best-effort prior cleanup */\n");
        self.w("    if (bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) { close(s); return -1; }\n");
        self.w("    if (listen(s, (int)backlog) != 0) { close(s); return -1; }\n");
        self.w("    int flags = fcntl(s, F_GETFL, 0); fcntl(s, F_SETFL, flags | O_NONBLOCK);\n");
        self.w("    return s;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_unix_connect(const char* path) {\n");
        self.w("    int s = socket(__MAKA_AF_UNIX, __MAKA_SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    int flags = fcntl(s, F_GETFL, 0); fcntl(s, F_SETFL, flags | O_NONBLOCK);\n");
        self.w("    struct __maka_sockaddr_un sa; memset(&sa, 0, sizeof(sa));\n");
        self.w("    sa.sun_family = __MAKA_AF_UNIX;\n");
        self.w("    size_t pl = strlen(path); if (pl >= sizeof(sa.sun_path)) pl = sizeof(sa.sun_path) - 1;\n");
        self.w("    memcpy(sa.sun_path, path, pl);\n");
        self.w("    int r = connect(s, (struct sockaddr*)&sa, sizeof(sa));\n");
        self.w("    if (r == 0) return s;\n");
        self.w("    if (errno == EINPROGRESS) {\n");
        self.w("        __maka_wait_fd(s, MAKA_EV_WRITE);\n");
        self.w("        int err = 0; __maka_socklen_t elen = sizeof(err);\n");
        self.w("        getsockopt(s, __MAKA_SOL_SOCKET, __MAKA_SO_ERROR, &err, &elen);\n");
        self.w("        if (err == 0) return s;\n");
        self.w("    }\n");
        self.w("    close(s); return -1;\n");
        self.w("}\n");
        // File async IO via offload thread.  Each call spawns a one-shot
        // pthread that does the blocking pread/pwrite, then signals an
        // eventfd the calling fiber waits on.  Heavy per call (pthread
        // creation) but correct without an io_uring/AIO dependency.
        self.w("typedef struct {\n");
        self.w("    int fd;\n");
        self.w("    void* buf;\n");
        self.w("    int64_t len;\n");
        self.w("    int64_t offset;\n");
        self.w("    int64_t result;\n");
        self.w("    int efd;\n");
        self.w("    int is_write;\n");
        self.w("} __maka_aio_t;\n");
        self.w("extern long pread(int, void*, unsigned long, long);\n");
        self.w("extern long pwrite(int, const void*, unsigned long, long);\n");
        self.w("static void* __maka_aio_worker(void* arg) {\n");
        self.w("    __maka_aio_t* j = (__maka_aio_t*)arg;\n");
        self.w("    if (j->is_write) j->result = (int64_t)pwrite(j->fd, j->buf, (unsigned long)j->len, (long)j->offset);\n");
        self.w("    else             j->result = (int64_t)pread (j->fd, j->buf, (unsigned long)j->len, (long)j->offset);\n");
        self.w("    uint64_t v = 1; ssize_t w = write(j->efd, &v, sizeof(v)); (void)w;\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_file_read_async(int64_t fd, maka_unit* buf, int64_t cap, int64_t offset) {\n");
        self.w("    int efd = __maka_eventfd_create(0);\n");
        self.w("    if (efd < 0) return -1;\n");
        self.w("    __maka_aio_t* j = (__maka_aio_t*)calloc(1, sizeof(__maka_aio_t));\n");
        self.w("    j->fd = (int)fd; j->buf = (void*)buf; j->len = cap; j->offset = offset; j->efd = efd; j->is_write = 0;\n");
        self.w("    pthread_t t; pthread_create(&t, NULL, __maka_aio_worker, j); pthread_detach(t);\n");
        self.w("    (void)__maka_eventfd_recv(efd);\n");
        self.w("    int64_t r = j->result;\n");
        self.w("    free(j);\n");
        self.w("    close(efd);\n");
        self.w("    return r;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_file_write_async(int64_t fd, maka_unit* buf, int64_t len, int64_t offset) {\n");
        self.w("    int efd = __maka_eventfd_create(0);\n");
        self.w("    if (efd < 0) return -1;\n");
        self.w("    __maka_aio_t* j = (__maka_aio_t*)calloc(1, sizeof(__maka_aio_t));\n");
        self.w("    j->fd = (int)fd; j->buf = (void*)buf; j->len = len; j->offset = offset; j->efd = efd; j->is_write = 1;\n");
        self.w("    pthread_t t; pthread_create(&t, NULL, __maka_aio_worker, j); pthread_detach(t);\n");
        self.w("    (void)__maka_eventfd_recv(efd);\n");
        self.w("    int64_t r = j->result;\n");
        self.w("    free(j);\n");
        self.w("    close(efd);\n");
        self.w("    return r;\n");
        self.w("}\n");
        self.w("extern int open(const char*, int, ...);\n");
        self.w("#define __MAKA_O_RDONLY 0\n");
        self.w("#define __MAKA_O_WRONLY 1\n");
        self.w("#define __MAKA_O_RDWR   2\n");
        self.w("#define __MAKA_O_CREAT  64\n");
        self.w("#define __MAKA_O_TRUNC  512\n");
        self.w("static inline int64_t __maka_file_open(const char* path, int64_t flags, int64_t mode) {\n");
        self.w("    return (int64_t)open(path, (int)flags, (int)mode);\n");
        self.w("}\n");
        // DNS resolution via gethostbyname (legacy but doesn't require
        // pulling in netdb.h's struct addrinfo, which transitively brings
        // in sys/socket.h and conflicts with our forward decls).
        self.w("struct __maka_hostent { char* h_name; char** h_aliases; int h_addrtype; int h_length; char** h_addr_list; };\n");
        self.w("extern struct __maka_hostent* gethostbyname(const char*);\n");
        self.w("static inline int64_t __maka_dns_resolve_v4(const char* host) {\n");
        self.w("    struct __maka_hostent* he = gethostbyname(host);\n");
        self.w("    if (!he || !he->h_addr_list || !he->h_addr_list[0]) return -1;\n");
        self.w("    unsigned char* p = (unsigned char*)he->h_addr_list[0];\n");
        self.w("    return ((int64_t)p[0] << 24) | ((int64_t)p[1] << 16) | ((int64_t)p[2] << 8) | (int64_t)p[3];\n");
        self.w("}\n");
        // UDP helpers — bind a datagram socket, send/recv from a peer.
        self.w("static inline int64_t __maka_udp_open(int64_t port) {\n");
        self.w("    int s = socket(__MAKA_AF_INET, 2 /*SOCK_DGRAM*/, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa));\n");
        self.w("    sa.sin_family = __MAKA_AF_INET;\n");
        self.w("    sa.sin_addr.s_addr = htonl(__MAKA_INADDR_ANY);\n");
        self.w("    sa.sin_port = htons((unsigned short)port);\n");
        self.w("    if (port > 0 && bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) { close(s); return -1; }\n");
        self.w("    int flags = fcntl(s, F_GETFL, 0);\n");
        self.w("    fcntl(s, F_SETFL, flags | O_NONBLOCK);\n");
        self.w("    return s;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_udp_send_v4(int64_t fd, int64_t a, int64_t b, int64_t c, int64_t d, int64_t port, maka_unit* buf, int64_t len) {\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa));\n");
        self.w("    sa.sin_family = __MAKA_AF_INET;\n");
        self.w("    sa.sin_port = htons((unsigned short)port);\n");
        self.w("    unsigned char oct[4] = {(unsigned char)a, (unsigned char)b, (unsigned char)c, (unsigned char)d};\n");
        self.w("    memcpy(&sa.sin_addr.s_addr, oct, 4);\n");
        self.w("    while (1) {\n");
        self.w("        long n = sendto((int)fd, (void*)buf, (unsigned long)len, 0, (struct sockaddr*)&sa, sizeof(sa));\n");
        self.w("        if (n >= 0) return (int64_t)n;\n");
        self.w("        if (errno == EAGAIN || errno == EWOULDBLOCK) { __maka_wait_fd(fd, MAKA_EV_WRITE); continue; }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        // signalfd helpers (Linux-specific; non-Linux returns -1 from open).
        self.w("#ifdef __linux__\n");
        self.w("#include <sys/signalfd.h>\n");
        self.w("static inline int64_t __maka_signalfd_open(int64_t signum) {\n");
        self.w("    sigset_t mask; sigemptyset(&mask); sigaddset(&mask, (int)signum);\n");
        self.w("    pthread_sigmask(SIG_BLOCK, &mask, NULL);\n");
        self.w("    int fd = signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC);\n");
        self.w("    return fd;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_signalfd_recv(int64_t fd) {\n");
        self.w("    struct signalfd_siginfo si;\n");
        self.w("    while (1) {\n");
        self.w("        ssize_t n = read((int)fd, &si, sizeof(si));\n");
        self.w("        if (n == (ssize_t)sizeof(si)) return (int64_t)si.ssi_signo;\n");
        self.w("        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("#include <sys/timerfd.h>\n");
        self.w("#include <sys/eventfd.h>\n");
        self.w("#include <sys/inotify.h>\n");
        // inotify: kernel file-watch via reactor.  inotify_open creates an
        // inotify fd; inotify_add_path registers a path to watch.
        // inotify_recv_async yields until any event arrives, then returns
        // the first event's watch-descriptor (wd) or -1.
        self.w("static inline int64_t __maka_inotify_open(void) {\n");
        self.w("    return inotify_init1(IN_NONBLOCK | IN_CLOEXEC);\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_inotify_add(int64_t fd, const char* path, int64_t mask) {\n");
        self.w("    return inotify_add_watch((int)fd, path, (uint32_t)mask);\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_inotify_recv(int64_t fd) {\n");
        self.w("    char buf[sizeof(struct inotify_event) + 256] __attribute__((aligned(8)));\n");
        self.w("    while (1) {\n");
        self.w("        ssize_t n = read((int)fd, buf, sizeof(buf));\n");
        self.w("        if (n > 0) { struct inotify_event* e = (struct inotify_event*)buf; return (int64_t)e->wd; }\n");
        self.w("        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        // eventfd: kernel counter with read/write wakeups, useful for fiber
        // and thread coordination without a full socket pair.
        self.w("static inline int64_t __maka_eventfd_create(int64_t initial) {\n");
        self.w("    return eventfd((unsigned int)initial, EFD_NONBLOCK | EFD_CLOEXEC);\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_eventfd_signal(int64_t fd, int64_t n) {\n");
        self.w("    uint64_t v = (uint64_t)n;\n");
        self.w("    ssize_t w = write((int)fd, &v, sizeof(v));\n");
        self.w("    return (w == (ssize_t)sizeof(v)) ? 0 : -1;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_eventfd_recv(int64_t fd) {\n");
        self.w("    uint64_t v;\n");
        self.w("    while (1) {\n");
        self.w("        ssize_t r = read((int)fd, &v, sizeof(v));\n");
        self.w("        if (r == (ssize_t)sizeof(v)) return (int64_t)v;\n");
        self.w("        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_timerfd_create(int64_t initial_ns, int64_t interval_ns) {\n");
        self.w("    int fd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);\n");
        self.w("    if (fd < 0) return -1;\n");
        self.w("    struct itimerspec it = { .it_interval = { interval_ns / 1000000000LL, interval_ns % 1000000000LL },\n");
        self.w("                              .it_value    = { initial_ns / 1000000000LL,  initial_ns % 1000000000LL } };\n");
        self.w("    if (timerfd_settime(fd, 0, &it, NULL) != 0) { close(fd); return -1; }\n");
        self.w("    return fd;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_timerfd_recv(int64_t fd) {\n");
        self.w("    uint64_t exp;\n");
        self.w("    while (1) {\n");
        self.w("        ssize_t n = read((int)fd, &exp, sizeof(exp));\n");
        self.w("        if (n == (ssize_t)sizeof(exp)) return (int64_t)exp;\n");
        self.w("        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("#else\n");
        self.w("static inline int64_t __maka_signalfd_open(int64_t signum) { (void)signum; return -1; }\n");
        self.w("static inline int64_t __maka_signalfd_recv(int64_t fd) { (void)fd; return -1; }\n");
        self.w("static inline int64_t __maka_timerfd_create(int64_t a, int64_t b) { (void)a; (void)b; return -1; }\n");
        self.w("static inline int64_t __maka_timerfd_recv(int64_t fd) { (void)fd; return -1; }\n");
        self.w("static inline int64_t __maka_eventfd_create(int64_t initial) { (void)initial; return -1; }\n");
        self.w("static inline int64_t __maka_eventfd_signal(int64_t fd, int64_t n) { (void)fd; (void)n; return -1; }\n");
        self.w("static inline int64_t __maka_eventfd_recv(int64_t fd) { (void)fd; return -1; }\n");
        self.w("static inline int64_t __maka_inotify_open(void) { return -1; }\n");
        self.w("static inline int64_t __maka_inotify_add(int64_t fd, const char* p, int64_t m) { (void)fd; (void)p; (void)m; return -1; }\n");
        self.w("static inline int64_t __maka_inotify_recv(int64_t fd) { (void)fd; return -1; }\n");
        self.w("#endif\n");
        self.w("static inline int64_t __maka_udp_recv_async(int64_t fd, maka_unit* buf, int64_t cap) {\n");
        self.w("    while (1) {\n");
        self.w("        long n = recvfrom((int)fd, (void*)buf, (unsigned long)cap, 0, NULL, NULL);\n");
        self.w("        if (n >= 0) return (int64_t)n;\n");
        self.w("        if (errno == EAGAIN || errno == EWOULDBLOCK) { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        if (errno == EINTR) continue;\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
    }

    fn emit_func(&mut self, f: &HFunc) {
        // inline functions are spliced at each call site; no standalone C function emitted.
        if self.sym.func_sig(f.id).is_inline { return; }
        let sig = self.func_signature(f);
        self.wl(&format!("{} {{", sig));
        self.open();
        // For lifted capturing lambdas, emit env-extraction statements at the function entry.
        let s = self.sym.func_sig(f.id);
        if s.name.starts_with("__lambda_cap_") {
            // First param is __env; remaining params are the lambda's normal params.
            // The capture locals come AFTER the params in f.locals, named after the captured variables.
            // Initialize each capture local from __env->capture_name.
            // Identify the env struct id.
            let env_struct = if !f.params.is_empty() {
                let env_local = &f.locals[f.params[0].0 as usize];
                match &env_local.ty {
                    HType::Ref { inner, .. } => match inner.as_ref() {
                        HType::Struct(id) => Some(*id),
                        _ => None,
                    },
                    _ => None,
                }
            } else { None };
            if let Some(eid) = env_struct {
                let env_info = self.sym.struct_info(eid).clone();
                // Find capture locals: locals after the params that match env field names.
                let first_non_param = f.params.last().map(|l| l.0 + 1).unwrap_or(0);
                for cap_id in first_non_param..(f.locals.len() as u32) {
                    let li = &f.locals[cap_id as usize];
                    if env_info.fields.iter().any(|fi| fi.name == li.name) {
                        let lname = local_name(LocalId(cap_id), &li.name);
                        let ty = self.c_type(&li.ty);
                        self.wl(&format!("{} {} = __env_0->{};", ty, lname, c_ident(&li.name)));
                    }
                }
            }
        }
        self.emit_block(f, &f.body, true);
        self.close();
        self.wl("}");
        self.wl("");
    }

    fn emit_block(&mut self, f: &HFunc, b: &HBlock, is_top: bool) {
        if !is_top { self.wl("{"); self.open(); }
        for s in &b.stmts {
            self.emit_stmt(f, s);
        }
        // Null-collapse pointers whose deps just died.
        for id in &b.ptr_nulls {
            let li = &f.locals[id.0 as usize];
            self.wl(&format!("{} = NULL; /* collapse {} */", local_name(*id, &li.name), li.name));
        }
        // Free heap locals declared in this block (in reverse decl order).
        for id in &b.heap_to_free {
            let li = &f.locals[id.0 as usize];
            let n = local_name(*id, &li.name);
            let is_vec = matches!(&li.ty, HType::Heap { inner } if matches!(inner.as_ref(), HType::Vec { .. }));
            if is_vec {
                self.wl(&format!("free({}.data); /* drop heap vec {} */", n, li.name));
            } else {
                self.wl(&format!("free({}); /* drop heap {} */", n, li.name));
            }
        }
        if !is_top { self.close(); self.wl("}"); }
    }

    fn emit_stmt(&mut self, f: &HFunc, s: &HStmt) {
        match s {
            HStmt::Let { local, init, .. } => self.emit_let(f, *local, init),
            HStmt::Assign { op, place, value, .. } => self.emit_assign(f, *op, place, value),
            HStmt::ExprStmt(e) => {
                let s = self.emit_expr(f, e);
                self.wl(&format!("(void)({});", s));
            }
            HStmt::Return { value, heap_drops, .. } => {
                // Drop heap locals first.
                for id in heap_drops {
                    let li = &f.locals[id.0 as usize];
                    let n = local_name(*id, &li.name);
                    let is_vec = matches!(&li.ty, HType::Heap { inner } if matches!(inner.as_ref(), HType::Vec { .. }));
                    if is_vec {
                        self.wl(&format!("free({}.data); /* drop on return (vec) */", n));
                    } else {
                        self.wl(&format!("free({}); /* drop on return */", n));
                    }
                }
                match value {
                    Some(e) => {
                        let s = self.emit_expr(f, e);
                        if matches!(e.ty, HType::Unit) {
                            self.wl(&format!("(void)({}); return;", s));
                        } else {
                            self.wl(&format!("return {};", s));
                        }
                    }
                    None => self.wl("return;"),
                }
            }
            HStmt::If { cond, then_b, else_b, .. } => {
                let cs = self.emit_expr(f, cond);
                self.wl(&format!("if ({}) {{", cs));
                self.open();
                self.emit_block(f, then_b, true);
                self.close();
                if let Some(b) = else_b {
                    self.wl("} else {");
                    self.open();
                    self.emit_block(f, b, true);
                    self.close();
                }
                self.wl("}");
            }
            HStmt::While { cond, body, .. } => {
                let cs = self.emit_expr(f, cond);
                self.wl(&format!("while ({}) {{", cs));
                self.open();
                self.emit_block(f, body, true);
                self.close();
                self.wl("}");
            }
            HStmt::Block(b) => {
                self.wl("{");
                self.open();
                self.emit_block(f, b, true);
                self.close();
                self.wl("}");
            }
            HStmt::Unsafe(b, _) => {
                self.wl("/* unsafe */ {");
                self.open();
                self.emit_block(f, b, true);
                self.close();
                self.wl("}");
            }
            HStmt::Break(_) => self.wl("break;"),
            HStmt::Continue(_) => self.wl("continue;"),
            HStmt::Propagate { value, .. } => {
                // `propagate X;` ⇒ `return X;` from the enclosing C function.
                // Within an InlineCall expansion (which IS in the caller's C function), this exits
                // the caller, fulfilling the spec semantics.  `propagate;` (no value) emits a bare
                // `return;` — only valid when the caller returns `unit` (enforced by sema).
                match value {
                    Some(v) => {
                        let s = self.emit_expr(f, v);
                        self.wl(&format!("return {};", s));
                    }
                    None => self.wl("return;"),
                }
            }
            HStmt::ForC { init, cond, step, body, .. } => {
                self.emit_stmt(f, init);
                let cs = self.emit_expr(f, cond);
                let step_text = self.emit_step_expr(f, step);
                self.wl(&format!("for (; {}; {}) {{", cs, step_text));
                self.open();
                self.emit_block(f, body, true);
                self.close();
                self.wl("}");
            }
            HStmt::ForEach { var, src, body, .. } => {
                let li = &f.locals[var.0 as usize];
                let var_name = local_name(*var, &li.name);
                let var_ty = self.c_type(&li.ty);
                let src_c = self.emit_expr(f, src);
                self.wl("{");
                self.open();
                // Layout based on src type:
                let (len_str, elem_access) = match &src.ty {
                    HType::Array { len, .. } => {
                        // No copy — directly index the source array.
                        (format!("(maka_int){}", len), src_c.clone())
                    }
                    HType::Slice { .. } | HType::Vec { .. } => {
                        let src_key = self.type_key(&src.ty);
                        let _ = src_key;
                        let src_c_ty = self.c_type(&src.ty);
                        self.wl(&format!("{} __src = {};", src_c_ty, src_c));
                        if matches!(&src.ty, HType::Slice { .. }) {
                            ("__src.len".to_string(), "__src.ptr".to_string())
                        } else {
                            ("__src.len".to_string(), "__src.data".to_string())
                        }
                    }
                    HType::Heap { inner } if matches!(inner.as_ref(), HType::Vec { .. }) => {
                        let src_c_ty = self.c_type(&src.ty);
                        self.wl(&format!("{} __src = {};", src_c_ty, src_c));
                        ("__src.len".to_string(), "__src.data".to_string())
                    }
                    _ => {
                        ("0".to_string(), src_c.clone())
                    }
                };
                self.wl(&format!("{} {} = ({})0;", var_ty, var_name, var_ty));
                self.wl(&format!("for (maka_int __i = 0; __i < {}; __i += 1) {{", len_str));
                self.open();
                self.wl(&format!("{} = {}[__i];", var_name, elem_access));
                self.emit_block(f, body, true);
                self.close();
                self.wl("}");
                self.close();
                self.wl("}");
            }
        }
    }

    fn emit_step_expr(&mut self, f: &HFunc, step: &HStmt) -> String {
        if let HStmt::Assign { op, place, value, .. } = step {
            let p = self.emit_place(f, place);
            let v = self.emit_expr(f, value);
            return format!("{} {} {}", p, assign_op_c(*op), v);
        }
        "(void)0".into()
    }

    /// Expand an `inline` call as a GCC statement-expression. The body's `return X` becomes
    /// `__result = X; break;` (exiting a do-while), and `propagate X` becomes a real C `return X;`
    /// from the *outer* function (the C function that contains this expansion).
    fn emit_inline_expansion(&mut self, caller_f: &HFunc, callee: FuncId, args: &[HExpr], result_ty: &HType) -> String {
        // Find the inline HFunc by id.
        let inline_f = match self.sym.funcs.iter().find(|hf| hf.id == callee).cloned() {
            Some(hf) => hf,
            None => return "MAKA_UNIT".into(),
        };
        // Generate a unique tag for this expansion so local name renaming is unique.
        // (Stable enough: id + caller's name + arg count.)
        self.inline_call_seq += 1;
        let tag = format!("ix{}_{}", inline_f.id.0, self.inline_call_seq);
        let needs_value = !matches!(result_ty, HType::Unit);

        let mut out = String::new();
        out.push_str("__extension__ ({ ");

        // Emit argument-binding locals: for each param, declare a local of the param's type initialized with the arg expr.
        // Each inline-local needs a unique name in the caller's C function — append `_{tag}` to avoid collisions.
        let arg_strs: Vec<String> = args.iter().map(|a| self.emit_expr(caller_f, a)).collect();
        for (i, &pid) in inline_f.params.iter().enumerate() {
            let li = &inline_f.locals[pid.0 as usize];
            let pname = inline_local_name(&inline_f, pid, &tag);
            let ty = self.c_type(&li.ty);
            let arg_s = arg_strs.get(i).cloned().unwrap_or_else(|| "0".into());
            out.push_str(&format!("{} {} = {}; ", ty, pname, arg_s));
        }

        // Declare result var if needed.
        if needs_value {
            let res_c = self.c_type(result_ty);
            out.push_str(&format!("{} __r_{} = ({})0; ", res_c, tag, res_c));
        }

        // Declare all non-param locals up front so they're in scope for the whole expansion.
        for (i, li) in inline_f.locals.iter().enumerate() {
            let id = LocalId(i as u32);
            if inline_f.params.contains(&id) { continue; }
            let pname = inline_local_name(&inline_f, id, &tag);
            let ty = self.c_type(&li.ty);
            match &li.ty {
                HType::Array { .. } => {
                    out.push_str(&format!("{} = {{0}}; ", self.c_decl(&li.ty, &pname)));
                }
                _ => out.push_str(&format!("{} {} = ({})0; ", ty, pname, ty)),
            }
        }

        // Emit the body with rewrites.  Using `do/while(0)` + `break;` for `return`
        // would silently exit a user-written loop instead of the inline expansion when
        // `return` fires from inside a `while`/`for`.  Use a labeled goto so the
        // semantics of `return` are independent of any surrounding loops.
        out.push_str("do { ");
        for stmt in &inline_f.body.stmts {
            out.push_str(&self.emit_inline_stmt(&inline_f, stmt, &tag, result_ty));
        }
        out.push_str("} while (0); ");
        out.push_str(&format!("end_{}: ; ", tag));

        // Auto-free any heap/own locals the inline owns.  Ownership transferred
        // in via parameters lands here too: the lifetime pass marks the OUTER
        // binding as moved at the InlineCall site, so the outer scope-exit drop
        // is suppressed and the only free happens here, inside the splice.
        for id in &inline_f.body.heap_to_free {
            let li = &inline_f.locals[id.0 as usize];
            let n = inline_local_name(&inline_f, *id, &tag);
            let is_vec = matches!(&li.ty, HType::Heap { inner } if matches!(inner.as_ref(), HType::Vec { .. }));
            if is_vec {
                out.push_str(&format!("free({}.data); ", n));
            } else {
                out.push_str(&format!("free({}); ", n));
            }
        }

        if needs_value { out.push_str(&format!("__r_{}; ", tag)); }
        else { out.push_str("MAKA_UNIT; "); }
        out.push_str("})");
        out
    }

    /// Emit a statement from the inline function's body, with substitutions.
    fn emit_inline_stmt(&mut self, inline_f: &HFunc, s: &HStmt, tag: &str, result_ty: &HType) -> String {
        match s {
            HStmt::Let { local, init, .. } => {
                let li = &inline_f.locals[local.0 as usize];
                let n = inline_local_name(inline_f, *local, tag);
                let v = self.emit_inline_expr(inline_f, init, tag);
                match &li.ty {
                    HType::Heap { inner } if !matches!(inner.as_ref(), HType::Vec { .. }) => {
                        let ic = self.c_type(inner);
                        format!("{} = ({}*)malloc(sizeof({})); *{} = ({}); ", n, ic, ic, n, v)
                    }
                    _ => format!("{} = {}; ", n, v),
                }
            }
            HStmt::Assign { op, place, value, .. } => {
                let p = self.emit_inline_place(inline_f, place, tag);
                let v = self.emit_inline_expr(inline_f, value, tag);
                format!("{} {} {}; ", p, assign_op_c(*op), v)
            }
            HStmt::ExprStmt(e) => {
                let v = self.emit_inline_expr(inline_f, e, tag);
                format!("(void)({}); ", v)
            }
            HStmt::Return { value, .. } => {
                let needs = !matches!(result_ty, HType::Unit);
                let mut s = String::new();
                if let Some(v) = value {
                    let vs = self.emit_inline_expr(inline_f, v, tag);
                    if needs { s.push_str(&format!("__r_{} = ({}); ", tag, vs)); }
                    else { s.push_str(&format!("(void)({}); ", vs)); }
                }
                s.push_str(&format!("goto end_{}; ", tag));
                s
            }
            HStmt::Propagate { value, .. } => match value {
                Some(v) => {
                    let s = self.emit_inline_expr(inline_f, v, tag);
                    format!("return {}; ", s)
                }
                None => "return; ".into(),
            },
            HStmt::If { cond, then_b, else_b, .. } => {
                let cs = self.emit_inline_expr(inline_f, cond, tag);
                let mut s = format!("if ({}) {{ ", cs);
                for st in &then_b.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                s.push_str("} ");
                if let Some(b) = else_b {
                    s.push_str("else { ");
                    for st in &b.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                    s.push_str("} ");
                }
                s
            }
            HStmt::While { cond, body, .. } => {
                let cs = self.emit_inline_expr(inline_f, cond, tag);
                let mut s = format!("while ({}) {{ ", cs);
                for st in &body.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                s.push_str("} ");
                s
            }
            HStmt::Block(b) => {
                let mut s = String::from("{ ");
                for st in &b.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                s.push_str("} ");
                s
            }
            HStmt::Unsafe(b, _) => {
                let mut s = String::from("/* unsafe */ { ");
                for st in &b.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                s.push_str("} ");
                s
            }
            HStmt::Break(_) => "break; ".into(),
            HStmt::Continue(_) => "continue; ".into(),
            HStmt::ForC { init, cond, step, body, .. } => {
                let mut s = self.emit_inline_stmt(inline_f, init, tag, result_ty);
                let cs = self.emit_inline_expr(inline_f, cond, tag);
                let step_text = match step.as_ref() {
                    HStmt::Assign { op, place, value, .. } => {
                        let p = self.emit_inline_place(inline_f, place, tag);
                        let v = self.emit_inline_expr(inline_f, value, tag);
                        format!("{} {} {}", p, assign_op_c(*op), v)
                    }
                    _ => "(void)0".into(),
                };
                s.push_str(&format!("for (; {}; {}) {{ ", cs, step_text));
                for st in &body.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                s.push_str("} ");
                s
            }
            HStmt::ForEach { .. } => {
                "/* for-each in inline not supported */ ".into()
            }
        }
    }

    /// Emit an expression from the inline body, renaming local references with the tag.
    fn emit_inline_expr(&mut self, inline_f: &HFunc, e: &HExpr, tag: &str) -> String {
        match &e.kind {
            HExprKind::Local(id) => inline_local_name(inline_f, *id, tag),
            HExprKind::Bin { op, lhs, rhs } => {
                let l = self.emit_inline_expr(inline_f, lhs, tag);
                let r = self.emit_inline_expr(inline_f, rhs, tag);
                format!("(({}) {} ({}))", l, binop_c(*op), r)
            }
            HExprKind::Un { op, expr } => {
                let v = self.emit_inline_expr(inline_f, expr, tag);
                match op {
                    HUnOp::Neg => format!("(-({}))", v),
                    HUnOp::Not => format!("(!({}))", v),
                }
            }
            HExprKind::Unwrap { expr, skip_check: _ } => {
                let s = self.emit_inline_expr(inline_f, expr, tag);
                format!("(*({}))", s)
            }
            HExprKind::AddrOfRef { place, .. } => {
                if matches!(place.ty, HType::Dyn { .. }) {
                    return self.emit_inline_place(inline_f, place, tag);
                }
                let p = self.emit_inline_place(inline_f, place, tag);
                format!("(&({}))", p)
            }
            HExprKind::Field { base, field } => {
                let bs = self.emit_inline_expr(inline_f, base, tag);
                let (arrow, fname) = self.field_access(inline_f, base, *field);
                format!("({}{}{})", bs, arrow, fname)
            }
            HExprKind::Index { base, idx } => {
                let bs = self.emit_inline_expr(inline_f, base, tag);
                let is_ = self.emit_inline_expr(inline_f, idx, tag);
                self.index_access(inline_f, base, &bs, &is_)
            }
            HExprKind::DerefRef(inner) => {
                let s = self.emit_inline_expr(inline_f, inner, tag);
                format!("(*({}))", s)
            }
            HExprKind::HeapAlloc(inner) => {
                let ic = self.c_type(&inner.ty);
                let v = self.emit_inline_expr(inline_f, inner, tag);
                format!("(__extension__ ({{ {0}* __p = ({0}*)malloc(sizeof({0})); *__p = ({1}); __p; }}))", ic, v)
            }
            // Everything else: fall back to ordinary emit_expr but with a dummy HFunc that holds
            // the inline locals so name lookups resolve. For simplicity, use the inline_f directly.
            _ => self.emit_expr_with_tag(inline_f, e, tag),
        }
    }

    /// emit_expr but with local-name rewriting via `tag` for the given inline function's locals.
    fn emit_expr_with_tag(&mut self, inline_f: &HFunc, e: &HExpr, tag: &str) -> String {
        // For ad-hoc reuse: temporarily swap the local-name rendering. Simplest: emit normally
        // through self.emit_expr but recurse into more handlers. Most expressions have already been
        // covered above; this fallback handles literals and calls.
        match &e.kind {
            HExprKind::LitInt(n) => format!("(maka_int){}LL", n),
            HExprKind::LitFloat(v) => format!("(maka_float){}", v),
            HExprKind::LitBool(b) => if *b { "true".into() } else { "false".into() },
            HExprKind::LitChar(c) => format!("(maka_char){}u", *c as u32),
            HExprKind::LitStr(s) => format!("\"{}\"", c_escape(s)),
            HExprKind::LitNull => "NULL".into(),
            HExprKind::LitUnit => "MAKA_UNIT".into(),
            HExprKind::Call { callee, args } => {
                if callee.0 == u32::MAX - 2 {
                    if let Some(a) = args.first() {
                        let s = self.emit_inline_expr(inline_f, a, tag);
                        return format!("(maka_panic({}), MAKA_UNIT)", s);
                    }
                    return "(maka_panic(\"\"), MAKA_UNIT)".into();
                }
                if callee.0 == u32::MAX - 1 {
                    if let Some(a) = args.first() {
                        let s = self.emit_inline_expr(inline_f, a, tag);
                        return format!("(free((void*)({})), MAKA_UNIT)", s);
                    }
                    return "MAKA_UNIT".into();
                }
                if callee.0 == u32::MAX {
                    if args.len() == 1 {
                        let s = self.emit_inline_expr(inline_f, &args[0], tag);
                        let helper = self.log_helper(&args[0].ty);
                        return format!("({}({}), MAKA_UNIT)", helper, s);
                    }
                    return "MAKA_UNIT".into();
                }
                if callee.0 == u32::MAX - 3 {
                    if let Some(a) = args.first() {
                        let s = self.emit_inline_expr(inline_f, a, tag);
                        return format!("(Thread*)__maka_spawn_fiber(({}).code, ({}).env)", s, s);
                    }
                    return "NULL".into();
                }
                if callee.0 == u32::MAX - 15 {
                    if let Some(a) = args.first() {
                        let s = self.emit_inline_expr(inline_f, a, tag);
                        return format!("(Thread*)__maka_spawn_thread(({}).code, ({}).env)", s, s);
                    }
                    return "NULL".into();
                }
                if callee.0 == u32::MAX - 16 {
                    if let Some(a) = args.first() {
                        let s = self.emit_inline_expr(inline_f, a, tag);
                        return format!("(Thread*)__maka_spawn_job(({}).code, ({}).env)", s, s);
                    }
                    return "NULL".into();
                }
                if callee.0 == u32::MAX - 4 {
                    if let Some(a) = args.first() {
                        let s = self.emit_inline_expr(inline_f, a, tag);
                        return format!("(__maka_join((maka_unit*)({})), MAKA_UNIT)", s);
                    }
                    return "MAKA_UNIT".into();
                }
                if let Some(fname) = match callee.0 {
                    v if v == u32::MAX - 5 => Some("__maka_str_concat"),
                    v if v == u32::MAX - 8 => Some("__maka_str_concat_freel"),
                    v if v == u32::MAX - 9 => Some("__maka_str_concat_freer"),
                    v if v == u32::MAX - 10 => Some("__maka_str_concat_freeb"),
                    _ => None,
                } {
                    if args.len() == 2 {
                        let a = self.emit_inline_expr(inline_f, &args[0], tag);
                        let b = self.emit_inline_expr(inline_f, &args[1], tag);
                        return format!("{}({}, {})", fname, a, b);
                    }
                    return "((char*)0)".into();
                }
                if let Some(fname) = match callee.0 {
                    v if v == u32::MAX - 11 => Some("__maka_int_to_str"),
                    v if v == u32::MAX - 12 => Some("__maka_bool_to_str"),
                    v if v == u32::MAX - 13 => Some("__maka_float_to_str"),
                    v if v == u32::MAX - 14 => Some("__maka_char_to_str"),
                    _ => None,
                } {
                    if args.len() == 1 {
                        let a = self.emit_inline_expr(inline_f, &args[0], tag);
                        return format!("{}({})", fname, a);
                    }
                    return "((char*)0)".into();
                }
                if callee.0 == u32::MAX - 6 {
                    return "__maka_read_line()".into();
                }
                if callee.0 == u32::MAX - 7 {
                    return "__maka_read_int()".into();
                }
                let sig = self.sym.func_sig(*callee);
                let name = if sig.is_extern { sig.c_name.clone() }
                    else if sig.name == "main" && sig.logic.is_none() { "maka_main".to_string() }
                    else { c_ident(&sig.c_name) };
                let arg_s: Vec<String> = args.iter().map(|a| self.emit_inline_expr(inline_f, a, tag)).collect();
                let call = format!("{}({})", name, arg_s.join(", "));
                if matches!(sig.ret, HType::Unit) {
                    format!("({}, MAKA_UNIT)", call)
                } else { call }
            }
            HExprKind::Cast { expr, kind, to } => {
                if let CastKind::ToDyn { trait_name, struct_id } = kind {
                    return self.emit_to_dyn(inline_f, expr, trait_name, *struct_id);
                }
                let s = self.emit_inline_expr(inline_f, expr, tag);
                self.emit_cast(s, kind.clone(), to)
            }
            HExprKind::Struct { id, fields } => {
                let info = self.sym.struct_info(*id);
                let parts: Vec<String> = info.fields.iter().enumerate().map(|(i, f0)| {
                    if let Some((_, fe)) = fields.iter().find(|(j, _)| *j == i) {
                        let s = self.emit_inline_expr(inline_f, fe, tag);
                        format!(".{} = {}", c_ident(&f0.name), s)
                    } else {
                        format!(".{} = NULL", c_ident(&f0.name))
                    }
                }).collect();
                format!("(({}){{ {} }})", c_ident(&info.name), parts.join(", "))
            }
            HExprKind::VariantCtor { enum_id, variant, fields } => {
                let info = self.sym.enum_info(*enum_id);
                let v = &info.variants[*variant];
                if v.fields.is_empty() {
                    if info.is_simple() {
                        return format!("{}__{}", c_ident(&info.name), c_ident(&v.name));
                    }
                    return format!("(({0}){{ .tag = {1} }})", c_ident(&info.name), v.tag);
                }
                let parts: Vec<String> = v.fields.iter().enumerate().map(|(i, fi)| {
                    let s = fields.iter().find(|(idx, _)| *idx == i)
                        .map(|(_, e)| self.emit_inline_expr(inline_f, e, tag))
                        .unwrap_or_else(|| "0".into());
                    format!(".{} = {}", c_ident(&fi.name), s)
                }).collect();
                format!("(({0}){{ .tag = {1}, .payload.{2} = {{ {3} }} }})",
                    c_ident(&info.name), v.tag, c_ident(&v.name), parts.join(", "))
            }
            HExprKind::Match { scrutinee, arms, result_ty } => {
                // Simplified: emit normally via the inline_f context, where locals already get
                // their tagged names through emit_inline_expr.
                let _ = (scrutinee, arms, result_ty);
                self.emit_match_inline(inline_f, e, tag)
            }
            // Fallback: ordinary emit_expr (uses inline_f as the function context, which contains
            // the local-name table; matching is fine here).
            _ => self.emit_expr(inline_f, e),
        }
    }

    fn emit_match_inline(&mut self, inline_f: &HFunc, e: &HExpr, tag: &str) -> String {
        // Reuse standard emit_match but inline_f is passed as `f`. Local names within inline_f
        // get rendered via local_name (without tag), which would be wrong for the splice. To keep
        // this simple we restrict: inline match expressions must reference only their own locals
        // (no captures from outer scope), and the tag isn't needed inside the match.
        let _ = tag;
        self.emit_expr(inline_f, e)
    }

    fn emit_inline_place(&mut self, inline_f: &HFunc, e: &HExpr, tag: &str) -> String {
        match &e.kind {
            HExprKind::Local(id) => inline_local_name(inline_f, *id, tag),
            HExprKind::Field { base, field } => {
                let bs = self.emit_inline_expr(inline_f, base, tag);
                let (arrow, fname) = self.field_access(inline_f, base, *field);
                format!("{}{}{}", bs, arrow, fname)
            }
            HExprKind::Index { base, idx } => {
                let bs = self.emit_inline_expr(inline_f, base, tag);
                let is_ = self.emit_inline_expr(inline_f, idx, tag);
                self.index_access(inline_f, base, &bs, &is_)
            }
            HExprKind::Unwrap { expr, skip_check: _ } => {
                let s = self.emit_inline_expr(inline_f, expr, tag);
                format!("(*({}))", s)
            }
            _ => self.emit_inline_expr(inline_f, e, tag),
        }
    }

    fn emit_let(&mut self, f: &HFunc, id: LocalId, init: &HExpr) {
        let li = &f.locals[id.0 as usize];
        let name = local_name(id, &li.name);
        match &li.ty {
            HType::Heap { inner } => {
                // heap [*]T: emit as plain Vec_T struct, no extra heap layer.
                if matches!(inner.as_ref(), HType::Vec { .. }) {
                    let value_s = self.emit_expr(f, init);
                    self.wl(&format!("{} {} = {};", self.c_type(inner), name, value_s));
                    return;
                }
                // Detect move from another heap local: init is HExprKind::Local pointing at a heap local.
                if let HExprKind::Local(src) = init.kind {
                    if matches!(f.locals[src.0 as usize].ty, HType::Heap { .. }) {
                        // move
                        self.wl(&format!("{}* {} = {}; /* moved */", self.c_type(inner), name, local_name(src, &f.locals[src.0 as usize].name)));
                        return;
                    }
                }
                // Function call returning heap T: also transferred ownership; capture pointer directly.
                if let HExprKind::Call { .. } = init.kind {
                    if matches!(init.ty, HType::Heap { .. }) {
                        let s = self.emit_expr(f, init);
                        self.wl(&format!("{}* {} = {};", self.c_type(inner), name, s));
                        return;
                    }
                }
                // New allocation: value expression of type `T` lifted into heap slot.
                let value_s = self.emit_expr(f, init);
                self.wl(&format!("{}* {} = ({}*)malloc(sizeof({}));", self.c_type(inner), name, self.c_type(inner), self.c_type(inner)));
                self.wl(&format!("*{} = {};", name, value_s));
            }
            HType::Array { .. } => {
                let value_s = self.emit_expr(f, init);
                self.wl(&format!("{} = {};", self.c_decl(&li.ty, &name), value_s));
            }
            _ => {
                let value_s = self.emit_expr(f, init);
                let prefix = if li.thread_local { "static __thread " } else { "" };
                self.wl(&format!("{}{} = {};", prefix, self.c_decl(&li.ty, &name), value_s));
            }
        }
    }

    fn emit_assign(&mut self, f: &HFunc, op: HAssignOp, place: &HExpr, value: &HExpr) {
        let lhs = self.emit_place(f, place);
        let rhs = self.emit_expr(f, value);

        // Heap reassignment: replace value in slot (§6.6).
        if let HExprKind::Local(id) = place.kind {
            let lty = &f.locals[id.0 as usize].ty;
            if matches!(lty, HType::Heap { .. }) {
                // *p = newval;
                self.wl(&format!("*{} = {};", lhs, rhs));
                return;
            }
            // Write through a `&mut T` local (e.g. a by-mut-ref captured binding inside a closure).
            if matches!(lty, HType::Ref { mutable: true, .. }) {
                self.wl(&format!("*{} {} {};", lhs, assign_op_c(op), rhs));
                return;
            }
        }
        // Pointer unwrap on LHS like `p! = v` → `*p = v;`
        if let HExprKind::Unwrap { expr, skip_check: _ } = &place.kind {
            let inner = self.emit_expr(f, expr);
            self.wl(&format!("*({}) {} {};", inner, assign_op_c(op), rhs));
            return;
        }
        self.wl(&format!("{} {} {};", lhs, assign_op_c(op), rhs));
    }

    /// Emit an lvalue suitable for the LHS of an assignment.
    fn emit_place(&mut self, f: &HFunc, e: &HExpr) -> String {
        match &e.kind {
            HExprKind::Local(id) => local_name(*id, &f.locals[id.0 as usize].name),
            HExprKind::GlobalRef(gid) => self.sym.globals[gid.0 as usize].c_name.clone(),
            HExprKind::Field { base, field } => {
                let base_s = self.emit_expr(f, base);
                let (arrow, fname) = self.field_access(f, base, *field);
                format!("{}{}{}", base_s, arrow, fname)
            }
            HExprKind::Index { base, idx } => {
                let base_s = self.emit_expr(f, base);
                let idx_s = self.emit_expr(f, idx);
                self.index_access(f, base, &base_s, &idx_s)
            }
            HExprKind::Unwrap { expr, skip_check: _ } => {
                let inner = self.emit_expr(f, expr);
                format!("(*({}))", inner)
            }
            _ => self.emit_expr(f, e),
        }
    }

    fn field_access(&self, f: &HFunc, base: &HExpr, field: usize) -> (&'static str, String) {
        fn peel(t: &HType) -> StructId {
            match t {
                HType::Struct(id) => *id,
                HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } | HType::Heap { inner } => peel(inner),
                _ => panic!("field access on non-struct: {:?}", t),
            }
        }
        let sid = peel(&base.ty);
        let info = self.sym.struct_info(sid);
        let fname = c_ident(&info.fields[field].name);
        let arrow = match base.ty {
            HType::Ref { .. } | HType::Ptr { .. } | HType::RawPtr { .. } | HType::OwnPtr { .. } | HType::Heap { .. } => "->",
            _ => ".",
        };
        let _ = f;
        (arrow, fname)
    }

    fn index_access(&self, _f: &HFunc, base: &HExpr, base_s: &str, idx_s: &str) -> String {
        match &base.ty {
            HType::Array { len, .. } => {
                format!("(({})[maka_check_idx((maka_int)({}), (maka_int){}, \"array idx\")])", base_s, idx_s, len)
            }
            HType::Slice { .. } => {
                format!("(({}).ptr[maka_check_idx((maka_int)({}), ({}).len, \"slice idx\")])", base_s, idx_s, base_s)
            }
            HType::Vec { .. } => {
                format!("(({}).data[maka_check_idx((maka_int)({}), ({}).len, \"vec idx\")])", base_s, idx_s, base_s)
            }
            HType::Heap { inner } => match inner.as_ref() {
                HType::Array { len, .. } => format!("((*{})[maka_check_idx((maka_int)({}), (maka_int){}, \"array idx\")])", base_s, idx_s, len),
                // heap [*]T: base is Vec_T (no deref)
                HType::Vec { .. } => format!("(({}).data[maka_check_idx((maka_int)({}), ({}).len, \"vec idx\")])", base_s, idx_s, base_s),
                _ => format!("(({})[{}])", base_s, idx_s),
            },
            _ => format!("(({})[{}])", base_s, idx_s),
        }
    }

    fn emit_expr(&mut self, f: &HFunc, e: &HExpr) -> String {
        match &e.kind {
            HExprKind::LitInt(n) => format!("(maka_int){}LL", n),
            HExprKind::LitFloat(v) => format!("(maka_float){}", v),
            HExprKind::LitBool(b) => if *b { "true".into() } else { "false".into() },
            HExprKind::LitChar(c) => format!("(maka_char){}u", *c as u32),
            HExprKind::LitStr(s) => format!("\"{}\"", c_escape(s)),
            HExprKind::LitNull => "NULL".into(),
            HExprKind::LitUnit => "MAKA_UNIT".into(),
            HExprKind::Local(id) => local_name(*id, &f.locals[id.0 as usize].name),
            HExprKind::GlobalRef(gid) => self.sym.globals[gid.0 as usize].c_name.clone(),
            HExprKind::CallIndirect { callee, args } => {
                let c = self.emit_expr(f, callee);
                let mut arg_strs: Vec<String> = Vec::new();
                // First arg passed to the raw fn ptr is the captured env (NULL for plain fns).
                // Use a temp so we don't re-evaluate the callee expression.
                let mut all_args = String::from("__c.env");
                for a in args {
                    let s = self.emit_expr(f, a);
                    arg_strs.push(s.clone());
                    all_args.push_str(", ");
                    all_args.push_str(&s);
                }
                let ret_ty = match &callee.ty {
                    HType::FnPtr { ret, .. } => (**ret).clone(),
                    _ => HType::Unit,
                };
                let unit_pad = if matches!(ret_ty, HType::Unit) { ", MAKA_UNIT" } else { "" };
                let key = match &callee.ty {
                    HType::FnPtr { ret, params } => fn_sig_key(ret, params),
                    _ => "x".to_string(),
                };
                if matches!(ret_ty, HType::Unit) {
                    format!("(__extension__ ({{ Callable_{0} __c = ({1}); __c.code({2}){3}; }}))", key, c, all_args, unit_pad)
                } else {
                    format!("(__extension__ ({{ Callable_{0} __c = ({1}); __c.code({2}); }}))", key, c, all_args)
                }
            }
            HExprKind::InlineCall { callee, args } => {
                self.emit_inline_expansion(f, *callee, args, &e.ty)
            }
            HExprKind::Transfer(inner) => self.emit_expr(f, inner),
            HExprKind::EnumTag(inner) => {
                let s = self.emit_expr(f, inner);
                if let HType::Enum(eid) = &inner.ty {
                    if self.sym.enum_info(*eid).is_simple() {
                        // Simple enum: the C value IS the tag.
                        return format!("(maka_int)({})", s);
                    }
                }
                format!("(maka_int)(({}).tag)", s)
            }
            HExprKind::SliceLen(inner) => {
                let s = self.emit_expr(f, inner);
                match &inner.ty {
                    HType::Slice { .. } => format!("({}).len", s),
                    HType::Vec { .. } => format!("({}).len", s),
                    HType::Heap { inner: i } => match i.as_ref() {
                        HType::Vec { .. } => format!("({}).len", s),
                        HType::Array { len, .. } => format!("(maka_int){}", len),
                        _ => "0".into(),
                    },
                    HType::Array { len, .. } => format!("(maka_int){}", len),
                    _ => "0".into(),
                }
            }
            HExprKind::Closure { lifted, env_struct, env_values } => {
                let sig = self.sym.func_sig(*lifted);
                // We need the callable's fn-pointer signature to emit Callable_KEY{...}.
                // The lambda's fn-ptr type is (lifted.ret, lifted_params_minus_env)
                // where lifted_params_minus_env is everything after the env param.
                let lambda_params: Vec<HType> = sig.param_tys.iter().skip(1).cloned().collect();
                let key = fn_sig_key(&sig.ret, &lambda_params);
                let env_struct_name = c_ident(&self.sym.struct_info(*env_struct).name);
                let env_fields_info = self.sym.struct_info(*env_struct).fields.clone();
                let mut field_inits = Vec::new();
                for (i, fi) in env_fields_info.iter().enumerate() {
                    let v = env_values.get(i)
                        .map(|e| self.emit_expr(f, e))
                        .unwrap_or_else(|| "0".into());
                    field_inits.push(format!(".{} = {}", c_ident(&fi.name), v));
                }
                let lifted_c_name = c_ident(&sig.c_name);
                format!(
                    "(__extension__ ({{ {0}* __env = ({0}*)malloc(sizeof({0})); *__env = ({0}){{ {1} }}; (Callable_{2}){{ .code = (fn_{2}_raw){3}_closure_trampoline, .env = __env }}; }}))",
                    env_struct_name, field_inits.join(", "), key, lifted_c_name
                )
            }
            HExprKind::FnRef(fid) => {
                let sig = self.sym.func_sig(*fid);
                let target_name = if sig.is_extern { sig.c_name.clone() }
                    else if sig.name == "main" && sig.logic.is_none() { "maka_main".to_string() }
                    else { c_ident(&sig.c_name) };
                let key = fn_sig_key(&sig.ret, &sig.param_tys);
                let tram_name = format!("__tramp_{}", target_name);
                format!("((Callable_{0}){{ .code = (fn_{0}_raw){1}, .env = NULL }})", key, tram_name)
            }
            HExprKind::EnumVariant(eid, vi) => {
                let info = self.sym.enum_info(*eid);
                let v = &info.variants[*vi];
                // For simple enums, the variant is its integer tag constant.
                if info.is_simple() {
                    format!("{}__{}", c_ident(&info.name), c_ident(&v.name))
                } else {
                    // For tagged enums with payload, this is a constructor — only the tag value
                    // is materialized here when used directly (no payload data).
                    format!("(({0}){{ .tag = {1} }})", c_ident(&info.name), v.tag)
                }
            }
            HExprKind::Bin { op, lhs, rhs } => {
                let l = self.emit_expr(f, lhs);
                let r = self.emit_expr(f, rhs);
                let opc = binop_c(*op);
                // Tagged enum eq/ne: compare `.tag` on each side.  Simple
                // (payload-less) enums are already represented as integers in
                // C, so a direct compare works there.
                if matches!(op, HBinOp::Eq | HBinOp::Ne) {
                    if let (HType::Enum(le), HType::Enum(re)) = (&lhs.ty, &rhs.ty) {
                        if le == re && !self.sym.enum_info(*le).is_simple() {
                            return format!("(({}).tag {} ({}).tag)", l, opc, r);
                        }
                    }
                }
                format!("(({}) {} ({}))", l, opc, r)
            }
            HExprKind::Un { op, expr } => {
                let v = self.emit_expr(f, expr);
                match op {
                    HUnOp::Neg => format!("(-({}))", v),
                    HUnOp::Not => format!("(!({}))", v),
                }
            }
            HExprKind::Unwrap { expr, skip_check: _ } => {
                let s = self.emit_expr(f, expr);
                format!("(*({}))", s)
            }
            HExprKind::AddrOfRef { place, .. } => {
                // For dyn fat-pointers, `&m` is just `m` (the fat pointer already encapsulates indirection).
                if matches!(place.ty, HType::Dyn { .. }) {
                    return self.emit_place(f, place);
                }
                let p = self.emit_place(f, place);
                format!("(&({}))", p)
            }
            HExprKind::Field { base, field } => {
                let bs = self.emit_expr(f, base);
                let (arrow, fname) = self.field_access(f, base, *field);
                format!("({}{}{})", bs, arrow, fname)
            }
            HExprKind::Index { base, idx } => {
                let bs = self.emit_expr(f, base);
                let is_ = self.emit_expr(f, idx);
                self.index_access(f, base, &bs, &is_)
            }
            HExprKind::Call { callee, args } => {
                // Built-in `panic(msg)`.
                if callee.0 == u32::MAX - 2 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(maka_panic({}), MAKA_UNIT)", s);
                    }
                    return "(maka_panic(\"\"), MAKA_UNIT)".into();
                }
                // Built-in `free`
                if callee.0 == u32::MAX - 1 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(free((void*)({})), MAKA_UNIT)", s);
                    }
                    return "MAKA_UNIT".into();
                }
                // Built-in `log`
                if callee.0 == u32::MAX {
                    // Dispatch by first arg type
                    if args.len() == 1 {
                        let s = self.emit_expr(f, &args[0]);
                        let helper = self.log_helper(&args[0].ty);
                        return format!("({}({}), MAKA_UNIT)", helper, s);
                    }
                    return "MAKA_UNIT".into();
                }
                // Built-in `spawn(closure)` — fiber tier.  Compound-stmt expr
                // wraps the closure once so its env malloc happens exactly once
                // (emitting `(s).code, (s).env` would expand and re-allocate).
                if callee.0 == u32::MAX - 3 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_spawn_fiber(__cb.code, __cb.env); }}))", s);
                    }
                    return "NULL".into();
                }
                // Built-in `thread(closure)` — kernel thread tier.
                if callee.0 == u32::MAX - 15 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_spawn_thread(__cb.code, __cb.env); }}))", s);
                    }
                    return "NULL".into();
                }
                // Built-in `job(closure)` — work-pool tier.
                if callee.0 == u32::MAX - 16 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_spawn_job(__cb.code, __cb.env); }}))", s);
                    }
                    return "NULL".into();
                }
                // Built-in `spawn_pool(closure)` — fiber on background pool.
                if callee.0 == u32::MAX - 37 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_spawn_pool(__cb.code, __cb.env); }}))", s);
                    }
                    return "NULL".into();
                }
                // Built-in `join(slice_of_handles)` — wait for all handles.
                // Codegen extracts the slice's ptr+len and calls the runtime.
                if callee.0 == u32::MAX - 17 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        // The slice value (or borrow thereof) carries `.ptr` and
                        // `.len`.  For `&[]*Thread` we deref via (*s).; for `[]*Thread`
                        // direct access works.  The codegen for `Ref` wraps the
                        // value in `&v` which dereferences cleanly via (s).ptr too —
                        // so either form lands at the same access pattern.
                        return format!(
                            "(__maka_join_all_i64((maka_unit**)({0}).ptr, ({0}).len, NULL), MAKA_UNIT)",
                            s
                        );
                    }
                    return "MAKA_UNIT".into();
                }
                // Built-in `select(slice_of_handles)` — race; first ready wins,
                // losers are cancelled.
                if callee.0 == u32::MAX - 18 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!(
                            "(__maka_select_first_i64((maka_unit**)({0}).ptr, ({0}).len, NULL), MAKA_UNIT)",
                            s
                        );
                    }
                    return "MAKA_UNIT".into();
                }
                // Built-in `yield_now()` — cooperative yield.
                if callee.0 == u32::MAX - 20 {
                    return "(__maka_yield_now(), MAKA_UNIT)".into();
                }
                // par_map_bytes(in_ptr, n, in_sz, out_sz, body) — generic.
                if callee.0 == u32::MAX - 38 {
                    if args.len() == 5 {
                        let ip = self.emit_expr(f, &args[0]);
                        let n  = self.emit_expr(f, &args[1]);
                        let isz = self.emit_expr(f, &args[2]);
                        let osz = self.emit_expr(f, &args[3]);
                        let body = self.emit_expr(f, &args[4]);
                        return format!(
                            "(__extension__ ({{ Callable_unit_Pmunit_Pmunit_ __cb = ({4}); \
                             (maka_unit*)maka_par_map_bytes((void*)({0}), (int64_t)({1}), (int64_t)({2}), (int64_t)({3}), (void*)__cb.code, (void*)__cb.env); }}))",
                            ip, n, isz, osz, body
                        );
                    }
                    return "NULL".into();
                }
                // Built-in `detach(*Thread)`.
                if callee.0 == u32::MAX - 33 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(__maka_detach((maka_unit*)({})), MAKA_UNIT)", s);
                    }
                    return "MAKA_UNIT".into();
                }
                // Built-in `cancel(*Thread)`.
                if callee.0 == u32::MAX - 23 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(__maka_cancel((maka_unit*)({})), MAKA_UNIT)", s);
                    }
                    return "MAKA_UNIT".into();
                }
                // Built-in `try_join(*Thread) -> bool`.
                if callee.0 == u32::MAX - 24 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("((bool)(__maka_try_join((maka_unit*)({}), NULL) != 0))", s);
                    }
                    return "0".into();
                }
                // Built-in `join_timeout(*Thread, int) -> bool`.
                if callee.0 == u32::MAX - 25 {
                    if args.len() == 2 {
                        let h = self.emit_expr(f, &args[0]);
                        let ms = self.emit_expr(f, &args[1]);
                        return format!(
                            "((bool)(__maka_join_timeout((maka_unit*)({}), (int64_t)({}), NULL) != 0))",
                            h, ms
                        );
                    }
                    return "0".into();
                }
                // once_do(o, init) builtin: split Callable to (code, env).
                if callee.0 == u32::MAX - 32 {
                    if args.len() == 2 {
                        let o = self.emit_expr(f, &args[0]);
                        let cb = self.emit_expr(f, &args[1]);
                        return format!(
                            "(__extension__ ({{ Callable_unit_ __cb = ({1}); (maka_once_do((maka_unit*)({0}), (void*)__cb.code, (void*)__cb.env), MAKA_UNIT); }}))",
                            o, cb
                        );
                    }
                    return "MAKA_UNIT".into();
                }
                // par_for_each_float(slice, body)
                if callee.0 == u32::MAX - 34 {
                    if args.len() == 2 {
                        let s = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        return format!(
                            "(__extension__ ({{ Callable_unit_float_ __cb = ({1}); (__maka_par_for_each_f64(({0}), __cb.code, __cb.env), MAKA_UNIT); }}))",
                            s, b
                        );
                    }
                    return "MAKA_UNIT".into();
                }
                // par_map_float(slice, fn)
                if callee.0 == u32::MAX - 35 {
                    if args.len() == 2 {
                        let s = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        return format!(
                            "(__extension__ ({{ Callable_float_float_ __cb = ({1}); __maka_par_map_float(({0}), __cb.code, __cb.env); }}))",
                            s, b
                        );
                    }
                    return "(Slice_maka_float){0}".into();
                }
                // par_reduce_float(slice, init, combine)
                if callee.0 == u32::MAX - 36 {
                    if args.len() == 3 {
                        let s = self.emit_expr(f, &args[0]);
                        let init = self.emit_expr(f, &args[1]);
                        let b = self.emit_expr(f, &args[2]);
                        return format!(
                            "(__extension__ ({{ Callable_float_float_float_ __cb = ({2}); __maka_par_reduce_float(({0}), (double)({1}), __cb.code, __cb.env); }}))",
                            s, init, b
                        );
                    }
                    return "0.0".into();
                }
                // par_for_each(slice, body)
                if callee.0 == u32::MAX - 27 {
                    if args.len() == 2 {
                        let s = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        return format!(
                            "(__extension__ ({{ Callable_unit_ __cb = ({1}); (__maka_par_for_each_i64(({0}), __cb.code, __cb.env), MAKA_UNIT); }}))",
                            s, b
                        );
                    }
                    return "MAKA_UNIT".into();
                }
                // par_map_int(slice, fn)
                if callee.0 == u32::MAX - 28 {
                    if args.len() == 2 {
                        let s = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        return format!(
                            "(__extension__ ({{ Callable_int_int_ __cb = ({1}); __maka_par_map_int_slice(({0}), __cb.code, __cb.env); }}))",
                            s, b
                        );
                    }
                    return "(Slice_maka_int){0}".into();
                }
                // par_reduce_int(slice, init, combine)
                if callee.0 == u32::MAX - 29 {
                    if args.len() == 3 {
                        let s = self.emit_expr(f, &args[0]);
                        let init = self.emit_expr(f, &args[1]);
                        let b = self.emit_expr(f, &args[2]);
                        return format!(
                            "(__extension__ ({{ Callable_int_int_int_ __cb = ({2}); __maka_par_reduce_int_slice(({0}), (int64_t)({1}), __cb.code, __cb.env); }}))",
                            s, init, b
                        );
                    }
                    return "0".into();
                }
                // par_filter_int(slice, pred)
                if callee.0 == u32::MAX - 30 {
                    if args.len() == 2 {
                        let s = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        return format!(
                            "(__extension__ ({{ Callable_bool_int_ __cb = ({1}); __maka_par_filter_int(({0}), __cb.code, __cb.env); }}))",
                            s, b
                        );
                    }
                    return "(Slice_maka_int){0}".into();
                }
                // par_scan_int(slice, combine)
                if callee.0 == u32::MAX - 31 {
                    if args.len() == 2 {
                        let s = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        return format!(
                            "(__extension__ ({{ Callable_int_int_int_ __cb = ({1}); __maka_par_scan_int(({0}), __cb.code, __cb.env); }}))",
                            s, b
                        );
                    }
                    return "(Slice_maka_int){0}".into();
                }
                // Built-in `select_timeout(slice, int) -> int`.
                if callee.0 == u32::MAX - 26 {
                    if args.len() == 2 {
                        let s = self.emit_expr(f, &args[0]);
                        let ms = self.emit_expr(f, &args[1]);
                        return format!(
                            "({{ int64_t __out_i = -1; \
                                 (void)__maka_select_timeout_i64((maka_unit**)({0}).ptr, ({0}).len, (int64_t)({1}), &__out_i); \
                                 __out_i; }})",
                            s, ms
                        );
                    }
                    return "(-1)".into();
                }
                // Built-in `par_map_int(start, end, fn)` — produce a `[]int`.
                if callee.0 == u32::MAX - 22 {
                    if args.len() == 3 {
                        let a = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        let body = self.emit_expr(f, &args[2]);
                        return format!(
                            "(__extension__ ({{ Callable_int_int_ __cb = ({2}); __maka_par_map_int((int64_t)({0}), (int64_t)({1}), __cb.code, __cb.env); }}))",
                            a, b, body
                        );
                    }
                    return "((Slice_maka_int){ .ptr = NULL, .len = 0 })".into();
                }
                // Built-in `par_reduce_int(start, end, init, combine)`.
                if callee.0 == u32::MAX - 21 {
                    if args.len() == 4 {
                        let a = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        let init = self.emit_expr(f, &args[2]);
                        let body = self.emit_expr(f, &args[3]);
                        return format!(
                            "(__extension__ ({{ Callable_int_int_int_ __cb = ({3}); __maka_par_reduce_int((int64_t)({0}), (int64_t)({1}), (int64_t)({2}), __cb.code, __cb.env); }}))",
                            a, b, init, body
                        );
                    }
                    return "0".into();
                }
                // Built-in `par_for_range(start, end, closure)` — dispatch
                // chunks across the job pool.
                if callee.0 == u32::MAX - 19 {
                    if args.len() == 3 {
                        let a = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        let body = self.emit_expr(f, &args[2]);
                        return format!(
                            "(__extension__ ({{ Callable_unit_int_ __cb = ({2}); (__maka_par_for_range((int64_t)({0}), (int64_t)({1}), __cb.code, __cb.env), MAKA_UNIT); }}))",
                            a, b, body
                        );
                    }
                    return "MAKA_UNIT".into();
                }
                // Built-in `join(*Thread)` — pthread join + free handle.
                if callee.0 == u32::MAX - 4 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(__maka_join((maka_unit*)({})), MAKA_UNIT)", s);
                    }
                    return "MAKA_UNIT".into();
                }
                // Built-in string concat — four variants depending on whether each
                // operand is borrowed (`string`) or owned (`own *char` from a prior
                // concat / `read_line`).  Owned operands are freed inside the helper.
                if let Some(fname) = match callee.0 {
                    v if v == u32::MAX - 5 => Some("__maka_str_concat"),
                    v if v == u32::MAX - 8 => Some("__maka_str_concat_freel"),
                    v if v == u32::MAX - 9 => Some("__maka_str_concat_freer"),
                    v if v == u32::MAX - 10 => Some("__maka_str_concat_freeb"),
                    _ => None,
                } {
                    if args.len() == 2 {
                        let a = self.emit_expr(f, &args[0]);
                        let b = self.emit_expr(f, &args[1]);
                        return format!("{}({}, {})", fname, a, b);
                    }
                    return "((char*)0)".into();
                }
                // Built-in `format(...)` placeholder converters.
                if let Some(fname) = match callee.0 {
                    v if v == u32::MAX - 11 => Some("__maka_int_to_str"),
                    v if v == u32::MAX - 12 => Some("__maka_bool_to_str"),
                    v if v == u32::MAX - 13 => Some("__maka_float_to_str"),
                    v if v == u32::MAX - 14 => Some("__maka_char_to_str"),
                    _ => None,
                } {
                    if args.len() == 1 {
                        let a = self.emit_expr(f, &args[0]);
                        return format!("{}({})", fname, a);
                    }
                    return "((char*)0)".into();
                }
                // Built-in `read_line()` — owns a heap NUL-terminated buffer.
                if callee.0 == u32::MAX - 6 {
                    return "__maka_read_line()".into();
                }
                // Built-in `read_int()` — reads one int from stdin.
                if callee.0 == u32::MAX - 7 {
                    return "__maka_read_int()".into();
                }
                let sig = self.sym.func_sig(*callee);

                // Dynamic-dispatch: if the first arg is a dyn (or a reference to one), call through the vtable.
                if !args.is_empty() {
                    if let Some(_traits) = strip_to_dyn(&args[0].ty) {
                        // We want the underlying Dyn value, not its address. If the expression is
                        // `&mut x`/AddrOfRef of a dyn place, skip the `&`.
                        let recv_s = match &args[0].kind {
                            HExprKind::AddrOfRef { place, .. } => self.emit_place(f, place),
                            _ => self.emit_expr(f, &args[0]),
                        };
                        let mut rest: Vec<String> = Vec::new();
                        for a in &args[1..] { rest.push(self.emit_expr(f, a)); }
                        let rest_s = rest.join(", ");
                        let comma = if rest_s.is_empty() { "" } else { ", " };
                        let call = format!("({0}).vtbl->{1}(({0}).data{2}{3})", recv_s, c_ident(&sig.name), comma, rest_s);
                        return if matches!(sig.ret, HType::Unit) {
                            format!("({}, MAKA_UNIT)", call)
                        } else { call };
                    }
                }
                let name = if sig.is_extern {
                    sig.c_name.clone()
                } else if sig.name == "main" && sig.logic.is_none() {
                    "maka_main".to_string()
                } else { c_ident(&sig.c_name) };
                let arg_s: Vec<String> = args.iter().map(|a| self.emit_expr(f, a)).collect();
                let call = format!("{}({})", name, arg_s.join(", "));
                if matches!(sig.ret, HType::Unit) {
                    format!("({}, MAKA_UNIT)", call)
                } else {
                    call
                }
            }
            HExprKind::Cast { expr, kind, to } => {
                // For ToDyn casts we need the source's address (the data pointer).
                if let CastKind::ToDyn { trait_name, struct_id } = kind {
                    return self.emit_to_dyn(f, expr, trait_name, *struct_id);
                }
                let s = self.emit_expr(f, expr);
                self.emit_cast(s, kind.clone(), to)
            }
            HExprKind::CheckedCast { expr, kind, to } => {
                let s = self.emit_expr(f, expr);
                self.emit_checked_cast(s, kind.clone(), &expr.ty, to)
            }
            HExprKind::Struct { id, fields } => {
                let info = self.sym.struct_info(*id);
                let parts: Vec<String> = info.fields.iter().enumerate().map(|(i, f0)| {
                    if let Some((_, fe)) = fields.iter().find(|(j, _)| *j == i) {
                        let s = self.emit_expr(f, fe);
                        format!(".{} = {}", c_ident(&f0.name), s)
                    } else {
                        // pointer default null
                        format!(".{} = NULL", c_ident(&f0.name))
                    }
                }).collect();
                let _ = parts.is_empty();
                format!("(({}){{ {} }})", c_ident(&info.name), parts.join(", "))
            }
            HExprKind::ArrayLit(elems) => {
                let parts: Vec<String> = elems.iter().map(|e| self.emit_expr(f, e)).collect();
                // For an array-typed context: emit `{a, b, c}`
                // For slice context: we wrap into a Slice_T struct using a compound literal of the array.
                let eff_ty = e.ty.strip_heap();
                match eff_ty {
                    HType::Array { elem, .. } => {
                        let _ = elem;
                        format!("{{ {} }}", parts.join(", "))
                    }
                    HType::Slice { elem, .. } => {
                        let key = self.type_key(elem);
                        let elem_c = self.c_type(elem);
                        format!("((Slice_{0}){{ .ptr = (({1}[]){{ {2} }}), .len = {3} }})", key, elem_c, parts.join(", "), elems.len())
                    }
                    HType::Vec { elem } => {
                        // Initial vector content goes into a malloc'd buffer.
                        // For empty `[]` we just zero it out.
                        if elems.is_empty() {
                            let key = self.type_key(elem);
                            format!("((Vec_{0}){{ .data = NULL, .len = 0, .cap = 0 }})", key)
                        } else {
                            let key = self.type_key(elem);
                            let elem_c = self.c_type(elem);
                            let n = elems.len();
                            // Use a static initializer + memcpy via compound literal isn't trivial in C; we use malloc+copy.
                            // Emit a comma expression. We can't easily here. Fall back to using a local helper inline via stmt-expr (gcc/clang extension).
                            format!("(__extension__ ({{ {0}* __d = ({0}*)malloc(sizeof({0})*{1}); {0} __s[] = {{ {2} }}; memcpy(__d, __s, sizeof(__s)); (Vec_{3}){{ .data = __d, .len = {1}, .cap = {1} }}; }}))", elem_c, n, parts.join(", "), key)
                        }
                    }
                    _ => format!("{{ {} }}", parts.join(", ")),
                }
            }
            HExprKind::DropWrite(inner) => self.emit_expr(f, inner),
            HExprKind::DerefRef(inner) => {
                let s = self.emit_expr(f, inner);
                format!("(*({}))", s)
            }
            HExprKind::HeapAlloc(inner) => {
                let inner_c = self.c_type(&inner.ty);
                let v = self.emit_expr(f, inner);
                format!("(__extension__ ({{ {0}* __p = ({0}*)malloc(sizeof({0})); *__p = ({1}); __p; }}))", inner_c, v)
            }
            HExprKind::Match { scrutinee, arms, result_ty } => {
                self.emit_match(f, scrutinee, arms, result_ty)
            }
            HExprKind::VariantCtor { enum_id, variant, fields } => {
                let info = self.sym.enum_info(*enum_id);
                let v = &info.variants[*variant];
                if v.fields.is_empty() {
                    // No payload — same as bare variant.
                    return if info.is_simple() {
                        format!("{}__{}", c_ident(&info.name), c_ident(&v.name))
                    } else {
                        format!("(({0}){{ .tag = {1} }})", c_ident(&info.name), v.tag)
                    };
                }
                // Tagged with payload.
                let parts: Vec<String> = v.fields.iter().enumerate().map(|(i, fi)| {
                    let s = fields.iter().find(|(idx, _)| *idx == i)
                        .map(|(_, e)| self.emit_expr(f, e))
                        .unwrap_or_else(|| "0".into());
                    format!(".{} = {}", c_ident(&fi.name), s)
                }).collect();
                format!("(({0}){{ .tag = {1}, .payload.{2} = {{ {3} }} }})",
                    c_ident(&info.name), v.tag, c_ident(&v.name), parts.join(", "))
            }
            HExprKind::ArrayToSlice { base, len } => {
                let bs = self.emit_expr(f, base);
                let elem = match &e.ty {
                    HType::Slice { elem, .. } => self.type_key(elem),
                    _ => "maka_int".to_string(),
                };
                format!("((Slice_{0}){{ .ptr = (({1})), .len = {2} }})", elem, bs, len)
            }
        }
    }

    fn log_helper(&self, t: &HType) -> &'static str {
        match t {
            HType::Int | HType::Enum(_) | HType::SizedInt { .. } => "maka_log_int",
            HType::Float => "maka_log_float",
            HType::Bool => "maka_log_bool",
            HType::Char => "maka_log_char",
            HType::Str => "maka_log_str",
            HType::Ptr { .. } | HType::RawPtr { .. } | HType::OwnPtr { .. } | HType::Ref { .. } | HType::Heap { .. } => "maka_log_ptr",
            _ => "maka_log_ptr",
        }
    }

    fn emit_match(&mut self, f: &HFunc, scrutinee: &HExpr, arms: &[HMatchArm], result_ty: &HType) -> String {
        // Always emit as a GCC statement-expression that yields the result value.
        // Strategy:
        //   ({
        //     SCRUT_T __s = SCRUT;
        //     RES_T __r;
        //     do {
        //       <for each arm, in order>:
        //         if (<predicate>) { <bindings>; { body; __r = <value>; } break; }
        //     } while (0);
        //     __r;
        //   })
        let s = self.emit_expr(f, scrutinee);
        let scrut_c = self.c_type(&scrutinee.ty);
        let res_c = self.c_type(result_ty);
        let needs_value = !matches!(result_ty, HType::Unit);

        let mut body = String::new();
        body.push_str("__extension__ ({ ");
        body.push_str(&format!("{} __s = {}; ", scrut_c, s));
        if needs_value {
            body.push_str(&format!("{} __r = ({})0; ", res_c, res_c));
        }
        body.push_str("do { ");

        // Determine enum-id for variant matching if applicable.
        let _enum_eid: Option<EnumId> = match scrutinee.ty.clone() {
            HType::Enum(id) => Some(id),
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } | HType::Heap { inner } =>
                if let HType::Enum(id) = *inner { Some(id) } else { None },
            _ => None,
        };

        for a in arms {
            let pred = self.match_predicate(&a.kind);
            let mut bindings = self.match_bindings(f, &a.kind);
            // Scrut binding: bind __s to a local before guard/body.
            if let Some(local) = a.scrut_binding {
                let li = &f.locals[local.0 as usize];
                let n = local_name(local, &li.name);
                let ty = self.c_type(&li.ty);
                bindings.push_str(&format!("{} {} = __s; ", ty, n));
            }
            // If the arm has bindings (scrut or destructure), put guard *inside* the arm body
            // so the bindings are in scope.
            let guard = a.guard.as_ref().map(|g| self.emit_expr(f, g));
            let needs_body_guard = a.scrut_binding.is_some() || matches!(&a.kind, HArmKind::Variant { .. });
            let (pred_combined, body_guard) = if needs_body_guard && guard.is_some() {
                (pred, guard)
            } else {
                let combined = match (pred, guard) {
                    (p, Some(g)) if p == "1" => format!("({})", g),
                    (p, Some(g)) => format!("(({}) && ({}))", p, g),
                    (p, None) => p,
                };
                (combined, None)
            };
            body.push_str(&format!("if ({}) {{ {}", pred_combined, bindings));
            if let Some(g) = body_guard {
                body.push_str(&format!("if ({}) {{ ", g));
            }
            for stmt in &a.body.stmts {
                let prev_len = self.out.len();
                self.emit_stmt(f, stmt);
                let emitted = self.out[prev_len..].to_string();
                self.out.truncate(prev_len);
                body.push_str(&emitted);
            }
            if let Some(v) = &a.value {
                let vs = self.emit_expr(f, v);
                if needs_value { body.push_str(&format!("__r = ({}); ", vs)); }
                else { body.push_str(&format!("(void)({}); ", vs)); }
            }
            // Close any open guard `if`, with `break;` so a matched arm exits.
            // If we put guard into the arm body, we must NOT break unconditionally outside it.
            let has_body_guard = a.scrut_binding.is_some() && a.guard.is_some();
            if has_body_guard {
                // close inner if with break; close outer arm-if without break (fall-through to next arm).
                body.push_str(" break; } } ");
            } else {
                body.push_str(" break; } ");
            }
        }
        body.push_str("} while (0); ");
        if needs_value { body.push_str("__r; "); } else { body.push_str("MAKA_UNIT; "); }
        body.push_str("})");
        body
    }

    fn match_predicate(&self, k: &HArmKind) -> String {
        match k {
            HArmKind::Else => "1".to_string(),
            HArmKind::Null => "(__s == NULL)".to_string(),
            HArmKind::Lit(e) => {
                let s = self.literal_str(e);
                format!("(__s == {})", s)
            }
            HArmKind::Variant { variant, lit_checks, enum_id, .. } => {
                let info = self.sym.enum_info(*enum_id);
                let v = &info.variants[*variant];
                let tag_expr = if info.is_simple() {
                    "__s".to_string()
                } else {
                    "__s.tag".to_string()
                };
                let mut parts = vec![format!("({} == {})", tag_expr, v.tag)];
                for (i, c) in lit_checks.iter().enumerate() {
                    if let Some(ce) = c {
                        let s = self.literal_str(ce);
                        let fname = &v.fields[i].name;
                        parts.push(format!("(__s.payload.{}.{} == {})", c_ident(&v.name), c_ident(fname), s));
                    }
                }
                parts.join(" && ")
            }
        }
    }

    fn literal_str(&self, e: &HExpr) -> String {
        match &e.kind {
            HExprKind::LitInt(n) => format!("(maka_int){}LL", n),
            HExprKind::LitBool(b) => if *b { "true".into() } else { "false".into() },
            HExprKind::LitChar(c) => format!("(maka_char){}u", *c as u32),
            HExprKind::LitFloat(v) => format!("(maka_float){}", v),
            HExprKind::LitNull => "NULL".into(),
            _ => "0".into(),
        }
    }

    /// Emit a global's initializer as a C constant expression.  Walks the same
    /// shapes a static initializer accepts: literals, unary minus on literals,
    /// and arithmetic over the same.  For anything more, the C compiler will
    /// produce a "not a constant expression" error - which is the right move.
    fn emit_global_init(&self, e: &HExpr) -> String {
        match &e.kind {
            HExprKind::LitInt(n) => format!("(maka_int){}LL", n),
            HExprKind::LitBool(b) => if *b { "true".into() } else { "false".into() },
            HExprKind::LitChar(c) => format!("(maka_char){}u", *c as u32),
            HExprKind::LitFloat(v) => format!("(maka_float){}", v),
            HExprKind::LitStr(s) => format!("{:?}", s),
            HExprKind::LitNull => "NULL".into(),
            HExprKind::LitUnit => "MAKA_UNIT".into(),
            HExprKind::Un { op: HUnOp::Neg, expr } => format!("(-({}))", self.emit_global_init(expr)),
            HExprKind::Bin { op, lhs, rhs } => {
                let l = self.emit_global_init(lhs);
                let r = self.emit_global_init(rhs);
                format!("(({}) {} ({}))", l, binop_c(*op), r)
            }
            _ => "0".into(),
        }
    }

    fn match_bindings(&self, f: &HFunc, k: &HArmKind) -> String {
        match k {
            HArmKind::Variant { enum_id, variant, bindings, .. } => {
                let info = self.sym.enum_info(*enum_id);
                let v = &info.variants[*variant];
                let mut out = String::new();
                for (i, b) in bindings.iter().enumerate() {
                    if let Some(local) = b {
                        let li = &f.locals[local.0 as usize];
                        let name = local_name(*local, &li.name);
                        let fname = &v.fields[i].name;
                        let ty = self.c_type(&li.ty);
                        out.push_str(&format!("{} {} = __s.payload.{}.{}; ", ty, name, c_ident(&v.name), c_ident(fname)));
                    }
                }
                out
            }
            _ => String::new(),
        }
    }

    fn emit_to_dyn(&mut self, f: &HFunc, expr: &HExpr, trait_name: &str, struct_id: StructId) -> String {
        // Source might be `&T`, `&mut T`, or `T`. For all three, we take the address of the value.
        let sname = self.sym.struct_info(struct_id).name.clone();
        let inner_c = c_ident(&sname);
        // Need an address. If expr is already a reference (Ref), emit_expr returns `&local` form.
        let s = self.emit_expr(f, expr);
        // If the expression's type is a Ref, its C value is already `T*`. Else, take address.
        let data_expr = match &expr.ty {
            HType::Ref { .. } => format!("(void*)({})", s),
            _ => format!("(void*)(&({}))", s),
        };
        let _ = inner_c;
        format!("((Dyn_{0}){{ .data = {1}, .vtbl = &{0}_vtbl_for_{2} }})",
                c_ident(trait_name), data_expr, c_ident(&sname))
    }

    fn emit_cast(&self, s: String, kind: CastKind, to: &HType) -> String {
        let to_c = self.c_type(to);
        match kind {
            CastKind::Numeric | CastKind::SignChange | CastKind::EnumToInt | CastKind::CharIntInt | CastKind::Identity => {
                format!("(({}){})", to_c, s)
            }
            // Reinterpret: a plain C cast — works for ptr↔ptr, ptr↔intptr_t, etc.
            // The `(uintptr_t)` round-trip silences GCC's "incompatible pointer types"
            // warnings on direct *T↔*U casts.
            CastKind::Reinterpret => format!("(({})(uintptr_t)({}))", to_c, s),
            _ => format!("(({}){})", to_c, s),
        }
    }

    fn emit_checked_cast(&self, s: String, kind: CastKind, _from: &HType, to: &HType) -> String {
        match kind {
            CastKind::IntToEnumChecked => {
                if let HType::Enum(eid) = to {
                    let info = self.sym.enum_info(*eid);
                    // Build a runtime check expression: result = (s in {variants...}) ? new T(s) : NULL.
                    // Since result type is *T (Enum) — represented as `T*` — we malloc a temp and return its addr.
                    let mut cond_parts = Vec::new();
                    for v in &info.variants {
                        cond_parts.push(format!("__v == {}", v.tag));
                    }
                    let cond = if cond_parts.is_empty() { "0".to_string() } else { cond_parts.join(" || ") };
                    let ec = c_ident(&info.name);
                    return format!("(__extension__ ({{ maka_int __v = ({0}); {1}* __r = NULL; if ({2}) {{ __r = ({1}*)malloc(sizeof({1})); *__r = ({1})__v; }} __r; }}))", s, ec, cond);
                }
                format!("({})", s)
            }
            CastKind::IntToCharChecked => {
                // Accept any non-negative int < 0x110000
                format!("(__extension__ ({{ maka_int __v = ({0}); maka_char* __r = NULL; if (__v >= 0 && __v < 0x110000) {{ __r = (maka_char*)malloc(sizeof(maka_char)); *__r = (maka_char)__v; }} __r; }}))", s)
            }
            _ => format!("({})", s),
        }
    }
}

fn binop_c(op: HBinOp) -> &'static str {
    use HBinOp::*;
    match op {
        Add => "+", Sub => "-", Mul => "*", Div => "/", Mod => "%",
        Eq => "==", Ne => "!=", Lt => "<", Le => "<=", Gt => ">", Ge => ">=",
        And => "&&", Or => "||",
        BitAnd => "&", BitOr => "|", BitXor => "^", Shl => "<<", Shr => ">>",
    }
}
fn assign_op_c(op: HAssignOp) -> &'static str {
    match op {
        HAssignOp::Assign => "=", HAssignOp::Add => "+=", HAssignOp::Sub => "-=",
        HAssignOp::Mul => "*=", HAssignOp::Div => "/=", HAssignOp::Mod => "%=",
    }
}

fn c_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn c_ident(name: &str) -> String {
    // mangle to ascii-only identifier
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' { out.push(c); }
        else { out.push_str(&format!("_{:02x}", c as u32)); }
    }
    out
}

fn inline_local_name(_inline_f: &HFunc, id: LocalId, tag: &str) -> String {
    format!("_il_{}_{}", id.0, tag)
}

/// Generates a deterministic key for a function-pointer signature, used in C typedef names
/// like `Callable_KEY` and trampoline names like `fn_NAME_trampoline`.
fn fn_sig_key(ret: &HType, params: &[HType]) -> String {
    let mut s = ret.key();
    s.push('_');
    for p in params { s.push_str(&p.key()); s.push('_'); }
    // Stay alphanumeric-friendly.
    s.replace('\'', "x").replace('+', "p")
}

fn local_name(id: LocalId, name: &str) -> String {
    format!("{}_{}", c_ident(name), id.0)
}
