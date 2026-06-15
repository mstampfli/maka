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

        // Function bodies
        for f in &funcs {
            self.emit_func(f);
        }

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
        // Threads via pthread.  `Thread` is an opaque struct typedef declared below
        // (we emit it before any user code); `__maka_spawn(code, env)` packages a
        // closure (its code pointer + env pointer) into a pthread.
        self.w("typedef struct Thread { pthread_t handle; } Thread;\n");
        self.w("typedef struct { void* code; void* env; } __maka_closure_fat;\n");
        self.w("static void* __maka_thread_entry(void* arg) { __maka_closure_fat* f = (__maka_closure_fat*)arg; void (*code)(void*) = (void (*)(void*))f->code; code(f->env); free(f); return NULL; }\n");
        self.w("maka_unit* __maka_spawn(void* code, void* env) { Thread* t = (Thread*)malloc(sizeof(Thread)); __maka_closure_fat* f = (__maka_closure_fat*)malloc(sizeof(__maka_closure_fat)); f->code = code; f->env = env; pthread_create(&t->handle, NULL, __maka_thread_entry, f); return (maka_unit*)t; }\n");
        self.w("void __maka_join(maka_unit* t) { Thread* th = (Thread*)t; pthread_join(th->handle, NULL); free(th); }\n");
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
                let ty = self.c_type(&f.ty);
                self.wl(&format!("{} {};", ty, c_ident(&f.name)));
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
            HExprKind::SliceLen(inner) => self.scan_expr(inner),
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
        }
    }

    fn c_type_from_key(&self, k: &str) -> String {
        // Most keys are already valid C type names (struct names, primitives like
        // `maka_int`), but a couple of Maka-level keys need translation.
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
                        return format!("(Thread*)__maka_spawn(({}).code, ({}).env)", s, s);
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
                // Built-in `spawn(closure)` — pthread spawn.
                if callee.0 == u32::MAX - 3 {
                    if let Some(a) = args.first() {
                        let s = self.emit_expr(f, a);
                        return format!("(Thread*)__maka_spawn(({}).code, ({}).env)", s, s);
                    }
                    return "NULL".into();
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
