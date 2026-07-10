//! Maka → C codegen. Produces a portable C99/C11 translation of an HIR module.
//!
//! Heap T values are represented as `T*` allocated with `malloc` (single owner);
//! freed at scope exit unless moved. Pointer unwrap (`p!`) emits a runtime null check
//! when not narrowed.

use maka_sema::*;

pub fn emit(m: &HirModule) -> String {
    emit_with_debug(m, &std::collections::HashMap::new())
}

/// Like `emit`, but emits `#line` directives mapping generated C back to Maka
/// source, using `debug_files` (module path -> `.maka` file).  The driver passes
/// this for a normal build so gdb/lldb/VS Code debug Maka source directly.
pub fn emit_with_debug(
    m: &HirModule,
    debug_files: &std::collections::HashMap<Vec<String>, String>,
) -> String {
    let mut cx = Cx::new(&m.sym, &m.cincludes, &m.cblocks);
    cx.debug_files = debug_files.clone();
    cx.emit_module();
    cx.out
}

/// Freestanding emit — no libc / no stdio / no malloc.  The prologue
/// drops every `#include` and every runtime helper that calls libc; the
/// allocator (`alloc T { ... }` and `free p;`) is rewritten to call
/// user-provided extern symbols `__maka_alloc(usize)` and `__maka_free(*mut)`.
/// `log` and `panic` likewise lower to `__maka_log` / `__maka_panic` externs.
/// The caller (kernel author / boot image) is expected to provide those four
/// symbols in their own translation unit before linking.
pub fn emit_freestanding(m: &HirModule) -> String {
    let mut cx = Cx::new(&m.sym, &m.cincludes, &m.cblocks);
    cx.freestanding = true;
    cx.emit_module();
    cx.out
}

/// Proven upper bound of a loop counter, for bounds-check elision.
#[derive(Clone, Copy)]
enum BceBound {
    /// Counter stays in `[0, N)` for a compile-time constant N.
    Const(i64),
    /// Counter stays in `[0, len)` of the slice/vec held by this local id.
    SliceLen(u32),
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
    /// `Rust<T>` opaque type labels seen — emit an `extern __maka_drop_<T>` decl.
    rust_drop_types: std::collections::BTreeSet<String>,
    /// Dyn instantiations seen: (Trait, StructId) — generate vtables for these.
    dyn_insts: std::collections::BTreeSet<(String, u32)>,
    /// Traits seen — generate Dyn_Trait struct typedef for these.
    dyn_traits: std::collections::BTreeSet<String>,
    /// Runtime existential re-witness targets `(src_trait, to_traits)` seen
    /// (`v has X` on a `dyn`/`some` value).  After the scan populates `dyn_insts`,
    /// each is expanded: every concrete type packed as `dyn src_trait` that also
    /// implements all of `to_traits` gets a vtable emitted for that (combined) trait
    /// set, so the runtime switch can re-witness it.
    dyn_witness_targets: std::collections::BTreeSet<(String, Vec<String>)>,
    /// Combined (multi-trait) vtables needed for narrowing re-witness: a `dyn (Y+X)`
    /// keeps both interfaces.  `combined_vtbls` drives the `Dyn_<key>`/`<key>_vtbl`
    /// typedefs; `combined_insts` drives the per-type `<key>_vtbl_for_T` instances.
    /// `key` is `traits.join("_")` and the vtbl unions all constituent methods.
    combined_vtbls: std::collections::BTreeSet<Vec<String>>,
    combined_insts: std::collections::BTreeSet<(Vec<String>, u32)>,
    /// Fat-callable signatures seen — emit `Callable_KEY` typedef + raw-fn-ptr typedef.
    callable_sigs: std::collections::BTreeMap<String, (HType, Vec<HType>)>,
    /// Aggregate type keys already walked by `note_type`, so recursion into
    /// struct/enum field types terminates on self-referential types.
    noted_aggregates: std::collections::HashSet<String>,
    /// Functions whose names appear as values — emit trampolines.
    fn_trampolines: std::collections::BTreeSet<u32>,
    /// Lifted-lambda function ids that need closure trampolines (env cast + call).
    closure_trampolines: std::collections::BTreeSet<u32>,
    /// Names of struct/enum types that (transitively) own heap resources and so
    /// get a generated `__maka_drop_<Name>` recursive-free function.
    drop_owns: std::collections::HashSet<String>,
    /// Locals stored in C as a `T*` alias into existing storage (not a `T` value
    /// copy) - a read-only for-each element bound to `&elem[i]`, or a by-ref
    /// struct parameter.  Every use of such a local is emitted as `(*name)`.
    aliased_locals: std::collections::HashSet<u32>,
    /// The chosen C name of each local of the function currently being emitted,
    /// keyed by `LocalId.0`.  Built once per function (`build_local_names`): a
    /// local keeps its readable Maka name when that is unambiguous, and only
    /// falls back to the unique `name_<id>` spelling on a genuine clash (a
    /// shadowed name, a function/type of the same name, a runtime symbol).  This
    /// is what makes the emitted C - and therefore gdb/lldb/DWARF - readable.
    cur_local_names: std::collections::HashMap<u32, String>,
    /// In-scope loop induction variables proven to stay within `[0, bound)` by a
    /// for-range loop guard - `(counter local id, bound)`.  Lets indexing skip
    /// the bounds check: a constant bound covers fixed arrays of length >= it; a
    /// `SliceLen(s)` bound covers indexing that exact slice/vec local `s`.
    bounded_vars: Vec<(u32, BceBound)>,
    /// Per-emission counter for inline expansions: each statement-expression gets
    /// a unique tag so labels and locals never collide across multiple call sites
    /// of the same inline within the same C function.
    inline_call_seq: u32,
    /// Stack of result types for the inline expansions currently being spliced,
    /// so a `match` nested in an inline body can lower a `return` in an arm to the
    /// right `__r_<tag>` assignment.  Innermost expansion is last.
    inline_ret_stack: Vec<HType>,
    /// Per-active-inline-expansion stack of the CALLER's owning locals (resolved C
    /// name + type) live at the call site.  A `propagate` inside the expansion
    /// early-returns the caller, so it frees these first or they leak.
    inline_propagate_drops: Vec<Vec<(String, HType)>>,
    /// Like `inline_propagate_drops` but for a `break`/`continue` spliced from the
    /// inline that targets the caller's loop: the caller's loop-body owning locals
    /// to free before the jump (the jump skips the loop body's scope-exit drops).
    inline_loop_drops: Vec<Vec<(String, HType)>>,
    /// Return type of the C function currently being emitted.  A `propagate`
    /// spliced from an inline body lowers to a bare `return` of THIS function, so
    /// its value must match this type even when the inline helper was monomorphized
    /// with different type args (e.g. try_ok's Result<T,E>.Err into the caller's
    /// Result<U,E>).
    cur_c_func_ret: HType,
    /// `*Thread` handle locals in the current function that are NEVER read after
    /// declaration (genuinely abandoned - not joined/detached/stored/returned).
    /// Such a handle leaks its runtime Thread struct, so it is auto-detached at
    /// its declaring block's exit.  Read handles are excluded, so join/store
    /// paths are never double-reaped.
    cur_abandoned_handles: std::collections::HashSet<u32>,
    /// Stack of (synthetic `__yield` local, C end-label) for the value-producing
    /// match arms currently being emitted.  A `yield` lowers to an assignment of
    /// that local; assigning it must ALSO jump to the arm's end label so a `yield`
    /// nested in an `if`/`while` terminates the arm instead of falling through to a
    /// later `yield` that overwrites it.  Innermost arm is last.
    yield_exit: Vec<(LocalId, String)>,
    /// Monotonic counter for unique `yield`-arm end labels.
    yield_label_seq: u32,
    /// Monotonic counter for the temp binding a `free deep p` uses inside its
    /// GNU statement-expression, so two deep-frees never share a temp name.
    free_deep_seq: u32,
    /// When Some(tag), sub-expression emission (`emit_sub`) resolves local names
    /// via the inline expansion's tagged scope instead of the normal one.  Set only
    /// while emitting a builtin-sentinel call spliced from an inline body, so the one
    /// `emit_builtin_sentinel` serves both the normal and the inline paths.
    inline_tag: Option<String>,
    /// Freestanding mode — emit a minimal libc-free prologue and route the
    /// allocator / panic / log / atomic-runtime hooks to user-provided
    /// extern symbols (`__maka_alloc`, `__maka_free`, `__maka_panic`,
    /// `__maka_log`).  Set by `emit_freestanding()`.
    pub freestanding: bool,
    /// Debug source maps: module path -> the `.maka` file it was parsed from.
    /// When non-empty, codegen emits `#line N "file.maka"` directives so the
    /// binary's DWARF references Maka source and gdb/lldb/VS Code debug Maka
    /// lines directly instead of the generated C.  Empty = no directives.
    debug_files: std::collections::HashMap<Vec<String>, String>,
    /// The `.maka` file of the function currently being emitted (its module's
    /// entry in `debug_files`), used to stamp each statement's `#line`.
    cur_debug_file: Option<String>,
}

impl<'a> Cx<'a> {
    fn new(sym: &'a SymTab, cincludes: &'a [String], cblocks: &'a [String]) -> Self {
        Self {
            sym, out: String::new(), indent: 0,
            module_cincludes: cincludes,
            module_cblocks: cblocks,
            slice_types: Default::default(), vec_types: Default::default(),
            rust_drop_types: Default::default(),
            dyn_insts: Default::default(), dyn_traits: Default::default(),
            dyn_witness_targets: Default::default(),
            combined_vtbls: Default::default(), combined_insts: Default::default(),
            callable_sigs: Default::default(),
            fn_trampolines: Default::default(),
            closure_trampolines: Default::default(),
            drop_owns: Default::default(),
            aliased_locals: Default::default(),
            cur_local_names: Default::default(),
            bounded_vars: Vec::new(),
            inline_call_seq: 0,
            inline_ret_stack: Vec::new(),
            inline_propagate_drops: Vec::new(),
            inline_loop_drops: Vec::new(),
            cur_c_func_ret: HType::Unit,
            cur_abandoned_handles: Default::default(),
            yield_exit: Vec::new(),
            yield_label_seq: 0,
            free_deep_seq: 0,
            inline_tag: None,
            noted_aggregates: Default::default(),
            freestanding: false,
            debug_files: Default::default(),
            cur_debug_file: None,
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

        // First pass: scan for slice/vec/dyn types used in signatures and function
        // bodies so we can typedef them.  Skip generic templates (non-empty
        // type_params) - only their concrete instantiations are emitted; a template
        // carries TyVar types (e.g. `Vec<V>`) that would produce bogus typedefs.
        let funcs: Vec<HFunc> = self.sym.funcs.clone().into_iter()
            .filter(|f| self.sym.func_sig(f.id).type_params.is_empty())
            .collect();
        for f in &funcs {
            self.scan_func(f);
        }
        // Expand runtime re-witness targets now that `dyn_insts` is complete: for
        // each `v has X` (src_trait -> to_trait), every concrete type packed as
        // `dyn src_trait` that ALSO implements to_trait needs a `to_trait` vtable
        // (and the `Dyn_to_trait` typedef) so the runtime switch can re-witness it.
        // Fixpoint: a re-witness source can be a COMBINED existential produced by a
        // prior narrowing (nested `if (d has A) { if (d has B) {..} }`), so keep
        // expanding until no new (single or combined) vtable is registered.  Sources
        // come from both `dyn_insts` (single-trait) and `combined_insts` (joined).
        let witness_targets: Vec<(String, Vec<String>)> = self.dyn_witness_targets.iter().cloned().collect();
        loop {
            let avail = self.dyn_source_insts();
            let mut changed = false;
            for (src, to_traits) in &witness_targets {
                let single = to_traits.len() == 1;
                // Register the target TYPEDEF unconditionally: the re-witness produces a
                // `Dyn_<key>` value even when NO concrete type matches (all-null switch ->
                // vtbl NULL -> membership false), so the typedef must exist regardless.
                if single { changed |= self.dyn_traits.insert(to_traits[0].clone()); }
                else { changed |= self.combined_vtbls.insert(to_traits.clone()); }
                for (key, sid_n) in &avail {
                    if key != src { continue; }
                    if !to_traits.iter().all(|tr| self.struct_impls_trait_cg(StructId(*sid_n), tr)) { continue; }
                    if single { changed |= self.dyn_insts.insert((to_traits[0].clone(), *sid_n)); }
                    else { changed |= self.combined_insts.insert((to_traits.clone(), *sid_n)); }
                }
            }
            if !changed { break; }
        }
        // Structs/enums: forward-decls, then (internally) dyn+callable typedefs
        // that reference struct POINTERS, then the struct BODIES.  A `data` struct
        // can embed a trait-object VALUE by field (`data Holder { &dyn Trait f; }`),
        // so the Dyn_Trait fat-pointer typedef must be defined before the struct
        // body - but AFTER the struct forward-decls (its vtable takes struct ptrs).
        // The scan above populated the dyn/callable sets used there.
        self.emit_structs_and_enums();
        // Callable typedefs for types used only in function signatures (the
        // dyn typedefs were already emitted inside emit_structs_and_enums, after
        // the forward-decls; emitting them again would redefine them).
        self.emit_callable_typedefs();
        // Slice/vec typedefs come AFTER structs: a `Vec<Cat>` buffer embeds the
        // struct by value, so the element struct must already be defined.
        self.emit_slice_typedefs();
        self.emit_vec_typedefs();

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
                    let pn = s.param_names.get(i).cloned().unwrap_or_else(|| format!("p{}", i));
                    self.c_extern_param_decl(t, &pn)
                }).collect();
                params.push_str(&parts.join(", "));
                if s.is_variadic {
                    if !s.param_tys.is_empty() { params.push_str(", "); }
                    params.push_str("...");
                }
            }
            self.wl(&format!("extern {} {}({});", ret, s.c_name, params));
        }

        // Drop shims for `Rust<T>` opaque handles (generated in the Rust sidecar).
        for label in &self.rust_drop_types.clone() {
            self.wl(&format!("extern void __maka_drop_{}(void*);", c_ident(label)));
        }

        // Forward decls (non-extern).  Build the local-name table first so a
        // prototype's parameter names match the definition's (the table is a
        // deterministic function of `f`, so both passes agree).
        for f in &funcs {
            if self.sym.func_sig(f.id).is_inline { continue; }
            self.cur_local_names = self.build_local_names(f);
            let sig = self.func_signature(f);
            self.wl(&format!("{};", sig));
        }
        self.wl("");

        // Vtable instances (need function forward decls to be visible).
        self.emit_dyn_vtable_instances();
        self.emit_combined_vtable_instances();
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

        // Recursive drop glue for owning types, before any function body that
        // might call it.
        self.compute_drop_owns();
        self.emit_drop_glue();

        // Function bodies
        for f in &funcs {
            self.emit_func(f);
        }
        // TCP socket helper bodies — emitted last so that the sys/socket.h
        // include is positioned AFTER all user code.  Otherwise socket.h's
        // pollution (e.g. the bare `accept` symbol) clobbers user functions
        // sharing those names.
        if !self.freestanding {
            self.emit_socket_helpers();
        }

        // Synthesize a C `int main` that calls Maka `main`.  Two surface shapes:
        //
        //   unit main()                  → ignore argc/argv
        //   unit main([]string args)     → build a `Slice_str` from (argc, argv)
        //
        // The slice form receives a borrowed view of argv; Maka code may not free
        // it.  argv[0] is the program name, matching every other language.
        //
        // Freestanding mode skips this shim entirely: there's no libc-style
        // `int main(argc, argv)` entry point on a kernel target.  The OS
        // author's boot code calls `maka_main()` directly from their `_start`.
        if self.freestanding {
            return;
        }
        let user_main = funcs.iter().find(|f| f.name == "main");
        match user_main {
            Some(f) => {
                let sig = self.sym.func_sig(f.id);
                let takes_args = sig.param_tys.len() == 1
                    && matches!(&sig.param_tys[0], HType::Slice { elem, .. } if matches!(elem.as_ref(), HType::Str));
                if takes_args {
                    let call_returning_int = matches!(f.ret, HType::Int);
                    if call_returning_int {
                        self.wl("int main(int argc, char** argv) { __maka_rt_set_args(argc, argv); Slice_str args = { .ptr = (const char**)argv, .len = (maka_int)argc }; return (int)maka_main(args); }");
                    } else {
                        self.wl("int main(int argc, char** argv) { __maka_rt_set_args(argc, argv); Slice_str args = { .ptr = (const char**)argv, .len = (maka_int)argc }; maka_main(args); return 0; }");
                    }
                } else {
                    match f.ret {
                        HType::Int => self.wl("int main(int argc, char** argv) { __maka_rt_set_args(argc, argv); return (int)maka_main(); }"),
                        _ => self.wl("int main(int argc, char** argv) { __maka_rt_set_args(argc, argv); maka_main(); return 0; }"),
                    }
                }
            }
            None => self.wl("int main(void) { return 0; }"),
        }
    }

    /// Minimal freestanding prologue — no libc, no stdio, no stdlib.  Pulls
    /// in only the headers C requires every freestanding implementation to
    /// provide (`<stdint.h>`, `<stdbool.h>`, `<stddef.h>`, `<stdarg.h>` and
    /// the C11 `<stdatomic.h>` for the `__atomic_*` intrinsics).  Allocator,
    /// log, and panic hooks become extern symbols the OS author defines.
    /// User cblocks are pasted verbatim as in the regular path.
    fn emit_freestanding_prologue(&mut self) {
        self.w("// freestanding mode - no libc, no stdio, no malloc.\n");
        self.w("// User must provide: __maka_alloc, __maka_free, __maka_panic, __maka_log_int, __maka_log_str.\n");
        self.w("#include <stdint.h>\n");
        self.w("#include <stdbool.h>\n");
        self.w("#include <stddef.h>\n");
        self.w("#include <stdarg.h>\n");
        // User-requested system headers from `cinclude "name.h";` directives.
        // In a kernel build the user will typically use cinclude only for
        // their own headers; libc system headers will fail to compile under
        // `-ffreestanding -nostdinc`.
        let extras: Vec<String> = self.module_cincludes.to_vec();
        for h in &extras { self.w(&format!("#include <{}>\n", h)); }
        self.w("typedef int64_t maka_int;\ntypedef double maka_float;\ntypedef uint8_t maka_char;\n");
        self.w("typedef struct { int dummy; } maka_unit;\n");
        self.w("static maka_unit MAKA_UNIT = {0};\n");
        // Runtime hooks the OS author supplies — declared, never defined here.
        self.w("extern void* __maka_alloc(size_t sz);\n");
        self.w("extern void  __maka_free(void* p);\n");
        self.w("extern void  __maka_panic(const char* msg);\n");
        self.w("extern void  __maka_log_int(maka_int v);\n");
        self.w("extern void  __maka_log_str(const char* s);\n");
        // Macro-redirect all in-prologue malloc/free/panic/log call sites in
        // existing codegen paths so we don't have to change every emit point.
        // `printf`/`puts`/`fprintf` are NOT redirected — any code path that
        // would emit one of those (string concat helpers, read_line, etc.)
        // is not reachable in freestanding mode because the stdlib that uses
        // them isn't auto-included and the helper emit blocks are guarded.
        self.w("#define malloc(sz)            __maka_alloc((size_t)(sz))\n");
        self.w("#define free(p)               __maka_free((void*)(p))\n");
        self.w("#define maka_panic(s)         __maka_panic(s)\n");
        self.w("#define maka_log_int(v)       __maka_log_int(v)\n");
        self.w("#define maka_log_str(s)       __maka_log_str(s)\n");
        // Minimal helpers the rest of codegen assumes are present.
        // `maka_check_idx` is emitted by array/slice access; route to panic.
        self.w("static inline maka_int maka_check_idx(maka_int i, maka_int len, const char* msg){ if(i<0||i>=len) maka_panic(msg); return i; }\n");
        // `maka_check_div` guards an integer division/modulo divisor: a zero
        // divisor is C undefined behaviour (a hardware trap), so panic cleanly
        // instead.  Type-preserving so the division keeps its operand width.
        self.w("#define maka_check_div(d) (__extension__ ({ __typeof__(d) __mdz = (d); if (__mdz == 0) maka_panic(\"divide by zero\"); __mdz; }))\n");
        // Signed division/modulo: panic on /0 AND on the MIN/-1 overflow (C UB),
        // the same way zero and shift overflow panic.  `mn` is the type minimum.
        self.w("#define maka_divs(a, b, mn) (__extension__ ({ __typeof__(a) __dva = (a); __typeof__(b) __dvb = (b); if (__dvb == 0) maka_panic(\"divide by zero\"); if (__dva == (mn) && __dvb == -1) maka_panic(\"division overflow\"); __dva / __dvb; }))\n");
        self.w("#define maka_mods(a, b, mn) (__extension__ ({ __typeof__(a) __dma = (a); __typeof__(b) __dmb = (b); if (__dmb == 0) maka_panic(\"divide by zero\"); if (__dma == (mn) && __dmb == -1) maka_panic(\"division overflow\"); __dma % __dmb; }))\n");
        // Saturating float->int conversion (C makes out-of-range / NaN UB): NaN->0,
        // below-range->MIN, above-range->MAX.  `hi` is 2^(bits-1) (signed) or 2^bits
        // (unsigned) as a double; saturated bounds are returned as ints, never routed
        // through double (which would re-round INT64_MAX back out of range).
        self.w("#define maka_f2i(f, T, hi, mn, mx) (__extension__ ({ double __ff = (double)(f); T __fr; if (__ff != __ff) __fr = 0; else if (__ff >= (hi)) __fr = (mx); else if (__ff < -(hi)) __fr = (mn); else __fr = (T)__ff; __fr; }))\n");
        self.w("#define maka_f2u(f, T, hi, mx) (__extension__ ({ double __ff = (double)(f); T __fr; if (__ff != __ff) __fr = 0; else if (__ff <= 0.0) __fr = 0; else if (__ff >= (hi)) __fr = (mx); else __fr = (T)__ff; __fr; }))\n");
        self.w("#define maka_check_shift(n, w) (__extension__ ({ __typeof__(n) __msh = (n); if ((uint64_t)(__msh) >= (uint64_t)(w)) maka_panic(\"shift amount out of range\"); __msh; }))\n");
        // Atomic intrinsics — `__atomic_*` are compiler builtins (gcc/clang
        // emit them inline, no libc).  The runtime helpers that wrap them
        // in `maka_atomic_*` symbols are kept here for any extern decl that
        // referenced them.
        self.w("static maka_int maka_atomic_load_i64(maka_int* p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }\n");
        self.w("static void     maka_atomic_store_i64(maka_int* p, maka_int v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }\n");
        self.w("static maka_int maka_atomic_fetch_add_i64(maka_int* p, maka_int d) { return __atomic_fetch_add(p, d, __ATOMIC_SEQ_CST); }\n");
        self.w("static maka_int maka_atomic_fetch_sub_i64(maka_int* p, maka_int d) { return __atomic_fetch_sub(p, d, __ATOMIC_SEQ_CST); }\n");
        self.w("static void     maka_fence(maka_int ord) { (void)ord; __atomic_thread_fence(__ATOMIC_SEQ_CST); }\n");
    }

    fn emit_prologue(&mut self) {
        self.w("// generated by makac\n");
        if self.freestanding {
            self.emit_freestanding_prologue();
            return;
        }
        // The hosted runtime below uses POSIX threads (pthread_*) and the
        // GCC/clang atomic builtins (__atomic_*, <stdatomic.h>), neither of which
        // MSVC's cl.exe provides - a cl build otherwise fails with a cryptic
        // C1189 ("C atomics require C11") + C2085 cascade.  Fail early with an
        // actionable message instead.  This fires only for genuine MSVC cl:
        // mingw gcc does not define _MSC_VER, and clang-cl defines __clang__ and
        // does have the atomic builtins.  (Full cl support needs a pthread->Win32
        // + atomics->Interlocked runtime port; see GAPS.md #6.)
        self.w("#if defined(_MSC_VER) && !defined(__clang__)\n");
        self.w("#error \"The Maka runtime uses POSIX threads and GCC/clang atomics; compile the emitted C with mingw gcc or clang (clang-cl), not MSVC cl. Full cl support is unimplemented - see GAPS.md #6.\"\n");
        self.w("#endif\n");
        // Feature-test macros must be defined BEFORE any libc include — once
        // <stdio.h> is parsed, redefining them is a no-op.  On Darwin we need
        // both _DARWIN_C_SOURCE (for non-POSIX extensions like ucontext) and
        // _POSIX_C_SOURCE 200809 (for getline / strdup / clock_gettime).
        self.w("#ifdef __APPLE__\n");
        self.w("#define _DARWIN_C_SOURCE 1\n");
        self.w("#define _POSIX_C_SOURCE 200809L\n");
        self.w("#define _XOPEN_SOURCE 700\n");
        self.w("#elif defined(_WIN32)\n");
        self.w("#define _XOPEN_SOURCE 600\n");
        self.w("#else\n");
        // glibc: _GNU_SOURCE exposes getline, syscall, the socket/inet helpers
        // (send/recv/ntohl), and waitpid.  It must be defined before any libc
        // include.  Older `_XOPEN_SOURCE 600` (POSIX 2004) hid getline/syscall,
        // which modern gcc (14+) then rejects as implicit declarations.
        self.w("#ifndef _GNU_SOURCE\n#define _GNU_SOURCE 1\n#endif\n");
        self.w("#endif\n");
        self.w("#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <stdbool.h>\n#include <string.h>\n#include <wchar.h>\n#include <stdarg.h>\n");
        // POSIX system headers used by the runtime: process control, fds, and
        // process wait.  Emitted up front so the functions are declared before
        // any runtime code calls them (modern gcc rejects implicit decls).
        // Socket/inet headers are deliberately NOT pulled here on Linux - they
        // declare `accept`, which would clash with a user function named
        // `accept` (test 114); the socket runtime self-declares its ABI and
        // calls accept via syscall.  Windows uses the winsock shims below.
        self.w("#ifndef _WIN32\n");
        self.w("#include <unistd.h>\n");
        self.w("#include <fcntl.h>\n");
        self.w("#include <sys/types.h>\n");
        self.w("#include <sys/wait.h>\n");
        // The scheduler's wake-pipe drain calls send/recv early (before the
        // socket runtime is emitted), so forward-declare them here.  Using the
        // POSIX ssize_t/size_t signature keeps this compatible with the real
        // <sys/socket.h> the macOS/BSD socket runtime pulls in later.
        self.w("extern ssize_t send(int, const void*, size_t, int);\n");
        self.w("extern ssize_t recv(int, void*, size_t, int);\n");
        self.w("#endif\n");
        // Windows compatibility headers must come BEFORE any code that uses
        // POSIX functions (read_line, etc.), since the shim provides the
        // declarations.  On non-Windows this expands to a couple of POSIX
        // includes.
        self.w("#ifdef _WIN32\n");
        self.w("#define WIN32_LEAN_AND_MEAN\n");
        self.w("#define NOMINMAX\n");      /* keep windows.h from defining min/max macros */
        self.w("#include <winsock2.h>\n");
        self.w("#include <ws2tcpip.h>\n");
        self.w("#include <windows.h>\n");
        // WIN32_LEAN_AND_MEAN drops the multimedia timer; pull it in explicitly
        // for timeBeginPeriod (the runtime raises the timer resolution to 1 ms).
        self.w("#include <timeapi.h>\n");
        self.w("#include <io.h>\n");
        self.w("#include <fcntl.h>\n");
        // stat/fstat/struct stat used by the file helpers below (mingw provides
        // these through <sys/stat.h>; MSVC maps them to _stat/_fstat).
        self.w("#include <sys/types.h>\n");
        self.w("#include <sys/stat.h>\n");
        self.w("#ifdef max\n#undef max\n#endif\n");
        self.w("#ifdef min\n#undef min\n#endif\n");
        self.w("typedef SSIZE_T ssize_t;\n");
        // pipe() on Windows: emulate with a TCP loopback socket pair so
        // that WSAPoll can wait on the read fd in the reactor.  Otherwise
        // _pipe() returns CRT file descriptors that WSAPoll can't poll.
        // The fds returned ARE SOCKETs cast to int — callers can use
        // recv/send (mapped by our shim) and closesocket transparently.
        self.w("static inline int pipe(int* fds) {\n");
        self.w("    SOCKET listener = socket(AF_INET, SOCK_STREAM, 0);\n");
        self.w("    if (listener == INVALID_SOCKET) return -1;\n");
        self.w("    struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));\n");
        self.w("    addr.sin_family = AF_INET;\n");
        self.w("    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);\n");
        self.w("    addr.sin_port = 0;\n");
        self.w("    if (bind(listener, (struct sockaddr*)&addr, sizeof(addr)) != 0) { closesocket(listener); return -1; }\n");
        self.w("    int alen = sizeof(addr);\n");
        self.w("    if (getsockname(listener, (struct sockaddr*)&addr, &alen) != 0) { closesocket(listener); return -1; }\n");
        self.w("    if (listen(listener, 1) != 0) { closesocket(listener); return -1; }\n");
        // Connect with a non-blocking socket so a stalled connect can't hang
        // the whole scheduler, then accept with a bounded WSAPoll spin.  If
        // either side stalls past the budget we bail out with -1 instead of
        // freezing.
        self.w("    SOCKET writer = socket(AF_INET, SOCK_STREAM, 0);\n");
        self.w("    if (writer == INVALID_SOCKET) { closesocket(listener); return -1; }\n");
        self.w("    u_long nb = 1; ioctlsocket(writer, FIONBIO, &nb);\n");
        self.w("    int r = connect(writer, (struct sockaddr*)&addr, sizeof(addr));\n");
        self.w("    if (r != 0 && WSAGetLastError() != WSAEWOULDBLOCK) { closesocket(listener); closesocket(writer); return -1; }\n");
        self.w("    /* Wait up to ~2 seconds for accept to fire.  Loopback should be ms-fast. */\n");
        self.w("    WSAPOLLFD lpf; lpf.fd = listener; lpf.events = POLLRDNORM; lpf.revents = 0;\n");
        self.w("    if (WSAPoll(&lpf, 1, 2000) <= 0) { closesocket(listener); closesocket(writer); return -1; }\n");
        self.w("    SOCKET reader = accept(listener, NULL, NULL);\n");
        self.w("    closesocket(listener);\n");
        self.w("    if (reader == INVALID_SOCKET) { closesocket(writer); return -1; }\n");
        // Caller (aux_alloc) sets both to non-blocking; restore blocking on
        // writer in case the user code expects sync send().
        self.w("    nb = 0; ioctlsocket(writer, FIONBIO, &nb);\n");
        self.w("    fds[0] = (int)reader;\n");
        self.w("    fds[1] = (int)writer;\n");
        self.w("    return 0;\n");
        self.w("}\n");
        self.w("static inline long getline(char** buf, size_t* cap, FILE* f) {\n");
        self.w("    (void)cap; if (!*buf) *buf = (char*)malloc(4096);\n");
        self.w("    if (!fgets(*buf, 4096, f)) return -1;\n");
        self.w("    return (long)strlen(*buf);\n");
        self.w("}\n");
        self.w("#endif\n");
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
        self.w("#define maka_check_div(d) (__extension__ ({ __typeof__(d) __mdz = (d); if (__mdz == 0) maka_panic(\"divide by zero\"); __mdz; }))\n");
        // Signed division/modulo: panic on /0 AND on the MIN/-1 overflow (C UB),
        // the same way zero and shift overflow panic.  `mn` is the type minimum.
        self.w("#define maka_divs(a, b, mn) (__extension__ ({ __typeof__(a) __dva = (a); __typeof__(b) __dvb = (b); if (__dvb == 0) maka_panic(\"divide by zero\"); if (__dva == (mn) && __dvb == -1) maka_panic(\"division overflow\"); __dva / __dvb; }))\n");
        self.w("#define maka_mods(a, b, mn) (__extension__ ({ __typeof__(a) __dma = (a); __typeof__(b) __dmb = (b); if (__dmb == 0) maka_panic(\"divide by zero\"); if (__dma == (mn) && __dmb == -1) maka_panic(\"division overflow\"); __dma % __dmb; }))\n");
        // Saturating float->int conversion (C makes out-of-range / NaN UB): NaN->0,
        // below-range->MIN, above-range->MAX.  `hi` is 2^(bits-1) (signed) or 2^bits
        // (unsigned) as a double; saturated bounds are returned as ints, never routed
        // through double (which would re-round INT64_MAX back out of range).
        self.w("#define maka_f2i(f, T, hi, mn, mx) (__extension__ ({ double __ff = (double)(f); T __fr; if (__ff != __ff) __fr = 0; else if (__ff >= (hi)) __fr = (mx); else if (__ff < -(hi)) __fr = (mn); else __fr = (T)__ff; __fr; }))\n");
        self.w("#define maka_f2u(f, T, hi, mx) (__extension__ ({ double __ff = (double)(f); T __fr; if (__ff != __ff) __fr = 0; else if (__ff <= 0.0) __fr = 0; else if (__ff >= (hi)) __fr = (mx); else __fr = (T)__ff; __fr; }))\n");
        self.w("#define maka_check_shift(n, w) (__extension__ ({ __typeof__(n) __msh = (n); if ((uint64_t)(__msh) >= (uint64_t)(w)) maka_panic(\"shift amount out of range\"); __msh; }))\n");
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
        // `format(...)` with all-scalar placeholders lowers to one of these (a
        // single allocation), instead of a chain of per-arg concat mallocs.
        self.w("static char* __maka_format1(const char* fmt, ...) { va_list ap; va_start(ap, fmt); va_list ap2; va_copy(ap2, ap); int n = vsnprintf((char*)0, 0, fmt, ap); va_end(ap); char* r = (char*)malloc((size_t)(n < 0 ? 0 : n) + 1); vsnprintf(r, (size_t)(n < 0 ? 0 : n) + 1, fmt, ap2); va_end(ap2); return r; }\n");
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
        // Forward decls for reactor reg cleanup — the close shims (Windows
        // winsock_close + POSIX close_fd) call these before releasing the fd
        // so a recycled fd number doesn't inherit a stale events_mask.
        self.w("static void __maka_fd_arm(int fd, int events_mask);\n");
        self.w("static void __maka_fd_reg_drop(int fd);\n");
        // Forward decls so the close shims can wake fibers parked on a fd
        // before the fd disappears.  Real definitions are emitted later
        // in the scheduler block.
        self.w("struct maka_fiber_s;\n");
        // The typedef name `maka_fiber_t` must be visible to the Win32
        // socket close shim (emitted soon after this block) which uses it
        // as the field/list element type.  The full struct body is filled
        // in much later in the scheduler block; an incomplete-type typedef
        // here is enough for pointer use in the close shim.
        self.w("typedef struct maka_fiber_s maka_fiber_t;\n");
        // maka_fd_waiters is now a maka_sched_state field (shared across a
        // Pool's workers); its accessor macro is defined after the struct, and
        // the close shim that used to reference it early is relocated below it.
        self.w("extern __thread int maka_sched_inited;\n");
        self.w("static void __maka_ready_enqueue(struct maka_fiber_s* f);\n");
        // Forward decls for sched_state ref helpers — ready_enqueue calls
        // them from inside the prologue, before the actual definitions
        // appear in the scheduler block.
        self.w("struct maka_sched_state_s;\n");
        self.w("static inline void __maka_sched_state_ref(struct maka_sched_state_s* s);\n");
        self.w("static inline void __maka_sched_state_unref(struct maka_sched_state_s* s);\n");
        self.w("static inline struct maka_sched_state_s* __maka_sched_validate_and_ref(struct maka_sched_state_s* candidate);\n");
        self.w("static inline struct maka_sched_state_s* __maka_sched_validate_and_ref_epoch(struct maka_sched_state_s* candidate, int64_t expected_epoch);\n");
        // Forward decls for the listdir/str_split builtins so the user-code
        // call sites (which precede the definitions in the same translation
        // unit) see a non-implicit prototype.
        self.w("const char** __maka_rt_file_listdir(const char* path, int64_t* out_n);\n");
        self.w("const char** __maka_rt_str_split(const char* s, const char* sep, int64_t* out_n);\n");
        // ============================================================
        // Windows compatibility layer: every POSIX call the rest of the
        // runtime issues is either provided by mingw-w64's compat headers
        // (pthread, sys/time, sys/stat, etc.) or shimmed below to its
        // Win32 equivalent.  The rest of the runtime code stays POSIX-
        // shaped and these shims translate.
        // ============================================================
        self.w("#ifdef _WIN32\n");
        self.w("#include <stdatomic.h>\n");
        self.w("#include <pthread.h>\n");
        // sysconf
        self.w("#define _SC_NPROCESSORS_ONLN 84\n");
        self.w("static inline long sysconf(int name) {\n");
        self.w("    if (name == _SC_NPROCESSORS_ONLN) { SYSTEM_INFO si; GetSystemInfo(&si); return (long)si.dwNumberOfProcessors; }\n");
        self.w("    return -1;\n");
        self.w("}\n");
        // Memory protection — mmap PROT_NONE + mprotect emulated with VirtualAlloc/VirtualProtect
        self.w("#define PROT_NONE 0\n");
        self.w("#define PROT_READ 1\n");
        self.w("#define PROT_WRITE 2\n");
        self.w("#define MAP_PRIVATE 0\n");
        self.w("#define MAP_ANONYMOUS 0\n");
        self.w("#define MAP_FAILED ((void*)-1)\n");
        self.w("static inline void* mmap(void* a, size_t l, int p, int f, int fd, long off) {\n");
        self.w("    (void)a; (void)f; (void)fd; (void)off;\n");
        self.w("    void* r = VirtualAlloc(NULL, l, MEM_RESERVE, PAGE_NOACCESS);\n");
        self.w("    if (!r) return MAP_FAILED;\n");
        self.w("    if (p != PROT_NONE) {\n");
        self.w("        DWORD prot = PAGE_NOACCESS;\n");
        self.w("        if ((p & PROT_READ) && (p & PROT_WRITE)) prot = PAGE_READWRITE;\n");
        self.w("        else if (p & PROT_READ) prot = PAGE_READONLY;\n");
        self.w("        if (!VirtualAlloc(r, l, MEM_COMMIT, prot)) { VirtualFree(r, 0, MEM_RELEASE); return MAP_FAILED; }\n");
        self.w("    }\n");
        self.w("    return r;\n");
        self.w("}\n");
        self.w("static inline int munmap(void* a, size_t l) { (void)l; VirtualFree(a, 0, MEM_RELEASE); return 0; }\n");
        self.w("static inline int mprotect(void* a, size_t l, int p) {\n");
        self.w("    DWORD prot = PAGE_NOACCESS;\n");
        self.w("    if ((p & PROT_READ) && (p & PROT_WRITE)) prot = PAGE_READWRITE;\n");
        self.w("    else if (p & PROT_READ) prot = PAGE_READONLY;\n");
        self.w("    void* r = VirtualAlloc(a, l, MEM_COMMIT, prot);\n");
        self.w("    return r ? 0 : -1;\n");
        self.w("}\n");
        // ucontext via Win32 Fibers
        self.w("typedef struct { void* fiber; struct { void* ss_sp; size_t ss_size; } uc_stack; void* uc_link; } ucontext_t;\n");
        self.w("static __thread int __maka_win_fiber_inited = 0;\n");
        self.w("static inline void __maka_win_fiber_init(void) {\n");
        self.w("    if (__maka_win_fiber_inited) return;\n");
        self.w("    __maka_win_fiber_inited = 1;\n");
        self.w("    ConvertThreadToFiber(NULL);\n");
        self.w("}\n");
        self.w("static inline int getcontext(ucontext_t* c) { __maka_win_fiber_init(); c->fiber = GetCurrentFiber(); return 0; }\n");
        self.w("static inline void makecontext(ucontext_t* c, void (*f)(void), int n) {\n");
        self.w("    (void)n; __maka_win_fiber_init();\n");
        self.w("    c->fiber = CreateFiber(c->uc_stack.ss_size ? c->uc_stack.ss_size : 0, (LPFIBER_START_ROUTINE)(void*)f, NULL);\n");
        self.w("}\n");
        self.w("static inline int swapcontext(ucontext_t* save, ucontext_t* restore) {\n");
        self.w("    __maka_win_fiber_init();\n");
        self.w("    save->fiber = GetCurrentFiber();\n");
        self.w("    SwitchToFiber(restore->fiber);\n");
        self.w("    return 0;\n");
        self.w("}\n");
        // sockaddr_in: mingw aliases sin_addr.s_addr to S_un.S_addr already.
        // O_NONBLOCK is N/A on Windows; ioctlsocket(FIONBIO) instead.  Provide
        // fcntl no-op shims so existing F_GETFL/F_SETFL calls compile.
        self.w("#define F_GETFL 3\n");
        self.w("#define F_SETFL 4\n");
        self.w("#ifndef O_NONBLOCK\n");
        self.w("#define O_NONBLOCK 0x4000\n");
        self.w("#endif\n");
        self.w("static inline int fcntl(int fd, int cmd, ...) {\n");
        self.w("    if (cmd == F_GETFL) return 0;\n");
        self.w("    if (cmd == F_SETFL) {\n");
        self.w("        u_long nb = 1;\n");
        self.w("        ioctlsocket((SOCKET)fd, FIONBIO, &nb);\n");
        self.w("        return 0;\n");
        self.w("    }\n");
        self.w("    return -1;\n");
        self.w("}\n");
        // Map POSIX errno names to winsock equivalents so the reactor's
        // `if (errno == EAGAIN)` checks work on socket calls (which set
        // WSAGetLastError, not errno).  errno macro gets redirected too.
        self.w("#include <errno.h>\n");
        self.w("#undef EAGAIN\n");
        self.w("#undef EWOULDBLOCK\n");
        self.w("#undef EINTR\n");
        self.w("#undef EINPROGRESS\n");
        self.w("#define EAGAIN      WSAEWOULDBLOCK\n");
        self.w("#define EWOULDBLOCK WSAEWOULDBLOCK\n");
        self.w("#define EINTR       WSAEINTR\n");
        self.w("#define EINPROGRESS WSAEWOULDBLOCK\n");
        self.w("#undef errno\n");
        self.w("#define errno (WSAGetLastError())\n");
        // Socket-aware read/write/close: on Windows sockets aren't file
        // descriptors, so the runtime's `read(fd, ...)` calls won't work
        // on SOCKETs.  Use recv/send/closesocket for sockets; mingw's
        // `_read` / `_write` / `_close` for regular file fds.  We define
        // `read`/`write`/`close` macros that try the socket path first
        // (cheap WSAGetLastError check on failure) then fall back.
        self.w("static inline SSIZE_T __maka_winsock_read(int fd, void* buf, size_t n) {\n");
        self.w("    int r = recv((SOCKET)fd, (char*)buf, (int)n, 0);\n");
        self.w("    if (r >= 0) return r;\n");
        self.w("    if (WSAGetLastError() == WSAENOTSOCK) return (SSIZE_T)_read(fd, buf, (unsigned)n);\n");
        self.w("    return r;\n");
        self.w("}\n");
        self.w("static inline SSIZE_T __maka_winsock_write(int fd, const void* buf, size_t n) {\n");
        self.w("    int r = send((SOCKET)fd, (const char*)buf, (int)n, 0);\n");
        self.w("    if (r >= 0) return r;\n");
        self.w("    if (WSAGetLastError() == WSAENOTSOCK) return (SSIZE_T)_write(fd, buf, (unsigned)n);\n");
        self.w("    return r;\n");
        self.w("}\n");
        // Forward decl only — the body accesses maka_fiber_s fields, which
        // aren't visible until the scheduler block much later in the file.
        // The body is emitted there (search "__maka_winsock_close body").
        self.w("static int __maka_winsock_close(int fd);\n");
        self.w("#define read  __maka_winsock_read\n");
        self.w("#define write __maka_winsock_write\n");
        self.w("#define close __maka_winsock_close\n");
        // Initialize winsock once via a GCC/mingw constructor.
        self.w("static _Atomic int __maka_wsa_inited = 0;\n");
        self.w("static inline void __maka_wsa_init(void) {\n");
        self.w("    int e = 0;\n");
        self.w("    if (!atomic_compare_exchange_strong(&__maka_wsa_inited, &e, 1)) return;\n");
        self.w("    WSADATA wsa;\n");
        self.w("    WSAStartup(MAKEWORD(2, 2), &wsa);\n");
        self.w("}\n");
        self.w("__attribute__((constructor)) static void __maka_wsa_ctor(void) { __maka_wsa_init(); }\n");
        // Force stdout/stderr binary mode so `printf("%d\n", ...)` outputs LF
        // (not CRLF) — matches POSIX semantics and the expected file format.
        // Also force line buffering so log() output appears promptly even
        // when the program later hangs in a reactor wait.
        self.w("__attribute__((constructor)) static void __maka_bin_stdio(void) {\n");
        self.w("    _setmode(_fileno(stdin),  _O_BINARY);\n");
        self.w("    _setmode(_fileno(stdout), _O_BINARY);\n");
        self.w("    _setmode(_fileno(stderr), _O_BINARY);\n");
        self.w("    setvbuf(stdout, NULL, _IONBF, 0);\n");
        self.w("    setvbuf(stderr, NULL, _IONBF, 0);\n");
        // Raise the Windows multimedia timer to 1 ms so nanosleep / WaitableTimer
        // resolutions match the rest of the runtime.  Default tick is 15.6 ms
        // which makes sub-50 ms sleep_ms / timerfd round badly.
        self.w("#ifdef _WIN32\n");
        self.w("    timeBeginPeriod(1);\n");
        self.w("#endif\n");
        self.w("}\n");
        // mingw's pread/pwrite are sometimes missing.  Provide robust shims:
        //   * Use 64-bit offset (Win64 `long` is 32-bit — using it would cap
        //     positional IO at 2 GiB and silently truncate file pointers).
        //   * Treat ReadFile/WriteFile failure with ERROR_HANDLE_EOF as a
        //     clean 0-byte EOF rather than -1 (matches POSIX behavior).
        //   * Loop over chunks larger than DWORD_MAX so very big buffers
        //     don't get silently truncated.
        self.w("static inline ssize_t pread (int fd, void* buf, size_t n, int64_t off) {\n");
        self.w("    HANDLE h = (HANDLE)_get_osfhandle(fd);\n");
        self.w("    if (h == INVALID_HANDLE_VALUE) return -1;\n");
        self.w("    size_t total = 0;\n");
        self.w("    while (total < n) {\n");
        self.w("        size_t remain = n - total;\n");
        self.w("        DWORD chunk = remain > 0x40000000u ? 0x40000000u : (DWORD)remain;\n");
        self.w("        int64_t pos = off + (int64_t)total;\n");
        self.w("        OVERLAPPED ov = {0};\n");
        self.w("        ov.Offset     = (DWORD)((uint64_t)pos & 0xFFFFFFFFull);\n");
        self.w("        ov.OffsetHigh = (DWORD)((uint64_t)pos >> 32);\n");
        self.w("        DWORD got = 0;\n");
        self.w("        if (!ReadFile(h, (char*)buf + total, chunk, &got, &ov)) {\n");
        self.w("            DWORD e = GetLastError();\n");
        self.w("            if (e == ERROR_HANDLE_EOF) return (ssize_t)total;\n");
        self.w("            return total == 0 ? -1 : (ssize_t)total;\n");
        self.w("        }\n");
        self.w("        if (got == 0) return (ssize_t)total;  /* EOF */\n");
        self.w("        total += got;\n");
        self.w("        if (got < chunk) break;\n");
        self.w("    }\n");
        self.w("    return (ssize_t)total;\n");
        self.w("}\n");
        self.w("static inline ssize_t pwrite(int fd, const void* buf, size_t n, int64_t off) {\n");
        self.w("    HANDLE h = (HANDLE)_get_osfhandle(fd);\n");
        self.w("    if (h == INVALID_HANDLE_VALUE) return -1;\n");
        self.w("    size_t total = 0;\n");
        self.w("    while (total < n) {\n");
        self.w("        size_t remain = n - total;\n");
        self.w("        DWORD chunk = remain > 0x40000000u ? 0x40000000u : (DWORD)remain;\n");
        self.w("        int64_t pos = off + (int64_t)total;\n");
        self.w("        OVERLAPPED ov = {0};\n");
        self.w("        ov.Offset     = (DWORD)((uint64_t)pos & 0xFFFFFFFFull);\n");
        self.w("        ov.OffsetHigh = (DWORD)((uint64_t)pos >> 32);\n");
        self.w("        DWORD wrote = 0;\n");
        self.w("        if (!WriteFile(h, (const char*)buf + total, chunk, &wrote, &ov)) {\n");
        self.w("            return total == 0 ? -1 : (ssize_t)total;\n");
        self.w("        }\n");
        self.w("        if (wrote == 0) break;\n");
        self.w("        total += wrote;\n");
        self.w("        if (wrote < chunk) break;\n");
        self.w("    }\n");
        self.w("    return (ssize_t)total;\n");
        self.w("}\n");
        // CLOCK_MONOTONIC for clock_gettime
        self.w("#ifndef CLOCK_MONOTONIC\n");
        self.w("#define CLOCK_MONOTONIC 1\n");
        self.w("#endif\n");
        self.w("#else\n");
        // POSIX path:
        self.w("#include <pthread.h>\n");
        self.w("#include <stdatomic.h>\n");
        // ucontext.h is deprecated on macOS (POSIX.1-2008 removed it) —
        // _DARWIN_C_SOURCE + _XOPEN_SOURCE were set at the top of the
        // prologue, before any libc includes, so this picks them up.
        self.w("#include <ucontext.h>\n");
        self.w("#endif\n");
        self.w("maka_unit* maka_mutex_new(void) { pthread_mutex_t* m = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t)); pthread_mutex_init(m, NULL); return (maka_unit*)m; }\n");
        self.w("void maka_mutex_lock(maka_unit* m) { pthread_mutex_lock((pthread_mutex_t*)m); }\n");
        self.w("void maka_mutex_unlock(maka_unit* m) { pthread_mutex_unlock((pthread_mutex_t*)m); }\n");
        self.w("void maka_mutex_destroy(maka_unit* m) { pthread_mutex_destroy((pthread_mutex_t*)m); free(m); }\n");
        // RwLock: FIBER-AWARE, defined with the other fiber-aware sync primitives
        // (Mutex/WaitGroup/Once) after the scheduler infrastructure - a plain
        // pthread_rwlock blocks the whole OS thread, deadlocking the cooperative
        // scheduler when a fiber holds the read lock across a yield and another
        // fiber write-locks.  Forward-declare here; define near maka_fmutex.
        self.w("maka_unit* maka_rwlock_new(void);\n");
        self.w("void maka_rwlock_read_lock(maka_unit* r);\n");
        self.w("void maka_rwlock_write_lock(maka_unit* r);\n");
        self.w("void maka_rwlock_unlock(maka_unit* r);\n");
        self.w("void maka_rwlock_destroy(maka_unit* r);\n");
        // Spinlock: pthread_spinlock_t on Linux/BSD/Windows, os_unfair_lock
        // on macOS (Darwin removed pthread_spinlock_t).  os_unfair_lock is
        // the Apple-recommended replacement and has the same surface.
        self.w("#ifdef __APPLE__\n");
        self.w("#include <os/lock.h>\n");
        self.w("maka_unit* maka_spinlock_new(void) { os_unfair_lock* s = (os_unfair_lock*)malloc(sizeof(os_unfair_lock)); *s = OS_UNFAIR_LOCK_INIT; return (maka_unit*)s; }\n");
        self.w("void maka_spinlock_lock(maka_unit* s) { os_unfair_lock_lock((os_unfair_lock*)s); }\n");
        self.w("void maka_spinlock_unlock(maka_unit* s) { os_unfair_lock_unlock((os_unfair_lock*)s); }\n");
        self.w("void maka_spinlock_destroy(maka_unit* s) { free(s); }\n");
        self.w("#else\n");
        self.w("maka_unit* maka_spinlock_new(void) { pthread_spinlock_t* s = (pthread_spinlock_t*)malloc(sizeof(pthread_spinlock_t)); pthread_spin_init(s, PTHREAD_PROCESS_PRIVATE); return (maka_unit*)s; }\n");
        self.w("void maka_spinlock_lock(maka_unit* s) { pthread_spin_lock((pthread_spinlock_t*)s); }\n");
        self.w("void maka_spinlock_unlock(maka_unit* s) { pthread_spin_unlock((pthread_spinlock_t*)s); }\n");
        self.w("void maka_spinlock_destroy(maka_unit* s) { pthread_spin_destroy((pthread_spinlock_t*)s); free(s); }\n");
        self.w("#endif\n");
        // ===== Maka primitive runtime helpers =====
        // futex_wait / futex_wake — direct kernel-wait/wake-on-address.
        //   Linux:   syscall(SYS_futex, ...)
        //   Windows: WaitOnAddress / WakeByAddressSingle (Vista+)
        //   macOS:   __ulock_wait / __ulock_wake (private but stable since 10.12)
        // thread_yield — sched_yield on POSIX / SwitchToThread on Win.
        // syscall — generic kernel call with up to 6 args (variadic shim).
        self.w("#ifdef __linux__\n");
        self.w("#include <linux/futex.h>\n");
        self.w("#include <sys/syscall.h>\n");
        self.w("#include <unistd.h>\n");
        self.w("#include <sched.h>\n");
        self.w("static int __maka_futex_wait(const int* addr, int expected) {\n");
        self.w("    return (int)syscall(SYS_futex, (int*)addr, FUTEX_WAIT, expected, NULL, NULL, 0);\n");
        self.w("}\n");
        self.w("static int __maka_futex_wake(const int* addr, int n) {\n");
        self.w("    return (int)syscall(SYS_futex, (int*)addr, FUTEX_WAKE, n, NULL, NULL, 0);\n");
        self.w("}\n");
        self.w("static void __maka_thread_yield(void) { sched_yield(); }\n");
        self.w("static long __maka_syscall(long n, long a1, long a2, long a3, long a4, long a5, long a6) {\n");
        self.w("    return syscall(n, a1, a2, a3, a4, a5, a6);\n");
        self.w("}\n");
        self.w("#elif defined(_WIN32)\n");
        // WaitOnAddress + WakeByAddress live in Synchronization.lib (mingw
        // also auto-links via this comment).
        self.w("#pragma comment(lib, \"Synchronization.lib\")\n");
        self.w("static int __maka_futex_wait(const int* addr, int expected) {\n");
        self.w("    int local = expected;\n");
        self.w("    return WaitOnAddress((volatile void*)addr, &local, sizeof(int), INFINITE) ? 0 : -1;\n");
        self.w("}\n");
        self.w("static int __maka_futex_wake(const int* addr, int n) {\n");
        self.w("    if (n <= 1) { WakeByAddressSingle((void*)addr); } else { WakeByAddressAll((void*)addr); }\n");
        self.w("    return 0;\n");
        self.w("}\n");
        self.w("static void __maka_thread_yield(void) { SwitchToThread(); }\n");
        self.w("static long __maka_syscall(long n, long a1, long a2, long a3, long a4, long a5, long a6) {\n");
        // Windows has no unified syscall surface; this is a stub.  Returns
        // -1; we don't touch errno because the Win32 codegen earlier in this
        // prologue #defines errno to a function-call macro (not an lvalue).
        // Users targeting Windows should reach for the typed Win32 APIs
        // directly, not raw syscalls.
        self.w("    (void)n;(void)a1;(void)a2;(void)a3;(void)a4;(void)a5;(void)a6; return -1;\n");
        self.w("}\n");
        self.w("#else\n");
        // Darwin/BSD — fall back to a spin-yield emulation for futex (no
        // kernel address-waits in the public API on Darwin); sched_yield()
        // exists on every POSIX so use it for yield.
        self.w("#include <sched.h>\n");
        self.w("#include <sys/syscall.h>\n");
        self.w("#include <unistd.h>\n");
        self.w("static int __maka_futex_wait(const int* addr, int expected) {\n");
        // Polling fallback — checks every 1 ms until *addr != expected.  Not
        // ideal but correct.  Future: __ulock_wait on Darwin.
        self.w("    while (__atomic_load_n(addr, __ATOMIC_SEQ_CST) == expected) {\n");
        self.w("        struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL);\n");
        self.w("    }\n");
        self.w("    return 0;\n");
        self.w("}\n");
        self.w("static int __maka_futex_wake(const int* addr, int n) { (void)addr; (void)n; return 0; }\n");
        self.w("static void __maka_thread_yield(void) { sched_yield(); }\n");
        self.w("static long __maka_syscall(long n, long a1, long a2, long a3, long a4, long a5, long a6) {\n");
        self.w("    return syscall(n, a1, a2, a3, a4, a5, a6);\n");
        self.w("}\n");
        self.w("#endif\n");
        // Channel: a simple unbounded queue of int64_t protected by a mutex + condvar.
        self.w("typedef struct maka_chan_node_t { maka_int v; struct maka_chan_node_t* next; } maka_chan_node_t;\n");
        self.w("typedef struct { pthread_mutex_t m; pthread_cond_t c; maka_chan_node_t* head; maka_chan_node_t* tail; maka_int count; int closed; int waiters; pthread_cond_t drained_cv; } maka_channel_t;\n");
        self.w("maka_unit* maka_channel_new(void) { maka_channel_t* ch = (maka_channel_t*)calloc(1, sizeof(maka_channel_t)); pthread_mutex_init(&ch->m, NULL); pthread_cond_init(&ch->c, NULL); pthread_cond_init(&ch->drained_cv, NULL); return (maka_unit*)ch; }\n");
        self.w("void maka_channel_send(maka_unit* p, maka_int v) { maka_channel_t* ch = (maka_channel_t*)p; maka_chan_node_t* n = (maka_chan_node_t*)malloc(sizeof(maka_chan_node_t)); n->v = v; n->next = NULL; pthread_mutex_lock(&ch->m); if (ch->closed) { free(n); pthread_mutex_unlock(&ch->m); return; } if (ch->tail) ch->tail->next = n; else ch->head = n; ch->tail = n; ch->count++; pthread_cond_signal(&ch->c); pthread_mutex_unlock(&ch->m); }\n");
        // recv: re-test 'closed' on every wake — if destroy is in progress we
        // bail out so the destroy can pthread_cond_destroy safely.
        self.w("maka_int maka_channel_recv(maka_unit* p) {\n");
        self.w("    maka_channel_t* ch = (maka_channel_t*)p;\n");
        self.w("    pthread_mutex_lock(&ch->m);\n");
        self.w("    ch->waiters++;\n");
        self.w("    while (!ch->head && !ch->closed) pthread_cond_wait(&ch->c, &ch->m);\n");
        self.w("    ch->waiters--;\n");
        self.w("    if (ch->waiters == 0) pthread_cond_signal(&ch->drained_cv);\n");
        self.w("    maka_int v = 0;\n");
        self.w("    if (ch->head) {\n");
        self.w("        maka_chan_node_t* n = ch->head; ch->head = n->next;\n");
        self.w("        if (!ch->head) ch->tail = NULL; ch->count--;\n");
        self.w("        v = n->v; free(n);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&ch->m);\n");
        self.w("    return v;\n");
        self.w("}\n");
        self.w("void maka_channel_destroy(maka_unit* p) {\n");
        self.w("    maka_channel_t* ch = (maka_channel_t*)p;\n");
        // Close, broadcast, wait for all waiters to leave the cv before
        // destroying the mutex/cv (POSIX UB otherwise).
        self.w("    pthread_mutex_lock(&ch->m);\n");
        self.w("    ch->closed = 1;\n");
        self.w("    while (ch->head) { maka_chan_node_t* n = ch->head; ch->head = n->next; free(n); }\n");
        self.w("    pthread_cond_broadcast(&ch->c);\n");
        self.w("    while (ch->waiters > 0) pthread_cond_wait(&ch->drained_cv, &ch->m);\n");
        self.w("    pthread_mutex_unlock(&ch->m);\n");
        self.w("    pthread_mutex_destroy(&ch->m); pthread_cond_destroy(&ch->c); pthread_cond_destroy(&ch->drained_cv); free(ch);\n");
        self.w("}\n");
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
        self.w("    int waiters;\n");
        self.w("    pthread_cond_t drained_cv;\n");
        // Fiber-park support (mirrors maka_fmutex_t): a spawned (non-anchor) fiber
        // blocking on an empty recv parks itself onto `fiber_waiters` and yields to
        // the scheduler instead of pthread_cond_wait'ing (which would block the whole
        // OS thread and deadlock the sibling producer fiber).  `inflight_parkers`
        // lets destroy() wait until all armed-but-not-finalized parkers settle.
        self.w("    _Atomic int    inflight_parkers;\n");
        self.w("    maka_fiber_t*  fiber_waiters;\n");
        self.w("} maka_bchan_t;\n");
        // Late-defined (needs the scheduler): re-enqueue one (all=0) or all (all=1)
        // parked fiber(s) as ready.  Called by send/close/destroy.
        self.w("void __maka_bchan_wake(maka_bchan_t* c, int all);\n");
        self.w("maka_unit* maka_chan_bytes_new(int64_t item_size) {\n");
        // Clamp to [0, INT_MAX] so the (int) cast can't truncate to negative,
        // which would later promote to a giant size_t in malloc/memcpy and
        // corrupt the heap.  Negative values are also rejected.
        self.w("    if (item_size < 0) item_size = 0;\n");
        self.w("    if (item_size > 0x7fffffffLL) item_size = 0x7fffffffLL;\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)calloc(1, sizeof(maka_bchan_t));\n");
        self.w("    c->item_size = (int)item_size;\n");
        self.w("    pthread_mutex_init(&c->m, NULL);\n");
        self.w("    pthread_cond_init(&c->c, NULL);\n");
        self.w("    pthread_cond_init(&c->drained_cv, NULL);\n");
        self.w("    return (maka_unit*)c;\n");
        self.w("}\n");
        self.w("void maka_chan_bytes_send(maka_unit* p, maka_unit* src) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)p;\n");
        self.w("    maka_bnode_t* n = (maka_bnode_t*)malloc(sizeof(maka_bnode_t) + (size_t)c->item_size);\n");
        self.w("    memcpy(n->data, (void*)src, (size_t)c->item_size);\n");
        self.w("    n->next = NULL;\n");
        self.w("    pthread_mutex_lock(&c->m);\n");
        // Refuse to push to a closed channel — destroy() may free the
        // mutex/cv as soon as it sees waiters==0, and pushing on a
        // half-destroyed channel races on freed memory.
        self.w("    if (c->closed) { free(n); pthread_mutex_unlock(&c->m); return; }\n");
        self.w("    if (c->tail) c->tail->next = n; else c->head = n;\n");
        self.w("    c->tail = n; c->count++;\n");
        self.w("    pthread_cond_signal(&c->c);\n");
        self.w("    pthread_mutex_unlock(&c->m);\n");
        self.w("    __maka_bchan_wake(c, 0);\n");
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
        self.w("    __maka_bchan_wake(c, 1);\n");
        self.w("}\n");
        self.w("void maka_chan_bytes_destroy(maka_unit* p) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)p;\n");
        self.w("    pthread_mutex_lock(&c->m);\n");
        self.w("    c->closed = 1;\n");
        self.w("    while (c->head) { maka_bnode_t* n = c->head; c->head = n->next; free(n); }\n");
        self.w("    pthread_cond_broadcast(&c->c);\n");
        self.w("    pthread_mutex_unlock(&c->m);\n");
        // Wake every parked fiber (they see `closed` and return) and wait until all
        // pthread waiters AND in-flight/parked fibers have settled before freeing —
        // POSIX UB otherwise, and freeing under a parked fiber would corrupt it.
        self.w("    while (1) {\n");
        self.w("        __maka_bchan_wake(c, 1);\n");
        self.w("        pthread_mutex_lock(&c->m);\n");
        self.w("        if (c->waiters == 0 && atomic_load(&c->inflight_parkers) == 0 && !c->fiber_waiters) { pthread_mutex_unlock(&c->m); break; }\n");
        self.w("        pthread_cond_wait(&c->drained_cv, &c->m);\n");
        self.w("        pthread_mutex_unlock(&c->m);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_destroy(&c->m); pthread_cond_destroy(&c->c); pthread_cond_destroy(&c->drained_cv);\n");
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
        self.w("#include <unistd.h>\n#include <time.h>\n#include <errno.h>\n#include <signal.h>\n#include <fcntl.h>\n");
        // sys/stat + sys/types pulled in for stat/mkdir/struct stat — Linux
        // includes them transitively but macOS doesn't, so make it explicit.
        self.w("#ifndef _WIN32\n");
        self.w("#include <sys/types.h>\n");
        self.w("#include <sys/stat.h>\n");
        self.w("#endif\n");
        self.w("typedef struct Thread {\n");
        self.w("    pthread_t       handle;\n");
        self.w("    pthread_mutex_t done_mutex;\n");
        self.w("    pthread_cond_t  done_cond;\n");
        self.w("    int             done_flag;       /* set to 1 when work finishes */\n");
        self.w("    int64_t         result;          /* type-erased return value */\n");
        self.w("    int             is_job;          /* 1 for job-pool work item */\n");
        self.w("    int             is_fiber;        /* 1 for cooperative fiber */\n");
        self.w("    _Atomic int     detached;        /* 1 if user opted out of join (informational) */\n");
        // is_failed: 1 iff pthread_create failed for this Thread; in that
        // case `handle` is uninitialized so join paths must skip pthread_join.
        self.w("    _Atomic int     is_failed;\n");
        // cancel_requested: set by __maka_cancel when the target's home_sched
        // is not yet pinned (spawn_pool pre-pickup); the pool worker checks
        // this before invoking the fiber body so cancel doesn't silently
        // become a no-op race.
        self.w("    _Atomic int     cancel_requested;\n");
        // Set by file_async wrappers while a detached AIO worker still holds
        // refs to caller-owned buf/fd.  cancel_fiber_local refuses to free
        // a fiber whose Thread has this set — the fiber finishes naturally
        // once eventfd_recv returns.
        self.w("    _Atomic int     aio_in_flight;\n");
        // Refcount-based cleanup ownership.  Both the spawner-side handle and
        // the runner-side completion hold one ref each.  detach() and join()
        // drop the handle ref; the runner drops its ref after setting
        // done_flag + broadcasting.  Last drop frees, no detached race.
        self.w("    _Atomic int     refcount;\n");
        // home_sched: the scheduler on which the runner runs (for fibers).
        // Used by cross-thread cancel() / select() to post requests onto
        // the right thread's inbox instead of walking the wrong queues.
        // Stored as void* so the forward decl doesn't require the full type.
        self.w("    _Atomic(void*)  home_sched;\n");
        self.w("    _Atomic int64_t home_sched_epoch;\n");
        // env_drop: per-closure recursive drop for the capture env, run before the
        // env is freed so an owner MOVED into the thread (capture-by-value of
        // `own *T`) is freed by the thread - even on cancel, where the body never
        // runs.  NULL when the env owns no heap.  calloc'd, so default NULL.
        self.w("    void (*env_drop)(void*);\n");
        // Fiber joiners parked on this Thread's completion.  A joining FIBER
        // must yield its worker (park here) rather than block it, or a Pool
        // deadlocks when every worker blocks on a child that needs a free
        // worker.  Drained + re-enqueued by the runner's completion path under
        // done_mutex.  (Distinct from the fiber's own f->waiters so a joiner
        // can park without having to locate the running fiber.)
        self.w("    maka_fiber_t* fiber_waiters;\n");
        self.w("} Thread;\n");
        // Bump/drop refs.  free(t) is illegal — always go through unref.
        self.w("static inline void __maka_thread_ref(Thread* t) {\n");
        self.w("    if (t) atomic_fetch_add(&t->refcount, 1);\n");
        self.w("}\n");
        self.w("static inline void __maka_thread_unref(Thread* t) {\n");
        self.w("    if (!t) return;\n");
        self.w("    if (atomic_fetch_sub(&t->refcount, 1) == 1) {\n");
        self.w("        pthread_mutex_destroy(&t->done_mutex);\n");
        self.w("        pthread_cond_destroy(&t->done_cond);\n");
        self.w("        free(t);\n");
        self.w("    }\n");
        self.w("}\n");
        // Free a thread/fiber/job capture env, first running its recursive drop
        // (t->env_drop) so any owner MOVED into the thread is freed exactly once,
        // here, by the thread - regardless of whether the body ran (cancel).
        self.w("static inline void __maka_free_env(Thread* t, void* env) {\n");
        self.w("    if (env && t && t->env_drop) t->env_drop(env);\n");
        self.w("    free(env);\n");
        self.w("}\n");
        // Allocator that initializes the mutex/cond + refcount=2 (one for the
        // returned handle, one for the runner side).  Every Thread alloc site
        // should go through this so the ownership invariant holds.
        self.w("static inline Thread* __maka_thread_new(void) {\n");
        self.w("    Thread* t = (Thread*)calloc(1, sizeof(Thread));\n");
        self.w("    pthread_mutex_init(&t->done_mutex, NULL);\n");
        self.w("    pthread_cond_init (&t->done_cond,  NULL);\n");
        self.w("    atomic_init(&t->refcount, 2);\n");
        self.w("    return t;\n");
        self.w("}\n");
        // pthread_create wrapper that handles EAGAIN by marking the Thread
        // done (so join() doesn't hang), dropping the runner-side ref, and
        // freeing the worker arg.  Callers can then proceed as if the worker
        // ran instantly to completion.
        // Convert a UTF-8 path to a freshly-malloc'd UTF-16 string for Win32
        // *W file APIs.  Caller frees.  Returns NULL on invalid input.
        self.w("#ifdef _WIN32\n");
        self.w("static inline WCHAR* __maka_path_to_w(const char* path) {\n");
        self.w("    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);\n");
        self.w("    if (wlen <= 0) return NULL;\n");
        self.w("    WCHAR* w = (WCHAR*)calloc((size_t)wlen, sizeof(WCHAR));\n");
        self.w("    if (!w) return NULL;\n");
        self.w("    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, w, wlen) <= 0) { free(w); return NULL; }\n");
        self.w("    return w;\n");
        self.w("}\n");
        self.w("#endif\n");
        self.w("static inline int __maka_spawn_pthread(pthread_t* h, void* (*entry)(void*), void* arg, Thread* completion) {\n");
        self.w("    int rc = pthread_create(h, NULL, entry, arg);\n");
        self.w("    if (rc != 0 && completion) {\n");
        // DO NOT free arg here — the caller may still use it after join
        // returns immediately (e.g. par_reduce reads chs[c]->partial).  The
        // is_failed flag tells the join path to skip pthread_join (handle is
        // uninitialized).  The caller's normal cleanup loop will free the arg.
        self.w("        atomic_store(&completion->is_failed, 1);\n");
        self.w("        pthread_mutex_lock(&completion->done_mutex);\n");
        self.w("        completion->done_flag = 1;\n");
        self.w("        pthread_cond_broadcast(&completion->done_cond);\n");
        self.w("        pthread_mutex_unlock(&completion->done_mutex);\n");
        self.w("        __maka_thread_unref(completion);  /* runner ref */\n");
        self.w("    }\n");
        self.w("    return rc;\n");
        self.w("}\n");
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
        // Free the capture env (running env_drop) BEFORE dropping the runner
        // ref: the unref can free the Thread (when join/detach already dropped
        // the other ref), and __maka_free_env reads t->env_drop.
        self.w("    __maka_free_env(a->h, a->env);\n");
        // Drop the runner-side ref so join/detach can free.  Without this,
        // every thread() leaks a Thread + its mutex/cond.
        self.w("    __maka_thread_unref(a->h);\n");
        self.w("    free(a);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        // thread() — kernel thread tier, default ~8 MB stack.
        self.w("maka_unit* __maka_spawn_thread(void* code, void* env, void* env_drop) {\n");
        self.w("    Thread* t = __maka_thread_new();\n");
        self.w("    t->env_drop = (void(*)(void*))env_drop;\n");
        self.w("    __maka_handle_args_t* a = (__maka_handle_args_t*)malloc(sizeof(__maka_handle_args_t));\n");
        self.w("    a->code = code; a->env = env; a->h = t;\n");
        self.w("    __maka_spawn_pthread(&t->handle, __maka_handle_entry, a, t);\n");
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
        self.w("#ifndef _WIN32\n");
        self.w("#include <sys/mman.h>\n");
        self.w("#endif\n");
        self.w("#define MAKA_FIBER_STACK_SIZE (64 * 1024)\n");
        self.w("#define MAKA_FIBER_SLAB_RESERVE (1024 * 1024) /* 1 MB VM per fiber */\n");
        self.w("#define MAKA_FIBER_GUARD_PAGE 4096\n");
        // Lightweight fiber context switch: custom callee-saved-register swap on
        // x86_64/aarch64 (Linux + macOS), ucontext / native-fiber fallback
        // elsewhere (incl. Windows).  Drop-in for the old swapcontext.
        self.w(r#"/* ---- maka_ctx_switch / maka_ctx_make ---------------------------------
 * A hand-written callee-saved register swap replaces ucontext swapcontext
 * (which does a sigprocmask syscall per switch, ~1us) with a ~15ns save and
 * restore.  A context is just a saved stack pointer; all register state lives
 * on the fiber's own stack.  x86_64 and aarch64 (Linux + macOS) take the asm
 * path; Windows and every other arch fall back to ucontext / native OS fibers,
 * so the runtime still builds and runs everywhere.  FP control state (x86
 * MXCSR / aarch64 FPCR rounding mode) is thread-shared and intentionally NOT
 * swapped per fiber - do not change the FP rounding mode across a yield. */
#if (defined(__x86_64__) || defined(__aarch64__)) && !defined(_WIN32)
#define MAKA_ASM_CTX 1
typedef struct maka_mctx_s { void* sp; } maka_mctx_t;
extern void maka_ctx_switch(maka_mctx_t* from, maka_mctx_t* to);
#if defined(__APPLE__)
#define MAKA_CTX_SYM "_maka_ctx_switch"
#else
#define MAKA_CTX_SYM "maka_ctx_switch"
#endif
#if defined(__x86_64__)
__asm__(
".text\n"
".globl " MAKA_CTX_SYM "\n"
".p2align 4\n"
MAKA_CTX_SYM ":\n"
"  pushq %rbp\n"
"  pushq %rbx\n"
"  pushq %r12\n"
"  pushq %r13\n"
"  pushq %r14\n"
"  pushq %r15\n"
"  movq  %rsp, (%rdi)\n"
"  movq  (%rsi), %rsp\n"
"  popq  %r15\n"
"  popq  %r14\n"
"  popq  %r13\n"
"  popq  %r12\n"
"  popq  %rbx\n"
"  popq  %rbp\n"
"  ret\n"
);
static void maka_ctx_make(maka_mctx_t* c, void* base, size_t size, void (*entry)(void)) {
    uintptr_t top = ((uintptr_t)base + size) & ~(uintptr_t)0xF; /* 16-align top */
    void** sp = (void**)top;
    *(--sp) = (void*)0;       /* pad: keep the return slot 16-aligned */
    *(--sp) = (void*)entry;   /* return address the switch `ret` pops */
    *(--sp) = (void*)0;       /* rbp */
    *(--sp) = (void*)0;       /* rbx */
    *(--sp) = (void*)0;       /* r12 */
    *(--sp) = (void*)0;       /* r13 */
    *(--sp) = (void*)0;       /* r14 */
    *(--sp) = (void*)0;       /* r15 */
    c->sp = (void*)sp;
}
#else /* __aarch64__ */
__asm__(
".text\n"
".globl " MAKA_CTX_SYM "\n"
".p2align 4\n"
MAKA_CTX_SYM ":\n"
"  sub  sp, sp, #0xa0\n"
"  stp  x19, x20, [sp, #0x00]\n"
"  stp  x21, x22, [sp, #0x10]\n"
"  stp  x23, x24, [sp, #0x20]\n"
"  stp  x25, x26, [sp, #0x30]\n"
"  stp  x27, x28, [sp, #0x40]\n"
"  stp  x29, x30, [sp, #0x50]\n"
"  stp  d8,  d9,  [sp, #0x60]\n"
"  stp  d10, d11, [sp, #0x70]\n"
"  stp  d12, d13, [sp, #0x80]\n"
"  stp  d14, d15, [sp, #0x90]\n"
"  mov  x2, sp\n"
"  str  x2, [x0]\n"
"  ldr  x2, [x1]\n"
"  mov  sp, x2\n"
"  ldp  x19, x20, [sp, #0x00]\n"
"  ldp  x21, x22, [sp, #0x10]\n"
"  ldp  x23, x24, [sp, #0x20]\n"
"  ldp  x25, x26, [sp, #0x30]\n"
"  ldp  x27, x28, [sp, #0x40]\n"
"  ldp  x29, x30, [sp, #0x50]\n"
"  ldp  d8,  d9,  [sp, #0x60]\n"
"  ldp  d10, d11, [sp, #0x70]\n"
"  ldp  d12, d13, [sp, #0x80]\n"
"  ldp  d14, d15, [sp, #0x90]\n"
"  add  sp, sp, #0xa0\n"
"  ret\n"
);
static void maka_ctx_make(maka_mctx_t* c, void* base, size_t size, void (*entry)(void)) {
    uintptr_t top = ((uintptr_t)base + size) & ~(uintptr_t)0xF;
    top -= 0xa0;                              /* one 160-byte save frame */
    uint64_t* fr = (uint64_t*)top;
    for (int i = 0; i < 20; i++) fr[i] = 0;
    fr[11] = (uint64_t)(uintptr_t)entry;      /* x30 (lr) slot -> entry */
    c->sp = (void*)top;
}
#endif
#else /* portable fallback: ucontext (Windows native fibers via the shims) */
typedef ucontext_t maka_mctx_t;
static inline void maka_ctx_switch(maka_mctx_t* from, maka_mctx_t* to) { swapcontext(from, to); }
static void maka_ctx_make(maka_mctx_t* c, void* base, size_t size, void (*entry)(void)) {
    getcontext(c);
    c->uc_stack.ss_sp = base;
    c->uc_stack.ss_size = size;
    c->uc_link = NULL;
    makecontext(c, entry, 0);
}
#endif
"#);
        // Per-thread free-list of slabs.  Reusing a slab avoids the mmap/munmap
        // cost on every spawn — same trick goroutines use.
        self.w("typedef struct maka_slab_s {\n");
        self.w("    void* base;                  /* mmap base (guard page at top) */\n");
        self.w("    void* stack_top;             /* start of usable stack region */\n");
        self.w("    size_t stack_size;           /* usable (committed) stack bytes */\n");
        self.w("    size_t reserve;              /* mmap'd VM span (for munmap) */\n");
        self.w("    int cacheable;               /* 1 = default-size, returns to the cache */\n");
        self.w("    struct maka_slab_s* next;    /* free-list link */\n");
        self.w("} maka_slab_t;\n");
        // Only default-size (64 KB) slabs are cached; a custom-size fiber stack
        // (Pool with a user stack size) gets a dedicated mmap that is munmap'd on
        // free rather than pooled - keeps the hot path a fixed-size free-list.
        self.w("static __thread maka_slab_t* maka_slab_pool = NULL;\n");
        self.w("static maka_slab_t* __maka_slab_alloc(size_t stack) {\n");
        self.w("    if (stack == 0) stack = MAKA_FIBER_STACK_SIZE;\n");
        self.w("    if (stack == (size_t)MAKA_FIBER_STACK_SIZE && maka_slab_pool) {\n");
        self.w("        maka_slab_t* s = maka_slab_pool;\n");
        self.w("        maka_slab_pool = s->next; s->next = NULL; s->stack_size = stack;\n");
        self.w("        return s;\n");
        self.w("    }\n");
        self.w("#ifdef _WIN32\n");
        // Win32 Fibers manage their own stack — slab is a stub carrying the size
        // (CreateFiber in the shim uses ctx.ss_size = stack_size).
        self.w("    maka_slab_t* s = (maka_slab_t*)calloc(1, sizeof(maka_slab_t));\n");
        self.w("    if (s) { s->stack_size = stack; s->cacheable = (stack == (size_t)MAKA_FIBER_STACK_SIZE); }\n");
        self.w("    return s; /* may be NULL on OOM; callers must handle */\n");
        self.w("#else\n");
        self.w("    /* Reserve VM (PROT_NONE), commit the top `stack` bytes RW, leave the\n");
        self.w("       rest below as a PROT_NONE overflow guard.  Default size uses the\n");
        self.w("       1 MB reserve (poolable); a custom size reserves stack + one\n");
        self.w("       default-size guard span. */\n");
        self.w("    size_t reserve = (stack == (size_t)MAKA_FIBER_STACK_SIZE) ? (size_t)MAKA_FIBER_SLAB_RESERVE : (stack + (size_t)MAKA_FIBER_STACK_SIZE);\n");
        self.w("    void* base = mmap(NULL, reserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);\n");
        self.w("    if (base == MAP_FAILED) return NULL;\n");
        self.w("    void* commit_start = (char*)base + reserve - stack;\n");
        self.w("    if (mprotect(commit_start, stack, PROT_READ | PROT_WRITE) != 0) {\n");
        self.w("        munmap(base, reserve);\n");
        self.w("        return NULL;\n");
        self.w("    }\n");
        self.w("    maka_slab_t* s = (maka_slab_t*)malloc(sizeof(maka_slab_t));\n");
        self.w("    if (!s) { munmap(base, reserve); return NULL; }\n");
        self.w("    s->base = base;\n");
        self.w("    s->stack_top = commit_start;\n");
        self.w("    s->stack_size = stack;\n");
        self.w("    s->reserve = reserve;\n");
        self.w("    s->cacheable = (stack == (size_t)MAKA_FIBER_STACK_SIZE);\n");
        self.w("    s->next = NULL;\n");
        self.w("    return s;\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("static void __maka_slab_free(maka_slab_t* s) {\n");
        // Default-size slabs return to the per-thread free-list (cheap reuse).
        // A custom-size stack is released immediately (never pooled).
        self.w("    if (s->cacheable) { s->next = maka_slab_pool; maka_slab_pool = s; return; }\n");
        self.w("#ifndef _WIN32\n");
        self.w("    if (s->base) munmap(s->base, s->reserve);\n");
        self.w("#endif\n");
        self.w("    free(s);\n");
        self.w("}\n");
        // Release THIS thread's cached slabs (unmap the reserves + free the
        // structs).  Immortal global-pool workers never call this - it exists
        // for explicit Pool workers, which exit on pool_shutdown and would
        // otherwise leak their per-thread slab cache.
        self.w("static void __maka_slab_drain(void) {\n");
        self.w("    maka_slab_t* s = maka_slab_pool; maka_slab_pool = NULL;\n");
        self.w("    while (s) {\n");
        self.w("        maka_slab_t* nx = s->next;\n");
        self.w("#ifndef _WIN32\n");
        self.w("        if (s->base) munmap(s->base, s->reserve);\n");
        self.w("#endif\n");
        self.w("        free(s); s = nx;\n");
        self.w("    }\n");
        self.w("}\n");
        // home_sched is set at fiber-spawn time and points to the scheduler
        // state of the pthread the fiber was created on.  swapcontext is only
        // safe between contexts in the same pthread, so cross-thread wakes
        // must route through the home thread's remote-wake queue.
        self.w("struct maka_sched_state_s;\n");
        self.w("typedef struct maka_fiber_s {\n");
        self.w("    maka_mctx_t ctx;\n");
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
        // Atomic so cross-thread enqueuers can read with acquire ordering.
        self.w("    _Atomic(void*) home_sched;\n");
        self.w("    _Atomic int64_t home_sched_epoch;  /* snapshot of home_sched->epoch at spawn */\n");
        // Dedupe flag for ready_enqueue — prevents self-loop / double-link if
        // a remote wake fires concurrently with a local one.
        self.w("    _Atomic int in_queue;\n");
        self.w("    struct maka_fiber_s* next;        /* ready / sleep / fd-wait queue link */\n");
        self.w("    struct maka_fiber_s* waiters;     /* fibers blocked on this fiber */\n");
        self.w("    struct maka_fiber_s* next_waiter; /* waiter list link */\n");
        self.w("} maka_fiber_t;\n");
        // __maka_winsock_close body is relocated below the maka_sched_state
        // struct + accessor macros (it walks maka_fd_waiters, now a struct
        // field reached through maka_sched_state).
        // Per-pthread scheduler state, addressable from other threads.
        // remote_wake_head is the cross-thread inbox protected by remote_mu;
        // wake_pipe_w is the self-pipe write end the remote pusher pings so
        // the home thread's epoll_wait returns.
        // Lightweight remote cancel request — a Thread* the home scheduler
        // should reap.  Allocated by the cross-thread cancel(), freed by the
        // home scheduler after processing.
        self.w("typedef struct maka_cancel_req_s {\n");
        self.w("    Thread* target;\n");
        self.w("    struct maka_cancel_req_s* next;\n");
        self.w("} maka_cancel_req_t;\n");
        // Cross-thread close request — asks the home scheduler to wake any
        // local fiber waiting on `closed_fd`.  Without this, a fiber parked
        // on thread A's fd_waiters waiting on fd X hangs forever when
        // thread B calls close_fd(X).
        self.w("typedef struct maka_close_req_s {\n");
        self.w("    int closed_fd;\n");
        self.w("    struct maka_close_req_s* next;\n");
        self.w("} maka_close_req_t;\n");
        // --- Per-worker lock-free FIFO run queue (work-stealing) ---------------
        // A fiber scheduler needs FIFO (fair `yield`: a yielding fiber lets peers
        // run before it), so this is a Go/Tokio-style ring - push at tail, pop or
        // steal at head - NOT the LIFO Chase-Lev the `job` pool uses (jobs are
        // fork-join, where LIFO-local is optimal; fibers are cooperative, where
        // LIFO would let a yielding fiber immediately re-run itself).  Single
        // producer (the owner pushes tail), multi consumer (owner pops + thieves
        // steal, both CAS head).  Overflow spills to the group's global linked
        // run queue (run_head/tail); cross-thread wakes land there too.  head and
        // tail are int64 so they never wrap in practice (no ABA).
        self.w("#define MAKA_FRQ_CAP 256\n");
        self.w("typedef struct {\n");
        self.w("    _Atomic int64_t head;\n");
        self.w("    _Atomic int64_t tail;\n");
        self.w("    maka_fiber_t* buf[MAKA_FRQ_CAP];\n");
        self.w("} maka_frunq_t;\n");
        // Owner push at tail (single producer).  Returns 0 if full (caller spills
        // to the group's global queue).
        self.w("static inline int __maka_frq_push(maka_frunq_t* q, maka_fiber_t* f) {\n");
        self.w("    int64_t t = atomic_load_explicit(&q->tail, memory_order_relaxed);\n");
        self.w("    int64_t h = atomic_load_explicit(&q->head, memory_order_acquire);\n");
        self.w("    if (t - h >= MAKA_FRQ_CAP) return 0;\n");
        self.w("    q->buf[t % MAKA_FRQ_CAP] = f;\n");
        self.w("    atomic_thread_fence(memory_order_release);\n");
        self.w("    atomic_store_explicit(&q->tail, t + 1, memory_order_relaxed);\n");
        self.w("    return 1;\n");
        self.w("}\n");
        // Pop (owner) / steal (thief) at head - identical, both CAS head, FIFO.
        self.w("static inline maka_fiber_t* __maka_frq_take(maka_frunq_t* q) {\n");
        self.w("    for (;;) {\n");
        self.w("        int64_t h = atomic_load_explicit(&q->head, memory_order_acquire);\n");
        self.w("        atomic_thread_fence(memory_order_seq_cst);\n");
        self.w("        int64_t t = atomic_load_explicit(&q->tail, memory_order_acquire);\n");
        self.w("        if (h >= t) return NULL;\n");
        self.w("        maka_fiber_t* f = q->buf[h % MAKA_FRQ_CAP];\n");
        self.w("        int64_t expected = h;\n");
        self.w("        if (atomic_compare_exchange_strong_explicit(&q->head, &expected, h + 1, memory_order_seq_cst, memory_order_relaxed)) return f;\n");
        self.w("    }\n");
        self.w("}\n");
        // Scheduler group: the set of worker threads that share a run queue.  The
        // HOT path is per-worker lock-free FIFO deques (`frunqs`) with stealing;
        // `run_head`/`run_tail` is the group's global overflow + cross-thread
        // landing queue (guarded by `lock`, a no-op for a lone cooperative
        // worker).  n_workers==1 is a lone worker (cooperative `spawn`).
        self.w("struct maka_fd_reg_s;\n");
        self.w("typedef struct maka_sched_group_s {\n");
        self.w("    pthread_mutex_t lock;\n");
        self.w("    int n_workers;\n");
        self.w("    maka_fiber_t* run_head;\n");
        self.w("    maka_fiber_t* run_tail;\n");
        // Slice 2: when a group has N>1 workers (a Pool), an idle worker blocks on
        // `cv` for a ready fiber; enqueue signals it.  `closed` + `inflight` drive
        // shutdown.  Unused for a lone worker (n_workers==1).
        self.w("    pthread_cond_t cv;\n");
        self.w("    int closed;\n");
        self.w("    _Atomic long inflight;\n");
        // Exactly one worker drives the shared reactor (epoll_wait) at a time;
        // `polling` is 1 while that worker is in the reactor.  Others cv-wait.
        self.w("    int polling;\n");
        // is_pool: 1 for an explicit Pool's shared group (even a Pool of 1),
        // 0 for a lone cooperative worker (spawn/main) or a spawn_pool worker.
        // Gates all pool scheduling behavior: locking, cv coordination, the
        // poll-owner idle wait (wait-for-work instead of return-to-anchor), and
        // inflight accounting.  Cooperative groups keep zero overhead.
        self.w("    int is_pool;\n");
        // Per-worker lock-free FIFO deques (the work-stealing hot path) + the
        // atomic id dispenser workers claim their slot from on attach.  frunqs
        // has n_workers slots; frunqs[id] is owned by worker `id`, stolen from
        // by peers.  run_head/tail above is the shared overflow/cross-thread
        // fallback.
        self.w("    _Atomic(maka_frunq_t*) *frunqs;\n");
        self.w("    _Atomic int worker_seq;\n");
        self.w("} maka_sched_group_t;\n");
        self.w("typedef struct maka_sched_state_s {\n");
        self.w("    pthread_mutex_t remote_mu;\n");
        self.w("    maka_fiber_t* remote_wake_head;\n");
        // Cross-thread cancel inbox.  Same locking + same wake_pipe; the
        // drain function pulls + processes both.
        self.w("    maka_cancel_req_t* remote_cancel_head;\n");
        // Cross-thread close inbox — populated by close_fd() on other
        // threads to wake any local fiber parked on the now-closed fd.
        self.w("    maka_close_req_t* remote_close_head;\n");
        self.w("    int wake_pipe_r;\n");
        self.w("    int wake_pipe_w;\n");
        // Refcount-based teardown.  The owning thread holds 1; every
        // cross-thread enqueuer bumps before locking remote_mu, drops after.
        // Cleanup (pthread_key destructor) drops the owner ref last; last
        // drop frees.  Prevents UAF when a remote thread is still posting
        // wake/cancel requests as the home thread exits.
        self.w("    _Atomic int refcount;\n");
        // Monotonic epoch — incremented from a global atomic at alloc time.
        // Prevents a recycled (malloc-reused) sched_state address from
        // passing validation against a fiber that was bound to the old
        // scheduler that lived at the same address.
        self.w("    int64_t epoch;\n");
        self.w("    maka_sched_group_t* group;   /* shared ready queue for this worker's group */\n");
        // Reactor + timer state.  Moved out of per-thread TLS so a Pool's
        // workers, which share one sched_state, share one reactor: a fiber
        // parked on sleep/IO is serviced by whichever worker owns the poll,
        // so it does NOT pin the worker it parked on.  For a lone worker this
        // is its own private reactor exactly as before.
        self.w("    maka_fiber_t* sleep_head;    /* timer list (min-deadline scan) */\n");
        self.w("    maka_fiber_t* fd_waiters;    /* fibers parked on an fd */\n");
        self.w("    int epoll_fd;                /* shared reactor fd (-1 until first use) */\n");
        self.w("    struct maka_fd_reg_s* fd_regs; /* fd -> armed-mask registry (shared) */\n");
        // kqueue (macOS/BSD) armed-filter cache.  On those platforms epoll_fd
        // holds the kqueue fd; this per-fd cache moves out of TLS into the
        // sched_state so a Pool's workers share ONE kq with a consistent
        // arm/disarm view (a lone/spawn_pool worker keeps its own, as before).
        self.w("#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)\n");
        self.w("    int kq_armed_in[1024];\n");
        self.w("    int kq_armed_out[1024];\n");
        self.w("#endif\n");
        self.w("} maka_sched_state_t;\n");
        self.w("static _Atomic int64_t __maka_sched_epoch_ctr = 1;\n");
        self.w("static __thread maka_sched_state_t* maka_sched_state = NULL;\n");
        // This worker's own lock-free deque + its id within the group (the slot
        // it owns in group->frunqs).  Set in sched_init (lone worker id 0) and
        // in the pool worker attach.
        self.w("static __thread maka_frunq_t* maka_my_frunq = NULL;\n");
        self.w("static __thread int maka_my_worker_id = 0;\n");
        // xorshift32 for random steal-victim selection (per-worker, no shared state).
        self.w("static __thread unsigned int maka_steal_rng = 2463534242u;\n");
        self.w("static __thread maka_mctx_t maka_sched_ctx;\n");
        self.w("static __thread maka_fiber_t* maka_current_fiber = NULL;\n");
        // The ready queue lives in the worker's group (shared across a Pool's
        // workers).  Macros keep every access site unchanged; a lone worker's
        // group is size 1, so this is the current per-thread queue verbatim.
        self.w("#define maka_ready_head (maka_sched_state->group->run_head)\n");
        self.w("#define maka_ready_tail (maka_sched_state->group->run_tail)\n");
        // Reactor/timer accessors — reach the (possibly shared) sched_state
        // fields.  Every existing access site stays textually unchanged; a lone
        // worker's sched_state is private, so this is the old TLS behavior.
        self.w("#define maka_sleep_head (maka_sched_state->sleep_head)\n");
        self.w("#define maka_fd_waiters (maka_sched_state->fd_waiters)\n");
        self.w("#define maka_epoll_fd (maka_sched_state->epoll_fd)\n");
        self.w("#define maka_fd_regs (maka_sched_state->fd_regs)\n");
        // Group lock: a Pool's N workers share one sched_state, so the run
        // queue + timer list + fd-waiter list they all touch need mutual
        // exclusion.  A lone worker (n_workers==1) never contends, so the
        // lock is a NO-OP there and the cooperative/spawn/job path is
        // provably unchanged.  The mutex is RECURSIVE (see __maka_grp_mtx_init)
        // so nested critical sections (a locked phase that calls
        // __maka_ready_enqueue, which locks again) are safe by construction;
        // the one rule is that cond_wait runs at lock depth 1 (the idle path
        // never nests), so it fully releases.
        self.w("#define MAKA_GRP_LOCK()   do { if (maka_sched_state->group->is_pool) pthread_mutex_lock(&maka_sched_state->group->lock); } while (0)\n");
        self.w("#define MAKA_GRP_UNLOCK() do { if (maka_sched_state->group->is_pool) pthread_mutex_unlock(&maka_sched_state->group->lock); } while (0)\n");
        // Relocated close shim (see forward note above): now that the struct +
        // accessors are in scope it can walk maka_fd_waiters correctly.
        self.w("#ifdef _WIN32\n");
        self.w("static int __maka_winsock_close(int fd) {\n");
        self.w("    if (maka_sched_inited) {\n");
        self.w("        maka_fiber_t** prev = &maka_fd_waiters;\n");
        self.w("        while (*prev) {\n");
        self.w("            maka_fiber_t* w = *prev;\n");
        self.w("            if (w->waiting_fd == fd) {\n");
        self.w("                *prev = w->next; w->next = NULL;\n");
        self.w("                w->waiting_fd = -1; w->waiting_events = 0;\n");
        self.w("                w->wait_deadline_ns = 0; w->wait_timed_out = 0;\n");
        self.w("                __maka_ready_enqueue(w);\n");
        self.w("            } else {\n");
        self.w("                prev = &(*prev)->next;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    __maka_fd_arm(fd, 0);\n");
        self.w("    __maka_fd_reg_drop(fd);\n");
        self.w("    if (closesocket((SOCKET)fd) == 0) return 0;\n");
        self.w("    if (WSAGetLastError() == WSAENOTSOCK) return _close(fd);\n");
        self.w("    return -1;\n");
        self.w("}\n");
        self.w("#endif\n");
        // Non-static so the forward decl in the top of the prologue can find it.
        self.w("__thread int maka_sched_inited = 0;\n");
        // Park-request mechanism: parker arms maka_pending_park before
        // swapcontext-out; scheduler's finalize-park block (right after the
        // swap returns) does the actual list push under the parker's lock.
        // Eliminates the parker-unlock vs swap-save race entirely.
        self.w("typedef struct maka_park_req_s {\n");
        self.w("    pthread_mutex_t* lock;\n");
        self.w("    maka_fiber_t**   head;\n");
        self.w("    int (*should_park)(void*);\n");
        self.w("    void* arg;\n");
        // Optional in-flight counter + drained_cv pointers.  Parker increments
        // *inflight under *lock BEFORE arming; finalize-park decrements after
        // push/skip and signals drained_cv on drop-to-zero.  Lets destroy()
        // wait until all armed-but-not-pushed parkers have finalized.  Set to
        // NULL when the primitive is lifecycle-managed by refcount instead.
        self.w("    _Atomic int*    inflight;\n");
        self.w("    pthread_cond_t* drained_cv;\n");
        // kind: 0 = primitive waiter (the fields above); 1 = sleep (defer the
        // push onto maka_sleep_head); 2 = fd (defer push onto maka_fd_waiters +
        // epoll arm).  Sleep/fd defer their list insertion to AFTER the fiber
        // has fully switched out, so in a Pool a peer worker cannot wake+run
        // the fiber before its context is saved (double-run).  This is the
        // same race the waiter finalize already prevents, extended to the
        // timer + reactor.
        self.w("    int kind;\n");
        self.w("} maka_park_req_t;\n");
        self.w("static __thread maka_park_req_t* maka_pending_park = NULL;\n");
        // Reused per-worker park request for the sleep/fd deferred insertion.
        self.w("static __thread maka_park_req_t maka_timerfd_park;\n");
        // Blocking-syscall watchdog: each scheduler updates `last_tick_ns`
        // at the top of every loop iteration.  If MAKA_WATCHDOG_MS is set in
        // the env, a global watchdog thread periodically checks every
        // registered scheduler — if its tick hasn't advanced past the
        // threshold AND it has pending work, we warn on stderr.  The
        // assumption is that the fiber is stuck in a blocking syscall.
        // Forward decl so the tick can carry a sched_state pointer.
        self.w("struct maka_sched_state_s;\n");
        self.w("typedef struct maka_sched_tick_s {\n");
        self.w("    struct maka_sched_state_s* owner;  /* the sched_state this tick belongs to */\n");
        self.w("    int64_t epoch;                     /* mirrored from owner->epoch */\n");
        self.w("    _Atomic int64_t last_tick_ns;\n");
        self.w("    _Atomic int     has_work;\n");
        self.w("    _Atomic int     warned;\n");
        self.w("    struct maka_sched_tick_s* next;\n");
        self.w("} maka_sched_tick_t;\n");
        self.w("static pthread_mutex_t __maka_ticks_mu = PTHREAD_MUTEX_INITIALIZER;\n");
        self.w("static maka_sched_tick_t* __maka_ticks_head = NULL;\n");
        self.w("static __thread maka_sched_tick_t* __maka_my_tick = NULL;\n");
        self.w("static _Atomic int __maka_watchdog_started = 0;\n");
        // _Atomic: several scheduler-init threads check-then-set this concurrently
        // (each computes the same value from MAKA_WATCHDOG_MS, so it is idempotent,
        // but the plain accesses are a C11 data race).  Atomic access is TSan-clean.
        self.w("static _Atomic int64_t __maka_watchdog_threshold_ns = 0;\n");
        // The awaited completion (a refcounted Thread, NOT the fiber): the
        // fiber is freed by the completion handler as soon as it finishes, so
        // holding the fiber here and reading `->completion` later is a
        // use-after-free.  The Thread outlives the fiber (the joiner holds a ref).
        self.w("static __thread Thread* maka_join_target = NULL;\n");
        self.w("static __thread maka_fiber_t* maka_anchor_fiber = NULL;\n");
        self.w("static __thread int maka_anchor_wake_on_finish = 0;\n");
        self.w("static __thread int64_t maka_anchor_deadline_ns = 0; /* 0 = none; otherwise scheduler caps its timeout so anchor wakes by this */\n");
        // Reactor backend selection.  Three backends:
        //   * Linux           — epoll(7) used directly.
        //   * macOS / *BSD    — kqueue(2), via a shim exposing the epoll API.
        //   * Any other POSIX — poll(2) fallback (slower; rebuilds pollfd[]
        //                       from maka_fd_regs each scheduler tick).
        //
        // Windows is a known gap: the fiber model relies on ucontext_t
        // (getcontext/swapcontext/makecontext) which doesn't exist on
        // Windows.  A real port would replace those with Win32 Fibers
        // (CreateFiber/SwitchToFiber) and the reactor with WSAPoll or IOCP.
        // That's a multi-day refactor that needs actual Windows hardware
        // to verify, deliberately deferred.
        self.w("#ifdef _WIN32\n");
        self.w("#include <winsock2.h>\n");
        self.w("#include <ws2tcpip.h>\n");
        self.w("#define poll WSAPoll\n");
        self.w("typedef unsigned long nfds_t;\n");
        self.w("#else\n");
        self.w("#include <poll.h>\n");
        self.w("#endif\n");
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
        self.w("typedef struct maka_epoll_event_t { int events; union { int fd; void* ptr; } data; } maka_epoll_event_t;\n");
        self.w("#define epoll_event maka_epoll_event_t\n");
        // kqueue: lazy-open kq with FD_CLOEXEC + per-thread instance.  Each
        // thread runs an independent scheduler with its own __thread waiter
        // list, so they must each have their own kq — a process-global kq
        // would deliver events for thread A's fds to thread B's kevent call.
        // EPOLL_CTL_MOD diffs old vs new mask so dropped filters get
        // EV_DELETE'd; EPOLL_CTL_DEL splits the two filters into separate
        // kevent calls so ENOENT on one doesn't suppress the other.
        // The kq fd lives in maka_epoll_fd (the sched_state reactor field),
        // shared across a Pool's workers exactly like the Linux epoll fd; the
        // armed-filter cache lives in the sched_state too so arm/disarm stays
        // consistent whichever worker services the shared kq.
        self.w("static inline int epoll_create1(int flags) {\n");
        self.w("    (void)flags;\n");
        self.w("    int kq = kqueue();\n");
        self.w("    if (kq >= 0) { int f = fcntl(kq, F_GETFD); if (f >= 0) fcntl(kq, F_SETFD, f | FD_CLOEXEC); }\n");
        self.w("    return kq;\n");
        self.w("}\n");
        self.w("static inline int epoll_ctl(int ep, int op, int fd, struct epoll_event* e) {\n");
        self.w("    if (ep < 0) return -1;\n");
        self.w("    int want_in  = e && (e->events & EPOLLIN);\n");
        self.w("    int want_out = e && (e->events & EPOLLOUT);\n");
        // For fds >= 1024 we can't cache armed state in the static table;
        // pretend "not armed if we want it" and "armed if we want to drop it"
        // so the change always gets emitted (EV_ADD is idempotent, EV_DELETE
        // of an unarmed filter just returns ENOENT via EV_RECEIPT).
        self.w("    int armed_in  = (fd >= 0 && fd < 1024) ? maka_sched_state->kq_armed_in [fd] : !want_in;\n");
        self.w("    int armed_out = (fd >= 0 && fd < 1024) ? maka_sched_state->kq_armed_out[fd] : !want_out;\n");
        self.w("    if (op == EPOLL_CTL_DEL) { want_in = 0; want_out = 0; armed_in = 1; armed_out = 1; }\n");
        self.w("    /* Issue each filter change separately with EV_RECEIPT so we\n");
        self.w("       see per-change errors (kevent stops processing on first\n");
        self.w("       error when not using EV_RECEIPT). */\n");
        self.w("    int rc = 0;\n");
        self.w("    struct kevent change, recv;\n");
        self.w("    if (want_in && !armed_in) {\n");
        self.w("        EV_SET(&change, fd, EVFILT_READ, EV_ADD | EV_RECEIPT, 0, 0, NULL);\n");
        self.w("        (void)kevent(ep, &change, 1, &recv, 1, NULL);\n");
        self.w("        if (recv.data == 0 && fd >= 0 && fd < 1024) maka_sched_state->kq_armed_in[fd] = 1;\n");
        self.w("        else if (recv.data != 0) rc = -1;\n");
        self.w("    } else if (!want_in && armed_in) {\n");
        self.w("        EV_SET(&change, fd, EVFILT_READ, EV_DELETE | EV_RECEIPT, 0, 0, NULL);\n");
        self.w("        (void)kevent(ep, &change, 1, &recv, 1, NULL);\n");
        self.w("        if (fd >= 0 && fd < 1024) maka_sched_state->kq_armed_in[fd] = 0;\n");
        self.w("    }\n");
        self.w("    if (want_out && !armed_out) {\n");
        self.w("        EV_SET(&change, fd, EVFILT_WRITE, EV_ADD | EV_RECEIPT, 0, 0, NULL);\n");
        self.w("        (void)kevent(ep, &change, 1, &recv, 1, NULL);\n");
        self.w("        if (recv.data == 0 && fd >= 0 && fd < 1024) maka_sched_state->kq_armed_out[fd] = 1;\n");
        self.w("        else if (recv.data != 0) rc = -1;\n");
        self.w("    } else if (!want_out && armed_out) {\n");
        self.w("        EV_SET(&change, fd, EVFILT_WRITE, EV_DELETE | EV_RECEIPT, 0, 0, NULL);\n");
        self.w("        (void)kevent(ep, &change, 1, &recv, 1, NULL);\n");
        self.w("        if (fd >= 0 && fd < 1024) maka_sched_state->kq_armed_out[fd] = 0;\n");
        self.w("    }\n");
        self.w("    return rc;\n");
        self.w("}\n");
        self.w("static inline int __maka_epoll_wait_kq(int ep, struct epoll_event* evs, int max, int timeout_ms) {\n");
        self.w("    if (ep < 0) return 0;\n");
        self.w("    struct kevent kevs[32]; if (max > 32) max = 32;\n");
        self.w("    struct timespec ts; struct timespec* pts = NULL;\n");
        self.w("    if (timeout_ms >= 0) { ts.tv_sec = timeout_ms/1000; ts.tv_nsec = (timeout_ms%1000)*1000000; pts = &ts; }\n");
        self.w("    int n;\n");
        self.w("    /* Retry on EINTR - otherwise the scheduler treats a signal\n");
        self.w("       interrupt as a full quantum elapsed and busy-loops. */\n");
        self.w("    do { n = kevent(ep, NULL, 0, kevs, max, pts); }\n");
        self.w("    while (n < 0 && errno == EINTR);\n");
        self.w("    if (n < 0) return 0;  /* other errors: treat as no events */\n");
        self.w("    int out = 0;\n");
        self.w("    for (int i = 0; i < n; i++) {\n");
        self.w("        evs[out].events = 0;\n");
        self.w("        if (kevs[i].filter == EVFILT_READ)  evs[out].events |= EPOLLIN;\n");
        self.w("        if (kevs[i].filter == EVFILT_WRITE) evs[out].events |= EPOLLOUT;\n");
        /* EV_EOF on READ means peer closed AND no bytes remain.
           Don't set EPOLLHUP for WRITE-side EV_EOF (it means writer
           half-closed, but a fiber waiting on READ shouldn't get
           spuriously woken). */
        self.w("        if ((kevs[i].flags & EV_EOF) && kevs[i].filter == EVFILT_READ && kevs[i].data == 0)\n");
        self.w("            evs[out].events |= EPOLLHUP;\n");
        self.w("        if (kevs[i].flags & EV_ERROR)       evs[out].events |= EPOLLERR;\n");
        self.w("        evs[out].data.fd = (int)kevs[i].ident;\n");
        self.w("        out++;\n");
        self.w("    }\n");
        self.w("    return out;\n");
        self.w("}\n");
        self.w("#define epoll_wait(ep, evs, max, t) __maka_epoll_wait_kq((ep), (evs), (max), (t))\n");
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
        self.w("typedef struct maka_epoll_event_t { int events; union { int fd; void* ptr; } data; } maka_epoll_event_t;\n");
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
        // maka_fd_regs is a sched_state field (accessor macro above) so a Pool's
        // workers share one fd->armed-mask registry on the shared reactor.
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
        self.w("    int rv;\n");
        self.w("    do { rv = poll(pfds, (nfds_t)n, timeout_ms); }\n");
        self.w("    while (rv < 0 && errno == EINTR);\n");
        self.w("    int out = 0;\n");
        self.w("    if (rv > 0) {\n");
        self.w("        for (int j = 0; j < n && out < max; j++) {\n");
        self.w("            if (pfds[j].revents) {\n");
        self.w("                int re = pfds[j].revents;\n");
        // WSAPoll doesn't report POLLHUP on graceful peer close.  Probe with
        // a zero-byte MSG_PEEK recv: if recv == 0 the peer half-closed —
        // synthesize POLLHUP so waiters get woken instead of blocking forever.
        self.w("#ifdef _WIN32\n");
        self.w("                if ((re & POLLIN) && !(re & POLLHUP)) {\n");
        self.w("                    char c; int p = recv(pfds[j].fd, &c, 1, MSG_PEEK);\n");
        self.w("                    if (p == 0) re |= POLLHUP;\n");
        self.w("                }\n");
        self.w("#endif\n");
        self.w("                evs[out].events = re;\n");
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
        // No scheduler => no reactor registry (maka_fd_regs is
        // maka_sched_state->fd_regs); nothing was registered, nothing to drop.
        self.w("    if (!maka_sched_state) return;\n");
        self.w("    maka_fd_reg_t** prev = &maka_fd_regs;\n");
        self.w("    while (*prev) {\n");
        self.w("        if ((*prev)->fd == fd) {\n");
        self.w("            maka_fd_reg_t* r = *prev; *prev = r->next; free(r); return;\n");
        self.w("        }\n");
        self.w("        prev = &(*prev)->next;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("#define MAKA_EV_READ  1\n");
        self.w("#define MAKA_EV_WRITE 2\n");
        self.w("static __thread char maka_sched_stack[256 * 1024];\n");
        // Is there any ready fiber to run - this worker's deque, the group's
        // global overflow/cross-thread queue, or (pool) any peer's deque?  Used
        // by the watchdog, the pool idle re-check, and the anchor wait-loops
        // that decide whether to drive the scheduler vs block.  Lock-free reads;
        // a stale answer is harmless (the actual dequeue re-checks).
        self.w("static inline int __maka_have_ready(void) {\n");
        self.w("    if (maka_my_frunq && atomic_load_explicit(&maka_my_frunq->head, memory_order_acquire) < atomic_load_explicit(&maka_my_frunq->tail, memory_order_acquire)) return 1;\n");
        self.w("    if (maka_ready_head) return 1;\n");
        self.w("    if (maka_sched_state->group->is_pool) {\n");
        self.w("        maka_sched_group_t* g = maka_sched_state->group;\n");
        self.w("        if (g->frunqs) for (int i = 0; i < g->n_workers; i++) {\n");
        self.w("            maka_frunq_t* q = atomic_load_explicit(&g->frunqs[i], memory_order_relaxed);\n");
        self.w("            if (q && atomic_load_explicit(&q->head, memory_order_acquire) < atomic_load_explicit(&q->tail, memory_order_acquire)) return 1;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    return 0;\n");
        self.w("}\n");
        self.w("static void __maka_ready_enqueue(maka_fiber_t* f) {\n");
        // Dedupe: if the fiber is already queued somewhere (state==0 with a
        // non-self next), enqueueing again would either splice it into a
        // self-loop or duplicate it in two queues.  Both are catastrophic.
        self.w("    if (atomic_exchange(&f->in_queue, 1) != 0) return;\n");
        self.w("    f->state = 0; f->next = NULL;\n");
        // Cross-thread wake?  Acquire-load home_sched, then VALIDATE it's
        // still in the global registry under __maka_ticks_mu (which the dtor
        // also holds when unlinking).  The validate-and-ref helper bumps
        // refcount inside the lock, so the sched can't be freed mid-use.
        self.w("    maka_sched_state_t* candidate = (maka_sched_state_t*)atomic_load_explicit((_Atomic(void*)*)&f->home_sched, memory_order_acquire);\n");
        self.w("    if (candidate && candidate != maka_sched_state) {\n");
        self.w("        maka_sched_state_t* home = __maka_sched_validate_and_ref_epoch(candidate, atomic_load_explicit(&f->home_sched_epoch, memory_order_acquire));\n");
        self.w("        if (!home) {\n");
        // Home thread is gone — set done_flag on the fiber's completion so
        // any pending join() exits instead of hanging forever.
        self.w("            if (f->completion) {\n");
        self.w("                pthread_mutex_lock(&f->completion->done_mutex);\n");
        self.w("                f->completion->done_flag = 1;\n");
        self.w("                pthread_cond_broadcast(&f->completion->done_cond);\n");
        self.w("                pthread_mutex_unlock(&f->completion->done_mutex);\n");
        self.w("            }\n");
        self.w("            return;\n");
        self.w("        }\n");
        self.w("        pthread_mutex_lock(&home->remote_mu);\n");
        self.w("        f->next = home->remote_wake_head;\n");
        self.w("        home->remote_wake_head = f;\n");
        self.w("        int wfd = home->wake_pipe_w;\n");
        self.w("        pthread_mutex_unlock(&home->remote_mu);\n");
        // wake_pipe is a SOCKET pair on every platform now (so send/recv work);
        // EAGAIN/EWOULDBLOCK is fine — the pending byte is already there.
        self.w("        if (wfd > 0) { char b = 1; (void)send((int)wfd, &b, 1, 0); }\n");
        self.w("        __maka_sched_state_unref(home);\n");
        self.w("        return;\n");
        self.w("    }\n");
        // Local wake: push to THIS worker's lock-free deque (hot path, no lock).
        // Only when it is full (>256 ready fibers on one worker) spill to the
        // group's global queue under the lock.  A lone cooperative worker never
        // contends the lock (it is a no-op there).
        self.w("    if (!(maka_my_frunq && __maka_frq_push(maka_my_frunq, f))) {\n");
        self.w("        MAKA_GRP_LOCK();\n");
        self.w("        if (maka_ready_tail) { maka_ready_tail->next = f; maka_ready_tail = f; }\n");
        self.w("        else { maka_ready_head = maka_ready_tail = f; }\n");
        self.w("        MAKA_GRP_UNLOCK();\n");
        self.w("    }\n");
        // Pool: wake a peer so it can steal this fiber (idle peers cv-wait; the
        // poll-owner is in epoll_wait, woken via the shared wake_pipe).
        self.w("    if (maka_sched_state->group->is_pool) {\n");
        self.w("        pthread_cond_signal(&maka_sched_state->group->cv);\n");
        self.w("        if (maka_sched_state->group->polling && maka_sched_state->wake_pipe_w > 0) { char b = 1; (void)send(maka_sched_state->wake_pipe_w, &b, 1, 0); }\n");
        self.w("    }\n");
        self.w("    MAKA_GRP_UNLOCK();\n");
        self.w("}\n");
        // Drain remote-wake inbox into local ready queue.  Called at the top
        // of the scheduler loop and after any epoll_wait that read from the
        // wake_pipe.  Reverse the LIFO push so FIFO semantics are preserved.
        // Forward decl so drain can invoke the local cancel helper before it
        // is textually defined.
        self.w("static int __maka_cancel_fiber_local(Thread* t);\n");
        self.w("static void __maka_drain_remote_wakes(void) {\n");
        self.w("    if (!maka_sched_state) return;\n");
        // Drain the wake pipe FIRST, BEFORE snapshotting and unlocking.
        // Otherwise a remote pusher that runs between our unlock and recv
        // would have its wake byte consumed but its fiber missed by our
        // snapshot — silent lost-wake under level-triggered EPOLLIN.
        self.w("    if (maka_sched_state->wake_pipe_r >= 0) {\n");
        self.w("        char drain[64]; (void)recv(maka_sched_state->wake_pipe_r, drain, sizeof(drain), 0);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&maka_sched_state->remote_mu);\n");
        self.w("    maka_fiber_t* head = maka_sched_state->remote_wake_head;\n");
        self.w("    maka_sched_state->remote_wake_head = NULL;\n");
        // Also pull the cancel + close inboxes under the same lock so the
        // requests and the wake byte stay consistent.
        self.w("    maka_cancel_req_t* creq = maka_sched_state->remote_cancel_head;\n");
        self.w("    maka_sched_state->remote_cancel_head = NULL;\n");
        self.w("    maka_close_req_t* clreq = maka_sched_state->remote_close_head;\n");
        self.w("    maka_sched_state->remote_close_head = NULL;\n");
        self.w("    pthread_mutex_unlock(&maka_sched_state->remote_mu);\n");
        // Splice wakes (already drained the pipe above to plug the unlock→recv
        // window where a remote splice+signal would be lost).  Run queue +
        // fd_waiters are shared across a Pool's workers, so splice + close-req
        // processing runs under the group lock; a batch landed, so broadcast
        // the cv to wake all idle workers (not just one).
        self.w("    MAKA_GRP_LOCK();\n");
        self.w("    maka_fiber_t* prev = NULL;\n");
        self.w("    while (head) { maka_fiber_t* nx = head->next; head->next = prev; prev = head; head = nx; }\n");
        self.w("    while (prev) {\n");
        self.w("        maka_fiber_t* nx = prev->next; prev->next = NULL;\n");
        // Keep in_queue == 1: the fiber is still queued, just moved from the
        // remote inbox to the global run queue.  Clearing it here would break
        // the dedup invariant (in_queue==1 iff queued) and let a concurrent
        // lock-free frq_push double-queue it -> double dispatch.  Dequeue clears
        // it when the fiber is actually taken to run.
        self.w("        if (maka_ready_tail) { maka_ready_tail->next = prev; maka_ready_tail = prev; }\n");
        self.w("        else { maka_ready_head = maka_ready_tail = prev; }\n");
        self.w("        prev->state = 0; prev = nx;\n");
        self.w("    }\n");
        // NOW process close requests (waiters now visible) and cancels.
        self.w("    while (clreq) {\n");
        self.w("        maka_close_req_t* nx2 = clreq->next;\n");
        self.w("        int cfd = clreq->closed_fd;\n");
        self.w("        maka_fiber_t** prev2 = &maka_fd_waiters;\n");
        self.w("        while (*prev2) {\n");
        self.w("            maka_fiber_t* w = *prev2;\n");
        self.w("            if (w->waiting_fd == cfd) {\n");
        self.w("                *prev2 = w->next; w->next = NULL;\n");
        self.w("                w->waiting_fd = -1; w->waiting_events = 0;\n");
        self.w("                w->wait_deadline_ns = 0; w->wait_timed_out = 0;\n");
        self.w("                __maka_ready_enqueue(w);\n");
        self.w("            } else {\n");
        self.w("                prev2 = &(*prev2)->next;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        __maka_fd_recompute(cfd);\n");
        self.w("        free(clreq); clreq = nx2;\n");
        self.w("    }\n");
        self.w("    if (maka_sched_state->group->is_pool) pthread_cond_broadcast(&maka_sched_state->group->cv);\n");
        self.w("    MAKA_GRP_UNLOCK();\n");
        self.w("    while (creq) {\n");
        self.w("        maka_cancel_req_t* nx = creq->next;\n");
        self.w("        Thread* tgt = creq->target;\n");
        self.w("        if (!__maka_cancel_fiber_local(tgt)) {\n");
        self.w("            pthread_mutex_lock(&tgt->done_mutex);\n");
        self.w("            tgt->done_flag = 1;\n");
        self.w("            pthread_cond_broadcast(&tgt->done_cond);\n");
        self.w("            pthread_mutex_unlock(&tgt->done_mutex);\n");
        self.w("        }\n");
        self.w("        __maka_thread_unref(tgt);\n");
        self.w("        free(creq); creq = nx;\n");
        self.w("    }\n");
        self.w("}\n");
        // Reap a lazily-cancelled fiber pulled from a run deque: the fiber never
        // ran (cancel walks the sleep/fd/waiter lists but cannot remove from a
        // lock-free deque, so a ready fiber is flagged and reaped here instead).
        // Mirrors the natural-completion teardown so refcounts stay balanced:
        // env_drop frees any owner moved into it, joiners wake, inflight drops.
        self.w("static void __maka_reap_cancelled(maka_fiber_t* f) {\n");
        self.w("    if (maka_sched_state->group->is_pool && f != maka_anchor_fiber) atomic_fetch_sub(&maka_sched_state->group->inflight, 1);\n");
        self.w("    Thread* t = f->completion;\n");
        self.w("    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("    t->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&t->done_cond);\n");
        self.w("    maka_fiber_t* drain = f->waiters; f->waiters = NULL;\n");
        self.w("    maka_fiber_t* jdrain = t->fiber_waiters; t->fiber_waiters = NULL;\n");
        self.w("    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("    while (drain) { maka_fiber_t* nx = drain->next_waiter; drain->next_waiter = NULL; __maka_ready_enqueue(drain); drain = nx; }\n");
        self.w("    while (jdrain) { maka_fiber_t* nx = jdrain->next_waiter; jdrain->next_waiter = NULL; __maka_ready_enqueue(jdrain); jdrain = nx; }\n");
        self.w("    __maka_free_env(t, f->entry_env);\n");
        self.w("    __maka_slab_free(f->slab);\n");
        self.w("    free(f);\n");
        self.w("    __maka_thread_unref(t);\n");
        self.w("}\n");
        // Dequeue: own lock-free deque (hot) -> group global overflow/cross-thread
        // queue -> steal from a random peer (pool).  Lazily-cancelled fibers are
        // reaped and skipped.
        self.w("static maka_fiber_t* __maka_ready_dequeue(void) {\n");
        self.w("    for (;;) {\n");
        self.w("        maka_fiber_t* f = NULL;\n");
        self.w("        if (maka_my_frunq) f = __maka_frq_take(maka_my_frunq);\n");
        self.w("        if (!f) {\n");
        self.w("            MAKA_GRP_LOCK();\n");
        self.w("            f = maka_ready_head;\n");
        self.w("            if (f) { maka_ready_head = f->next; if (!maka_ready_head) maka_ready_tail = NULL; f->next = NULL; }\n");
        self.w("            MAKA_GRP_UNLOCK();\n");
        self.w("        }\n");
        self.w("        if (!f && maka_sched_state->group->is_pool) {\n");
        self.w("            maka_sched_group_t* g = maka_sched_state->group;\n");
        self.w("            int n = g->n_workers;\n");
        self.w("            if (n > 1 && g->frunqs) {\n");
        self.w("                maka_steal_rng ^= maka_steal_rng << 13; maka_steal_rng ^= maka_steal_rng >> 17; maka_steal_rng ^= maka_steal_rng << 5;\n");
        self.w("                int start = (int)(maka_steal_rng % (unsigned)n);\n");
        self.w("                for (int i = 0; i < n; i++) {\n");
        self.w("                    int v = (start + i) % n;\n");
        self.w("                    if (v == maka_my_worker_id) continue;\n");
        self.w("                    maka_frunq_t* vq = atomic_load_explicit(&g->frunqs[v], memory_order_relaxed);\n");
        self.w("                    if (vq) { f = __maka_frq_take(vq); if (f) break; }\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        if (!f) return NULL;\n");
        self.w("        atomic_store(&f->in_queue, 0);\n");
        self.w("        if (f->completion && atomic_load(&f->completion->cancel_requested) && f != maka_anchor_fiber) {\n");
        self.w("            __maka_reap_cancelled(f);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        return f;\n");
        self.w("    }\n");
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
        self.w("                fprintf(stderr, \"maka: scheduler stuck - fibers haven't yielded for %.2fs (likely a blocking syscall inside a fiber)\\n\", (double)(now - last) / 1e9);\n");
        self.w("                atomic_store(&t->warned, 1);\n");
        self.w("            }\n");
        self.w("            if (now - last <= threshold && atomic_load(&t->warned)) atomic_store(&t->warned, 0);\n");
        self.w("        }\n");
        self.w("        pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("    }\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static void __maka_watchdog_register(void) {\n");
        // ALWAYS link the per-thread tick into __maka_ticks_head.  Cross-
        // thread cancel/select/wake validation walks this list, so without
        // the tick the validation always fails and silently no-ops every
        // cross-thread operation.  The watchdog itself is opt-in via the
        // MAKA_WATCHDOG_MS env var.
        self.w("    if (__maka_my_tick) return;\n");
        self.w("    __maka_my_tick = (maka_sched_tick_t*)calloc(1, sizeof(maka_sched_tick_t));\n");
        self.w("    __maka_my_tick->owner = maka_sched_state;\n");
        self.w("    __maka_my_tick->epoch = maka_sched_state->epoch;\n");
        self.w("    atomic_init(&__maka_my_tick->last_tick_ns, __maka_now_ns());\n");
        self.w("    pthread_mutex_lock(&__maka_ticks_mu);\n");
        self.w("    __maka_my_tick->next = __maka_ticks_head;\n");
        self.w("    __maka_ticks_head = __maka_my_tick;\n");
        self.w("    pthread_mutex_unlock(&__maka_ticks_mu);\n");
        // Only spin up the watchdog pthread when MAKA_WATCHDOG_MS is set.
        self.w("    if (__maka_watchdog_threshold_ns == 0) {\n");
        self.w("        const char* env = getenv(\"MAKA_WATCHDOG_MS\");\n");
        self.w("        int64_t ms = env ? atoll(env) : 0;\n");
        self.w("        if (ms <= 0) { __maka_watchdog_threshold_ns = -1; return; }\n");
        self.w("        __maka_watchdog_threshold_ns = ms * 1000000LL;\n");
        self.w("    }\n");
        self.w("    if (__maka_watchdog_threshold_ns < 0) return;\n");
        self.w("    int expected = 0;\n");
        self.w("    if (atomic_compare_exchange_strong(&__maka_watchdog_started, &expected, 1)) {\n");
        self.w("        pthread_t w;\n");
        self.w("        if (pthread_create(&w, NULL, __maka_watchdog_loop, NULL) == 0) {\n");
        self.w("            pthread_detach(w);\n");
        self.w("        } else {\n");
        // Reset the flag so a later register call can retry instead of
        // leaving the watchdog permanently disabled.
        self.w("            atomic_store(&__maka_watchdog_started, 0);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        // pthread_key destructor implementation — defined here where the
        // ticks list + anchor fiber are in scope.  The earlier forward decl
        // ensures pthread_key_create can register the wrapper.
        self.w("static void __maka_sched_state_key_dtor_impl(void* p) {\n");
        // Unlink + free the per-thread watchdog tick.  Without this, every
        // exited pool worker leaves a dangling entry in __maka_ticks_head
        // that the watchdog keeps dereferencing (UAF + memory leak).
        self.w("    if (__maka_my_tick) {\n");
        self.w("        pthread_mutex_lock(&__maka_ticks_mu);\n");
        self.w("        maka_sched_tick_t** prev = &__maka_ticks_head;\n");
        self.w("        while (*prev) {\n");
        self.w("            if (*prev == __maka_my_tick) { *prev = (*prev)->next; break; }\n");
        self.w("            prev = &(*prev)->next;\n");
        self.w("        }\n");
        self.w("        pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("        free(__maka_my_tick); __maka_my_tick = NULL;\n");
        self.w("    }\n");
        // Free the anchor fiber (allocated in sched_init).
        self.w("    if (maka_anchor_fiber) {\n");
        self.w("        free(maka_anchor_fiber); maka_anchor_fiber = NULL;\n");
        self.w("    }\n");
        // Run reactor cleanup BEFORE nulling TLS — __maka_sched_state_cleanup
        // checks (s == maka_sched_state) to decide whether to touch the
        // __thread reactor registry.  If we null first, cleanup always takes
        // the remote branch and the local reactor reg leaks.
        self.w("    __maka_sched_state_unref((maka_sched_state_t*)p);\n");
        self.w("    maka_sched_state = NULL;\n");
        self.w("}\n");
        // Reactor service: one epoll_wait with `timeout_ms`, then move every
        // ready fd + timed-out fd waiter onto the run queue.  Shared by the
        // cooperative idle path (n_workers==1, the group lock is a no-op) and
        // the Pool poll-owner (n_workers>1, the group lock serializes the
        // shared fd_waiters list).  ONE reactor mechanism, not two.
        self.w("static void __maka_reactor_epoll(int64_t timeout_ms) {\n");
        self.w("    if (maka_epoll_fd < 0) return;\n");
        self.w("    struct epoll_event evs[32];\n");
        self.w("    int n = epoll_wait(maka_epoll_fd, evs, 32, (int)timeout_ms);\n");
        self.w("    MAKA_GRP_LOCK();\n");
        self.w("    for (int i = 0; i < n; i++) {\n");
        self.w("        int evfd = evs[i].data.fd;\n");
        self.w("        int err_hup = evs[i].events & (EPOLLERR | EPOLLHUP);\n");
        self.w("        maka_fiber_t** prev2 = &maka_fd_waiters;\n");
        self.w("        while (*prev2) {\n");
        self.w("            maka_fiber_t* w = *prev2;\n");
        self.w("            if (w->waiting_fd == evfd && (err_hup || (evs[i].events & w->waiting_events))) {\n");
        self.w("                *prev2 = w->next; w->next = NULL;\n");
        self.w("                w->waiting_fd = -1; w->waiting_events = 0;\n");
        self.w("                w->wait_deadline_ns = 0; w->wait_timed_out = 0;\n");
        self.w("                __maka_ready_enqueue(w);\n");
        self.w("            } else {\n");
        self.w("                prev2 = &(*prev2)->next;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        __maka_fd_recompute(evfd);\n");
        self.w("    }\n");
        self.w("    int64_t now2 = __maka_now_ns();\n");
        self.w("    maka_fiber_t** prev3 = &maka_fd_waiters;\n");
        self.w("    int timed_out_fds[32]; int n_timed_out = 0;\n");
        self.w("    while (*prev3) {\n");
        self.w("        maka_fiber_t* f = *prev3;\n");
        self.w("        if (f->wait_deadline_ns != 0 && f->wait_deadline_ns <= now2) {\n");
        self.w("            int tfd = f->waiting_fd;\n");
        self.w("            *prev3 = f->next; f->next = NULL;\n");
        self.w("            f->waiting_fd = -1; f->waiting_events = 0;\n");
        self.w("            f->wait_deadline_ns = 0;\n");
        self.w("            f->wait_timed_out = 1;\n");
        self.w("            __maka_ready_enqueue(f);\n");
        self.w("            if (n_timed_out < 32 && tfd >= 0) timed_out_fds[n_timed_out++] = tfd;\n");
        self.w("        } else {\n");
        self.w("            prev3 = &(*prev3)->next;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    for (int k = 0; k < n_timed_out; k++) __maka_fd_recompute(timed_out_fds[k]);\n");
        self.w("    MAKA_GRP_UNLOCK();\n");
        self.w("}\n");
        self.w("static void __maka_scheduler_loop(void) {\n");
        self.w("    while (1) {\n");
        self.w("        int64_t now = __maka_now_ns();\n");
        // Drain any wakes pushed by other threads since the last iteration.
        self.w("        __maka_drain_remote_wakes();\n");
        self.w("        if (__maka_my_tick) atomic_store(&__maka_my_tick->last_tick_ns, now);\n");
        self.w("        /* Move expired sleepers to ready (shared list -> group lock) */\n");
        self.w("        MAKA_GRP_LOCK();\n");
        // has_work reads the shared run/sleep/fd lists, so compute it under the
        // group lock (serialized with dequeue's write); a no-op lock for a lone
        // worker keeps it race-free there too.
        self.w("        if (__maka_my_tick) atomic_store(&__maka_my_tick->has_work, (__maka_have_ready() || maka_sleep_head || maka_fd_waiters) ? 1 : 0);\n");
        self.w("        maka_fiber_t** prev = &maka_sleep_head;\n");
        self.w("        int __woke = 0;\n");
        self.w("        while (*prev) {\n");
        self.w("            maka_fiber_t* sf = *prev;\n");
        self.w("            if (sf->wake_at_ns <= now) {\n");
        self.w("                *prev = sf->next; sf->next = NULL;\n");
        // Wake to the group's GLOBAL run queue under the lock we already hold -
        // NOT the lock-free per-worker deque.  The timer expiry runs on every
        // worker, so keeping the sleep->ready transition fully under the group
        // lock keeps it serialized with dequeue + the fd/close paths, instead of
        // racing a peer's lock-free steal of a just-woken (and about-to-be-freed)
        // sleeper.  Dedup on in_queue as usual.
        self.w("                if (atomic_exchange(&sf->in_queue, 1) == 0) {\n");
        self.w("                    sf->state = 0;\n");
        self.w("                    if (maka_ready_tail) { maka_ready_tail->next = sf; maka_ready_tail = sf; }\n");
        self.w("                    else { maka_ready_head = maka_ready_tail = sf; }\n");
        self.w("                    __woke = 1;\n");
        self.w("                }\n");
        self.w("            } else { prev = &sf->next; }\n");
        self.w("        }\n");
        // Wake idle/poll-owner peers so they pick up the newly-ready sleepers.
        self.w("        if (__woke && maka_sched_state->group->is_pool) {\n");
        self.w("            pthread_cond_broadcast(&maka_sched_state->group->cv);\n");
        self.w("            if (maka_sched_state->group->polling && maka_sched_state->wake_pipe_w > 0) { char b = 1; (void)send(maka_sched_state->wake_pipe_w, &b, 1, 0); }\n");
        self.w("        }\n");
        self.w("        MAKA_GRP_UNLOCK();\n");
        self.w("        /* Is the awaited target done? */\n");
        self.w("        if (maka_join_target) {\n");
        self.w("            Thread* tgt = maka_join_target;\n");
        self.w("            int done = 0;\n");
        self.w("            pthread_mutex_lock(&tgt->done_mutex);\n");
        self.w("            done = tgt->done_flag;\n");
        self.w("            pthread_mutex_unlock(&tgt->done_mutex);\n");
        self.w("            if (done) {\n");
        self.w("                maka_fiber_t* anchor = maka_anchor_fiber;\n");
        self.w("                maka_join_target = NULL;\n");
        self.w("                maka_current_fiber = anchor;\n");
        self.w("                maka_ctx_switch(&maka_sched_ctx, &anchor->ctx);\n");
        self.w("                continue;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        /* Pick a ready fiber */\n");
        self.w("        maka_fiber_t* f = __maka_ready_dequeue();\n");
        self.w("        if (f) {\n");
        self.w("            maka_current_fiber = f;\n");
        self.w("            f->state = 1;\n");
        self.w("            maka_ctx_switch(&maka_sched_ctx, &f->ctx);\n");
        // Finalize park: parkers set maka_pending_park before swapcontext-out;
        // here, after the swap save has fully completed, we do the actual
        // push onto the target list under the appropriate lock.  Eliminates
        // the parker-unlock vs. swap-save race.
        // Did the fiber PARK or YIELD (vs complete)?  If so, after the finalize
        // re-queues/parks it, `f` belongs to the group again - a peer can steal,
        // run, and FREE it - so this worker must NOT touch `f` afterward.  Only a
        // fiber that did NOT park has completed and is still exclusively ours.
        self.w("            int __parked = (maka_pending_park != NULL);\n");
        self.w("            if (maka_pending_park) {\n");
        self.w("                maka_park_req_t* p = maka_pending_park;\n");
        self.w("                maka_pending_park = NULL;\n");
        self.w("                if (p->kind == 1) {\n");
        // Deferred sleep insertion — f is fully switched out, so it is now
        // safe for a peer worker's timer scan to expire + run it.
        self.w("                    MAKA_GRP_LOCK();\n");
        self.w("                    f->next = maka_sleep_head; maka_sleep_head = f;\n");
        self.w("                    if (maka_sched_state->group->is_pool && maka_sched_state->group->polling && maka_sched_state->wake_pipe_w > 0) { char b = 1; (void)send(maka_sched_state->wake_pipe_w, &b, 1, 0); }\n");
        self.w("                    MAKA_GRP_UNLOCK();\n");
        self.w("                } else if (p->kind == 2) {\n");
        // Deferred fd insertion + epoll arm for the union interest mask.
        self.w("                    MAKA_GRP_LOCK();\n");
        self.w("                    f->next = maka_fd_waiters; maka_fd_waiters = f;\n");
        self.w("                    __maka_fd_recompute(f->waiting_fd);\n");
        self.w("                    if (maka_sched_state->group->is_pool && maka_sched_state->group->polling && maka_sched_state->wake_pipe_w > 0) { char b = 1; (void)send(maka_sched_state->wake_pipe_w, &b, 1, 0); }\n");
        self.w("                    MAKA_GRP_UNLOCK();\n");
        self.w("                } else if (p->kind == 3) {\n");
        // Deferred yield: re-enqueue as ready now that the context is saved.
        self.w("                    __maka_ready_enqueue(f);\n");
        self.w("                } else {\n");
        self.w("                int __mp_enqueue_ready = 0;\n");
        self.w("                pthread_mutex_lock(p->lock);\n");
        self.w("                if (p->should_park(p->arg)) {\n");
        self.w("                    f->next_waiter = *p->head;\n");
        self.w("                    *p->head = f;\n");
        self.w("                } else {\n");
        self.w("                    f->state = 0;\n");
        self.w("                    __mp_enqueue_ready = 1;\n");
        self.w("                }\n");
        // Decrement the in-flight counter under the lock so destroy's
        // drained_cv wait sees a consistent (inflight, fiber_waiters) view.
        self.w("                if (p->inflight) {\n");
        self.w("                    if (atomic_fetch_sub(p->inflight, 1) == 1 && p->drained_cv) {\n");
        self.w("                        pthread_cond_signal(p->drained_cv);\n");
        self.w("                    }\n");
        self.w("                }\n");
        self.w("                pthread_mutex_unlock(p->lock);\n");
        self.w("                if (__mp_enqueue_ready) __maka_ready_enqueue(f);\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("            maka_current_fiber = NULL;\n");
        // Only a fiber that did NOT park/yield has completed and is still ours to
        // touch (a parked/yielded fiber may already be running on a peer).
        self.w("            if (!__parked && f->state == 4) {\n");
        // Pool bookkeeping: a submitted/spawned fiber in this group finished,
        // so pool_shutdown's drain condition (inflight==0) advances.  The
        // per-worker anchor is not a submitted fiber, so it is excluded.
        self.w("                if (maka_sched_state->group->is_pool && f != maka_anchor_fiber) atomic_fetch_sub(&maka_sched_state->group->inflight, 1);\n");
        self.w("                /* Fiber finished: mark completion + wake waiters. */\n");
        // Hold done_mutex around the waiters drain so the join parker's
        // finalize push (which takes done_mutex + checks done_flag) is
        // serialized with this drain.  Snapshot list under lock, drain after.
        self.w("                Thread* fcompl = f->completion;\n");
        self.w("                pthread_mutex_lock(&fcompl->done_mutex);\n");
        self.w("                fcompl->done_flag = 1;\n");
        self.w("                pthread_cond_broadcast(&fcompl->done_cond);\n");
        self.w("                maka_fiber_t* drain = f->waiters; f->waiters = NULL;\n");
        // Fiber joiners parked on the Thread (yielded their worker via join)
        // are drained under the same lock so the parker's finalize (which also
        // takes done_mutex + re-checks done_flag) can't lose the wake.
        self.w("                maka_fiber_t* jdrain = fcompl->fiber_waiters; fcompl->fiber_waiters = NULL;\n");
        self.w("                pthread_mutex_unlock(&fcompl->done_mutex);\n");
        self.w("                while (drain) {\n");
        self.w("                    maka_fiber_t* nx = drain->next_waiter; drain->next_waiter = NULL;\n");
        self.w("                    __maka_ready_enqueue(drain); drain = nx;\n");
        self.w("                }\n");
        self.w("                while (jdrain) {\n");
        self.w("                    maka_fiber_t* nx = jdrain->next_waiter; jdrain->next_waiter = NULL;\n");
        self.w("                    __maka_ready_enqueue(jdrain); jdrain = nx;\n");
        self.w("                }\n");
        // Defense-in-depth: in normal control flow file_*_async clears
        // aio_in_flight before returning, so this loop is essentially dead
        // code on the natural completion path.  But if the body somehow
        // returns while a detached AIO worker still holds buf/fd refs, busy-
        // yield until the worker signals (microseconds) before freeing the
        // slab.  Mirrors __maka_cancel_fiber_local at line 2668.
        self.w("                if (f != maka_anchor_fiber) {\n");
        self.w("                    while (atomic_load(&fcompl->aio_in_flight)) { struct timespec __mk_aio_ts = { 0, 1000 }; nanosleep(&__mk_aio_ts, NULL); }\n");
        self.w("                    __maka_free_env(f->completion, f->entry_env); __maka_slab_free(f->slab); free(f);\n");
        self.w("                }\n");
        self.w("                /* Drop the runner-side ref; last drop frees.\n");
        self.w("                   Both joiner + detacher race against this on\n");
        self.w("                   __maka_thread_unref - refcount makes it safe. */\n");
        self.w("                __maka_thread_unref(fcompl);\n");
        self.w("                /* If anchor is parked in a select loop, return so it can re-poll. */\n");
        self.w("                if (maka_anchor_wake_on_finish) {\n");
        self.w("                    maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                    maka_ctx_switch(&maka_sched_ctx, &maka_anchor_fiber->ctx);\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        /* No ready fibers - wait for sleepers, fd events, or fall out. */\n");
        // Pool-mode idle (n_workers>1): the group shares one reactor, so exactly
        // one worker (the poll-owner) drives epoll_wait while the rest cv-wait.
        // A worker exits only when the pool is closed AND fully drained.
        self.w("        if (maka_sched_state->group->is_pool) {\n");
        self.w("            maka_sched_group_t* g = maka_sched_state->group;\n");
        self.w("            MAKA_GRP_LOCK();\n");
        // Another worker may have enqueued between the dequeue and here.
        self.w("            if (__maka_have_ready()) { MAKA_GRP_UNLOCK(); continue; }\n");
        // Drained + closed + nothing parked anywhere -> this worker is done.
        // Wake the peers so they re-check the exit condition too.
        self.w("            if (g->closed && atomic_load(&g->inflight) == 0 && !maka_sleep_head && !maka_fd_waiters) {\n");
        self.w("                pthread_cond_broadcast(&g->cv);\n");
        self.w("                MAKA_GRP_UNLOCK();\n");
        // Switch back to this worker's anchor (the ctx that entered the loop),
        // NOT `return` — the scheduler context was ctx_make'd and has no valid
        // return frame.  The anchor resumes in __maka_pool_h_worker and exits.
        self.w("                maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                maka_ctx_switch(&maka_sched_ctx, &maka_anchor_fiber->ctx);\n");
        self.w("                continue;\n");
        self.w("            }\n");
        self.w("            if (!g->polling) {\n");
        // Become the poll-owner.  Compute the nearest timer/fd deadline (or
        // block indefinitely on the wake_pipe if nothing is parked but work
        // may still arrive cross-thread), then drive the shared reactor.
        self.w("                int64_t timeout_ms = -1;\n");
        self.w("                int64_t now_ns = __maka_now_ns();\n");
        self.w("                for (maka_fiber_t* s = maka_sleep_head; s; s = s->next) {\n");
        self.w("                    int64_t d = s->wake_at_ns - now_ns;\n");
        self.w("                    int64_t dms = d <= 0 ? 0 : (d / 1000000LL < 1 ? 1 : d / 1000000LL);\n");
        self.w("                    if (timeout_ms < 0 || dms < timeout_ms) timeout_ms = dms;\n");
        self.w("                }\n");
        self.w("                for (maka_fiber_t* w = maka_fd_waiters; w; w = w->next) {\n");
        self.w("                    if (w->wait_deadline_ns == 0) continue;\n");
        self.w("                    int64_t d = w->wait_deadline_ns - now_ns;\n");
        self.w("                    int64_t dms = d <= 0 ? 0 : (d / 1000000LL < 1 ? 1 : d / 1000000LL);\n");
        self.w("                    if (timeout_ms < 0 || dms < timeout_ms) timeout_ms = dms;\n");
        self.w("                }\n");
        // Bound the wait so `closed` (shutdown) and any wake that raced with
        // this thread committing to epoll_wait are always re-checked promptly.
        // New work stays immediate via the unconditional wake_pipe ping; this
        // cap only backstops the level-triggered wake_pipe TOCTOU (a peer's
        // drain can consume the shutdown byte before this epoll_wait starts).
        self.w("                if (timeout_ms < 0 || timeout_ms > 250) timeout_ms = 250;\n");
        self.w("                g->polling = 1;\n");
        self.w("                MAKA_GRP_UNLOCK();\n");
        self.w("                __maka_reactor_epoll(timeout_ms);\n");
        self.w("                MAKA_GRP_LOCK();\n");
        self.w("                g->polling = 0;\n");
        self.w("                pthread_cond_broadcast(&g->cv);\n");
        self.w("                MAKA_GRP_UNLOCK();\n");
        self.w("                continue;\n");
        self.w("            } else {\n");
        // Another worker drives the reactor.  Wait for a signal (enqueue,
        // poll-owner finish, or shutdown).  BOUNDED wait: the enqueue cv-signal
        // that wakes a stealer is not synchronized with this wait (the deque
        // push is lock-free), so a signal can race this thread committing to the
        // wait and be lost.  A short timeout makes a missed steal-wake self-heal
        // (the worker re-checks + steals within the bound) - the same backstop
        // the poll-owner's bounded epoll_wait uses.  cond_timedwait at lock
        // depth 1 fully releases the recursive mutex.
        self.w("                struct timespec __ts; clock_gettime(CLOCK_REALTIME, &__ts);\n");
        self.w("                __ts.tv_nsec += 2000000L; if (__ts.tv_nsec >= 1000000000L) { __ts.tv_sec++; __ts.tv_nsec -= 1000000000L; }\n");
        self.w("                pthread_cond_timedwait(&g->cv, &g->lock, &__ts);\n");
        self.w("                MAKA_GRP_UNLOCK();\n");
        self.w("                continue;\n");
        self.w("            }\n");
        self.w("        }\n");
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
        self.w("            int anchor_deadline_expired = 0;\n");
        self.w("            if (maka_anchor_deadline_ns != 0) {\n");
        self.w("                int64_t delta = maka_anchor_deadline_ns - now_ns;\n");
        self.w("                if (delta <= 0) { anchor_deadline_expired = 1; }\n");
        self.w("                else {\n");
        self.w("                    int64_t dms = delta / 1000000LL;\n");
        self.w("                    if (dms < 1) dms = 1;\n");
        self.w("                    if (timeout_ms < 0 || dms < timeout_ms) timeout_ms = dms;\n");
        self.w("                }\n");
        self.w("            }\n");
        // If the anchor's deadline has expired AND no other deadline shorter
        // than that fires, hand back to anchor IMMEDIATELY instead of calling
        // epoll_wait(0).  Windows WSAPoll with timeout=0 returns instantly
        // without sleeping, which would spin until anchor checks its flag.
        self.w("            if (anchor_deadline_expired && maka_anchor_fiber) {\n");
        self.w("                maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                maka_ctx_switch(&maka_sched_ctx, &maka_anchor_fiber->ctx);\n");
        self.w("                continue;\n");
        self.w("            }\n");
        self.w("            if (have_fd_waiters && maka_epoll_fd >= 0) {\n");
        self.w("                __maka_reactor_epoll(timeout_ms);\n");
        self.w("            } else if (have_sleepers) {\n");
        self.w("                struct timespec ts = { timeout_ms / 1000, (timeout_ms % 1000) * 1000000 };\n");
        self.w("                nanosleep(&ts, NULL);\n");
        self.w("            }\n");
        self.w("            /* Anchor deadline reached?  Hand control back so the\n");
        self.w("               caller's timeout primitive can finish. */\n");
        self.w("            if (maka_anchor_deadline_ns != 0 && __maka_now_ns() >= maka_anchor_deadline_ns && maka_anchor_fiber) {\n");
        self.w("                maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                maka_ctx_switch(&maka_sched_ctx, &maka_anchor_fiber->ctx);\n");
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
        self.w("            maka_ctx_switch(&maka_sched_ctx, &maka_anchor_fiber->ctx);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("}\n");
        // pthread_key-based cleanup for per-thread sched_state.  Workers that
        // exit (e.g. spawn_pool background pthreads) tear down their state
        // via this destructor — without it, every worker exit leaks 2 fds
        // (the wake_pipe pair) plus the sched_state allocation.
        self.w("static pthread_key_t __maka_sched_state_key;\n");
        self.w("static pthread_once_t __maka_sched_state_once = PTHREAD_ONCE_INIT;\n");
        // sched_state ref/unref — last drop tears down the structure.
        self.w("static void __maka_sched_state_cleanup(void* p);\n");
        self.w("static inline void __maka_sched_state_ref(maka_sched_state_t* s) {\n");
        self.w("    if (s) atomic_fetch_add(&s->refcount, 1);\n");
        self.w("}\n");
        self.w("static inline void __maka_sched_state_unref(maka_sched_state_t* s) {\n");
        self.w("    if (!s) return;\n");
        self.w("    if (atomic_fetch_sub(&s->refcount, 1) == 1) __maka_sched_state_cleanup((void*)s);\n");
        self.w("}\n");
        // Validate that `candidate` is still owned by a live thread (its tick
        // is still in the global registry) and atomically bump its ref.  The
        // ticks list is locked by the dtor when unlinking, so this check is
        // race-free against cleanup.
        self.w("static inline maka_sched_state_t* __maka_sched_validate_and_ref_epoch(maka_sched_state_t* candidate, int64_t expected_epoch) {\n");
        self.w("    if (!candidate) return NULL;\n");
        self.w("    pthread_mutex_lock(&__maka_ticks_mu);\n");
        self.w("    for (maka_sched_tick_t* tk = __maka_ticks_head; tk; tk = tk->next) {\n");
        self.w("        if (tk->owner == candidate && tk->epoch == expected_epoch) {\n");
        self.w("            atomic_fetch_add(&candidate->refcount, 1);\n");
        self.w("            pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("            return candidate;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        // Backwards-compat wrapper for sites that don't yet pass an epoch.
        self.w("static inline maka_sched_state_t* __maka_sched_validate_and_ref(maka_sched_state_t* candidate) {\n");
        self.w("    if (!candidate) return NULL;\n");
        self.w("    pthread_mutex_lock(&__maka_ticks_mu);\n");
        self.w("    for (maka_sched_tick_t* tk = __maka_ticks_head; tk; tk = tk->next) {\n");
        self.w("        if (tk->owner == candidate) {\n");
        self.w("            atomic_fetch_add(&candidate->refcount, 1);\n");
        self.w("            pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("            return candidate;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static void __maka_sched_state_cleanup(void* p) {\n");
        self.w("    maka_sched_state_t* s = (maka_sched_state_t*)p;\n");
        self.w("    if (!s) return;\n");
        // Drain pending remote_cancel + remote_wake + remote_close requests
        // so target Threads don't leak runner refs, queued fibers don't stay
        // stuck with in_queue == 1, and close-requests don't get silently
        // dropped (fd_waiters that the close was supposed to wake would
        // otherwise miss it on shutdown).
        self.w("    pthread_mutex_lock(&s->remote_mu);\n");
        self.w("    maka_cancel_req_t* creq = s->remote_cancel_head;\n");
        self.w("    s->remote_cancel_head = NULL;\n");
        self.w("    maka_fiber_t* wake = s->remote_wake_head;\n");
        self.w("    s->remote_wake_head = NULL;\n");
        self.w("    maka_close_req_t* clreq = s->remote_close_head;\n");
        self.w("    s->remote_close_head = NULL;\n");
        self.w("    pthread_mutex_unlock(&s->remote_mu);\n");
        // Pending close requests: just free them — the home thread is gone
        // so there are no local fd_waiters to wake on this thread.
        self.w("    while (clreq) { maka_close_req_t* nx = clreq->next; free(clreq); clreq = nx; }\n");
        // Clear in_queue on queued fibers so they're not permanently stuck,
        // then free them (slab + entry_env + struct) and drop the runner-side
        // ref so the Thread reaches refcount 0.  Without freeing the fiber
        // struct we'd leak ~256 KB per pending cross-thread wake at shutdown.
        self.w("    while (wake) {\n");
        self.w("        maka_fiber_t* nx = wake->next; wake->next = NULL;\n");
        self.w("        atomic_store(&wake->in_queue, 0);\n");
        self.w("        Thread* tcompl = wake->completion;\n");
        self.w("        if (tcompl) {\n");
        self.w("            pthread_mutex_lock(&tcompl->done_mutex);\n");
        self.w("            tcompl->done_flag = 1;\n");
        self.w("            pthread_cond_broadcast(&tcompl->done_cond);\n");
        self.w("            pthread_mutex_unlock(&tcompl->done_mutex);\n");
        self.w("        }\n");
        self.w("        if (wake->entry_env) __maka_free_env(wake->completion, wake->entry_env);\n");
        self.w("        if (wake->slab) __maka_slab_free(wake->slab);\n");
        self.w("        free(wake);\n");
        self.w("        if (tcompl) __maka_thread_unref(tcompl);\n");
        self.w("        wake = nx;\n");
        self.w("    }\n");
        self.w("    while (creq) {\n");
        self.w("        maka_cancel_req_t* nx = creq->next;\n");
        self.w("        Thread* tgt = creq->target;\n");
        self.w("        pthread_mutex_lock(&tgt->done_mutex);\n");
        self.w("        tgt->done_flag = 1;\n");
        self.w("        pthread_cond_broadcast(&tgt->done_cond);\n");
        self.w("        pthread_mutex_unlock(&tgt->done_mutex);\n");
        // Drop BOTH refs: the one held by the cancel request (taken in
        // remote_post_cancel) AND the runner-side ref (the fiber will never
        // reach its own completion path on this dead scheduler).
        self.w("        __maka_thread_unref(tgt);  /* request ref */\n");
        self.w("        __maka_thread_unref(tgt);  /* runner ref */\n");
        self.w("        free(creq); creq = nx;\n");
        self.w("    }\n");
        // Drop reactor reg for the wake_pipe before closing — but ONLY if
        // we're the owning thread (i.e. this call came from the pthread_key
        // dtor, not a remote unref).  __maka_fd_arm / __maka_fd_reg_drop
        // touch __thread reactor state that belongs to whoever's running.
        // If we're a remote unref dropping the last ref, just leak the
        // reactor reg — the owning thread's dtor will have done its part.
        self.w("    if (s == maka_sched_state) {\n");
        self.w("        if (s->wake_pipe_r >= 0) { __maka_fd_arm(s->wake_pipe_r, 0); __maka_fd_reg_drop(s->wake_pipe_r); close(s->wake_pipe_r); }\n");
        self.w("        if (s->wake_pipe_w >= 0) close(s->wake_pipe_w);\n");
        self.w("    } else {\n");
        // Best-effort: still close the fds so they don't leak.  The reactor
        // table will be cleaned up by the owning thread's dtor (which set
        // s == NULL on its TLS, so we know we're not it).
        self.w("        if (s->wake_pipe_r >= 0) close(s->wake_pipe_r);\n");
        self.w("        if (s->wake_pipe_w >= 0) close(s->wake_pipe_w);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_destroy(&s->remote_mu);\n");
        // Close the shared reactor + free the fd->mask registry.  Only close a
        // real epoll fd (the poll fallback uses a fake epoll_fd of 0 = stdin;
        // kqueue keeps its fd in per-thread TLS, cleaned on that thread).
        self.w("#if MAKA_USE_EPOLL || MAKA_USE_KQUEUE\n");
        self.w("    if (s->epoll_fd >= 0) close(s->epoll_fd);\n");
        self.w("#endif\n");
        self.w("    { maka_fd_reg_t* r = s->fd_regs; while (r) { maka_fd_reg_t* nx = r->next; free(r); r = nx; } }\n");
        // Free the group (lone or shared Pool) — the sched_state owns it and the
        // last refcount drop lands here.
        // Free the per-worker deques (empty by now: a Pool drains to inflight==0
        // before its workers exit) + the deque array, then the group itself.
        self.w("    if (s->group) {\n");
        self.w("        if (s->group->frunqs) { for (int i = 0; i < s->group->n_workers; i++) free(atomic_load_explicit(&s->group->frunqs[i], memory_order_relaxed)); free(s->group->frunqs); }\n");
        self.w("        pthread_mutex_destroy(&s->group->lock); pthread_cond_destroy(&s->group->cv); free(s->group);\n");
        self.w("    }\n");
        self.w("    free(s);\n");
        self.w("}\n");
        self.w("static void __maka_sched_state_key_dtor_impl(void* p);\n");
        self.w("static void __maka_sched_state_key_dtor(void* p) { __maka_sched_state_key_dtor_impl(p); }\n");
        self.w("static void __maka_sched_key_init(void) {\n");
        self.w("    pthread_key_create(&__maka_sched_state_key, __maka_sched_state_key_dtor);\n");
        self.w("}\n");
        self.w("static void __maka_sched_register_thread(maka_sched_state_t* s) {\n");
        self.w("    pthread_once(&__maka_sched_state_once, __maka_sched_key_init);\n");
        self.w("    pthread_setspecific(__maka_sched_state_key, s);\n");
        self.w("}\n");
        // Recursive so nested group critical sections (a locked scheduler
        // phase that calls __maka_ready_enqueue) don't self-deadlock.
        self.w("static void __maka_grp_mtx_init(pthread_mutex_t* m) {\n");
        self.w("    pthread_mutexattr_t a; pthread_mutexattr_init(&a);\n");
        self.w("    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);\n");
        self.w("    pthread_mutex_init(m, &a);\n");
        self.w("    pthread_mutexattr_destroy(&a);\n");
        self.w("}\n");
        self.w("static void __maka_sched_init(void) {\n");
        self.w("    if (maka_sched_inited) return;\n");
        self.w("    maka_sched_inited = 1;\n");
        // watchdog_register now also links the tick (which requires
        // maka_sched_state to be set), so move the call below the alloc.
        // Per-thread scheduler state + self-socketpair for cross-thread wakes.
        // socketpair(2) lives in <sys/socket.h>; Linux/Windows don't pull
        // that header in the prologue, so forward-declare locally.  AF_UNIX
        // and SOCK_STREAM are 1 on every modern platform.
        self.w("#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__) && !defined(__DragonFly__)\n");
        self.w("    extern int socketpair(int domain, int type, int protocol, int sv[2]);\n");
        self.w("#ifndef AF_UNIX\n");
        self.w("#define AF_UNIX 1\n");
        self.w("#endif\n");
        self.w("#ifndef SOCK_STREAM\n");
        self.w("#define SOCK_STREAM 1\n");
        self.w("#endif\n");
        self.w("#endif\n");
        self.w("    maka_sched_state = (maka_sched_state_t*)calloc(1, sizeof(maka_sched_state_t));\n");
        // epoll_fd defaults to 0 under calloc, but 0 is a valid fd; the lazy
        // reactor create keys off (epoll_fd < 0), so it must start at -1.
        self.w("    maka_sched_state->epoll_fd = -1;\n");
        self.w("    pthread_mutex_init(&maka_sched_state->remote_mu, NULL);\n");
        // Lone-worker group (size 1): its own ready queue, no locking.  A Pool
        // (slice 2) will replace this with a shared group of N workers.
        self.w("    maka_sched_state->group = (maka_sched_group_t*)calloc(1, sizeof(maka_sched_group_t));\n");
        self.w("    __maka_grp_mtx_init(&maka_sched_state->group->lock);\n");
        self.w("    pthread_cond_init(&maka_sched_state->group->cv, NULL);\n");
        self.w("    maka_sched_state->group->n_workers = 1;\n");
        // Lone worker: one deque, this thread owns slot 0.
        self.w("    maka_sched_state->group->frunqs = (_Atomic(maka_frunq_t*)*)calloc(1, sizeof(_Atomic(maka_frunq_t*)));\n");
        self.w("    maka_my_frunq = (maka_frunq_t*)calloc(1, sizeof(maka_frunq_t));\n");
        self.w("    atomic_store_explicit(&maka_sched_state->group->frunqs[0], maka_my_frunq, memory_order_relaxed);\n");
        self.w("    maka_my_worker_id = 0;\n");
        self.w("    atomic_store(&maka_sched_state->group->worker_seq, 1);\n");
        // Owner ref — released by sched_state_cleanup on thread exit.
        self.w("    atomic_init(&maka_sched_state->refcount, 1);\n");
        self.w("    maka_sched_state->epoch = atomic_fetch_add(&__maka_sched_epoch_ctr, 1);\n");
        // NOTE: watchdog_register (which links the tick into __maka_ticks_head
        // for cross-thread validation) is deferred until AFTER the wake_pipe
        // sockets are armed — otherwise a remote validator would see the
        // sched with wake_pipe_w == 0 (calloc default) and write a stray
        // wake byte to fd 0.
        // Use a TCP-loopback socket pair (already abstracted by the Windows
        // pipe() shim) instead of POSIX pipe() — recv()/send() on a real
        // POSIX pipe fd returns ENOTSOCK, so the wake byte would never get
        // through on Linux/macOS.  socketpair(AF_UNIX, SOCK_STREAM) is the
        // POSIX portable equivalent; on Windows the pipe() macro is our
        // TCP-pair shim, so go through that there.
        self.w("    int wp[2]; int ok;\n");
        self.w("#ifdef _WIN32\n");
        self.w("    ok = (pipe(wp) == 0);\n");
        self.w("    if (ok) { u_long nb = 1; ioctlsocket((SOCKET)wp[0], FIONBIO, &nb); ioctlsocket((SOCKET)wp[1], FIONBIO, &nb); }\n");
        self.w("#else\n");
        self.w("    ok = (socketpair(AF_UNIX, SOCK_STREAM, 0, wp) == 0);\n");
        self.w("    if (ok) {\n");
        self.w("        int f0 = fcntl(wp[0], F_GETFL, 0); fcntl(wp[0], F_SETFL, f0 | O_NONBLOCK);\n");
        self.w("        int f1 = fcntl(wp[1], F_GETFL, 0); fcntl(wp[1], F_SETFL, f1 | O_NONBLOCK);\n");
        self.w("    }\n");
        self.w("#endif\n");
        self.w("    if (ok) {\n");
        self.w("        maka_sched_state->wake_pipe_r = wp[0];\n");
        self.w("        maka_sched_state->wake_pipe_w = wp[1];\n");
        // Arm the read end in the reactor so a remote-wake send unblocks
        // epoll_wait/kevent/WSAPoll.  No fiber waits on it; the drain at the
        // top of the next scheduler iteration handles the readable byte.
        self.w("        __maka_fd_arm(wp[0], MAKA_EV_READ);\n");
        self.w("    } else { maka_sched_state->wake_pipe_r = maka_sched_state->wake_pipe_w = -1; }\n");
        // Now link the tick — wake_pipe is initialized, so any cross-thread
        // wake byte sent through validate-and-ref reaches a real socket.
        self.w("    __maka_watchdog_register();\n");
        // Register a pthread_key destructor so worker pthreads tear down the
        // sched_state on exit (closes both wake_pipe sockets + frees state).
        // Without this, every spawn_pool worker that exits leaks 2 fds + the
        // sched struct.  Key lives in a separate static helper (see below).
        self.w("    __maka_sched_register_thread(maka_sched_state);\n");
        self.w("    /* anchor represents the calling (main or pthread) context */\n");
        self.w("    maka_anchor_fiber = (maka_fiber_t*)calloc(1, sizeof(maka_fiber_t));\n");
        self.w("    maka_anchor_fiber->state = 1;\n");
        // Epoch BEFORE pointer release so a remote acquire-load sees a
        // consistent (pointer, epoch) pair.
        self.w("    atomic_store_explicit(&maka_anchor_fiber->home_sched_epoch, maka_sched_state->epoch, memory_order_relaxed);\n");
        self.w("    atomic_store_explicit(&maka_anchor_fiber->home_sched, (void*)maka_sched_state, memory_order_release);\n");
        self.w("    maka_current_fiber = maka_anchor_fiber;\n");
        self.w("    maka_ctx_make(&maka_sched_ctx, maka_sched_stack, sizeof(maka_sched_stack), __maka_scheduler_loop);\n");
        self.w("}\n");
        self.w("static void __maka_fiber_entry(void) {\n");
        self.w("    maka_fiber_t* f = maka_current_fiber;\n");
        self.w("    f->entry_code(f->entry_env);\n");
        self.w("    f->state = 4;\n");
        self.w("    maka_ctx_switch(&f->ctx, &maka_sched_ctx);\n");
        self.w("}\n");
        // spawn(): create a userspace cooperative fiber.
        self.w("maka_unit* __maka_spawn_fiber(void* code, void* env, void* env_drop) {\n");
        self.w("    __maka_sched_init();\n");
        self.w("    Thread* t = __maka_thread_new();\n");
        self.w("    t->is_fiber = 1;\n");
        self.w("    t->env_drop = (void(*)(void*))env_drop;\n");
        self.w("    maka_fiber_t* f = (maka_fiber_t*)calloc(1, sizeof(maka_fiber_t));\n");
        self.w("    if (!f) {\n");
        self.w("        atomic_store(&t->is_failed, 1);\n");
        self.w("        pthread_mutex_lock(&t->done_mutex); t->done_flag = 1; pthread_cond_broadcast(&t->done_cond); pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("        __maka_thread_unref(t);  /* runner ref */\n");
        self.w("        return (maka_unit*)t;\n");
        self.w("    }\n");
        self.w("    f->slab = __maka_slab_alloc(MAKA_FIBER_STACK_SIZE);\n");
        // Slab allocation can fail under heavy OOM / vm.max_map_count limits.
        // Mark the Thread failed + done so a joiner picks up cleanly without
        // pthread_join() / readying a fiber with no usable stack.
        self.w("    if (!f->slab) {\n");
        self.w("        free(f);\n");
        self.w("        atomic_store(&t->is_failed, 1);\n");
        self.w("        pthread_mutex_lock(&t->done_mutex); t->done_flag = 1; pthread_cond_broadcast(&t->done_cond); pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("        __maka_thread_unref(t);  /* runner ref */\n");
        self.w("        return (maka_unit*)t;\n");
        self.w("    }\n");
        self.w("    f->entry_code = (void(*)(void*))code;\n");
        self.w("    f->entry_env = env;\n");
        self.w("    f->completion = t;\n");
        self.w("    f->state = 0;\n");
        self.w("    f->waiting_fd = -1;\n");
        self.w("    f->waiting_events = 0;\n");
        self.w("    f->wait_deadline_ns = 0;\n");
        self.w("    f->wait_timed_out = 0;\n");
        // Epoch BEFORE pointer release.
        self.w("    atomic_store_explicit(&f->home_sched_epoch, maka_sched_state->epoch, memory_order_relaxed);\n");
        self.w("    atomic_store_explicit(&f->home_sched, (void*)maka_sched_state, memory_order_release);\n");
        self.w("    atomic_store_explicit(&t->home_sched_epoch, maka_sched_state->epoch, memory_order_relaxed);\n");
        self.w("    atomic_store_explicit((_Atomic(void*)*)&t->home_sched, (void*)maka_sched_state, memory_order_release);\n");
        self.w("    maka_ctx_make(&f->ctx, f->slab->stack_top, f->slab->stack_size, __maka_fiber_entry);\n");
        // A spawn() from inside a Pool worker creates a child on the shared
        // group; count it in-flight so pool_shutdown waits for it too (matched
        // by the completion-path decrement).
        self.w("    if (maka_sched_state->group->is_pool) atomic_fetch_add(&maka_sched_state->group->inflight, 1);\n");
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
        self.w("static int __maka_pool_q_push(maka_fiber_t* f) {\n");
        self.w("    pthread_mutex_lock(&__maka_pool_q.lock);\n");
        // Queue closed (e.g. all workers failed init) — refuse the push so
        // the caller can mark the Thread done instead of leaking the fiber.
        self.w("    if (__maka_pool_q.closed) { pthread_mutex_unlock(&__maka_pool_q.lock); return -1; }\n");
        self.w("    f->next = NULL;\n");
        self.w("    if (__maka_pool_q.tail) __maka_pool_q.tail->next = f;\n");
        self.w("    else __maka_pool_q.head = f;\n");
        self.w("    __maka_pool_q.tail = f;\n");
        self.w("    pthread_cond_signal(&__maka_pool_q.cond);\n");
        self.w("    pthread_mutex_unlock(&__maka_pool_q.lock);\n");
        self.w("    return 0;\n");
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
        // If a cancel arrived before this worker picked up the fiber, drop
        // the fiber here without running its body — sets done_flag + drops
        // runner ref so any joiner exits.
        self.w("        if (atomic_load(&f->completion->cancel_requested)) {\n");
        self.w("            Thread* tcompl = f->completion;\n");
        self.w("            pthread_mutex_lock(&tcompl->done_mutex);\n");
        self.w("            tcompl->done_flag = 1;\n");
        self.w("            pthread_cond_broadcast(&tcompl->done_cond);\n");
        self.w("            pthread_mutex_unlock(&tcompl->done_mutex);\n");
        self.w("            __maka_free_env(f->completion, f->entry_env); __maka_slab_free(f->slab); free(f);\n");
        self.w("            __maka_thread_unref(tcompl);  /* runner ref */\n");
        self.w("            continue;\n");
        self.w("        }\n");
        // Now that the fiber has been pinned to this worker, claim it as our
        // own.  Without this, cross-thread wakes (cond_signal, chan_send,
        // etc.) would route through home_sched==NULL and either fall back
        // to a local enqueue on the wrong thread or skip the wake entirely.
        // Epoch BEFORE pointer release on both fiber and its Thread mirror.
        self.w("        atomic_store_explicit(&f->home_sched_epoch, maka_sched_state->epoch, memory_order_relaxed);\n");
        self.w("        atomic_store_explicit(&f->home_sched, (void*)maka_sched_state, memory_order_release);\n");
        self.w("        atomic_store_explicit(&f->completion->home_sched_epoch, maka_sched_state->epoch, memory_order_relaxed);\n");
        self.w("        atomic_store_explicit((_Atomic(void*)*)&f->completion->home_sched, (void*)maka_sched_state, memory_order_release);\n");
        self.w("        __maka_ready_enqueue(f);\n");
        self.w("        /* Drive scheduler until our local queue is empty + nothing waiting. */\n");
        self.w("        maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
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
        // Count actually-started workers — if pthread_create fails for some,
        // we still know the real worker count; if it fails for ALL workers,
        // we mark the queue closed so enqueue can fail fast instead of
        // deadlocking joiners waiting for work that will never run.
        self.w("    int started = 0;\n");
        self.w("    for (int i = 0; i < (int)n; i++) {\n");
        self.w("        pthread_t w;\n");
        self.w("        if (pthread_create(&w, NULL, __maka_pool_worker, NULL) == 0) {\n");
        self.w("            pthread_detach(w); started++;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    __maka_pool_n_workers = started;\n");
        self.w("    if (started == 0) {\n");
        self.w("        pthread_mutex_lock(&__maka_pool_q.lock);\n");
        self.w("        __maka_pool_q.closed = 1;\n");
        self.w("        pthread_cond_broadcast(&__maka_pool_q.cond);\n");
        self.w("        pthread_mutex_unlock(&__maka_pool_q.lock);\n");
        self.w("    }\n");
        self.w("}\n");
        // spawn_pool(): spawn a fiber that runs on the background pool.
        self.w("maka_unit* __maka_spawn_pool(void* code, void* env, void* env_drop) {\n");
        self.w("    __maka_pool_init();\n");
        self.w("    Thread* t = __maka_thread_new();\n");
        self.w("    t->is_fiber = 1;\n");
        self.w("    t->env_drop = (void(*)(void*))env_drop;\n");
        self.w("    maka_fiber_t* f = (maka_fiber_t*)calloc(1, sizeof(maka_fiber_t));\n");
        self.w("    if (!f) {\n");
        self.w("        atomic_store(&t->is_failed, 1);\n");
        self.w("        pthread_mutex_lock(&t->done_mutex); t->done_flag = 1; pthread_cond_broadcast(&t->done_cond); pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("        __maka_thread_unref(t);  /* runner ref */\n");
        self.w("        return (maka_unit*)t;\n");
        self.w("    }\n");
        self.w("    f->slab = __maka_slab_alloc(MAKA_FIBER_STACK_SIZE);\n");
        self.w("    if (!f->slab) {\n");
        self.w("        free(f);\n");
        self.w("        atomic_store(&t->is_failed, 1);\n");
        self.w("        pthread_mutex_lock(&t->done_mutex); t->done_flag = 1; pthread_cond_broadcast(&t->done_cond); pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("        __maka_thread_unref(t);  /* runner ref */\n");
        self.w("        return (maka_unit*)t;\n");
        self.w("    }\n");
        self.w("    f->entry_code = (void(*)(void*))code;\n");
        self.w("    f->entry_env = env;\n");
        self.w("    f->completion = t;\n");
        self.w("    f->state = 0;\n");
        self.w("    f->waiting_fd = -1; f->waiting_events = 0;\n");
        self.w("    f->wait_deadline_ns = 0; f->wait_timed_out = 0;\n");
        self.w("    maka_ctx_make(&f->ctx, f->slab->stack_top, f->slab->stack_size, __maka_fiber_entry);\n");
        self.w("    if (__maka_pool_q_push(f) != 0) {\n");
        // Pool refused (queue closed) — simulate immediate completion so
        // join() doesn't hang, free the fiber, drop the runner ref.
        self.w("        pthread_mutex_lock(&t->done_mutex);\n");
        self.w("        t->done_flag = 1;\n");
        self.w("        pthread_cond_broadcast(&t->done_cond);\n");
        self.w("        pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("        __maka_free_env(f->completion, f->entry_env); __maka_slab_free(f->slab); free(f);\n");
        self.w("        __maka_thread_unref(t);  /* runner ref */\n");
        self.w("    }\n");
        self.w("    return (maka_unit*)t;\n");
        self.w("}\n");
        // ====================================================================
        // Explicit worker Pool: `Pool p = pool(N)` / `spawn_on(p, f)` /
        // `pool_shutdown(p)`.  A user-sized set of N worker threads, each
        // running its own hardened per-thread fiber scheduler and draining a
        // shared intake queue.  `spawn_on` fans a fiber across the pool's
        // workers; a worker that grabs a fiber runs it there (cross-worker
        // migration/stealing is a follow-up).  `pool_shutdown` closes the
        // intake and joins the workers once their fibers drain.  Reuses the
        // single-thread scheduler unchanged, so nothing about `spawn()` moves.
        // ====================================================================
        self.w(r#"typedef struct maka_pool_s {
    pthread_t* workers;
    int n_workers;                 /* worker threads actually started */
    int req_workers;               /* requested size = shared group size */
    maka_sched_state_t* sched;     /* shared scheduler; created by the first worker */
    pthread_mutex_t init_mu;
    pthread_cond_t init_cv;
    int creating;                  /* 1 once a worker has begun creating `sched` */
    int closed;                    /* pool_shutdown requested */
    size_t stack_bytes;            /* fiber stack size for spawn_on/job_on (0 = default) */
} maka_pool_t;
/* Each worker runs the SHARED scheduler over the SHARED run queue + reactor, so
   a parked-then-woken fiber resumes on whichever worker is free (M:N), and a
   sleep/IO-parked fiber frees its worker (serviced by the poll-owner). */
static void* __maka_pool_h_worker(void* arg) {
    maka_pool_t* pool = (maka_pool_t*)arg;
    pthread_mutex_lock(&pool->init_mu);
    if (!pool->creating && !pool->sched) {
        /* First worker: build the shared scheduler via the normal init (which
           arms the wake_pipe on this thread's TLS reactor -> becomes the shared
           reactor), then promote its group to a shared Pool group. */
        pool->creating = 1;
        pthread_mutex_unlock(&pool->init_mu);
        __maka_sched_init();
        maka_sched_state->group->is_pool = 1;
        maka_sched_state->group->n_workers = pool->req_workers;
        /* Grow the per-worker deque array to N slots (slot 0 is this creator's,
           set by sched_init); peers claim slots 1..N-1 via worker_seq. */
        {
            _Atomic(maka_frunq_t*)* fr = (_Atomic(maka_frunq_t*)*)calloc((size_t)pool->req_workers, sizeof(_Atomic(maka_frunq_t*)));
            atomic_store_explicit(&fr[0], atomic_load_explicit(&maka_sched_state->group->frunqs[0], memory_order_relaxed), memory_order_relaxed);
            free(maka_sched_state->group->frunqs);
            maka_sched_state->group->frunqs = fr;
        }
        pthread_mutex_lock(&pool->init_mu);
        pool->sched = maka_sched_state;
        pthread_cond_broadcast(&pool->init_cv);
        pthread_mutex_unlock(&pool->init_mu);
    } else {
        while (!pool->sched) pthread_cond_wait(&pool->init_cv, &pool->init_mu);
        maka_sched_state_t* shared = pool->sched;
        pthread_mutex_unlock(&pool->init_mu);
        /* Attach to the shared scheduler: shared sched_state, own per-worker
           anchor/ctx/tick.  A ref + the pthread-key registration make this
           worker's exit tear its ref + tick + anchor down automatically. */
        maka_sched_state = shared;
        maka_sched_inited = 1;
        __maka_sched_state_ref(shared);
        /* Claim a worker id + own deque slot in the shared group. */
        maka_my_worker_id = atomic_fetch_add(&shared->group->worker_seq, 1);
        maka_my_frunq = (maka_frunq_t*)calloc(1, sizeof(maka_frunq_t));
        atomic_store_explicit(&shared->group->frunqs[maka_my_worker_id], maka_my_frunq, memory_order_relaxed);
        __maka_watchdog_register();
        __maka_sched_register_thread(shared);
        maka_anchor_fiber = (maka_fiber_t*)calloc(1, sizeof(maka_fiber_t));
        maka_anchor_fiber->state = 1;
        atomic_store_explicit(&maka_anchor_fiber->home_sched_epoch, shared->epoch, memory_order_relaxed);
        atomic_store_explicit(&maka_anchor_fiber->home_sched, (void*)shared, memory_order_release);
        maka_current_fiber = maka_anchor_fiber;
        maka_ctx_make(&maka_sched_ctx, maka_sched_stack, sizeof(maka_sched_stack), __maka_scheduler_loop);
    }
    /* Run until the pool is closed AND fully drained (the loop returns to the
       anchor from the pool-mode idle path only on that exit condition). */
    maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);
    __maka_slab_drain();  /* release this worker's cached slabs before exit */
    return NULL;
}
maka_unit* __maka_pool_new(maka_int n, maka_int stack_bytes) {
    if (n < 1) n = 1;
    if (n > 1024) n = 1024;
    maka_pool_t* pool = (maka_pool_t*)calloc(1, sizeof(maka_pool_t));
    if (!pool) return NULL;
    pthread_mutex_init(&pool->init_mu, NULL);
    pthread_cond_init(&pool->init_cv, NULL);
    pool->req_workers = (int)n;
    /* Clamp the stack size: 0 -> default; otherwise round up to a page and
       floor at the default so a too-small request can't overflow immediately. */
    if (stack_bytes <= 0) pool->stack_bytes = 0;
    else { size_t sb = (size_t)stack_bytes; if (sb < (size_t)MAKA_FIBER_STACK_SIZE) sb = (size_t)MAKA_FIBER_STACK_SIZE; sb = (sb + 4095u) & ~(size_t)4095u; pool->stack_bytes = sb; }
    pool->workers = (pthread_t*)calloc((size_t)n, sizeof(pthread_t));
    if (!pool->workers) { free(pool); return NULL; }
    int started = 0;
    for (maka_int i = 0; i < n; i++) {
        if (pthread_create(&pool->workers[started], NULL, __maka_pool_h_worker, pool) == 0) started++;
    }
    pool->n_workers = started;
    if (started == 0) {
        /* Nothing to run on; mark closed so spawn_on fails fast. */
        pthread_mutex_lock(&pool->init_mu);
        pool->closed = 1;
        pthread_cond_broadcast(&pool->init_cv);
        pthread_mutex_unlock(&pool->init_mu);
    }
    return (maka_unit*)pool;
}
static void __maka_pool_submit_fail(Thread* t, maka_fiber_t* f) {
    if (f) { if (f->slab) __maka_slab_free(f->slab); free(f); }
    atomic_store(&t->is_failed, 1);
    pthread_mutex_lock(&t->done_mutex); t->done_flag = 1;
    pthread_cond_broadcast(&t->done_cond); pthread_mutex_unlock(&t->done_mutex);
    __maka_thread_unref(t);
}
maka_unit* __maka_pool_spawn_on(maka_unit* poolv, void* code, void* env, void* env_drop) {
    maka_pool_t* pool = (maka_pool_t*)poolv;
    Thread* t = __maka_thread_new();
    t->is_fiber = 1;
    t->env_drop = (void(*)(void*))env_drop;
    maka_fiber_t* f = (maka_fiber_t*)calloc(1, sizeof(maka_fiber_t));
    maka_slab_t* slab = f ? __maka_slab_alloc(pool ? pool->stack_bytes : 0) : NULL;
    /* Wait for the shared scheduler to come up (the first worker may still be
       creating it) unless the pool is already closed. */
    maka_sched_state_t* sched = NULL;
    if (pool) {
        pthread_mutex_lock(&pool->init_mu);
        while (!pool->sched && !pool->closed) pthread_cond_wait(&pool->init_cv, &pool->init_mu);
        sched = pool->closed ? NULL : pool->sched;
        pthread_mutex_unlock(&pool->init_mu);
    }
    if (!pool || !sched || !f || !slab) {
        if (slab && f) { f->slab = slab; } else if (slab) { __maka_slab_free(slab); }
        __maka_pool_submit_fail(t, f);
        return (maka_unit*)t;
    }
    f->slab = slab;
    f->entry_code = (void(*)(void*))code; f->entry_env = env; f->completion = t; f->state = 0;
    f->waiting_fd = -1; f->waiting_events = 0; f->wait_deadline_ns = 0; f->wait_timed_out = 0;
    maka_ctx_make(&f->ctx, f->slab->stack_top, f->slab->stack_size, __maka_fiber_entry);
    /* Home both the fiber and its completion Thread on the shared scheduler so
       wakes/joins route there. */
    atomic_store_explicit(&f->home_sched_epoch, sched->epoch, memory_order_relaxed);
    atomic_store_explicit(&f->home_sched, (void*)sched, memory_order_release);
    atomic_store_explicit(&t->home_sched_epoch, sched->epoch, memory_order_relaxed);
    atomic_store_explicit((_Atomic(void*)*)&t->home_sched, (void*)sched, memory_order_release);
    /* Count it as in-flight BEFORE enqueue so a worker cannot drain + complete +
       decrement before this add lands (which would let shutdown exit early). */
    atomic_fetch_add(&sched->group->inflight, 1);
    /* Submit through the proven cross-thread wake seam: posts to the shared
       remote-wake inbox + pings the shared wake_pipe, so the poll-owner drains
       it onto the shared run queue and any free worker picks it up. */
    __maka_ready_enqueue(f);
    return (maka_unit*)t;
}
void __maka_pool_free(maka_unit* poolv) {
    maka_pool_t* pool = (maka_pool_t*)poolv;
    if (!pool) return;
    pthread_mutex_lock(&pool->init_mu);
    pool->closed = 1;
    /* Wait for worker setup to finish (the first worker publishes pool->sched)
       so we close the group they actually run the scheduler loop under.  A pool
       torn down immediately after creation (never spawned on) would otherwise
       read a NULL sched, skip the group-close below, and the workers - which
       enter the scheduler loop right after publishing pool->sched - would never
       see the close, so their join here would hang. */
    while (!pool->sched && pool->n_workers > 0) pthread_cond_wait(&pool->init_cv, &pool->init_mu);
    maka_sched_state_t* sched = pool->sched;
    pthread_cond_broadcast(&pool->init_cv);  /* release a spawn_on awaiting sched */
    pthread_mutex_unlock(&pool->init_mu);
    if (sched) {
        /* Mark the shared group closed + wake every worker (idle cv-waiters via
           broadcast, the poll-owner via the wake_pipe) so they re-check the
           drain/exit condition.  Submitted work still runs to completion first
           (workers exit only when inflight hits 0). */
        pthread_mutex_lock(&sched->group->lock);
        sched->group->closed = 1;
        pthread_cond_broadcast(&sched->group->cv);
        pthread_mutex_unlock(&sched->group->lock);
        if (sched->wake_pipe_w > 0) { char b = 1; (void)send(sched->wake_pipe_w, &b, 1, 0); }
    }
    for (int i = 0; i < pool->n_workers; i++) pthread_join(pool->workers[i], NULL);
    /* The shared sched_state is freed by the last worker's pthread-key dtor
       (refcount drop to 0), so we must not touch it after the joins. */
    pthread_mutex_destroy(&pool->init_mu);
    pthread_cond_destroy(&pool->init_cv);
    free(pool->workers);
    /* NOT free(pool): the `own *PoolBody` handle owns the maka_pool_t block, so
       Maka libc-frees it after this teardown (drop = logical teardown, allocator
       frees the block - the Rust model).  Freeing it here would double-free. */
}
"#);
        // Cooperative sleep / yield primitives.
        self.w("void __maka_yield_now(void) {\n");
        self.w("    if (!maka_current_fiber || maka_current_fiber == maka_anchor_fiber) return;\n");
        self.w("    maka_fiber_t* me = maka_current_fiber;\n");
        // Defer the re-enqueue to the finalize block (after this context is
        // saved) - enqueueing before the switch would let a peer steal + run us
        // on a half-saved context.  kind 3 = re-enqueue as ready.
        self.w("    maka_timerfd_park.kind = 3;\n");
        self.w("    maka_pending_park = &maka_timerfd_park;\n");
        self.w("    maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("}\n");
        self.w("static void __maka_sleep_fiber(int64_t nanos) {\n");
        self.w("    maka_fiber_t* me = maka_current_fiber;\n");
        self.w("    me->wake_at_ns = __maka_now_ns() + nanos;\n");
        self.w("    me->state = 3;\n");
        // Defer the sleep-list insertion to the scheduler's finalize-park block
        // (after this ctx save completes), so a peer Pool worker can't expire +
        // run us mid-switch.
        self.w("    maka_timerfd_park.kind = 1;\n");
        self.w("    maka_pending_park = &maka_timerfd_park;\n");
        self.w("    maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
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
        // No scheduler on this thread => no reactor (maka_epoll_fd is
        // maka_sched_state->epoll_fd).  A bare `main` that uses the blocking
        // solo async path never armed anything, so close_fd's disarm has
        // nothing to do; without this guard the epoll_fd macro derefs NULL.
        self.w("    if (!maka_sched_state) return;\n");
        self.w("    if (maka_epoll_fd < 0) { maka_epoll_fd = epoll_create1(EPOLL_CLOEXEC); }\n");
        self.w("    if (events_mask == 0) {\n");
        self.w("        epoll_ctl(maka_epoll_fd, EPOLL_CTL_DEL, fd, NULL);\n");
        self.w("        __maka_fd_reg_drop(fd);\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("    maka_fd_reg_t* r = __maka_fd_reg_ensure(fd);\n");
        // Always issue MOD even if the cached mask matches.  The kernel may
        // have evicted the registration out-of-band (close() auto-cleans
        // epoll, kqueue EV_EOF deletes filters); the ENOENT recovery below
        // would never fire if we short-circuited here.
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
        // The wake_pipe is always armed for EPOLLIN by sched_init and never
        // has a fiber waiter — recomputing it would EPOLL_CTL_DEL it and
        // permanently disable cross-thread wakes for this scheduler.  Skip.
        self.w("    if (maka_sched_state && fd == maka_sched_state->wake_pipe_r) return;\n");
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
        self.w("    maka_current_fiber->wait_deadline_ns = 0;\n");
        self.w("    maka_current_fiber->wait_timed_out = 0;\n");
        // Defer fd_waiters insertion + epoll arm to the finalize-park block so a
        // peer Pool worker cannot see the fd ready and run us mid-switch.
        self.w("    maka_timerfd_park.kind = 2;\n");
        self.w("    maka_pending_park = &maka_timerfd_park;\n");
        self.w("    maka_ctx_switch(&maka_current_fiber->ctx, &maka_sched_ctx);\n");
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
        // Defer fd_waiters insertion + epoll arm to the finalize-park block.
        self.w("    maka_timerfd_park.kind = 2;\n");
        self.w("    maka_pending_park = &maka_timerfd_park;\n");
        self.w("    maka_ctx_switch(&maka_current_fiber->ctx, &maka_sched_ctx);\n");
        self.w("    /* Scheduler resumes us either on fd event or on deadline. */\n");
        self.w("    return maka_current_fiber->wait_timed_out ? 0 : 1;\n");
        self.w("}\n");
        self.w("#include <fcntl.h>\n");
        // TCP helpers — defined in an inner block so the sys/socket.h /
        // netinet/in.h symbol pollution (especially the bare `accept`
        // function name) doesn't escape into user code.  We define the
        // runtime functions inside `extern "C"`-equivalent scope and
        // reference them by their __maka_ names from the rest of the file.
        self.w("/* TCP runtime - scoped includes to avoid name clashes. */\n");
        self.w("static inline int64_t __maka_tcp_listen(int64_t port, int64_t backlog);\n");
        self.w("static inline int64_t __maka_tcp_listen_any(int64_t port, int64_t backlog);\n");
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
        self.w("static inline int64_t __maka_tls_server_init(const char* cert_pem, const char* key_pem);\n");
        self.w("static inline maka_unit* __maka_tls_server_accept_new(int64_t fd);\n");
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
        // _Atomic: the job-pool init writes this after spawning the work-stealing
        // workers, which read it concurrently in their steal loop (a C11 data
        // race on the plain global).  Atomic access is TSan-clean.
        self.w("static _Atomic long __maka_n_workers = 0;\n");
        self.w("static __maka_ws_deque_t* __maka_ws_deques = NULL;\n");
        self.w("static __thread int __maka_ws_worker_id = -1;\n");
        // Idle workers sleep on this condvar.  job() (and steal-source pushes)
        // broadcast to wake them up.  Avoids the ~80% CPU "active idle" of a
        // pure spin-then-nanosleep loop on machines with many cores.
        self.w("static pthread_mutex_t __maka_ws_idle_mu = PTHREAD_MUTEX_INITIALIZER;\n");
        self.w("static pthread_cond_t  __maka_ws_idle_cv = PTHREAD_COND_INITIALIZER;\n");
        self.w("static _Atomic int     __maka_ws_pending = 0;  /* jobs currently in any deque */\n");
        self.w("static inline void __maka_ws_notify(void) {\n");
        self.w("    pthread_mutex_lock(&__maka_ws_idle_mu);\n");
        self.w("    pthread_cond_broadcast(&__maka_ws_idle_cv);\n");
        self.w("    pthread_mutex_unlock(&__maka_ws_idle_mu);\n");
        self.w("}\n");
        self.w("static __thread unsigned int __maka_ws_rng = 0;\n");
        self.w("static unsigned int __maka_ws_rand(void) {\n");
        self.w("    /* xorshift32 - fast, no global state contention */\n");
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
        self.w("        /* Last item - race with a stealer. */\n");
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
        self.w("    __maka_free_env(item->h, item->env);\n");
        // Drop the runner-side ref — joiner/detacher will drop the spawner ref.
        self.w("    __maka_thread_unref(item->h);\n");
        self.w("}\n");
        self.w("static void* __maka_ws_worker(void* arg) {\n");
        self.w("    int id = (int)(intptr_t)arg;\n");
        self.w("    __maka_ws_worker_id = id;\n");
        self.w("    __maka_ws_deque_t* mine = &__maka_ws_deques[id];\n");
        self.w("    int idle_iters = 0;\n");
        self.w("    while (1) {\n");
        self.w("        __maka_job_entry_t item;\n");
        self.w("        if (__maka_ws_pop(mine, &item)) {\n");
        self.w("            atomic_fetch_sub(&__maka_ws_pending, 1);\n");
        self.w("            __maka_ws_run(&item);\n");
        self.w("            idle_iters = 0; continue;\n");
        self.w("        }\n");
        self.w("        /* Try to steal from a random victim. */\n");
        self.w("        if (__maka_n_workers > 1) {\n");
        self.w("            int v = (int)(__maka_ws_rand() % (unsigned int)__maka_n_workers);\n");
        self.w("            if (v == id) v = (v + 1) % __maka_n_workers;\n");
        self.w("            if (__maka_ws_steal(&__maka_ws_deques[v], &item)) {\n");
        self.w("                atomic_fetch_sub(&__maka_ws_pending, 1);\n");
        self.w("                __maka_ws_run(&item);\n");
        self.w("                idle_iters = 0; continue;\n");
        self.w("        }\n");
        self.w("        }\n");
        // Short spin first — fresh work often shows up within a few hundred
        // cycles after a push.  Then fall through to a condvar wait so the
        // worker doesn't burn CPU while there's no work anywhere.
        self.w("        idle_iters++;\n");
        self.w("        if (idle_iters < 64) {\n");
        self.w("            for (volatile int s = 0; s < 32; s++) {}\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        pthread_mutex_lock(&__maka_ws_idle_mu);\n");
        self.w("        while (atomic_load(&__maka_ws_pending) == 0) {\n");
        self.w("            pthread_cond_wait(&__maka_ws_idle_cv, &__maka_ws_idle_mu);\n");
        self.w("        }\n");
        self.w("        pthread_mutex_unlock(&__maka_ws_idle_mu);\n");
        self.w("        idle_iters = 0;\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static _Atomic int __maka_job_pool_inited = 0;\n");
        self.w("static void __maka_job_pool_init(void) {\n");
        // Atomic CAS so only one thread performs initialization even when
        // multiple callers race on first job() (mirror __maka_pool_init).
        self.w("    int expected = 0;\n");
        self.w("    if (!atomic_compare_exchange_strong(&__maka_job_pool_inited, &expected, 1)) return;\n");
        self.w("    long n = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (n < 1) n = 1;\n");
        self.w("    if (n > 64) n = 64;\n");
        self.w("    __maka_n_workers = n;\n");
        self.w("    __maka_ws_deques = (__maka_ws_deque_t*)calloc((size_t)n, sizeof(__maka_ws_deque_t));\n");
        self.w("    for (long i = 0; i < n; i++) {\n");
        self.w("        atomic_init(&__maka_ws_deques[i].top, 0);\n");
        self.w("        atomic_init(&__maka_ws_deques[i].bottom, 0);\n");
        self.w("    }\n");
        // Track actually-started workers — pass `started` (dense [0..started))
        // as the worker id so live workers occupy contiguous deque slots.
        // Without this, sparse ids leave gaps and __maka_spawn_job's RNG
        // round-robin pushes into dead deques that nobody steals from.
        self.w("    long started = 0;\n");
        self.w("    for (long i = 0; i < n; i++) {\n");
        self.w("        pthread_t w;\n");
        self.w("        if (pthread_create(&w, NULL, __maka_ws_worker, (void*)(intptr_t)started) == 0) {\n");
        self.w("            pthread_detach(w); started++;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    __maka_n_workers = started;\n");
        self.w("}\n");
        // job() — push to a worker's deque (round-robin from non-worker callers,
        // own-deque from worker callers).
        self.w("static __thread int __maka_job_rr = 0;\n");
        self.w("maka_unit* __maka_spawn_job(void* code, void* env, void* env_drop) {\n");
        self.w("    __maka_job_pool_init();\n");
        self.w("    Thread* t = __maka_thread_new();\n");
        self.w("    t->is_job = 1;\n");
        self.w("    t->env_drop = (void(*)(void*))env_drop;\n");
        self.w("    __maka_job_entry_t item = { code, env, t };\n");
        self.w("    if (__maka_ws_worker_id >= 0) {\n");
        self.w("        /* Caller is a worker - push to own deque (LIFO, cache-warm). */\n");
        self.w("        if (__maka_ws_push(&__maka_ws_deques[__maka_ws_worker_id], item)) {\n");
        self.w("            atomic_fetch_add(&__maka_ws_pending, 1);\n");
        self.w("            __maka_ws_notify();\n");
        self.w("            return (maka_unit*)t;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    /* Non-worker caller, or own deque full: round-robin push. */\n");
        self.w("    for (long try_count = 0; try_count < __maka_n_workers; try_count++) {\n");
        self.w("        int target = (__maka_job_rr + (int)try_count) % (int)__maka_n_workers;\n");
        self.w("        if (__maka_ws_push(&__maka_ws_deques[target], item)) {\n");
        self.w("            __maka_job_rr = (target + 1) % (int)__maka_n_workers;\n");
        self.w("            atomic_fetch_add(&__maka_ws_pending, 1);\n");
        self.w("            __maka_ws_notify();\n");
        self.w("            return (maka_unit*)t;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    /* All deques full: fall back to a dedicated pthread. */\n");
        self.w("    __maka_handle_args_t* a = (__maka_handle_args_t*)malloc(sizeof(__maka_handle_args_t));\n");
        self.w("    a->code = code; a->env = env; a->h = t;\n");
        self.w("    __maka_spawn_pthread(&t->handle, __maka_handle_entry, a, t);\n");
        self.w("    return (maka_unit*)t;\n");
        self.w("}\n");
        // Back-compat: `__maka_spawn` is still emitted, aliased to `spawn` (fiber).
        // Existing tests that build `spawn(closure)` and previously hit the
        // pthread path now hit the smaller-stack fiber path — same observable
        // behavior, less RAM.
        self.w("maka_unit* __maka_spawn(void* code, void* env) { return __maka_spawn_fiber(code, env, NULL); }\n");
        // ====================================================================
        // JOIN — block on a single handle and reclaim its result.
        // ====================================================================
        self.w("static int __maka_pp_join(void* p) { return ((Thread*)p)->done_flag == 0; }\n");
        self.w("int64_t __maka_join_result(maka_unit* h) {\n");
        self.w("    Thread* t = (Thread*)h;\n");
        self.w("    /* Fast path: already done. */\n");
        self.w("    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("    int done = t->done_flag;\n");
        self.w("    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("    if (!done && maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        // Joiner is a FIBER: park on the target Thread's fiber-waiter list and
        // yield the worker back to the scheduler.  Never block the worker
        // (cond_wait / nanosleep) - in a Pool that deadlocks, since the child
        // being joined may need this very worker to run.  should_park re-checks
        // done_flag under done_mutex; the completion path drains us under the
        // same lock, so there is no lost wake.
        self.w("        maka_fiber_t* me = maka_current_fiber;\n");
        self.w("        while (1) {\n");
        self.w("            pthread_mutex_lock(&t->done_mutex);\n");
        self.w("            int d = t->done_flag;\n");
        self.w("            pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("            if (d) break;\n");
        self.w("            maka_park_req_t req; req.lock = &t->done_mutex; req.head = &t->fiber_waiters;\n");
        self.w("            req.should_park = __maka_pp_join; req.arg = t; req.inflight = NULL; req.drained_cv = NULL; req.kind = 0;\n");
        self.w("            maka_pending_park = &req;\n");
        self.w("            me->state = 2;\n");
        self.w("            maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("        }\n");
        self.w("    } else if (!done) {\n");
        self.w("        /* If the scheduler has ready/sleeping/fd-waiting fibers, drive\n");
        self.w("           it instead of pthread-cond-waiting (which would freeze it). */\n");
        self.w("        if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
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
        self.w("                maka_join_target = t; /* the Thread (outlives the fiber), not target */\n");
        self.w("                /* Switch into the scheduler; it'll swap back to us when\n");
        self.w("                   target finishes. */\n");
        // Sub-fiber-safe: enqueue me on target->waiters so the natural
        // completion path resumes us; the scheduler at fiber-finished
        // drains target->waiters via __maka_ready_enqueue.
        // Strategy Y: arm pending_park; scheduler pushes onto target->waiters
        // under done_mutex AFTER swap-save completes (no race with the
        // fiber-finished drain re-running with a stale ctx).
        self.w("                if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("                    maka_fiber_t* me = maka_current_fiber;\n");
        self.w("                    maka_park_req_t req = { .lock = &t->done_mutex, .head = &target->waiters,\n");
        self.w("                                            .should_park = __maka_pp_join, .arg = t };\n");
        self.w("                    maka_pending_park = &req;\n");
        self.w("                    me->state = 2;\n");
        self.w("                    maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("                } else {\n");
        self.w("                    maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("                    maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                }\n");
        self.w("                maka_join_target = NULL;\n");
        self.w("            } else {\n");
        self.w("                /* Not a fiber we own - it's a pthread thread or a job. Drive\n");
        self.w("                   scheduler ALSO so any waiting fibers can still progress\n");
        self.w("                   while we periodically poll the foreign handle. */\n");
        self.w("                while (1) {\n");
        self.w("                    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("                    int d = t->done_flag;\n");
        self.w("                    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("                    if (d) break;\n");
        self.w("                    if (__maka_have_ready() || maka_sleep_head || maka_fd_waiters) {\n");
        self.w("                        maka_join_target = NULL;\n");
        // Sub-fiber-safe: re-enqueue ME as ready before yielding so the
        // scheduler picks it up again on the next iteration to re-check
        // the foreign handle.  Otherwise swapcontext orphans the fiber.
        self.w("                        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("                            maka_fiber_t* me = maka_current_fiber;\n");
        self.w("                            __maka_ready_enqueue(me);\n");
        self.w("                            maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("                        } else {\n");
        self.w("                            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("                            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("                        }\n");
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
        self.w("    if (!t->is_job && !t->is_fiber) { if (!atomic_load(&t->is_failed)) pthread_join(t->handle, NULL); }\n");
        self.w("    __maka_thread_unref(t);\n");
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
        self.w("        if (!t->is_job && !t->is_fiber) { if (!atomic_load(&t->is_failed)) pthread_join(t->handle, NULL); }\n");
        self.w("        __maka_thread_unref(t);\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("    atomic_store(&t->detached, 1);\n");
        // For thread tier, also pthread_detach so the OS thread reaps itself.
        self.w("    if (!t->is_job && !t->is_fiber) pthread_detach(t->handle);\n");
        // Drop the spawner-side ref — runner will drop its own when finishing.
        self.w("    __maka_thread_unref(t);\n");
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
        self.w("    if (!t->is_job && !t->is_fiber) { if (!atomic_load(&t->is_failed)) pthread_join(t->handle, NULL); }\n");
        self.w("    __maka_thread_unref(t);\n");
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
        self.w("            if (!t->is_job && !t->is_fiber) { if (!atomic_load(&t->is_failed)) pthread_join(t->handle, NULL); }\n");
        self.w("            __maka_thread_unref(t);\n");
        self.w("            return 1;\n");
        self.w("        }\n");
        self.w("        int64_t now = __maka_now_ms();\n");
        self.w("        if (now >= deadline_ms) {\n");
        self.w("            maka_anchor_deadline_ns = prev_anchor_deadline;\n");
        self.w("            return 0;\n");
        self.w("        }\n");
        // A spawned (non-anchor) fiber must NOT swapcontext through the anchor's
        // context slot - that clobbers the anchor's saved continuation and strands
        // this fiber (it is on no ready/sleep queue) -> deadlock.  Yield as a fiber
        // (re-enqueue self, run the scheduler) and re-check the loop.
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("            __maka_yield_now();\n");
        self.w("        } else if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
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
        // Local fiber cancel: walk the CURRENT thread's queues, unlink the
        // fiber, free its stack, set done_flag, drop runner-side ref.
        // Returns 1 if the fiber was found on this thread, 0 otherwise.
        self.w("static int __maka_cancel_fiber_local(Thread* t) {\n");
        // Walks the shared run/sleep/fd lists — hold the group lock (recursive,
        // so a caller that already holds it is fine).
        self.w("    MAKA_GRP_LOCK();\n");
        self.w("    maka_fiber_t** prev = &maka_ready_head;\n");
        self.w("    maka_fiber_t* found = NULL;\n");
        self.w("    int found_in = 0;  /* 1=ready 2=sleep 3=fd_waiters */\n");
        self.w("    while (*prev) {\n");
        self.w("        if ((*prev)->completion == t) {\n");
        self.w("            found = *prev; *prev = found->next;\n");
        // Update tail correctly: if found was tail, the new tail is the prev
        // pointer's container — i.e. the node whose next we just rewrote.
        // We can recover it from the "prev" pointer's owner.  Easier: scan
        // to find the new tail after unlink if found == tail.
        self.w("            if (maka_ready_tail == found) {\n");
        self.w("                maka_fiber_t* nt = maka_ready_head;\n");
        self.w("                if (!nt) maka_ready_tail = NULL;\n");
        self.w("                else { while (nt->next) nt = nt->next; maka_ready_tail = nt; }\n");
        self.w("            }\n");
        self.w("            found_in = 1; break;\n");
        self.w("        }\n");
        self.w("        prev = &(*prev)->next;\n");
        self.w("    }\n");
        self.w("    if (!found) {\n");
        self.w("        prev = &maka_sleep_head;\n");
        self.w("        while (*prev) {\n");
        self.w("            if ((*prev)->completion == t) { found = *prev; *prev = found->next; found_in = 2; break; }\n");
        self.w("            prev = &(*prev)->next;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    int cancelled_fd = -1;\n");
        self.w("    if (!found) {\n");
        self.w("        prev = &maka_fd_waiters;\n");
        self.w("        while (*prev) {\n");
        self.w("            if ((*prev)->completion == t) {\n");
        self.w("                found = *prev; *prev = found->next;\n");
        self.w("                cancelled_fd = found->waiting_fd;\n");
        self.w("                found_in = 3; break;\n");
        self.w("            }\n");
        self.w("            prev = &(*prev)->next;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    if (!found) { MAKA_GRP_UNLOCK(); return 0; }\n");
        // If a detached AIO worker still holds refs to caller-owned buf/fd,
        // we MUST NOT free the fiber's slab/entry_env — the worker is about
        // to write the result into buf/efd that the caller (this fiber)
        // owns.  Leave the fiber where it is; set done_flag so join() can
        // exit; the worker's eventfd_signal will wake the fiber and it'll
        // complete naturally.  The natural completion path frees the slab.
        self.w("    if (atomic_load(&t->aio_in_flight)) {\n");
        // Re-link the fiber where we unlinked it.  If it was on fd_waiters,
        // it's still parked waiting for the eventfd byte (which the AIO
        // worker WILL send) — re-link there.  Otherwise (was ready/sleep)
        // re-enqueue as ready so the scheduler resumes it later.
        self.w("        if (found_in == 3) {\n");
        self.w("            found->next = maka_fd_waiters; maka_fd_waiters = found;\n");
        self.w("        } else {\n");
        self.w("            atomic_store(&found->in_queue, 0);\n");
        self.w("            __maka_ready_enqueue(found);\n");
        self.w("        }\n");
        // Do NOT flip done_flag here — the joiner could read it before the
        // fiber writes t->result.  Let natural completion own the flip.
        self.w("        MAKA_GRP_UNLOCK();\n");
        self.w("        return 1;\n");
        self.w("    }\n");
        // Cancelled pool fiber is destroyed here (not via the completion path),
        // so account for it in the group's in-flight count or shutdown hangs.
        self.w("    if (maka_sched_state->group->is_pool && found != maka_anchor_fiber) atomic_fetch_sub(&maka_sched_state->group->inflight, 1);\n");
        self.w("    __maka_free_env(found->completion, found->entry_env); __maka_slab_free(found->slab); free(found);\n");
        self.w("    if (cancelled_fd >= 0) __maka_fd_recompute(cancelled_fd);\n");
        self.w("    MAKA_GRP_UNLOCK();\n");
        self.w("    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("    t->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&t->done_cond);\n");
        self.w("    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("    __maka_thread_unref(t);  /* runner-side ref */\n");
        self.w("    return 1;\n");
        self.w("}\n");
        // Post a cancel request onto the target's home scheduler's inbox.
        // The home scheduler will drain it on its next loop iteration and
        // execute __maka_cancel_fiber_local() against its own queues.
        self.w("static void __maka_remote_post_cancel(maka_sched_state_t* sched, Thread* t) {\n");
        self.w("    if (!sched) return;\n");
        // Bump the target Thread (so it can't be freed between post and drain).
        // We hold a sched_state ref already (from the caller bumping it before
        // entering the cross-thread branch); the drain path doesn't need an
        // extra one because the sched is guaranteed live for the duration.
        self.w("    __maka_thread_ref(t);\n");
        self.w("    maka_cancel_req_t* r = (maka_cancel_req_t*)malloc(sizeof(maka_cancel_req_t));\n");
        self.w("    r->target = t;\n");
        self.w("    pthread_mutex_lock(&sched->remote_mu);\n");
        self.w("    r->next = sched->remote_cancel_head;\n");
        self.w("    sched->remote_cancel_head = r;\n");
        self.w("    int wfd = sched->wake_pipe_w;\n");
        self.w("    pthread_mutex_unlock(&sched->remote_mu);\n");
        self.w("    if (wfd > 0) { char b = 1; (void)send((int)wfd, &b, 1, 0); }\n");
        self.w("}\n");
        self.w("void __maka_cancel(maka_unit* h) {\n");
        self.w("    Thread* t = (Thread*)h;\n");
        self.w("    pthread_mutex_lock(&t->done_mutex);\n");
        self.w("    int already_done = t->done_flag;\n");
        self.w("    pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("    if (already_done) { (void)__maka_join_result(h); return; }\n");
        self.w("    if (t->is_fiber) {\n");
        // Set cancel_requested first so a spawn_pool worker that picks up the
        // fiber AFTER this cancel can detect it and skip the body.
        self.w("        atomic_store(&t->cancel_requested, 1);\n");
        // Cross-thread cancel: the target lives on another scheduler.  Don't
        // walk our own queues — post a cancel request onto the home thread's
        // inbox and let it run __maka_cancel_fiber_local() there.  We drop
        // only the spawner ref here; the home scheduler drops the runner ref.
        self.w("        maka_sched_state_t* cand = (maka_sched_state_t*)atomic_load(&t->home_sched);\n");
        self.w("        if (cand && cand != maka_sched_state) {\n");
        // Validate+ref under the ticks mutex so cleanup can't race-free the sched.
        self.w("            maka_sched_state_t* home = __maka_sched_validate_and_ref_epoch(cand, atomic_load_explicit(&t->home_sched_epoch, memory_order_acquire));\n");
        self.w("            if (home) {\n");
        self.w("                __maka_remote_post_cancel(home, t);\n");
        self.w("                __maka_sched_state_unref(home);\n");
        self.w("            } else {\n");
        // Home died; just mark done so any pending join() exits.
        self.w("                pthread_mutex_lock(&t->done_mutex);\n");
        self.w("                t->done_flag = 1;\n");
        self.w("                pthread_cond_broadcast(&t->done_cond);\n");
        self.w("                pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("                __maka_thread_unref(t);  /* runner */\n");
        self.w("            }\n");
        self.w("            __maka_thread_unref(t);  /* spawner */\n");
        self.w("            return;\n");
        self.w("        }\n");
        self.w("        if (!__maka_cancel_fiber_local(t)) {\n");
        // Not in local queues — try the spawn_pool queue (fiber may be
        // queued there waiting for a worker to pick it up).  If found,
        // execute the worker's cancel-on-pickup epilogue inline.
        self.w("            pthread_mutex_lock(&__maka_pool_q.lock);\n");
        self.w("            maka_fiber_t** pq = &__maka_pool_q.head;\n");
        self.w("            maka_fiber_t* pprev = NULL;\n");
        self.w("            maka_fiber_t* pfound = NULL;\n");
        self.w("            while (*pq) {\n");
        self.w("                if ((*pq)->completion == t) {\n");
        self.w("                    pfound = *pq; *pq = pfound->next;\n");
        // Tail tracking: if we removed the tail, new tail is the predecessor
        // (or NULL if we also removed the head — pprev would be NULL there).
        self.w("                    if (__maka_pool_q.tail == pfound) __maka_pool_q.tail = pprev;\n");
        self.w("                    break;\n");
        self.w("                }\n");
        self.w("                pprev = *pq;\n");
        self.w("                pq = &(*pq)->next;\n");
        self.w("            }\n");
        self.w("            pthread_mutex_unlock(&__maka_pool_q.lock);\n");
        self.w("            if (pfound) {\n");
        self.w("                pthread_mutex_lock(&t->done_mutex);\n");
        self.w("                t->done_flag = 1;\n");
        self.w("                pthread_cond_broadcast(&t->done_cond);\n");
        self.w("                pthread_mutex_unlock(&t->done_mutex);\n");
        self.w("                __maka_free_env(pfound->completion, pfound->entry_env); __maka_slab_free(pfound->slab); free(pfound);\n");
        self.w("                __maka_thread_unref(t);  /* runner-side ref */\n");
        self.w("            }\n");
        // Otherwise the fiber's body is currently running — cancel_requested
        // is already set; do NOT touch done_flag here.  Body checks the flag
        // at yield points; natural completion owns the flip.
        self.w("        }\n");
        self.w("    } else if (!t->is_job) {\n");
        self.w("        pthread_cancel(t->handle);\n");
        self.w("        if (!atomic_load(&t->is_failed)) pthread_join(t->handle, NULL);\n");
        // pthread cancellation is DEFERRED, and a compute body has no cancellation
        // points, so the thread almost always runs its epilogue to completion -
        // and the epilogue (__maka_handle_entry) sets done_flag=1 THEN drops the
        // runner-side ref.  Dropping it again here double-drops the refcount -> a
        // heap-use-after-free on the trailing spawner unref.  Only drop the runner
        // ref if the thread was actually cancelled BEFORE its epilogue ran
        // (done_flag still 0); pthread_join has fully reaped it so this read is
        // unsynchronized-safe.
        self.w("        if (!t->done_flag) __maka_thread_unref(t);  /* runner ref (cancelled early) */\n");
        self.w("    }\n");
        // Drop the spawner-side ref — caller is giving up the handle.
        self.w("    __maka_thread_unref(t);\n");
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
        self.w("               pthread_join.  Jobs: just wait - cancellation isn't supported. */\n");
        self.w("            for (int64_t j = 0; j < n; j++) {\n");
        self.w("                if (j == i) continue;\n");
        self.w("                Thread* l = (Thread*)handles[j];\n");
        self.w("                if (l->is_fiber) {\n");
        // Cross-thread: post a cancel onto the loser's home scheduler so it
        // walks its own queues; drop only the spawner ref here.
        self.w("                    maka_sched_state_t* lcand = (maka_sched_state_t*)atomic_load(&l->home_sched);\n");
        self.w("                    if (lcand && lcand != maka_sched_state) {\n");
        self.w("                        maka_sched_state_t* lhome = __maka_sched_validate_and_ref_epoch(lcand, atomic_load_explicit(&l->home_sched_epoch, memory_order_acquire));\n");
        self.w("                        if (lhome) {\n");
        self.w("                            __maka_remote_post_cancel(lhome, l);\n");
        self.w("                            __maka_sched_state_unref(lhome);\n");
        self.w("                        } else {\n");
        self.w("                            pthread_mutex_lock(&l->done_mutex);\n");
        self.w("                            l->done_flag = 1;\n");
        self.w("                            pthread_cond_broadcast(&l->done_cond);\n");
        self.w("                            pthread_mutex_unlock(&l->done_mutex);\n");
        self.w("                            __maka_thread_unref(l);  /* runner */\n");
        self.w("                        }\n");
        self.w("                        __maka_thread_unref(l);  /* spawner */\n");
        self.w("                        continue;\n");
        self.w("                    }\n");
        self.w("                    if (__maka_cancel_fiber_local(l)) {\n");
        // cancel_fiber_local drops runner ref; drop spawner ref here.
        self.w("                        __maka_thread_unref(l);\n");
        self.w("                    } else {\n");
        self.w("                        pthread_mutex_lock(&l->done_mutex);\n");
        self.w("                        l->done_flag = 1;\n");
        self.w("                        pthread_cond_broadcast(&l->done_cond);\n");
        self.w("                        pthread_mutex_unlock(&l->done_mutex);\n");
        self.w("                        __maka_thread_unref(l);\n");
        self.w("                    }\n");
        self.w("                } else if (!l->is_job) {\n");
        self.w("                    pthread_cancel(l->handle);\n");
        self.w("                    if (!atomic_load(&l->is_failed)) pthread_join(l->handle, NULL);\n");
        // Runner + spawner: pthread_cancel skipped the runner's epilogue
        // unref, so drop both refs here.
        self.w("                    __maka_thread_unref(l);\n");
        self.w("                    __maka_thread_unref(l);\n");
        self.w("                } else {\n");
        self.w("                    (void)__maka_join_result(handles[j]);\n");
        self.w("                }\n");
        self.w("            }\n");
        self.w("            /* Reap the winner. */\n");
        self.w("            (void)__maka_join_result(handles[i]);\n");
        self.w("            return r;\n");
        self.w("        }\n");
        self.w("        /* No winner yet - drive scheduler so fibers can progress. */\n");
        // Non-anchor fiber: yield as a fiber (see join_timeout) instead of driving
        // through the anchor context slot.
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("            __maka_yield_now();\n");
        self.w("        } else if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head)) {\n");
        self.w("            maka_anchor_wake_on_finish = 1;\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
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
        // Non-anchor fiber: yield as a fiber (see join_timeout) instead of driving
        // through the anchor context slot.
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("            __maka_yield_now();\n");
        self.w("        } else if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            maka_anchor_wake_on_finish = 1;\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
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
        self.w("        if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_reduce_chunk_t* ch = (__maka_reduce_chunk_t*)malloc(sizeof(__maka_reduce_chunk_t));\n");
        self.w("        ch->start = start + c * per;\n");
        self.w("        ch->end = ch->start + per; if (ch->end > end) ch->end = end;\n");
        self.w("        ch->combine = (int64_t(*)(void*, int64_t, int64_t))code;\n");
        self.w("        ch->env = env;\n");
        self.w("        ch->partial = 0;\n");
        self.w("        ch->completion = th;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_reduce_chunk_entry, ch, th);\n");
        self.w("        handles[c] = th; chunks_arr[c] = ch;\n");
        self.w("    }\n");
        self.w("    int64_t acc = init;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&handles[c]->is_failed)) pthread_join(handles[c]->handle, NULL);\n");
        self.w("        /* Combine partial using the same combine function with two\n");
        self.w("           int args -- works for sum/max/min where the function\n");
        self.w("           is associative on integers regardless of the\n");
        self.w("           \"index\" position. */\n");
        self.w("        acc = chunks_arr[c]->combine(env, acc, chunks_arr[c]->partial);\n");
        self.w("        __maka_thread_unref(handles[c]);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
        self.w("    free(c);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("Slice_maka_int __maka_par_map_int(int64_t start, int64_t end, void* code, void* env) {\n");
        self.w("    if (start >= end) {\n");
        self.w("        Slice_maka_int empty = { .ptr = NULL, .len = 0 };\n");
        self.w("        return empty;\n");
        self.w("    }\n");
        self.w("    int64_t total = end - start;\n");
        // Guard the output-buffer size multiply: `total` is caller-controlled
        // (end-start), so 8*total can wrap size_t and under-allocate `out`,
        // letting the workers write past it (heap-buffer-overflow).  Mirror the
        // par_map_bytes __builtin_mul_overflow guards.
        self.w("    size_t __out_sz;\n");
        self.w("    if (__builtin_mul_overflow((size_t)total, sizeof(int64_t), &__out_sz)) {\n");
        self.w("        Slice_maka_int empty = { .ptr = NULL, .len = 0 };\n");
        self.w("        return empty;\n");
        self.w("    }\n");
        self.w("    int64_t* out = (int64_t*)malloc(__out_sz);\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1;\n");
        self.w("    if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > total) chunks = total;\n");
        self.w("    int64_t per = (total + chunks - 1) / chunks;\n");
        self.w("    Thread** handles = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        // Track each chunk arg so we can free it (and run inline on spawn
        // failure) instead of leaking it through the no-arg-free wrapper.
        self.w("    __maka_map_chunk_t** chs = (__maka_map_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_map_chunk_t* ch = (__maka_map_chunk_t*)malloc(sizeof(__maka_map_chunk_t));\n");
        self.w("        ch->start = start + c * per;\n");
        self.w("        ch->end = ch->start + per; if (ch->end > end) ch->end = end;\n");
        self.w("        ch->fn = (int64_t(*)(void*, int64_t))code;\n");
        self.w("        ch->env = env;\n");
        self.w("        ch->out = out - start;\n");
        self.w("        ch->completion = th;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        if (__maka_spawn_pthread(&th->handle, __maka_map_chunk_entry, ch, th) != 0) {\n");
        // EAGAIN: run the chunk inline so out[] doesn't have uninitialized
        // slots; the wrapper already marked Thread done + dropped runner ref.
        self.w("            for (int64_t i = ch->start; i < ch->end; i++) ch->out[i] = ch->fn(ch->env, i);\n");
        self.w("        }\n");
        self.w("        handles[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        int failed = atomic_load(&handles[c]->is_failed);\n");
        self.w("        if (!failed) pthread_join(handles[c]->handle, NULL);\n");
        // Worker frees its own ch on success; on failure free now.
        self.w("        if (failed) free(chs[c]);\n");
        self.w("        __maka_thread_unref(handles[c]);\n");
        self.w("    }\n");
        self.w("    free(handles); free(chs);\n");
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
        self.w("    __maka_par_chunk_t** chs = (__maka_par_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_par_chunk_t* ch = (__maka_par_chunk_t*)malloc(sizeof(__maka_par_chunk_t));\n");
        self.w("        ch->start = start + c * per;\n");
        self.w("        ch->end = ch->start + per; if (ch->end > end) ch->end = end;\n");
        self.w("        ch->code = (void(*)(void*, int64_t))code;\n");
        self.w("        ch->env = env;\n");
        self.w("        ch->completion = th;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        if (__maka_spawn_pthread(&th->handle, __maka_par_chunk_entry, ch, th) != 0) {\n");
        // EAGAIN: run chunk inline.
        self.w("            for (int64_t i = ch->start; i < ch->end; i++) ch->code(ch->env, i);\n");
        self.w("        }\n");
        self.w("        handles[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        int failed = atomic_load(&handles[c]->is_failed);\n");
        self.w("        (void)__maka_join_result((maka_unit*)handles[c]);\n");
        // Worker frees ch on success; on failure free now.
        self.w("        if (failed) free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(handles); free(chs);\n");
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
        self.w("    pthread_cond_t  drained_cv;\n");
        self.w("    int             pth_waiters;\n");
        // Strategy Y: parkers armed but not yet pushed by scheduler's
        // finalize-park.  destroy waits for this to reach 0 alongside
        // pth_waiters + fiber_waiters before tearing the primitive down.
        self.w("    _Atomic int     inflight_parkers;\n");
        self.w("    maka_fiber_t*   fiber_waiters;\n");
        self.w("} maka_fmutex_t;\n");
        self.w("maka_unit* maka_fmutex_new(void) {\n");
        self.w("    maka_fmutex_t* m = (maka_fmutex_t*)calloc(1, sizeof(maka_fmutex_t));\n");
        self.w("    atomic_init(&m->locked, 0);\n");
        self.w("    pthread_mutex_init(&m->kw_mu, NULL);\n");
        self.w("    pthread_cond_init(&m->kw_cv, NULL);\n");
        self.w("    pthread_cond_init(&m->drained_cv, NULL);\n");
        self.w("    return (maka_unit*)m;\n");
        self.w("}\n");
        self.w("static int __maka_pp_fmutex(void* p) { return atomic_load(&((maka_fmutex_t*)p)->locked) != 0; }\n");
        self.w("void maka_fmutex_lock(maka_unit* p) {\n");
        self.w("    maka_fmutex_t* m = (maka_fmutex_t*)p;\n");
        self.w("    while (1) {\n");
        self.w("        int expected = 0;\n");
        self.w("        if (atomic_compare_exchange_strong(&m->locked, &expected, 1)) return;\n");
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        // Strategy Y + in-flight counter: take kw_mu, re-check predicate, bump
        // inflight_parkers UNDER the lock so destroy's wait sees it before we
        // release.  Then drop lock, arm the park request, yield.  Scheduler's
        // finalize-park does the push (or skip + ready-re-enqueue) and the
        // decrement under kw_mu.  Destroy waits on drained_cv until
        // inflight_parkers reaches 0 — no parker-released-then-finalize-park
        // window where destroy can sneak past and free the primitive.
        self.w("            pthread_mutex_lock(&m->kw_mu);\n");
        self.w("            if (atomic_load(&m->locked) == 0) { pthread_mutex_unlock(&m->kw_mu); continue; }\n");
        self.w("            atomic_fetch_add(&m->inflight_parkers, 1);\n");
        self.w("            pthread_mutex_unlock(&m->kw_mu);\n");
        self.w("            maka_fiber_t* me = maka_current_fiber;\n");
        self.w("            maka_park_req_t req = { .lock = &m->kw_mu, .head = &m->fiber_waiters,\n");
        self.w("                                    .should_park = __maka_pp_fmutex, .arg = m,\n");
        self.w("                                    .inflight = &m->inflight_parkers, .drained_cv = &m->drained_cv };\n");
        self.w("            maka_pending_park = &req;\n");
        self.w("            me->state = 2;\n");
        self.w("            maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("        } else if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            pthread_mutex_lock(&m->kw_mu);\n");
        self.w("            m->pth_waiters++;\n");
        self.w("            while (atomic_load(&m->locked) != 0) pthread_cond_wait(&m->kw_cv, &m->kw_mu);\n");
        self.w("            m->pth_waiters--;\n");
        self.w("            if (m->pth_waiters == 0) pthread_cond_signal(&m->drained_cv);\n");
        self.w("            pthread_mutex_unlock(&m->kw_mu);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("void maka_fmutex_unlock(maka_unit* p) {\n");
        self.w("    maka_fmutex_t* m = (maka_fmutex_t*)p;\n");
        // Take kw_mu BEFORE clearing locked so a concurrent push happens-before
        // the unlock observes the list, eliminating the missed-wake window.
        self.w("    pthread_mutex_lock(&m->kw_mu);\n");
        self.w("    atomic_store(&m->locked, 0);\n");
        self.w("    maka_fiber_t* w = m->fiber_waiters;\n");
        self.w("    if (w) { m->fiber_waiters = w->next_waiter; w->next_waiter = NULL; }\n");
        self.w("    pthread_cond_signal(&m->kw_cv);\n");
        self.w("    pthread_mutex_unlock(&m->kw_mu);\n");
        self.w("    if (w) __maka_ready_enqueue(w);\n");
        self.w("}\n");
        self.w("void maka_fmutex_destroy(maka_unit* p) {\n");
        self.w("    maka_fmutex_t* m = (maka_fmutex_t*)p;\n");
        self.w("    pthread_mutex_lock(&m->kw_mu);\n");
        // Zero the predicate so any broadcast-woken pthread waiter exits
        // its while-loop, and any Strategy Y finalize-park sees should_park=0
        // and re-enqueues the parker as ready instead of pushing.
        self.w("    atomic_store(&m->locked, 0);\n");
        self.w("    pthread_cond_broadcast(&m->kw_cv);\n");
        // Drain-loop: pull current fiber_waiters under lock, release, enqueue,
        // re-acquire; then wait on drained_cv if there are still pth_waiters
        // OR in-flight Strategy Y parkers that haven't finalized yet.
        // Loop re-runs because in-flight parkers can push fresh waiters as
        // they finalize (even though predicate is false now — race with the
        // parker's bump-before-arm).  Wait protected by drained_cv signals
        // from fmutex_unlock pth_waiter drain AND finalize-park decrement.
        self.w("    while (1) {\n");
        self.w("        maka_fiber_t* w = m->fiber_waiters; m->fiber_waiters = NULL;\n");
        self.w("        if (w) {\n");
        self.w("            pthread_mutex_unlock(&m->kw_mu);\n");
        self.w("            while (w) { maka_fiber_t* nx = w->next_waiter; w->next_waiter = NULL; __maka_ready_enqueue(w); w = nx; }\n");
        self.w("            pthread_mutex_lock(&m->kw_mu);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        if (m->pth_waiters > 0 || atomic_load(&m->inflight_parkers) > 0) {\n");
        self.w("            pthread_cond_wait(&m->drained_cv, &m->kw_mu);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        break;\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&m->kw_mu);\n");
        self.w("    pthread_mutex_destroy(&m->kw_mu); pthread_cond_destroy(&m->kw_cv); pthread_cond_destroy(&m->drained_cv);\n");
        self.w("    free(m);\n");
        self.w("}\n");
        // Fiber-aware RwLock (forward-declared up with the raw locks).  A single
        // unlock disambiguates read vs write by the writer flag: a write lock
        // means writer=1 & readers=0, a read lock means writer=0 & readers>0, so
        // they never overlap.  On unlock we wake ALL waiters and let each re-check
        // its own predicate (readers block on writer!=0; writers block on
        // writer!=0 || readers!=0) - simple and deadlock-free.
        self.w("typedef struct {\n");
        self.w("    _Atomic int     writer;\n");
        self.w("    _Atomic int     readers;\n");
        self.w("    pthread_mutex_t kw_mu;\n");
        self.w("    pthread_cond_t  kw_cv;\n");
        self.w("    pthread_cond_t  drained_cv;\n");
        self.w("    int             pth_waiters;\n");
        self.w("    _Atomic int     inflight_parkers;\n");
        self.w("    maka_fiber_t*   fiber_waiters;\n");
        self.w("} maka_rwlock_t;\n");
        self.w("maka_unit* maka_rwlock_new(void) {\n");
        self.w("    maka_rwlock_t* r = (maka_rwlock_t*)calloc(1, sizeof(maka_rwlock_t));\n");
        self.w("    atomic_init(&r->writer, 0); atomic_init(&r->readers, 0);\n");
        self.w("    pthread_mutex_init(&r->kw_mu, NULL);\n");
        self.w("    pthread_cond_init(&r->kw_cv, NULL);\n");
        self.w("    pthread_cond_init(&r->drained_cv, NULL);\n");
        self.w("    return (maka_unit*)r;\n");
        self.w("}\n");
        self.w("static int __maka_pp_rwlock_rd(void* p) { return atomic_load(&((maka_rwlock_t*)p)->writer) != 0; }\n");
        self.w("static int __maka_pp_rwlock_wr(void* p) { maka_rwlock_t* r = (maka_rwlock_t*)p; return atomic_load(&r->writer) != 0 || atomic_load(&r->readers) != 0; }\n");
        self.w("void maka_rwlock_read_lock(maka_unit* p) {\n");
        self.w("    maka_rwlock_t* r = (maka_rwlock_t*)p;\n");
        self.w("    while (1) {\n");
        self.w("        pthread_mutex_lock(&r->kw_mu);\n");
        self.w("        if (atomic_load(&r->writer) == 0) { atomic_fetch_add(&r->readers, 1); pthread_mutex_unlock(&r->kw_mu); return; }\n");
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("            atomic_fetch_add(&r->inflight_parkers, 1);\n");
        self.w("            pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("            maka_fiber_t* me = maka_current_fiber;\n");
        self.w("            maka_park_req_t req = { .lock = &r->kw_mu, .head = &r->fiber_waiters,\n");
        self.w("                                    .should_park = __maka_pp_rwlock_rd, .arg = r,\n");
        self.w("                                    .inflight = &r->inflight_parkers, .drained_cv = &r->drained_cv };\n");
        self.w("            maka_pending_park = &req;\n");
        self.w("            me->state = 2;\n");
        self.w("            maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("        } else if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            r->pth_waiters++;\n");
        self.w("            while (atomic_load(&r->writer) != 0) pthread_cond_wait(&r->kw_cv, &r->kw_mu);\n");
        self.w("            r->pth_waiters--;\n");
        self.w("            if (r->pth_waiters == 0) pthread_cond_signal(&r->drained_cv);\n");
        self.w("            pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("void maka_rwlock_write_lock(maka_unit* p) {\n");
        self.w("    maka_rwlock_t* r = (maka_rwlock_t*)p;\n");
        self.w("    while (1) {\n");
        self.w("        pthread_mutex_lock(&r->kw_mu);\n");
        self.w("        if (atomic_load(&r->writer) == 0 && atomic_load(&r->readers) == 0) { atomic_store(&r->writer, 1); pthread_mutex_unlock(&r->kw_mu); return; }\n");
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("            atomic_fetch_add(&r->inflight_parkers, 1);\n");
        self.w("            pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("            maka_fiber_t* me = maka_current_fiber;\n");
        self.w("            maka_park_req_t req = { .lock = &r->kw_mu, .head = &r->fiber_waiters,\n");
        self.w("                                    .should_park = __maka_pp_rwlock_wr, .arg = r,\n");
        self.w("                                    .inflight = &r->inflight_parkers, .drained_cv = &r->drained_cv };\n");
        self.w("            maka_pending_park = &req;\n");
        self.w("            me->state = 2;\n");
        self.w("            maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("        } else if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            r->pth_waiters++;\n");
        self.w("            while (atomic_load(&r->writer) != 0 || atomic_load(&r->readers) != 0) pthread_cond_wait(&r->kw_cv, &r->kw_mu);\n");
        self.w("            r->pth_waiters--;\n");
        self.w("            if (r->pth_waiters == 0) pthread_cond_signal(&r->drained_cv);\n");
        self.w("            pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("void maka_rwlock_unlock(maka_unit* p) {\n");
        self.w("    maka_rwlock_t* r = (maka_rwlock_t*)p;\n");
        self.w("    pthread_mutex_lock(&r->kw_mu);\n");
        self.w("    if (atomic_load(&r->writer)) atomic_store(&r->writer, 0);\n");
        self.w("    else atomic_fetch_sub(&r->readers, 1);\n");
        self.w("    maka_fiber_t* w = r->fiber_waiters; r->fiber_waiters = NULL;\n");
        self.w("    pthread_cond_broadcast(&r->kw_cv);\n");
        self.w("    pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("    while (w) { maka_fiber_t* nx = w->next_waiter; w->next_waiter = NULL; __maka_ready_enqueue(w); w = nx; }\n");
        self.w("}\n");
        self.w("void maka_rwlock_destroy(maka_unit* p) {\n");
        self.w("    maka_rwlock_t* r = (maka_rwlock_t*)p;\n");
        self.w("    pthread_mutex_lock(&r->kw_mu);\n");
        self.w("    atomic_store(&r->writer, 0); atomic_store(&r->readers, 0);\n");
        self.w("    pthread_cond_broadcast(&r->kw_cv);\n");
        self.w("    while (1) {\n");
        self.w("        maka_fiber_t* w = r->fiber_waiters; r->fiber_waiters = NULL;\n");
        self.w("        if (w) {\n");
        self.w("            pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("            while (w) { maka_fiber_t* nx = w->next_waiter; w->next_waiter = NULL; __maka_ready_enqueue(w); w = nx; }\n");
        self.w("            pthread_mutex_lock(&r->kw_mu);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        if (r->pth_waiters > 0 || atomic_load(&r->inflight_parkers) > 0) {\n");
        self.w("            pthread_cond_wait(&r->drained_cv, &r->kw_mu);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        break;\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&r->kw_mu);\n");
        self.w("    pthread_mutex_destroy(&r->kw_mu); pthread_cond_destroy(&r->kw_cv); pthread_cond_destroy(&r->drained_cv);\n");
        self.w("    free(r);\n");
        self.w("}\n");
        // WaitGroup.
        self.w("typedef struct {\n");
        self.w("    _Atomic int64_t count;\n");
        self.w("    pthread_mutex_t kw_mu;\n");
        self.w("    pthread_cond_t  kw_cv;\n");
        self.w("    pthread_cond_t  drained_cv;\n");
        self.w("    int             pth_waiters;\n");
        self.w("    _Atomic int     inflight_parkers;\n");
        self.w("    maka_fiber_t*   fiber_waiters;\n");
        self.w("} maka_wg_t;\n");
        self.w("maka_unit* maka_wg_new(void) {\n");
        self.w("    maka_wg_t* w = (maka_wg_t*)calloc(1, sizeof(maka_wg_t));\n");
        self.w("    atomic_init(&w->count, 0);\n");
        self.w("    pthread_mutex_init(&w->kw_mu, NULL);\n");
        self.w("    pthread_cond_init(&w->kw_cv, NULL);\n");
        self.w("    pthread_cond_init(&w->drained_cv, NULL);\n");
        self.w("    return (maka_unit*)w;\n");
        self.w("}\n");
        self.w("void maka_wg_add(maka_unit* p, int64_t n) {\n");
        self.w("    atomic_fetch_add(&((maka_wg_t*)p)->count, n);\n");
        self.w("}\n");
        self.w("void maka_wg_done(maka_unit* p) {\n");
        self.w("    maka_wg_t* w = (maka_wg_t*)p;\n");
        self.w("    int64_t prev = atomic_fetch_sub(&w->count, 1);\n");
        self.w("    if (prev <= 1) {\n");
        // Snapshot under kw_mu so concurrent wg_wait pushes don't corrupt.
        self.w("        pthread_mutex_lock(&w->kw_mu);\n");
        self.w("        maka_fiber_t* head = w->fiber_waiters;\n");
        self.w("        w->fiber_waiters = NULL;\n");
        self.w("        pthread_cond_broadcast(&w->kw_cv);\n");
        self.w("        pthread_mutex_unlock(&w->kw_mu);\n");
        self.w("        while (head) {\n");
        self.w("            maka_fiber_t* nx = head->next_waiter; head->next_waiter = NULL;\n");
        self.w("            __maka_ready_enqueue(head); head = nx;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("static int __maka_pp_wg(void* p) { return atomic_load(&((maka_wg_t*)p)->count) > 0; }\n");
        self.w("void maka_wg_wait(maka_unit* p) {\n");
        self.w("    maka_wg_t* w = (maka_wg_t*)p;\n");
        self.w("    while (atomic_load(&w->count) > 0) {\n");
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        // Strategy Y + in-flight counter (see fmutex_lock for full rationale).
        self.w("            pthread_mutex_lock(&w->kw_mu);\n");
        self.w("            if (atomic_load(&w->count) == 0) { pthread_mutex_unlock(&w->kw_mu); break; }\n");
        self.w("            atomic_fetch_add(&w->inflight_parkers, 1);\n");
        self.w("            pthread_mutex_unlock(&w->kw_mu);\n");
        self.w("            maka_fiber_t* me = maka_current_fiber;\n");
        self.w("            maka_park_req_t req = { .lock = &w->kw_mu, .head = &w->fiber_waiters,\n");
        self.w("                                    .should_park = __maka_pp_wg, .arg = w,\n");
        self.w("                                    .inflight = &w->inflight_parkers, .drained_cv = &w->drained_cv };\n");
        self.w("            maka_pending_park = &req;\n");
        self.w("            me->state = 2;\n");
        self.w("            maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("        } else if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            /* On the anchor: drive the scheduler so fibers can complete\n");
        self.w("               and call wg_done.  pthread_cond_wait would freeze them. */\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            pthread_mutex_lock(&w->kw_mu);\n");
        self.w("            w->pth_waiters++;\n");
        self.w("            while (atomic_load(&w->count) > 0) pthread_cond_wait(&w->kw_cv, &w->kw_mu);\n");
        self.w("            w->pth_waiters--;\n");
        self.w("            if (w->pth_waiters == 0) pthread_cond_signal(&w->drained_cv);\n");
        self.w("            pthread_mutex_unlock(&w->kw_mu);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("void maka_wg_destroy(maka_unit* p) {\n");
        self.w("    maka_wg_t* w = (maka_wg_t*)p;\n");
        self.w("    pthread_mutex_lock(&w->kw_mu);\n");
        // Zero the predicate so broadcast-woken pthread waiters exit AND
        // Strategy Y finalize-park skips the push.  Then drain-loop with
        // in-flight-aware wait (see fmutex_destroy for rationale).
        self.w("    atomic_store(&w->count, 0);\n");
        self.w("    pthread_cond_broadcast(&w->kw_cv);\n");
        self.w("    while (1) {\n");
        self.w("        maka_fiber_t* head = w->fiber_waiters; w->fiber_waiters = NULL;\n");
        self.w("        if (head) {\n");
        self.w("            pthread_mutex_unlock(&w->kw_mu);\n");
        self.w("            while (head) { maka_fiber_t* nx = head->next_waiter; head->next_waiter = NULL; __maka_ready_enqueue(head); head = nx; }\n");
        self.w("            pthread_mutex_lock(&w->kw_mu);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        if (w->pth_waiters > 0 || atomic_load(&w->inflight_parkers) > 0) {\n");
        self.w("            pthread_cond_wait(&w->drained_cv, &w->kw_mu);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        break;\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&w->kw_mu);\n");
        self.w("    pthread_mutex_destroy(&w->kw_mu); pthread_cond_destroy(&w->kw_cv); pthread_cond_destroy(&w->drained_cv);\n");
        self.w("    free(w);\n");
        self.w("}\n");
        // Once.
        self.w("typedef struct {\n");
        self.w("    _Atomic int state;\n");
        self.w("    _Atomic int runner_in_flight;\n");
        self.w("    _Atomic int inflight_parkers;\n");
        self.w("    pthread_mutex_t mu;\n");
        self.w("    pthread_cond_t  cv;\n");
        self.w("    pthread_cond_t  drained_cv;\n");
        self.w("    int             waiters;\n");
        self.w("    maka_fiber_t*   fiber_waiters;\n");
        self.w("} maka_once_t;\n");
        self.w("maka_unit* maka_once_new(void) {\n");
        self.w("    maka_once_t* o = (maka_once_t*)calloc(1, sizeof(maka_once_t));\n");
        self.w("    atomic_init(&o->state, 0);\n");
        self.w("    pthread_mutex_init(&o->mu, NULL);\n");
        self.w("    pthread_cond_init(&o->cv, NULL);\n");
        self.w("    pthread_cond_init(&o->drained_cv, NULL);\n");
        self.w("    return (maka_unit*)o;\n");
        self.w("}\n");
        self.w("static int __maka_pp_once(void* p) { return atomic_load(&((maka_once_t*)p)->state) != 2; }\n");
        self.w("void maka_once_do(maka_unit* p, void* code, void* env) {\n");
        self.w("    maka_once_t* o = (maka_once_t*)p;\n");
        self.w("    int expected = 0;\n");
        self.w("    if (atomic_compare_exchange_strong(&o->state, &expected, 1)) {\n");
        // Bump runner_in_flight before invoking the body so destroy can't
        // tear down the mutex/cv while we still need them for publish.
        self.w("        atomic_fetch_add(&o->runner_in_flight, 1);\n");
        self.w("        ((void(*)(void*))code)(env);\n");
        self.w("        pthread_mutex_lock(&o->mu);\n");
        self.w("        atomic_store(&o->state, 2);\n");
        self.w("        maka_fiber_t* head = o->fiber_waiters; o->fiber_waiters = NULL;\n");
        self.w("        pthread_cond_broadcast(&o->cv);\n");
        self.w("        atomic_fetch_sub(&o->runner_in_flight, 1);\n");
        self.w("        pthread_cond_signal(&o->drained_cv);\n");
        self.w("        pthread_mutex_unlock(&o->mu);\n");
        // Drain the waiter list outside the lock to avoid lock-order issues
        // with __maka_ready_enqueue (which may take remote_mu on cross-thread).
        self.w("        while (head) { maka_fiber_t* nx = head->next_waiter; head->next_waiter = NULL; __maka_ready_enqueue(head); head = nx; }\n");
        self.w("        return;\n");
        self.w("    }\n");
        // Late caller: park on per-once waiter list (NOT busy-yield).  The
        // winner's broadcast drains the list when state flips to 2.
        // Strategy Y + in-flight counter (see fmutex_lock for full rationale).
        self.w("    if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("        pthread_mutex_lock(&o->mu);\n");
        self.w("        if (atomic_load(&o->state) == 2) { pthread_mutex_unlock(&o->mu); return; }\n");
        self.w("        atomic_fetch_add(&o->inflight_parkers, 1);\n");
        self.w("        pthread_mutex_unlock(&o->mu);\n");
        self.w("        maka_fiber_t* me = maka_current_fiber;\n");
        self.w("        maka_park_req_t req = { .lock = &o->mu, .head = &o->fiber_waiters,\n");
        self.w("                                .should_park = __maka_pp_once, .arg = o,\n");
        self.w("                                .inflight = &o->inflight_parkers, .drained_cv = &o->drained_cv };\n");
        self.w("        maka_pending_park = &req;\n");
        self.w("        me->state = 2;\n");
        self.w("        maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("        return;\n");
        self.w("    }\n");
        // Anchor (or no scheduler): the once RUNNER may be a spawned, co-resident
        // fiber that only the scheduler can resume, so we must DRIVE the scheduler
        // while there is schedulable work rather than freezing it in a plain
        // cond_wait - that would deadlock (the runner never resumes to publish
        // state=2).  Mirrors maka_fmutex_lock / maka_wg_wait.
        self.w("    while (atomic_load(&o->state) != 2) {\n");
        self.w("        if (maka_sched_inited && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("        } else {\n");
        self.w("            pthread_mutex_lock(&o->mu);\n");
        self.w("            if (atomic_load(&o->state) == 2) { pthread_mutex_unlock(&o->mu); break; }\n");
        self.w("            o->waiters++;\n");
        self.w("            pthread_cond_wait(&o->cv, &o->mu);\n");
        self.w("            o->waiters--;\n");
        self.w("            if (o->waiters == 0) pthread_cond_signal(&o->drained_cv);\n");
        self.w("            pthread_mutex_unlock(&o->mu);\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("}\n");
        self.w("void maka_once_destroy(maka_unit* p) {\n");
        self.w("    maka_once_t* o = (maka_once_t*)p;\n");
        // Force state→2 so any future cond_wait predicate exits AND Strategy
        // Y finalize-park sees should_park=0 (won't push).  Drain-loop with
        // in-flight-aware wait (see fmutex_destroy for rationale).
        self.w("    pthread_mutex_lock(&o->mu);\n");
        self.w("    atomic_store(&o->state, 2);\n");
        self.w("    pthread_cond_broadcast(&o->cv);\n");
        self.w("    while (1) {\n");
        self.w("        maka_fiber_t* head = o->fiber_waiters; o->fiber_waiters = NULL;\n");
        self.w("        if (head) {\n");
        self.w("            pthread_mutex_unlock(&o->mu);\n");
        self.w("            while (head) { maka_fiber_t* nx = head->next_waiter; head->next_waiter = NULL; __maka_ready_enqueue(head); head = nx; }\n");
        self.w("            pthread_mutex_lock(&o->mu);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        if (o->waiters > 0 || atomic_load(&o->runner_in_flight) > 0 || atomic_load(&o->inflight_parkers) > 0) {\n");
        self.w("            pthread_cond_wait(&o->drained_cv, &o->mu);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        break;\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&o->mu);\n");
        self.w("    pthread_mutex_destroy(&o->mu); pthread_cond_destroy(&o->cv); pthread_cond_destroy(&o->drained_cv);\n");
        self.w("    free(o);\n");
        self.w("}\n");
        // Fiber-aware byte-channel recv.  On the anchor with pending fiber
        // work, drive the scheduler in short bursts so other fibers can run
        // (and possibly post to the channel) instead of blocking the worker.
        // Fiber-park predicate + waker for the byte channel (defined here, after the
        // scheduler, so they can touch maka_fiber_t / __maka_ready_enqueue).
        self.w("static int __maka_pp_bchan(void* p) { maka_bchan_t* c = (maka_bchan_t*)p; return !c->head && !c->closed; }\n");
        self.w("void __maka_bchan_wake(maka_bchan_t* c, int all) {\n");
        self.w("    pthread_mutex_lock(&c->m);\n");
        self.w("    maka_fiber_t* w = c->fiber_waiters;\n");
        self.w("    if (all) { c->fiber_waiters = NULL; }\n");
        self.w("    else if (w) { c->fiber_waiters = w->next_waiter; w->next_waiter = NULL; }\n");
        self.w("    pthread_mutex_unlock(&c->m);\n");
        self.w("    while (w) { maka_fiber_t* nx = all ? w->next_waiter : NULL; w->next_waiter = NULL; __maka_ready_enqueue(w); w = nx; }\n");
        self.w("}\n");
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
        // A spawned (non-anchor) fiber must NOT pthread_cond_wait here - that blocks
        // the whole OS thread / scheduler, so a sibling producer fiber never runs
        // (deadlock).  Park onto fiber_waiters and yield to the scheduler instead;
        // send/close re-enqueue us as ready.  Mirrors maka_fmutex_lock's park branch.
        self.w("        if (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) {\n");
        self.w("            atomic_fetch_add(&c->inflight_parkers, 1);\n");
        self.w("            pthread_mutex_unlock(&c->m);\n");
        self.w("            maka_fiber_t* me = maka_current_fiber;\n");
        self.w("            maka_park_req_t req = { .lock = &c->m, .head = &c->fiber_waiters,\n");
        self.w("                                    .should_park = __maka_pp_bchan, .arg = c,\n");
        self.w("                                    .inflight = &c->inflight_parkers, .drained_cv = &c->drained_cv };\n");
        self.w("            maka_pending_park = &req;\n");
        self.w("            me->state = 2;\n");
        self.w("            maka_ctx_switch(&me->ctx, &maka_sched_ctx);\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        pthread_mutex_unlock(&c->m);\n");
        self.w("        if (maka_sched_inited && maka_current_fiber == maka_anchor_fiber\n");
        self.w("            && (__maka_have_ready() || maka_sleep_head || maka_fd_waiters)) {\n");
        self.w("            maka_ctx_switch(&maka_anchor_fiber->ctx, &maka_sched_ctx);\n");
        self.w("            maka_current_fiber = maka_anchor_fiber;\n");
        self.w("            continue;\n");
        self.w("        }\n");
        self.w("        pthread_mutex_lock(&c->m);\n");
        self.w("        c->waiters++;\n");
        self.w("        while (!c->head && !c->closed) pthread_cond_wait(&c->c, &c->m);\n");
        // Handle the wake WHILE STILL HOLDING c->m and RETURN - do NOT decrement
        // waiters, unlock, then loop back to re-acquire c: a concurrent destroy
        // (whose drain condition is satisfied the instant waiters hits 0) would
        // free(c) in that gap, and the re-lock/re-read would be a use-after-free.
        self.w("        maka_bnode_t* wn = c->head;\n");
        self.w("        if (wn) {\n");
        self.w("            c->head = wn->next; if (!c->head) c->tail = NULL;\n");
        self.w("            c->count--;\n");
        self.w("            memcpy((void*)dst, wn->data, (size_t)c->item_size);\n");
        self.w("        } else {\n");
        self.w("            memset((void*)dst, 0, (size_t)c->item_size);\n");
        self.w("        }\n");
        self.w("        c->waiters--;\n");
        self.w("        if (c->waiters == 0) pthread_cond_signal(&c->drained_cv);\n");
        self.w("        pthread_mutex_unlock(&c->m);\n");
        self.w("        if (wn) free(wn);\n");
        self.w("        return;\n");
        self.w("    }\n");
        self.w("}\n");
        // Non-blocking dequeue: takes c->m ONCE, pops head if present (returns 1),
        // else returns 0 without ever waiting.  The try_recv shims must route
        // through this - doing count()>0 then a separate blocking recv() is a
        // TOCTOU: with >1 consumer another can steal the item in the gap, so the
        // loser's recv() blocks forever on an empty-but-open channel.
        self.w("int64_t maka_chan_bytes_try_recv(maka_unit* p, maka_unit* dst) {\n");
        self.w("    maka_bchan_t* c = (maka_bchan_t*)p;\n");
        self.w("    pthread_mutex_lock(&c->m);\n");
        self.w("    maka_bnode_t* n = c->head;\n");
        self.w("    if (n) {\n");
        self.w("        c->head = n->next; if (!c->head) c->tail = NULL;\n");
        self.w("        c->count--;\n");
        self.w("        memcpy((void*)dst, n->data, (size_t)c->item_size);\n");
        self.w("        pthread_mutex_unlock(&c->m);\n");
        self.w("        free(n);\n");
        self.w("        return 1;\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&c->m);\n");
        self.w("    return 0;\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.body = (void(*)(void*, int64_t))code;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_each_entry, ch, th);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = out; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.fn = (int64_t(*)(void*, int64_t))code;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_map_slice_entry, ch, th);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->env = env; ch->completion = th; ch->init = init;\n");
        self.w("        ch->code.combine = (int64_t(*)(void*, int64_t, int64_t))code;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_reduce_slice_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    int64_t acc = init;\n");
        self.w("    int64_t (*combine)(void*, int64_t, int64_t) = (int64_t(*)(void*, int64_t, int64_t))code;\n");
        self.w("    int merged_first = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        if (!merged_first) { acc = chs[c]->out_acc; merged_first = 1; }\n");
        self.w("        else { acc = combine(env, acc, chs[c]->out_acc); }\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("        free(chs[c]);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = tmp + ch->i_start;\n");
        self.w("        ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.pred = (int(*)(void*, int64_t))code;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_filter_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    int64_t total = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        total += chs[c]->out_len;\n");
        self.w("    }\n");
        self.w("    int64_t* out = (int64_t*)malloc(sizeof(int64_t) * (size_t)(total > 0 ? total : 1));\n");
        self.w("    int64_t w = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        memcpy(out + w, tmp + chs[c]->i_start, sizeof(int64_t) * (size_t)chs[c]->out_len);\n");
        self.w("        w += chs[c]->out_len;\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("        free(chs[c]);\n");
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
        self.w("            double (*combine)(void*, double, double);\n");
        self.w("            int  (*pred)(void*, double); } code;\n");
        self.w("    double init;\n");
        self.w("    double out_acc;\n");
        self.w("    int64_t out_len;\n");
        self.w("} __maka_fslice_chunk_t;\n");

        self.w("static void* __maka_par_each_f_entry(void* arg) {\n");
        self.w("    __maka_fslice_chunk_t* c = (__maka_fslice_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) c->code.body(c->env, c->in_ptr[i]);\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_fslice_chunk_t* ch = (__maka_fslice_chunk_t*)calloc(1, sizeof(__maka_fslice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.body = (void(*)(void*, double))code;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_each_f_entry, ch, th);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_fslice_chunk_t* ch = (__maka_fslice_chunk_t*)calloc(1, sizeof(__maka_fslice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = out; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.fn = (double(*)(void*, double))code;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_map_f_entry, ch, th);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_fslice_chunk_t* ch = (__maka_fslice_chunk_t*)calloc(1, sizeof(__maka_fslice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->env = env; ch->completion = th; ch->init = init;\n");
        self.w("        ch->code.combine = (double(*)(void*, double, double))code;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_reduce_f_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    double acc = init;\n");
        self.w("    double (*combine)(void*, double, double) = (double(*)(void*, double, double))code;\n");
        self.w("    int merged_first = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        if (!merged_first) { acc = chs[c]->out_acc; merged_first = 1; }\n");
        self.w("        else { acc = combine(env, acc, chs[c]->out_acc); }\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("        free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(hs); free(chs);\n");
        self.w("    return acc;\n");
        self.w("}\n");

        // par_filter_float: same 2-pass design as par_filter_int, but predicate
        // takes double and the input/output slices are float.
        self.w("static void* __maka_par_filter_f_entry(void* arg) {\n");
        self.w("    __maka_fslice_chunk_t* c = (__maka_fslice_chunk_t*)arg;\n");
        self.w("    int64_t w = 0;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) {\n");
        self.w("        if (c->code.pred(c->env, c->in_ptr[i])) c->out_ptr[w++] = c->in_ptr[i];\n");
        self.w("    }\n");
        self.w("    c->out_len = w;\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    __maka_thread_unref(c->completion);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("Slice_maka_float __maka_par_filter_float(Slice_maka_float s, void* code, void* env) {\n");
        self.w("    Slice_maka_float empty = { .ptr = NULL, .len = 0 };\n");
        self.w("    if (s.len <= 0) return empty;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    double* tmp = (double*)malloc(sizeof(double) * (size_t)s.len);\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_fslice_chunk_t** chs = (__maka_fslice_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_fslice_chunk_t* ch = (__maka_fslice_chunk_t*)calloc(1, sizeof(__maka_fslice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = tmp + ch->i_start;\n");
        self.w("        ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.pred = (int(*)(void*, double))code;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_filter_f_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    int64_t total = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        total += chs[c]->out_len;\n");
        self.w("    }\n");
        self.w("    double* out = (double*)malloc(sizeof(double) * (size_t)(total > 0 ? total : 1));\n");
        self.w("    int64_t w = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        memcpy(out + w, tmp + chs[c]->i_start, sizeof(double) * (size_t)chs[c]->out_len);\n");
        self.w("        w += chs[c]->out_len;\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("        free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(hs); free(chs); free(tmp);\n");
        self.w("    Slice_maka_float res = { .ptr = out, .len = total };\n");
        self.w("    return res;\n");
        self.w("}\n");

        // par_scan_float: same 2-pass scan as par_scan_int.  Local pass
        // saves the chunk's tail in out_acc; offset pass runs a sequential
        // combine over those tails to derive each chunk's offset, then
        // folds the offset back across the chunk's prefix.
        self.w("static void* __maka_par_scan_f_local_entry(void* arg) {\n");
        self.w("    __maka_fslice_chunk_t* c = (__maka_fslice_chunk_t*)arg;\n");
        self.w("    if (c->i_start < c->i_end) {\n");
        self.w("        double acc = c->in_ptr[c->i_start];\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static void* __maka_par_scan_f_offset_entry(void* arg) {\n");
        self.w("    __maka_fslice_chunk_t* c = (__maka_fslice_chunk_t*)arg;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) {\n");
        self.w("        c->out_ptr[i] = c->code.combine(c->env, c->init, c->out_ptr[i]);\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    __maka_thread_unref(c->completion);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("Slice_maka_float __maka_par_scan_float(Slice_maka_float s, void* code, void* env) {\n");
        self.w("    Slice_maka_float empty = { .ptr = NULL, .len = 0 };\n");
        self.w("    if (s.len <= 0) return empty;\n");
        self.w("    double* out = (double*)malloc(sizeof(double) * (size_t)s.len);\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > s.len) chunks = s.len;\n");
        self.w("    int64_t per = (s.len + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_fslice_chunk_t** chs = (__maka_fslice_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    double (*combine)(void*, double, double) = (double(*)(void*, double, double))code;\n");
        self.w("    /* pass 1 */\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_fslice_chunk_t* ch = (__maka_fslice_chunk_t*)calloc(1, sizeof(__maka_fslice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = out; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.combine = combine;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_scan_f_local_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    double* offsets = (double*)malloc(sizeof(double) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        offsets[c] = chs[c]->out_acc;\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("    }\n");
        self.w("    /* pass 2: running offsets across chunks (chunk 0 stays). */\n");
        self.w("    double running = offsets[0];\n");
        self.w("    for (int64_t c = 1; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        chs[c]->completion = th; chs[c]->init = running;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_scan_f_offset_entry, chs[c], th);\n");
        self.w("        hs[c] = th;\n");
        self.w("        running = combine(env, running, offsets[c]);\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 1; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("        free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(chs[0]);\n");
        self.w("    free(hs); free(chs); free(offsets);\n");
        self.w("    Slice_maka_float res = { .ptr = out, .len = s.len };\n");
        self.w("    return res;\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
        self.w("    free(c); return NULL;\n");
        self.w("}\n");
        // Returns a malloc'd buffer of (n * out_item_size) bytes; caller is
        // responsible for freeing (via free() since malloc was used).
        self.w("void* maka_par_map_bytes(void* in_ptr, int64_t n, int64_t in_sz, int64_t out_sz, void* code, void* env) {\n");
        self.w("    if (n <= 0 || out_sz <= 0 || in_sz < 0) return NULL;\n");
        // Checked multiplication: refuse to malloc when n * out_sz would
        // overflow int64_t (wrap to negative → giant size_t → heap corruption).
        self.w("    size_t total;\n");
        self.w("    if (__builtin_mul_overflow((size_t)n, (size_t)out_sz, &total)) return NULL;\n");
        self.w("    char* out = (char*)malloc(total);\n");
        self.w("    if (!out) return NULL;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > n) chunks = n;\n");
        self.w("    int64_t per = (n + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_bytes_chunk_t* ch = (__maka_bytes_chunk_t*)calloc(1, sizeof(__maka_bytes_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > n) ch->i_end = n;\n");
        self.w("        ch->in_ptr = (char*)in_ptr;\n");
        self.w("        ch->out_ptr = out;\n");
        self.w("        ch->in_sz = in_sz; ch->out_sz = out_sz;\n");
        self.w("        ch->env = env; ch->completion = th;\n");
        self.w("        ch->body = (void(*)(void*, void*, void*))code;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_map_bytes_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) (void)__maka_join_result((maka_unit*)hs[c]);\n");
        self.w("    free(hs);\n");
        self.w("    return (void*)out;\n");
        self.w("}\n");
        // par_filter_bytes: 2-pass parallel filter over an opaque-item buffer.
        // Pass 1: each chunk runs `pred(env, item)` over its slice and copies
        // the survivors into a per-chunk temp region.  Pass 2: compact the
        // survivors into a single output buffer.  out_n receives the count.
        self.w("typedef struct {\n");
        self.w("    int64_t i_start, i_end;\n");
        self.w("    char* in_ptr;\n");
        self.w("    char* tmp_ptr;\n");
        self.w("    int64_t item_sz;\n");
        self.w("    int (*pred)(void*, void*);\n");
        self.w("    void* env;\n");
        self.w("    int64_t out_count;\n");
        self.w("    Thread* completion;\n");
        self.w("} __maka_bytes_filter_chunk_t;\n");
        self.w("static void* __maka_par_filter_bytes_entry(void* arg) {\n");
        self.w("    __maka_bytes_filter_chunk_t* c = (__maka_bytes_filter_chunk_t*)arg;\n");
        self.w("    int64_t w = 0;\n");
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) {\n");
        self.w("        void* it = (void*)(c->in_ptr + i * c->item_sz);\n");
        self.w("        if (c->pred(c->env, it)) {\n");
        self.w("            memcpy(c->tmp_ptr + w * c->item_sz, it, (size_t)c->item_sz); w++;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    c->out_count = w;\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    __maka_thread_unref(c->completion);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("void* maka_par_filter_bytes(void* in_ptr, int64_t n, int64_t item_sz, int64_t* out_n, void* code, void* env) {\n");
        self.w("    *out_n = 0;\n");
        self.w("    if (n <= 0 || item_sz <= 0) return NULL;\n");
        self.w("    size_t tmp_total;\n");
        self.w("    if (__builtin_mul_overflow((size_t)n, (size_t)item_sz, &tmp_total)) return NULL;\n");
        // Per-chunk temp buffer is item_sz * per (max worst-case survivors per
        // chunk).  After parallel pass, we walk chunks and copy survivors into
        // the final packed buffer.
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > n) chunks = n;\n");
        self.w("    int64_t per = (n + chunks - 1) / chunks;\n");
        self.w("    char* tmp = (char*)malloc(tmp_total);\n");
        self.w("    if (!tmp) return NULL;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_bytes_filter_chunk_t** chs = (__maka_bytes_filter_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_bytes_filter_chunk_t* ch = (__maka_bytes_filter_chunk_t*)calloc(1, sizeof(__maka_bytes_filter_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > n) ch->i_end = n;\n");
        self.w("        ch->in_ptr = (char*)in_ptr;\n");
        self.w("        ch->tmp_ptr = tmp + (ch->i_start * item_sz);\n");
        self.w("        ch->item_sz = item_sz; ch->env = env; ch->completion = th;\n");
        self.w("        ch->pred = (int(*)(void*, void*))code;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_filter_bytes_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    int64_t total = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        total += chs[c]->out_count;\n");
        self.w("    }\n");
        // total is bounded by sum of chunk counts which is ≤ n, so n*item_sz
        // already passed the overflow check above — but recompute safely.
        self.w("    size_t out_total;\n");
        self.w("    if (total <= 0) { out_total = 1; }\n");
        self.w("    else if (__builtin_mul_overflow((size_t)total, (size_t)item_sz, &out_total)) { free(tmp); return NULL; }\n");
        self.w("    char* out = (char*)malloc(out_total);\n");
        self.w("    if (!out) { free(tmp); return NULL; }\n");
        self.w("    int64_t w = 0;\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        memcpy(out + w * item_sz, tmp + chs[c]->i_start * item_sz,\n");
        self.w("               (size_t)(chs[c]->out_count * item_sz));\n");
        self.w("        w += chs[c]->out_count;\n");
        self.w("        __maka_thread_unref(hs[c]); free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(hs); free(chs); free(tmp);\n");
        self.w("    *out_n = total;\n");
        self.w("    return (void*)out;\n");
        self.w("}\n");
        // par_scan_bytes: 2-pass parallel inclusive scan over arbitrary items.
        // combine(env, prev_acc, cur, new_acc) writes the next accumulator
        // through new_acc, given prev_acc and the current input item.  Pass 1
        // per-chunk local scan; pass 2 folds the running per-chunk tail into
        // every chunk except chunk 0.
        self.w("typedef struct {\n");
        self.w("    int64_t i_start, i_end;\n");
        self.w("    char* in_ptr;\n");
        self.w("    char* out_ptr;\n");
        self.w("    char* offset_ptr;     /* item-sized scratch for offset pass */\n");
        self.w("    int64_t item_sz;\n");
        self.w("    void (*combine)(void*, void*, void*, void*);\n");
        self.w("    void* env;\n");
        self.w("    Thread* completion;\n");
        self.w("} __maka_bytes_scan_chunk_t;\n");
        self.w("static void* __maka_par_scan_bytes_local_entry(void* arg) {\n");
        self.w("    __maka_bytes_scan_chunk_t* c = (__maka_bytes_scan_chunk_t*)arg;\n");
        self.w("    if (c->i_start < c->i_end) {\n");
        // First item: copy directly (no prior accumulator).
        self.w("        memcpy(c->out_ptr + c->i_start * c->item_sz,\n");
        self.w("               c->in_ptr  + c->i_start * c->item_sz, (size_t)c->item_sz);\n");
        self.w("        for (int64_t i = c->i_start + 1; i < c->i_end; i++) {\n");
        self.w("            c->combine(c->env,\n");
        self.w("                       (void*)(c->out_ptr + (i - 1) * c->item_sz),\n");
        self.w("                       (void*)(c->in_ptr  +       i * c->item_sz),\n");
        self.w("                       (void*)(c->out_ptr +       i * c->item_sz));\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    __maka_thread_unref(c->completion);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static void* __maka_par_scan_bytes_offset_entry(void* arg) {\n");
        self.w("    __maka_bytes_scan_chunk_t* c = (__maka_bytes_scan_chunk_t*)arg;\n");
        // Apply running offset by folding it into each item via combine.
        self.w("    for (int64_t i = c->i_start; i < c->i_end; i++) {\n");
        self.w("        c->combine(c->env,\n");
        self.w("                   (void*)c->offset_ptr,\n");
        self.w("                   (void*)(c->out_ptr + i * c->item_sz),\n");
        self.w("                   (void*)(c->out_ptr + i * c->item_sz));\n");
        self.w("    }\n");
        self.w("    pthread_mutex_lock(&c->completion->done_mutex);\n");
        self.w("    c->completion->done_flag = 1;\n");
        self.w("    pthread_cond_broadcast(&c->completion->done_cond);\n");
        self.w("    pthread_mutex_unlock(&c->completion->done_mutex);\n");
        self.w("    __maka_thread_unref(c->completion);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("void* maka_par_scan_bytes(void* in_ptr, int64_t n, int64_t item_sz, void* code, void* env) {\n");
        self.w("    if (n <= 0 || item_sz <= 0) return NULL;\n");
        self.w("    size_t total;\n");
        self.w("    if (__builtin_mul_overflow((size_t)n, (size_t)item_sz, &total)) return NULL;\n");
        self.w("    char* out = (char*)malloc(total);\n");
        self.w("    if (!out) return NULL;\n");
        self.w("    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);\n");
        self.w("    if (nprocs < 1) nprocs = 1; if (nprocs > 16) nprocs = 16;\n");
        self.w("    int64_t chunks = (int64_t)nprocs;\n");
        self.w("    if (chunks > n) chunks = n;\n");
        self.w("    int64_t per = (n + chunks - 1) / chunks;\n");
        self.w("    Thread** hs = (Thread**)malloc(sizeof(Thread*) * (size_t)chunks);\n");
        self.w("    __maka_bytes_scan_chunk_t** chs = (__maka_bytes_scan_chunk_t**)malloc(sizeof(void*) * (size_t)chunks);\n");
        self.w("    void (*combine)(void*, void*, void*, void*) = (void(*)(void*, void*, void*, void*))code;\n");
        self.w("    /* pass 1: per-chunk local prefix */\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_bytes_scan_chunk_t* ch = (__maka_bytes_scan_chunk_t*)calloc(1, sizeof(__maka_bytes_scan_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > n) ch->i_end = n;\n");
        self.w("        ch->in_ptr = (char*)in_ptr; ch->out_ptr = out;\n");
        self.w("        ch->item_sz = item_sz; ch->env = env; ch->completion = th;\n");
        self.w("        ch->combine = combine;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_scan_bytes_local_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) { if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL); __maka_thread_unref(hs[c]); }\n");
        self.w("    /* pass 2: running offset across chunks (chunk 0 stays). */\n");
        // Compute prefix offsets sequentially.  running starts as chunk 0's
        // tail; each chunk's offset_ptr holds the running prior-tail value.
        self.w("    char* running = (char*)malloc((size_t)item_sz);\n");
        self.w("    memcpy(running, out + (chs[0]->i_end - 1) * item_sz, (size_t)item_sz);\n");
        self.w("    char* tmp = (char*)malloc((size_t)item_sz);\n");
        self.w("    for (int64_t c = 1; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        chs[c]->completion = th;\n");
        self.w("        chs[c]->offset_ptr = (char*)malloc((size_t)item_sz);\n");
        self.w("        memcpy(chs[c]->offset_ptr, running, (size_t)item_sz);\n");
        // Update running by combining current running with this chunk's tail.
        self.w("        memcpy(tmp, out + (chs[c]->i_end - 1) * item_sz, (size_t)item_sz);\n");
        self.w("        combine(env, running, tmp, running);\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_scan_bytes_offset_entry, chs[c], th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 1; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("        free(chs[c]->offset_ptr); free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(chs[0]);\n");
        self.w("    free(hs); free(chs); free(running); free(tmp);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("    __maka_thread_unref(c->completion);\n");
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
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        __maka_slice_chunk_t* ch = (__maka_slice_chunk_t*)calloc(1, sizeof(__maka_slice_chunk_t));\n");
        self.w("        ch->i_start = c * per; ch->i_end = ch->i_start + per;\n");
        self.w("        if (ch->i_end > s.len) ch->i_end = s.len;\n");
        self.w("        ch->in_ptr = s.ptr; ch->out_ptr = out; ch->env = env; ch->completion = th;\n");
        self.w("        ch->code.combine = combine;\n");
        self.w("        chs[c] = ch;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_scan_local_entry, ch, th);\n");
        self.w("        hs[c] = th;\n");
        self.w("    }\n");
        self.w("    int64_t* offsets = (int64_t*)malloc(sizeof(int64_t) * (size_t)chunks);\n");
        self.w("    for (int64_t c = 0; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        offsets[c] = chs[c]->out_acc;\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("    }\n");
        self.w("    /* pass 2: apply running offset across chunks (chunk 0 stays). */\n");
        self.w("    int64_t running = offsets[0];\n");
        self.w("    for (int64_t c = 1; c < chunks; c++) {\n");
        self.w("        Thread* th = __maka_thread_new();\n");
        self.w("        chs[c]->completion = th; chs[c]->init = running;\n");
        self.w("        __maka_spawn_pthread(&th->handle, __maka_par_scan_offset_entry, chs[c], th);\n");
        self.w("        hs[c] = th;\n");
        self.w("        running = combine(env, running, offsets[c]);\n");
        self.w("    }\n");
        self.w("    for (int64_t c = 1; c < chunks; c++) {\n");
        self.w("        if (!atomic_load(&hs[c]->is_failed)) pthread_join(hs[c]->handle, NULL);\n");
        self.w("        __maka_thread_unref(hs[c]);\n");
        self.w("        free(chs[c]);\n");
        self.w("    }\n");
        self.w("    free(chs[0]);\n");
        self.w("    free(hs); free(chs); free(offsets);\n");
        self.w("    Slice_maka_int res = { .ptr = out, .len = s.len };\n");
        self.w("    return res;\n");
        self.w("}\n");
        self.w("\n");
    }

    /// Does the nominal type named `name` carry a user `Drop` impl?  Mirrors
    /// sema's `type_impls_drop`: such a type is owning (its `drop` must run) and
    /// move-only even when it holds no heap fields.
    fn type_impls_drop(&self, name: &str) -> bool {
        self.sym.trait_impls.get("Drop").is_some_and(|s| s.contains(name))
    }

    /// The C name of `T`'s `Drop::drop` method, if `T has Drop`.  Looked up from
    /// the `has Drop` impl record keyed by the receiver type name.
    fn drop_method_cname(&self, type_name: &str) -> Option<String> {
        for h in &self.sym.has_impls {
            if h.attr_name != "Drop" || h.type_key != type_name { continue; }
            for &fid in &h.func_ids {
                let sig = self.sym.func_sig(fid);
                if sig.name == "drop" {
                    return Some(c_ident(&sig.c_name));
                }
            }
        }
        None
    }

    /// Does a value of this type, by itself, own heap resources that must be
    /// freed?  `own *T` / `own &T` own their pointee; structs/enums own if any
    /// field/variant does; arrays own if their element does.
    fn drop_ty_owns(&self, ty: &HType) -> bool {
        match ty {
            HType::OwnPtr { .. } | HType::Heap { .. } => true,
            // A by-value `Vec<T>` owns its malloc'd buffer.
            HType::Vec { .. } => true,
            // `Rust<T>` owns a boxed Rust value, dropped via a generated shim.
            HType::RustOpaque(_) => true,
            // A closure value owns its malloc'd capture env.  Drives drop-glue
            // for structs/enums that *store* a closure field; bare closure locals
            // are scheduled separately by the lifetime pass (no double free).
            HType::FnPtr { .. } => true,
            HType::Struct(id) => self.drop_owns.contains(&self.sym.struct_info(*id).name),
            HType::Enum(id) => self.drop_owns.contains(&self.sym.enum_info(*id).name),
            HType::Array { elem, .. } => self.drop_ty_owns(elem),
            _ => false,
        }
    }

    /// Fixpoint: mark every concrete struct/enum that transitively owns heap
    /// resources.  Seeds on `own *T` / `own &T` fields, propagates through
    /// by-value struct/enum/array fields.
    fn compute_drop_owns(&mut self) {
        let structs = self.sym.structs.clone();
        let enums = self.sym.enums.clone();
        // Seed: a `has Drop` type is owning by its destructor alone, even with no
        // owning fields.  Seeding before the fixpoint lets containers of a Drop
        // type inherit ownership through the normal field propagation below.
        for s in &structs {
            if !s.type_params.is_empty() { continue; }
            if self.type_impls_drop(&s.name) { self.drop_owns.insert(s.name.clone()); }
        }
        for e in &enums {
            if self.type_impls_drop(&e.name) { self.drop_owns.insert(e.name.clone()); }
        }
        loop {
            let mut changed = false;
            for s in &structs {
                if !s.type_params.is_empty() || s.name == "Thread" { continue; }
                if self.drop_owns.contains(&s.name) { continue; }
                if s.fields.iter().any(|f| self.drop_ty_owns(&f.ty)) {
                    self.drop_owns.insert(s.name.clone());
                    changed = true;
                }
            }
            for e in &enums {
                if e.is_simple() || self.drop_owns.contains(&e.name) { continue; }
                if e.variants.iter().any(|v| v.fields.iter().any(|f| self.drop_ty_owns(&f.ty))) {
                    self.drop_owns.insert(e.name.clone());
                    changed = true;
                }
            }
            if !changed { break; }
        }
    }

    /// Emit `__maka_drop_<Name>` for every owning struct/enum: recursively frees
    /// the owned fields of `*p` (but not `p` itself).  Forward-declared first so
    /// mutually recursive types resolve.
    fn emit_drop_glue(&mut self) {
        let structs: Vec<_> = self.sym.structs.iter()
            .filter(|s| s.type_params.is_empty() && s.name != "Thread" && self.drop_owns.contains(&s.name))
            .cloned().collect();
        let enums: Vec<_> = self.sym.enums.iter()
            .filter(|e| !e.is_simple() && self.drop_owns.contains(&e.name))
            .cloned().collect();
        if structs.is_empty() && enums.is_empty() { return; }
        self.wl("/* ---- recursive drop glue ---- */");
        for s in &structs { self.wl(&format!("static void __maka_drop_{0}(struct {0}* p);", c_ident(&s.name))); }
        for e in &enums { self.wl(&format!("static void __maka_drop_{0}(struct {0}* p);", c_ident(&e.name))); }
        for s in &structs {
            let cn = c_ident(&s.name);
            // A struct with EXACTLY ONE self-referential `own *Self` field (a
            // linked-list spine) drops that chain ITERATIVELY, not recursively -
            // one C stack frame per node would blow the stack on a long list (the
            // Box-chain problem).  The one self field is walked in a loop; every
            // other owning field (and the user `Drop`) is dropped per node.  A tree
            // (>1 self field) still recurses for now.
            let sid = self.sym.struct_by_name(&s.name).map(|(id, _)| id.0);
            let self_ptr_fields: Vec<&str> = s.fields.iter().filter_map(|f| match &f.ty {
                HType::OwnPtr { inner, .. } if matches!(inner.as_ref(), HType::Struct(id) if Some(id.0) == sid) =>
                    Some(f.name.as_str()),
                _ => None,
            }).collect();
            self.wl(&format!("static void __maka_drop_{0}(struct {0}* p) {{", cn));
            self.open();
            if let &[spine] = self_ptr_fields.as_slice() {
                // Iterative spine drop.  Per-node body (user Drop + non-spine owning
                // fields) is emitted for `p`, then for each `__it` in the chain.
                self.emit_node_body_drop("p", s, spine);
                self.wl(&format!("struct {0}* __it = p->{1};", cn, c_ident(spine)));
                self.wl("while (__it) {");
                self.open();
                self.wl(&format!("struct {0}* __nx = __it->{1};", cn, c_ident(spine)));
                self.emit_node_body_drop("__it", s, spine);
                self.wl("free(__it);");
                self.wl("__it = __nx;");
                self.close();
                self.wl("}");
            } else {
                if let Some(dropfn) = self.drop_method_cname(&s.name) {
                    self.wl(&format!("{}(p);", dropfn));
                }
                for fld in &s.fields {
                    if !self.drop_ty_owns(&fld.ty) { continue; }
                    let lv = format!("p->{}", c_ident(&fld.name));
                    self.emit_field_drop(&lv, &fld.ty, 0);
                }
            }
            self.close();
            self.wl("}");
        }
        for e in &enums {
            self.wl(&format!("static void __maka_drop_{0}(struct {0}* p) {{", c_ident(&e.name)));
            self.open();
            if let Some(dropfn) = self.drop_method_cname(&e.name) {
                self.wl(&format!("{}(p);", dropfn));
            }
            self.wl("switch (p->tag) {");
            self.open();
            for v in &e.variants {
                if !v.fields.iter().any(|f| self.drop_ty_owns(&f.ty)) { continue; }
                self.wl(&format!("case {}: {{", v.tag));
                self.open();
                for fld in &v.fields {
                    if !self.drop_ty_owns(&fld.ty) { continue; }
                    let lv = format!("p->payload.{}.{}", c_ident(&v.name), c_ident(&fld.name));
                    self.emit_field_drop(&lv, &fld.ty, 0);
                }
                self.close();
                self.wl("} break;");
            }
            self.close();
            self.wl("}");
            self.close();
            self.wl("}");
        }
        self.wl("");
    }

    /// Emit statements that free what the owned lvalue `lv` (of type `ty`) holds,
    /// including `lv` itself when it is an owning pointer.
    /// The per-node drop body used by the ITERATIVE spine drop: run the struct's
    /// user `Drop` on `base`, then drop each owning field EXCEPT `skip_field` (the
    /// self-referential spine, walked by the loop).  `base` is a `struct S*`.
    fn emit_node_body_drop(&mut self, base: &str, s: &StructInfo, skip_field: &str) {
        if let Some(dropfn) = self.drop_method_cname(&s.name) {
            self.wl(&format!("{}({});", dropfn, base));
        }
        for fld in &s.fields {
            if fld.name == skip_field { continue; }
            if !self.drop_ty_owns(&fld.ty) { continue; }
            let lv = format!("{}->{}", base, c_ident(&fld.name));
            self.emit_field_drop(&lv, &fld.ty, 0);
        }
    }

    fn emit_field_drop(&mut self, lv: &str, ty: &HType, depth: usize) {
        match ty {
            HType::OwnPtr { inner, .. } if matches!(inner.as_ref(), HType::Dyn { .. }) => {
                // `own *dyn Trait` collapses to the fat-pointer VALUE `{void* data;
                // vtbl}` in C, so `free(lv)` would free a struct value (invalid).
                // The owned erased object is `lv.data` - free that.
                self.wl(&format!("if (({0}).data) {{ free(({0}).data); }}", lv));
            }
            HType::OwnPtr { inner, .. } => {
                self.wl(&format!("if ({}) {{", lv));
                self.open();
                self.emit_pointee_drop(lv, inner, depth);
                // A FOREIGN opaque pointee (`extern data`) is freed by its own
                // Drop (the C destructor), not by Maka - emit no libc-free.
                let foreign = matches!(inner.as_ref(), HType::Struct(sid) if self.sym.struct_info(*sid).is_foreign);
                if !foreign {
                    self.wl(&format!("free({});", lv));
                }
                self.close();
                self.wl("}");
            }
            HType::Heap { inner } => {
                if let HType::Vec { elem } = inner.as_ref() {
                    // `own &[*]T` owns a malloc'd buffer.  If the elements own heap
                    // (e.g. `[*]own *int`), drop each before freeing the buffer.
                    if self.drop_ty_owns(elem) {
                        let i = format!("__v{}", depth);
                        self.wl(&format!("for (size_t {0} = 0; {0} < ({1}).len; {0}++) {{", i, lv));
                        self.open();
                        let elem_lv = format!("({}).data[{}]", lv, i);
                        self.emit_field_drop(&elem_lv, elem, depth + 1);
                        self.close();
                        self.wl("}");
                    }
                    self.wl(&format!("free({}.data);", lv));
                } else {
                    // Null guard: a field moved out (invalidated) is NULL here.
                    self.wl(&format!("if ({}) {{", lv));
                    self.open();
                    self.emit_pointee_drop(lv, inner, depth);
                    self.wl(&format!("free({});", lv));
                    self.close();
                    self.wl("}");
                }
            }
            HType::Struct(id) if self.drop_ty_owns(ty) => {
                self.wl(&format!("__maka_drop_{}(&({}));", c_ident(&self.sym.struct_info(*id).name), lv));
            }
            HType::Enum(id) if self.drop_ty_owns(ty) => {
                self.wl(&format!("__maka_drop_{}(&({}));", c_ident(&self.sym.enum_info(*id).name), lv));
            }
            HType::Array { len, elem } if self.drop_ty_owns(elem) => {
                let i = format!("__d{}", depth);
                self.wl(&format!("for (maka_int {0} = 0; {0} < {1}; {0}++) {{", i, len));
                self.open();
                let elem_lv = format!("({})[{}]", lv, i);
                self.emit_field_drop(&elem_lv, elem, depth + 1);
                self.close();
                self.wl("}");
            }
            // A dense existential column `Vec<some X>`: the element type is hidden,
            // so drop each element through the per-column witness drop-fn (NULL for
            // POD elements) striding by `__esz`, then free the buffer.
            HType::Vec { elem } if matches!(elem.as_ref(), HType::Dyn { locked: true, .. }) => {
                self.wl(&format!("if (({0}).__drop) {{ for (maka_int __v{1} = 0; __v{1} < ({0}).len; __v{1}++) {{ ({0}).__drop((char*)({0}).data + __v{1} * ({0}).__esz); }} }}", lv, depth));
                self.wl(&format!("free(({}).data);", lv));
            }
            // A by-value `Vec<T>`: drop owning elements, then free the buffer.
            HType::Vec { elem } => {
                if self.drop_ty_owns(elem) {
                    let i = format!("__v{}", depth);
                    self.wl(&format!("for (size_t {0} = 0; {0} < ({1}).len; {0}++) {{", i, lv));
                    self.open();
                    let elem_lv = format!("({}).data[{}]", lv, i);
                    self.emit_field_drop(&elem_lv, elem, depth + 1);
                    self.close();
                    self.wl("}");
                }
                self.wl(&format!("free({}.data);", lv));
            }
            // `Rust<T>`: hand the boxed pointer back to Rust to drop (generated shim).
            HType::RustOpaque(label) => {
                self.wl(&format!("if ({0}) {{ __maka_drop_{1}((void*)({0})); }}", lv, c_ident(label)));
            }
            // A non-escaping closure local: free its malloc'd capture env.  Only
            // scheduled by the lifetime pass when the closure provably does not
            // escape (so spawn'd closures, whose env the fiber frees, are excluded).
            HType::FnPtr { .. } => {
                self.wl(&format!("if (({0}).env) {{ free(({0}).env); }}", lv));
            }
            _ => {}
        }
    }

    /// Drop what the pointee owns, given `ptr` is a pointer to a value of
    /// `pointee` type (does not free `ptr` itself).
    fn emit_pointee_drop(&mut self, ptr: &str, pointee: &HType, depth: usize) {
        match pointee {
            HType::Struct(id) if self.drop_ty_owns(pointee) => {
                self.wl(&format!("__maka_drop_{}({});", c_ident(&self.sym.struct_info(*id).name), ptr));
            }
            HType::Enum(id) if self.drop_ty_owns(pointee) => {
                self.wl(&format!("__maka_drop_{}({});", c_ident(&self.sym.enum_info(*id).name), ptr));
            }
            HType::Array { .. } | HType::OwnPtr { .. } | HType::Heap { .. }
            | HType::Vec { .. } | HType::FnPtr { .. } | HType::RustOpaque(_) if self.drop_ty_owns(pointee) => {
                // A heap-allocated value that itself owns resources: a boxed
                // closure (`own *T(..)`) whose env must be freed, an owned Vec
                // buffer, a Rust box, etc.  Drop through `*ptr` before the OwnPtr
                // arm frees `ptr`.
                self.emit_field_drop(&format!("(*({}))", ptr), pointee, depth);
            }
            _ => {}
        }
    }

    /// Names of struct/enum types embedded BY VALUE in `ty` (directly or as an
    /// array element).  Pointer/ref/slice/vec fields are excluded - they only
    /// need a forward declaration, so they impose no definition ordering.
    fn value_dep_names(&self, ty: &HType, out: &mut Vec<String>) {
        match ty {
            HType::Struct(id) => out.push(self.sym.struct_info(*id).name.clone()),
            HType::Enum(id) => {
                let e = self.sym.enum_info(*id);
                if !e.is_simple() { out.push(e.name.clone()); }
            }
            HType::Array { elem, .. } => self.value_dep_names(elem, out),
            _ => {}
        }
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

        // Dyn (trait-object) typedefs go here - AFTER the struct/enum forward-decls
        // (their vtables take struct POINTERS, satisfied by the forward-decls) and
        // BEFORE the struct BODIES (a struct field can embed a `Dyn_T` fat-pointer
        // by value, so its typedef must exist first).  Idempotent.
        self.emit_dyn_typedefs();
        self.emit_combined_typedefs();

        // Full definitions, emitted in dependency order.  A type that embeds
        // another BY VALUE (a struct field, an enum-variant field, or an array
        // element) must be defined after it; pointer/slice/ref fields only need
        // the forward decls above, so they impose no ordering and recursive
        // types still work.
        let structs = self.sym.structs.clone();
        let enums = self.sym.enums.clone();
        for s in &structs {
            // Skip generic templates - their fields carry TyVar types (`Vec<V>`)
            // that would emit bogus typedefs.  Only instantiations are emitted.
            if !s.type_params.is_empty() { continue; }
            for f in &s.fields { self.note_type(&f.ty); }
        }
        // Enum variant payload fields likewise embed container/slice/callable
        // types by value (e.g. `List { Vec<int> xs }`), so their element typedefs
        // must be emitted before the variant payload struct that uses them.
        for e in &enums {
            if !e.type_params.is_empty() { continue; }
            for v in &e.variants {
                for f in &v.fields { self.note_type(&f.ty); }
            }
        }
        // FnPtr struct fields (and slice/vec-of-closure element types) need their
        // Callable typedef defined before the struct/container body that uses it.
        self.emit_callable_typedefs();
        self.emit_slice_typedefs();
        self.emit_vec_typedefs();

        // Index every emittable type by name: (is_enum, index).
        let mut kind_of: std::collections::HashMap<String, (bool, usize)> = std::collections::HashMap::new();
        let mut all_names: Vec<String> = Vec::new();
        for (i, s) in structs.iter().enumerate() {
            if !s.type_params.is_empty() || s.name == "Thread" { continue; }
            if kind_of.contains_key(&s.name) { continue; }
            kind_of.insert(s.name.clone(), (false, i));
            all_names.push(s.name.clone());
        }
        for (i, e) in enums.iter().enumerate() {
            if e.is_simple() { continue; }
            if kind_of.contains_key(&e.name) { continue; }
            kind_of.insert(e.name.clone(), (true, i));
            all_names.push(e.name.clone());
        }

        // Fixpoint topological order: emit a type once all its by-value deps
        // are already out.
        let mut emitted: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut ordered: Vec<String> = Vec::new();
        loop {
            let mut progressed = false;
            for name in &all_names {
                if emitted.contains(name) { continue; }
                let (is_enum, idx) = kind_of[name];
                let mut deps: Vec<String> = Vec::new();
                if is_enum {
                    for v in &enums[idx].variants {
                        for f in &v.fields { self.value_dep_names(&f.ty, &mut deps); }
                    }
                } else {
                    for f in &structs[idx].fields { self.value_dep_names(&f.ty, &mut deps); }
                }
                let ready = deps.iter().all(|d| !kind_of.contains_key(d) || emitted.contains(d));
                if ready {
                    ordered.push(name.clone());
                    emitted.insert(name.clone());
                    progressed = true;
                }
            }
            if !progressed { break; }
        }
        // Any remaining (only possible with an illegal by-value cycle) fall back
        // to declaration order so something is still emitted.
        for name in &all_names {
            if emitted.insert(name.clone()) { ordered.push(name.clone()); }
        }

        for name in &ordered {
            let (is_enum, idx) = kind_of[name];
            if is_enum {
                let e = &enums[idx];
                // Per-variant payload structs (named EnumName_VariantName).
                for v in &e.variants {
                    if v.fields.is_empty() { continue; }
                    self.wl("typedef struct {");
                    self.open();
                    for f in &v.fields {
                        let ty = self.c_type(&f.ty);
                        self.wl(&format!("{} {};", ty, c_ident(&f.name)));
                    }
                    self.close();
                    self.wl(&format!("}} {0}_{1}_Payload;", c_ident(&e.name), c_ident(&v.name)));
                }
                // The enum struct itself: tag + union of variant payloads.
                self.wl(&format!("struct {0} {{", c_ident(&e.name)));
                self.open();
                self.wl("maka_int tag;");
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
            } else {
                let s = &structs[idx];
                self.wl(&format!("struct {} {{", c_ident(&s.name)));
                self.open();
                for f in &s.fields {
                    // `c_decl` places the field name correctly for array-of-T
                    // fields (`T name[N]`) vs the standard `T name` form.
                    let decl = self.c_decl(&f.ty, &c_ident(&f.name));
                    self.wl(&format!("{};", decl));
                }
                self.close();
                self.wl("};");
            }
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
            HStmt::Break { .. } | HStmt::Continue { .. } => {}
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
                match kind {
                    // A multi-trait `dyn (A+B)` value needs a COMBINED vtable (union of
                    // methods); a single-trait one uses the plain per-trait vtable.
                    CastKind::ToDyn { traits, struct_id } => {
                        if traits.len() == 1 {
                            self.dyn_traits.insert(traits[0].clone());
                            self.dyn_insts.insert((traits[0].clone(), struct_id.0));
                        } else {
                            self.combined_vtbls.insert(traits.clone());
                            self.combined_insts.insert((traits.clone(), struct_id.0));
                        }
                    }
                    CastKind::PackSomeVec { trait_name, struct_id }
                    | CastKind::DowncastSomeVec { trait_name, struct_id } => {
                        self.dyn_traits.insert(trait_name.clone());
                        self.dyn_insts.insert((trait_name.clone(), struct_id.0));
                    }
                    _ => {}
                }
                self.scan_expr(expr);
            }
            HExprKind::DynRewitness { value, src_trait, to_traits } => {
                self.dyn_witness_targets.insert((src_trait.clone(), to_traits.clone()));
                self.scan_expr(value);
            }
            HExprKind::DynHasVtbl(v) => self.scan_expr(v),
            HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) => self.scan_expr(expr),
            HExprKind::ArrayToSlice { base, .. } => self.scan_expr(base),
            HExprKind::DerefRef(inner) => self.scan_expr(inner),
            HExprKind::HeapAlloc(inner) => self.scan_expr(inner),
            HExprKind::Free(inner, _) => self.scan_expr(inner),
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
            HType::RustOpaque(label) => { self.rust_drop_types.insert(label.clone()); }
            HType::Heap { inner } => self.note_type(inner),
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } => self.note_type(inner),
            HType::Array { elem, .. } => self.note_type(elem),
            HType::Dyn { traits, .. } => for t in traits { self.dyn_traits.insert(t.clone()); },
            HType::FnPtr { ret, params } => {
                let key = fn_sig_key(ret, params);
                self.callable_sigs.insert(key, ((**ret).clone(), params.clone()));
                self.note_type(ret);
                for p in params { self.note_type(p); }
            }
            // Recurse into aggregate field / variant types so a Slice/Vec/FnPtr
            // used only as a struct field or enum payload still gets its typedef
            // (e.g. a `int(int)` closure field needs its `Callable_*` typedef).
            // The `noted_aggregates` guard makes self-referential types terminate.
            HType::Struct(id) => {
                let key = self.type_key(t);
                if self.noted_aggregates.insert(key) {
                    let fields: Vec<HType> = self.sym.struct_info(*id).fields.iter().map(|f| f.ty.clone()).collect();
                    for ft in fields { self.note_type(&ft); }
                }
            }
            HType::Enum(id) => {
                let key = self.type_key(t);
                if self.noted_aggregates.insert(key) {
                    let tys: Vec<HType> = self.sym.enum_info(*id).variants.iter()
                        .flat_map(|v| v.fields.iter().map(|f| f.ty.clone())).collect();
                    for ft in tys { self.note_type(&ft); }
                }
            }
            _ => {}
        }
    }

    fn emit_dyn_typedefs(&mut self) {
        // Typedef structs only (Movement_vtbl + Dyn_Movement). Vtable instances are emitted
        // separately AFTER function forward declarations (via emit_dyn_vtable_instances).
        let traits: Vec<String> = self.dyn_traits.iter().cloned().collect();
        for tn in &traits {
            // A trait's methods are the sigs tagged with its name -
            // an `attr`'s `has`-impl methods.
            let funcs = self.sym.trait_method_funcs(tn);
            if funcs.is_empty() {
                continue;
            }
            // Take *unique* method names of the trait.
            let mut name_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for fid in &funcs {
                let s = self.sym.func_sig(*fid);
                name_set.insert(s.name.clone());
            }
            // Emit vtable struct.
            self.wl(&format!("typedef struct {0}_vtbl {{", c_ident(tn)));
            self.open();
            for n in &name_set {
                // For each declared name, find one signature to derive the function-pointer shape.
                let sig = funcs.iter().filter_map(|fid| {
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

            // Dyn struct (per-value existential: {data, vtbl}).
            self.wl(&format!("typedef struct {{ void* data; const {0}_vtbl* vtbl; }} Dyn_{0};", c_ident(tn)));
            // SomeVec struct (`Vec<some X>`): a DENSE homogeneous column - a Vec
            // header over raw `T` values, plus ONE hoisted witness (the trait vtbl
            // for the hidden `T` and its element size, for striding).  Iterating
            // hands out `Dyn_X` fat pointers `{data + j*esz, vtbl}`, so dispatch
            // reuses the ordinary dyn path.
            self.wl(&format!("typedef struct {{ void* data; maka_int len; maka_int cap; const {0}_vtbl* __vtbl; maka_int __esz; void (*__drop)(void*); }} SomeVec_{0};", c_ident(tn)));
        }
    }

    fn emit_callable_typedefs(&mut self) {
        // For each fat-callable signature, emit:
        //   typedef RetType (*fn_KEY_raw)(void*, ParamTypes);
        //   typedef struct { fn_KEY_raw code; void* env; } Callable_KEY;
        let sigs: Vec<(String, HType, Vec<HType>)> = self.callable_sigs.iter()
            .map(|(k, (r, ps))| (k.clone(), r.clone(), ps.clone())).collect();
        for (key, ret, params) in &sigs {
            // Guard against double emission: this runs both before struct bodies
            // (so FnPtr struct fields resolve) and again in the prologue (for
            // FnPtr types that appear only in function bodies).
            let decl = format!("typedef struct {{ fn_{0}_raw code; void* env; }} Callable_{0};", key);
            if self.out.contains(&decl) { continue; }
            let ret_c = self.c_ret_type(ret);
            let mut parts: Vec<String> = vec!["void*".to_string()];
            for p in params { parts.push(self.c_type(p)); }
            let plist = parts.join(", ");
            self.wl(&format!("typedef {} (*fn_{}_raw)({});", ret_c, key, plist));
            self.wl(&decl);
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
                else if sig.name == "main" && sig.trait_name.is_none() { "maka_main".to_string() }
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
                // The trampoline's C return type is `void` (c_ret_type maps unit
                // to void), so a bare `return;` - not `return MAKA_UNIT;`, which
                // modern gcc rejects as a value-return from a void function.
                self.wl("return;");
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
                // void trampoline -> bare `return;` (see emit_trampolines).
                self.wl("return;");
            } else {
                self.wl(&format!("return {}({});", lifted_c, call_args.join(", ")));
            }
            self.close();
            self.wl("}");
            // The signature key for the callable produced.
            let _ = key;
        }
    }

    /// Does struct `sid` have a buildable `trait_name` vtable, i.e. an impl of
    /// every trait method taking `&Struct(sid)` as receiver?  Mirrors the `chosen`
    /// lookup in `emit_dyn_vtable_instances` (so a candidate we admit is always
    /// vtable-buildable) and the sema `struct_satisfies_trait` used for `as dyn`.
    fn struct_impls_trait_cg(&self, sid: StructId, trait_name: &str) -> bool {
        let funcs = self.sym.trait_method_funcs(trait_name);
        if funcs.is_empty() { return false; }
        let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for fid in &funcs { names.insert(self.sym.func_sig(*fid).name.clone()); }
        names.iter().all(|n| funcs.iter().any(|fid| {
            let s = self.sym.func_sig(*fid);
            s.name == *n
                && matches!(s.param_tys.first(),
                    Some(HType::Ref { inner, .. }) if matches!(inner.as_ref(), HType::Struct(id) if *id == sid))
        }))
    }

    /// Every existential instantiation available as a re-witness SOURCE, keyed by
    /// its runtime vtable key: single-trait (`dyn_insts`) and combined (`combined_insts`,
    /// key = `traits.join("_")`).  A narrowed value's vtbl is a combined one, so a
    /// nested `has` must find it here.
    fn dyn_source_insts(&self) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> = self.dyn_insts.iter().map(|(t, s)| (t.clone(), *s)).collect();
        v.extend(self.combined_insts.iter().map(|(traits, s)| (traits.join("_"), *s)));
        v
    }

    /// The concrete types whose source vtbl is `src_trait` (single or combined) and
    /// that implement ALL of `to_traits` (struct names, for the re-witness switch),
    /// de-duplicated in stable order.
    fn dyn_witness_candidates(&self, src_trait: &str, to_traits: &[String]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (key, sid_n) in self.dyn_source_insts() {
            if key != src_trait { continue; }
            if !to_traits.iter().all(|tr| self.struct_impls_trait_cg(StructId(sid_n), tr)) { continue; }
            let name = self.sym.struct_info(StructId(sid_n)).name.clone();
            if !out.contains(&name) { out.push(name); }
        }
        out
    }

    /// Union of the method funcs of several traits (for a combined narrowing vtbl).
    fn combined_method_funcs(&self, traits: &[String]) -> Vec<FuncId> {
        let mut v = Vec::new();
        for t in traits { v.extend(self.sym.trait_method_funcs(t)); }
        v
    }

    /// Emit `Dyn_<key>` + `<key>_vtbl` typedefs for each combined (multi-trait)
    /// re-witness target; the vtbl unions all constituent traits' methods so a
    /// `dyn (Y+X)` value dispatches either interface directly.
    fn emit_combined_typedefs(&mut self) {
        let combos: Vec<Vec<String>> = self.combined_vtbls.iter().cloned().collect();
        for traits in &combos {
            let funcs = self.combined_method_funcs(traits);
            if funcs.is_empty() { continue; }
            let key = c_ident(&traits.join("_"));
            let mut name_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for fid in &funcs { name_set.insert(self.sym.func_sig(*fid).name.clone()); }
            self.wl(&format!("typedef struct {0}_vtbl {{", key));
            self.open();
            for n in &name_set {
                let Some(sig) = funcs.iter().find(|fid| self.sym.func_sig(**fid).name == *n)
                    .map(|fid| self.sym.func_sig(*fid).clone()) else { continue; };
                let ret = self.c_ret_type(&sig.ret);
                let mut parts: Vec<String> = Vec::new();
                for (i, p) in sig.param_tys.iter().enumerate() {
                    if i == 0 { parts.push("void*".into()); } else { parts.push(self.c_type(p)); }
                }
                let pstr = if parts.is_empty() { "void".to_string() } else { parts.join(", ") };
                self.wl(&format!("{} (*{})({});", ret, c_ident(n), pstr));
            }
            self.close();
            self.wl(&format!("}} {0}_vtbl;", key));
            self.wl(&format!("typedef struct {{ void* data; const {0}_vtbl* vtbl; }} Dyn_{0};", key));
        }
    }

    /// Emit the per-type `<key>_vtbl_for_T` instances for combined re-witness vtbls.
    fn emit_combined_vtable_instances(&mut self) {
        let insts: Vec<(Vec<String>, u32)> = self.combined_insts.iter().cloned().collect();
        for (traits, sid_n) in &insts {
            let sid = StructId(*sid_n);
            let funcs = self.combined_method_funcs(traits);
            if funcs.is_empty() { continue; }
            let sname = self.sym.struct_info(sid).name.clone();
            let key = c_ident(&traits.join("_"));
            let mut name_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for fid in &funcs { name_set.insert(self.sym.func_sig(*fid).name.clone()); }
            self.wl(&format!("static const {0}_vtbl {0}_vtbl_for_{1} = {{", key, c_ident(&sname)));
            self.open();
            for n in &name_set {
                let chosen = funcs.iter().find_map(|fid| {
                    let s = self.sym.func_sig(*fid);
                    if s.name != *n || s.param_tys.is_empty() { return None; }
                    if let HType::Ref { inner, .. } = &s.param_tys[0] {
                        if let HType::Struct(id) = inner.as_ref() {
                            if *id == sid { return Some(s.c_name.clone()); }
                        }
                    }
                    None
                });
                if let Some(cn) = chosen { self.wl(&format!(".{} = (void*){},", c_ident(n), cn)); }
            }
            self.close();
            self.wl("};");
        }
    }

    fn emit_dyn_vtable_instances(&mut self) {
        let insts: Vec<(String, u32)> = self.dyn_insts.iter().cloned().collect();
        for (tn, sid_n) in &insts {
            let sid = StructId(*sid_n);
            let sname = self.sym.struct_info(sid).name.clone();
            let funcs = self.sym.trait_method_funcs(tn);
            if funcs.is_empty() {
                continue;
            }
            // For each distinct method name, find the matching impl for this struct
            // (the `has X for T` method).
            let mut name_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for fid in &funcs {
                let s = self.sym.func_sig(*fid);
                name_set.insert(s.name.clone());
            }
            self.wl(&format!("static const {0}_vtbl {0}_vtbl_for_{1} = {{", c_ident(tn), c_ident(&sname)));
            self.open();
            for n in &name_set {
                // Find the FuncSig for this T.
                let chosen = funcs.iter().find_map(|fid| {
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
        let mut keys: Vec<String> = self.vec_types.iter().cloned().collect();
        // Inner Vec typedefs must precede the outer ones that embed them:
        // `Vec<Vec<int>>` (element key "Vec_maka_int") refers to `Vec<int>`
        // (key "maka_int"), so emit by ascending nesting depth.  A plain
        // BTreeSet sorts "Vec_..." before "maka_int" and breaks the C build.
        keys.sort_by_key(|k| (k.matches("Vec_").count(), k.clone()));
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
            HType::SizedFloat { bits: 32 } => "float".into(),
            HType::SizedFloat { bits: 64 } => "double".into(),
            HType::SizedFloat { bits } => format!("_unknown_f{}", bits),
            HType::Bool => "bool".into(),
            HType::Char => "maka_char".into(),
            HType::Unit => "maka_unit".into(),
            HType::Struct(id) => c_ident(&self.sym.struct_info(*id).name),
            HType::Enum(id) => c_ident(&self.sym.enum_info(*id).name),
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. } | HType::OwnPtr { inner, .. } => {
                // A reference/pointer to a `dyn Trait` collapses to the fat-pointer
                // value itself (mirrors c_type), so a container element type key
                // (`Vec<&dyn Trait>`) resolves to `Dyn_Trait`, not `Dyn_Trait*`.
                if matches!(inner.as_ref(), HType::Dyn { .. }) {
                    self.type_key(inner)
                } else if matches!(inner.as_ref(), HType::Str) {
                    // A pointer to a string decays to a single char buffer (mirrors
                    // c_type): owned -> `char*` (str_own), borrowed -> `const char*` (str).
                    if matches!(t, HType::OwnPtr { .. }) { "str_own".into() } else { "str".into() }
                } else {
                    format!("p_{}", self.type_key(inner))
                }
            }
            HType::Heap { inner } if matches!(inner.as_ref(), HType::Str) => "str_own".into(),
            HType::Heap { inner } => format!("h_{}", self.type_key(inner)),
            HType::Array { len, elem } => format!("a{}_{}", len, self.type_key(elem)),
            HType::Slice { elem, .. } => format!("Slice_{}", self.type_key(elem)),
            // `Vec<some X>` is a DENSE existential column (one hoisted witness),
            // distinct from `Vec<dyn X>` (per-element fat pointers).
            HType::Vec { elem } => match elem.as_ref() {
                HType::Dyn { traits, locked: true } => format!("SomeVec_{}", traits.join("_")),
                _ => format!("Vec_{}", self.type_key(elem)),
            },
            HType::Str => "str".into(),
            HType::NullT => "nullptr_t".into(),
            HType::Dyn { traits, .. } => format!("Dyn_{}", traits.join("_")),
            // Key a closure/fn-pointer by its fat-callable type name, so a
            // container element (`Vec<int(int)>`) resolves via c_type_from_key to
            // the `Callable_*` struct, not a bogus raw-fn-ptr name.
            HType::FnPtr { ret, params } => format!("Callable_{}", fn_sig_key(ret, params)),
            HType::TyVar(n) => format!("T_{}", n),
            HType::AssocType { on, segment, .. } => format!("AT_{}_{}", self.type_key(on), segment),
            HType::GenericPattern { template_name, args, .. } => {
                let inner: Vec<String> = args.iter().map(|a| self.type_key(a)).collect();
                format!("GP_{}__{}", template_name, inner.join("_"))
            }
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
        // Heap keys (`own &T`, encoded `h_<inner>`) land as `<inner>*` in C,
        // matching c_type - except `own &[*]T` (a Vec), which is the Vec struct
        // by value with no extra indirection.
        if let Some(rest) = k.strip_prefix("h_") {
            if rest.starts_with("Vec_") {
                return self.c_type_from_key(rest);
            }
            return format!("{}*", self.c_type_from_key(rest));
        }
        match k {
            "str" => "const char*".into(),
            // owned string buffer (`own *string` / `own &string`) - a single freeable char*.
            "str_own" => "char*".into(),
            other => other.to_string(),
        }
    }

    // A Maka pointer whose pointee is a fixed array or a string is represented in C
    // as a single element pointer (the array data / a `char*`).  Unwrap (`p!`) and
    // explicit deref (`*p`) of such a pointer are therefore no-ops at the C level.
    // Shared by every (inline and regular) emission site so the two cannot drift.
    fn ptr_pointee_decays(ptr_ty: &HType) -> bool {
        matches!(ptr_ty,
            HType::Ptr { inner, .. } | HType::OwnPtr { inner, .. } | HType::RawPtr { inner, .. }
            | HType::Ref { inner, .. } | HType::Heap { inner }
            if matches!(inner.as_ref(), HType::Array { .. } | HType::Str))
    }

    fn c_type(&self, t: &HType) -> String {
        match t {
            HType::Int => "maka_int".into(),
            HType::SizedInt { signed, bits: 0 } => if *signed { "intptr_t".into() } else { "uintptr_t".into() },
            HType::SizedInt { signed, bits } => format!("{}int{}_t", if *signed {""} else {"u"}, bits),
            HType::Float => "maka_float".into(),
            HType::SizedFloat { bits: 32 } => "float".into(),
            HType::SizedFloat { bits: 64 } => "double".into(),
            HType::SizedFloat { bits } => format!("_unknown_f{}", bits),
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
                // A string is an array of chars, so a pointer to one decays to a single
                // `char*` (a nullable string) - never a `char**`.  `own *string` owns its
                // buffer (mutable/freeable); a borrowed `*string`/`&string` is read-only.
                if matches!(inner.as_ref(), HType::Str) {
                    return if matches!(t, HType::OwnPtr { .. }) { "char*".into() } else { "const char*".into() };
                }
                // A pointer to a fixed array (`own *[N]T`, `&[N]T`, ...) decays to a
                // pointer-to-element in C.  C cannot spell `int (*)[N]` in our prefix
                // type model, and every observation (`p![i]`, `p!.len`) wants the
                // element pointer anyway; the length N rides along in the Maka type.
                if let HType::Array { elem, .. } = inner.as_ref() {
                    return format!("{}*", self.c_type(elem));
                }
                format!("{}*", self.c_type(inner))
            }
            HType::Heap { inner } => match inner.as_ref() {
                // heap [*]T is just the Vec struct (no extra indirection)
                HType::Vec { .. } => self.c_type(inner),
                HType::Array { elem, .. } => format!("{}*", self.c_type(elem)),
                // `own &string` is a single owned `char*` (a string is an array of chars).
                HType::Str => "char*".into(),
                _ => format!("{}*", self.c_type(inner)),
            },
            HType::Array { len, elem } => {
                // arrays are only valid in fixed-size contexts; use raw element type and
                // append [N] at the declaration site. For type-name use a pointer-decay.
                format!("{}[{}]", self.c_type(elem), len)
            }
            HType::Slice { elem, .. } => format!("Slice_{}", self.type_key(elem)),
            HType::Vec { elem } => match elem.as_ref() {
                HType::Dyn { traits, locked: true } => format!("SomeVec_{}", traits.join("_")),
                _ => format!("Vec_{}", self.type_key(elem)),
            },
            HType::Dyn { traits, .. } => format!("Dyn_{}", c_ident(&traits.join("_"))),
            HType::FnPtr { ret, params } => {
                format!("Callable_{}", fn_sig_key(ret, params))
            }
            HType::TyVar(_) => "void*".into(), // erased; not expected at codegen
            HType::AssocType { .. } => "void*".into(), // erased; should have been resolved at mono
            HType::GenericPattern { .. } => "void*".into(), // pattern-only; never reaches codegen for real instantiations
            // `Rust<T>` is `own *mut unit` at the C layer (= `void*`).
            HType::RustOpaque(_) => "void*".into(),
        }
    }

    /// Like c_type but suitable for declaring a variable named `name` with that type.
    fn c_decl(&self, t: &HType, name: &str) -> String {
        match t {
            HType::Array { .. } => {
                // All array dimensions go after the declarator name, with the base
                // element type as the prefix: `T name[d0][d1]...`.  A nested array
                // ([2][3]int) must emit `maka_int name[2][3]`, not the invalid
                // `maka_int[3] name[2]`.
                let mut dims = String::new();
                let mut cur = t;
                while let HType::Array { len, elem } = cur {
                    dims.push_str(&format!("[{}]", len));
                    cur = elem;
                }
                format!("{} {}{}", self.c_type(cur), name, dims)
            }
            HType::FnPtr { ret, params } => {
                format!("Callable_{} {}", fn_sig_key(ret, params), name)
            }
            _ => format!("{} {}", self.c_type(t), name),
        }
    }

    fn func_signature(&self, f: &HFunc) -> String {
        let ret = self.c_ret_type(&f.ret);
        let sig = self.sym.func_sig(f.id);
        let is_main = sig.name == "main" && sig.trait_name.is_none();
        let mangled = if is_main {
            "maka_main".to_string()
        } else {
            c_ident(&sig.c_name)
        };
        // Module-private functions get internal linkage: everything compiles to
        // a single C translation unit, so non-`pub` functions are never referenced
        // externally, and `static` lets the C compiler inline/optimize them freely
        // (a large win for small hot helpers - vector math, etc.).  `main` stays
        // external for the C entry shim; `pub` and `extern` are left external.
        let linkage = if !sig.is_pub && !sig.is_export && !sig.is_extern && !is_main { "static " } else { "" };
        let mut out = format!("{}{} {}(", linkage, ret, mangled);
        if f.params.is_empty() { out.push_str("void"); }
        else {
            let parts: Vec<String> = f.params.iter().map(|id| {
                let li = &f.locals[id.0 as usize];
                self.c_decl(&li.ty.strip_heap().clone(), &self.local_name(*id, &li.name))
            }).collect();
            // For heap params we still use T* (heap-storage). Use c_decl on the heap-wrapped type instead.
            let parts2: Vec<String> = f.params.iter().map(|id| {
                let li = &f.locals[id.0 as usize];
                let n = self.local_name(*id, &li.name);
                // A by-value array param can't be a real value in a C signature
                // (arrays decay to pointers), so it arrives as a decayed pointer
                // under a mangled name; the function BODY copies it into a real
                // local array (value semantics - the callee's mutations do not
                // touch the caller).  `&[N]T` / `&mut [N]T` are `Ref`s, genuine
                // borrows, and are left untouched.
                if matches!(&li.ty, HType::Array { .. }) {
                    self.c_decl(&li.ty, &format!("{}__pv", n))
                } else {
                    self.c_decl(&li.ty, &n)
                }
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
        self.w("#ifndef _WIN32\n");
        self.w("#include <sys/syscall.h>\n");
        self.w("#endif\n");
        // Manual forward decls avoid pulling sys/socket.h's declaration of
        // `accept` into scope (which would conflict with user-defined funcs
        // named `accept`, see test 114).  All socket calls go through these
        // shims; `accept` is invoked via syscall(2) inside the helper below.
        // On Windows winsock2.h is already in scope from the top of the
        // prologue — it provides sockaddr_in / htons / sendto / recvfrom /
        // accept etc.  Skip the manual forward decls to avoid type clashes.
        self.w("#ifndef _WIN32\n");
        // On Darwin/BSD, sockaddr_in / sockaddr_un have a leading sin_len /
        // sun_len byte the kernel checks — bind() returns EAFNOSUPPORT if it
        // sees zero there.  Pull the real headers on those targets so the
        // ABI matches.  Linux's struct has no sin_len, so the local layout
        // works there.
        self.w("#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)\n");
        self.w("#include <sys/types.h>\n");
        self.w("#include <sys/socket.h>\n");
        self.w("#include <netinet/in.h>\n");
        self.w("#include <netinet/tcp.h>\n");
        self.w("#include <arpa/inet.h>\n");
        self.w("#include <sys/un.h>\n");
        self.w("typedef socklen_t __maka_socklen_t;\n");
        // Darwin/BSD sockaddr_in has a leading sin_len; bind/connect return
        // EAFNOSUPPORT when it's zero.  Macro is a no-op on Linux where the
        // field doesn't exist.
        self.w("#define __MAKA_SA_LEN_INIT(sa) ((sa).sin_len = sizeof(sa))\n");
        self.w("#else\n");
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
        self.w("#define __MAKA_SA_LEN_INIT(sa) ((void)0)\n");
        self.w("#endif\n");
        self.w("extern unsigned short htons(unsigned short);\n");
        self.w("extern unsigned int   htonl(unsigned int);\n");
        self.w("extern unsigned short ntohs(unsigned short);\n");
        self.w("extern unsigned int   ntohl(unsigned int);\n");
        self.w("#else\n");
        self.w("typedef int __maka_socklen_t;\n");
        self.w("#define __MAKA_SA_LEN_INIT(sa) ((void)0)\n");
        self.w("#endif\n");
        // Use the system-header values when present (macOS/BSD include them
        // via our <sys/socket.h> include; Linux/Windows fall back to the
        // Linux/winsock values which are the same in practice for AF_INET
        // and SOCK_STREAM but differ for SOL_SOCKET / SO_*).
        self.w("#ifndef AF_INET\n");
        self.w("#define __MAKA_AF_INET     2\n");
        self.w("#else\n");
        self.w("#define __MAKA_AF_INET     AF_INET\n");
        self.w("#endif\n");
        self.w("#ifndef SOCK_STREAM\n");
        self.w("#define __MAKA_SOCK_STREAM 1\n");
        self.w("#else\n");
        self.w("#define __MAKA_SOCK_STREAM SOCK_STREAM\n");
        self.w("#endif\n");
        self.w("#ifndef INADDR_ANY\n");
        self.w("#define __MAKA_INADDR_ANY  0u\n");
        self.w("#else\n");
        self.w("#define __MAKA_INADDR_ANY  INADDR_ANY\n");
        self.w("#endif\n");
        self.w("#ifndef SOL_SOCKET\n");
        self.w("#define __MAKA_SOL_SOCKET  1\n");
        self.w("#else\n");
        self.w("#define __MAKA_SOL_SOCKET  SOL_SOCKET\n");
        self.w("#endif\n");
        self.w("#ifndef SO_REUSEADDR\n");
        self.w("#define __MAKA_SO_REUSEADDR 2\n");
        self.w("#else\n");
        self.w("#define __MAKA_SO_REUSEADDR SO_REUSEADDR\n");
        self.w("#endif\n");
        self.w("#ifndef SO_ERROR\n");
        self.w("#define __MAKA_SO_ERROR    4\n");
        self.w("#else\n");
        self.w("#define __MAKA_SO_ERROR    SO_ERROR\n");
        self.w("#endif\n");
        // tcp_listen binds INADDR_ANY (0.0.0.0) — accepts on every interface.
        // On Windows the firewall prompts the first time a binary calls this
        // (allow once and it's remembered).  Same behavior as tcp_listen_any;
        // the two are kept as separate names for source-compat with code that
        // already used either spelling.
        self.w("static inline int64_t __maka_tcp_listen(int64_t port, int64_t backlog) {\n");
        self.w("    int s = socket(__MAKA_AF_INET, __MAKA_SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    int one = 1;\n");
        self.w("    setsockopt(s, __MAKA_SOL_SOCKET, __MAKA_SO_REUSEADDR, (char*)&one, sizeof(one));\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa)); __MAKA_SA_LEN_INIT(sa);\n");
        self.w("    sa.sin_family = __MAKA_AF_INET;\n");
        self.w("    sa.sin_addr.s_addr = htonl(__MAKA_INADDR_ANY);   /* 0.0.0.0 */\n");
        self.w("    sa.sin_port = htons((unsigned short)port);\n");
        self.w("    if (bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) { close(s); return -1; }\n");
        self.w("    if (listen(s, (int)backlog) != 0) { close(s); return -1; }\n");
        self.w("    int flags = fcntl(s, F_GETFL, 0);\n");
        self.w("    fcntl(s, F_SETFL, flags | O_NONBLOCK);\n");
        self.w("    return s;\n");
        self.w("}\n");
        // Same as tcp_listen but binds INADDR_ANY (0.0.0.0) — accepts on all
        // interfaces.  On Windows the firewall will prompt the first time
        // a binary calls this.  Use for production servers.
        self.w("static inline int64_t __maka_tcp_listen_any(int64_t port, int64_t backlog) {\n");
        self.w("    int s = socket(__MAKA_AF_INET, __MAKA_SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    int one = 1;\n");
        self.w("    setsockopt(s, __MAKA_SOL_SOCKET, __MAKA_SO_REUSEADDR, (char*)&one, sizeof(one));\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa)); __MAKA_SA_LEN_INIT(sa);\n");
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
        // Call accept via direct syscall on Linux to avoid clashing with a
        // user-named Maka function called `accept` (see tests/programs/114_*).
        // On macOS/BSD/Windows, SYS_accept numbers differ (or don't exist
        // for BSDs that don't expose SYS_*); use the libc accept directly.
        self.w("#ifdef _WIN32\n");
        self.w("        int c = (int)accept((int)listen_fd, NULL, NULL);\n");
        self.w("#elif defined(__linux__)\n");
        self.w("        int c = (int)syscall(SYS_accept, (int)listen_fd, (void*)0, (void*)0);\n");
        self.w("#else\n");
        // macOS/BSD: <sys/socket.h> is already in scope (pulled at the top of
        // the prologue), so the real accept() prototype is visible.  Don't
        // re-declare with our own signature — that triggers a redeclaration
        // error against the SDK's `int accept(int, struct sockaddr*, socklen_t*)`.
        self.w("        int c = accept((int)listen_fd, NULL, NULL);\n");
        self.w("#endif\n");
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
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa)); __MAKA_SA_LEN_INIT(sa);\n");
        self.w("    sa.sin_family = __MAKA_AF_INET;\n");
        self.w("    sa.sin_port = htons((unsigned short)port);\n");
        self.w("    unsigned char octets[4] = {(unsigned char)a, (unsigned char)b, (unsigned char)c, (unsigned char)d};\n");
        self.w("    memcpy(&sa.sin_addr.s_addr, octets, 4);\n");
        self.w("    int r = connect(s, (struct sockaddr*)&sa, sizeof(sa));\n");
        self.w("    if (r == 0) return s;\n");
        self.w("    if (errno == EINPROGRESS) {\n");
        self.w("        __maka_wait_fd(s, MAKA_EV_WRITE);\n");
        self.w("        int err = 0; __maka_socklen_t elen = sizeof(err);\n");
        self.w("        getsockopt(s, __MAKA_SOL_SOCKET, __MAKA_SO_ERROR, (char*)&err, &elen);\n");
        self.w("        if (err == 0) return s;\n");
        self.w("        close(s); return -1;\n");
        self.w("    }\n");
        self.w("    close(s); return -1;\n");
        self.w("}\n");
        // Drop reactor registration first so the per-fd events_mask doesn't
        // outlive the fd into recycled lifetimes; otherwise __maka_fd_arm
        // short-circuits and the new fd never registers.  Also wake every
        // fiber parked on that fd — local AND on other schedulers.
        self.w("static inline int64_t __maka_close_fd(int64_t fd) {\n");
        self.w("    if (maka_sched_inited) {\n");
        self.w("        maka_fiber_t** prev = &maka_fd_waiters;\n");
        self.w("        while (*prev) {\n");
        self.w("            maka_fiber_t* w = *prev;\n");
        self.w("            if (w->waiting_fd == (int)fd) {\n");
        self.w("                *prev = w->next; w->next = NULL;\n");
        self.w("                w->waiting_fd = -1; w->waiting_events = 0;\n");
        self.w("                w->wait_deadline_ns = 0; w->wait_timed_out = 0;\n");
        self.w("                __maka_ready_enqueue(w);\n");
        self.w("            } else {\n");
        self.w("                prev = &(*prev)->next;\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("    }\n");
        // Broadcast a remote close request to every other live scheduler so
        // they wake their own fd_waiters on this fd.
        self.w("    pthread_mutex_lock(&__maka_ticks_mu);\n");
        self.w("    for (maka_sched_tick_t* tk = __maka_ticks_head; tk; tk = tk->next) {\n");
        self.w("        maka_sched_state_t* o = tk->owner;\n");
        self.w("        if (!o || o == maka_sched_state) continue;\n");
        self.w("        maka_close_req_t* cr = (maka_close_req_t*)malloc(sizeof(maka_close_req_t));\n");
        self.w("        cr->closed_fd = (int)fd;\n");
        self.w("        pthread_mutex_lock(&o->remote_mu);\n");
        self.w("        cr->next = o->remote_close_head;\n");
        self.w("        o->remote_close_head = cr;\n");
        self.w("        int wfd = o->wake_pipe_w;\n");
        self.w("        pthread_mutex_unlock(&o->remote_mu);\n");
        self.w("        if (wfd > 0) { char b = 1; (void)send((int)wfd, &b, 1, 0); }\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&__maka_ticks_mu);\n");
        self.w("    __maka_fd_arm((int)fd, 0);\n");
        self.w("    __maka_fd_reg_drop((int)fd);\n");
        self.w("    return close((int)fd);\n");
        self.w("}\n");
        // HTTP request line parser — given a request buffer, returns the
        // method, path, and Content-Length as offset+len in pieces.
        // Caller exposes them via separate getters (Maka can't return tuples).
        self.w("static __thread int __maka_http_method_off = -1;\n");
        self.w("static __thread int __maka_http_method_len = 0;\n");
        self.w("static __thread int __maka_http_path_off = -1;\n");
        self.w("static __thread int __maka_http_path_len = 0;\n");
        self.w("static __thread int __maka_http_body_off = -1;\n");
        self.w("static __thread int64_t __maka_http_content_length = -1;\n");
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
        self.w("                int64_t v = 0;\n");
        self.w("                while (p < len && buf[p] >= '0' && buf[p] <= '9') {\n");
        self.w("                    if (v > (int64_t)9223372036854775807LL / 10) { v = -1; break; }\n");
        self.w("                    v = v * 10 + (buf[p] - '0'); p++; }\n");
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
        // to a fiber" pattern from needing a cblock helper.  Don't re-extern
        // on Darwin/BSD — their <unistd.h> already declares pipe() with an
        // asm-aliased prototype that conflicts with a re-declaration.  On
        // Linux/Windows the runtime needs the forward decl since we don't
        // pull <unistd.h> on all paths.
        self.w("#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__) && !defined(__DragonFly__)\n");
        self.w("extern int pipe(int*);\n");
        self.w("#endif\n");
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
        self.w("static SSL_CTX* __maka_tls_server_ctx = NULL;\n");
        self.w("static inline int64_t __maka_tls_server_init(const char* cert_pem, const char* key_pem) {\n");
        self.w("    __maka_tls_init_once();\n");
        self.w("    if (__maka_tls_server_ctx) SSL_CTX_free(__maka_tls_server_ctx);\n");
        self.w("    __maka_tls_server_ctx = SSL_CTX_new(TLS_server_method());\n");
        self.w("    if (!__maka_tls_server_ctx) return -1;\n");
        self.w("    if (SSL_CTX_use_certificate_file(__maka_tls_server_ctx, cert_pem, SSL_FILETYPE_PEM) <= 0) return -1;\n");
        self.w("    if (SSL_CTX_use_PrivateKey_file (__maka_tls_server_ctx, key_pem, SSL_FILETYPE_PEM) <= 0) return -1;\n");
        self.w("    return 0;\n");
        self.w("}\n");
        self.w("static inline maka_unit* __maka_tls_server_accept_new(int64_t fd) {\n");
        self.w("    if (!__maka_tls_server_ctx) return NULL;\n");
        self.w("    SSL* s = SSL_new(__maka_tls_server_ctx);\n");
        self.w("    if (!s) return NULL;\n");
        self.w("    SSL_set_fd(s, (int)fd);\n");
        self.w("    while (1) {\n");
        self.w("        int r = SSL_accept(s);\n");
        self.w("        if (r == 1) return (maka_unit*)s;\n");
        self.w("        int e = SSL_get_error(s, r);\n");
        self.w("        int sfd = SSL_get_fd(s);\n");
        self.w("        if (e == SSL_ERROR_WANT_READ)  { __maka_wait_fd(sfd, MAKA_EV_READ); continue; }\n");
        self.w("        if (e == SSL_ERROR_WANT_WRITE) { __maka_wait_fd(sfd, MAKA_EV_WRITE); continue; }\n");
        self.w("        SSL_free(s); return NULL;\n");
        self.w("    }\n");
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
        self.w("static inline int64_t __maka_tls_server_init(const char* cert_pem, const char* key_pem) { (void)cert_pem; (void)key_pem; return -1; }\n");
        self.w("static inline maka_unit* __maka_tls_server_accept_new(int64_t fd) { (void)fd; return NULL; }\n");
        self.w("static inline maka_unit* __maka_tls_client_new(int64_t fd, const char* hostname) { (void)fd; (void)hostname; return NULL; }\n");
        self.w("static inline int64_t __maka_tls_handshake(maka_unit* p) { (void)p; return -1; }\n");
        self.w("static inline int64_t __maka_tls_read(maka_unit* p, maka_unit* buf, int64_t cap) { (void)p; (void)buf; (void)cap; return -1; }\n");
        self.w("static inline int64_t __maka_tls_write(maka_unit* p, maka_unit* buf, int64_t len) { (void)p; (void)buf; (void)len; return -1; }\n");
        self.w("static inline void __maka_tls_close(maka_unit* p) { (void)p; }\n");
        self.w("#endif\n");
        // Unix domain sockets — bind/connect by path.  On Darwin/BSD use the
        // real sockaddr_un from <sys/un.h> (has sun_len + correct AF_UNIX);
        // on Linux + non-Win, keep the manual layout (no sun_len, AF_UNIX=1).
        self.w("#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)\n");
        self.w("#define __maka_sockaddr_un sockaddr_un\n");
        self.w("#ifndef AF_UNIX\n");
        self.w("#define AF_UNIX 1\n");
        self.w("#endif\n");
        self.w("#define __MAKA_AF_UNIX AF_UNIX\n");
        self.w("#define __MAKA_SUN_LEN_INIT(sa) ((sa).sun_len = sizeof(sa))\n");
        self.w("#else\n");
        self.w("struct __maka_sockaddr_un { unsigned short sun_family; char sun_path[108]; };\n");
        self.w("#define __MAKA_AF_UNIX 1\n");
        self.w("#define __MAKA_SUN_LEN_INIT(sa) ((void)0)\n");
        self.w("#endif\n");
        self.w("#ifdef _WIN32\n");
        // Windows AF_UNIX exists on 1803+ but filesystem-path bind semantics
        // are clunky.  Emulate with TCP loopback keyed by path → port stored
        // in a tiny table.  Server creates a 127.0.0.1:0 listener and
        // registers the path → port mapping; client looks up and connects.
        self.w("#define __MAKA_UNIX_MAX 32\n");
        self.w("typedef struct { char path[256]; int port; } __maka_unix_entry_t;\n");
        self.w("static __maka_unix_entry_t __maka_unix_table[__MAKA_UNIX_MAX];\n");
        self.w("static pthread_mutex_t __maka_unix_mu = PTHREAD_MUTEX_INITIALIZER;\n");
        self.w("static int __maka_unix_register(const char* path, int port) {\n");
        self.w("    pthread_mutex_lock(&__maka_unix_mu);\n");
        self.w("    for (int i = 0; i < __MAKA_UNIX_MAX; i++) {\n");
        self.w("        if (__maka_unix_table[i].port == 0 || strcmp(__maka_unix_table[i].path, path) == 0) {\n");
        self.w("            strncpy(__maka_unix_table[i].path, path, sizeof(__maka_unix_table[i].path)-1);\n");
        self.w("            __maka_unix_table[i].port = port;\n");
        self.w("            pthread_mutex_unlock(&__maka_unix_mu); return 0;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&__maka_unix_mu); return -1;\n");
        self.w("}\n");
        self.w("static int __maka_unix_lookup(const char* path) {\n");
        self.w("    pthread_mutex_lock(&__maka_unix_mu);\n");
        self.w("    for (int i = 0; i < __MAKA_UNIX_MAX; i++) {\n");
        self.w("        if (__maka_unix_table[i].port != 0 && strcmp(__maka_unix_table[i].path, path) == 0) {\n");
        self.w("            int p = __maka_unix_table[i].port;\n");
        self.w("            pthread_mutex_unlock(&__maka_unix_mu); return p;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&__maka_unix_mu); return -1;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_unix_listen(const char* path, int64_t backlog) {\n");
        self.w("    int s = socket(AF_INET, SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa)); __MAKA_SA_LEN_INIT(sa);\n");
        self.w("    sa.sin_family = AF_INET;\n");
        self.w("    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);\n");
        self.w("    sa.sin_port = 0;\n");
        self.w("    if (bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) { closesocket(s); return -1; }\n");
        self.w("    int sl = sizeof(sa);\n");
        self.w("    if (getsockname(s, (struct sockaddr*)&sa, &sl) != 0) { closesocket(s); return -1; }\n");
        self.w("    if (listen(s, (int)backlog) != 0) { closesocket(s); return -1; }\n");
        self.w("    u_long nb = 1; ioctlsocket((SOCKET)s, FIONBIO, &nb);\n");
        self.w("    if (__maka_unix_register(path, (int)ntohs(sa.sin_port)) != 0) { closesocket(s); return -1; }\n");
        self.w("    return s;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_unix_connect(const char* path) {\n");
        self.w("    /* Allow the listener side a brief moment to register if the\n");
        self.w("       client raced ahead. */\n");
        self.w("    int port = -1;\n");
        self.w("    for (int tries = 0; tries < 200; tries++) {\n");
        self.w("        port = __maka_unix_lookup(path);\n");
        self.w("        if (port > 0) break;\n");
        self.w("        Sleep(5);\n");
        self.w("    }\n");
        self.w("    if (port <= 0) return -1;\n");
        self.w("    int s = socket(AF_INET, SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa)); __MAKA_SA_LEN_INIT(sa);\n");
        self.w("    sa.sin_family = AF_INET;\n");
        self.w("    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);\n");
        self.w("    sa.sin_port = htons((u_short)port);\n");
        self.w("    if (connect(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) { closesocket(s); return -1; }\n");
        self.w("    u_long nb = 1; ioctlsocket((SOCKET)s, FIONBIO, &nb);\n");
        self.w("    return s;\n");
        self.w("}\n");
        self.w("#else\n");
        self.w("static inline int64_t __maka_unix_listen(const char* path, int64_t backlog) {\n");
        self.w("    int s = socket(__MAKA_AF_UNIX, __MAKA_SOCK_STREAM, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    struct __maka_sockaddr_un sa; memset(&sa, 0, sizeof(sa)); __MAKA_SUN_LEN_INIT(sa);\n");
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
        self.w("    struct __maka_sockaddr_un sa; memset(&sa, 0, sizeof(sa)); __MAKA_SUN_LEN_INIT(sa);\n");
        self.w("    sa.sun_family = __MAKA_AF_UNIX;\n");
        self.w("    size_t pl = strlen(path); if (pl >= sizeof(sa.sun_path)) pl = sizeof(sa.sun_path) - 1;\n");
        self.w("    memcpy(sa.sun_path, path, pl);\n");
        self.w("    int r = connect(s, (struct sockaddr*)&sa, sizeof(sa));\n");
        self.w("    if (r == 0) return s;\n");
        self.w("    if (errno == EINPROGRESS) {\n");
        self.w("        __maka_wait_fd(s, MAKA_EV_WRITE);\n");
        self.w("        int err = 0; __maka_socklen_t elen = sizeof(err);\n");
        self.w("        getsockopt(s, __MAKA_SOL_SOCKET, __MAKA_SO_ERROR, (char*)&err, &elen);\n");
        self.w("        if (err == 0) return s;\n");
        self.w("    }\n");
        self.w("    close(s); return -1;\n");
        self.w("}\n");
        self.w("#endif\n");
        // File async IO via offload thread.  Each call spawns a one-shot
        // pthread that does the blocking pread/pwrite, then signals an
        // eventfd the calling fiber waits on.  Heavy per call (pthread
        // creation) but correct without an io_uring/AIO dependency.
        self.w("typedef struct {\n");
        self.w("    int fd;\n");
        self.w("    void* buf;\n");
        self.w("    int64_t len;\n");
        self.w("    int64_t offset;\n");
        self.w("    _Atomic int64_t result;\n");
        self.w("    int efd;\n");
        self.w("    int is_write;\n");
        // Refcount: both the worker pthread (detached) and the calling fiber
        // hold a ref.  Each drops after it's done with `j`; last drop frees.
        // Lets the fiber's eventfd_recv yield to other fibers (no pthread_join
        // blocking the whole scheduler thread) while still avoiding UAF.
        self.w("    _Atomic int refcount;\n");
        self.w("} __maka_aio_t;\n");
        // Linux uses real eventfd (single kernel fd); the aux-table machinery
        // is non-Linux only.  Provide a no-op stub so file_read_async /
        // file_write_async can call the release unconditionally.  Non-Linux
        // gets a forward decl here; the real body is defined alongside the
        // aux table further down.
        self.w("#ifdef __linux__\n");
        self.w("static inline void __maka_aux_release(int read_fd) { (void)read_fd; }\n");
        self.w("#else\n");
        self.w("static void __maka_aux_release(int read_fd);\n");
        self.w("#endif\n");
        self.w("#ifndef _WIN32\n");
        // Use size_t / off_t (POSIX-correct) — `long` is 32-bit on Win64 and
        // would silently cap large-file IO.  The Windows path uses our own
        // 64-bit-clean shims declared earlier in the prologue.
        self.w("#include <sys/types.h>\n");
        self.w("extern ssize_t pread (int, void*,       size_t, off_t);\n");
        self.w("extern ssize_t pwrite(int, const void*, size_t, off_t);\n");
        self.w("#endif\n");
        self.w("static void* __maka_aio_worker(void* arg) {\n");
        self.w("    __maka_aio_t* j = (__maka_aio_t*)arg;\n");
        self.w("    int64_t r;\n");
        self.w("#ifdef _WIN32\n");
        self.w("    if (j->is_write) r = (int64_t)pwrite(j->fd, j->buf, (size_t)j->len, (int64_t)j->offset);\n");
        self.w("    else             r = (int64_t)pread (j->fd, j->buf, (size_t)j->len, (int64_t)j->offset);\n");
        self.w("#else\n");
        self.w("    if (j->is_write) r = (int64_t)pwrite(j->fd, j->buf, (size_t)j->len, (off_t)j->offset);\n");
        self.w("    else             r = (int64_t)pread (j->fd, j->buf, (size_t)j->len, (off_t)j->offset);\n");
        self.w("#endif\n");
        // Release-store: pairs with the acquire-load in the caller after eventfd_recv.
        self.w("    atomic_store_explicit(&j->result, r, memory_order_release);\n");
        self.w("#ifdef _WIN32\n");
        self.w("    (void)__maka_eventfd_signal(j->efd, 1);\n");
        self.w("#else\n");
        self.w("    uint64_t v = 1; ssize_t w = write(j->efd, &v, sizeof(v)); (void)w;\n");
        self.w("#endif\n");
        // Drop worker ref; if caller already dropped, free.
        self.w("    if (atomic_fetch_sub(&j->refcount, 1) == 1) free(j);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_file_read_async(int64_t fd, maka_unit* buf, int64_t cap, int64_t offset) {\n");
        self.w("    int efd = __maka_eventfd_create(0);\n");
        self.w("    if (efd < 0) return -1;\n");
        self.w("    __maka_aio_t* j = (__maka_aio_t*)calloc(1, sizeof(__maka_aio_t));\n");
        self.w("    j->fd = (int)fd; j->buf = (void*)buf; j->len = cap; j->offset = offset; j->efd = efd; j->is_write = 0;\n");
        self.w("    atomic_init(&j->refcount, 2);  /* worker + caller */\n");
        self.w("    pthread_t t;\n");
        self.w("    if (pthread_create(&t, NULL, __maka_aio_worker, j) != 0) {\n");
        self.w("        free(j);\n");
        self.w("#ifndef __linux__\n");
        self.w("        __maka_aux_release(efd);\n");
        self.w("#endif\n");
        self.w("        close(efd);\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("    pthread_detach(t);\n");
        // Tag the calling fiber's completion so cancel won't free it while
        // the detached worker still holds buf/fd refs.  Cleared after recv.
        self.w("    Thread* aio_owner = (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) ? maka_current_fiber->completion : NULL;\n");
        self.w("    if (aio_owner) atomic_store(&aio_owner->aio_in_flight, 1);\n");
        self.w("    (void)__maka_eventfd_recv(efd);\n");
        self.w("    if (aio_owner) atomic_store(&aio_owner->aio_in_flight, 0);\n");
        // Acquire-load: pairs with worker's release-store of result.
        self.w("    int64_t r = atomic_load_explicit(&j->result, memory_order_acquire);\n");
        self.w("    if (atomic_fetch_sub(&j->refcount, 1) == 1) free(j);\n");
        self.w("#ifndef __linux__\n");
        self.w("    __maka_aux_release(efd);\n");
        self.w("#endif\n");
        self.w("    close(efd);\n");
        self.w("    return r;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_file_write_async(int64_t fd, maka_unit* buf, int64_t len, int64_t offset) {\n");
        self.w("    int efd = __maka_eventfd_create(0);\n");
        self.w("    if (efd < 0) return -1;\n");
        self.w("    __maka_aio_t* j = (__maka_aio_t*)calloc(1, sizeof(__maka_aio_t));\n");
        self.w("    j->fd = (int)fd; j->buf = (void*)buf; j->len = len; j->offset = offset; j->efd = efd; j->is_write = 1;\n");
        self.w("    atomic_init(&j->refcount, 2);  /* worker + caller */\n");
        self.w("    pthread_t t;\n");
        self.w("    if (pthread_create(&t, NULL, __maka_aio_worker, j) != 0) {\n");
        self.w("        free(j);\n");
        self.w("#ifndef __linux__\n");
        self.w("        __maka_aux_release(efd);\n");
        self.w("#endif\n");
        self.w("        close(efd);\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("    pthread_detach(t);\n");
        // Tag the calling fiber's completion so cancel won't free it while
        // the detached worker still holds buf/fd refs.  Cleared after recv.
        self.w("    Thread* aio_owner = (maka_current_fiber && maka_current_fiber != maka_anchor_fiber) ? maka_current_fiber->completion : NULL;\n");
        self.w("    if (aio_owner) atomic_store(&aio_owner->aio_in_flight, 1);\n");
        self.w("    (void)__maka_eventfd_recv(efd);\n");
        self.w("    if (aio_owner) atomic_store(&aio_owner->aio_in_flight, 0);\n");
        // Acquire-load: pairs with worker's release-store of result.
        self.w("    int64_t r = atomic_load_explicit(&j->result, memory_order_acquire);\n");
        self.w("    if (atomic_fetch_sub(&j->refcount, 1) == 1) free(j);\n");
        self.w("#ifndef __linux__\n");
        self.w("    __maka_aux_release(efd);\n");
        self.w("#endif\n");
        self.w("    close(efd);\n");
        self.w("    return r;\n");
        self.w("}\n");
        // open() is declared by <fcntl.h> with an asm-aliased prototype on
        // Darwin/BSD; re-declaring it here would conflict.  Linux's
        // <fcntl.h> is permissive but Maka doesn't pull it everywhere, so
        // keep the extern decl for Linux only.
        self.w("#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__) && !defined(__DragonFly__)\n");
        self.w("extern int open(const char*, int, ...);\n");
        self.w("#endif\n");
        self.w("#define __MAKA_O_RDONLY 0\n");
        self.w("#define __MAKA_O_WRONLY 1\n");
        self.w("#define __MAKA_O_RDWR   2\n");
        self.w("#define __MAKA_O_CREAT  64\n");
        self.w("#define __MAKA_O_TRUNC  512\n");
        self.w("static inline int64_t __maka_file_open(const char* path, int64_t flags, int64_t mode) {\n");
        self.w("#ifdef _WIN32\n");
        // Maka uses Linux O_* values; remap to mingw's O_* before calling open().
        self.w("    int wf = 0;\n");
        self.w("    int access = (int)flags & 3;\n");
        self.w("    if (access == 0) wf |= _O_RDONLY;\n");
        self.w("    else if (access == 1) wf |= _O_WRONLY;\n");
        self.w("    else if (access == 2) wf |= _O_RDWR;\n");
        self.w("    if (flags & 64)    wf |= _O_CREAT;\n");
        self.w("    if (flags & 512)   wf |= _O_TRUNC;\n");
        self.w("    if (flags & 1024)  wf |= _O_APPEND;\n");
        self.w("    /* O_NONBLOCK (2048) has no mingw equivalent - silently ignored. */\n");
        self.w("    wf |= _O_BINARY;\n");
        self.w("    WCHAR* wpath = __maka_path_to_w(path);\n");
        self.w("    if (!wpath) return -1;\n");
        self.w("    int rc = _wopen(wpath, wf, (int)mode);\n");
        self.w("    free(wpath); return (int64_t)rc;\n");
        self.w("#else\n");
        self.w("    return (int64_t)open(path, (int)flags, (int)mode);\n");
        self.w("#endif\n");
        self.w("}\n");
        // Filesystem ops: unlink, rename, mkdir, file_size, file_sync (fsync),
        // file_truncate.  Cross-platform: Windows uses _unlink/_mkdir/_chsize64.
        // Note the symbols use a `_rt` suffix to dodge mingw's `_unlink`/`_mkdir`
        // macros and to avoid linkage clashes with Maka's extern decls.
        self.w("int64_t __maka_rt_file_unlink(const char* path) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    WCHAR* wp = __maka_path_to_w(path); if (!wp) return -1;\n");
        self.w("    int rc = _wunlink(wp); free(wp); return rc;\n");
        self.w("#else\n");
        self.w("    return unlink(path);\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_rename(const char* from, const char* to) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    WCHAR* wf = __maka_path_to_w(from); if (!wf) return -1;\n");
        self.w("    WCHAR* wt = __maka_path_to_w(to);   if (!wt) { free(wf); return -1; }\n");
        self.w("    int rc = _wrename(wf, wt); free(wf); free(wt); return rc;\n");
        self.w("#else\n");
        self.w("    return rename(from, to);\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_mkdir(const char* path, int64_t mode) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    (void)mode;\n");
        self.w("    WCHAR* wp = __maka_path_to_w(path); if (!wp) return -1;\n");
        self.w("    int rc = _wmkdir(wp); free(wp); return rc;\n");
        self.w("#else\n");
        self.w("    return mkdir(path, (mode_t)mode);\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_size(const char* path) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    struct __stat64 st;\n");
        self.w("    WCHAR* wp = __maka_path_to_w(path); if (!wp) return -1;\n");
        self.w("    int sr = _wstat64(wp, &st); free(wp); if (sr != 0) return -1;\n");
        self.w("    return (int64_t)st.st_size;\n");
        self.w("#else\n");
        self.w("    struct stat st;\n");
        self.w("    if (stat(path, &st) != 0) return -1;\n");
        self.w("    return (int64_t)st.st_size;\n");
        self.w("#endif\n");
        self.w("}\n");
        // Whole-file read: returns a freshly-malloc'd, NUL-terminated buffer of
        // the file's contents (NULL on error).  Maka's `read_file` wraps this as
        // an owned String, so the return type is `char*` to match.
        self.w("char* __maka_rt_read_file(const char* path) {\n");
        self.w("    FILE* fp = fopen(path, \"rb\"); if (!fp) return (char*)0;\n");
        self.w("    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return (char*)0; }\n");
        self.w("    long n = ftell(fp); if (n < 0) { fclose(fp); return (char*)0; }\n");
        self.w("    rewind(fp);\n");
        self.w("    char* buf = (char*)malloc((size_t)n + 1); if (!buf) { fclose(fp); return (char*)0; }\n");
        self.w("    size_t rd = fread(buf, 1, (size_t)n, fp); fclose(fp); buf[rd] = 0; return (char*)buf;\n");
        self.w("}\n");
        // Owned substring: a freshly-malloc'd copy of s[start .. start+len].
        // Returns `char*` so Maka can own/free it (the borrowed
        // `__maka_rt_str_substring` would leak it inside a Vec<String>).
        // `slen` is the caller-known source length (callers already have it, so we
        // avoid an O(n) strlen here).  start/len are clamped into [0, slen] so an
        // out-of-range request copies only valid bytes instead of reading past the
        // buffer (safe by construction, never an OOB read).
        self.w("char* __maka_rt_substr_owned(const char* s, int64_t start, int64_t len, int64_t slen) {\n");
        self.w("    if (slen < 0) slen = 0;\n");
        self.w("    if (start < 0) start = 0; if (start > slen) start = slen;\n");
        self.w("    if (len < 0) len = 0; if (len > slen - start) len = slen - start;\n");
        self.w("    char* r = (char*)malloc((size_t)len + 1);\n");
        self.w("    for (int64_t k = 0; k < len; k++) r[k] = s[start + k];\n");
        self.w("    r[len] = 0; return (char*)r;\n");
        self.w("}\n");
        // Whole-file write: writes `len` bytes; returns 0 on success, -1 on error.
        self.w("int64_t __maka_rt_write_file(const char* path, const char* data, int64_t len) {\n");
        self.w("    FILE* fp = fopen(path, \"wb\"); if (!fp) return -1;\n");
        self.w("    size_t wr = fwrite(data, 1, (size_t)len, fp); fclose(fp);\n");
        self.w("    return (wr == (size_t)len) ? 0 : -1;\n");
        self.w("}\n");
        // FNV-1a 64-bit hash of a NUL-terminated string (for HashMap).
        self.w("int64_t __maka_rt_str_hash(const char* s) {\n");
        self.w("    uint64_t h = 1469598103934665603ULL;\n");
        self.w("    while (*s) { h ^= (unsigned char)(*s++); h *= 1099511628211ULL; }\n");
        self.w("    return (int64_t)(h & 0x7fffffffffffffffULL);\n");
        self.w("}\n");
        // Byte at index `i` of a string (0..len-1), or -1 out of range.  Gives
        // Maka byte-level read access to strings (which aren't indexable).
        self.w("int64_t __maka_rt_str_byte(const char* s, int64_t i) {\n");
        self.w("    if (i < 0 || (uint64_t)i >= strlen(s)) return -1;\n");
        self.w("    return (int64_t)(unsigned char)s[i];\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_sync(int64_t fd) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    HANDLE h = (HANDLE)_get_osfhandle((int)fd);\n");
        self.w("    if (h == INVALID_HANDLE_VALUE) return -1;\n");
        self.w("    return FlushFileBuffers(h) ? 0 : -1;\n");
        self.w("#else\n");
        self.w("    return fsync((int)fd);\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_truncate(int64_t fd, int64_t len) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    return _chsize_s((int)fd, (__int64)len);\n");
        self.w("#else\n");
        self.w("    return ftruncate((int)fd, (off_t)len);\n");
        self.w("#endif\n");
        self.w("}\n");
        // Channel close + try_recv shims — pure wrappers around the existing
        // bytes-channel impls so the stdlib can expose them with typed
        // signatures.
        self.w("static inline int64_t __maka_chan_try_recv_int(maka_unit* p, int64_t* out) {\n");
        self.w("    /* fast non-blocking peek: if count > 0, do the recv (which won't block). */\n");
        self.w("    if (!p || !out) return 0;\n");
        self.w("    int64_t v = 0; if (!maka_chan_bytes_try_recv(p, (maka_unit*)&v)) return 0;\n");
        self.w("    *out = v; return 1;\n");
        self.w("}\n");
        // Convenience: atomic_bool / atomic_ptr — internally back to the int64
        // atomic.  bool packs 0/1; ptr round-trips through (intptr_t).
        // Float atomics need a small helper since C11 atomic_load on float
        // isn't reliably supported by older mingw — use atomic int64 + bits.
        self.w("maka_unit* __maka_atomic_bool_new(int64_t v) { return maka_atomic_i64_new(v ? 1 : 0); }\n");
        self.w("int64_t __maka_atomic_bool_get(maka_unit* a) { return maka_atomic_i64_load(a) ? 1 : 0; }\n");
        self.w("void    __maka_atomic_bool_set(maka_unit* a, int64_t v) { maka_atomic_i64_store(a, v ? 1 : 0); }\n");
        self.w("maka_unit* __maka_atomic_ptr_new(maka_unit* p) { return maka_atomic_i64_new((int64_t)(intptr_t)p); }\n");
        self.w("maka_unit* __maka_atomic_ptr_get(maka_unit* a) { return (maka_unit*)(intptr_t)maka_atomic_i64_load(a); }\n");
        self.w("void       __maka_atomic_ptr_set(maka_unit* a, maka_unit* p) { maka_atomic_i64_store(a, (int64_t)(intptr_t)p); }\n");
        // --- Misc filesystem + env + process + string conversion + RNG helpers.
        self.w("int64_t __maka_rt_file_exists(const char* path) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    struct __stat64 st;\n");
        self.w("    WCHAR* wp = __maka_path_to_w(path); if (!wp) return 0;\n");
        self.w("    int sr = _wstat64(wp, &st); free(wp); return sr == 0 ? 1 : 0;\n");
        self.w("#else\n");
        self.w("    struct stat st; return stat(path, &st) == 0 ? 1 : 0;\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_is_dir(const char* path) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    struct __stat64 st;\n");
        self.w("    WCHAR* wp = __maka_path_to_w(path); if (!wp) return 0;\n");
        self.w("    int sr = _wstat64(wp, &st); free(wp); if (sr != 0) return 0;\n");
        // mingw-w64's `sys/stat.h` may hide both `S_IFDIR` and `_S_IFDIR`
        // depending on which POSIX feature macros are defined.  Use the
        // literal bit mask — it's stable in the Windows FAT/NTFS metadata
        // layout (`0x4000`).
        self.w("    return ((st.st_mode & 0xF000) == 0x4000) ? 1 : 0;\n");
        self.w("#else\n");
        // Use the raw mode mask instead of the S_ISDIR macro so we don't
        // depend on <sys/stat.h> being pulled in everywhere; 040000 octal is
        // the directory bit on every POSIX-compliant system.
        self.w("    struct stat st; if (stat(path, &st) != 0) return 0;\n");
        self.w("    return ((st.st_mode & 0170000) == 0040000) ? 1 : 0;\n");
        self.w("#endif\n");
        self.w("}\n");
        // Returns the env var value as a freshly-malloc'd string, or "" if
        // unset.  NULL would segfault Maka's printf-based log().
        self.w("char* __maka_rt_env_get(const char* name) {\n");
        self.w("    const char* v = getenv(name);\n");
        // Return a fresh empty string (not a literal) so callers that free()
        // returned strings don't crash on unset env vars.
        self.w("    if (!v) { char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    size_t n = strlen(v);\n");
        self.w("    char* s = (char*)malloc(n + 1);\n");
        self.w("    memcpy(s, v, n + 1);\n");
        self.w("    return s;\n");
        self.w("}\n");
        self.w("void __maka_rt_process_exit(int64_t code) { exit((int)code); }\n");
        // String conversions + search.
        self.w("int64_t __maka_rt_str_to_int(const char* s) {\n");
        self.w("    if (!s) return 0;\n");
        self.w("    char* end; long long v = strtoll(s, &end, 10); return (int64_t)v;\n");
        self.w("}\n");
        self.w("char* __maka_rt_int_to_str(int64_t n) {\n");
        self.w("    char buf[32]; int len = snprintf(buf, sizeof(buf), \"%lld\", (long long)n);\n");
        self.w("    if (len < 0) len = 0;\n");
        self.w("    char* s = (char*)malloc((size_t)len + 1);\n");
        self.w("    memcpy(s, buf, (size_t)len + 1);\n");
        self.w("    return (char*)s;\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_str_find(const char* s, const char* needle) {\n");
        self.w("    if (!s || !needle) return -1;\n");
        self.w("    const char* p = strstr(s, needle);\n");
        self.w("    return p ? (int64_t)(p - s) : -1;\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_str_starts_with(const char* s, const char* prefix) {\n");
        self.w("    if (!s || !prefix) return 0;\n");
        self.w("    size_t pl = strlen(prefix);\n");
        self.w("    return strncmp(s, prefix, pl) == 0 ? 1 : 0;\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_str_ends_with(const char* s, const char* suffix) {\n");
        self.w("    if (!s || !suffix) return 0;\n");
        self.w("    size_t sl = strlen(s), xl = strlen(suffix);\n");
        self.w("    if (xl > sl) return 0;\n");
        self.w("    return strcmp(s + sl - xl, suffix) == 0 ? 1 : 0;\n");
        self.w("}\n");
        self.w("char* __maka_rt_str_to_upper(const char* s) {\n");
        self.w("    if (!s) { char* e = (char*)malloc(1); e[0] = 0; return (char*)e; }\n");
        self.w("    size_t l = strlen(s);\n");
        self.w("    char* o = (char*)malloc(l + 1);\n");
        self.w("    for (size_t i = 0; i < l; i++) {\n");
        self.w("        char c = s[i]; o[i] = (c >= 'a' && c <= 'z') ? (c - 32) : c;\n");
        self.w("    }\n");
        self.w("    o[l] = 0; return (char*)o;\n");
        self.w("}\n");
        self.w("char* __maka_rt_str_to_lower(const char* s) {\n");
        self.w("    if (!s) { char* e = (char*)malloc(1); e[0] = 0; return (char*)e; }\n");
        self.w("    size_t l = strlen(s);\n");
        self.w("    char* o = (char*)malloc(l + 1);\n");
        self.w("    for (size_t i = 0; i < l; i++) {\n");
        self.w("        char c = s[i]; o[i] = (c >= 'A' && c <= 'Z') ? (c + 32) : c;\n");
        self.w("    }\n");
        self.w("    o[l] = 0; return (char*)o;\n");
        self.w("}\n");
        self.w("char* __maka_rt_str_trim(const char* s) {\n");
        self.w("    if (!s) { char* e = (char*)malloc(1); e[0] = 0; return (char*)e; }\n");
        self.w("    const char* p = s;\n");
        self.w("    while (*p == ' ' || *p == '\\t' || *p == '\\n' || *p == '\\r') p++;\n");
        self.w("    const char* e = s + strlen(s);\n");
        self.w("    while (e > p && (e[-1] == ' ' || e[-1] == '\\t' || e[-1] == '\\n' || e[-1] == '\\r')) e--;\n");
        self.w("    size_t l = (size_t)(e - p);\n");
        self.w("    char* o = (char*)malloc(l + 1);\n");
        self.w("    memcpy(o, p, l); o[l] = 0; return (char*)o;\n");
        self.w("}\n");
        self.w("char* __maka_rt_str_replace(const char* s, const char* from, const char* to) {\n");
        // Always return a freshly malloc'd string so callers that free() the
        // result don't crash on degenerate inputs.
        self.w("    if (!s) { char* e = (char*)malloc(1); e[0] = 0; return (char*)e; }\n");
        self.w("    if (!from || !*from || !to) {\n");
        self.w("        size_t sl = strlen(s);\n");
        self.w("        char* o = (char*)malloc(sl + 1);\n");
        self.w("        memcpy(o, s, sl + 1); return (char*)o;\n");
        self.w("    }\n");
        self.w("    size_t fl = strlen(from), tl = strlen(to);\n");
        // Count occurrences for sizing.
        self.w("    size_t count = 0;\n");
        self.w("    for (const char* p = s; (p = strstr(p, from)); p += fl) count++;\n");
        self.w("    size_t sl = strlen(s);\n");
        self.w("    size_t out_len = sl + count * (tl > fl ? (tl - fl) : 0) - count * (fl > tl ? (fl - tl) : 0);\n");
        self.w("    char* o = (char*)malloc(out_len + 1);\n");
        self.w("    char* w = o;\n");
        self.w("    const char* r = s;\n");
        self.w("    while (1) {\n");
        self.w("        const char* p = strstr(r, from);\n");
        self.w("        if (!p) { size_t rem = strlen(r); memcpy(w, r, rem); w += rem; break; }\n");
        self.w("        size_t pref = (size_t)(p - r);\n");
        self.w("        memcpy(w, r, pref); w += pref;\n");
        self.w("        memcpy(w, to, tl); w += tl;\n");
        self.w("        r = p + fl;\n");
        self.w("    }\n");
        self.w("    *w = 0; return (char*)o;\n");
        self.w("}\n");
        // Random — use a per-thread xorshift seeded from clock + addr.
        self.w("static __thread uint64_t __maka_rt_rng_state = 0;\n");
        self.w("static inline uint64_t __maka_rt_rng_next(void) {\n");
        self.w("    if (__maka_rt_rng_state == 0) {\n");
        self.w("        __maka_rt_rng_state = (uint64_t)__maka_now_ns() ^ (uint64_t)(uintptr_t)&__maka_rt_rng_state;\n");
        self.w("        if (__maka_rt_rng_state == 0) __maka_rt_rng_state = 0x9E3779B97F4A7C15ULL;\n");
        self.w("    }\n");
        self.w("    uint64_t x = __maka_rt_rng_state;\n");
        self.w("    x ^= x << 13; x ^= x >> 7; x ^= x << 17;\n");
        self.w("    __maka_rt_rng_state = x; return x;\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_random_int(int64_t lo, int64_t hi) {\n");
        self.w("    if (hi <= lo) return lo;\n");
        // Cast each bound to uint64 BEFORE subtraction so INT64_MIN..INT64_MAX
        // ranges don't trigger signed overflow UB.  Then rejection-sample to
        // eliminate modulo bias.
        self.w("    uint64_t range = (uint64_t)hi - (uint64_t)lo;\n");
        self.w("    uint64_t limit = UINT64_MAX - (UINT64_MAX % range);\n");
        self.w("    uint64_t r;\n");
        self.w("    do { r = __maka_rt_rng_next(); } while (r >= limit);\n");
        self.w("    return (int64_t)((uint64_t)lo + (r % range));\n");
        self.w("}\n");
        self.w("double __maka_rt_random_float(void) {\n");
        // 53 bits → [0, 1)
        self.w("    return (double)(__maka_rt_rng_next() >> 11) * (1.0 / 9007199254740992.0);\n");
        self.w("}\n");
        // chan_try_recv_int — non-blocking peek + recv.  Returns 1 + writes
        // *out, or 0 if the channel was empty.
        self.w("int64_t __maka_rt_chan_try_recv_int(maka_unit* p, int64_t* out) {\n");
        self.w("    if (!p || !out) return 0;\n");
        self.w("    int64_t v = 0; if (!maka_chan_bytes_try_recv(p, (maka_unit*)&v)) return 0;\n");
        self.w("    *out = v; return 1;\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_chan_try_recv_float(maka_unit* p, double* out) {\n");
        self.w("    if (!p || !out) return 0;\n");
        self.w("    double v = 0; if (!maka_chan_bytes_try_recv(p, (maka_unit*)&v)) return 0;\n");
        self.w("    *out = v; return 1;\n");
        self.w("}\n");
        // env_set + args + tcp_connect(string).
        self.w("int64_t __maka_rt_env_set(const char* name, const char* value) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    return _putenv_s(name, value);\n");
        self.w("#else\n");
        self.w("    return setenv(name, value, 1);\n");
        self.w("#endif\n");
        self.w("}\n");
        // The runtime caches argc/argv from main; stdlib exposes args_count + arg_at.
        self.w("static int __maka_rt_argc = 0;\n");
        self.w("static char** __maka_rt_argv = NULL;\n");
        self.w("static void __maka_rt_set_args(int argc, char** argv) {\n");
        self.w("    __maka_rt_argc = argc; __maka_rt_argv = argv;\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_args_count(void) { return (int64_t)__maka_rt_argc; }\n");
        self.w("const char* __maka_rt_arg_at(int64_t i) {\n");
        // arg_at returns a BORROWED view of argv[i] (lives for the program), so the
        // out-of-range path must also return a non-owned view - a static "" - not a
        // malloc'd buffer (which would leak, since the caller never frees a view).
        self.w("    if (i < 0 || i >= (int64_t)__maka_rt_argc) { return \"\"; }\n");
        self.w("    return __maka_rt_argv[i];\n");
        self.w("}\n");
        // tcp_connect by hostname — resolve via gethostbyname then connect.
        self.w("int64_t __maka_rt_tcp_connect_host(const char* host, int64_t port) {\n");
        self.w("    int64_t addr = __maka_dns_resolve_v4(host);\n");
        self.w("    if (addr < 0) return -1;\n");
        self.w("    int a = (int)((addr >> 24) & 0xFF);\n");
        self.w("    int b = (int)((addr >> 16) & 0xFF);\n");
        self.w("    int c = (int)((addr >> 8)  & 0xFF);\n");
        self.w("    int d = (int)( addr        & 0xFF);\n");
        self.w("    return __maka_tcp_connect_v4(a, b, c, d, (int)port);\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_tcp_set_nodelay(int64_t fd, int64_t on) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    int v = (int)on;\n");
        self.w("    return setsockopt((SOCKET)fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&v, sizeof(v));\n");
        self.w("#else\n");
        self.w("    int v = (int)on;\n");
        self.w("    return setsockopt((int)fd, 6 /* IPPROTO_TCP */, 1 /* TCP_NODELAY */, &v, sizeof(v));\n");
        self.w("#endif\n");
        self.w("}\n");
        // Random bytes — fill `len` bytes of `buf` with cryptographically
        // non-secure pseudorandom data (xorshift64*).
        self.w("int64_t __maka_rt_random_bytes(maka_unit* buf, int64_t len) {\n");
        self.w("    uint8_t* p = (uint8_t*)buf;\n");
        self.w("    int64_t i = 0;\n");
        self.w("    while (i + 8 <= len) {\n");
        self.w("        uint64_t r = __maka_rt_rng_next();\n");
        self.w("        memcpy(p + i, &r, 8); i += 8;\n");
        self.w("    }\n");
        self.w("    if (i < len) {\n");
        self.w("        uint64_t r = __maka_rt_rng_next();\n");
        self.w("        memcpy(p + i, &r, (size_t)(len - i));\n");
        self.w("    }\n");
        self.w("    return len;\n");
        self.w("}\n");
        // String slicing + splitting + mtime + listdir.
        self.w("char* __maka_rt_str_substring(const char* s, int64_t start, int64_t len) {\n");
        self.w("    if (!s) { char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    size_t slen = strlen(s);\n");
        self.w("    if (start < 0) start = 0;\n");
        self.w("    if ((size_t)start > slen) start = (int64_t)slen;\n");
        self.w("    if (len < 0) len = 0;\n");
        self.w("    if (len > (int64_t)(slen - (size_t)start)) len = (int64_t)(slen - (size_t)start);\n");
        self.w("    char* o = (char*)malloc((size_t)len + 1);\n");
        self.w("    memcpy(o, s + start, (size_t)len); o[len] = 0;\n");
        self.w("    return o;\n");
        self.w("}\n");
        // file_listdir + str_split — return a fresh `const char**` array of
        // malloc'd strings, writing the count through `*out_n`.  Stdlib
        // wraps these into a Slice<string> so the user never sees the raw
        // pointer pair.  Caller owns the array + each entry.
        self.w("#ifndef _WIN32\n");
        self.w("#include <dirent.h>\n");
        self.w("#endif\n");
        self.w("const char** __maka_rt_file_listdir(const char* path, int64_t* out_n) {\n");
        self.w("    *out_n = 0;\n");
        self.w("#ifdef _WIN32\n");
        // UTF-8 path → UTF-16 + "\\*" pattern → FindFirstFileW; convert each
        // returned cFileName back to UTF-8.  Handles non-ANSI paths.
        self.w("    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);\n");
        self.w("    if (wlen <= 0) return NULL;\n");
        self.w("    WCHAR* wpattern = (WCHAR*)calloc((size_t)(wlen + 3), sizeof(WCHAR));\n");
        self.w("    if (!wpattern) return NULL;\n");
        self.w("    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpattern, wlen) <= 0) { free(wpattern); return NULL; }\n");
        // Replace trailing NUL with backslash-star-NUL.
        self.w("    wpattern[wlen - 1] = L'\\\\'; wpattern[wlen] = L'*'; wpattern[wlen + 1] = 0;\n");
        self.w("    WIN32_FIND_DATAW fd;\n");
        self.w("    HANDLE h = FindFirstFileW(wpattern, &fd);\n");
        self.w("    free(wpattern);\n");
        self.w("    if (h == INVALID_HANDLE_VALUE) return NULL;\n");
        self.w("    size_t cap = 16, n = 0;\n");
        self.w("    const char** arr = (const char**)malloc(cap * sizeof(const char*));\n");
        self.w("    do {\n");
        self.w("        if (fd.cFileName[0] == L'.' && (fd.cFileName[1] == 0 || (fd.cFileName[1] == L'.' && fd.cFileName[2] == 0))) continue;\n");
        self.w("        int u8len = WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, NULL, 0, NULL, NULL);\n");
        self.w("        if (u8len <= 0) continue;\n");
        self.w("        if (n == cap) {\n");
        self.w("            const char** narr = (const char**)realloc(arr, cap * 2 * sizeof(const char*));\n");
        // On realloc failure: free already-accumulated entries + original
        // array and bail out with 0 entries.  Without the check, the
        // following arr[n++] would segfault.
        self.w("            if (!narr) {\n");
        self.w("                for (size_t i = 0; i < n; i++) free((void*)arr[i]);\n");
        self.w("                free(arr); *out_n = 0;\n");
        self.w("#ifdef _WIN32\n");
        self.w("                FindClose(h);\n");
        self.w("#else\n");
        self.w("                closedir(d);\n");
        self.w("#endif\n");
        self.w("                return NULL;\n");
        self.w("            }\n");
        self.w("            arr = narr; cap *= 2;\n");
        self.w("        }\n");
        self.w("        char* s = (char*)malloc((size_t)u8len);\n");
        self.w("        WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, s, u8len, NULL, NULL);\n");
        self.w("        arr[n++] = s;\n");
        self.w("    } while (FindNextFileW(h, &fd));\n");
        self.w("    FindClose(h);\n");
        self.w("    *out_n = (int64_t)n; return arr;\n");
        self.w("#else\n");
        self.w("    DIR* d = opendir(path);\n");
        self.w("    if (!d) return NULL;\n");
        self.w("    size_t cap = 16, n = 0;\n");
        self.w("    const char** arr = (const char**)malloc(cap * sizeof(const char*));\n");
        self.w("    struct dirent* ent;\n");
        self.w("    while ((ent = readdir(d))) {\n");
        self.w("        if (strcmp(ent->d_name, \".\") == 0 || strcmp(ent->d_name, \"..\") == 0) continue;\n");
        self.w("        if (n == cap) {\n");
        self.w("            const char** narr = (const char**)realloc(arr, cap * 2 * sizeof(const char*));\n");
        // On realloc failure: free already-accumulated entries + original
        // array and bail out with 0 entries.  Without the check, the
        // following arr[n++] would segfault.
        self.w("            if (!narr) {\n");
        self.w("                for (size_t i = 0; i < n; i++) free((void*)arr[i]);\n");
        self.w("                free(arr); *out_n = 0;\n");
        self.w("#ifdef _WIN32\n");
        self.w("                FindClose(h);\n");
        self.w("#else\n");
        self.w("                closedir(d);\n");
        self.w("#endif\n");
        self.w("                return NULL;\n");
        self.w("            }\n");
        self.w("            arr = narr; cap *= 2;\n");
        self.w("        }\n");
        self.w("        size_t l = strlen(ent->d_name);\n");
        self.w("        char* s = (char*)malloc(l + 1);\n");
        self.w("        memcpy(s, ent->d_name, l + 1);\n");
        self.w("        arr[n++] = s;\n");
        self.w("    }\n");
        self.w("    closedir(d);\n");
        self.w("    *out_n = (int64_t)n; return arr;\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("const char** __maka_rt_str_split(const char* s, const char* sep, int64_t* out_n) {\n");
        self.w("    *out_n = 0;\n");
        self.w("    if (!s || !sep || !*sep) return NULL;\n");
        self.w("    size_t cap = 8, n = 0;\n");
        self.w("    const char** arr = (const char**)malloc(cap * sizeof(const char*));\n");
        self.w("    size_t sep_len = strlen(sep);\n");
        self.w("    const char* cur = s;\n");
        self.w("    while (1) {\n");
        self.w("        const char* p = strstr(cur, sep);\n");
        self.w("        size_t l = p ? (size_t)(p - cur) : strlen(cur);\n");
        self.w("        if (n == cap) {\n");
        self.w("            const char** narr = (const char**)realloc(arr, cap * 2 * sizeof(const char*));\n");
        self.w("            if (!narr) {\n");
        self.w("                for (size_t i = 0; i < n; i++) free((void*)arr[i]);\n");
        self.w("                free(arr); *out_n = 0;\n");
        self.w("                return NULL;\n");
        self.w("            }\n");
        self.w("            arr = narr; cap *= 2;\n");
        self.w("        }\n");
        self.w("        char* tok = (char*)malloc(l + 1);\n");
        self.w("        memcpy(tok, cur, l); tok[l] = 0;\n");
        self.w("        arr[n++] = tok;\n");
        self.w("        if (!p) break;\n");
        self.w("        cur = p + sep_len;\n");
        self.w("    }\n");
        self.w("    *out_n = (int64_t)n; return arr;\n");
        self.w("}\n");
        // chan_try_recv_bytes: peek count, recv N bytes into caller's buffer.
        // The caller decides item size based on the channel they created with.
        self.w("int64_t __maka_rt_chan_try_recv_bytes(maka_unit* p, maka_unit* out) {\n");
        self.w("    if (!p || !out) return 0;\n");
        self.w("    return maka_chan_bytes_try_recv(p, out);\n");
        self.w("}\n");
        // Process + env_remove + file_copy/chmod/realpath + udp_recv_from.
        self.w("int64_t __maka_rt_process_id(void) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    return (int64_t)GetCurrentProcessId();\n");
        self.w("#else\n");
        self.w("    return (int64_t)getpid();\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_env_remove(const char* name) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    return _putenv_s(name, \"\");\n");
        self.w("#else\n");
        self.w("    return unsetenv(name);\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_process_run(int64_t argc, const char** argv) {\n");
        self.w("    if (argc <= 0 || !argv || !argv[0]) return -1;\n");
        self.w("#ifdef _WIN32\n");
        // MSVC CommandLineToArgvW round-trip rules: each argument is wrapped
        // in double quotes; backslashes are doubled iff they precede a quote
        // (or the closing quote); embedded quotes are escaped with backslash.
        self.w("    size_t cap = 1024; char* cmd = (char*)malloc(cap); size_t len = 0;\n");
        self.w("    for (int64_t i = 0; i < argc; i++) {\n");
        self.w("        const char* a = argv[i]; size_t l = strlen(a);\n");
        // Worst case: every char needs an escape backslash + 2 surrounding quotes + space.
        self.w("        while (len + l * 2 + 4 > cap) {\n");
        self.w("            char* nc = (char*)realloc(cmd, cap * 2);\n");
        self.w("            if (!nc) { free(cmd); return -1; }\n");
        self.w("            cmd = nc; cap *= 2;\n");
        self.w("        }\n");
        self.w("        if (i) cmd[len++] = ' ';\n");
        self.w("        cmd[len++] = '\"';\n");
        self.w("        size_t j = 0;\n");
        self.w("        while (j < l) {\n");
        self.w("            size_t bs = 0;\n");
        self.w("            while (j < l && a[j] == '\\\\') { bs++; j++; }\n");
        self.w("            if (j == l) {\n");
        // Trailing backslashes: double them so they don't escape the closing quote.
        self.w("                while (bs--) { cmd[len++] = '\\\\'; cmd[len++] = '\\\\'; }\n");
        self.w("            } else if (a[j] == '\"') {\n");
        self.w("                while (bs--) { cmd[len++] = '\\\\'; cmd[len++] = '\\\\'; }\n");
        self.w("                cmd[len++] = '\\\\'; cmd[len++] = '\"'; j++;\n");
        self.w("            } else {\n");
        self.w("                while (bs--) { cmd[len++] = '\\\\'; }\n");
        self.w("                cmd[len++] = a[j++];\n");
        self.w("            }\n");
        self.w("        }\n");
        self.w("        cmd[len++] = '\"';\n");
        self.w("    }\n");
        self.w("    cmd[len] = 0;\n");
        // UTF-8 cmd → UTF-16 → CreateProcessW so non-ANSI args work.
        self.w("    int wclen = MultiByteToWideChar(CP_UTF8, 0, cmd, -1, NULL, 0);\n");
        self.w("    if (wclen <= 0) { free(cmd); return -1; }\n");
        self.w("    WCHAR* wcmd = (WCHAR*)calloc((size_t)wclen, sizeof(WCHAR));\n");
        self.w("    if (!wcmd) { free(cmd); return -1; }\n");
        self.w("    if (MultiByteToWideChar(CP_UTF8, 0, cmd, -1, wcmd, wclen) <= 0) { free(cmd); free(wcmd); return -1; }\n");
        self.w("    STARTUPINFOW si; memset(&si, 0, sizeof(si)); si.cb = sizeof(si);\n");
        self.w("    PROCESS_INFORMATION pi; memset(&pi, 0, sizeof(pi));\n");
        self.w("    BOOL ok = CreateProcessW(NULL, wcmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);\n");
        self.w("    free(cmd); free(wcmd);\n");
        self.w("    if (!ok) return -1;\n");
        self.w("    WaitForSingleObject(pi.hProcess, INFINITE);\n");
        self.w("    DWORD code = 0; GetExitCodeProcess(pi.hProcess, &code);\n");
        self.w("    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);\n");
        self.w("    return (int64_t)code;\n");
        self.w("#else\n");
        self.w("    pid_t pid = fork();\n");
        self.w("    if (pid < 0) return -1;\n");
        self.w("    if (pid == 0) {\n");
        self.w("        if (argc < 0 || argc >= (int64_t)(SIZE_MAX / sizeof(char*)) - 1) _exit(127);\n");
        self.w("        char** av = (char**)malloc((size_t)(argc + 1) * sizeof(char*));\n");
        self.w("        if (!av) _exit(127);\n");
        self.w("        for (int64_t i = 0; i < argc; i++) av[i] = (char*)argv[i];\n");
        self.w("        av[argc] = NULL;\n");
        self.w("        execvp(av[0], av);\n");
        self.w("        _exit(127);\n");
        self.w("    }\n");
        self.w("    int status; waitpid(pid, &status, 0);\n");
        self.w("    if (WIFEXITED(status)) return (int64_t)WEXITSTATUS(status);\n");
        self.w("    return -1;\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_copy(const char* src, const char* dst) {\n");
        self.w("    int sfd = (int)__maka_file_open(src, 0, 0);\n");
        self.w("    if (sfd < 0) return -1;\n");
        // Same-file guard: opening dst with O_TRUNC would empty src before we read
        // it (silent data loss).  If src and dst are the same inode, the copy is a
        // no-op success.
        self.w("    { struct stat __ss, __ds; if (fstat(sfd, &__ss) == 0 && stat(dst, &__ds) == 0\n");
        self.w("        && __ss.st_dev == __ds.st_dev && __ss.st_ino == __ds.st_ino) { close(sfd); return 0; } }\n");
        self.w("    int dfd = (int)__maka_file_open(dst, 1 | 64 | 512, 420);\n");
        self.w("    if (dfd < 0) { close(sfd); return -1; }\n");
        self.w("    char buf[8192]; int64_t total = 0; ssize_t n;\n");
        self.w("    while (1) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("        n = (ssize_t)_read(sfd, buf, sizeof(buf));\n");
        self.w("#else\n");
        self.w("        n = read(sfd, buf, sizeof(buf));\n");
        self.w("#endif\n");
        // Distinguish error from EOF: n<0 must propagate, not be silently
        // reported as a clean copy.
        self.w("        if (n < 0) { close(sfd); close(dfd); return -1; }\n");
        self.w("        if (n == 0) break;\n");
        self.w("#ifdef _WIN32\n");
        self.w("        ssize_t w = (ssize_t)_write(dfd, buf, (unsigned int)n);\n");
        self.w("#else\n");
        self.w("        ssize_t w = write(dfd, buf, (size_t)n);\n");
        self.w("#endif\n");
        self.w("        if (w != n) { close(sfd); close(dfd); return -1; }\n");
        self.w("        total += w;\n");
        self.w("    }\n");
        // Return 0 on success / -1 on error to match the documented stdlib
        // contract for file_* — the byte count is accessible via file_size.
        self.w("    close(sfd); close(dfd); return total >= 0 ? 0 : -1;\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_chmod(const char* path, int64_t mode) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    WCHAR* wp = __maka_path_to_w(path); if (!wp) return -1;\n");
        self.w("    int rc = _wchmod(wp, (int)mode); free(wp); return rc;\n");
        self.w("#else\n");
        self.w("    return chmod(path, (mode_t)mode);\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("char* __maka_rt_file_realpath(const char* path) {\n");
        self.w("#ifdef _WIN32\n");
        // UTF-8 → UTF-16, query required length first to avoid silent truncation.
        self.w("    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);\n");
        self.w("    if (wlen <= 0) { char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    WCHAR* wpath = (WCHAR*)calloc((size_t)wlen, sizeof(WCHAR));\n");
        self.w("    if (!wpath) { char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen) <= 0) { free(wpath); char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    DWORD need = GetFullPathNameW(wpath, 0, NULL, NULL);\n");
        self.w("    if (need == 0) { free(wpath); char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    WCHAR* wbuf = (WCHAR*)calloc((size_t)need, sizeof(WCHAR));\n");
        self.w("    if (!wbuf) { free(wpath); char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    GetFullPathNameW(wpath, need, wbuf, NULL);\n");
        self.w("    free(wpath);\n");
        self.w("    int u8len = WideCharToMultiByte(CP_UTF8, 0, wbuf, -1, NULL, 0, NULL, NULL);\n");
        self.w("    if (u8len <= 0) { free(wbuf); char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    char* o = (char*)malloc((size_t)u8len);\n");
        self.w("    if (!o) { free(wbuf); char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    WideCharToMultiByte(CP_UTF8, 0, wbuf, -1, o, u8len, NULL, NULL);\n");
        self.w("    free(wbuf); return o;\n");
        self.w("#else\n");
        self.w("    char buf[4096];\n");
        self.w("    if (!realpath(path, buf)) { char* e = (char*)malloc(1); e[0] = 0; return e; }\n");
        self.w("    size_t l = strlen(buf);\n");
        self.w("    char* o = (char*)malloc(l + 1);\n");
        self.w("    memcpy(o, buf, l + 1); return o;\n");
        self.w("#endif\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_udp_recv_from(int64_t fd, maka_unit* buf, int64_t cap, int64_t* peer_v4, int64_t* peer_port) {\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa)); __MAKA_SA_LEN_INIT(sa);\n");
        self.w("#ifdef _WIN32\n");
        self.w("    int sl = sizeof(sa);\n");
        self.w("#else\n");
        self.w("    __maka_socklen_t sl = sizeof(sa);\n");
        self.w("#endif\n");
        self.w("    long n = recvfrom((int)fd, (void*)buf, (unsigned long)cap, 0, (struct sockaddr*)&sa, &sl);\n");
        self.w("    if (n < 0) return -1;\n");
        self.w("    if (peer_v4)   *peer_v4   = (int64_t)ntohl(sa.sin_addr.s_addr);\n");
        self.w("    if (peer_port) *peer_port = (int64_t)ntohs(sa.sin_port);\n");
        self.w("    return (int64_t)n;\n");
        self.w("}\n");
        self.w("int64_t __maka_rt_file_mtime_ns(const char* path) {\n");
        self.w("#ifdef _WIN32\n");
        self.w("    struct __stat64 st;\n");
        self.w("    WCHAR* wp = __maka_path_to_w(path); if (!wp) return -1;\n");
        self.w("    int sr = _wstat64(wp, &st); free(wp); if (sr != 0) return -1;\n");
        self.w("    return (int64_t)st.st_mtime * 1000000000LL;\n");
        self.w("#else\n");
        self.w("    struct stat st; if (stat(path, &st) != 0) return -1;\n");
        // Use st_mtime (second precision) — portable across Linux/Darwin/BSD
        // without requiring _POSIX_C_SOURCE 200809L for st_mtim / st_mtimespec.
        self.w("    return (int64_t)st.st_mtime * 1000000000LL;\n");
        self.w("#endif\n");
        self.w("}\n");
        // DNS resolution via gethostbyname.  On macOS/BSD we include <netdb.h>
        // directly (the SDK declaration conflicts with our forward decl when
        // <sys/event.h> transitively pulls it in).  On Linux we keep the
        // forward decl to avoid the addrinfo header bloat.  Windows uses the
        // winsock2.h-supplied gethostbyname.
        self.w("#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)\n");
        self.w("#include <netdb.h>\n");
        self.w("#define __maka_hostent_t struct hostent\n");
        self.w("#elif !defined(_WIN32)\n");
        self.w("struct __maka_hostent { char* h_name; char** h_aliases; int h_addrtype; int h_length; char** h_addr_list; };\n");
        self.w("extern struct __maka_hostent* gethostbyname(const char*);\n");
        self.w("#define __maka_hostent_t struct __maka_hostent\n");
        self.w("#else\n");
        self.w("#define __maka_hostent_t struct hostent\n");
        self.w("#endif\n");
        self.w("static inline int64_t __maka_dns_resolve_v4(const char* host) {\n");
        self.w("    __maka_hostent_t* he = gethostbyname(host);\n");
        self.w("    if (!he || !he->h_addr_list || !he->h_addr_list[0]) return -1;\n");
        self.w("    unsigned char* p = (unsigned char*)he->h_addr_list[0];\n");
        self.w("    return ((int64_t)p[0] << 24) | ((int64_t)p[1] << 16) | ((int64_t)p[2] << 8) | (int64_t)p[3];\n");
        self.w("}\n");
        // UDP helpers — bind a datagram socket, send/recv from a peer.
        self.w("static inline int64_t __maka_udp_open(int64_t port) {\n");
        self.w("    int s = socket(__MAKA_AF_INET, 2 /*SOCK_DGRAM*/, 0);\n");
        self.w("    if (s < 0) return -1;\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa)); __MAKA_SA_LEN_INIT(sa);\n");
        self.w("    sa.sin_family = __MAKA_AF_INET;\n");
        self.w("    sa.sin_addr.s_addr = htonl(__MAKA_INADDR_ANY);\n");
        self.w("    sa.sin_port = htons((unsigned short)port);\n");
        self.w("    if (port > 0 && bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) { close(s); return -1; }\n");
        self.w("    int flags = fcntl(s, F_GETFL, 0);\n");
        self.w("    fcntl(s, F_SETFL, flags | O_NONBLOCK);\n");
        self.w("    return s;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_udp_send_v4(int64_t fd, int64_t a, int64_t b, int64_t c, int64_t d, int64_t port, maka_unit* buf, int64_t len) {\n");
        self.w("    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa)); __MAKA_SA_LEN_INIT(sa);\n");
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
        // Windows / non-Linux POSIX equivalents.  Each "fd" is a TCP loopback
        // socket pair (read end exposed; write end stashed in a per-fd table)
        // so that WSAPoll/poll can wait on it via the existing reactor path.
        // Counts (eventfd, timerfd) are tracked atomically alongside.
        self.w("#define __MAKA_AUX_MAX 256\n");
        // `used` flag instead of `read_fd == 0` sentinel: real fd 0 (stdin) can
        // be closed and recycled by pipe(); using it as "empty" mis-marks the
        // slot free and the next aux_alloc clobbers a live registration.
        self.w("typedef struct { int read_fd; int write_fd; _Atomic int64_t count; int used; } maka_aux_t;\n");
        self.w("static maka_aux_t __maka_aux[__MAKA_AUX_MAX];\n");
        self.w("static _Atomic int __maka_aux_n = 0;\n");
        self.w("static pthread_mutex_t __maka_aux_mu = PTHREAD_MUTEX_INITIALIZER;\n");
        self.w("static int __maka_aux_alloc(void) {\n");
        self.w("    int fds[2]; if (pipe(fds) != 0) return -1;\n");
        self.w("    /* The recv loop in __maka_eventfd_recv expects EWOULDBLOCK\n");
        self.w("       on an empty fd, so the underlying sockets must be\n");
        self.w("       non-blocking. */\n");
        self.w("#ifdef _WIN32\n");
        self.w("    { u_long nb = 1; ioctlsocket((SOCKET)fds[0], FIONBIO, &nb); ioctlsocket((SOCKET)fds[1], FIONBIO, &nb); }\n");
        self.w("#else\n");
        self.w("    { int f0 = fcntl(fds[0], F_GETFL, 0); fcntl(fds[0], F_SETFL, f0 | O_NONBLOCK);\n");
        self.w("      int f1 = fcntl(fds[1], F_GETFL, 0); fcntl(fds[1], F_SETFL, f1 | O_NONBLOCK); }\n");
        self.w("#endif\n");
        self.w("    pthread_mutex_lock(&__maka_aux_mu);\n");
        self.w("    int i;\n");
        self.w("    for (i = 0; i < __MAKA_AUX_MAX; i++) if (!__maka_aux[i].used) break;\n");
        self.w("    if (i == __MAKA_AUX_MAX) {\n");
        // Table full — close the pipe pair we just opened so we don't leak
        // two fds on every exhausted call.
        self.w("        pthread_mutex_unlock(&__maka_aux_mu);\n");
        self.w("        close(fds[0]); close(fds[1]);\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("    __maka_aux[i].read_fd  = fds[0];\n");
        self.w("    __maka_aux[i].write_fd = fds[1];\n");
        self.w("    __maka_aux[i].used     = 1;\n");
        self.w("    atomic_init(&__maka_aux[i].count, 0);\n");
        self.w("    pthread_mutex_unlock(&__maka_aux_mu);\n");
        self.w("    return fds[0];\n");
        self.w("}\n");
        self.w("static maka_aux_t* __maka_aux_find(int read_fd) {\n");
        self.w("    for (int i = 0; i < __MAKA_AUX_MAX; i++)\n");
        self.w("        if (__maka_aux[i].used && __maka_aux[i].read_fd == read_fd) return &__maka_aux[i];\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        // Release the aux slot for a read_fd that's about to be closed.
        // Without this, the slot stays used=1 and a recycled fd number gets
        // mapped to the stale (closed) write_fd → signals vanish.
        self.w("static void __maka_aux_release(int read_fd) {\n");
        self.w("    pthread_mutex_lock(&__maka_aux_mu);\n");
        self.w("    for (int i = 0; i < __MAKA_AUX_MAX; i++) {\n");
        self.w("        if (__maka_aux[i].used && __maka_aux[i].read_fd == read_fd) {\n");
        self.w("            int wfd = __maka_aux[i].write_fd;\n");
        self.w("            __maka_aux[i].used = 0;\n");
        self.w("            __maka_aux[i].read_fd = -1;\n");
        self.w("            __maka_aux[i].write_fd = -1;\n");
        self.w("            pthread_mutex_unlock(&__maka_aux_mu);\n");
        // Closing the write end here would race with any worker thread
        // about to send() — leave that to whoever owns it.  Closing once
        // the read side is gone is enough for the reactor side: WSAPoll
        // on the now-closed read socket would return POLLNVAL.
        self.w("            (void)wfd; return;\n");
        self.w("        }\n");
        self.w("    }\n");
        self.w("    pthread_mutex_unlock(&__maka_aux_mu);\n");
        self.w("}\n");
        // eventfd: kernel counter with read/write wakeups.  Emulated via a
        // socket pair + atomic counter.  signal writes a wake byte; recv
        // consumes any pending bytes and returns the snapshot+reset counter.
        self.w("static inline int64_t __maka_eventfd_create(int64_t initial) {\n");
        self.w("    int fd = __maka_aux_alloc();\n");
        self.w("    if (fd < 0) return -1;\n");
        self.w("    maka_aux_t* a = __maka_aux_find(fd);\n");
        self.w("    if (a) atomic_store(&a->count, initial);\n");
        self.w("    return fd;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_eventfd_signal(int64_t fd, int64_t n) {\n");
        self.w("    maka_aux_t* a = __maka_aux_find((int)fd); if (!a) return -1;\n");
        self.w("    atomic_fetch_add(&a->count, n);\n");
        // Send a wake byte.  EWOULDBLOCK means the receive buffer is full,
        // which means a byte is already waiting and will fire the reactor —
        // safe to ignore.  The atomic count is the authoritative state.
        self.w("    char b = 1; (void)send((int)a->write_fd, &b, 1, 0);\n");
        self.w("    return 0;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_eventfd_recv(int64_t fd) {\n");
        self.w("    maka_aux_t* a = __maka_aux_find((int)fd); if (!a) return -1;\n");
        self.w("    while (1) {\n");
        // Count is authoritative — even if a wake byte got dropped by send
        // EWOULDBLOCK, drain on every iteration and consume the count.
        self.w("        int64_t snap = atomic_load(&a->count);\n");
        self.w("        if (snap > 0) {\n");
        self.w("            char drain[64]; (void)recv((int)fd, drain, sizeof(drain), 0);\n");
        self.w("            int64_t c = atomic_exchange(&a->count, 0);\n");
        self.w("            if (c > 0) return c;\n");
        self.w("        }\n");
        self.w("        char drain[64];\n");
        self.w("        int got = recv((int)fd, drain, sizeof(drain), 0);\n");
        self.w("        if (got > 0) { int64_t c = atomic_exchange(&a->count, 0); return c; }\n");
        self.w("        if (got < 0 && errno == EWOULDBLOCK) { __maka_wait_fd(fd, MAKA_EV_READ); continue; }\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("}\n");
        // timerfd: CreateWaitableTimer / SetWaitableTimer in a background
        // thread that signals the socket pair every interval_ns nanoseconds.
        // Counter increments on each fire; recv consumes and returns count.
        self.w("typedef struct { int wfd; _Atomic int64_t* counter; int64_t initial_ns; int64_t interval_ns; } __maka_timer_arg_t;\n");
        self.w("static void* __maka_timer_thread(void* arg) {\n");
        self.w("    __maka_timer_arg_t* a = (__maka_timer_arg_t*)arg;\n");
        self.w("    struct timespec ts;\n");
        self.w("    ts.tv_sec  = a->initial_ns / 1000000000LL;\n");
        self.w("    ts.tv_nsec = a->initial_ns % 1000000000LL;\n");
        self.w("    if (a->initial_ns > 0) nanosleep(&ts, NULL);\n");
        self.w("    while (1) {\n");
        self.w("        atomic_fetch_add(a->counter, 1);\n");
        self.w("        char b = 1; send(a->wfd, &b, 1, 0);\n");
        self.w("        if (a->interval_ns == 0) break;\n");
        self.w("        ts.tv_sec  = a->interval_ns / 1000000000LL;\n");
        self.w("        ts.tv_nsec = a->interval_ns % 1000000000LL;\n");
        self.w("        nanosleep(&ts, NULL);\n");
        self.w("    }\n");
        self.w("    free(a);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_timerfd_create(int64_t initial_ns, int64_t interval_ns) {\n");
        self.w("    int fd = __maka_aux_alloc();\n");
        self.w("    if (fd < 0) return -1;\n");
        self.w("    maka_aux_t* a = __maka_aux_find(fd); if (!a) return -1;\n");
        self.w("    __maka_timer_arg_t* arg = (__maka_timer_arg_t*)calloc(1, sizeof(__maka_timer_arg_t));\n");
        self.w("    arg->wfd = a->write_fd; arg->counter = &a->count;\n");
        self.w("    arg->initial_ns = initial_ns; arg->interval_ns = interval_ns;\n");
        self.w("    pthread_t t;\n");
        self.w("    if (pthread_create(&t, NULL, __maka_timer_thread, arg) != 0) {\n");
        // EAGAIN: free the arg + release the aux slot (so subsequent timerfds
        // don't exhaust the table).  Caller sees -1 and can recover.
        self.w("        free(arg);\n");
        self.w("        __maka_aux_release(fd);\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("    pthread_detach(t);\n");
        self.w("    return fd;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_timerfd_recv(int64_t fd) {\n");
        self.w("    return __maka_eventfd_recv(fd);    /* same shape: drain + return counter */\n");
        self.w("}\n");
        // signalfd: stub for now (no SIGUSR1 on Windows; tests using
        // SIGINT/SIGTERM could be supported via signal() + writes to the
        // pair, but that's a deeper hook).
        self.w("static inline int64_t __maka_signalfd_open(int64_t signum) { (void)signum; return -1; }\n");
        self.w("static inline int64_t __maka_signalfd_recv(int64_t fd) { (void)fd; return -1; }\n");
        // inotify: Windows uses ReadDirectoryChangesW; macOS/BSD use kqueue
        // EVFILT_VNODE on the file itself.  Both push a wake-byte through the
        // aux socket pair so the reactor picks them up like a real inotify fd.
        self.w("typedef struct { int wfd; _Atomic int64_t* counter; char* path; int mask; } __maka_inotify_arg_t;\n");
        self.w("#ifdef _WIN32\n");
        self.w("static void* __maka_inotify_thread(void* arg) {\n");
        self.w("    __maka_inotify_arg_t* a = (__maka_inotify_arg_t*)arg;\n");
        // Convert UTF-8 path to UTF-16 so non-ANSI paths work.
        self.w("    int wlen = MultiByteToWideChar(CP_UTF8, 0, a->path, -1, NULL, 0);\n");
        self.w("    if (wlen <= 0) { free(a->path); free(a); return NULL; }\n");
        self.w("    WCHAR* wpath = (WCHAR*)calloc((size_t)wlen + 1, sizeof(WCHAR));\n");
        self.w("    if (!wpath) { free(a->path); free(a); return NULL; }\n");
        self.w("    if (MultiByteToWideChar(CP_UTF8, 0, a->path, -1, wpath, wlen) <= 0) { free(wpath); free(a->path); free(a); return NULL; }\n");
        self.w("    HANDLE h = CreateFileW(wpath, FILE_LIST_DIRECTORY,\n");
        self.w("        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,\n");
        self.w("        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);\n");
        self.w("    free(wpath);\n");
        self.w("    if (h == INVALID_HANDLE_VALUE) { free(a->path); free(a); return NULL; }\n");
        self.w("    char buf[4096]; DWORD br = 0;\n");
        self.w("    while (1) {\n");
        self.w("        if (!ReadDirectoryChangesW(h, buf, sizeof(buf), TRUE,\n");
        self.w("                FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_LAST_WRITE |\n");
        self.w("                FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_CREATION,\n");
        self.w("                &br, NULL, NULL)) break;\n");
        self.w("        atomic_fetch_add(a->counter, 1);\n");
        self.w("        char m = 1; send(a->wfd, &m, 1, 0);\n");
        self.w("    }\n");
        self.w("    CloseHandle(h); free(a->path); free(a);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("#else\n");
        // macOS/BSD: kqueue EVFILT_VNODE on the file.  Watches the file (not
        // its parent dir) which is the common case; if the user passed a
        // directory path it gets watched too.
        self.w("#include <fcntl.h>\n");
        self.w("static void* __maka_inotify_thread(void* arg) {\n");
        self.w("    __maka_inotify_arg_t* a = (__maka_inotify_arg_t*)arg;\n");
        self.w("    int wfd = open(a->path, O_RDONLY);\n");
        self.w("    if (wfd < 0) { free(a->path); free(a); return NULL; }\n");
        self.w("#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)\n");
        self.w("    int kq = kqueue();\n");
        self.w("    struct kevent kev;\n");
        self.w("    EV_SET(&kev, wfd, EVFILT_VNODE, EV_ADD | EV_CLEAR,\n");
        self.w("           NOTE_WRITE | NOTE_DELETE | NOTE_EXTEND | NOTE_ATTRIB | NOTE_LINK | NOTE_RENAME,\n");
        self.w("           0, NULL);\n");
        self.w("    kevent(kq, &kev, 1, NULL, 0, NULL);\n");
        self.w("    while (1) {\n");
        self.w("        struct kevent out;\n");
        self.w("        int n = kevent(kq, NULL, 0, &out, 1, NULL);\n");
        self.w("        if (n <= 0) break;\n");
        self.w("        atomic_fetch_add(a->counter, 1);\n");
        self.w("        char m = 1; (void)send(a->wfd, &m, 1, 0);\n");
        self.w("    }\n");
        self.w("    close(kq);\n");
        self.w("#endif\n");
        self.w("    close(wfd); free(a->path); free(a);\n");
        self.w("    return NULL;\n");
        self.w("}\n");
        self.w("#endif\n");
        self.w("static __thread int __maka_last_inotify_fd = -1;\n");
        self.w("static inline int64_t __maka_inotify_open(void) {\n");
        self.w("    int fd = __maka_aux_alloc();\n");
        self.w("    __maka_last_inotify_fd = fd;\n");
        self.w("    return fd;\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_inotify_add(int64_t fd, const char* path, int64_t mask) {\n");
        self.w("    maka_aux_t* a = __maka_aux_find((int)fd); if (!a) return -1;\n");
        self.w("    /* When the user watches a file, watch its parent directory and\n");
        self.w("       deliver any event in that directory (inotify_recv returns wd 1). */\n");
        self.w("    char* parent = (char*)malloc(strlen(path) + 1);\n");
        self.w("    strcpy(parent, path);\n");
        self.w("#ifdef _WIN32\n");
        self.w("    char* slash = strrchr(parent, '/'); if (!slash) slash = strrchr(parent, '\\\\');\n");
        self.w("    if (slash) *slash = 0; else { strcpy(parent, \".\"); }\n");
        self.w("#endif\n");
        self.w("    __maka_inotify_arg_t* arg = (__maka_inotify_arg_t*)calloc(1, sizeof(__maka_inotify_arg_t));\n");
        self.w("    arg->wfd = a->write_fd; arg->counter = &a->count; arg->path = parent; arg->mask = (int)mask;\n");
        self.w("    pthread_t t;\n");
        self.w("    if (pthread_create(&t, NULL, __maka_inotify_thread, arg) != 0) {\n");
        self.w("        free(arg->path); free(arg);\n");
        self.w("        return -1;\n");
        self.w("    }\n");
        self.w("    pthread_detach(t);\n");
        self.w("    return 1;     /* watch descriptor */\n");
        self.w("}\n");
        self.w("static inline int64_t __maka_inotify_recv(int64_t fd) {\n");
        self.w("    return __maka_eventfd_recv(fd) >= 0 ? 1 : -1;   /* return wd=1 on event */\n");
        self.w("}\n");
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

    /// The C name of a local in the function being emitted, from the per-function
    /// table built by `build_local_names`.  There is deliberately NO fallback: a
    /// missing entry means a local was named before its function's table was
    /// installed (a new emission path that forgot to set `cur_local_names`), which
    /// would silently emit a wrong or mangled name.  Fail loudly instead.
    fn local_name(&self, id: LocalId, name: &str) -> String {
        match self.cur_local_names.get(&id.0) {
            Some(n) => n.clone(),
            None => panic!(
                "codegen bug: local `{}` (id {}) has no name-table entry; a code \
                 path emitted a local without first building `cur_local_names` for \
                 its function (see build_local_names / emit_func / func_signature)",
                name, id.0
            ),
        }
    }

    /// Assign each of `f`'s locals a C name: its readable `c_ident(name)` when
    /// unambiguous, else the unique `name_<id>` form.  Reserves function, struct,
    /// enum, and global names (a local must never shadow a callable or a type in
    /// C) and the runtime's `maka`/`__` prefixes, and guarantees the resulting
    /// set is collision-free by construction.
    fn build_local_names(&self, f: &HFunc) -> std::collections::HashMap<u32, String> {
        let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
        for x in &self.sym.funcs {
            used.insert(c_ident(&x.name));
        }
        for x in &self.sym.sigs {
            used.insert(c_ident(&x.name));
        }
        for x in &self.sym.structs {
            used.insert(c_ident(&x.name));
        }
        for x in &self.sym.enums {
            used.insert(c_ident(&x.name));
        }
        for x in &self.sym.globals {
            used.insert(c_ident(&x.name));
        }
        let mut map = std::collections::HashMap::new();
        for (idx, li) in f.locals.iter().enumerate() {
            let id = idx as u32;
            let base = c_ident(&li.name);
            // Runtime C symbols all start with `maka` or `__`; keep the unique
            // form for a user local that would otherwise look like one.
            let reserved = base.starts_with("maka") || base.starts_with("__");
            let chosen = if !reserved && used.insert(base.clone()) {
                base
            } else {
                // The always-unique `name_<id>` form, extended if it somehow
                // collides (e.g. a user local literally named `foo_3`).
                let mut cand = local_name_mangled(LocalId(id), &li.name);
                while !used.insert(cand.clone()) {
                    cand.push('_');
                }
                cand
            };
            map.insert(id, chosen);
        }
        map
    }

    fn emit_func(&mut self, f: &HFunc) {
        // inline functions are spliced at each call site; no standalone C function emitted.
        if self.sym.func_sig(f.id).is_inline { return; }
        self.cur_local_names = self.build_local_names(f);
        self.cur_c_func_ret = self.sym.func_sig(f.id).ret.clone();
        // The `.maka` file backing this function's module, for `#line` stamping.
        self.cur_debug_file = self
            .debug_files
            .get(&self.sym.func_sig(f.id).module_path)
            .cloned();
        // Abandoned *Thread handles: declared but never referenced again (not
        // joined/detached/stored/returned).  Each leaks its runtime Thread
        // struct, so it is auto-detached at its declaring block's exit.
        self.cur_abandoned_handles = {
            let mut read = std::collections::HashSet::new();
            collect_local_reads_block(&f.body, &mut read);
            let mut lets = Vec::new();
            collect_let_locals(&f.body, &mut lets);
            let mut abandoned = std::collections::HashSet::new();
            for lid in lets {
                if read.contains(&lid.0) { continue; }
                let is_thread = matches!(&f.locals[lid.0 as usize].ty, HType::Ptr { inner, .. }
                    if matches!(inner.as_ref(), HType::Struct(sid) if self.sym.struct_info(*sid).name == "Thread"));
                if is_thread { abandoned.insert(lid.0); }
            }
            abandoned
        };
        let sig = self.func_signature(f);
        self.wl(&format!("{} {{", sig));
        self.open();
        // By-value array params: copy the incoming decayed pointer (`<name>__pv`,
        // see func_signature) into a real local array, so the callee works on an
        // INDEPENDENT copy (value semantics - a `[N]T` param is a copy, like any
        // array; mutations never reach the caller).  `&[N]T` borrows are untouched.
        for &pid in &f.params {
            let li = &f.locals[pid.0 as usize];
            if matches!(&li.ty, HType::Array { .. }) {
                let n = self.local_name(pid, &li.name);
                self.wl(&format!("{};", self.c_decl(&li.ty, &n)));
                self.wl(&format!("memcpy({0}, {0}__pv, sizeof({0}));", n));
            }
        }
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
                        let lname = self.local_name(LocalId(cap_id), &li.name);
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
            // Stamp each real block statement with its Maka source line so DWARF
            // maps back to `.maka`.  Only here (a genuine statement context) - NOT
            // in `emit_stmt` itself, which the match-expression / inline paths also
            // call while building single-line C expressions where a `#line`
            // preprocessor directive would be illegal.
            self.emit_line_directive(s);
            self.emit_stmt(f, s);
        }
        // Null-collapse pointers whose deps just died.
        for id in &b.ptr_nulls {
            let li = &f.locals[id.0 as usize];
            self.wl(&format!("{} = NULL; /* collapse {} */", self.local_name(*id, &li.name), li.name));
        }
        // Free heap locals declared in this block (in reverse decl order).
        for id in &b.heap_to_free {
            let (n, ty, nm) = {
                let li = &f.locals[id.0 as usize];
                (self.local_name(*id, &li.name), li.ty.clone(), li.name.clone())
            };
            self.wl(&format!("/* drop heap {} */", nm));
            self.emit_field_drop(&n, &ty, 0);
        }
        // Auto-detach *Thread handles declared in this block that are never read
        // again (abandoned).  Detach drops the spawner ref and lets the runner
        // self-reap, so the Thread struct is freed instead of leaked.  The guard
        // skips a null (failed spawn); read handles are excluded from the set, so
        // joined/detached/stored/returned handles are never double-reaped.
        if !self.cur_abandoned_handles.is_empty() {
            for s in &b.stmts {
                if let HStmt::Let { local, .. } = s {
                    if self.cur_abandoned_handles.contains(&local.0) {
                        let li = &f.locals[local.0 as usize];
                        let n = self.local_name(*local, &li.name);
                        self.wl(&format!("if ({0}) {{ __maka_detach((maka_unit*)({0})); }} /* auto-detach abandoned handle {1} */", n, li.name));
                    }
                }
            }
        }
        if !is_top { self.close(); self.wl("}"); }
    }

    /// Emit `#line N "file.maka"` for a statement (file + line, so there's no
    /// cross-function bleed).  A no-op when debug mapping is off or the statement
    /// has no source span.
    fn emit_line_directive(&mut self, s: &HStmt) {
        if let Some(file) = self.cur_debug_file.clone() {
            if let Some(line) = hstmt_line(s) {
                self.wl(&format!(
                    "#line {} \"{}\"",
                    line,
                    file.replace('\\', "\\\\").replace('"', "\\\"")
                ));
            }
        }
    }

    fn emit_stmt(&mut self, f: &HFunc, s: &HStmt) {
        match s {
            HStmt::Let { local, init, .. } => self.emit_let(f, *local, init),
            HStmt::Assign { op, place, value, drop_old, .. } => {
                self.emit_assign(f, *op, place, value, *drop_old);
                // A `yield` lowers to an assignment of an arm's synthetic `__yield`
                // local; terminate the arm so a yield nested in an `if`/`while`
                // wins instead of falling through to a later `yield`.
                if let HExprKind::Local(id) = place.kind {
                    let lbl = self.yield_exit.iter().rev()
                        .find(|(yv, _)| *yv == id).map(|(_, l)| l.clone());
                    if let Some(lbl) = lbl { self.wl(&format!("goto {};", lbl)); }
                }
            }
            HStmt::ExprStmt(e) => {
                let s = self.emit_expr(f, e);
                // Discarding an owning-VALUE temporary (a String / struct with heap-
                // owning fields produced by a fn / operator / match yield) must drop
                // it, else it leaks (SPEC 2.3).  Owning POINTERS are already hoisted
                // and freed by emit_expr.  Restrict to value-PRODUCING expressions
                // (not a place like a bare Local/Field, whose discard is a move the
                // lifetime pass already accounts for) so we never double-free.
                let is_ptr = matches!(&e.ty,
                    HType::OwnPtr { .. } | HType::Heap { .. } | HType::Ptr { .. }
                    | HType::RawPtr { .. } | HType::Ref { .. });
                let is_place = matches!(&e.kind,
                    HExprKind::Local(_) | HExprKind::Field { .. } | HExprKind::Index { .. }
                    | HExprKind::Unwrap { .. } | HExprKind::AddrOfRef { .. } | HExprKind::DerefRef(_));
                if !is_ptr && !is_place && self.drop_ty_owns(&e.ty) {
                    let ct = self.c_type(&e.ty);
                    let dc = self.capture_drops(&[("__maka_discard".to_string(), e.ty.clone())]);
                    self.wl(&format!("{{ {} __maka_discard = ({}); {} }}", ct, s, dc));
                } else {
                    self.wl(&format!("(void)({});", s));
                }
            }
            HStmt::Return { value, heap_drops, .. } => {
                // Evaluate the return value BEFORE dropping heap locals: the
                // expression may read a local that is about to be freed (e.g.
                // `return split_lines(text)`).  Stash it in a temp, then drop,
                // then return.  Locals moved out via the return value are not in
                // `heap_drops`, so their ownership transfers cleanly.
                let drops: Vec<(String, HType)> = heap_drops.iter().map(|id| {
                    let li = &f.locals[id.0 as usize];
                    (self.local_name(*id, &li.name), li.ty.clone())
                }).collect();
                match value {
                    Some(e) if !matches!(e.ty, HType::Unit) => {
                        self.wl("{");
                        self.open();
                        // Consuming position: null an owning field/element moved out
                        // via the return value before dropping the container locals.
                        let s = self.emit_move_consuming(f, e);
                        self.wl(&format!("{} = {};", self.c_decl(&e.ty, "__ret"), s));
                        for (n, ty) in &drops { self.wl("/* drop on return */"); self.emit_field_drop(n, ty, 0); }
                        self.wl("return __ret;");
                        self.close();
                        self.wl("}");
                    }
                    Some(e) => {
                        let s = self.emit_expr(f, e);
                        self.wl(&format!("(void)({});", s));
                        for (n, ty) in &drops { self.wl("/* drop on return */"); self.emit_field_drop(n, ty, 0); }
                        self.wl("return;");
                    }
                    None => {
                        for (n, ty) in &drops { self.wl("/* drop on return */"); self.emit_field_drop(n, ty, 0); }
                        self.wl("return;");
                    }
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
            HStmt::Break { heap_drops, .. } => {
                let drops: Vec<(String, HType)> = heap_drops.iter()
                    .map(|id| { let li = &f.locals[id.0 as usize]; (self.local_name(*id, &li.name), li.ty.clone()) }).collect();
                for (n, ty) in &drops { self.wl("/* drop on break */"); self.emit_field_drop(n, ty, 0); }
                self.wl("break;");
            }
            HStmt::Continue { heap_drops, .. } => {
                let drops: Vec<(String, HType)> = heap_drops.iter()
                    .map(|id| { let li = &f.locals[id.0 as usize]; (self.local_name(*id, &li.name), li.ty.clone()) }).collect();
                for (n, ty) in &drops { self.wl("/* drop on continue */"); self.emit_field_drop(n, ty, 0); }
                self.wl("continue;");
            }
            HStmt::Propagate { value, heap_drops, .. } => {
                // `propagate X;` ⇒ `return X;` from the enclosing C function, first
                // freeing THIS frame's live owning locals (like a `return`) - else
                // they leak on the propagate path.  Evaluate X into a temp BEFORE the
                // drops in case it reads one of them.
                let drops: Vec<(String, HType)> = heap_drops.iter().map(|id| {
                    let li = &f.locals[id.0 as usize];
                    (self.local_name(*id, &li.name), li.ty.clone())
                }).collect();
                match value {
                    Some(v) => {
                        let rt = self.c_type(&self.cur_c_func_ret.clone());
                        self.wl("{");
                        self.open();
                        let s = self.emit_move_consuming(f, v);
                        self.wl(&format!("{} __pv = {};", rt, s));
                        for (n, ty) in &drops { self.emit_field_drop(n, ty, 0); }
                        self.wl("return __pv;");
                        self.close();
                        self.wl("}");
                    }
                    None => {
                        for (n, ty) in &drops { self.emit_field_drop(n, ty, 0); }
                        self.wl("return;");
                    }
                }
            }
            HStmt::ForC { init, cond, step, body, .. } => {
                self.emit_stmt(f, init);
                let cs = self.emit_expr(f, cond);
                let step_text = self.emit_step_expr(f, step);
                self.wl(&format!("for (; {}; {}) {{", cs, step_text));
                self.open();
                // If this is a for-range counter provably within `[0, bound)`,
                // record it so array indexing in the body skips the bounds check.
                let bounded = forrange_bound(init, cond, body);
                if let Some(b) = bounded { self.bounded_vars.push(b); }
                self.emit_block(f, body, true);
                if bounded.is_some() { self.bounded_vars.pop(); }
                self.close();
                self.wl("}");
            }
            HStmt::ForEach { var, src, body, .. } => {
                let li = &f.locals[var.0 as usize];
                let var_name = self.local_name(*var, &li.name);
                let var_ty = self.c_type(&li.ty);
                let src_c = self.emit_expr(f, src);
                // Iterating a dense existential column `Vec<some X>`: the buffer is
                // raw `T` strided by `__esz`, and the binding is a `some X` (= a
                // `Dyn_X` fat pointer) built per element from the column's hoisted
                // witness - so trait-method dispatch on it reuses the ordinary dyn
                // path.  (Distinct from a normal Vec, whose elements are read directly.)
                if let HType::Vec { elem } = &src.ty {
                    if let HType::Dyn { traits, locked: true } = elem.as_ref() {
                        let dyn_c = format!("Dyn_{}", traits.join("_"));
                        self.wl("{");
                        self.open();
                        self.wl(&format!("{} __sc = {};", self.c_type(&src.ty), src_c));
                        self.wl("for (maka_int __i = 0; __i < __sc.len; __i += 1) {");
                        self.open();
                        self.wl(&format!(
                            "{dc} {vn} = {{ .data = (void*)((char*)__sc.data + __i * __sc.__esz), .vtbl = __sc.__vtbl }};",
                            dc = dyn_c, vn = var_name));
                        self.emit_block(f, body, true);
                        self.close();
                        self.wl("}");
                        self.close();
                        self.wl("}");
                        return;
                    }
                }
                self.wl("{");
                self.open();
                // Layout based on src type:
                let (len_str, elem_access) = match &src.ty {
                    HType::Array { len, .. } => {
                        // No copy — directly index the source array.
                        (format!("(maka_int){}", len), src_c.clone())
                    }
                    HType::Slice { .. } | HType::Vec { .. } => {
                        // If the body mutates a Vec Local source (e.g. push), its
                        // backing buffer may be realloc'd or the binding reassigned
                        // mid-iteration; a cached snapshot of `.data` would then
                        // dangle (use-after-free).  Read `.data` live each access and
                        // clamp the bound (see below) so iteration stays both
                        // memory-safe and terminating.  Non-mutating loops keep the
                        // cached snapshot (cheaper, and identical for them).
                        // Reread when the body may mutate the source container
                        // (push/realloc or a reassignment).  Root the source PLACE to
                        // its base local so a FIELD- or index-reached Vec
                        // (`for (x in b.xs) { b.xs.push(..) }`) is covered too, not
                        // just a bare Vec local - the field case snapshotted `.data`
                        // and dangled on the mid-loop realloc.
                        let reread = matches!(&src.ty, HType::Vec { .. })
                            && place_root_local(src).map_or(false, |r| block_mutates_local(body, r));
                        if reread {
                            // Bind a pointer to the container ONCE (so the source path
                            // is evaluated a single time) and read `.data`/`.len`
                            // through it each access, so a realloc is always seen.
                            let src_c_ty = self.c_type(&src.ty);
                            self.wl(&format!("{}* __vp = &({});", src_c_ty, src_c));
                            self.wl("maka_int __flen = (maka_int)__vp->len;");
                            // Bound = min(current len, snapshot len).  The snapshot
                            // cap gives snapshot semantics so a push cannot grow the
                            // iteration unboundedly (no infinite loop); the live cap
                            // keeps every index within the current buffer so a
                            // shrink/reassign cannot read out of bounds.  Data is
                            // read live each access.
                            ("((maka_int)__vp->len < __flen ? (maka_int)__vp->len : __flen)".to_string(),
                             "__vp->data".to_string())
                        } else {
                            let src_c_ty = self.c_type(&src.ty);
                            self.wl(&format!("{} __src = {};", src_c_ty, src_c));
                            if matches!(&src.ty, HType::Slice { .. }) {
                                ("__src.len".to_string(), "__src.ptr".to_string())
                            } else {
                                ("__src.len".to_string(), "__src.data".to_string())
                            }
                        }
                    }
                    HType::Heap { inner } if matches!(inner.as_ref(), HType::Vec { .. }) => {
                        let src_c_ty = self.c_type(&src.ty);
                        self.wl(&format!("{} __src = {};", src_c_ty, src_c));
                        ("__src.len".to_string(), "__src.data".to_string())
                    }
                    // Iterating a reference/pointer to a container (`for (T x in &vec)`):
                    // deref once to the container, then index its backing buffer.
                    // Without this the loop falls through to the scalar path, which
                    // emits a length of 0 and mis-indexes the pointer itself.
                    HType::Ref { inner, .. } | HType::Ptr { inner, .. }
                    | HType::OwnPtr { inner, .. } | HType::RawPtr { inner, .. }
                        if matches!(inner.as_ref(), HType::Vec { .. } | HType::Slice { .. }) =>
                    {
                        let inner_c_ty = self.c_type(inner);
                        self.wl(&format!("{} __src = *({});", inner_c_ty, src_c));
                        if matches!(inner.as_ref(), HType::Slice { .. }) {
                            ("__src.len".to_string(), "__src.ptr".to_string())
                        } else {
                            ("__src.len".to_string(), "__src.data".to_string())
                        }
                    }
                    HType::Ref { inner, .. } | HType::Ptr { inner, .. }
                    | HType::OwnPtr { inner, .. } | HType::RawPtr { inner, .. }
                        if matches!(inner.as_ref(), HType::Array { .. }) =>
                    {
                        // A reference to a fixed array emits the bare array name,
                        // which decays to a pointer; index it directly.
                        if let HType::Array { len, .. } = inner.as_ref() {
                            (format!("(maka_int){}", len), src_c.clone())
                        } else { unreachable!() }
                    }
                    _ => {
                        ("0".to_string(), src_c.clone())
                    }
                };
                // Alias the element (`T* x = &elem[i]`) instead of copying it
                // (`T x = elem[i]`) when it's a pure value-struct the body only
                // reads - avoids a per-iteration struct copy.  Safe only if: the
                // element owns no heap (so a move can't double-free), the loop var
                // isn't reassigned/`&mut`-borrowed, and the source isn't mutated
                // (which could realloc the backing buffer mid-iteration).
                // The element may be aliased (not copied) only if the source
                // container is not mutated mid-loop - rooted to the base local so a
                // field/index-reached source is guarded too (else a realloc would
                // dangle the alias).
                let src_local_ok = place_root_local(src).map_or(true, |r| !block_mutates_local(body, r));
                // An explicit reference binding (`for (&mut T x in &mut vec)`):
                // the loop var is a reference TO the element, not the element by
                // value, so bind it to the element's address.  Distinguished from
                // a by-value pointer element (`for (*Thread t in ts)`, where the
                // element type IS the pointer) by comparing the binding's pointee
                // to the source's element type.
                let li_inner: Option<&HType> = match &li.ty {
                    HType::Ref { inner, .. } | HType::Ptr { inner, .. }
                    | HType::OwnPtr { inner, .. } | HType::RawPtr { inner, .. } => Some(inner),
                    _ => None,
                };
                let elem_ty = foreach_elem_ty(&src.ty);
                let bind_by_ref = match (li_inner, elem_ty) {
                    // Bind by reference only when the loop var is genuinely one
                    // indirection deeper than the element (its pointee matches the
                    // element AND the var's own type is not already the element's).
                    // A `&dyn T` element is itself a fat pointer whose C type equals
                    // `dyn T`'s, so without the second check the var would be bound
                    // to the element's address (`Dyn_T x = &elem`) - invalid C.
                    (Some(inner), Some(et)) =>
                        self.c_type(inner) == self.c_type(et)
                        && self.c_type(&li.ty) != self.c_type(et),
                    _ => false,
                };
                let can_alias = !bind_by_ref
                    && matches!(&li.ty, HType::Struct(_))
                    && !self.drop_ty_owns(&li.ty)
                    && !block_mutates_local(body, var.0)
                    && src_local_ok;
                if bind_by_ref {
                    self.wl(&format!("for (maka_int __i = 0; __i < {}; __i += 1) {{", len_str));
                    self.open();
                    self.wl(&format!("{} {} = &({}[__i]);", var_ty, var_name, elem_access));
                    self.emit_block(f, body, true);
                    self.close();
                    self.wl("}");
                } else if matches!(&li.ty, HType::Array { .. }) {
                    // An array-typed element (`for ([N]T row in &grid)`) must be
                    // declared with its dimensions after the name and copied with
                    // memcpy: C forbids both `T[N] name` and assigning to an array
                    // with `=`.
                    self.wl(&format!("for (maka_int __i = 0; __i < {}; __i += 1) {{", len_str));
                    self.open();
                    self.wl(&format!("{};", self.c_decl(&li.ty, &var_name)));
                    self.wl(&format!("memcpy({}, {}[__i], sizeof({}));", var_name, elem_access, var_name));
                    self.emit_block(f, body, true);
                    self.close();
                    self.wl("}");
                } else if can_alias {
                    self.aliased_locals.insert(var.0);
                    self.wl(&format!("for (maka_int __i = 0; __i < {}; __i += 1) {{", len_str));
                    self.open();
                    self.wl(&format!("{}* {} = &({}[__i]);", var_ty, var_name, elem_access));
                    self.emit_block(f, body, true);
                    self.close();
                    self.wl("}");
                    self.aliased_locals.remove(&var.0);
                } else {
                    self.wl(&format!("{} {} = {{0}};", var_ty, var_name));
                    self.wl(&format!("for (maka_int __i = 0; __i < {}; __i += 1) {{", len_str));
                    self.open();
                    self.wl(&format!("{} = {}[__i];", var_name, elem_access));
                    self.emit_block(f, body, true);
                    self.close();
                    self.wl("}");
                }
                // A temporary owning container produced by a call/match (e.g.
                // `for (x in make_vec())`) has no other owner, so free it
                // (elements + buffer) after the loop.  A named place / borrow is
                // freed by its owner, so __src (a shallow copy) is left alone.
                if matches!(&src.kind, HExprKind::Call { .. } | HExprKind::Match { .. })
                    && matches!(&src.ty, HType::Vec { .. } | HType::Heap { .. })
                    && self.drop_ty_owns(&src.ty)
                {
                    self.emit_field_drop("__src", &src.ty, 0);
                }
                self.close();
                self.wl("}");
            }
        }
    }

    fn emit_step_expr(&mut self, f: &HFunc, step: &HStmt) -> String {
        if let HStmt::Assign { op, place, value, .. } = step {
            let p = self.emit_place(f, place);
            let v = self.emit_expr(f, value);
            return format!("{} = {}", p, self.compound_rhs(&p, *op, &v, &place.ty));
        }
        "(void)0".into()
    }

    /// Expand an `inline` call as a GCC statement-expression. The body's `return X` becomes
    /// `__result = X; break;` (exiting a do-while), and `propagate X` becomes a real C `return X;`
    /// from the *outer* function (the C function that contains this expansion).
    /// The C predicate for an inline match arm, over the tagged scrutinee temp
    /// `__s_<mt>` (referred to as `sref`).  Mirrors `match_predicate` but for the
    /// inline naming.  Handles enum variants (tag + optional payload lit-checks),
    /// `else`, `null`, and literal arms.
    fn inline_arm_pred(&self, a: &HMatchArm, sref: &str) -> String {
        match &a.kind {
            HArmKind::Else => "1".to_string(),
            HArmKind::Null => format!("({} == NULL)", sref),
            HArmKind::Lit(e) => format!("({} == {})", sref, self.literal_str(e)),
            HArmKind::Variant { variant, lit_checks, enum_id, .. } => {
                let info = self.sym.enum_info(*enum_id);
                let v = &info.variants[*variant];
                let tag_expr = if info.is_simple() { sref.to_string() } else { format!("{}.tag", sref) };
                let mut parts = vec![format!("({} == {})", tag_expr, v.tag)];
                for (i, c) in lit_checks.iter().enumerate() {
                    if let Some(ce) = c {
                        parts.push(format!("({}.payload.{}.{} == {})", sref, c_ident(&v.name), c_ident(&v.fields[i].name), self.literal_str(ce)));
                    }
                }
                format!("({})", parts.join(" && "))
            }
        }
    }

    /// Bind an inline match arm's variant payload fields and scrut binding from the
    /// tagged scrutinee temp `__s_<mt>` (`s` is the emitted scrutinee place, nulled
    /// where a binding is moved out so the scrutinee's drop does not double-free).
    /// Shared by the switch and if-chain forms.
    fn emit_inline_arm_bindings(&mut self, inline_f: &HFunc, a: &HMatchArm, scrutinee: &HExpr, s: &str, mt: &str, tag: &str) -> String {
        let mut out = String::new();
        if let HArmKind::Variant { enum_id, variant, bindings, .. } = &a.kind {
            let vinfo = self.sym.enum_info(*enum_id).variants[*variant].clone();
            let scrut_owns = matches!(&scrutinee.kind, HExprKind::Local(_)) && self.drop_ty_owns(&scrutinee.ty);
            for (i, b) in bindings.iter().enumerate() {
                if let Some(local) = b {
                    let lty = inline_f.locals[local.0 as usize].ty.clone();
                    let nm = inline_local_name(inline_f, *local, tag);
                    out.push_str(&format!("{} {} = __s_{}.payload.{}.{}; ",
                        self.c_type(&lty), nm, mt, c_ident(&vinfo.name), c_ident(&vinfo.fields[i].name)));
                    if scrut_owns && self.drop_ty_owns(&lty) && self.arm_moves_binding(a, *local) {
                        out.push_str(&format!("{}.payload.{}.{} = ({}){{0}}; ",
                            s, c_ident(&vinfo.name), c_ident(&vinfo.fields[i].name), self.c_type(&lty)));
                    }
                }
            }
        }
        if let Some(local) = a.scrut_binding {
            let li = &inline_f.locals[local.0 as usize];
            out.push_str(&format!("{} {} = __s_{}; ", self.c_type(&li.ty), inline_local_name(inline_f, local, tag), mt));
        }
        out
    }

    /// Emit an inline match arm's body, value assignment, and scope-exit drops
    /// (shared by the switch and if-chain forms).  `mt`/`fn_ret` thread the result
    /// temp and the enclosing expansion's return type for `return` inside the arm.
    fn emit_inline_arm_body(&mut self, inline_f: &HFunc, a: &HMatchArm, mt: &str, tag: &str, needs_value: bool, fn_ret: &HType) -> String {
        let mut out = String::new();
        let yexit_lbl = self.yield_exit_begin(inline_f, a);
        for st in &a.body.stmts { out.push_str(&self.emit_inline_stmt(inline_f, st, tag, fn_ret)); }
        self.yield_exit_end(yexit_lbl, &mut out);
        if let Some(v) = &a.value {
            let vs = self.emit_inline_expr(inline_f, v, tag);
            if needs_value { out.push_str(&format!("__rm_{} = ({}); ", mt, vs)); }
            else { out.push_str(&format!("(void)({}); ", vs)); }
        }
        out.push_str(&self.arm_body_cleanup(inline_f, &a.body, Some(tag)));
        out
    }

    /// Emit a `match` that appears inside an inline body, renaming locals with the
    /// expansion `tag` and routing arm bodies through `emit_inline_stmt` (so an arm
    /// `return` / `propagate` / `break` / `continue` lowers against the enclosing
    /// expansion / caller loop).  Uses a `switch` for a pure enum-tag dispatch, and
    /// an if-chain (label + goto, so a user break/continue targets the caller's
    /// loop, not the dispatch) for everything else - guards, lit-checks, non-enum
    /// scrutinees, or any arm that itself contains a break/continue.
    fn emit_inline_match(&mut self, inline_f: &HFunc, scrutinee: &HExpr, arms: &[HMatchArm], match_result_ty: &HType, tag: &str) -> String {
        self.inline_call_seq += 1;
        let mt = format!("{}_{}", tag, self.inline_call_seq);
        let s = self.emit_inline_expr(inline_f, scrutinee, tag);
        let scrut_c = self.c_type(&scrutinee.ty);
        let needs_value = !matches!(match_result_ty, HType::Unit);
        let res_c = self.c_type(match_result_ty);
        let fn_ret = self.inline_ret_stack.last().cloned().unwrap_or(HType::Unit);
        let mut body = String::new();
        body.push_str("__extension__ ({ ");
        body.push_str(&format!("{} __s_{} = {}; ", scrut_c, mt, s));
        if needs_value { body.push_str(&format!("{} __rm_{} = {{0}}; ", res_c, mt)); }

        // A switch captures `break`, so an arm containing a (caller-targeting)
        // break/continue must use the if-chain even for a pure enum dispatch.
        let arms_jump = arms.iter().any(|a|
            a.guard.as_ref().is_some_and(expr_has_break_continue)
            || block_has_break_continue(&a.body)
            || a.value.as_ref().is_some_and(expr_has_break_continue));
        let switch_eid = if arms_jump { None } else { self.match_as_switch(&scrutinee.ty, arms) };

        if let Some(eid) = switch_eid {
            let tag_expr = if self.sym.enum_info(eid).is_simple() { format!("__s_{}", mt) } else { format!("__s_{}.tag", mt) };
            body.push_str(&format!("switch ({}) {{ ", tag_expr));
            for a in arms {
                match &a.kind {
                    HArmKind::Variant { variant, enum_id, .. } => {
                        let t = self.sym.enum_info(*enum_id).variants[*variant].tag;
                        body.push_str(&format!("case {}: {{ ", t));
                    }
                    _ => body.push_str("default: { "),
                }
                body.push_str(&self.emit_inline_arm_bindings(inline_f, a, scrutinee, &s, &mt, tag));
                body.push_str(&self.emit_inline_arm_body(inline_f, a, &mt, tag, needs_value, &fn_ret));
                body.push_str("} break; ");
            }
            body.push_str("} ");
        } else {
            // If-chain: handles non-enum scrutinees, guards, lit-checks, AND any arm
            // containing a break/continue (the switch would swallow it).  Arms
            // terminate with goto, so a user break/continue falls through to the
            // caller's loop.
            self.yield_label_seq += 1;
            let mend = format!("__mend_{}", self.yield_label_seq);
            let sref = format!("__s_{}", mt);
            for a in arms {
                let pred = self.inline_arm_pred(a, &sref);
                let guard = a.guard.as_ref().map(|g| self.emit_inline_expr(inline_f, g, tag));
                let bindings = self.emit_inline_arm_bindings(inline_f, a, scrutinee, &s, &mt, tag);
                // A variant or scrut-bound arm's guard must run AFTER bindings (which
                // it may read), so place it inside the arm body, not in the predicate.
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
                if let Some(g) = &body_guard { body.push_str(&format!("if ({}) {{ ", g)); }
                body.push_str(&self.emit_inline_arm_body(inline_f, a, &mt, tag, needs_value, &fn_ret));
                if body_guard.is_some() {
                    body.push_str(&format!(" goto {}; }} }} ", mend));
                } else {
                    body.push_str(&format!(" goto {}; }} ", mend));
                }
            }
            body.push_str(&format!("{}: ; ", mend));
        }
        if needs_value { body.push_str(&format!("__rm_{}; ", mt)); }
        else { body.push_str("MAKA_UNIT; "); }
        body.push_str("})");
        body
    }

    /// Emit a spliced `propagate` value.  The value becomes a bare `return` of the
    /// enclosing C function (`cur_c_func_ret`).  When the inline helper was
    /// monomorphized with its own type args, the value may be a DIFFERENT
    /// instantiation of the same enum template than the caller's return type (e.g.
    /// try_ok's `Result<T,E>.Err` propagated into the caller's `Result<U,E>`): the
    /// constructed variant's data is identical, but the C struct type differs and
    /// the variant layouts may differ, so re-emit the variant against the caller's
    /// enum instead of returning the mismatched type.
    fn emit_inline_propagate_value(&mut self, inline_f: &HFunc, v: &HExpr, tag: &str) -> String {
        let target = self.cur_c_func_ret.clone();
        if !type_eq(&v.ty, &target) {
            if let (HExprKind::VariantCtor { variant, fields, .. }, HType::Enum(teid)) = (&v.kind, &target) {
                let tinfo = self.sym.enum_info(*teid).clone();
                if let Some(tv) = tinfo.variants.get(*variant) {
                    if tv.fields.is_empty() {
                        if tinfo.is_simple() {
                            return format!("{}__{}", c_ident(&tinfo.name), c_ident(&tv.name));
                        }
                        return format!("(({}){{ .tag = {} }})", c_ident(&tinfo.name), tv.tag);
                    }
                    let parts: Vec<String> = tv.fields.iter().enumerate().map(|(i, fi)| {
                        let s = fields.iter().find(|(idx, _)| *idx == i)
                            .map(|(_, e)| self.emit_inline_expr(inline_f, e, tag))
                            .unwrap_or_else(|| "0".into());
                        format!(".{} = {}", c_ident(&fi.name), s)
                    }).collect();
                    return format!("(({}){{ .tag = {}, .payload.{} = {{ {} }} }})",
                        c_ident(&tinfo.name), tv.tag, c_ident(&tv.name), parts.join(", "));
                }
            }
        }
        self.emit_inline_expr(inline_f, v, tag)
    }

    fn emit_inline_expansion(&mut self, caller_f: &HFunc, callee: FuncId, args: &[HExpr], result_ty: &HType, prop_drops: Vec<(String, HType)>, loop_drops: Vec<(String, HType)>) -> String {
        // Top-level expansion: args live in the caller's C function, so render
        // them with the ordinary emitter.
        let arg_strs: Vec<String> = args.iter().map(|a| self.emit_expr(caller_f, a)).collect();
        self.emit_inline_expansion_args(callee, &arg_strs, result_ty, prop_drops, loop_drops)
    }

    /// Inline expansion given PRE-RENDERED argument strings.  Splitting argument
    /// rendering out of the body expansion lets a NESTED `InlineCall` (one that
    /// appears inside an inline function's body) render its args in the OUTER
    /// inline's tagged scope (via `emit_inline_expr`), while a top-level call
    /// renders them in the caller's scope (via `emit_expr`).  The nested call
    /// still gets its own fresh `tag` for its locals, so names never collide.
    fn emit_inline_expansion_args(&mut self, callee: FuncId, arg_strs: &[String], result_ty: &HType, prop_drops: Vec<(String, HType)>, loop_drops: Vec<(String, HType)>) -> String {
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
        for (i, &pid) in inline_f.params.iter().enumerate() {
            let li = &inline_f.locals[pid.0 as usize];
            let pname = inline_local_name(&inline_f, pid, &tag);
            let arg_s = arg_strs.get(i).cloned().unwrap_or_else(|| "0".into());
            match &li.ty {
                // A by-value array param is an independent COPY (value semantics,
                // like any array).  In the inline splice, declare a real local
                // array and memcpy the caller's array into it, so writes to the
                // param don't reach the caller.  `&[N]T` borrows are `Ref`s and
                // hit the `_` arm (a plain pointer copy), untouched.
                HType::Array { .. } => {
                    out.push_str(&format!("{0}; memcpy({1}, ({2}), sizeof({1})); ",
                        self.c_decl(&li.ty, &pname), pname, arg_s));
                }
                _ => {
                    let ty = self.c_type(&li.ty);
                    out.push_str(&format!("{} {} = {}; ", ty, pname, arg_s));
                }
            }
        }

        // Declare result var if needed.
        if needs_value {
            let res_c = self.c_type(result_ty);
            out.push_str(&format!("{} __r_{} = {{0}}; ", res_c, tag));
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
                _ => out.push_str(&format!("{} {} = {{0}}; ", ty, pname)),
            }
        }

        // Emit the body in a PLAIN block, not `do { ... } while (0)`.  `return`
        // already exits via `goto end_<tag>`, so the do/while wrapper is dead weight -
        // and being a C loop it would CAPTURE a user `break`/`continue` written in the
        // inline body (which is meant to target the caller's loop), silently exiting
        // the match/expansion instead.  A plain block lets such a jump fall through to
        // the real enclosing loop; codegen emits its drops at the jump (below).
        out.push_str("{ ");
        self.inline_ret_stack.push(result_ty.clone());
        self.inline_propagate_drops.push(prop_drops);
        self.inline_loop_drops.push(loop_drops);
        for stmt in &inline_f.body.stmts {
            out.push_str(&self.emit_inline_stmt(&inline_f, stmt, &tag, result_ty));
        }
        self.inline_loop_drops.pop();
        self.inline_propagate_drops.pop();
        self.inline_ret_stack.pop();
        out.push_str("} ");
        out.push_str(&format!("end_{}: ; ", tag));

        // Auto-free any heap/own locals the inline owns.  Ownership transferred
        // in via parameters lands here too: the lifetime pass marks the OUTER
        // binding as moved at the InlineCall site, so the outer scope-exit drop
        // is suppressed and the only free happens here, inside the splice.
        //
        // Route through the canonical recursive drop (emit_field_drop via
        // capture_drops), NOT a hand-rolled per-type match: an `own *Struct` whose
        // struct owns heap, or a by-value `Vec<own *T>`, must have its pointee /
        // elements dropped first - a bare `free` on the outer pointer/buffer leaks
        // everything they transitively own.  Same mechanism as inline_block_drops.
        let scope_drops: Vec<(String, HType)> = inline_f.body.heap_to_free.iter().map(|id| {
            (inline_local_name(&inline_f, *id, &tag), inline_f.locals[id.0 as usize].ty.clone())
        }).collect();
        out.push_str(&self.capture_drops(&scope_drops));

        if needs_value { out.push_str(&format!("__r_{}; ", tag)); }
        else { out.push_str("MAKA_UNIT; "); }
        out.push_str("})");
        out
    }

    /// Emit a statement from the inline function's body, with substitutions.
    /// Drops for an inline block's scope-exit `heap_to_free` (owning locals of the
    /// inline frame going out of scope at the block end, incl. compensating frees
    /// the lifetime pass synthesized for a value moved on only one branch).  The
    /// non-inline block path emits these; the inline expansion must too or they leak.
    fn inline_block_drops(&mut self, inline_f: &HFunc, b: &HBlock, tag: &str) -> String {
        if b.heap_to_free.is_empty() { return String::new(); }
        let drops: Vec<(String, HType)> = b.heap_to_free.iter().map(|id| {
            let li = &inline_f.locals[id.0 as usize];
            (inline_local_name(inline_f, *id, tag), li.ty.clone())
        }).collect();
        self.capture_drops(&drops)
    }

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
                    // An array local is DECLARED up front (c_decl, a real array); a C
                    // array is not `=`-assignable, so initialise it via memcpy - an
                    // array-literal RHS becomes a compound literal to copy from
                    // (mirrors emit_assign's whole-array path).
                    HType::Array { .. } => {
                        let src = if matches!(&init.kind, HExprKind::ArrayLit(_)) {
                            format!("(__typeof__({})){}", n, v)
                        } else { v };
                        format!("memcpy({}, {}, sizeof({})); ", n, src, n)
                    }
                    _ => format!("{} = {}; ", n, v),
                }
            }
            HStmt::Assign { op, place, value, .. } => {
                let v = self.emit_inline_expr(inline_f, value, tag);
                // A `yield` assigns an arm's synthetic `__yield` local; terminate
                // the arm so a yield nested in an if/while wins (mirrors emit_stmt).
                let goto = if let HExprKind::Local(id) = place.kind {
                    self.yield_exit.iter().rev().find(|(yv, _)| *yv == id)
                        .map(|(_, l)| format!("goto {}; ", l)).unwrap_or_default()
                } else { String::new() };
                // A whole-array reassignment is not `=`-assignable in C; copy it
                // (mirrors emit_assign's array path).  Only for a plain Assign.
                if matches!(op, HAssignOp::Assign) && matches!(&place.ty, HType::Array { .. }) {
                    let lhs = self.emit_inline_place(inline_f, place, tag);
                    let src = if matches!(&value.kind, HExprKind::ArrayLit(_)) {
                        format!("(__typeof__({})){}", lhs, v)
                    } else { v };
                    return format!("memcpy({}, {}, sizeof({})); {}", lhs, src, lhs, goto);
                }
                // Write THROUGH a `&mut T` local, or a pointer-unwrap place `p!`: the
                // place renders as the POINTER, so it must be dereferenced (mirrors
                // emit_assign).  Without this an `&mut` mutation in an inline body -
                // `inline f(&mut int p) { p = v }`, or a by-ref foreach binding -
                // silently overwrites the pointer instead of the referent.
                let (lhs, slot_ty) = if let HExprKind::Local(id) = place.kind {
                    if let HType::Ref { mutable: true, inner } = &inline_f.locals[id.0 as usize].ty {
                        (format!("*{}", self.emit_inline_place(inline_f, place, tag)), (**inner).clone())
                    } else {
                        (self.emit_inline_place(inline_f, place, tag), place.ty.clone())
                    }
                } else if let HExprKind::Unwrap { expr, .. } = &place.kind {
                    (format!("*({})", self.emit_inline_expr(inline_f, expr, tag)), place.ty.clone())
                } else {
                    (self.emit_inline_place(inline_f, place, tag), place.ty.clone())
                };
                format!("{} = {}; {}", lhs, self.compound_rhs(&lhs, *op, &v, &slot_ty), goto)
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
            HStmt::Propagate { value, heap_drops, .. } => {
                // A `propagate` early-returns the OUTERMOST non-inline C frame, so
                // free the owning locals live in EVERY active inline frame between
                // here and that return - not just the innermost - else an outer
                // frame's locals leak (nested inline: try_ok inside an inline fn).
                // Frames hold disjoint locals (distinct functions / tagged names);
                // emit innermost-first (stack top -> bottom = reverse alloc order).
                // THIS inline frame's own live owning locals (heap_drops, filled by
                // the lifetime pass) come first - they were being missed entirely.
                let mut all_drops: Vec<(String, HType)> = heap_drops.iter().map(|id| {
                    let li = &inline_f.locals[id.0 as usize];
                    (inline_local_name(inline_f, *id, tag), li.ty.clone())
                }).collect();
                all_drops.extend(self.inline_propagate_drops.iter().rev().flatten().cloned());
                let dc = self.capture_drops(&all_drops);
                match value {
                    Some(v) => {
                        let s = self.emit_inline_propagate_value(inline_f, v, tag);
                        if dc.is_empty() {
                            format!("return {}; ", s)
                        } else {
                            // Evaluate the propagated value BEFORE the drops, in case
                            // it reads one of them, then drop, then return.
                            let rt = self.c_type(&self.cur_c_func_ret.clone());
                            format!("{{ {} __pv = {}; {}return __pv; }} ", rt, s, dc)
                        }
                    }
                    None => format!("{}return; ", dc),
                }
            }
            HStmt::If { cond, then_b, else_b, .. } => {
                let cs = self.emit_inline_expr(inline_f, cond, tag);
                let mut s = format!("if ({}) {{ ", cs);
                for st in &then_b.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                // Emit the block's scope-exit / compensating drops (a value moved on
                // only one branch gets a compensating free on the sibling; the
                // lifetime pass fills heap_to_free with it).  Non-inline block
                // codegen emits these; the inline expansion was dropping them.
                s.push_str(&self.inline_block_drops(inline_f, then_b, tag));
                s.push_str("} ");
                if let Some(b) = else_b {
                    s.push_str("else { ");
                    for st in &b.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                    s.push_str(&self.inline_block_drops(inline_f, b, tag));
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
            HStmt::Break { heap_drops, .. } | HStmt::Continue { heap_drops, .. } => {
                // A break/continue spliced from an inline body.  If it targets a loop
                // WITHIN the inline, heap_drops/inline_loop_drops are empty and this is
                // a plain jump.  If it is a caller-targeting jump (no inline-internal
                // loop), free the inline's own owning locals live here (heap_drops, set
                // return-like by the lifetime pass) PLUS the caller's loop-body owning
                // locals (threaded from the call site) - the jump skips both frames'
                // scope-exit drops.
                let inline_drops: Vec<(String, HType)> = heap_drops.iter().map(|id| {
                    let li = &inline_f.locals[id.0 as usize];
                    (inline_local_name(inline_f, *id, tag), li.ty.clone())
                }).collect();
                let mut dc = self.capture_drops(&inline_drops);
                // Free EVERY active inline frame's locals between here and the target
                // loop (innermost first), not just the nearest - a jump from a nested
                // inline must free each intermediate frame too (cf. the nested
                // propagate fix).  Each frame's set is its caller's locals to abandon.
                let caller_loop: Vec<(String, HType)> =
                    self.inline_loop_drops.iter().rev().flatten().cloned().collect();
                dc.push_str(&self.capture_drops(&caller_loop));
                let kw = if matches!(s, HStmt::Break { .. }) { "break" } else { "continue" };
                format!("{}{}; ", dc, kw)
            }
            HStmt::ForC { init, cond, step, body, .. } => {
                let mut s = self.emit_inline_stmt(inline_f, init, tag, result_ty);
                let cs = self.emit_inline_expr(inline_f, cond, tag);
                let step_text = match step.as_ref() {
                    HStmt::Assign { op, place, value, .. } => {
                        let p = self.emit_inline_place(inline_f, place, tag);
                        let v = self.emit_inline_expr(inline_f, value, tag);
                        format!("{} = {}", p, self.compound_rhs(&p, *op, &v, &place.ty))
                    }
                    _ => "(void)0".into(),
                };
                s.push_str(&format!("for (; {}; {}) {{ ", cs, step_text));
                let bounded = forrange_bound(init, cond, body);
                if let Some(b) = bounded { self.bounded_vars.push(b); }
                for st in &body.stmts { s.push_str(&self.emit_inline_stmt(inline_f, st, tag, result_ty)); }
                if bounded.is_some() { self.bounded_vars.pop(); }
                s.push_str("} ");
                s
            }
            HStmt::ForEach { var, src, body, .. } => {
                // Inline analogue of the non-inline ForEach (emit_stmt), emitted as
                // a statement-expression string with tag-threaded local names.  The
                // alias-elision optimisation is dropped here (elements are copied by
                // value or bound by reference), keeping the lowering simple and safe.
                let li = &inline_f.locals[var.0 as usize];
                let var_name = inline_local_name(inline_f, *var, tag);
                let var_ty = self.c_type(&li.ty);
                let src_c = self.emit_inline_expr(inline_f, src, tag);
                let mut s = String::from("{ ");
                let (len_str, elem_access) = match &src.ty {
                    HType::Array { len, .. } => (format!("(maka_int){}", len), src_c.clone()),
                    HType::Slice { .. } => {
                        s.push_str(&format!("{} __src = {}; ", self.c_type(&src.ty), src_c));
                        ("__src.len".to_string(), "__src.ptr".to_string())
                    }
                    HType::Vec { .. } => {
                        s.push_str(&format!("{} __src = {}; ", self.c_type(&src.ty), src_c));
                        ("__src.len".to_string(), "__src.data".to_string())
                    }
                    HType::Heap { inner } if matches!(inner.as_ref(), HType::Vec { .. }) => {
                        s.push_str(&format!("{} __src = {}; ", self.c_type(&src.ty), src_c));
                        ("__src.len".to_string(), "__src.data".to_string())
                    }
                    HType::Ref { inner, .. } | HType::Ptr { inner, .. }
                    | HType::OwnPtr { inner, .. } | HType::RawPtr { inner, .. }
                        if matches!(inner.as_ref(), HType::Vec { .. } | HType::Slice { .. }) =>
                    {
                        s.push_str(&format!("{} __src = *({}); ", self.c_type(inner), src_c));
                        if matches!(inner.as_ref(), HType::Slice { .. }) {
                            ("__src.len".to_string(), "__src.ptr".to_string())
                        } else {
                            ("__src.len".to_string(), "__src.data".to_string())
                        }
                    }
                    HType::Ref { inner, .. } | HType::Ptr { inner, .. }
                    | HType::OwnPtr { inner, .. } | HType::RawPtr { inner, .. }
                        if matches!(inner.as_ref(), HType::Array { .. }) =>
                    {
                        if let HType::Array { len, .. } = inner.as_ref() {
                            (format!("(maka_int){}", len), src_c.clone())
                        } else { unreachable!() }
                    }
                    _ => ("0".to_string(), src_c.clone()),
                };
                let li_inner: Option<&HType> = match &li.ty {
                    HType::Ref { inner, .. } | HType::Ptr { inner, .. }
                    | HType::OwnPtr { inner, .. } | HType::RawPtr { inner, .. } => Some(inner),
                    _ => None,
                };
                let elem_ty = foreach_elem_ty(&src.ty);
                let bind_by_ref = match (li_inner, elem_ty) {
                    (Some(inner), Some(et)) =>
                        self.c_type(inner) == self.c_type(et) && self.c_type(&li.ty) != self.c_type(et),
                    _ => false,
                };
                let body_s: String = body.stmts.iter()
                    .map(|st| self.emit_inline_stmt(inline_f, st, tag, result_ty)).collect();
                if bind_by_ref {
                    s.push_str(&format!("for (maka_int __i = 0; __i < {}; __i += 1) {{ {} {} = &({}[__i]); {}}} ",
                        len_str, var_ty, var_name, elem_access, body_s));
                } else if matches!(&li.ty, HType::Array { .. }) {
                    s.push_str(&format!("for (maka_int __i = 0; __i < {}; __i += 1) {{ {}; memcpy({}, {}[__i], sizeof({})); {}}} ",
                        len_str, self.c_decl(&li.ty, &var_name), var_name, elem_access, var_name, body_s));
                } else {
                    s.push_str(&format!("{} {} = {{0}}; for (maka_int __i = 0; __i < {}; __i += 1) {{ {} = {}[__i]; {}}} ",
                        var_ty, var_name, len_str, var_name, elem_access, body_s));
                }
                // A temporary owning container source (`for x in make_vec()`) has no
                // other owner: free it (elements + buffer) after the loop.
                if matches!(&src.kind, HExprKind::Call { .. } | HExprKind::Match { .. })
                    && matches!(&src.ty, HType::Vec { .. } | HType::Heap { .. })
                    && self.drop_ty_owns(&src.ty)
                {
                    s.push_str(&self.capture_drops(&[("__src".to_string(), src.ty.clone())]));
                }
                s.push_str("} ");
                s
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
                if needs_div_guard(*op, &lhs.ty, rhs) {
                    return self.emit_div(&l, *op, &r, &lhs.ty);
                }
                if matches!(op, HBinOp::Shl | HBinOp::Shr) {
                    return self.emit_shift(&l, *op, &r, &lhs.ty);
                }
                self.emit_arith(&l, *op, &r, &lhs.ty)
            }
            HExprKind::Un { op, expr } => {
                if let (HUnOp::Neg, HExprKind::LitInt(n)) = (op, &expr.kind) {
                    return c_int_lit(n.wrapping_neg());
                }
                let v = self.emit_inline_expr(inline_f, expr, tag);
                match op {
                    HUnOp::Neg => self.emit_neg(&v, &expr.ty),
                    HUnOp::Not => format!("(!({}))", v),
                    // C integer-promotes `~`, so cast back to the operand's type
                    // to mask the result into its width (e.g. `~(u8)0` is 255, not -1).
                    HUnOp::BitNot => format!("(({})(~({})))", self.c_type(&expr.ty), v),
                }
            }
            HExprKind::Unwrap { expr, skip_check: _ } => {
                let s = self.emit_inline_expr(inline_f, expr, tag);
                if Self::ptr_pointee_decays(&expr.ty) { return s; }
                format!("(*({}))", s)
            }
            HExprKind::AddrOfRef { place, .. } => {
                if matches!(place.ty, HType::Dyn { .. }) {
                    return self.emit_inline_place(inline_f, place, tag);
                }
                let p = self.emit_inline_place(inline_f, place, tag);
                if matches!(&place.ty, HType::Array { .. } | HType::Str) { return format!("({})", p); }
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
                self.index_access(inline_f, base, idx, &bs, &is_)
            }
            HExprKind::DerefRef(inner) => {
                let s = self.emit_inline_expr(inline_f, inner, tag);
                if Self::ptr_pointee_decays(&inner.ty) { return s; }
                format!("(*({}))", s)
            }
            HExprKind::HeapAlloc(inner) => {
                let ic = self.c_type(&inner.ty);
                let v = self.emit_inline_expr(inline_f, inner, tag);
                format!("(__extension__ ({{ {0}* __p = ({0}*)malloc(sizeof({0})); *__p = ({1}); __p; }}))", ic, v)
            }
            HExprKind::Free(inner, deep) => {
                let s = self.emit_inline_expr(inline_f, inner, tag);
                self.emit_free(s, &inner.ty, *deep)
            }
            HExprKind::Match { scrutinee, arms, result_ty } => {
                self.emit_inline_match(inline_f, scrutinee, arms, result_ty, tag)
            }
            // A nested inline call inside this inline's body: its args reference
            // THIS inline's params/locals, so render them in the current tagged
            // scope; the nested expansion then takes its own fresh tag.  Without
            // this the args would fall through to the untagged emitter and emit
            // undeclared `a_0`-style names.
            HExprKind::InlineCall { callee, args, propagate_drops, loop_jump_drops } => {
                let arg_strs: Vec<String> = args.iter()
                    .map(|a| self.emit_inline_expr(inline_f, a, tag))
                    .collect();
                // Nested inline call: the caller's locals here are the OUTER inline's
                // tagged locals, so resolve both drop sets in that tagged scope.
                let pd: Vec<(String, HType)> = propagate_drops.iter().map(|id| {
                    let li = &inline_f.locals[id.0 as usize];
                    (inline_local_name(inline_f, *id, tag), li.ty.clone())
                }).collect();
                let ld: Vec<(String, HType)> = loop_jump_drops.iter().map(|id| {
                    let li = &inline_f.locals[id.0 as usize];
                    (inline_local_name(inline_f, *id, tag), li.ty.clone())
                }).collect();
                self.emit_inline_expansion_args(*callee, &arg_strs, &e.ty, pd, ld)
            }
            // `.len` / enum-tag: emit the base in the tagged scope (the shared
            // `_c` helper is name-agnostic), else the base local would fall through
            // to the untagged emitter and reference an undeclared `x_0`.
            HExprKind::SliceLen(inner) => {
                let s = self.emit_inline_expr(inline_f, inner, tag);
                self.slice_len_c(&inner.ty, &s)
            }
            HExprKind::EnumTag(inner) => {
                let s = self.emit_inline_expr(inline_f, inner, tag);
                self.enum_tag_c(&inner.ty, &s)
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
            HExprKind::LitInt(n) => c_int_lit(*n),
            // `{:?}` (not `{}`): Rust's Display renders e.g. 1e30 as the
            // integer-looking "1000...000", which C parses as an integer literal
            // that overflows; Debug always emits a `.`/`e` form, round-trip exact
            // and a valid C double literal.
            HExprKind::LitFloat(v) => format!("(maka_float){:?}", v),
            HExprKind::LitBool(b) => if *b { "true".into() } else { "false".into() },
            HExprKind::LitChar(c) => format!("(maka_char){}u", *c as u32),
            HExprKind::LitStr(s) => format!("\"{}\"", c_escape(s)),
            HExprKind::LitNull => "NULL".into(),
            HExprKind::LitUnit => "MAKA_UNIT".into(),
            HExprKind::Call { callee, args } => {
                if is_builtin_sentinel(*callee) {
                    // Builtin call spliced from an inline body: reuse the single
                    // dispatch, tagging sub-expression emission so inline locals
                    // resolve to their tagged names.
                    let saved = self.inline_tag.replace(tag.to_string());
                    let r = self.emit_builtin_sentinel(inline_f, *callee, args, e);
                    self.inline_tag = saved;
                    return r;
                }
                let sig = self.sym.func_sig(*callee);
                let name = if sig.is_extern { sig.c_name.clone() }
                    else if sig.name == "main" && sig.trait_name.is_none() { "maka_main".to_string() }
                    else { c_ident(&sig.c_name) };
                let arg_s: Vec<String> = args.iter().map(|a| self.emit_inline_expr(inline_f, a, tag)).collect();
                let call = format!("{}({})", name, arg_s.join(", "));
                if matches!(sig.ret, HType::Unit) {
                    format!("({}, MAKA_UNIT)", call)
                } else { call }
            }
            HExprKind::Cast { expr, kind, to } => {
                if let CastKind::ToDyn { traits, struct_id } = kind {
                    return self.emit_to_dyn(inline_f, expr, traits, *struct_id);
                }
                let s = self.emit_inline_expr(inline_f, expr, tag);
                self.emit_cast(s, kind.clone(), to, &expr.ty)
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
                self.index_access(inline_f, base, idx, &bs, &is_)
            }
            HExprKind::Unwrap { expr, skip_check: _ } => {
                let s = self.emit_inline_expr(inline_f, expr, tag);
                format!("(*({}))", s)
            }
            _ => self.emit_inline_expr(inline_f, e, tag),
        }
    }

    /// Moving an owning value out of a place (`own *T px = c.x;`) invalidates the
    /// source field/element so the owner's drop skips it - no double free.  Only
    /// nulls when the source is genuinely owned (not read through a borrow).
    /// Emit `e` in a by-value CONSUMING position (call/ctor/return value).  A
    /// by-value read of an owning struct/array/Vec field-or-element (or an
    /// own-ptr/heap one) MOVES it out of its container, so capture the value and
    /// null the source slot in the same expression - otherwise the container's
    /// later drop double-frees the value now owned by the consumer.  A whole
    /// local moved this way is tracked by the compile-time moved set; only a
    /// field/index slot, which a monolithic container drop cannot selectively
    /// skip, needs the runtime null (the statement-form analogue is
    /// emit_move_out_null, used for Let/Assign).
    /// Emit a `<<` / `>>` shift with full well-definedness.  C left-shifts have
    /// three undefined cases: a shift amount >= the operand width, a negative
    /// amount, and a left shift into the sign bit of a signed type.  Guard the
    /// amount (panic if out of range, like maka_check_div) and, for a signed left
    /// shift, perform the shift in the unsigned counterpart and reinterpret back
    /// (well-defined two's-complement).  The result is cast to the operand type so
    /// a narrow type is masked to its width.
    fn emit_shift(&self, l: &str, op: HBinOp, r: &str, lhs_ty: &HType) -> String {
        let w = shift_width(lhs_ty);
        let checked = format!("maka_check_shift({}, {})", r, w);
        let st = self.c_type(lhs_ty);
        if matches!(op, HBinOp::Shl) && is_signed_int(lhs_ty) {
            let ut = unsigned_ctype(lhs_ty);
            format!("(({0})(({1})({2}) << {3}))", st, ut, l, checked)
        } else {
            format!("(({0})(({1}) {2} {3}))", st, l, binop_c(op), checked)
        }
    }

    /// Emit an integer `/` or `%` with full well-definedness.  Both have two
    /// undefined cases in C: a zero divisor and the signed `MIN / -1` overflow.
    /// A signed op routes through maka_divs/maka_mods (panic on either, like
    /// shift overflow); an unsigned op (which cannot overflow) keeps the cheaper
    /// divide-by-zero-only guard.
    fn emit_div(&self, l: &str, op: HBinOp, r: &str, lhs_ty: &HType) -> String {
        if is_signed_int(lhs_ty) {
            let mac = if matches!(op, HBinOp::Mod) { "maka_mods" } else { "maka_divs" };
            return format!("{}({}, {}, {})", mac, l, r, signed_min_lit(lhs_ty));
        }
        format!("(({}) {} maka_check_div({}))", l, binop_c(op), r)
    }

    /// Emit a binary op, making signed `+` / `-` / `*` overflow WRAP instead of
    /// being C undefined behaviour: compute in the unsigned counterpart (which
    /// wraps by definition) and reinterpret back.  Branchless and zero-cost, and
    /// identical to the raw op for any non-overflowing value.  Non-arithmetic and
    /// unsigned ops keep the plain form.
    fn emit_arith(&self, l: &str, op: HBinOp, r: &str, lhs_ty: &HType) -> String {
        // C's `%` is integer-only; floating-point modulo needs fmod.
        // __builtin_fmod avoids a <math.h>/-lm dependency.
        if matches!(op, HBinOp::Mod) && is_float_ty(lhs_ty) {
            return format!("(({0})__builtin_fmod((double)({1}), (double)({2})))", self.c_type(lhs_ty), l, r);
        }
        if matches!(op, HBinOp::Add | HBinOp::Sub | HBinOp::Mul) && is_signed_int(lhs_ty) {
            let st = self.c_type(lhs_ty);
            let ut = unsigned_ctype(lhs_ty);
            return format!("(({0})(({1})({2}) {3} ({1})({4})))", st, ut, l, binop_c(op), r);
        }
        // Unsigned NARROW types (uint8_t/uint16_t) integer-promote to signed `int`
        // in C, so u8/u16 add/sub/mul can overflow int (UB).  Compute in unsigned
        // int (which never promotes to signed) and truncate back.  u32/u64 are >=
        // int width, stay unsigned, and wrap by definition, so they use the plain form.
        if matches!(op, HBinOp::Add | HBinOp::Sub | HBinOp::Mul) {
            let ct = self.c_type(lhs_ty);
            if ct == "uint8_t" || ct == "uint16_t" {
                return format!("(({0})((unsigned int)({1}) {2} (unsigned int)({3})))", ct, l, binop_c(op), r);
            }
        }
        format!("(({}) {} ({}))", l, binop_c(op), r)
    }

    /// The RHS of a compound assignment `place OP= value`, rewritten as the value
    /// of `place = <this>` so it goes through the SAME guarded/wrapped arithmetic
    /// as `place = place OP value`: `/=`/`%=` get the divide-by-zero and
    /// INT_MIN/-1 guard, `+=`/`-=`/`*=` get the signed-overflow wrap.  Emitting a
    /// raw C `place OP= value` bypasses both (UB).  `lhs` must be a side-effect-
    /// free lvalue string (it is evaluated twice for a non-Assign op).
    fn compound_rhs(&self, lhs: &str, op: HAssignOp, rhs: &str, lhs_ty: &HType) -> String {
        match op {
            HAssignOp::Assign => rhs.to_string(),
            HAssignOp::Add => self.emit_arith(lhs, HBinOp::Add, rhs, lhs_ty),
            HAssignOp::Sub => self.emit_arith(lhs, HBinOp::Sub, rhs, lhs_ty),
            HAssignOp::Mul => self.emit_arith(lhs, HBinOp::Mul, rhs, lhs_ty),
            HAssignOp::Div => self.emit_div(lhs, HBinOp::Div, rhs, lhs_ty),
            HAssignOp::Mod => self.emit_div(lhs, HBinOp::Mod, rhs, lhs_ty),
        }
    }

    /// Emit unary negation; signed `-MIN` is C UB, so negate in the unsigned
    /// counterpart (wraps) and reinterpret back.  Unsigned negation is already
    /// well-defined in C.
    fn emit_neg(&self, v: &str, ty: &HType) -> String {
        if is_signed_int(ty) {
            let st = self.c_type(ty);
            let ut = unsigned_ctype(ty);
            return format!("(({0})(-({1})({2})))", st, ut, v);
        }
        format!("(-({}))", v)
    }

    /// Emit a sub-expression, honouring the active inline expansion tag.  When
    /// `inline_tag` is set (a builtin-sentinel call spliced from an inline body),
    /// locals must render with their tagged names; otherwise this is plain
    /// `emit_expr`.  Every sub-emission inside `emit_builtin_sentinel` goes through
    /// here so the single dispatch is correct in both the normal and inline paths.
    fn emit_sub(&mut self, f: &HFunc, e: &HExpr) -> String {
        if let Some(tag) = self.inline_tag.clone() {
            self.emit_inline_expr(f, e, &tag)
        } else {
            self.emit_expr(f, e)
        }
    }

    /// `emit_sub` for an lvalue/place (see `emit_sub`).
    fn emit_place_sub(&mut self, f: &HFunc, e: &HExpr) -> String {
        if let Some(tag) = self.inline_tag.clone() {
            self.emit_inline_place(f, e, &tag)
        } else {
            self.emit_place(f, e)
        }
    }

    /// C for `.len` on a slice / vec / array (or a borrow / pointer to one), given
    /// the already-emitted base string `s`.  Shared by the normal and the inline
    /// emitter so the base can be rendered tagged or untagged by the caller.
    fn slice_len_c(&self, inner_ty: &HType, s: &str) -> String {
        match inner_ty {
            HType::Slice { .. } => format!("({}).len", s),
            HType::Vec { .. } => format!("({}).len", s),
            HType::Heap { inner: i } => match i.as_ref() {
                HType::Vec { .. } => format!("({}).len", s),
                HType::Array { len, .. } => format!("(maka_int){}", len),
                _ => "0".into(),
            },
            // `.len` through a borrow / non-owning pointer to an array, vec, or
            // slice.  `&Vec<T>` / `&[]T` is a pointer to the {data/ptr,len,cap}
            // struct, so the count is at `(*base).len`.
            HType::Ref { inner: i, .. } | HType::Ptr { inner: i, .. } | HType::OwnPtr { inner: i, .. } => match i.as_ref() {
                HType::Vec { .. } | HType::Slice { .. } => format!("({})->len", s),
                HType::Array { len, .. } => format!("(maka_int){}", len),
                _ => "0".into(),
            },
            HType::Array { len, .. } => format!("(maka_int){}", len),
            _ => "0".into(),
        }
    }

    /// C for an enum's discriminant tag, given the already-emitted base `s` (see
    /// `slice_len_c` for why the base is passed in pre-emitted).
    fn enum_tag_c(&self, inner_ty: &HType, s: &str) -> String {
        if let HType::Enum(eid) = inner_ty {
            if self.sym.enum_info(*eid).is_simple() {
                // Simple enum: the C value IS the tag.
                return format!("(maka_int)({})", s);
            }
        }
        format!("(maka_int)(({}).tag)", s)
    }

    fn emit_move_consuming(&mut self, f: &HFunc, e: &HExpr) -> String {
        let s = self.emit_sub(f, e);
        // A bare `own *T` Local moved into an owning destination (this fn only runs
        // at owning-destination sites) AUTO-NULLS the source at the move site, so a
        // same-statement reassign isn't clobbered and its scope-exit free is a
        // no-op (`free(NULL)`).  A borrow of the same local is coerced to `*T`/`&T`
        // (different kind/type), so this hits moves only.
        if let HExprKind::Local(id) = &e.kind {
            let place = self.local_name(*id, &f.locals[id.0 as usize].name);
            let ty = self.c_type(&e.ty);
            if matches!(&e.ty, HType::OwnPtr { .. }) {
                return format!("(__extension__ ({{ {0} __mc = {1}; {2} = NULL; __mc; }}))", ty, s, place);
            }
            // Field-granular struct moved by value: copy it, then clear ALL its
            // owning fields in the source (own* -> NULL, Vec -> empty, nested ->
            // recurse) so the source stays a valid value and its buffer isn't
            // freed twice.  (Filtering own* only was the pass-by-value double-free.)
            if self.struct_own_ptr_granular(&e.ty) {
                let nulls = self.field_granular_null_stmts(&place, &e.ty);
                return format!("(__extension__ ({{ {0} __mc = {1}; {2}__mc; }}))", ty, s, nulls);
            }
        }
        if !matches!(&e.kind, HExprKind::Field { .. } | HExprKind::Index { .. }) { return s; }
        if !move_out_owned_place(e) { return s; }
        let nullv = if matches!(&e.ty, HType::OwnPtr { .. } | HType::Heap { .. }) {
            "NULL".to_string()
        } else if self.drop_ty_owns(&e.ty) {
            format!("({}){{0}}", self.c_type(&e.ty))
        } else {
            return s;
        };
        let place = self.emit_place_sub(f, e);
        let ty = self.c_type(&e.ty);
        format!("(__extension__ ({{ {0} __mc = {1}; {2} = {3}; __mc; }}))", ty, s, place, nullv)
    }

    /// A struct all of whose owning fields are `own *T` (nullable) - the
    /// field-granular-movable shape: copying it copies the plain fields and moves
    /// (nulls) the own* fields, leaving the source usable.  Excludes own& / Vec /
    /// nested-owning fields (those keep whole-value move).  Mirrors sema's
    /// `is_own_ptr_granular`.
    /// An owning field clearable to a moved-out sentinel whose stale use is CAUGHT:
    /// `own *T` (NULL, caught by the `!` deref proof) or a nested all-`own *T`
    /// field-granular struct.  Mirrors sema's `is_zeroable_owning`.  `Vec`/`String`
    /// (no proof gate -> a cleared-to-empty field reads back silently, and for
    /// String's `len-1` length that is `-1`) and `own &T` (non-nullable) are
    /// EXCLUDED: their container whole-moves-and-poisons instead.
    fn struct_field_zeroable(&self, ty: &HType) -> bool {
        match ty {
            HType::OwnPtr { .. } => true,
            HType::Struct(_) => self.struct_own_ptr_granular(ty),
            _ => false,
        }
    }
    fn struct_own_ptr_granular(&self, ty: &HType) -> bool {
        if let HType::Struct(sid) = ty {
            let si = self.sym.struct_info(*sid);
            si.fields.iter().all(|fi| !self.drop_ty_owns(&fi.ty) || self.struct_field_zeroable(&fi.ty))
                && si.fields.iter().any(|fi| self.drop_ty_owns(&fi.ty))
        } else { false }
    }

    /// The C statements clearing a field-granular struct's owning fields at
    /// `place` to their moved-out sentinel (own* -> NULL, nested -> recurse), as a
    /// string so BOTH move paths share it: `wl` lines (`null_own_ptr_fields`) and a
    /// statement-expression (`emit_move_consuming`).  Only `own *T` (and nested
    /// all-`own *T` structs) are cleared here - a field-granular struct has no
    /// `Vec`/`String`/`own &T` owning fields by construction (`struct_field_zeroable`).
    fn field_granular_null_stmts(&self, place: &str, ty: &HType) -> String {
        let mut out = String::new();
        if let HType::Struct(sid) = ty {
            for fld in self.sym.struct_info(*sid).fields.clone().iter() {
                let fp = format!("({}).{}", place, c_ident(&fld.name));
                match &fld.ty {
                    HType::OwnPtr { .. } => out.push_str(&format!("{} = NULL; ", fp)),
                    HType::Struct(_) if self.struct_own_ptr_granular(&fld.ty) =>
                        out.push_str(&self.field_granular_null_stmts(&fp, &fld.ty)),
                    _ => {}
                }
            }
        }
        out
    }

    /// Clear a field-granular struct's owning fields at `place` to their moved-out
    /// sentinel (own* -> NULL, Vec -> empty, nested -> recurse).  The plain fields
    /// are copied by the struct assignment; only ownership moves.
    fn null_own_ptr_fields(&mut self, place: &str, ty: &HType) {
        let stmts = self.field_granular_null_stmts(place, ty);
        if !stmts.is_empty() { self.wl(&stmts); }
    }

    fn emit_move_out_null(&mut self, f: &HFunc, src: &HExpr) {
        // A bare `own *T` Local moved out (into an owning binding) auto-nulls at
        // the source, so its scope-exit free is `free(NULL)` and a read sees null.
        if let HExprKind::Local(id) = &src.kind {
            if matches!(&src.ty, HType::OwnPtr { .. }) {
                let place = self.local_name(*id, &f.locals[id.0 as usize].name);
                self.wl(&format!("{} = NULL;", place));
            } else if self.struct_own_ptr_granular(&src.ty) {
                // Field-granular struct move: null only the own* fields.
                let place = self.local_name(*id, &f.locals[id.0 as usize].name);
                let ty = src.ty.clone();
                self.null_own_ptr_fields(&place, &ty);
            }
            return;
        }
        if !matches!(&src.kind, HExprKind::Field { .. } | HExprKind::Index { .. }) { return; }
        if !move_out_owned_place(src) { return; }
        if matches!(&src.ty, HType::OwnPtr { .. } | HType::Heap { .. }) {
            let place = self.emit_place(f, src);
            self.wl(&format!("{} = NULL;", place));
        } else if self.drop_ty_owns(&src.ty) {
            // An owning VALUE (struct / Vec / array) moved out of a container slot:
            // zero the slot so the source container's later drop treats it as empty
            // (its owning fields are now null) instead of double-freeing the moved
            // value.  Pointers above just null; a value needs a full zero.
            let place = self.emit_place(f, src);
            let ty = self.c_type(&src.ty);
            self.wl(&format!("{} = ({}){{0}};", place, ty));
        }
    }

    fn emit_let(&mut self, f: &HFunc, id: LocalId, init: &HExpr) {
        let li = &f.locals[id.0 as usize];
        let name = self.local_name(id, &li.name);
        match &li.ty {
            HType::Heap { inner } => {
                // heap [*]T: emit as plain Vec_T struct, no extra heap layer.
                if matches!(inner.as_ref(), HType::Vec { .. }) {
                    let value_s = self.emit_expr(f, init);
                    self.wl(&format!("{} {} = {};", self.c_type(inner), name, value_s));
                    return;
                }
                // Pointer C type for this owning slot.  `own &[N]T` decays to an
                // element pointer (`T*`); `own &T` is `T*`.
                let ptr_ty = self.c_type(&li.ty);
                // Detect move from another heap local: init is HExprKind::Local pointing at a heap local.
                if let HExprKind::Local(src) = init.kind {
                    let sty = f.locals[src.0 as usize].ty.clone();
                    if matches!(sty, HType::Heap { .. }) {
                        // move (own &T: non-null owner, source is dead per poison)
                        self.wl(&format!("{} {} = {}; /* moved */", ptr_ty, name, self.local_name(src, &f.locals[src.0 as usize].name)));
                        return;
                    }
                    if matches!(sty, HType::OwnPtr { .. }) {
                        // move an `own *T` local: copy the pointer, then AUTO-NULL
                        // the source (so its scope-exit free is `free(NULL)` and a
                        // read sees null - the ownership moved to this binding).
                        let srcn = self.local_name(src, &f.locals[src.0 as usize].name);
                        self.wl(&format!("{0} {1} = {2}; {2} = NULL; /* own* move */", ptr_ty, name, srcn));
                        return;
                    }
                }
                // Function call returning heap T: also transferred ownership; capture pointer directly.
                if let HExprKind::Call { .. } = init.kind {
                    if matches!(init.ty, HType::Heap { .. }) {
                        let s = self.emit_expr(f, init);
                        self.wl(&format!("{} {} = {};", ptr_ty, name, s));
                        return;
                    }
                }
                // `alloc value` directly: HeapAlloc already emits a `T*` —
                // assign the pointer, don't re-alloc and deref-assign.
                if matches!(init.kind, HExprKind::HeapAlloc(_)) {
                    let s = self.emit_expr(f, init);
                    self.wl(&format!("{} {} = {};", ptr_ty, name, s));
                    return;
                }
                // A field read whose DECLARED field type is `own &T` (Heap) is a
                // real heap pointer (e.g. `own &Box tmp = p.a;`): transfer it
                // directly.  Using the declared field type (not init.ty) avoids
                // confusing it with a plain `T` value coerced to a Heap slot (a
                // struct literal `own &Node n = Node{..}`), which must allocate.
                if let HExprKind::Field { base, field } = &init.kind {
                    let base_sid = match &base.ty {
                        HType::Struct(id) => Some(*id),
                        HType::Ref { inner, .. } | HType::Heap { inner } | HType::Ptr { inner, .. } | HType::OwnPtr { inner, .. } => {
                            if let HType::Struct(id) = inner.as_ref() { Some(*id) } else { None }
                        }
                        _ => None,
                    };
                    let field_is_heap = base_sid
                        .and_then(|sid| self.sym.struct_info(sid).fields.get(*field).map(|fi| matches!(fi.ty, HType::Heap { .. })))
                        .unwrap_or(false);
                    if field_is_heap {
                        let s = self.emit_expr(f, init);
                        self.wl(&format!("{} {} = {};", ptr_ty, name, s));
                        // Ownership TRANSFERRED out of the field: null the source
                        // so neither a later drop-on-reassign of that field nor
                        // the container's drop frees the moved pointer again
                        // (take-and-replace stays single-owner; a read after the
                        // move faults loudly instead of aliasing a freed box).
                        self.emit_move_out_null(f, init);
                        return;
                    }
                }
                // A match-expression whose arms `yield` owning allocations
                // produces a heap pointer (its result temporary is typed `T*`),
                // so transfer it directly like the move / call / alloc / field
                // cases above.  A bare `T` value initializer (e.g. a struct
                // literal coerced into this owning slot, which also reports a
                // Heap type) still falls through to be lifted into a fresh box.
                if matches!(&init.kind, HExprKind::Match { .. }) && matches!(&init.ty, HType::Heap { .. }) {
                    let s = self.emit_expr(f, init);
                    self.wl(&format!("{} {} = {};", ptr_ty, name, s));
                    return;
                }
                // ZeroInit owning-pointer slot (the synthetic `__yield` local of a
                // value-producing arm): a NULL pointer, NOT a malloc'd box.  The
                // arm's `yield` then rebinds it via drop-on-reassign (freeing the
                // NULL box is a null-guarded no-op).  Lifting it into a fresh box
                // would deref a `T*` ZeroInit into a `T` slot (type error / leak).
                if matches!(&init.kind, HExprKind::ZeroInit) {
                    self.wl(&format!("{} {} = NULL;", ptr_ty, name));
                    return;
                }
                // New allocation: value expression of type `T` lifted into heap slot.
                let value_s = self.emit_expr(f, init);
                let ic = self.c_type(inner);
                self.wl(&format!("{} {} = ({})malloc(sizeof({}));", ptr_ty, name, ptr_ty, ic));
                self.wl(&format!("*{} = {};", name, value_s));
            }
            HType::Array { .. } => {
                let value_s = self.emit_expr(f, init);
                // A C array is not `=`-initializable from another array VALUE, and
                // it must copy (value semantics - `[N]T b = a` is an independent
                // copy, like any array).  An array LITERAL brace-inits directly;
                // anything else (another array local/field/index) declares then
                // memcpy's the bytes.  (Reassignment already memcpy's, ~10438.)
                if matches!(&init.kind, HExprKind::ArrayLit(_)) {
                    self.wl(&format!("{} = {};", self.c_decl(&li.ty, &name), value_s));
                } else {
                    self.wl(&format!("{};", self.c_decl(&li.ty, &name)));
                    self.wl(&format!("memcpy({}, {}, sizeof({}));", name, value_s, name));
                }
            }
            _ => {
                let value_s = self.emit_expr(f, init);
                let prefix = if li.thread_local { "static __thread " } else { "" };
                self.wl(&format!("{}{} = {};", prefix, self.c_decl(&li.ty, &name), value_s));
                self.emit_move_out_null(f, init);
            }
        }
    }

    fn emit_assign(&mut self, f: &HFunc, op: HAssignOp, place: &HExpr, value: &HExpr, drop_old: bool) {
        let lhs = self.emit_place(f, place);
        let rhs = self.emit_expr(f, value);

        // Reassigning an `own &T` (Heap) local rebinds the owning pointer: the
        // RHS is itself an owning pointer (the place type is `Heap{T}`, so the
        // value coerces to it), so this falls through to the general
        // drop-on-reassign path below (free the old box, assign the new pointer,
        // null the moved-from source).  Treating it as an in-place pointee write
        // (`*n = rhs`) both mismatches types (`T` slot vs `T*` value) and leaks
        // the previous allocation.
        if let HExprKind::Local(id) = place.kind {
            let lty = f.locals[id.0 as usize].ty.clone();
            // Write through a `&mut T` local (e.g. a by-mut-ref captured binding inside a closure).
            if let HType::Ref { mutable: true, inner } = &lty {
                let deref = format!("*{}", lhs);
                self.wl(&format!("{} = {};", deref, self.compound_rhs(&deref, op, &rhs, inner)));
                return;
            }
        }
        // Pointer unwrap on LHS like `p! = v` → `*p = v;`
        if let HExprKind::Unwrap { expr, skip_check: _ } = &place.kind {
            let inner = self.emit_expr(f, expr);
            let deref = format!("*({})", inner);
            self.wl(&format!("{} = {};", deref, self.compound_rhs(&deref, op, &rhs, &place.ty)));
            return;
        }
        // Whole-array reassignment: a C array is not `=`-assignable, so copy.
        // This keeps the bare-array representation (and its pointer-decay perf);
        // only the explicit reassignment costs a memcpy.  An array-literal RHS
        // becomes a compound literal so it can be copied from.
        if matches!(op, HAssignOp::Assign) && matches!(&place.ty, HType::Array { .. }) {
            // If the old contents own heap, drop them first (unless the new value
            // is derived from the same array).
            if self.drop_ty_owns(&place.ty) {
                if let Some(root) = place_root_local(place) {
                    if !expr_contains_local(value, root) { self.emit_field_drop(&lhs, &place.ty, 0); }
                }
            }
            let src = if matches!(&value.kind, HExprKind::ArrayLit(_)) {
                format!("(__typeof__({})){}", lhs, rhs)
            } else {
                rhs
            };
            self.wl(&format!("memcpy({}, {}, sizeof({}));", lhs, src, lhs));
            return;
        }
        // Drop-on-reassign: free the previous owning value before overwriting it.
        // Three cases for a self-referential RHS (it mentions the place's root):
        //   - RHS is independent of the place        -> drop old, then assign.
        //   - RHS uses the place only via a borrow    -> the old value is still live
        //     (`s = s + x`, `+` takes `&self`): compute the new value into a temp,
        //     then drop the old value, then assign (free it exactly once).
        //   - RHS MOVES a sub-value out of the place by value (`node = node!.next`,
        //     `bs[i] = bs[j]`) -> aliasing-safe order: compute the RHS into a temp
        //     (moving the sub-value out and NULLing its slot), drop the old place
        //     (whose moved-out slot is now null, so its recursive drop does NOT
        //     cascade), then move the temp in.  This frees the old node exactly once
        //     = a correct free-as-you-go iterative consume; ownership decides the
        //     meaning (a `*T` walker never owns the nodes, so it takes the plain
        //     no-drop path above; an `own *T` cursor consumes as it advances).
        if drop_old && matches!(op, HAssignOp::Assign) && self.drop_ty_owns(&place.ty) {
            if let Some(root) = place_root_local(place) {
                if !expr_contains_local(value, root) {
                    self.emit_field_drop(&lhs, &place.ty, 0);
                } else if !local_used_by_value(value, root) && !place_is_projection(place) {
                    // RHS reads the place only through a borrow (`s = s + x`).
                    let ty = self.c_type(&place.ty);
                    self.wl(&format!("{{ {} __reassign = {};", ty, rhs));
                    self.emit_field_drop(&lhs, &place.ty, 0);
                    self.wl(&format!("{} = __reassign; }}", lhs));
                    self.emit_move_out_null(f, value);
                    return;
                } else {
                    // RHS moves a sub-value out of the place (projection alias like
                    // `bs[i] = bs[j]`, or a bare-local self-walk like `node = node!.next`).
                    let ty = self.c_type(&place.ty);
                    self.wl(&format!("{{ {} __new = {};", ty, rhs));
                    self.emit_move_out_null(f, value);
                    self.emit_field_drop(&lhs, &place.ty, 0);
                    self.wl(&format!("{} = __new; }}", lhs));
                    return;
                }
            }
        }
        self.wl(&format!("{} = {};", lhs, self.compound_rhs(&lhs, op, &rhs, &place.ty)));
        self.emit_move_out_null(f, value);
        // Moving an owning *local* into an owning slot (a field/index or another
        // owning local) transfers ownership: null the source so it is not freed
        // again - by a later drop-on-reassign or at scope exit.  Mirrors the
        // field move-out null above for `nx = cur.next`.  Without this,
        // `cur.next = prev; prev = cur;` double-frees the node prev pointed at.
        if matches!(op, HAssignOp::Assign) {
            if let HExprKind::Local(sid) = value.kind {
                if matches!(&value.ty, HType::OwnPtr { .. } | HType::Heap { .. })
                    && self.drop_ty_owns(&place.ty)
                    && place_root_local(place) != Some(sid.0)
                {
                    let base = self.local_name(sid, &f.locals[sid.0 as usize].name);
                    let sname = if self.aliased_locals.contains(&sid.0) { format!("(*{})", base) } else { base };
                    self.wl(&format!("{} = NULL;", sname));
                }
            }
        }
    }

    /// Emit an lvalue suitable for the LHS of an assignment.
    fn emit_place(&mut self, f: &HFunc, e: &HExpr) -> String {
        match &e.kind {
            HExprKind::Local(id) => {
                let n = self.local_name(*id, &f.locals[id.0 as usize].name);
                if self.aliased_locals.contains(&id.0) { format!("(*{})", n) } else { n }
            }
            HExprKind::GlobalRef(gid) => self.sym.globals[gid.0 as usize].c_name.clone(),
            HExprKind::Field { base, field } => {
                let base_s = self.emit_expr(f, base);
                let (arrow, fname) = self.field_access(f, base, *field);
                format!("{}{}{}", base_s, arrow, fname)
            }
            HExprKind::Index { base, idx } => {
                let base_s = self.emit_expr(f, base);
                let idx_s = self.emit_expr(f, idx);
                self.index_access(f, base, idx, &base_s, &idx_s)
            }
            HExprKind::Unwrap { expr, skip_check: _ } => {
                let inner = self.emit_expr(f, expr);
                // Pointer-to-fixed-array: the element pointer is the array base (no deref).
                if matches!(&expr.ty,
                    HType::Ptr { inner: i, .. } | HType::OwnPtr { inner: i, .. } | HType::RawPtr { inner: i, .. } | HType::Ref { inner: i, .. } | HType::Heap { inner: i }
                    if matches!(i.as_ref(), HType::Array { .. })) {
                    return inner;
                }
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

    /// Can the access `base[idx]` be proven in-bounds at compile time, so the
    /// runtime check is unnecessary?  Only fixed arrays (static length) qualify:
    /// a constant index in range, or a for-range loop counter whose guard keeps
    /// it within `[0, bound)` with `bound <= len`.  (Slices/vectors have runtime
    /// length, so they are never elided.)
    /// Is `base[idx]` into a fixed array of length `len` provably in-bounds?
    fn idx_in_const_bound(&self, len: i64, idx: &HExpr) -> bool {
        match &idx.kind {
            HExprKind::LitInt(n) => *n >= 0 && *n < len,
            HExprKind::Local(id) => self.bounded_vars.iter().any(|(lid, b)| *lid == id.0 && matches!(b, BceBound::Const(ub) if *ub <= len)),
            _ => false,
        }
    }

    /// Is indexing the slice/vec `base` with `idx` provably in-bounds?  Only when
    /// `idx` is a loop counter bounded by *this exact slice's* own `.len`.
    fn idx_in_slice_bound(&self, base: &HExpr, idx: &HExpr) -> bool {
        if let (HExprKind::Local(b), HExprKind::Local(i)) = (&base.kind, &idx.kind) {
            return self.bounded_vars.iter().any(|(lid, bound)|
                *lid == i.0 && matches!(bound, BceBound::SliceLen(s) if *s == b.0));
        }
        false
    }

    fn index_access(&self, _f: &HFunc, base: &HExpr, idx: &HExpr, base_s: &str, idx_s: &str) -> String {
        match &base.ty {
            HType::Array { len, .. } => {
                if self.idx_in_const_bound(*len, idx) {
                    format!("(({})[(maka_int)({})])", base_s, idx_s)
                } else {
                    format!("(({})[maka_check_idx((maka_int)({}), (maka_int){}, \"array idx\")])", base_s, idx_s, len)
                }
            }
            HType::Slice { .. } => {
                if self.idx_in_slice_bound(base, idx) {
                    format!("(({}).ptr[(maka_int)({})])", base_s, idx_s)
                } else {
                    format!("(({}).ptr[maka_check_idx((maka_int)({}), ({}).len, \"slice idx\")])", base_s, idx_s, base_s)
                }
            }
            HType::Vec { .. } => {
                if self.idx_in_slice_bound(base, idx) {
                    format!("(({}).data[(maka_int)({})])", base_s, idx_s)
                } else {
                    format!("(({}).data[maka_check_idx((maka_int)({}), ({}).len, \"vec idx\")])", base_s, idx_s, base_s)
                }
            }
            // A string is an array of chars; index its byte, bounds-checked
            // against its length (strlen, since the length is not stored).
            HType::Str => format!("(({0})[maka_check_idx((maka_int)({1}), (maka_int)strlen({0}), \"string idx\")])", base_s, idx_s),
            HType::Heap { inner } => match inner.as_ref() {
                // heap fixed array: base is already the element pointer (no deref).
                HType::Array { len, .. } => {
                    if self.idx_in_const_bound(*len, idx) {
                        format!("(({})[(maka_int)({})])", base_s, idx_s)
                    } else {
                        format!("(({})[maka_check_idx((maka_int)({}), (maka_int){}, \"array idx\")])", base_s, idx_s, len)
                    }
                }
                // heap [*]T: base is Vec_T (no deref)
                HType::Vec { .. } => format!("(({}).data[maka_check_idx((maka_int)({}), ({}).len, \"vec idx\")])", base_s, idx_s, base_s),
                _ => format!("(({})[{}])", base_s, idx_s),
            },
            // Borrow / non-owning pointer to a fixed array: base is the element
            // pointer; index with the array's static bound.
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } if matches!(inner.as_ref(), HType::Array { .. }) => {
                let len = if let HType::Array { len, .. } = inner.as_ref() { *len } else { 0 };
                if self.idx_in_const_bound(len, idx) {
                    format!("(({})[(maka_int)({})])", base_s, idx_s)
                } else {
                    format!("(({})[maka_check_idx((maka_int)({}), (maka_int){}, \"array idx\")])", base_s, idx_s, len)
                }
            }
            // Borrow / pointer to a Vec or Slice: deref to the struct, then index
            // its buffer.  `&Vec<T>` is `Vec_T*`, so the element lives at
            // `(*base).data[i]` (Vec) / `(*base).ptr[i]` (Slice).
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::OwnPtr { inner, .. }
                if matches!(inner.as_ref(), HType::Vec { .. }) => {
                format!("(((*({0})).data)[maka_check_idx((maka_int)({1}), (*({0})).len, \"vec idx\")])", base_s, idx_s)
            }
            HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::OwnPtr { inner, .. }
                if matches!(inner.as_ref(), HType::Slice { .. }) => {
                format!("(((*({0})).ptr)[maka_check_idx((maka_int)({1}), (*({0})).len, \"slice idx\")])", base_s, idx_s)
            }
            _ => format!("(({})[{}])", base_s, idx_s),
        }
    }

    /// Emit a built-in "sentinel" call: panic / log / format / push / pop /
    /// spawn / channel / atomic / etc.  Callee ids in [u32::MAX-1024, u32::MAX]
    /// are reserved for these (see `is_builtin_sentinel`) and every one is handled
    /// here.  ONE dispatch shared by the normal emitter and inline-body expansion
    /// (which sets `self.inline_tag` so `emit_sub` renders inline-tagged local
    /// names) - so a new builtin can never work in one path yet crash the other.
    fn emit_builtin_sentinel(&mut self, f: &HFunc, callee: FuncId, args: &[HExpr], e: &HExpr) -> String {
        // Built-in `fn_code(f)` / `fn_env(f)` — project a Callable value into its
        // env-first code pointer / env pointer for a C `(cb, void* ctx)` API.  The
        // closure lowers to `Callable_KEY { code, env }`, so this is a field read.
        if callee.0 == u32::MAX - 63 || callee.0 == u32::MAX - 64 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                let key = match &a.ty {
                    HType::FnPtr { ret, params } => fn_sig_key(ret, params),
                    _ => "unit_".to_string(),
                };
                let field = if callee.0 == u32::MAX - 63 { "__cb.code" } else { "(maka_unit*)__cb.env" };
                return format!("(__extension__ ({{ Callable_{k} __cb = ({s}); {fld}; }}))",
                    k = key, s = s, fld = field);
            }
            return "NULL".into();
        }
        // Built-in float bit-reinterprets (memcpy, no strict-aliasing UB): exact
        // IEEE-754 bits in/out for serialization.
        if (u32::MAX - 68..=u32::MAX - 65).contains(&callee.0) {
            let s = args.first().map(|a| self.emit_sub(f, a)).unwrap_or_else(|| "0".into());
            let body = match callee.0 {
                x if x == u32::MAX - 65 => // f32_bits(f32) -> int
                    format!("float __f = (float)({s}); uint32_t __u; memcpy(&__u, &__f, 4); (maka_int)__u;", s = s),
                x if x == u32::MAX - 66 => // bits_f32(int) -> f32
                    format!("uint32_t __u = (uint32_t)({s}); float __f; memcpy(&__f, &__u, 4); __f;", s = s),
                x if x == u32::MAX - 67 => // f64_bits(float) -> int
                    format!("double __d = ({s}); int64_t __i; memcpy(&__i, &__d, 8); (maka_int)__i;", s = s),
                _ => // bits_f64(int) -> float
                    format!("int64_t __i = ({s}); double __d; memcpy(&__d, &__i, 8); __d;", s = s),
            };
            return format!("(__extension__ ({{ {} }}))", body);
        }
        // Built-in `panic(msg)`.
        if callee.0 == u32::MAX - 2 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("(maka_panic({}), MAKA_UNIT)", s);
            }
            return "(maka_panic(\"\"), MAKA_UNIT)".into();
        }
        // Built-in `free`
        if callee.0 == u32::MAX - 1 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("(free((void*)({})), MAKA_UNIT)", s);
            }
            return "MAKA_UNIT".into();
        }
        // Built-in `log`
        if callee.0 == u32::MAX {
            // Dispatch by first arg type
            if args.len() == 1 {
                let s = self.emit_sub(f, &args[0]);
                // Auto-deref a NON-NULL primitive reference (`&T` / `own &T`) so `log`
                // prints the value, not the address (SPEC 14: "auto-derefs primitive
                // refs"). Restricted to scalar pointees; nullable pointers
                // (`*T` / `own *T` / `raw *T`) keep printing the pointer (may be null).
                if let HType::Ref { inner, .. } | HType::Heap { inner } = &args[0].ty {
                    if matches!(inner.as_ref(),
                        HType::Int | HType::Enum(_) | HType::SizedInt { .. }
                        | HType::Float | HType::SizedFloat { .. } | HType::Bool | HType::Char)
                    {
                        return format!("({}(*({})), MAKA_UNIT)", self.log_helper(inner), s);
                    }
                }
                let helper = self.log_helper(&args[0].ty);
                return format!("({}({}), MAKA_UNIT)", helper, s);
            }
            return "MAKA_UNIT".into();
        }
        // Optimized `log(format(...))` -> printf, no allocation.  args[0]
        // is the printf format string; the rest are scalar values.
        if callee.0 == u32::MAX - 58 {
            let mut parts: Vec<String> = vec![self.emit_sub(f, &args[0])];
            for a in &args[1..] {
                let s = self.emit_sub(f, a);
                parts.push(printf_conv(s, &a.ty));
            }
            return format!("(printf({}), MAKA_UNIT)", parts.join(", "));
        }
        // `format(...)` with scalar placeholders -> one __maka_format1 alloc.
        if callee.0 == u32::MAX - 59 {
            let mut parts: Vec<String> = vec![self.emit_sub(f, &args[0])];
            for a in &args[1..] {
                let s = self.emit_sub(f, a);
                parts.push(printf_conv(s, &a.ty));
            }
            return format!("__maka_format1({})", parts.join(", "));
        }
        // `push(v, x)` - append, growing the buffer (realloc) on demand.
        // args[0] is `&mut Vec_T`; element size is taken from the buffer.
        if callee.0 == u32::MAX - 60 {
            let vp_ty = self.c_type(&args[0].ty);   // Vec_T*
            let vp = self.emit_sub(f, &args[0]);
            // The element is moved into the Vec: null an owning field/index
            // source so the caller's container drop does not double-free it.
            let x = self.emit_move_consuming(f, &args[1]);
            return format!("(__extension__ ({{ {0} __vp = {1}; if (__vp->len == __vp->cap) {{ __vp->cap = __vp->cap ? __vp->cap * 2 : 4; __vp->data = realloc(__vp->data, __vp->cap * sizeof(*__vp->data)); }} __vp->data[__vp->len++] = ({2}); MAKA_UNIT; }}))", vp_ty, vp, x);
        }
        // `pop(v)` -> last element, shrinking the length (panics if empty).
        if callee.0 == u32::MAX - 61 {
            let vp_ty = self.c_type(&args[0].ty);
            let vp = self.emit_sub(f, &args[0]);
            return format!("(__extension__ ({{ {0} __vp = {1}; if (__vp->len == 0) {{ maka_panic(\"pop from empty Vec\"); }} __vp->len--; __vp->data[__vp->len]; }}))", vp_ty, vp);
        }
        // `zeroed()` -> a zero-initialized value of the call's type.
        if callee.0 == u32::MAX - 62 {
            return format!("(({}){{0}})", self.c_type(&e.ty));
        }
        // Built-in `spawn(closure)` — fiber tier.  Compound-stmt expr
        // wraps the closure once so its env malloc happens exactly once
        // (emitting `(s).code, (s).env` would expand and re-allocate).
        if callee.0 == u32::MAX - 3 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_spawn_fiber(__cb.code, __cb.env, {}); }}))", s, self.spawn_env_drop(a));
            }
            return "NULL".into();
        }
        // Built-in `thread(closure)` — kernel thread tier.
        if callee.0 == u32::MAX - 15 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_spawn_thread(__cb.code, __cb.env, {}); }}))", s, self.spawn_env_drop(a));
            }
            return "NULL".into();
        }
        // Built-in `job(closure)` — work-pool tier.
        if callee.0 == u32::MAX - 16 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_spawn_job(__cb.code, __cb.env, {}); }}))", s, self.spawn_env_drop(a));
            }
            return "NULL".into();
        }
        // Built-in `spawn_pool(closure)` — fiber on background pool.
        if callee.0 == u32::MAX - 37 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_spawn_pool(__cb.code, __cb.env, {}); }}))", s, self.spawn_env_drop(a));
            }
            return "NULL".into();
        }
        // Built-in `spawn_on(closure, pool)` — fiber on an explicit Pool.  Emitted
        // closure-first (args[0]=closure, args[1]=pool) so the lifetime scans match.
        if callee.0 == u32::MAX - 70 {
            if args.len() == 2 {
                let s = self.emit_sub(f, &args[0]);
                let pool = self.emit_sub(f, &args[1]);
                return format!("(__extension__ ({{ Callable_unit_ __cb = ({}); (Thread*)__maka_pool_spawn_on((maka_unit*)({}), __cb.code, __cb.env, {}); }}))", s, pool, self.spawn_env_drop(&args[0]));
            }
            return "NULL".into();
        }
        // Built-in `join(slice_of_handles)` — wait for all handles.
        // Codegen extracts the slice's ptr+len and calls the runtime.
        if callee.0 == u32::MAX - 17 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
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
                let s = self.emit_sub(f, a);
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
                let ip = self.emit_sub(f, &args[0]);
                let n  = self.emit_sub(f, &args[1]);
                let isz = self.emit_sub(f, &args[2]);
                let osz = self.emit_sub(f, &args[3]);
                let body = self.emit_sub(f, &args[4]);
                return format!(
                    "(__extension__ ({{ Callable_unit_Pmunit_Pmunit_ __cb = ({4}); \
                     (maka_unit*)maka_par_map_bytes((void*)({0}), (int64_t)({1}), (int64_t)({2}), (int64_t)({3}), (void*)__cb.code, (void*)__cb.env); }}))",
                    ip, n, isz, osz, body
                );
            }
            return "NULL".into();
        }
        // file_listdir(path) -> Slice_str
        if callee.0 == u32::MAX - 43 {
            if let Some(a) = args.first() {
                let p = self.emit_sub(f, a);
                return format!(
                    "(__extension__ ({{ int64_t __n; const char** __p = __maka_rt_file_listdir(({0}), &__n); (Slice_str){{ .ptr = __p, .len = (maka_int)__n }}; }}))",
                    p
                );
            }
            return "(Slice_str){0}".into();
        }
        // str_split(s, sep) -> Slice_str
        if callee.0 == u32::MAX - 44 {
            if args.len() == 2 {
                let s = self.emit_sub(f, &args[0]);
                let sep = self.emit_sub(f, &args[1]);
                return format!(
                    "(__extension__ ({{ int64_t __n; const char** __p = __maka_rt_str_split(({0}), ({1}), &__n); (Slice_str){{ .ptr = __p, .len = (maka_int)__n }}; }}))",
                    s, sep
                );
            }
            return "(Slice_str){0}".into();
        }
        // ===== Concurrency primitives (irreducible base) =====
        // atomic_cas(&mut T p, T expected, T new) -> T (returns old).
        // __atomic_compare_exchange_n updates *expected to *p on
        // failure; either way `__exp` ends up holding the OLD value.
        if callee.0 == u32::MAX - 45 {
            if args.len() == 3 {
                let p = self.emit_sub(f, &args[0]);
                let exp = self.emit_sub(f, &args[1]);
                let new = self.emit_sub(f, &args[2]);
                let ty = self.c_type(&args[2].ty);
                return format!(
                    "(__extension__ ({{ {ty} __exp = ({exp}); \
                     __atomic_compare_exchange_n({p}, &__exp, ({new}), 0, \
                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); __exp; }}))",
                    ty = ty, p = p, exp = exp, new = new
                );
            }
            return "0".into();
        }
        // atomic_load(&const T p) -> T
        if callee.0 == u32::MAX - 46 {
            if let Some(a) = args.first() {
                let p = self.emit_sub(f, a);
                return format!("__atomic_load_n(({}), __ATOMIC_SEQ_CST)", p);
            }
            return "0".into();
        }
        // atomic_store(&mut T p, T v)
        if callee.0 == u32::MAX - 47 {
            if args.len() == 2 {
                let p = self.emit_sub(f, &args[0]);
                let v = self.emit_sub(f, &args[1]);
                return format!(
                    "(__atomic_store_n(({}), ({}), __ATOMIC_SEQ_CST), MAKA_UNIT)",
                    p, v
                );
            }
            return "MAKA_UNIT".into();
        }
        // atomic_fetch_add / sub / and / or / xor — all the same shape.
        if let Some(c_op) = match callee.0 {
            v if v == u32::MAX - 48 => Some("__atomic_fetch_add"),
            v if v == u32::MAX - 49 => Some("__atomic_fetch_sub"),
            v if v == u32::MAX - 50 => Some("__atomic_fetch_and"),
            v if v == u32::MAX - 51 => Some("__atomic_fetch_or"),
            v if v == u32::MAX - 52 => Some("__atomic_fetch_xor"),
            _ => None,
        } {
            if args.len() == 2 {
                let p = self.emit_sub(f, &args[0]);
                let v = self.emit_sub(f, &args[1]);
                return format!("{}(({}), ({}), __ATOMIC_SEQ_CST)", c_op, p, v);
            }
            return "0".into();
        }
        // atomic_fence(int order)
        if callee.0 == u32::MAX - 53 {
            if let Some(a) = args.first() {
                let o = self.emit_sub(f, a);
                // Map Maka order to __ATOMIC_* via a small dispatch.
                return format!(
                    "(__atomic_thread_fence((({}) == 1) ? __ATOMIC_ACQUIRE : \
                                            (({}) == 2) ? __ATOMIC_RELEASE : \
                                            (({}) == 3) ? __ATOMIC_ACQ_REL : \
                                                          __ATOMIC_SEQ_CST), MAKA_UNIT)",
                    o, o, o
                );
            }
            return "MAKA_UNIT".into();
        }
        // futex_wait(&const int addr, int expected) -> int
        if callee.0 == u32::MAX - 54 {
            if args.len() == 2 {
                let p = self.emit_sub(f, &args[0]);
                let v = self.emit_sub(f, &args[1]);
                return format!("(int64_t)__maka_futex_wait((const int*)({}), (int)({}))", p, v);
            }
            return "0".into();
        }
        // futex_wake(&const int addr, int n) -> int
        if callee.0 == u32::MAX - 55 {
            if args.len() == 2 {
                let p = self.emit_sub(f, &args[0]);
                let n = self.emit_sub(f, &args[1]);
                return format!("(int64_t)__maka_futex_wake((const int*)({}), (int)({}))", p, n);
            }
            return "0".into();
        }
        // thread_yield()
        if callee.0 == u32::MAX - 56 {
            return "(__maka_thread_yield(), MAKA_UNIT)".into();
        }
        // syscall(n, a1..a6) -> int — variadic, missing args = 0.
        if callee.0 == u32::MAX - 57 {
            let mut parts: Vec<String> = (0..7).map(|i| {
                if let Some(a) = args.get(i) {
                    format!("(long)({})", self.emit_sub(f, a))
                } else {
                    "0L".to_string()
                }
            }).collect();
            let n = parts.remove(0);
            return format!(
                "(int64_t)__maka_syscall({}, {})",
                n, parts.join(", ")
            );
        }
        // par_filter_bytes(in, n, item_sz, &mut out_n, pred)
        if callee.0 == u32::MAX - 41 {
            if args.len() == 5 {
                let ip = self.emit_sub(f, &args[0]);
                let n  = self.emit_sub(f, &args[1]);
                let isz = self.emit_sub(f, &args[2]);
                let outn = self.emit_sub(f, &args[3]);
                let body = self.emit_sub(f, &args[4]);
                return format!(
                    "(__extension__ ({{ Callable_bool_Pmunit_ __cb = ({4}); \
                     (maka_unit*)maka_par_filter_bytes((void*)({0}), (int64_t)({1}), (int64_t)({2}), (int64_t*)({3}), (void*)__cb.code, (void*)__cb.env); }}))",
                    ip, n, isz, outn, body
                );
            }
            return "NULL".into();
        }
        // par_scan_bytes(in, n, item_sz, combine)
        if callee.0 == u32::MAX - 42 {
            if args.len() == 4 {
                let ip = self.emit_sub(f, &args[0]);
                let n  = self.emit_sub(f, &args[1]);
                let isz = self.emit_sub(f, &args[2]);
                let body = self.emit_sub(f, &args[3]);
                return format!(
                    "(__extension__ ({{ Callable_unit_Pmunit_Pmunit_Pmunit_ __cb = ({3}); \
                     (maka_unit*)maka_par_scan_bytes((void*)({0}), (int64_t)({1}), (int64_t)({2}), (void*)__cb.code, (void*)__cb.env); }}))",
                    ip, n, isz, body
                );
            }
            return "NULL".into();
        }
        // Built-in `detach(*Thread)`.
        if callee.0 == u32::MAX - 33 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("(__maka_detach((maka_unit*)({})), MAKA_UNIT)", s);
            }
            return "MAKA_UNIT".into();
        }
        // Built-in `cancel(*Thread)`.
        if callee.0 == u32::MAX - 23 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("(__maka_cancel((maka_unit*)({})), MAKA_UNIT)", s);
            }
            return "MAKA_UNIT".into();
        }
        // Built-in `try_join(*Thread) -> bool`.
        if callee.0 == u32::MAX - 24 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
                return format!("((bool)(__maka_try_join((maka_unit*)({}), NULL) != 0))", s);
            }
            return "0".into();
        }
        // Built-in `join_timeout(*Thread, int) -> bool`.
        if callee.0 == u32::MAX - 25 {
            if args.len() == 2 {
                let h = self.emit_sub(f, &args[0]);
                let ms = self.emit_sub(f, &args[1]);
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
                let o = self.emit_sub(f, &args[0]);
                let cb = self.emit_sub(f, &args[1]);
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
                let s = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
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
                let s = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                return format!(
                    "(__extension__ ({{ Callable_float_float_ __cb = ({1}); Slice_maka_float __s = __maka_par_map_float(({0}), __cb.code, __cb.env); (Vec_maka_float){{ .data = __s.ptr, .len = __s.len, .cap = __s.len }}; }}))",
                    s, b
                );
            }
            return "((Vec_maka_float){ .data = NULL, .len = 0, .cap = 0 })".into();
        }
        // par_reduce_float(slice, init, combine)
        if callee.0 == u32::MAX - 36 {
            if args.len() == 3 {
                let s = self.emit_sub(f, &args[0]);
                let init = self.emit_sub(f, &args[1]);
                let b = self.emit_sub(f, &args[2]);
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
                let s = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
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
                let s = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                return format!(
                    "(__extension__ ({{ Callable_int_int_ __cb = ({1}); Slice_maka_int __s = __maka_par_map_int_slice(({0}), __cb.code, __cb.env); (Vec_maka_int){{ .data = __s.ptr, .len = __s.len, .cap = __s.len }}; }}))",
                    s, b
                );
            }
            return "((Vec_maka_int){ .data = NULL, .len = 0, .cap = 0 })".into();
        }
        // par_reduce_int(slice, init, combine)
        if callee.0 == u32::MAX - 29 {
            if args.len() == 3 {
                let s = self.emit_sub(f, &args[0]);
                let init = self.emit_sub(f, &args[1]);
                let b = self.emit_sub(f, &args[2]);
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
                let s = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                return format!(
                    "(__extension__ ({{ Callable_bool_int_ __cb = ({1}); Slice_maka_int __s = __maka_par_filter_int(({0}), __cb.code, __cb.env); (Vec_maka_int){{ .data = __s.ptr, .len = __s.len, .cap = __s.len }}; }}))",
                    s, b
                );
            }
            return "((Vec_maka_int){ .data = NULL, .len = 0, .cap = 0 })".into();
        }
        // par_filter_float(slice, pred)
        if callee.0 == u32::MAX - 39 {
            if args.len() == 2 {
                let s = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                return format!(
                    "(__extension__ ({{ Callable_bool_float_ __cb = ({1}); __maka_par_filter_float(({0}), __cb.code, __cb.env); }}))",
                    s, b
                );
            }
            return "(Slice_maka_float){0}".into();
        }
        // par_scan_float(slice, combine)
        if callee.0 == u32::MAX - 40 {
            if args.len() == 2 {
                let s = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                return format!(
                    "(__extension__ ({{ Callable_float_float_float_ __cb = ({1}); __maka_par_scan_float(({0}), __cb.code, __cb.env); }}))",
                    s, b
                );
            }
            return "(Slice_maka_float){0}".into();
        }
        // par_scan_int(slice, combine)
        if callee.0 == u32::MAX - 31 {
            if args.len() == 2 {
                let s = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                return format!(
                    "(__extension__ ({{ Callable_int_int_int_ __cb = ({1}); Slice_maka_int __s = __maka_par_scan_int(({0}), __cb.code, __cb.env); (Vec_maka_int){{ .data = __s.ptr, .len = __s.len, .cap = __s.len }}; }}))",
                    s, b
                );
            }
            return "((Vec_maka_int){ .data = NULL, .len = 0, .cap = 0 })".into();
        }
        // Built-in `select_timeout(slice, int) -> int`.
        if callee.0 == u32::MAX - 26 {
            if args.len() == 2 {
                let s = self.emit_sub(f, &args[0]);
                let ms = self.emit_sub(f, &args[1]);
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
                let a = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                let body = self.emit_sub(f, &args[2]);
                return format!(
                    "(__extension__ ({{ Callable_int_int_ __cb = ({2}); Slice_maka_int __s = __maka_par_map_int((int64_t)({0}), (int64_t)({1}), __cb.code, __cb.env); (Vec_maka_int){{ .data = __s.ptr, .len = __s.len, .cap = __s.len }}; }}))",
                    a, b, body
                );
            }
            return "((Vec_maka_int){ .data = NULL, .len = 0, .cap = 0 })".into();
        }
        // Built-in `par_reduce_int(start, end, init, combine)`.
        if callee.0 == u32::MAX - 21 {
            if args.len() == 4 {
                let a = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                let init = self.emit_sub(f, &args[2]);
                let body = self.emit_sub(f, &args[3]);
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
                let a = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
                let body = self.emit_sub(f, &args[2]);
                // par_for_range runs the closure synchronously, so a fresh
                // capturing closure's env can be freed right after (its
                // malloc'd env would otherwise leak).  Only free when the
                // arg is a closure literal - never a borrowed fn-ptr value.
                let free_env = if matches!(&args[2].kind, HExprKind::Closure { .. }) { " free(__cb.env);" } else { "" };
                return format!(
                    "(__extension__ ({{ Callable_unit_int_ __cb = ({2}); __maka_par_for_range((int64_t)({0}), (int64_t)({1}), __cb.code, __cb.env);{3} MAKA_UNIT; }}))",
                    a, b, body, free_env
                );
            }
            return "MAKA_UNIT".into();
        }
        // Built-in `join(*Thread)` — pthread join + free handle.
        if callee.0 == u32::MAX - 4 {
            if let Some(a) = args.first() {
                let s = self.emit_sub(f, a);
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
                let a = self.emit_sub(f, &args[0]);
                let b = self.emit_sub(f, &args[1]);
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
                let a = self.emit_sub(f, &args[0]);
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
        unreachable!("unhandled builtin sentinel: u32::MAX - {}", u32::MAX - callee.0)
    }

    fn emit_expr(&mut self, f: &HFunc, e: &HExpr) -> String {
        match &e.kind {
            HExprKind::LitInt(n) => c_int_lit(*n),
            // `{:?}` (not `{}`): Rust's Display renders e.g. 1e30 as the
            // integer-looking "1000...000", which C parses as an integer literal
            // that overflows; Debug always emits a `.`/`e` form, round-trip exact
            // and a valid C double literal.
            HExprKind::LitFloat(v) => format!("(maka_float){:?}", v),
            HExprKind::LitBool(b) => if *b { "true".into() } else { "false".into() },
            HExprKind::LitChar(c) => format!("(maka_char){}u", *c as u32),
            HExprKind::LitStr(s) => format!("\"{}\"", c_escape(s)),
            HExprKind::LitNull => "NULL".into(),
            HExprKind::LitUnit => "MAKA_UNIT".into(),
            HExprKind::Local(id) => {
                let n = self.local_name(*id, &f.locals[id.0 as usize].name);
                if self.aliased_locals.contains(&id.0) { format!("(*{})", n) } else { n }
            }
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
            HExprKind::InlineCall { callee, args, propagate_drops, loop_jump_drops } => {
                let pd: Vec<(String, HType)> = propagate_drops.iter().map(|id| {
                    let li = &f.locals[id.0 as usize];
                    (self.local_name(*id, &li.name), li.ty.clone())
                }).collect();
                let ld: Vec<(String, HType)> = loop_jump_drops.iter().map(|id| {
                    let li = &f.locals[id.0 as usize];
                    (self.local_name(*id, &li.name), li.ty.clone())
                }).collect();
                self.emit_inline_expansion(f, *callee, args, &e.ty, pd, ld)
            }
            HExprKind::Transfer(inner) => self.emit_expr(f, inner),
            HExprKind::EnumTag(inner) => {
                let s = self.emit_expr(f, inner);
                self.enum_tag_c(&inner.ty, &s)
            }
            HExprKind::SliceLen(inner) => {
                let s = self.emit_expr(f, inner);
                self.slice_len_c(&inner.ty, &s)
            }
            HExprKind::Closure { lifted, env_struct, env_values, .. } => {
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
                    else if sig.name == "main" && sig.trait_name.is_none() { "maka_main".to_string() }
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
                if needs_div_guard(*op, &lhs.ty, rhs) {
                    return self.emit_div(&l, *op, &r, &lhs.ty);
                }
                if matches!(op, HBinOp::Shl | HBinOp::Shr) {
                    return self.emit_shift(&l, *op, &r, &lhs.ty);
                }
                self.emit_arith(&l, *op, &r, &lhs.ty)
            }
            HExprKind::Un { op, expr } => {
                // Fold negation of an integer literal so `-9223372036854775808`
                // emits i64::MIN directly rather than `-(i64::MIN)`, which is a C
                // overflow.  wrapping_neg keeps every other value identical.
                if let (HUnOp::Neg, HExprKind::LitInt(n)) = (op, &expr.kind) {
                    return c_int_lit(n.wrapping_neg());
                }
                let v = self.emit_expr(f, expr);
                match op {
                    HUnOp::Neg => self.emit_neg(&v, &expr.ty),
                    HUnOp::Not => format!("(!({}))", v),
                    // C integer-promotes `~`, so cast back to the operand's type
                    // to mask the result into its width (e.g. `~(u8)0` is 255, not -1).
                    HUnOp::BitNot => format!("(({})(~({})))", self.c_type(&expr.ty), v),
                }
            }
            HExprKind::Unwrap { expr, skip_check: _ } => {
                let s = self.emit_expr(f, expr);
                // `p!` on a pointer-to-fixed-array or pointer-to-string yields the base,
                // which the element-pointer / char* representation already holds - no deref.
                if Self::ptr_pointee_decays(&expr.ty) {
                    return s;
                }
                format!("(*({}))", s)
            }
            HExprKind::AddrOfRef { place, .. } => {
                // For dyn fat-pointers, `&m` is just `m` (the fat pointer already encapsulates indirection).
                if matches!(place.ty, HType::Dyn { .. }) {
                    return self.emit_place(f, place);
                }
                let p = self.emit_place(f, place);
                // Address of a fixed array or a string decays to an element pointer in C -
                // `&[N]T` and `&string` are both represented as a single element pointer.
                if matches!(&place.ty, HType::Array { .. } | HType::Str) {
                    return format!("({})", p);
                }
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
                self.index_access(f, base, idx, &bs, &is_)
            }
            HExprKind::Call { callee, args } => {
                if is_builtin_sentinel(*callee) {
                    return self.emit_builtin_sentinel(f, *callee, args, e);
                }
                let sig = self.sym.func_sig(*callee);

                // Dynamic-dispatch: a TRAIT-METHOD call whose receiver is a dyn (or
                // a reference to one) goes through the vtable.  A free function that
                // merely takes a dyn first argument (sig.trait_name is None) is a normal
                // call - mirrors the method-first/free-fn-fallback resolution in sema.
                if !args.is_empty() && sig.trait_name.is_some() {
                    if let Some(_traits) = strip_to_dyn(&args[0].ty) {
                        // We want the underlying Dyn value, not its address. If the expression is
                        // `&mut x`/AddrOfRef of a dyn place, skip the `&`.
                        let recv_s = match &args[0].kind {
                            HExprKind::AddrOfRef { place, .. } => self.emit_place(f, place),
                            _ => self.emit_expr(f, &args[0]),
                        };
                        let mut rest: Vec<String> = Vec::new();
                        for a in &args[1..] { rest.push(self.emit_move_consuming(f, a)); }
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
                } else if sig.name == "main" && sig.trait_name.is_none() {
                    "maka_main".to_string()
                } else { c_ident(&sig.c_name) };
                // Owning field/element args are moved by value: emit_move_consuming
                // captures the value and nulls the caller's slot so the caller's
                // later container drop does not double-free it.
                let arg_s: Vec<String> = args.iter().enumerate().map(|(i, a)| {
                    // Passing a Maka function to an `extern` fn-pointer parameter (a
                    // bare C callback): emit the function's DIRECT C address, not the
                    // fat env-first `Callable` a Maka closure value is - the direct
                    // function `R f(P...)` already matches the C `R (*)(P...)`.
                    if sig.is_extern {
                        if let Some(pt @ HType::FnPtr { .. }) = sig.param_tys.get(i) {
                            if let HExprKind::FnRef(fid) = &a.kind {
                                let fs = self.sym.func_sig(*fid);
                                return if fs.is_extern { fs.c_name.clone() }
                                    else if fs.name == "main" && fs.trait_name.is_none() { "maka_main".to_string() }
                                    else { c_ident(&fs.c_name) };
                            }
                            // `fn_code(f)` yields a bare env-first code pointer; cast
                            // it to the param's exact callback type so a `void*` vs
                            // `maka_unit*` env-slot naming difference can't clash.
                            if let HExprKind::Call { callee: cc, .. } = &a.kind {
                                if cc.0 == u32::MAX - 63 {
                                    let cast = self.c_extern_param_decl(pt, "");
                                    return format!("({}){}", cast, self.emit_sub(f, a));
                                }
                            }
                        }
                    }
                    self.emit_move_consuming(f, a)
                }).collect();
                // A capturing closure LITERAL passed to a consume-only FnPtr param
                // is a one-shot temp the callee only calls (never stores), so its
                // malloc'd env can be freed right after the call returns.
                let consume = self.sym.funcs.iter().find(|hf| hf.id == *callee)
                    .map(consume_only_fnptr_params).unwrap_or_default();
                let frees: Vec<usize> = args.iter().enumerate().filter(|(i, a)|
                    consume.get(*i).copied().unwrap_or(false)
                    && matches!(&a.kind, HExprKind::Closure { env_values, .. } if !env_values.is_empty())
                ).map(|(i, _)| i).collect();
                if frees.is_empty() {
                    let call = format!("{}({})", name, arg_s.join(", "));
                    if matches!(sig.ret, HType::Unit) { format!("({}, MAKA_UNIT)", call) } else { call }
                } else {
                    let mut decls = String::new();
                    let mut free_s = String::new();
                    let cargs: Vec<String> = arg_s.iter().enumerate().map(|(i, s)| {
                        if frees.contains(&i) {
                            let tn = format!("__cca{}", i);
                            decls.push_str(&format!("{} {} = {}; ", self.c_type(&args[i].ty), tn, s));
                            free_s.push_str(&format!("free({}.env); ", tn));
                            tn
                        } else { s.clone() }
                    }).collect();
                    let call = format!("{}({})", name, cargs.join(", "));
                    if matches!(sig.ret, HType::Unit) {
                        format!("(__extension__ ({{ {}{}; {}MAKA_UNIT; }}))", decls, call, free_s)
                    } else {
                        format!("(__extension__ ({{ {}{} __r = {}; {}__r; }}))", decls, self.c_type(&sig.ret), call, free_s)
                    }
                }
            }
            HExprKind::Cast { expr, kind, to } => {
                // For ToDyn casts we need the source's address (the data pointer).
                if let CastKind::ToDyn { traits, struct_id } = kind {
                    return self.emit_to_dyn(f, expr, traits, *struct_id);
                }
                let s = self.emit_expr(f, expr);
                self.emit_cast(s, kind.clone(), to, &expr.ty)
            }
            HExprKind::CheckedCast { expr, kind, to } => {
                // Dead path: the parser no longer constructs CheckedCast.
                // Fall back to the regular cast emitter; new int→Enum and
                // *int→*Enum live in `emit_cast` (see CastKind cases).
                let s = self.emit_expr(f, expr);
                self.emit_cast(s, kind.clone(), to, &expr.ty)
            }
            HExprKind::Struct { id, fields } => {
                let info = self.sym.struct_info(*id);
                let parts: Vec<String> = info.fields.iter().enumerate().map(|(i, f0)| {
                    if let Some((_, fe)) = fields.iter().find(|(j, _)| *j == i) {
                        let s = self.emit_move_consuming(f, fe);
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
                let parts: Vec<String> = elems.iter().map(|e| self.emit_move_consuming(f, e)).collect();
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
                            // Write elements straight into the malloc'd buffer - no
                            // intermediate stack array + memcpy (gcc can't fuse that,
                            // since it can't prove __d doesn't alias the temp).
                            let stores: String = parts.iter().enumerate()
                                .map(|(i, p)| format!("__d[{}] = {}; ", i, p)).collect();
                            format!("(__extension__ ({{ {0}* __d = ({0}*)malloc(sizeof({0})*{1}); {2}(Vec_{3}){{ .data = __d, .len = {1}, .cap = {1} }}; }}))", elem_c, n, stores, key)
                        }
                    }
                    _ => format!("{{ {} }}", parts.join(", ")),
                }
            }
            HExprKind::DropWrite(inner) => self.emit_expr(f, inner),
            HExprKind::DerefRef(inner) => {
                let s = self.emit_expr(f, inner);
                // A pointer-to-array / pointer-to-string is already the element pointer
                // (char* for strings); dereferencing it is a no-op at the C level.
                if Self::ptr_pointee_decays(&inner.ty) {
                    return s;
                }
                format!("(*({}))", s)
            }
            HExprKind::HeapAlloc(inner) => {
                // `alloc [N]T` -> a heap fixed array (`own *[N]T`): malloc N elements
                // and fill them.  C forbids assigning to a whole array (`*__p = {...}`),
                // so write element-by-element into an element-typed buffer.
                if let HType::Array { len, elem } = &inner.ty {
                    let elem_c = self.c_type(elem);
                    if let HExprKind::ArrayLit(elems) = &inner.kind {
                        let stores: String = elems.iter().enumerate()
                            .map(|(i, el)| { let s = self.emit_move_consuming(f, el); format!("__d[{}] = {}; ", i, s) }).collect();
                        return format!("(__extension__ ({{ {0}* __d = ({0}*)malloc(sizeof({0})*{1}); {2}__d; }}))", elem_c, len, stores);
                    }
                    // Non-literal array value: stage in a temp, then copy element-wise.
                    let v = self.emit_move_consuming(f, inner);
                    return format!("(__extension__ ({{ {0} __s[{1}] = {2}; {0}* __d = ({0}*)malloc(sizeof({0})*{1}); for (size_t __i=0;__i<(size_t){1};__i++) __d[__i]=__s[__i]; __d; }}))", elem_c, len, v);
                }
                let inner_c = self.c_type(&inner.ty);
                // The value is moved into the heap box: null an owning field/index
                // source so the caller's container drop does not double-free it.
                let v = self.emit_move_consuming(f, inner);
                format!("(__extension__ ({{ {0}* __p = ({0}*)malloc(sizeof({0})); *__p = ({1}); __p; }}))", inner_c, v)
            }
            HExprKind::Free(inner, deep) => {
                let s = self.emit_expr(f, inner);
                self.emit_free(s, &inner.ty, *deep)
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
                        .map(|(_, e)| self.emit_move_consuming(f, e))
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
            // Typed all-zero value: NULL for any pointer-shaped type, an all-zero
            // compound literal (null buffers) for an owning composite or scalar.
            HExprKind::ZeroInit => {
                let ct = self.c_type(&e.ty);
                match &e.ty {
                    HType::Ptr { .. } | HType::RawPtr { .. } | HType::OwnPtr { .. } | HType::Heap { .. } =>
                        format!("(({})0)", ct),
                    _ => format!("(({}){{0}})", ct),
                }
            }
            // Runtime existential re-witness: `{ v.data, <switch v.vtbl -> to-vtbl or NULL> }`.
            // The switch compares the source vtbl against each `src_vtbl_for_T` (T packed
            // as `dyn src_trait` that implements all of `to_traits`) and yields the matching
            // `to_vtbl_for_T`; anything else yields NULL (the membership-fails case).  For a
            // multi-trait `to_traits` (narrowing keeps the source trait too) the target is a
            // combined `Dyn_Y_X` whose vtbl unions both interfaces - so both dispatch directly.
            HExprKind::DynRewitness { value, src_trait, to_traits } => {
                let vc = self.emit_expr(f, value);
                let src_c = c_ident(src_trait);
                let to_key = c_ident(&to_traits.join("_"));
                let cands = self.dyn_witness_candidates(src_trait, to_traits);
                let mut sw = String::new();
                for t in &cands {
                    let tc = c_ident(t);
                    sw.push_str(&format!("(__s.vtbl == &{0}_vtbl_for_{2}) ? &{1}_vtbl_for_{2} : ", src_c, to_key, tc));
                }
                sw.push_str(&format!("(const {}_vtbl*)0", to_key));
                format!("(__extension__ ({{ Dyn_{0} __s = ({1}); (Dyn_{2}){{ .data = __s.data, .vtbl = ({3}) }}; }}))",
                    src_c, vc, to_key, sw)
            }
            // `(dynval).vtbl != NULL` — the has-answer read off a re-witness result.
            HExprKind::DynHasVtbl(v) => {
                let vc = self.emit_expr(f, v);
                format!("(({}).vtbl != 0)", vc)
            }
        }
    }

    fn log_helper(&self, t: &HType) -> &'static str {
        match t {
            HType::Int | HType::Enum(_) | HType::SizedInt { .. } => "maka_log_int",
            HType::Float | HType::SizedFloat { .. } => "maka_log_float",
            HType::Bool => "maka_log_bool",
            HType::Char => "maka_log_char",
            HType::Str => "maka_log_str",
            // A `[N]char` VALUE logs as a string (it IS a string, Option 1); the
            // array decays to `const char*` when passed to `maka_log_str`.
            HType::Array { elem, .. } if matches!(elem.as_ref(), HType::Char) => "maka_log_str",
            HType::Ptr { .. } | HType::RawPtr { .. } | HType::OwnPtr { .. } | HType::Ref { .. } | HType::Heap { .. } => "maka_log_ptr",
            _ => "maka_log_ptr",
        }
    }

    /// A match is a pure tag dispatch - lowerable to a C `switch` (one tag read +
    /// jump table) instead of a linear if-chain - when the scrutinee is an enum
    /// value and every arm is a bare variant (no guard, no literal field-check)
    /// or a single `else`.  Returns the enum id when eligible.
    fn match_as_switch(&self, scrut_ty: &HType, arms: &[HMatchArm]) -> Option<EnumId> {
        let eid = if let HType::Enum(id) = scrut_ty { *id } else { return None };
        let mut seen_else = false;
        for a in arms {
            if a.guard.is_some() { return None; }
            match &a.kind {
                HArmKind::Variant { enum_id, lit_checks, .. } => {
                    if *enum_id != eid { return None; }
                    if lit_checks.iter().any(|c| c.is_some()) { return None; }
                }
                HArmKind::Else => { if seen_else { return None; } seen_else = true; }
                _ => return None, // Null/Lit: not an enum-tag match
            }
        }
        Some(eid)
    }

    // If `arm` produces its value through a synthetic `__yield` local, register an
    // end label and push it so a `yield` nested in the arm body (inside an
    // `if`/`while`) jumps here, terminating the arm, instead of falling through to
    // a later `yield` that would overwrite it.  Returns the label to emit after the
    // body via `yield_exit_end`.  `f` is the function whose locals `arm.value`
    // refers to (the inline callee for inline matches).
    fn yield_exit_begin(&mut self, f: &HFunc, arm: &HMatchArm) -> Option<String> {
        if let Some(HExpr { kind: HExprKind::Local(yv), .. }) = &arm.value {
            if f.locals[yv.0 as usize].name == "__yield" {
                self.yield_label_seq += 1;
                let lbl = format!("__yend_{}", self.yield_label_seq);
                self.yield_exit.push((*yv, lbl.clone()));
                return Some(lbl);
            }
        }
        None
    }

    fn yield_exit_end(&mut self, lbl: Option<String>, body: &mut String) {
        if let Some(lbl) = lbl {
            body.push_str(&format!("{}: ; ", lbl));
            self.yield_exit.pop();
        }
    }

    fn emit_match_switch(&mut self, f: &HFunc, scrutinee: &HExpr, arms: &[HMatchArm], result_ty: &HType, eid: EnumId) -> String {
        let mut s = self.emit_expr(f, scrutinee);
        let mut scrut_c = self.c_type(&scrutinee.ty);
        let via_ptr = enum_pointee(&scrutinee.ty).is_some();
        // Matching on a borrow / pointer to an enum: deref so __s holds the value.
        if let Some(enum_ty) = enum_pointee(&scrutinee.ty) {
            scrut_c = self.c_type(&enum_ty);
            s = format!("(*({}))", s);
        }
        // A freshly-produced owned-enum rvalue scrutinee (a call result or a
        // nested match value) is materialized into __s and has no other owner, so
        // __s must be dropped after the switch (with its moved-out payload fields
        // nulled first).  Anything that views a value owned elsewhere - a place
        // (local/field/index/global), a deref (`e!`), or a borrow (via_ptr) - is
        // freed by that owner, so __s (a shallow copy) must NOT be dropped here.
        let scrut_is_temp_owned = !via_ptr
            && self.drop_ty_owns(&scrutinee.ty)
            && matches!(&scrutinee.kind, HExprKind::Call { .. } | HExprKind::Match { .. });
        let res_c = self.c_type(result_ty);
        let needs_value = !matches!(result_ty, HType::Unit);
        let tag_expr = if self.sym.enum_info(eid).is_simple() { "__s" } else { "__s.tag" };
        let mut body = String::new();
        body.push_str("__extension__ ({ ");
        body.push_str(&format!("{} __s = {}; ", scrut_c, s));
        if needs_value { body.push_str(&format!("{} __r = {{0}}; ", res_c)); }
        body.push_str(&format!("switch ({}) {{ ", tag_expr));
        for a in arms {
            match &a.kind {
                HArmKind::Variant { variant, enum_id, .. } => {
                    let t = self.sym.enum_info(*enum_id).variants[*variant].tag;
                    body.push_str(&format!("case {}: {{ ", t));
                }
                HArmKind::Else => body.push_str("default: { "),
                _ => unreachable!(),
            }
            body.push_str(&self.match_bindings(f, &a.kind));
            if let Some(local) = a.scrut_binding {
                let li = &f.locals[local.0 as usize];
                body.push_str(&format!("{} {} = __s; ", self.c_type(&li.ty), self.local_name(local, &li.name)));
            }
            // Value-producing arm via a synthetic `__yield` local: register an end
            // label so a `yield` nested in the body terminates the arm.
            let yexit_lbl = self.yield_exit_begin(f, a);
            for stmt in &a.body.stmts {
                let prev_len = self.out.len();
                self.emit_stmt(f, stmt);
                let emitted = self.out[prev_len..].to_string();
                self.out.truncate(prev_len);
                body.push_str(&emitted);
            }
            self.yield_exit_end(yexit_lbl, &mut body);
            if let Some(v) = &a.value {
                // Null any owning payload field this arm moves out, so the
                // scrutinee's drop does not double-free the escaped box.
                body.push_str(&self.match_scrut_field_nulls(f, scrutinee, a));
                let vs = self.emit_expr(f, v);
                if needs_value { body.push_str(&format!("__r = ({}); ", vs)); }
                else { body.push_str(&format!("(void)({}); ", vs)); }
            }
            // Drop owning locals declared in the arm body (after the value).
            let cu = self.arm_body_cleanup(f, &a.body, None);
            body.push_str(&cu);
            body.push_str("} break; ");
        }
        body.push_str("} ");
        if scrut_is_temp_owned {
            body.push_str(&format!("__maka_drop_{}(&__s); ", scrut_c));
        }
        if needs_value { body.push_str("__r; "); } else { body.push_str("MAKA_UNIT; "); }
        body.push_str("})");
        body
    }

    /// Lower a `free value` node to a C expression of type `MAKA_UNIT`.
    /// `ptr_expr` is the already-emitted operand (a `raw *T`), `ptr_ty` its
    /// HType.  A shallow free is just `free((void*)p)`; a `free deep p` first
    /// runs the SAME recursive drop glue an owning value gets at scope exit, by
    /// treating the raw pointer as an `own *T` and reusing `emit_field_drop`, so
    /// every pointee shape (struct/enum graph, array-of-owning, `Vec`, nested)
    /// is reclaimed uniformly.  The whole thing is a GNU statement-expression so
    /// it stays a single expression usable in any position, and the operand is
    /// bound to a temp so it is evaluated exactly once.
    fn emit_free(&mut self, ptr_expr: String, ptr_ty: &HType, deep: bool) -> String {
        let shallow = || format!("(free((void*)({})), MAKA_UNIT)", ptr_expr);
        if !deep {
            return shallow();
        }
        let pointee = match ptr_ty {
            HType::RawPtr { inner, .. } | HType::Ptr { inner, .. } | HType::OwnPtr { inner, .. } => {
                inner.as_ref().clone()
            }
            // Not a pointer we can walk (should not reach here after the sema
            // `raw *T` check); degrade to a shallow free rather than miscompile.
            _ => return shallow(),
        };
        // `own *T` and `raw *T` share a C representation, so recast the operand
        // into an owning pointer and let the standard drop path free the graph.
        let own_ty = HType::OwnPtr { mutable: true, inner: Box::new(pointee) };
        let cty = self.c_type(&own_ty);
        self.free_deep_seq += 1;
        let tmp = format!("__fd{}", self.free_deep_seq);
        // emit_field_drop(OwnPtr) => `if (tmp) { <drop *tmp's owned fields>; free(tmp); }`.
        let drops = self.capture_drops(&[(tmp.clone(), own_ty)]);
        format!(
            "(__extension__ ({{ {cty} {tmp} = ({cty})({ptr_expr}); {drops} MAKA_UNIT; }}))"
        )
    }

    /// Emit drops for a resolved (c_name, type) list, captured as a C string
    /// (callers that build C into a `String` rather than self.out use this).
    fn capture_drops(&mut self, drops: &[(String, HType)]) -> String {
        if drops.is_empty() { return String::new(); }
        let prev = self.out.len();
        for (n, ty) in drops { self.emit_field_drop(n, ty, 0); }
        let s = self.out[prev..].to_string();
        self.out.truncate(prev);
        s
    }

    /// Scope-exit drops for a match-arm body, captured as a C string.  A match
    /// (and its `if`/`if-expr` sugar) is emitted into a `body` String, so an arm
    /// body's drops cannot be written straight to `self.out` like emit_block does -
    /// they are captured the same way the arm-body statements are.  The arm body is
    /// a real lexical scope (its heap_to_free is filled by the lifetime pass's
    /// visit_matches_in_expr), and these drops MUST run AFTER the arm value is
    /// evaluated (the value may read these locals) and only on the matched arm.
    /// `tag` is `Some` for an inline expansion (tagged local names) and `None`
    /// otherwise.
    fn arm_body_cleanup(&mut self, f: &HFunc, b: &HBlock, tag: Option<&str>) -> String {
        if b.heap_to_free.is_empty() { return String::new(); }
        let prev = self.out.len();
        for id in &b.heap_to_free {
            let (n, ty) = {
                let li = &f.locals[id.0 as usize];
                let n = match tag {
                    Some(t) => inline_local_name(f, *id, t),
                    None => self.local_name(*id, &li.name),
                };
                (n, li.ty.clone())
            };
            self.emit_field_drop(&n, &ty, 0);
        }
        let s = self.out[prev..].to_string();
        self.out.truncate(prev);
        s
    }

    fn emit_match(&mut self, f: &HFunc, scrutinee: &HExpr, arms: &[HMatchArm], result_ty: &HType) -> String {
        // Pure tag dispatch -> C `switch` (jump table) instead of an if-chain - but
        // ONLY when no arm body contains a break/continue: a `switch` captures
        // `break`, so a loop-targeting break inside an arm would exit the switch
        // instead of the enclosing loop.  The if-chain form below terminates arms
        // with a label+goto, so user break/continue fall through to the real loop.
        let arms_jump = arms.iter().any(|a|
            a.guard.as_ref().is_some_and(expr_has_break_continue)
            || block_has_break_continue(&a.body)
            || a.value.as_ref().is_some_and(expr_has_break_continue));
        if !arms_jump {
            if let Some(eid) = self.match_as_switch(&scrutinee.ty, arms) {
                return self.emit_match_switch(f, scrutinee, arms, result_ty, eid);
            }
        }
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
        let mut s = self.emit_expr(f, scrutinee);
        let mut scrut_c = self.c_type(&scrutinee.ty);
        // Matching on a borrow / pointer to an enum: deref so __s holds the enum
        // value (tag and payload are read with `.`, not on a pointer).
        if let Some(enum_ty) = enum_pointee(&scrutinee.ty) {
            scrut_c = self.c_type(&enum_ty);
            s = format!("(*({}))", s);
        }
        let res_c = self.c_type(result_ty);
        let needs_value = !matches!(result_ty, HType::Unit);

        // Terminate a matched arm with `goto <end>` rather than wrapping the arms in
        // a `do { ... } while (0)` and breaking out of it: a `do/while` is a C loop,
        // so a user `break`/`continue` written inside an arm body would target the
        // dispatch wrapper instead of the enclosing Maka loop (silently exiting the
        // match and then double-freeing the loop locals).  With a label, user
        // break/continue fall through to the real enclosing loop.
        self.yield_label_seq += 1;
        let mend = format!("__mend_{}", self.yield_label_seq);

        let mut body = String::new();
        body.push_str("__extension__ ({ ");
        body.push_str(&format!("{} __s = {}; ", scrut_c, s));
        if needs_value {
            body.push_str(&format!("{} __r = {{0}}; ", res_c));
        }

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
                let n = self.local_name(local, &li.name);
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
            let yexit_lbl = self.yield_exit_begin(f, a);
            for stmt in &a.body.stmts {
                let prev_len = self.out.len();
                self.emit_stmt(f, stmt);
                let emitted = self.out[prev_len..].to_string();
                self.out.truncate(prev_len);
                body.push_str(&emitted);
            }
            self.yield_exit_end(yexit_lbl, &mut body);
            if let Some(v) = &a.value {
                // Null any owning payload field this arm moves out (only once the
                // arm and its guard have committed), so the scrutinee's drop does
                // not double-free the escaped box.
                body.push_str(&self.match_scrut_field_nulls(f, scrutinee, a));
                let vs = self.emit_expr(f, v);
                if needs_value { body.push_str(&format!("__r = ({}); ", vs)); }
                else { body.push_str(&format!("(void)({}); ", vs)); }
            }
            // Drop owning locals declared in the arm body (after the value, which
            // may read them; before the break, so only the matched arm runs them).
            let cu = self.arm_body_cleanup(f, &a.body, None);
            body.push_str(&cu);
            // Close any open guard `if`, with `break;` so a matched arm exits.
            // If we put guard into the arm body, we must NOT break unconditionally
            // outside it.  This must mirror exactly when a body-guard `if` was
            // opened (`needs_body_guard && guard`) - i.e. for any variant or
            // scrut-bound arm with a guard, not only `as`-bound ones.
            let has_body_guard = needs_body_guard && a.guard.is_some();
            if has_body_guard {
                // close inner if with `goto end`; close outer arm-if without it (fall-through to next arm).
                body.push_str(&format!(" goto {}; }} }} ", mend));
            } else {
                body.push_str(&format!(" goto {}; }} ", mend));
            }
        }
        body.push_str(&format!("{}: ; ", mend));
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
            HExprKind::LitInt(n) => c_int_lit(*n),
            HExprKind::LitBool(b) => if *b { "true".into() } else { "false".into() },
            HExprKind::LitChar(c) => format!("(maka_char){}u", *c as u32),
            // `{:?}` (not `{}`): Rust's Display renders e.g. 1e30 as the
            // integer-looking "1000...000", which C parses as an integer literal
            // that overflows; Debug always emits a `.`/`e` form, round-trip exact
            // and a valid C double literal.
            HExprKind::LitFloat(v) => format!("(maka_float){:?}", v),
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
            HExprKind::LitInt(n) => c_int_lit(*n),
            HExprKind::LitBool(b) => if *b { "true".into() } else { "false".into() },
            HExprKind::LitChar(c) => format!("(maka_char){}u", *c as u32),
            // `{:?}` (not `{}`): Rust's Display renders e.g. 1e30 as the
            // integer-looking "1000...000", which C parses as an integer literal
            // that overflows; Debug always emits a `.`/`e` form, round-trip exact
            // and a valid C double literal.
            HExprKind::LitFloat(v) => format!("(maka_float){:?}", v),
            HExprKind::LitStr(s) => format!("{:?}", s),
            HExprKind::LitNull => "NULL".into(),
            HExprKind::LitUnit => "MAKA_UNIT".into(),
            HExprKind::Un { op: HUnOp::Neg, expr } => self.emit_neg(&self.emit_global_init(expr), &expr.ty),
            HExprKind::Bin { op, lhs, rhs } => {
                let l = self.emit_global_init(lhs);
                let r = self.emit_global_init(rhs);
                self.emit_arith(&l, *op, &r, &lhs.ty)
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
                        let name = self.local_name(*local, &li.name);
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

    /// True if evaluating `value` yields (moves out) the owning local `target`
    /// as part of its result value - i.e. the binding's ownership transfers into
    /// the match result (returned, stored, moved into an aggregate, or passed to
    /// a consuming call).  A bare read through the binding (`b!.field`) is NOT a
    /// move.  Used to decide whether a matched owning payload escapes.
    fn value_moves_binding(&self, value: &HExpr, target: LocalId) -> bool {
        match &value.kind {
            // A bare `Local` is a move only if it still carries an owning type here.
            // When the binding was coerced to a borrowed view (an owning `String`
            // passed to a `string` / `&T` / `*T` parameter, e.g. `str_len(body)`),
            // its type at this site no longer owns heap, so ownership does NOT
            // escape - the scrutinee keeps it and its drop frees it.  Treating
            // such a borrow as a move would null the scrutinee field and leak.
            HExprKind::Local(id) => *id == target && self.drop_ty_owns(&value.ty),
            HExprKind::Transfer(inner)
            | HExprKind::Cast { expr: inner, .. }
            | HExprKind::CheckedCast { expr: inner, .. }
            | HExprKind::DropWrite(inner)
            | HExprKind::HeapAlloc(inner) => self.value_moves_binding(inner, target),
            HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } =>
                fields.iter().any(|(_, fe)| self.value_moves_binding(fe, target)),
            HExprKind::ArrayLit(es) => es.iter().any(|e| self.value_moves_binding(e, target)),
            HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } =>
                args.iter().any(|a| self.value_moves_binding(a, target)),
            HExprKind::CallIndirect { callee, args } =>
                self.value_moves_binding(callee, target) || args.iter().any(|a| self.value_moves_binding(a, target)),
            // Operator operands and a unary operand can contain a move (e.g. a
            // call `eat(s)` inside `eat(s) > 0`).  A bare owning Local never appears
            // as a Bin/Un operand (operator overloads auto-borrow their operands),
            // so recursing here flags real moves without false positives.  Field /
            // Index-base / Unwrap / AddrOfRef are borrow positions (a bare Local
            // there is a read, not a move), so they are deliberately NOT recursed;
            // only Index's subscript expression is a value position.
            HExprKind::Bin { lhs, rhs, .. } =>
                self.value_moves_binding(lhs, target) || self.value_moves_binding(rhs, target),
            HExprKind::Un { expr, .. } => self.value_moves_binding(expr, target),
            HExprKind::Index { idx, .. } => self.value_moves_binding(idx, target),
            HExprKind::Closure { env_values, .. } =>
                env_values.iter().any(|v| self.value_moves_binding(v, target)),
            // Evaluating a nested match moves `target` if its scrutinee, any
            // guard, any arm value, OR any arm body moves it - not just the arm
            // values.  Without the body/guard check a move inside a nested match's
            // body would not null the outer scrutinee's field -> double-free.
            HExprKind::Match { scrutinee, arms, .. } =>
                self.value_moves_binding(scrutinee, target)
                    || arms.iter().any(|a| {
                        a.value.as_ref().is_some_and(|v| self.value_moves_binding(v, target))
                            || a.guard.as_ref().is_some_and(|g| self.value_moves_binding(g, target))
                            || self.stmts_move_binding(&a.body.stmts, target)
                    }),
            _ => false,
        }
    }

    /// Does this match arm move `target` (one of its own bindings) out of the
    /// scrutinee?  Covers an expression-form arm value as well as a statement-form
    /// `return`/`propagate` that carries the binding away (the inline-body shape:
    /// `Ok { value } { return value; }`).  Used to null the moved-out field of an
    /// owning scrutinee so its later drop does not double-free the escaped payload.
    fn arm_moves_binding(&self, arm: &HMatchArm, target: LocalId) -> bool {
        if arm.value.as_ref().is_some_and(|v| self.value_moves_binding(v, target)) {
            return true;
        }
        self.stmts_move_binding(&arm.body.stmts, target)
    }

    fn stmts_move_binding(&self, stmts: &[HStmt], target: LocalId) -> bool {
        stmts.iter().any(|s| match s {
            HStmt::Return { value: Some(v), .. } | HStmt::Propagate { value: Some(v), .. }
            | HStmt::ExprStmt(v) => self.value_moves_binding(v, target),
            HStmt::Let { init, .. } => self.value_moves_binding(init, target),
            HStmt::Assign { value, .. } => self.value_moves_binding(value, target),
            HStmt::If { then_b, else_b, .. } =>
                self.stmts_move_binding(&then_b.stmts, target)
                    || else_b.as_ref().is_some_and(|b| self.stmts_move_binding(&b.stmts, target)),
            HStmt::While { body, .. } | HStmt::ForEach { body, .. } =>
                self.stmts_move_binding(&body.stmts, target),
            HStmt::ForC { body, .. } => self.stmts_move_binding(&body.stmts, target),
            HStmt::Block(b) | HStmt::Unsafe(b, _) => self.stmts_move_binding(&b.stmts, target),
            _ => false,
        })
    }

    /// When matching an owning enum VALUE held in a place (a Local), an arm that
    /// moves an owning *pointer* payload out of the scrutinee (yields it as the
    /// result) must null that field in the scrutinee place, so the scrutinee's
    /// later drop glue does not double-free the now-escaped box.  Returns the C
    /// statements to splice after the arm's bindings (empty when none apply).
    fn match_scrut_field_nulls(&self, f: &HFunc, scrutinee: &HExpr, arm: &HMatchArm) -> String {
        let place = match &scrutinee.kind {
            HExprKind::Local(id) if matches!(scrutinee.ty, HType::Enum(_)) => {
                let li = &f.locals[id.0 as usize];
                self.local_name(*id, &li.name)
            }
            // A freshly-produced owned-enum rvalue scrutinee (call/match result)
            // lives in `__s` (dropped after the switch); null the fields it moves
            // out so that drop does not double-free the escaped payload.  Must
            // mirror `scrut_is_temp_owned` in emit_match_switch exactly.
            _ if matches!(scrutinee.ty, HType::Enum(_))
                && self.drop_ty_owns(&scrutinee.ty)
                && enum_pointee(&scrutinee.ty).is_none()
                && matches!(&scrutinee.kind, HExprKind::Call { .. } | HExprKind::Match { .. }) =>
            {
                "__s".to_string()
            }
            _ => return String::new(),
        };
        let HArmKind::Variant { enum_id, variant, bindings, .. } = &arm.kind else { return String::new(); };
        let info = self.sym.enum_info(*enum_id);
        let v = &info.variants[*variant];
        let mut out = String::new();
        for (i, b) in bindings.iter().enumerate() {
            let Some(local) = b else { continue; };
            let li = &f.locals[local.0 as usize];
            // Any owning binding (an `own *T`/`own &T` pointer OR an owning struct
            // like `String`/a `Vec`-backed value) that this arm moves out: zero the
            // scrutinee's field so its drop does not double-free the escaped value.
            // `arm_moves_binding` also catches a statement-form `return`/`propagate`,
            // not just the expression-form arm value.
            if !self.drop_ty_owns(&li.ty) { continue; }
            if self.arm_moves_binding(arm, *local) {
                out.push_str(&format!("{}.payload.{}.{} = ({}){{0}}; ",
                    place, c_ident(&v.name), c_ident(&v.fields[i].name), self.c_type(&li.ty)));
            }
        }
        out
    }

    fn emit_to_dyn(&mut self, f: &HFunc, expr: &HExpr, traits: &[String], struct_id: StructId) -> String {
        // Source might be `&T`, `&mut T`, or `T`. For all three, we take the address of the value.
        let sname = self.sym.struct_info(struct_id).name.clone();
        // Need an address. If expr is already a reference (Ref), emit_expr returns `&local` form.
        let s = self.emit_expr(f, expr);
        // If the expression's type is a Ref, its C value is already `T*`. Else, take address.
        let data_expr = match &expr.ty {
            HType::Ref { .. } => format!("(void*)({})", s),
            _ => format!("(void*)(&({}))", s),
        };
        // Single or combined (multi-trait) vtable, keyed by the joined trait set.
        let key = c_ident(&traits.join("_"));
        format!("((Dyn_{0}){{ .data = {1}, .vtbl = &{0}_vtbl_for_{2} }})",
                key, data_expr, c_ident(&sname))
    }

    /// The env's recursive-drop function for a spawn-closure argument, so the
    /// thread frees any owner MOVED into its capture env - even on cancel, where
    /// the body never runs.  NULL when the arg is not a literal closure, or its
    /// env owns no heap (only Shareable / by-value-copied captures, which the env
    /// must not free).  Paired with the lifetime pass marking those owner
    /// captures moved, so the enclosing scope no longer frees them.
    fn spawn_env_drop(&self, a: &HExpr) -> String {
        if let HExprKind::Closure { env_struct, .. } = &a.kind {
            let name = self.sym.struct_info(*env_struct).name.clone();
            if self.drop_owns.contains(&name) {
                return format!("(void*)__maka_drop_{}", c_ident(&name));
            }
        }
        "NULL".to_string()
    }

    /// Saturating float->int conversion (C makes the out-of-range / NaN cases UB).
    fn emit_float_to_int(&self, s: &str, to: &HType) -> String {
        let to_c = self.c_type(to);
        let bits = int_bits(to);
        if is_signed_int(to) {
            let (hi, mn, mx) = sat_signed_lits(bits);
            format!("maka_f2i({}, {}, {}, {}, {})", s, to_c, hi, mn, mx)
        } else {
            let (hi, mx) = sat_unsigned_lits(bits);
            format!("maka_f2u({}, {}, {}, {})", s, to_c, hi, mx)
        }
    }

    /// A parameter declaration for an `extern` (C-ABI) function.  A Maka
    /// fn-pointer parameter on an extern is a BARE C callback `R (*name)(P...)`,
    /// NOT the fat env-carrying `Callable_KEY` a Maka closure uses - a C function
    /// can only receive a plain function pointer.  (Reverse of a Maka fn-pointer
    /// value; the call site passes a named function's direct C address for it.)
    fn c_extern_param_decl(&self, t: &HType, name: &str) -> String {
        if let HType::FnPtr { ret, params } = t {
            let ret_c = self.c_ret_type(ret);
            let ps: Vec<String> = if params.is_empty() { vec!["void".to_string()] }
                else { params.iter().map(|p| self.c_type(p)).collect() };
            return format!("{} (*{})({})", ret_c, name, ps.join(", "));
        }
        format!("{} {}", self.c_type(t), name)
    }

    fn emit_cast(&self, s: String, kind: CastKind, to: &HType, from: &HType) -> String {
        let to_c = self.c_type(to);
        match kind {
            // float -> int is C-undefined out of range / on NaN; saturate it.
            CastKind::Numeric if is_float_ty(from) && is_int_ty(to) => {
                self.emit_float_to_int(&s, to)
            }
            CastKind::Numeric | CastKind::SignChange | CastKind::EnumToInt | CastKind::CharIntInt | CastKind::Identity => {
                format!("(({}){})", to_c, s)
            }
            // §3 `int as Enum` — runtime bounds-check against the variant
            // count, panic on out-of-range, return the Enum value.  Same
            // shape as `arr[i]`: a fail-fast guard, not a fallibility carrier.
            CastKind::IntToEnumChecked => {
                if let HType::Enum(eid) = to {
                    let info = self.sym.enum_info(*eid);
                    let cond = if info.variants.is_empty() {
                        "0".to_string()
                    } else {
                        info.variants.iter()
                            .map(|v| format!("__v == {}", v.tag))
                            .collect::<Vec<_>>()
                            .join(" || ")
                    };
                    let ec = c_ident(&info.name);
                    return format!(
                        "(__extension__ ({{ maka_int __v = ({0}); if (!({1})) {{ maka_panic(\"`int as {2}`: tag out of range\"); }} ({3})__v; }}))",
                        s, cond, info.name, ec,
                    );
                }
                format!("(({}){})", to_c, s)
            }
            // §6.6 `*int → *Enum` — peek at the pointee, validate the tag,
            // return the same pointer cast to `*Enum` (in range) or NULL
            // (out of range).  Failure-via-nullability — no panic, no
            // wrapping; the result *is* the requested `*Enum`.
            CastKind::IntPtrToEnumPtrChecked => {
                if let HType::Ptr { inner, .. } = to {
                    if let HType::Enum(eid) = inner.as_ref() {
                        let info = self.sym.enum_info(*eid);
                        let cond = if info.variants.is_empty() {
                            "0".to_string()
                        } else {
                            info.variants.iter()
                                .map(|v| format!("__v == {}", v.tag))
                                .collect::<Vec<_>>()
                                .join(" || ")
                        };
                        let ec = c_ident(&info.name);
                        return format!(
                            "(__extension__ ({{ maka_int* __p = ({0}); maka_int __v = *__p; ({1}) ? ({2}*)__p : ({2}*)0; }}))",
                            s, cond, ec,
                        );
                    }
                }
                format!("(({}){})", to_c, s)
            }
            // Reinterpret: a plain C cast — works for ptr↔ptr, ptr↔intptr_t, etc.
            // The `(uintptr_t)` round-trip silences GCC's "incompatible pointer types"
            // warnings on direct *T↔*U casts.
            CastKind::Reinterpret => format!("(({})(uintptr_t)({}))", to_c, s),
            // Pack `Vec<T>` -> `Vec<some X>`: copy the buffer into a column that owns
            // its own storage (so the source `Vec<T>` and the column don't both free
            // one buffer), and attach the hoisted witness (the `X`-vtable for `T` and
            // `sizeof(T)`).  The column's buffer is freed by the SomeVec drop glue.
            CastKind::PackSomeVec { trait_name, struct_id } => {
                let sname = self.sym.struct_info(struct_id).name.clone();
                let ec = c_ident(&sname);
                // Per-element drop-fn: the struct's recursive drop glue if it owns
                // heap, else NULL (POD elements need only the buffer freed).
                let drop_fn = if self.drop_ty_owns(&HType::Struct(struct_id)) {
                    format!("(void(*)(void*))&__maka_drop_{}", ec)
                } else { "((void(*)(void*))0)".to_string() };
                // MOVE the buffer (transfer ownership - the source `Vec<T>` is marked
                // moved by the lifetime pass, so only the column frees it).  A copy
                // would be a SHALLOW copy that double-frees owning elements' inner heap.
                format!("(__extension__ ({{ {vt} __sv = ({src}); ({sty}){{ .data = __sv.data, .len = __sv.len, .cap = __sv.cap, .__vtbl = &{tr}_vtbl_for_{st}, .__esz = (maka_int)sizeof({ec}), .__drop = {df} }}; }}))",
                    vt = self.c_type(from), src = s, sty = to_c,
                    tr = c_ident(&trait_name), st = c_ident(&sname), ec = ec, df = drop_fn)
            }
            // Witness-checked downcast: alias the column's Vec header (data/len/cap
            // share layout) as `*Vec<T>` iff the hoisted vtable is `T`'s, else null.
            // `s` must be an lvalue (a stored column); it is read twice, both pure.
            CastKind::DowncastSomeVec { trait_name, struct_id } => {
                let sname = self.sym.struct_info(struct_id).name.clone();
                format!("((({src}).__vtbl == &{tr}_vtbl_for_{st}) ? ({tc})&({src}) : ({tc})0)",
                    src = s, tr = c_ident(&trait_name), st = c_ident(&sname), tc = to_c)
            }
            // Devirtualization: the receiver's concrete type is statically proven,
            // so view the fat pointer's `.data` as `&T` (`T*`) for a direct call.
            CastKind::DynDataRef { struct_id: _ } => {
                format!("(({})({}).data)", to_c, s)
            }
            _ => format!("(({}){})", to_c, s),
        }
    }

}

/// The element type produced by iterating `t` in a for-each, peeling any
/// reference/pointer/heap wrapper around the container.  Used to decide whether
/// a loop binding is the element by value or a reference to it.
fn foreach_elem_ty(t: &HType) -> Option<&HType> {
    match t {
        HType::Array { elem, .. } | HType::Slice { elem, .. } | HType::Vec { elem } => Some(elem),
        HType::Heap { inner }
        | HType::Ref { inner, .. } | HType::Ptr { inner, .. }
        | HType::OwnPtr { inner, .. } | HType::RawPtr { inner, .. } => foreach_elem_ty(inner),
        _ => None,
    }
}

/// If a `ForC` is a for-range counter provably within `[0, bound)` whose counter
/// is never mutated in the body, return `(local id, exclusive bound)`.  Used to
/// elide array bounds checks on `arr[i]` when `bound <= array length`.
fn forrange_bound(init: &HStmt, cond: &HExpr, body: &HBlock) -> Option<(u32, BceBound)> {
    // init: `Let { local: iv, init: LitInt(lo) }` with lo >= 0
    let iv = match init {
        HStmt::Let { local, init, .. } => match &init.kind {
            HExprKind::LitInt(n) if *n >= 0 => local.0,
            _ => return None,
        },
        _ => return None,
    };
    let (op, rhs) = match &cond.kind {
        HExprKind::Bin { op, lhs, rhs } => {
            if !matches!(&lhs.kind, HExprKind::Local(id) if id.0 == iv) { return None; }
            (op, rhs)
        }
        _ => return None,
    };
    let bound = match (op, &rhs.kind) {
        // Constant upper bound (covers fixed arrays).
        (HBinOp::Lt, HExprKind::LitInt(b)) if *b >= 0 => BceBound::Const(*b),
        (HBinOp::Le, HExprKind::LitInt(b)) if *b >= 0 => BceBound::Const(b.checked_add(1)?),
        // `i < s.len` of a slice/vec local s (exclusive only; `<= len` is OOB).
        // s must not be mutated/reassigned in the body, or its len could change.
        (HBinOp::Lt, HExprKind::SliceLen(inner)) => match &inner.kind {
            HExprKind::Local(s) if !block_mutates_local(body, s.0) => BceBound::SliceLen(s.0),
            _ => return None,
        },
        _ => return None,
    };
    // The counter must not be reassigned or `&mut`-borrowed in the body, or the
    // `[0, bound)` guarantee is void (e.g. `for i in 0..3 { i = 10; arr[i] }`).
    if block_mutates_local(body, iv) { return None; }
    Some((iv, bound))
}

fn block_mutates_local(b: &HBlock, iv: u32) -> bool {
    b.stmts.iter().any(|s| stmt_mutates_local(s, iv))
}

/// Is the place's root local `iv`?  Peels field/index/unwrap/deref, so `p`,
/// `p.x`, `p.arr[i]`, `p!` all root at `p`.  Used to detect writes *through* iv.
fn place_root_is(e: &HExpr, iv: u32) -> bool {
    match &e.kind {
        HExprKind::Local(id) => id.0 == iv,
        HExprKind::Field { base, .. } | HExprKind::Index { base, .. } => place_root_is(base, iv),
        HExprKind::Unwrap { expr, .. } | HExprKind::DerefRef(expr) => place_root_is(expr, iv),
        _ => false,
    }
}

/// The root local id of a place expression, if it bottoms out in a local.
/// If `t` is a borrow / pointer to an enum (`&E`, `*E`, `own *E`, ...), return
/// the pointee enum type, so a `match` on it can deref to the value.
/// Does this block (recursively, including nested match-arm bodies) contain a
/// `break`/`continue`?  A `match` is emitted as a `switch` (or, when fixed, a
/// labeled if-chain); a `switch` is a C break target, so an arm body containing a
/// loop-targeting break/continue must NOT use the switch form, or the jump would
/// exit the switch instead of the enclosing Maka loop.  Over-approximates (descends
/// into nested loops too) - that only costs the jump-table optimization, never
/// correctness.
fn block_has_break_continue(b: &HBlock) -> bool {
    b.stmts.iter().any(stmt_has_break_continue)
}
fn stmt_has_break_continue(s: &HStmt) -> bool {
    match s {
        HStmt::Break { .. } | HStmt::Continue { .. } => true,
        HStmt::If { then_b, else_b, .. } => block_has_break_continue(then_b) || else_b.as_ref().is_some_and(block_has_break_continue),
        HStmt::While { body, .. } | HStmt::Block(body) | HStmt::Unsafe(body, _)
        | HStmt::ForC { body, .. } | HStmt::ForEach { body, .. } => block_has_break_continue(body),
        HStmt::Let { init, .. } => expr_has_break_continue(init),
        HStmt::Assign { value, .. } => expr_has_break_continue(value),
        HStmt::ExprStmt(e) => expr_has_break_continue(e),
        HStmt::Return { value, .. } | HStmt::Propagate { value, .. } => value.as_ref().is_some_and(expr_has_break_continue),
    }
}
fn expr_has_break_continue(e: &HExpr) -> bool {
    // break/continue only occur in statement position, reachable through a nested
    // match's arm bodies; find any such match.
    match &e.kind {
        HExprKind::Match { scrutinee, arms, .. } =>
            expr_has_break_continue(scrutinee) || arms.iter().any(|a|
                a.guard.as_ref().is_some_and(expr_has_break_continue)
                || block_has_break_continue(&a.body)
                || a.value.as_ref().is_some_and(expr_has_break_continue)),
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => args.iter().any(expr_has_break_continue),
        HExprKind::CallIndirect { callee, args } => expr_has_break_continue(callee) || args.iter().any(expr_has_break_continue),
        HExprKind::Bin { lhs, rhs, .. } => expr_has_break_continue(lhs) || expr_has_break_continue(rhs),
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. } | HExprKind::Cast { expr, .. }
        | HExprKind::CheckedCast { expr, .. } | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr)
        | HExprKind::HeapAlloc(expr) | HExprKind::Free(expr, _) | HExprKind::SliceLen(expr)
        | HExprKind::EnumTag(expr) | HExprKind::Transfer(expr) | HExprKind::ArrayToSlice { base: expr, .. }
        | HExprKind::AddrOfRef { place: expr, .. } | HExprKind::Field { base: expr, .. } => expr_has_break_continue(expr),
        HExprKind::Index { base, idx } => expr_has_break_continue(base) || expr_has_break_continue(idx),
        HExprKind::Closure { env_values, .. } => env_values.iter().any(expr_has_break_continue),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => fields.iter().any(|(_, fe)| expr_has_break_continue(fe)),
        HExprKind::ArrayLit(es) => es.iter().any(expr_has_break_continue),
        _ => false,
    }
}

fn enum_pointee(t: &HType) -> Option<HType> {
    match t {
        HType::Ref { inner, .. } | HType::Ptr { inner, .. } | HType::RawPtr { inner, .. }
        | HType::OwnPtr { inner, .. } | HType::Heap { inner } => {
            if matches!(inner.as_ref(), HType::Enum(_)) { Some((**inner).clone()) } else { None }
        }
        _ => None,
    }
}

fn place_root_local(e: &HExpr) -> Option<u32> {
    match &e.kind {
        HExprKind::Local(id) => Some(id.0),
        HExprKind::Field { base, .. } | HExprKind::Index { base, .. } => place_root_local(base),
        HExprKind::Unwrap { expr, .. } | HExprKind::DerefRef(expr) => place_root_local(expr),
        _ => None,
    }
}

/// Is this place a PROJECTION of its root (a field or index chain) rather than
/// the bare root local itself?  A projection's old value only flows into the
/// RHS if the RHS reads that same projection - which the aliasing-safe
/// reassign order handles - so drop-then-move applies.  The bare local keeps
/// the leave-alone rule (`node = node.next` list walking).
fn place_is_projection(e: &HExpr) -> bool {
    match &e.kind {
        HExprKind::Index { .. } | HExprKind::Field { .. } => true,
        HExprKind::Unwrap { expr, .. } | HExprKind::DerefRef(expr) => place_is_projection(expr),
        _ => false,
    }
}

/// Does `e` reference local `id` anywhere?  Used as the drop-on-reassign guard:
/// if the RHS reads the slot being overwritten, dropping the old value first
/// could free something the RHS still needs (or already moved).
/// True if `id` is used by VALUE (read or moved) anywhere in `e` - i.e. in any
/// position OTHER than purely as the target of a borrow (`&id`, `&id.f`, `&id[k]`).
/// Drop-on-reassign uses this: if an assignment's RHS uses the place's local only
/// via a borrow, the old value is still live and must be dropped after the new
/// value is computed; if it reads/moves the local by value, the new value is
/// derived from the old one and the old must NOT be dropped.
fn local_used_by_value(e: &HExpr, id: u32) -> bool {
    match &e.kind {
        HExprKind::Local(l) => l.0 == id,
        HExprKind::AddrOfRef { place, .. } => addrof_place_value_use(place, id),
        HExprKind::Field { base, .. } => local_used_by_value(base, id),
        HExprKind::Index { base, idx } => local_used_by_value(base, id) || local_used_by_value(idx, id),
        HExprKind::Unwrap { expr, .. } | HExprKind::Un { expr, .. }
        | HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. }
        | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr)
        | HExprKind::HeapAlloc(expr) | HExprKind::Free(expr, _)
        | HExprKind::SliceLen(expr) | HExprKind::EnumTag(expr)
        | HExprKind::ArrayToSlice { base: expr, .. } => local_used_by_value(expr, id),
        HExprKind::Bin { lhs, rhs, .. } => local_used_by_value(lhs, id) || local_used_by_value(rhs, id),
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => args.iter().any(|a| local_used_by_value(a, id)),
        HExprKind::CallIndirect { callee, args } => local_used_by_value(callee, id) || args.iter().any(|a| local_used_by_value(a, id)),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => fields.iter().any(|(_, fe)| local_used_by_value(fe, id)),
        HExprKind::ArrayLit(es) => es.iter().any(|e| local_used_by_value(e, id)),
        HExprKind::Closure { env_values, .. } => env_values.iter().any(|v| local_used_by_value(v, id)),
        HExprKind::Transfer(inner) => local_used_by_value(inner, id),
        HExprKind::Match { scrutinee, arms, .. } => local_used_by_value(scrutinee, id)
            || arms.iter().any(|a| a.guard.as_ref().map_or(false, |g| local_used_by_value(g, id))
                || a.value.as_ref().map_or(false, |v| local_used_by_value(v, id))),
        _ => false,
    }
}

/// AddrOfRef arm of `local_used_by_value`: a place-chain rooted at a Local is
/// borrowed (not a value use), but index expressions inside it are value uses.
fn addrof_place_value_use(place: &HExpr, id: u32) -> bool {
    match &place.kind {
        HExprKind::Local(_) => false,
        HExprKind::Field { base, .. } => addrof_place_value_use(base, id),
        HExprKind::Index { base, idx } => addrof_place_value_use(base, id) || local_used_by_value(idx, id),
        HExprKind::Unwrap { expr, .. } | HExprKind::DerefRef(expr) => addrof_place_value_use(expr, id),
        _ => local_used_by_value(place, id),
    }
}

fn expr_contains_local(e: &HExpr, id: u32) -> bool {
    match &e.kind {
        HExprKind::Local(l) => l.0 == id,
        HExprKind::Field { base, .. } => expr_contains_local(base, id),
        HExprKind::Index { base, idx } => expr_contains_local(base, id) || expr_contains_local(idx, id),
        HExprKind::Unwrap { expr, .. } | HExprKind::Un { expr, .. }
        | HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. }
        | HExprKind::DropWrite(expr) | HExprKind::DerefRef(expr)
        | HExprKind::HeapAlloc(expr) | HExprKind::Free(expr, _)
        | HExprKind::SliceLen(expr) | HExprKind::EnumTag(expr)
        | HExprKind::ArrayToSlice { base: expr, .. } => expr_contains_local(expr, id),
        HExprKind::AddrOfRef { place, .. } => expr_contains_local(place, id),
        HExprKind::Bin { lhs, rhs, .. } => expr_contains_local(lhs, id) || expr_contains_local(rhs, id),
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => args.iter().any(|a| expr_contains_local(a, id)),
        HExprKind::CallIndirect { callee, args } => expr_contains_local(callee, id) || args.iter().any(|a| expr_contains_local(a, id)),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } => fields.iter().any(|(_, fe)| expr_contains_local(fe, id)),
        HExprKind::ArrayLit(es) => es.iter().any(|e| expr_contains_local(e, id)),
        HExprKind::Closure { env_values, .. } => env_values.iter().any(|v| expr_contains_local(v, id)),
        HExprKind::Transfer(inner) => expr_contains_local(inner, id),
        // Arm BODIES too: a match RHS that reads the reassigned place in an arm body
        // (e.g. `x = match c { 0 { ...x!.v...; yield ... } }`) DOES use the place, so
        // the drop-on-reassign must not free the old value before the RHS runs.
        HExprKind::Match { scrutinee, arms, .. } => expr_contains_local(scrutinee, id)
            || arms.iter().any(|a| a.guard.as_ref().map_or(false, |g| expr_contains_local(g, id))
                || block_contains_local(&a.body, id)
                || a.value.as_ref().map_or(false, |v| expr_contains_local(v, id))),
        _ => false,
    }
}

fn block_contains_local(b: &HBlock, id: u32) -> bool { b.stmts.iter().any(|s| stmt_contains_local(s, id)) }
fn stmt_contains_local(s: &HStmt, id: u32) -> bool {
    match s {
        HStmt::Let { init, .. } => expr_contains_local(init, id),
        HStmt::Assign { place, value, .. } => expr_contains_local(place, id) || expr_contains_local(value, id),
        HStmt::ExprStmt(e) => expr_contains_local(e, id),
        HStmt::Return { value: Some(v), .. } | HStmt::Propagate { value: Some(v), .. } => expr_contains_local(v, id),
        HStmt::If { cond, then_b, else_b, .. } => expr_contains_local(cond, id) || block_contains_local(then_b, id) || else_b.as_ref().is_some_and(|b| block_contains_local(b, id)),
        HStmt::While { cond, body, .. } => expr_contains_local(cond, id) || block_contains_local(body, id),
        HStmt::ForC { init, cond, step, body, .. } => stmt_contains_local(init, id) || expr_contains_local(cond, id) || stmt_contains_local(step, id) || block_contains_local(body, id),
        HStmt::ForEach { src, body, .. } => expr_contains_local(src, id) || block_contains_local(body, id),
        HStmt::Block(b) | HStmt::Unsafe(b, _) => block_contains_local(b, id),
        _ => false,
    }
}

/// Is this place rooted in something we own (a value local/global, or reached
/// through an owning pointer) rather than a borrow?  Only then is it sound to
/// null the place on move-out.  Moving out through a `&T`/`*T` borrow is not.
/// A callee whose `FuncId` is a reserved built-in sentinel (panic/log/format/push/
/// pop/spawn/channel/atomic/...) rather than a real user function.  Sema reserves the
/// top `[u32::MAX-1024, u32::MAX]` of the id space for these (see the mirror check in
/// typeck: `fid.0 < u32::MAX - 1024` == "a real, inline-able function").  Such calls
/// are lowered by `emit_builtin_sentinel`, never by the `func_sig` path.
fn is_builtin_sentinel(callee: FuncId) -> bool {
    callee.0 >= u32::MAX - 1024
}

fn move_out_owned_place(e: &HExpr) -> bool {
    match &e.kind {
        HExprKind::Field { base, .. } | HExprKind::Index { base, .. } => match &base.ty {
            HType::Struct(_) | HType::Enum(_) | HType::Array { .. } | HType::Vec { .. } => move_out_owned_place(base),
            // A `&mut` borrow grants mutable access to owned data, so moving out
            // of one of its fields/elements is valid (and must null the slot).
            // A `&T` borrow / `*T` / `raw *T` is not owned - do not null.
            HType::Ref { mutable: true, .. } => move_out_owned_place(base),
            HType::OwnPtr { .. } | HType::Heap { .. } => true,
            _ => false,
        },
        HExprKind::Unwrap { expr, .. } => matches!(&expr.ty, HType::OwnPtr { .. } | HType::Heap { .. }),
        HExprKind::Local(_) | HExprKind::GlobalRef(_) => true,
        _ => false,
    }
}

fn stmt_mutates_local(s: &HStmt, iv: u32) -> bool {
    match s {
        HStmt::Let { init, .. } => expr_mutates_local(init, iv),
        HStmt::Assign { place, value, .. } => {
            // A write to any place rooted at iv (iv itself, iv.field, iv.arr[j]).
            place_root_is(place, iv)
                || expr_mutates_local(place, iv) || expr_mutates_local(value, iv)
        }
        HStmt::ExprStmt(e) => expr_mutates_local(e, iv),
        HStmt::Return { value, .. } => value.as_ref().map_or(false, |v| expr_mutates_local(v, iv)),
        HStmt::Propagate { value, .. } => value.as_ref().map_or(false, |v| expr_mutates_local(v, iv)),
        HStmt::If { cond, then_b, else_b, .. } => {
            expr_mutates_local(cond, iv) || block_mutates_local(then_b, iv)
                || else_b.as_ref().map_or(false, |b| block_mutates_local(b, iv))
        }
        HStmt::While { cond, body, .. } => expr_mutates_local(cond, iv) || block_mutates_local(body, iv),
        HStmt::Block(b) | HStmt::Unsafe(b, _) => block_mutates_local(b, iv),
        HStmt::ForC { init, cond, step, body, .. } =>
            stmt_mutates_local(init, iv) || expr_mutates_local(cond, iv)
                || stmt_mutates_local(step, iv) || block_mutates_local(body, iv),
        HStmt::ForEach { src, body, .. } => expr_mutates_local(src, iv) || block_mutates_local(body, iv),
        HStmt::Break { .. } | HStmt::Continue { .. } => false,
    }
}

/// Conservatively, does `e` reassign or `&mut`-borrow local `iv`?  Exhaustive
/// (no wildcard) so a new HExpr variant forces an explicit decision here.
fn expr_mutates_local(e: &HExpr, iv: u32) -> bool {
    match &e.kind {
        HExprKind::AddrOfRef { mutable, place } =>
            (*mutable && place_root_is(place, iv)) || expr_mutates_local(place, iv),
        HExprKind::Field { base, .. } => expr_mutates_local(base, iv),
        HExprKind::Index { base, idx } => expr_mutates_local(base, iv) || expr_mutates_local(idx, iv),
        HExprKind::Call { args, .. } | HExprKind::InlineCall { args, .. } => args.iter().any(|a| expr_mutates_local(a, iv)),
        HExprKind::CallIndirect { callee, args } => expr_mutates_local(callee, iv) || args.iter().any(|a| expr_mutates_local(a, iv)),
        HExprKind::Bin { lhs, rhs, .. } => expr_mutates_local(lhs, iv) || expr_mutates_local(rhs, iv),
        HExprKind::Un { expr, .. } | HExprKind::Unwrap { expr, .. }
        | HExprKind::Cast { expr, .. } | HExprKind::CheckedCast { expr, .. } => expr_mutates_local(expr, iv),
        HExprKind::DropWrite(e) | HExprKind::DerefRef(e) | HExprKind::HeapAlloc(e)
        | HExprKind::Free(e, _) | HExprKind::Transfer(e) | HExprKind::SliceLen(e)
        | HExprKind::EnumTag(e) | HExprKind::DynHasVtbl(e)
        | HExprKind::DynRewitness { value: e, .. } => expr_mutates_local(e, iv),
        HExprKind::ArrayToSlice { base, .. } => expr_mutates_local(base, iv),
        HExprKind::Struct { fields, .. } | HExprKind::VariantCtor { fields, .. } =>
            fields.iter().any(|(_, fe)| expr_mutates_local(fe, iv)),
        HExprKind::ArrayLit(es) => es.iter().any(|x| expr_mutates_local(x, iv)),
        HExprKind::Closure { env_values, .. } => env_values.iter().any(|x| expr_mutates_local(x, iv)),
        HExprKind::Match { scrutinee, arms, .. } => {
            expr_mutates_local(scrutinee, iv)
                || arms.iter().any(|a|
                    a.guard.as_ref().map_or(false, |g| expr_mutates_local(g, iv))
                    || a.value.as_ref().map_or(false, |v| expr_mutates_local(v, iv))
                    || block_mutates_local(&a.body, iv))
        }
        // No sub-expressions: cannot mutate anything.
        HExprKind::LitInt(_) | HExprKind::LitFloat(_) | HExprKind::LitBool(_)
        | HExprKind::LitChar(_) | HExprKind::LitStr(_) | HExprKind::LitNull
        | HExprKind::LitUnit | HExprKind::ZeroInit | HExprKind::Local(_) | HExprKind::EnumVariant(_, _)
        | HExprKind::FnRef(_) | HExprKind::GlobalRef(_) => false,
    }
}

/// Wrap an emitted value for a printf argument, matching the conversion spec
/// chosen in typeck: bool -> "true"/"false" (%s), char -> int (%c), integers ->
/// long long (%lld), floats -> double (%g); strings pass through (%s).
fn printf_conv(s: String, ty: &HType) -> String {
    match ty {
        HType::Bool => format!("(({}) ? \"true\" : \"false\")", s),
        HType::Char => format!("(int)({})", s),
        HType::Int | HType::SizedInt { .. } => format!("(long long)({})", s),
        HType::Float | HType::SizedFloat { .. } => format!("(double)({})", s),
        _ => s,
    }
}

/// An integer `/` or `%` whose divisor is not a known-non-zero literal needs a
/// runtime zero-check (`maka_check_div`): a zero divisor is C undefined
/// behaviour.  Float division is defined (inf/nan) so it is never guarded, and a
/// Bit width of an integer operand for shift bounds: a valid shift amount is in
/// `0..width`.  `int` and pointer-sized ints are 64; `char`/`u8` are 8.
fn shift_width(ty: &HType) -> u32 {
    match ty {
        HType::Char => 8,
        HType::SizedInt { bits: 0, .. } => 64,
        HType::SizedInt { bits, .. } => *bits as u32,
        _ => 64,
    }
}

/// Whether a left shift of this type needs the unsigned-domain workaround for the
/// sign bit (`int` and signed sized ints; `char`/`u8` are unsigned).
fn is_signed_int(ty: &HType) -> bool {
    matches!(ty, HType::Int) || matches!(ty, HType::SizedInt { signed: true, .. })
}

fn is_float_ty(ty: &HType) -> bool {
    matches!(ty, HType::Float | HType::SizedFloat { .. })
}

fn is_int_ty(ty: &HType) -> bool {
    matches!(ty, HType::Int | HType::SizedInt { .. } | HType::Char)
}

/// Bit width of an integer type for saturation thresholds (usize/isize = 64).
fn int_bits(ty: &HType) -> u32 {
    match ty {
        HType::Char => 8,
        HType::SizedInt { bits: 0, .. } => 64,
        HType::SizedInt { bits, .. } => *bits as u32,
        _ => 64,
    }
}

/// `(2^(bits-1) as double-literal, MIN int-literal, MAX int-literal)` for a
/// signed integer type, used by the saturating float->int conversion.
fn sat_signed_lits(bits: u32) -> (&'static str, &'static str, &'static str) {
    match bits {
        8 => ("128.0", "(-128)", "127"),
        16 => ("32768.0", "(-32768)", "32767"),
        32 => ("2147483648.0", "(-2147483647 - 1)", "2147483647"),
        _ => ("9223372036854775808.0", "(-9223372036854775807LL - 1)", "9223372036854775807LL"),
    }
}

/// `(2^bits as double-literal, MAX uint-literal)` for an unsigned integer type.
fn sat_unsigned_lits(bits: u32) -> (&'static str, &'static str) {
    match bits {
        8 => ("256.0", "255"),
        16 => ("65536.0", "65535"),
        32 => ("4294967296.0", "4294967295U"),
        _ => ("18446744073709551616.0", "18446744073709551615ULL"),
    }
}

/// The minimum value of a signed integer type as a C literal, used by maka_divs
/// / maka_mods to detect the `MIN / -1` overflow.  Written without the bare
/// `INT*_MIN` token (which C would parse as `-(literal)`, itself overflowing).
fn signed_min_lit(ty: &HType) -> String {
    match shift_width(ty) {
        8 => "(-128)".to_string(),
        16 => "(-32768)".to_string(),
        32 => "(-2147483647 - 1)".to_string(),
        _ => "(-9223372036854775807LL - 1)".to_string(),
    }
}

/// Same-width unsigned C type, used to perform a signed left shift in the
/// unsigned domain before reinterpreting the bits back.
fn unsigned_ctype(ty: &HType) -> String {
    match ty {
        HType::Int => "uint64_t".into(),
        HType::SizedInt { bits: 0, .. } => "uintptr_t".into(),
        HType::SizedInt { bits, .. } => format!("uint{}_t", bits),
        HType::Char => "maka_char".into(),
        _ => "uint64_t".into(),
    }
}

/// literal non-zero divisor is provably safe so the check is elided.
fn needs_div_guard(op: HBinOp, lhs_ty: &HType, rhs: &HExpr) -> bool {
    if !matches!(op, HBinOp::Div | HBinOp::Mod) { return false; }
    if !matches!(lhs_ty, HType::Int | HType::SizedInt { .. } | HType::Char) { return false; }
    // A divisor that is a known nonzero literal cannot be /0; but a literal `-1`
    // can still overflow a signed `MIN`, so it still needs the guard.
    match &rhs.kind {
        HExprKind::LitInt(n) => *n == 0 || (*n == -1 && is_signed_int(lhs_ty)),
        _ => true,
    }
}

/// Emit an i64 literal as valid C.  Every value spells fine as `<n>LL` except
/// i64::MIN, whose magnitude (9223372036854775808) overflows a signed long long
/// when written as `-9223372036854775808LL`; spell it as `(-MAX - 1)` instead.
/// (Large unsigned bit patterns reach here as their signed reinterpretation,
/// e.g. u64::MAX as -1, and a cast to the target type restores the bits.)
fn c_int_lit(n: i64) -> String {
    if n == i64::MIN {
        "(maka_int)(-9223372036854775807LL - 1)".to_string()
    } else {
        format!("(maka_int){}LL", n)
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
    // A Maka identifier that is a C keyword but not a Maka keyword (`double`,
    // `void`, `long`, `struct`, `switch`, `static`, ...) would emit an illegal C
    // name and fail to compile.  Suffix `_` to disambiguate.  Done inside
    // `c_ident` so EVERY emission site (a declaration and each of its uses)
    // mangles identically.  Globals are already prefixed (`__maka_global__`) and
    // externs keep their raw ABI name, so neither routes through here.
    if is_c_keyword(&out) {
        out.push('_');
    }
    out
}

/// The C11 keyword set.  Maka's own keywords (`int`, `char`, `for`, ...) can
/// never be identifiers, so the overlap is harmless; the ones that matter are
/// the C keywords that ARE legal Maka identifiers (`double`, `void`, `long`,
/// `short`, `signed`, `unsigned`, `struct`, `union`, `switch`, `case`,
/// `default`, `do`, `goto`, `sizeof`, `static`, `register`, `volatile`,
/// `restrict`, `typedef`, `auto`).
/// The Maka source line of a statement, for `#line` stamping.  `None` for a
/// statement with no meaningful span (line 0 = synthesized).
fn hstmt_line(s: &HStmt) -> Option<u32> {
    let sp = match s {
        HStmt::Let { span, .. }
        | HStmt::Assign { span, .. }
        | HStmt::Return { span, .. }
        | HStmt::If { span, .. }
        | HStmt::While { span, .. }
        | HStmt::Break { span, .. }
        | HStmt::Continue { span, .. }
        | HStmt::ForC { span, .. }
        | HStmt::ForEach { span, .. }
        | HStmt::Propagate { span, .. } => *span,
        HStmt::Unsafe(_, span) => *span,
        HStmt::ExprStmt(e) => e.span,
        HStmt::Block(b) => b.span,
    };
    if sp.line == 0 { None } else { Some(sp.line) }
}

fn is_c_keyword(s: &str) -> bool {
    matches!(
        s,
        "auto" | "break" | "case" | "char" | "const" | "continue" | "default"
            | "do" | "double" | "else" | "enum" | "extern" | "float" | "for"
            | "goto" | "if" | "inline" | "int" | "long" | "register" | "restrict"
            | "return" | "short" | "signed" | "sizeof" | "static" | "struct"
            | "switch" | "typedef" | "union" | "unsigned" | "void" | "volatile"
            | "while" | "_Bool" | "_Complex" | "_Imaginary" | "_Alignas"
            | "_Alignof" | "_Atomic" | "_Generic" | "_Noreturn"
            | "_Static_assert" | "_Thread_local"
    )
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

/// The always-unique, always-safe C name of a local: `name_<id>`.  Used as the
/// fallback whenever a readable name would be ambiguous or is not available.
fn local_name_mangled(id: LocalId, name: &str) -> String {
    format!("{}_{}", c_ident(name), id.0)
}

/// Collect every local id that appears as an expression (a read or a write
/// PLACE) anywhere in the block.  Used to find `*Thread` handles that are never
/// referenced after declaration (so they are abandoned and can be auto-detached
/// at scope exit).  Exhaustive over sub-expression-bearing variants; a missed
/// variant would UNDER-count and risk over-reaping, so all are handled and only
/// genuine leaves fall through.
fn collect_local_reads_block(b: &HBlock, out: &mut std::collections::HashSet<u32>) {
    for s in &b.stmts { collect_local_reads_stmt(s, out); }
}
fn collect_local_reads_stmt(s: &HStmt, out: &mut std::collections::HashSet<u32>) {
    match s {
        HStmt::Let { init, .. } => collect_local_reads_expr(init, out),
        HStmt::Assign { place, value, .. } => { collect_local_reads_expr(place, out); collect_local_reads_expr(value, out); }
        HStmt::ExprStmt(e) => collect_local_reads_expr(e, out),
        HStmt::Return { value: Some(e), .. } | HStmt::Propagate { value: Some(e), .. } => collect_local_reads_expr(e, out),
        HStmt::Return { value: None, .. } | HStmt::Propagate { value: None, .. }
        | HStmt::Break { .. } | HStmt::Continue { .. } => {}
        HStmt::If { cond, then_b, else_b, .. } => {
            collect_local_reads_expr(cond, out);
            collect_local_reads_block(then_b, out);
            if let Some(b) = else_b { collect_local_reads_block(b, out); }
        }
        HStmt::While { cond, body, .. } => { collect_local_reads_expr(cond, out); collect_local_reads_block(body, out); }
        HStmt::ForC { init, cond, step, body, .. } => {
            collect_local_reads_stmt(init, out); collect_local_reads_expr(cond, out);
            collect_local_reads_stmt(step, out); collect_local_reads_block(body, out);
        }
        HStmt::ForEach { src, body, .. } => { collect_local_reads_expr(src, out); collect_local_reads_block(body, out); }
        HStmt::Block(b) | HStmt::Unsafe(b, _) => collect_local_reads_block(b, out),
    }
}
fn collect_local_reads_expr(e: &HExpr, out: &mut std::collections::HashSet<u32>) {
    use HExprKind::*;
    match &e.kind {
        Local(id) => { out.insert(id.0); }
        Field { base, .. } | ArrayToSlice { base, .. } => collect_local_reads_expr(base, out),
        Index { base, idx } => { collect_local_reads_expr(base, out); collect_local_reads_expr(idx, out); }
        Call { args, .. } | InlineCall { args, .. } => for a in args { collect_local_reads_expr(a, out); },
        CallIndirect { callee, args } => { collect_local_reads_expr(callee, out); for a in args { collect_local_reads_expr(a, out); } }
        Bin { lhs, rhs, .. } => { collect_local_reads_expr(lhs, out); collect_local_reads_expr(rhs, out); }
        Un { expr, .. } | Unwrap { expr, .. } | Cast { expr, .. } | CheckedCast { expr, .. }
        | DropWrite(expr) | DerefRef(expr) | HeapAlloc(expr) | Free(expr, _) | Transfer(expr)
        | SliceLen(expr) | EnumTag(expr) | DynHasVtbl(expr)
        | DynRewitness { value: expr, .. } => collect_local_reads_expr(expr, out),
        AddrOfRef { place, .. } => collect_local_reads_expr(place, out),
        Struct { fields, .. } | VariantCtor { fields, .. } => for (_, fe) in fields { collect_local_reads_expr(fe, out); },
        ArrayLit(es) => for e2 in es { collect_local_reads_expr(e2, out); },
        Closure { env_values, .. } => for v in env_values { collect_local_reads_expr(v, out); },
        Match { scrutinee, arms, .. } => {
            collect_local_reads_expr(scrutinee, out);
            for a in arms {
                if let Some(g) = &a.guard { collect_local_reads_expr(g, out); }
                collect_local_reads_block(&a.body, out);
                if let Some(v) = &a.value { collect_local_reads_expr(v, out); }
            }
        }
        // Leaves (no sub-expression): literals, ZeroInit, EnumVariant tag, FnRef,
        // GlobalRef.  None can reference a local.
        LitInt(..) | LitFloat(..) | LitBool(..) | LitChar(..) | LitStr(..) | LitNull | LitUnit
        | ZeroInit | EnumVariant { .. } | FnRef(..) | GlobalRef(..) => {}
    }
}

/// Collect the local id and declaring block of every `let` in the function, so
/// each abandoned `*Thread` handle is detached in the block it was declared in
/// (per loop iteration, not once at function exit).
fn collect_let_locals(b: &HBlock, out: &mut Vec<LocalId>) {
    for s in &b.stmts {
        match s {
            HStmt::Let { local, .. } => out.push(*local),
            HStmt::If { then_b, else_b, .. } => {
                collect_let_locals(then_b, out);
                if let Some(eb) = else_b { collect_let_locals(eb, out); }
            }
            HStmt::While { body, .. } | HStmt::ForC { body, .. } | HStmt::ForEach { body, .. }
            | HStmt::Block(body) | HStmt::Unsafe(body, _) => collect_let_locals(body, out),
            _ => {}
        }
    }
}
