// Minimal user-provided runtime that a real kernel would replace with
// its own implementation.  Bump allocator, no-op free, halt-on-panic.
#include <stdint.h>
#include <stddef.h>

static unsigned char heap[4096];
static size_t heap_off = 0;

void* __maka_alloc(size_t sz) {
    sz = (sz + 7) & ~(size_t)7;
    if (heap_off + sz > sizeof(heap)) return 0;
    void* p = &heap[heap_off];
    heap_off += sz;
    return p;
}
void __maka_free(void* p) { (void)p; }
void __maka_panic(const char* msg) { (void)msg; for (;;) {} }
void __maka_log_int(int64_t v) { (void)v; }
void __maka_log_str(const char* s) { (void)s; }

extern void maka_main(void);
void _start(void) { maka_main(); for (;;) {} }
