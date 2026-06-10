#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

typedef int64_t maka_int;

struct counter_arg { maka_int* counter; maka_int iters; };

static void* maka_thread_entry(void* p) {
    struct counter_arg* a = (struct counter_arg*)p;
    for (maka_int i = 0; i < a->iters; i++) {
        __atomic_fetch_add(a->counter, 1, __ATOMIC_SEQ_CST);
    }
    return NULL;
}

void* maka_thread_spawn_counter(maka_int* counter, maka_int iters) {
    pthread_t* tid = (pthread_t*)malloc(sizeof(pthread_t));
    struct counter_arg* a = (struct counter_arg*)malloc(sizeof(struct counter_arg));
    a->counter = counter;
    a->iters = iters;
    pthread_create(tid, NULL, maka_thread_entry, a);
    return tid;
}

void maka_thread_join(void* handle) {
    pthread_t* tid = (pthread_t*)handle;
    void* ret = NULL;
    pthread_join(*tid, &ret);
    free(tid);
}
