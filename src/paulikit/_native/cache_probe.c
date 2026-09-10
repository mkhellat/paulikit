#include "cache_probe.h"

#include <float.h>
#include <stdlib.h>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#define PAULIKIT_HAVE_MMAP 1
#endif

/* CPU pinning: Linux-only (sched_setaffinity has no portable POSIX
 * equivalent - macOS/BSD have different, weaker affinity APIs not
 * worth chasing here). A no-op elsewhere: pin_to_one_cpu()/
 * restore_affinity() below become empty stubs, and the repeat-minimum
 * strategy (see cache_probe.h) is left to do the whole job of
 * rejecting preemption noise on those platforms. */
#if defined(__linux__)
#include <sched.h>

static int pin_to_one_cpu(cpu_set_t *saved_mask) {
    if (sched_getaffinity(0, sizeof(*saved_mask), saved_mask) != 0) {
        return 0; /* couldn't read the current mask - skip pinning */
    }
    cpu_set_t single;
    CPU_ZERO(&single);
    /* Pin to whichever CPU we're already running on, not CPU 0 -
     * avoids fighting an external affinity restriction (e.g. a Slurm
     * cgroup/cpuset only granting a subset of CPUs) that CPU 0 might
     * not even be inside. */
    int current_cpu = sched_getcpu();
    if (current_cpu < 0) {
        return 0;
    }
    CPU_SET(current_cpu, &single);
    return sched_setaffinity(0, sizeof(single), &single) == 0;
}

static void restore_affinity(const cpu_set_t *saved_mask) {
    sched_setaffinity(0, sizeof(*saved_mask), saved_mask);
}
#else
typedef int cpu_set_t; /* unused placeholder */
static int pin_to_one_cpu(cpu_set_t *saved_mask) {
    (void)saved_mask;
    return 0;
}
static void restore_affinity(const cpu_set_t *saved_mask) {
    (void)saved_mask;
}
#endif

/* Prime stride for the scrambled traversal, matching configure's own
 * probe exactly (104729 - a prime, chosen only to avoid a trivial
 * sequential/power-of-two stride a hardware prefetcher could learn).
 * index_{k+1} = (index_k + 104729) mod n_elems. */
static const uint64_t SCRAMBLE_STRIDE = 104729;

/* Hardware cycle counter, one implementation per architecture -
 * mirrors configure's own asm probe's choice of instruction exactly
 * (see cache_probe.h's docstring for why: immune to DVFS-driven
 * frequency scaling, unlike clock_gettime). */
#if defined(__x86_64__)
#include <x86intrin.h>
static inline uint64_t read_cycle_counter(void) {
    unsigned int aux;
    return __rdtscp(&aux);
}
#elif defined(__aarch64__)
static inline uint64_t read_cycle_counter(void) {
    uint64_t value;
    __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(value));
    return value;
}
#elif defined(__riscv) && __riscv_xlen == 64
static inline uint64_t read_cycle_counter(void) {
    uint64_t value;
    __asm__ __volatile__("rdtime %0" : "=r"(value));
    return value;
}
#else
/* No known cycle-counter instruction for this architecture - caller
 * (cache_probe.pyx) treats an all-zero-cycles result as "probe
 * unusable" and falls back to declared-size detection. */
static inline uint64_t read_cycle_counter(void) {
    return 0;
}
#endif

/* Builds a scrambled cyclic permutation over n_elems 8-byte slots in
 * buf: buf[i] holds the index of the next slot to visit. Every slot
 * is visited exactly once per full cycle (a single cycle covering all
 * n_elems slots, not several disjoint short cycles) because gcd(stride,
 * n_elems) is forced to 1 by construction below. */
static void build_scrambled_chain(uint64_t *buf, uint64_t n_elems) {
    uint64_t index = 0;
    for (uint64_t i = 0; i < n_elems; i++) {
        uint64_t next = (index + SCRAMBLE_STRIDE) % n_elems;
        buf[index] = next;
        index = next;
    }
}

/* One timed pointer-chase pass over an already-built, already-warmed
 * chain. Returns cycles for `reps` accesses. */
static double timed_pass(uint64_t *buf, int64_t reps) {
    volatile uint64_t cursor = 0;
    uint64_t c0 = read_cycle_counter();
    for (int64_t i = 0; i < reps; i++) {
        cursor = buf[cursor];
    }
    uint64_t c1 = read_cycle_counter();
    (void)cursor;
    return (double)(c1 - c0);
}

size_t cache_probe_run(
    size_t min_size_bytes,
    size_t n_sizes,
    int64_t reps,
    int repeats,
    cache_probe_sample *out
) {
    if (repeats < 1) {
        repeats = 1;
    }

    cpu_set_t saved_mask;
    int pinned = pin_to_one_cpu(&saved_mask);

    size_t written = 0;
    size_t size = min_size_bytes;

    for (size_t s = 0; s < n_sizes; s++, size <<= 1) {
        uint64_t n_elems = (uint64_t)(size / sizeof(uint64_t));
        if (n_elems < 2) {
            continue; /* too small to form a >1-element cycle */
        }

#if PAULIKIT_HAVE_MMAP
        uint64_t *buf = (uint64_t *)mmap(
            NULL, size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
        );
        if (buf == MAP_FAILED) {
            break; /* stop early rather than crash - see header */
        }
#else
        uint64_t *buf = (uint64_t *)malloc(size);
        if (buf == NULL) {
            break;
        }
#endif

        build_scrambled_chain(buf, n_elems);

        /* Warm-up walk: 3x the buffer's own element count, matching
         * configure's own probe (enough to bring the whole working
         * set through the cache hierarchy before timing starts).
         *
         * A second, separate call to probe_cache_boundaries() in the
         * same process, made shortly after a first call already
         * walked a much larger buffer, can see this warm-up as
         * insufficient - stale TLB/cache state from the earlier large
         * buffer inflates the very next call's small-buffer readings
         * (investigated and characterized, not fixed here).
         * Deliberately NOT changed to compensate: the only shipped
         * caller, autotune.recommended_chunk_size, calls this at most
         * ONCE per process (module-level cache, see autotune.py) -
         * every real invocation is a first, isolated call, which was
         * independently confirmed reliable (10/10 fresh-process
         * trials in that same investigation). A larger warm-up floor
         * was tried and found to only partially help (diminishing
         * returns, rising runtime cost, never fully eliminated) for a
         * scenario production code cannot reach - reverted rather
         * than shipped as an unnecessary cost. */
        volatile uint64_t warm_cursor = 0;
        for (uint64_t i = 0; i < n_elems * 3; i++) {
            warm_cursor = buf[warm_cursor];
        }
        (void)warm_cursor;

        double min_cycles = DBL_MAX;
        for (int r = 0; r < repeats; r++) {
            double cycles = timed_pass(buf, reps);
            if (cycles < min_cycles) {
                min_cycles = cycles;
            }
        }

        out[written].buffer_size_bytes = size;
        out[written].cycles_per_access = min_cycles / (double)reps;
        written++;

#if PAULIKIT_HAVE_MMAP
        munmap(buf, size);
#else
        free(buf);
#endif
    }

    if (pinned) {
        restore_affinity(&saved_mask);
    }

    return written;
}
