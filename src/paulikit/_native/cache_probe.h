/* Empirical cache-latency boundary probe.
 *
 * Portable C reimplementation of the pointer-chase microbenchmark
 * `configure --probe-cache-latency` already runs (per-architecture
 * raw assembly, see configure's own comments): each buffer size gets
 * a scrambled-stride traversal order (index = (index + prime) mod
 * n_elems - defeats simple sequential-stride prefetch detection, not
 * a full shuffle), a warm-up walk, then a timed pointer chase. Here
 * it is one portable C loop (compiled per-target by the build's own
 * C compiler) instead of three hand-written asm variants, since this
 * runs at Python-import/first-use time (a C toolchain always exists
 * by then), not at configure time (before any toolchain is known to
 * exist - configure's own reason for raw asm there).
 *
 * Why this exists at all rather than reading declared cache sizes:
 * `lscpu`'s "L2 1 MiB" was found to be an aggregate across cores, not
 * the per-core size a single-threaded chunk computation actually
 * sees (confirmed 256 KiB per-core on the machine that finding was
 * made on) - declared-topology sources are ambiguous in exactly this
 * way. An empirical probe measures what the hardware actually does,
 * sidestepping the ambiguity entirely.
 *
 * Timing source: raw hardware cycle counters (RDTSCP on x86_64,
 * CNTVCT_EL0 on aarch64, the `rdtime` CSR read on riscv64 - the same
 * three instructions configure's own asm probe uses), NOT
 * clock_gettime(). A first C implementation used clock_gettime and
 * found the middle cache-size region wildly noisy (the same 64 KiB
 * buffer read anywhere from 3ns to 42ns/access across runs) - traced
 * to this machine's `powersave` CPU-frequency governor swinging
 * 400 MHz<->3.3 GHz during the timed loop, an 8x range matching the
 * noise magnitude almost exactly. Hardware cycle counters are
 * immune to DVFS (the invariant TSC on x86_64 and its ARM/RISC-V
 * equivalents tick at a fixed reference rate regardless of the
 * core's actual clock speed) - confirmed by configure's own asm
 * probe getting clean, sharp boundaries on the very same noisy
 * hardware and load. Boundary detection only ever compares *ratios*
 * between rows (see cache_probe.pyx), so raw cycle counts need no
 * frequency calibration to be useful.
 *
 * Even with cycle counters, a second, distinct noise source remained:
 * occasional scheduler preemption landing mid-measurement on some
 * row, which - unlike DVFS - a single occurrence fully corrupts (the
 * whole timed loop's cycle delta includes however long the process
 * was descheduled). Two mitigations, both applied by
 * cache_probe_run(): (1) the probe pins itself to one CPU for its own
 * duration via sched_setaffinity (Linux; a no-op elsewhere - see
 * cache_probe.c), restoring the caller's original affinity mask
 * afterward, to reduce migration-driven noise specifically; (2) each
 * buffer size is measured `repeats` times (not just `reps` pointer
 * chases once), keeping the *minimum* cycle count across repeats - a
 * preemption can only inflate a reading (extra descheduled time adds
 * to the measured delta), never deflate it below the true cache
 * latency, so the minimum of a few repeats robustly rejects
 * preemption outliers without needing to detect them explicitly.
 */

#ifndef PAULIKIT_CACHE_PROBE_H
#define PAULIKIT_CACHE_PROBE_H

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One (buffer_size, cycles_per_access) measurement. */
typedef struct {
    size_t buffer_size_bytes;
    double cycles_per_access;
} cache_probe_sample;

/* Runs the pointer-chase probe over `n_sizes` buffer sizes, doubling
 * from `min_size_bytes` (inclusive). Writes up to `n_sizes` samples
 * into `out` (caller-allocated, must have room for n_sizes entries)
 * and returns the number actually written (fewer than n_sizes only if
 * an allocation fails partway through, e.g. a memory-constrained
 * container - the probe stops early rather than crashing). Each
 * sample uses `reps` timed pointer-chase iterations (300000 matches
 * configure's own probe's rep count).
 *
 * `repeats` (>= 1) repeated timed measurements are taken per buffer
 * size, keeping the minimum cycle count - see this header's own
 * docstring for why minimum-of-repeats robustly rejects preemption
 * outliers.
 *
 * Not thread-safe to call concurrently with itself (no shared mutable
 * state beyond the caller-provided output buffer and the process-wide
 * CPU affinity mask it temporarily changes, but see cache_probe.pyx
 * for why this is invoked at most once per process regardless). */
size_t cache_probe_run(
    size_t min_size_bytes,
    size_t n_sizes,
    int64_t reps,
    int repeats,
    cache_probe_sample *out
);

#ifdef __cplusplus
}
#endif

#endif /* PAULIKIT_CACHE_PROBE_H */
