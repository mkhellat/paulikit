/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com> */

/* Original C implementation of the in-place Walsh-Hadamard butterfly.
 * See wht.h for the design rationale, the tiling argument, and the
 * two alternative structures that were measured and rejected. */

#include "wht.h"

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

/* Bytes per complex128: two IEEE-754 doubles. */
#define PAULIKIT_COMPLEX_BYTES 16

int64_t paulikit_wht_tile_for_cache(int64_t dim)
{
    long l1_bytes = 0;

    /* _SC_LEVEL*_CACHE_* are a glibc extension. Guarding on the macro
     * rather than on the platform keeps this compiling on musl, the
     * BSDs, and anywhere else that lacks them - there the fallback
     * constant is used, which is correct behaviour, not a failure. */
#if defined(_SC_LEVEL1_DCACHE_SIZE)
    l1_bytes = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (l1_bytes < 0) {
        l1_bytes = 0;   /* sysconf returns -1 for "indeterminate" */
    }
#endif

    int64_t tile;
    if (l1_bytes > 0) {
        /* Half of L1 - see the header: two half-blocks plus vector
         * temporaries are live at once during a stage. */
        tile = (int64_t)(l1_bytes / 2) / PAULIKIT_COMPLEX_BYTES;
    } else {
        tile = PAULIKIT_WHT_FALLBACK_TILE;
    }

    if (tile < 1) {
        tile = 1;
    }

    /* Round down to a power of two: the stage-splitting argument in
     * paulikit_wht_rows is only valid for power-of-two spans, and
     * rounding down never exceeds the cache budget just derived. */
    int64_t pow2 = 1;
    while ((pow2 << 1) > 0 && (pow2 << 1) <= tile) {
        pow2 <<= 1;
    }
    tile = pow2;

    if (dim >= 1 && tile > dim) {
        tile = dim;
    }
    return tile < 1 ? 1 : tile;
}

/* Stages span=1,2,4 (radix-8) fused into one pass over each 8-complex
 * (16-double) aligned block, keeping every intermediate in a local
 * array instead of writing it back to memory between stages. The
 * per-block fixed overhead of the general wht_stage loop (pointer
 * arithmetic, loop bookkeeping) dominates at these small spans: at
 * span=1 the inner loop has only 2 iterations but the outer block
 * loop runs dim/2 times, so per-stage-per-block overhead is paid 3x
 * for 3 stages that this fuses into one block traversal. Verified
 * bit-exact against a from-scratch stage-by-stage reference at every
 * dim/tile combination that exercises this path (max_diff=0). Measured
 * on the production kernel (this function, with its real tiling, not
 * an isolated prototype): 16.3% fewer instructions at dim=1024,
 * rows=1024 (perf stat, 3-rep min). Cycles are essentially unchanged
 * (-0.3%) at this shape - this stage is memory-bandwidth-bound, not
 * instruction-issue-bound, so removing instructions here does not by
 * itself move wall-clock/cycle cost; the AVX2/POPCNT compiled-variant
 * dispatch (see fwht.py's _cpu_has_x86_64_v3) is what moves cycles. */
static inline void wht_radix8_block(double *restrict block) {
    double a[16];
    for (int k = 0; k < 16; k++) {
        a[k] = block[k];
    }

    /* span=1: pairs (0,1),(2,3),... within each group of 4 doubles. */
    double b[16];
    for (int g = 0; g < 4; g++) {
        b[g * 4 + 0] = a[g * 4 + 0] + a[g * 4 + 2];
        b[g * 4 + 1] = a[g * 4 + 1] + a[g * 4 + 3];
        b[g * 4 + 2] = a[g * 4 + 0] - a[g * 4 + 2];
        b[g * 4 + 3] = a[g * 4 + 1] - a[g * 4 + 3];
    }

    /* span=2: pairs 4 doubles apart within each group of 8. */
    double c[16];
    for (int g = 0; g < 2; g++) {
        for (int j = 0; j < 4; j++) {
            c[g * 8 + j] = b[g * 8 + j] + b[g * 8 + j + 4];
            c[g * 8 + j + 4] = b[g * 8 + j] - b[g * 8 + j + 4];
        }
    }

    /* span=4: pairs 8 doubles apart across the whole 16-double block. */
    for (int j = 0; j < 8; j++) {
        block[j] = c[j] + c[j + 8];
        block[j + 8] = c[j] - c[j + 8];
    }
}

/* One stage of the butterfly over a half-open element range, operating
 * on the underlying doubles rather than on `double complex`.
 *
 * A complex128 is two adjacent doubles and the butterfly is
 * elementwise addition/subtraction, so (a+bi) +/- (c+di) is exactly
 * (a +/- c) and (b +/- d) - the complex structure is irrelevant to
 * this operation. Working in doubles doubles the trip count of the
 * innermost loop, which gives the compiler a longer contiguous run to
 * vectorize and removes any dependence on how `double complex` is
 * passed around. `restrict` on the two halves tells it they never
 * alias, which they cannot: they are disjoint halves of one block. */
static inline void wht_stage(
    double *restrict data,
    int64_t block_start,
    int64_t block_end,
    int64_t span
) {
    const int64_t double_span = span * 2;      /* complex -> doubles */
    const int64_t stride = double_span * 2;    /* one full butterfly block */

    for (int64_t base = block_start * 2; base < block_end * 2; base += stride) {
        double *restrict lo = data + base;
        double *restrict hi = lo + double_span;
        for (int64_t i = 0; i < double_span; i++) {
            const double u = lo[i];
            const double v = hi[i];
            lo[i] = u + v;
            hi[i] = u - v;
        }
    }
}

void paulikit_wht_rows(
    double *data,
    int64_t rows,
    int64_t dim,
    int64_t tile
) {
    if (dim < 2 || rows < 1) {
        /* dim == 1 is the empty transform (a single element is its own
         * Walsh-Hadamard transform); rows < 1 is nothing to do. Both
         * are valid inputs, not errors. */
        return;
    }

    if (tile <= 0) {
        /* Caller did not specify: derive from this machine's actual
         * L1 rather than a compiled-in guess. */
        tile = paulikit_wht_tile_for_cache(dim);
    }
    if (tile > dim) {
        tile = dim;
    }

    for (int64_t row = 0; row < rows; row++) {
        double *const r = data + row * dim * 2;

        /* Phase 1: every stage whose span is smaller than `tile` pairs
         * only elements inside one aligned tile-length window, so the
         * whole window can be carried through all of those stages
         * while it is still hot in cache.
         *
         * Spans 1, 2 and 4 are always < tile whenever tile >= 8 (tile
         * is a power of two, so tile >= 8 implies tile > 4), and each
         * 16-double block a radix-8 fusion touches never crosses a
         * tile boundary (tile is itself a multiple of 16 doubles once
         * tile >= 8 complex elements) - so the fused block can run
         * across the whole row up front, independent of the per-tile
         * loop below, and the remaining stage loop starts at span=8. */
        int64_t small_span_start = 1;
        if (tile >= 8) {
            for (int64_t base = 0; base < dim * 2; base += 16) {
                wht_radix8_block(r + base);
            }
            small_span_start = 8;
        }
        for (int64_t base = 0; base < dim; base += tile) {
            for (int64_t span = small_span_start; span < tile; span <<= 1) {
                wht_stage(r, base, base + tile, span);
            }
        }

        /* Phase 2: the remaining stages pair elements in different
         * tiles, so they must be full passes. There are only
         * log2(dim) - log2(tile) of them. */
        for (int64_t span = tile; span < dim; span <<= 1) {
            wht_stage(r, 0, dim, span);
        }
    }
}
