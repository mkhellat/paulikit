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
         * while it is still hot in cache. */
        for (int64_t base = 0; base < dim; base += tile) {
            for (int64_t span = 1; span < tile; span <<= 1) {
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
