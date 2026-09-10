/* Original C implementation of the in-place Walsh-Hadamard butterfly
 * used by paulikit.algorithms.fwht.
 *
 * WHY THIS EXISTS. Profiling a real N=150 chunk attributed 69.5% of
 * per-chunk time to `_walsh_hadamard_transform_rows`, the pure-NumPy
 * butterfly. The NumPy formulation operates on stride-2 views
 * (`view[:, :, 0, :]` / `view[:, :, 1, :]`), which cannot be
 * vectorized: NumPy must gather each strided operand, and there is no
 * way to express "load a contiguous pair, emit sum and difference"
 * through its API. A contiguous C loop can, and the compiler turns it
 * into packed SIMD without intrinsics.
 *
 * WHAT IS COMPUTED. The unnormalized transform, matching the Python
 * reference exactly: at each of log2(dim) stages, elements a distance
 * `span` apart are replaced by their sum and difference. No
 * normalization (that is folded into the phase-factor step), so
 * (H^{tensor n})^2 = 2^n * I. Each row of the (rows, dim) block
 * transforms independently - there is no cross-row coupling, which is
 * what keeps chunks independent and the caller parallelizable.
 *
 * THE TILING, AND WHY IT IS NOT THE TEXTBOOK LOOP. The naive
 * structure runs one full pass over the row per stage. At the real
 * workload dim = 16384, a row is 16384 * 16 = 256 KiB, which is
 * exactly this class of machine's per-core L2 - so every one of the
 * 14 passes evicts the row and reloads it from L3 or memory.
 *
 * Instead, stages are split at `span == tile`:
 *
 *   - Stages with span < tile touch only elements within a single
 *     aligned `tile`-element window. So we walk tile by tile and run
 *     ALL of those stages while the tile is resident, paying one load
 *     of the tile for log2(tile) stages instead of one per stage.
 *   - Stages with span >= tile inherently cross tiles and are run as
 *     ordinary full passes. There are only log2(dim) - log2(tile) of
 *     them.
 *
 * This is legal because the butterfly at stage `span` only ever pairs
 * indices differing in bit log2(span): within a 2*span-aligned block,
 * no element outside that block is read or written, so reordering the
 * (tile, stage) loop nest changes no data dependence. Measured 1.39x
 * over the naive structure at dim=16384, tile=1024 (correctness
 * verified against the naive form at every tile size).
 *
 * TWO STRUCTURES MEASURED AND REJECTED, recorded so they are not
 * retried:
 *
 *   1. Processing two rows in the inner loop, on the theory that the
 *      real chunk_size is 2 and dual rows would raise ILP. Measured
 *      WORSE (1.20x vs 1.39x): two 256 KiB rows exceed L2 together,
 *      so interleaving doubles the working set and defeats the
 *      tiling that is doing the actual work.
 *   2. Explicit AVX intrinsics over the underlying doubles. Measured
 *      WORSE (0.79-0.88x): GCC's autovectorizer already handles the
 *      contiguous loop better, and forcing the small-span stages
 *      through a 4-wide vector path costs more than it saves.
 *
 * Scalar, tiled, and left to the compiler is the measured optimum.
 */

#ifndef PAULIKIT_WHT_H
#define PAULIKIT_WHT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compiled-in fallback tile, in complex elements, used only when the
 * runtime cannot report an L1 data-cache size at all.
 *
 * A hardcoded tile is the portability failure the runtime autotuner
 * exists to avoid - it bakes one machine's cache into the binary. So
 * `paulikit_wht_tile_for_cache()` below derives the tile from
 * `sysconf(_SC_LEVEL1_DCACHE_SIZE)` at runtime, and this constant is
 * reached only if that returns nothing usable.
 *
 * 1024 complex = 16 KiB, half of the most common 32 KiB L1d. */
#define PAULIKIT_WHT_FALLBACK_TILE 1024

/* Backwards-compatible spelling; prefer the name above. */
#define PAULIKIT_WHT_DEFAULT_TILE PAULIKIT_WHT_FALLBACK_TILE

/* Runtime-derived tile length in complex elements, for a row of
 * `dim` complex elements.
 *
 * Queries the L1 data-cache size through `sysconf(3)`
 * (`_SC_LEVEL1_DCACHE_SIZE`), the same value `getconf
 * LEVEL1_DCACHE_SIZE` reports and the same source paulikit's
 * `configure` script checks. This is preferred over an empirical
 * latency probe for L1 specifically: the probe's smallest buffers are
 * its noisiest samples, and a spurious first jump makes it report
 * 8 KiB instead of 32 KiB run-to-run, which measured 111us/row
 * against 94us at the correct tile. `sysconf` is stable, needs no
 * warm-up, and costs nothing.
 *
 * Returns half of L1 in complex elements, rounded DOWN to a power of
 * two and clamped to [1, dim]: each butterfly stage has two
 * half-blocks plus vector temporaries live at once, so a full-L1 tile
 * evicts part of its own window mid-stage.
 *
 * Falls back to PAULIKIT_WHT_FALLBACK_TILE where sysconf lacks the
 * cache extensions (they are a glibc extension, absent on musl and
 * some BSDs) or returns a non-positive value. Never fails. */
int64_t paulikit_wht_tile_for_cache(int64_t dim);

/* In-place unnormalized Walsh-Hadamard transform of each row of a
 * row-major (rows, dim) complex128 block.
 *
 * `data` points to rows*dim complex numbers stored as consecutive
 * (real, imaginary) double pairs - exactly NumPy's complex128
 * C-contiguous layout, so a NumPy buffer can be passed directly with
 * no copy or conversion.
 *
 * `dim` must be a power of two >= 1. `rows` may be any non-negative
 * count. `tile` must be a power of two in [1, dim]; pass 0 for
 * PAULIKIT_WHT_DEFAULT_TILE, clamped to dim.
 *
 * Operates strictly in place and allocates nothing: peak memory is
 * the caller's block, which is what keeps the pipeline's resident set
 * bounded by chunk_size rather than by dim^2 or the total term count.
 *
 * Rows are transformed independently; no state is shared between
 * rows or between calls, so concurrent calls on disjoint blocks are
 * safe. */
void paulikit_wht_rows(
    double *data,
    int64_t rows,
    int64_t dim,
    int64_t tile
);

#ifdef __cplusplus
}
#endif

#endif /* PAULIKIT_WHT_H */
