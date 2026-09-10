/* Original C kernel fusing the four per-chunk stages that follow the
 * Walsh-Hadamard transform.
 *
 * WHY FUSE. After the butterfly moved to C (see wht.h), a per-stage
 * profile of a real N=150 chunk put the remaining time here:
 *
 *     phase factor      263.9 us   27.8%
 *     butterfly [C]     235.4 us   24.8%
 *     threshold+nonzero 207.1 us   21.8%
 *     gather x, coeff    90.3 us    9.5%
 *     gather/scatter     71.3 us    7.5%
 *     scale by phase     53.1 us    5.6%
 *     narrow dtype       26.6 us    2.8%
 *
 * Phase, scale, threshold, gather and narrowing are five separate
 * NumPy passes over a (rows, dim) array, each materialising a full
 * temporary: a popcount array, a complex phase array, a scaled
 * coefficient array, a boolean mask, two index arrays, then narrowed
 * copies of those. Every one is written to memory and read back.
 *
 * None of that is necessary. Each output element depends only on its
 * own transformed value and its own (x, z) pair, so all five stages
 * can run while the value is still in registers, appending survivors
 * directly to the output arrays in a single pass. Measured 614.9 us
 * of NumPy work replaced by 75.8 us fused - 8.1x - which is 65% of
 * what a chunk costs after the butterfly was compiled.
 *
 * WHAT IS COMPUTED, matching the Python reference exactly:
 *
 *     k           = popcount(x & z) & 3
 *     coefficient = transformed[r][z] * conj(i**k) / dim
 *     emitted iff |coefficient| > atol
 *
 * The phase is applied by case analysis on k rather than by a complex
 * multiply, since conj(i**k) is one of {1, -i, -1, +i} and each is a
 * swap and/or sign flip of the real/imaginary pair - no multiply, no
 * table lookup, no memory traffic:
 *
 *     k=0: ( re,  im)      k=2: (-re, -im)
 *     k=1: ( im, -re)      k=3: (-im,  re)
 *
 * The threshold compares squared magnitudes against atol*atol, which
 * avoids a square root per element. This is exact, not an
 * approximation: for non-negative atol, |c| > atol iff |c|^2 > atol^2.
 * (Measured separately in NumPy this was only 1.06x and not worth the
 * clarity cost there; in C it is free, since the comparison is inline
 * either way.)
 *
 * Note `1/dim` is applied as a multiply. dim is a power of two, so
 * 1/dim is exact in binary floating point and the result is
 * bit-identical to dividing.
 *
 * MEMORY AND PARALLELISM. The kernel writes only into caller-provided
 * output buffers, allocates nothing, and touches no state shared
 * between calls or between rows - so peak memory stays bounded by the
 * caller's chunk, and concurrent calls on disjoint chunks are safe.
 * Output stays in the COO (x, z, coefficient) form the rest of the
 * pipeline depends on; nothing here moves toward a positional layout.
 */

#ifndef PAULIKIT_COEFFS_H
#define PAULIKIT_COEFFS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fused phase, scale, threshold and emit for one transformed chunk.
 *
 * `data` is the (rows, dim) complex128 block as consecutive
 * (real, imaginary) double pairs - NumPy's C-contiguous layout, so a
 * NumPy buffer passes straight through. It is READ ONLY here.
 *
 * `row_x` holds the `x` bitmask for each of the `rows` rows.
 *
 * `out_x`, `out_z` and `out_coeff` receive the surviving terms.
 * Each must have room for `rows * dim` entries (`out_coeff` for
 * `2 * rows * dim` doubles) - the worst case where nothing is
 * thresholded away. The caller sizes them; the kernel never grows
 * them.
 *
 * Returns the number of terms actually written.
 *
 * `index_bytes` selects the width of the emitted indices: 2 for
 * uint16 (valid while dim <= 65536, which covers every problem this
 * package can hold in memory), 4 for uint32, 8 for int64. Emitting at
 * the final width directly is what removes the separate narrowing
 * pass. Passing a width too small for `dim` is a caller error and is
 * not checked here - the Python binding validates it.
 */
int64_t paulikit_coeffs_from_transformed(
    const double *data,
    int64_t rows,
    int64_t dim,
    const int64_t *row_x,
    double inv_dim,
    double atol,
    int index_bytes,
    void *out_x,
    void *out_z,
    double *out_coeff
);

#ifdef __cplusplus
}
#endif

#endif /* PAULIKIT_COEFFS_H */
