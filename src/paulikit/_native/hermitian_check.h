/* Original C kernel for the Hermiticity-violation check on a chunk's
 * surviving coefficients.
 *
 * WHY THIS EXISTS. `_check_hermitian_violation` (fwht.py) does this
 * as four separate NumPy passes: `np.abs(coefficient_values)`,
 * `np.abs(coefficient_values.imag)`, `np.maximum(atol, 1e-6*c_abs)`,
 * then the comparison and `.any()` - each materializing a full-size
 * temporary array, each a separate pass over memory. Isolated
 * measurement (perf stat instructions:u, with the test array's own
 * construction cost correctly subtracted out) found a single fused
 * pass could cut this to as little as 41% of the 4-pass cost - the
 * gap between "1 comparison + 1 any()" and "4 materializing passes"
 * over the same data. Structurally the same rationale as `gather.c`
 * and `coeffs.c`: NumPy's per-stage temporaries are the cost, not the
 * comparison itself.
 *
 * WHAT IS COMPUTED, matching the Python reference exactly:
 *
 *     violation iff |im| > max(atol, 1e-6 * |c|)
 *
 * where `|c|` is the FULL complex magnitude (not `|re|`) - matching
 * `_build_real_terms`'s identical check exactly, which this function
 * is the array-yielding path's counterpart to. The squared-magnitude
 * form avoids a square root per element, exact (not approximate) for
 * non-negative atol: `|im| > max(atol, 1e-6*|c|)` iff
 * `im^2 > max(atol^2, 1e-12*|c|^2)`, and `|c|^2 = re^2 + im^2` needs
 * no square root either.
 *
 * MEMORY AND PARALLELISM. Read-only over `data`; allocates nothing;
 * touches no state shared between calls. Stops at the FIRST violation
 * found (matching the Python reference's "labels only the single
 * offending term" contract) rather than always scanning to the end -
 * a real, if input-dependent, additional saving on top of the
 * pass-count reduction, though not counted in the ~41% headline
 * figure above (that used a violation-free array, this kernel's own
 * worst case).
 */

#ifndef PAULIKIT_HERMITIAN_CHECK_H
#define PAULIKIT_HERMITIAN_CHECK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Scans `n` complex128 coefficients (as consecutive real/imaginary
 * double pairs - NumPy's C-contiguous complex128 layout) for the
 * first Hermiticity violation.
 *
 * `data` points to `n` complex numbers. `atol` is the same tolerance
 * `_check_hermitian_violation`'s Python reference takes.
 *
 * Returns the 0-based index of the first violating element, or -1 if
 * none. `n < 1` always returns -1 (no elements, no violation).
 */
int64_t paulikit_first_hermitian_violation(
    const double *data,
    int64_t n,
    double atol
);

#ifdef __cplusplus
}
#endif

#endif /* PAULIKIT_HERMITIAN_CHECK_H */
