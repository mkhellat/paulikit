/* oneTBB-parallelized variant of pauli_label_batch.
 * Each term's label is independent (see pauli_label.h's
 * pauli_label_batch docstring), so this is embarrassingly parallel -
 * tbb::parallel_for over terms, no synchronization needed since every
 * thread writes to a disjoint slice of `out`.
 *
 * C++ (not C) because oneTBB is a C++ template library with no
 * portable C API; extern "C" linkage keeps the exported symbol
 * callable from the Cython binding (or any other C-ABI caller)
 * exactly like the serial pauli_label_batch.
 */

#ifndef PAULIKIT_PAULI_LABEL_PARALLEL_H
#define PAULIKIT_PAULI_LABEL_PARALLEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void pauli_label_batch_parallel(
    const uint32_t *x_masks,
    const uint32_t *z_masks,
    int64_t n_terms,
    int n_qubits,
    char *out
);

#ifdef __cplusplus
}
#endif

#endif /* PAULIKIT_PAULI_LABEL_PARALLEL_H */
