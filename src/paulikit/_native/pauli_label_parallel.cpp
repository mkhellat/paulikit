#include "pauli_label_parallel.h"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

/* Same letter table as pauli_label.c - duplicated rather than shared
 * across the C/C++ boundary to keep this file self-contained and the
 * serial kernel (pauli_label.c) untouched as the correctness
 * baseline. */
static const char LETTER_TABLE[4] = {'I', 'Z', 'X', 'Y'};

void pauli_label_batch_parallel(
    const uint32_t *x_masks,
    const uint32_t *z_masks,
    int64_t n_terms,
    int n_qubits,
    char *out
) {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, n_terms),
        [=](const tbb::blocked_range<int64_t> &range) {
            for (int64_t term = range.begin(); term != range.end(); ++term) {
                uint32_t x_mask = x_masks[term];
                uint32_t z_mask = z_masks[term];
                char *dest = out + term * (int64_t)n_qubits;
                for (int qubit = 0; qubit < n_qubits; qubit++) {
                    int bit = n_qubits - 1 - qubit;
                    unsigned xj = (x_mask >> bit) & 1u;
                    unsigned zj = (z_mask >> bit) & 1u;
                    dest[qubit] = LETTER_TABLE[(xj << 1) | zj];
                }
            }
        }
    );
}
