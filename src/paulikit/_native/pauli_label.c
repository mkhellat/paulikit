#include "pauli_label.h"

/* Indexed by (x_j << 1) | z_j, matching the Python reference's
 * `letters[(xj, zj)]` dict: (0,0)='I', (0,1)='Z', (1,0)='X', (1,1)='Y'. */
static const char LETTER_TABLE[4] = {'I', 'Z', 'X', 'Y'};

void pauli_label(uint32_t x_mask, uint32_t z_mask, int n_qubits, char *out) {
    for (int qubit = 0; qubit < n_qubits; qubit++) {
        int bit = n_qubits - 1 - qubit;
        unsigned xj = (x_mask >> bit) & 1u;
        unsigned zj = (z_mask >> bit) & 1u;
        out[qubit] = LETTER_TABLE[(xj << 1) | zj];
    }
    out[n_qubits] = '\0';
}

void pauli_label_batch(
    const uint32_t *x_masks,
    const uint32_t *z_masks,
    int64_t n_terms,
    int n_qubits,
    char *out
) {
    for (int64_t term = 0; term < n_terms; term++) {
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
