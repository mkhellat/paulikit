/* Original C port of paulikit.algorithms.fwht.pauli_label.
 *
 * Converts an (x, z) symplectic bitmask pair to an IXYZ Pauli-string
 * label, matching the exact convention of the pure-Python reference
 * in src/paulikit/algorithms/fwht.py: for qubit j (0-indexed from the
 * left/most-significant end of the label string), read bit position
 * (n_qubits - 1 - j) of both x_mask and z_mask, then map
 * (x_j, z_j) -> (0,0)='I', (1,0)='X', (0,1)='Z', (1,1)='Y'.
 *
 * Scoping rationale: pauli_label was confirmed (2026-08-16 profiling)
 * to be the dominant cost in fwht_pauli_terms, not the FWHT math
 * itself.
 */

#ifndef PAULIKIT_PAULI_LABEL_H
#define PAULIKIT_PAULI_LABEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Writes exactly n_qubits ASCII characters (one of 'I','X','Y','Z')
 * followed by a null terminator into `out`. Caller owns `out` and
 * must ensure it has room for at least (n_qubits + 1) bytes. Does not
 * allocate. `n_qubits` must be in [1, 32] (x_mask/z_mask are 32-bit);
 * callers are trusted not to violate this (see module docstring
 * convention: validated once at the NumPy-array boundary in
 * fwht_pauli_coefficients, not re-validated per term). */
void pauli_label(uint32_t x_mask, uint32_t z_mask, int n_qubits, char *out);

/* Batch entry point: computes pauli_label for n_terms (x, z) pairs in
 * one call, writing each label's n_qubits characters (no null
 * terminators between entries) contiguously into `out`, which must
 * have room for at least (n_terms * n_qubits) bytes. Batching the
 * call itself is the point: 1.26M individual Python->C calls would
 * just move the per-call overhead bottleneck rather than remove it. */
void pauli_label_batch(
    const uint32_t *x_masks,
    const uint32_t *z_masks,
    int64_t n_terms,
    int n_qubits,
    char *out
);

#ifdef __cplusplus
}
#endif

#endif /* PAULIKIT_PAULI_LABEL_H */
