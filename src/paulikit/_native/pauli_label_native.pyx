# cython: language_level=3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Cython binding for the serial pauli_label C kernel.

Optional: paulikit stays pip-installable without a C toolchain. Needs
only C and Cython >= 3.0 — no C++ and no oneTBB. The oneTBB-parallel
entry point lives in ``pauli_label_parallel_native`` when that module
is built.

Exposes:

- ``pauli_label(x_mask, z_mask, n_qubits)`` — single-term label.
- ``pauli_label_batch(x_masks, z_masks, n_qubits)`` — batch labels as
  a Python ``list[str]``.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint32_t, int64_t

cnp.import_array()

cdef extern from "pauli_label.h":
    void c_pauli_label "pauli_label"(uint32_t x_mask, uint32_t z_mask, int n_qubits, char *out)
    void c_pauli_label_batch "pauli_label_batch"(
        const uint32_t *x_masks,
        const uint32_t *z_masks,
        int64_t n_terms,
        int n_qubits,
        char *out,
    )


def pauli_label(unsigned int x_mask, unsigned int z_mask, int n_qubits):
    """Single-term IXYZ label, matching fwht.pauli_label exactly."""
    cdef bytes buf = bytes(n_qubits + 1)
    cdef char *out = buf
    c_pauli_label(x_mask, z_mask, n_qubits, out)
    return out[:n_qubits].decode("ascii")


def pauli_label_batch(cnp.ndarray[uint32_t, ndim=1] x_masks,
                       cnp.ndarray[uint32_t, ndim=1] z_masks,
                       int n_qubits):
    """Batch IXYZ labels for parallel arrays of (x, z) masks.

    Returns a Python list of str, one call into the C batch kernel
    regardless of how many terms there are.
    """
    if x_masks.shape[0] != z_masks.shape[0]:
        raise ValueError("x_masks and z_masks must have the same length")

    cdef int64_t n_terms = x_masks.shape[0]
    cdef cnp.ndarray[uint32_t, ndim=1] x_c = np.ascontiguousarray(x_masks, dtype=np.uint32)
    cdef cnp.ndarray[uint32_t, ndim=1] z_c = np.ascontiguousarray(z_masks, dtype=np.uint32)

    cdef bytes buf = bytes(n_terms * n_qubits)
    cdef char *out = buf
    c_pauli_label_batch(<uint32_t *>x_c.data, <uint32_t *>z_c.data, n_terms, n_qubits, out)

    result = [None] * n_terms
    cdef int64_t i
    for i in range(n_terms):
        result[i] = out[i * n_qubits: (i + 1) * n_qubits].decode("ascii")
    return result
