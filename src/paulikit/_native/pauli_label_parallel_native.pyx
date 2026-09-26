# cython: language_level=3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Cython binding for the oneTBB-parallel pauli_label C++ kernel.

Built only when C++ and oneTBB are available. The serial path is
``pauli_label_native`` and does not depend on this module. End-to-end
label APIs that return ``list[str]`` spend most of their time in
Python string construction, so this parallel fill is opt-in and not
the wheel default.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint32_t, int64_t

cnp.import_array()

cdef extern from "pauli_label_parallel.h":
    void c_pauli_label_batch_parallel "pauli_label_batch_parallel"(
        const uint32_t *x_masks,
        const uint32_t *z_masks,
        int64_t n_terms,
        int n_qubits,
        char *out,
    )


def pauli_label_batch_parallel(cnp.ndarray[uint32_t, ndim=1] x_masks,
                                cnp.ndarray[uint32_t, ndim=1] z_masks,
                                int n_qubits):
    """Same contract as ``pauli_label_native.pauli_label_batch``."""
    if x_masks.shape[0] != z_masks.shape[0]:
        raise ValueError("x_masks and z_masks must have the same length")

    cdef int64_t n_terms = x_masks.shape[0]
    cdef cnp.ndarray[uint32_t, ndim=1] x_c = np.ascontiguousarray(x_masks, dtype=np.uint32)
    cdef cnp.ndarray[uint32_t, ndim=1] z_c = np.ascontiguousarray(z_masks, dtype=np.uint32)

    cdef bytes buf = bytes(n_terms * n_qubits)
    cdef char *out = buf
    c_pauli_label_batch_parallel(<uint32_t *>x_c.data, <uint32_t *>z_c.data, n_terms, n_qubits, out)

    result = [None] * n_terms
    cdef int64_t i
    for i in range(n_terms):
        result[i] = out[i * n_qubits: (i + 1) * n_qubits].decode("ascii")
    return result
