# cython: language_level=3
"""Cython binding for the pauli_label C kernel - paulikit's optional
compiled fast path.

Originally built and benchmarked standalone as part of a four-binding
comparison across alternative binding approaches (Cython won on both
speed and effort). This copy is the one actually shipped and wired
into ``paulikit.algorithms.fwht.fwht_pauli_terms``.

This extension is OPTIONAL: paulikit must remain pip-installable
without a C++ toolchain or oneTBB (the NumPy/SciPy prebuilt-wheel
model is the eventual packaging target, not yet adopted). If this
module fails to build or import, callers fall back to the
pure-Python ``pauli_label`` loop with a visible warning - see
``fwht.py``. Do not make this import a hard requirement anywhere in
the package.

Exposes two entry points mirroring the pure-Python reference's two
use cases:

- ``pauli_label(x_mask, z_mask, n_qubits)`` - single-term, drop-in
  replacement for ``paulikit.algorithms.fwht.pauli_label``.
- ``pauli_label_batch(x_masks, z_masks, n_qubits)`` - given two
  equal-length NumPy uint32 arrays, returns a Python list of label
  strings, calling into the C batch kernel once rather than once per
  term (avoiding per-call FFI overhead at 1M+ terms is the point of
  the batch kernel).
- ``pauli_label_batch_parallel(x_masks, z_masks, n_qubits)`` - same
  contract as ``pauli_label_batch``, but calls the oneTBB-parallelized
  C++ kernel instead of the serial one. Not currently used by
  ``fwht_pauli_terms`` - parallelizing this specific loop was found
  to barely help end to end, since Python str construction (not the
  C loop) dominates wall-clock time - kept available for
  completeness, not wired in by default.
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

cdef extern from "pauli_label_parallel.h":
    void c_pauli_label_batch_parallel "pauli_label_batch_parallel"(
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


def pauli_label_batch_parallel(cnp.ndarray[uint32_t, ndim=1] x_masks,
                                cnp.ndarray[uint32_t, ndim=1] z_masks,
                                int n_qubits):
    """Same contract as pauli_label_batch, using the oneTBB-parallel
    C++ kernel instead of the serial C one."""
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
