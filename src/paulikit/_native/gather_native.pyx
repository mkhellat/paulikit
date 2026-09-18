# cython: language_level=3
"""Cython binding for the fully-dense per-chunk XOR-gather kernel.

Replaces the NumPy fancy-index gather
(``operator[p_indices, q_range]``) paulikit.algorithms.fwht's
fully-dense fast path used for
``gathered_chunk[row, q] = operator[x_row ^ q, q]`` with a tight C
loop over the underlying doubles. See ``gather.h`` for the design.

Optional, like the other extensions here: ``fwht.py`` falls back to
the NumPy fancy-index gather if this module is absent, with identical
results.

The GIL is released around the kernel call - it touches only the
buffers handed to it and no Python objects.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

cnp.import_array()

cdef extern from "gather.h":
    void paulikit_gather_dense_chunk(
        const double *operator_data, int64_t dim, int64_t chunk_start,
        int64_t rows, double *out_data
    ) nogil


def gather_dense_chunk(cnp.ndarray operator, int64_t chunk_start, int64_t rows):
    """Direct XOR-gather of one chunk from a fully dense operator.

    Args:
        operator: C-contiguous 2-D ``complex128`` array of shape
            ``(dim, dim)`` with ``dim`` a power of two, every cell
            assumed nonzero (the caller's responsibility - see
            ``parallel_decompose_arrays``'s own ``assume_dense``
            docstring for what "assumed" means here: this kernel
            reads every cell regardless, so it is correct for ANY
            actual sparsity pattern, just not necessarily the fastest
            choice for one). Read only.
        chunk_start: First ``x`` value to gather (``0 <= chunk_start``).
        rows: Number of consecutive ``x`` values to gather
            (``chunk_start + rows <= dim``).

    Returns:
        A new ``(rows, dim)`` ``complex128`` array with
        ``out[row, q] == operator[(chunk_start + row) ^ q, q]``.

    Raises:
        ValueError: If ``operator`` is not 2-D complex128
            C-contiguous, not square, ``dim`` is not a power of two,
            or the requested chunk falls outside ``[0, dim)``.
    """
    if operator.ndim != 2:
        raise ValueError(f"expected a 2-D operator, got {operator.ndim}-D")
    if operator.dtype != np.complex128:
        raise ValueError(f"expected complex128, got {operator.dtype}")
    if not operator.flags["C_CONTIGUOUS"]:
        raise ValueError("operator must be C-contiguous")

    cdef int64_t dim = operator.shape[0]
    cdef int64_t dim1 = operator.shape[1]
    if dim1 != dim:
        raise ValueError(
            f"operator must be square, got shape ({dim}, {dim1})"
        )
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"dim must be a power of two, got {dim}")
    if chunk_start < 0 or rows < 0 or chunk_start + rows > dim:
        raise ValueError(
            f"chunk [{chunk_start}, {chunk_start + rows}) out of range "
            f"for dim={dim}"
        )

    cdef cnp.ndarray out = np.empty((rows, dim), dtype=np.complex128)
    if rows == 0 or dim == 0:
        return out

    cdef const double *op_data = <const double *> cnp.PyArray_DATA(operator)
    cdef double *out_data = <double *> cnp.PyArray_DATA(out)

    with nogil:
        paulikit_gather_dense_chunk(op_data, dim, chunk_start, rows, out_data)

    return out
