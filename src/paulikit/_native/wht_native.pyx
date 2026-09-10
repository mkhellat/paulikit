# cython: language_level=3
"""Cython binding for the Walsh-Hadamard butterfly C kernel -
paulikit's optional compiled fast path for the transform itself.

Profiling a real N=150 chunk attributed 69.5% of per-chunk
time to the pure-NumPy butterfly, which operates on stride-2 views
that NumPy cannot vectorize. This binding exposes the contiguous C
loop in ``wht.c``; see ``wht.h`` for the tiling design and the two
alternative structures that were measured and rejected.

Like ``pauli_label_native``, this extension is OPTIONAL. If it fails
to build or import, ``fwht.py`` falls back to the pure-Python
butterfly with identical results. Do not make this import a hard
requirement anywhere in the package.

The GIL is released around the kernel call: the transform touches
only the caller's buffer and no Python objects, so worker threads (and
any future threaded caller) are not serialized on it.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

cnp.import_array()

cdef extern from "wht.h":
    void paulikit_wht_rows(double *data, int64_t rows, int64_t dim,
                           int64_t tile) nogil
    int64_t paulikit_wht_tile_for_cache(int64_t dim) nogil
    int PAULIKIT_WHT_FALLBACK_TILE


def wht_rows_inplace(cnp.ndarray block, int64_t tile=0):
    """Apply the unnormalized Walsh-Hadamard transform to each row of
    ``block``, in place.

    Args:
        block: A C-contiguous 2-D ``complex128`` array of shape
            ``(rows, dim)`` with ``dim`` a power of two. Transformed
            in place; nothing is allocated.
        tile: Tile length in complex elements, a power of two in
            ``[1, dim]``. Pass 0 (default) for the kernel's tuned
            default.

    Returns:
        ``block`` itself, for API symmetry with the pure-Python
        reference's ``overwrite_input=True`` path.

    Raises:
        ValueError: If ``block`` is not 2-D, not complex128, not
            C-contiguous, or ``dim`` is not a power of two.
    """
    if block.ndim != 2:
        raise ValueError(f"expected a 2-D block, got {block.ndim}-D")
    if block.dtype != np.complex128:
        raise ValueError(f"expected complex128, got {block.dtype}")
    if not block.flags["C_CONTIGUOUS"]:
        raise ValueError("block must be C-contiguous")

    cdef int64_t rows = block.shape[0]
    cdef int64_t dim = block.shape[1]
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"dim must be a power of two, got {dim}")
    if tile < 0:
        raise ValueError(f"tile must be non-negative, got {tile}")

    # A zero-size block has nothing to transform and no buffer to
    # take an address of - return before touching the data pointer.
    if rows == 0 or dim < 2:
        return block

    cdef double *data = <double *> cnp.PyArray_DATA(block)
    with nogil:
        paulikit_wht_rows(data, rows, dim, tile)
    return block


def tile_for_cache(int64_t dim):
    """The tile the kernel derives for ``dim`` from this machine's L1
    data-cache size, via ``sysconf``. Exposed for tests and profiling
    harnesses; callers do not need to pass it, since ``tile=0`` makes
    the kernel query it itself."""
    return int(paulikit_wht_tile_for_cache(dim))


def fallback_tile():
    """The compiled-in tile used only where ``sysconf`` cannot report
    an L1 size. Exposed for tests."""
    return int(PAULIKIT_WHT_FALLBACK_TILE)
