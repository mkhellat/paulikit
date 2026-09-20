/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com> */

/* Original C kernel for the fully-dense per-chunk XOR-gather.
 *
 * WHY THIS EXISTS. For an operator with every cell nonzero (the
 * `assume_dense=True` / auto-detected-fully-dense case in
 * paulikit.algorithms.fwht), the Python-level fast path in
 * `_parallel_worker_chunk` reads
 *
 *     gathered_chunk[row, q] = operator[x_row ^ q, q]
 *
 * directly via NumPy fancy indexing (`operator[p_indices, q_range]`),
 * bypassing the general sort/searchsorted/scatter pipeline built for
 * sparse input. That fancy-index gather was measured (perf stat
 * instructions:u) to be a large share of the remaining per-chunk cost
 * once the sort-related overhead was removed - NumPy's generic
 * advanced-indexing machinery pays per-element bounds checks, dtype
 * dispatch and a temporary index array construction that a tight
 * XOR-and-copy loop does not need.
 *
 * WHAT IS COMPUTED, matching the Python reference exactly: for each of
 * `rows` consecutive x values starting at `chunk_start`, and each
 * `q` in `[0, dim)`,
 *
 *     out[row][q] = operator[(chunk_start + row) ^ q][q]
 *
 * `operator` is read-only and untouched; `out` is the caller's
 * pre-allocated (rows, dim) destination, matching
 * `_walsh_hadamard_transform_rows`'s expected input layout exactly so
 * the WHT kernel can run directly on this kernel's output with
 * `overwrite_input=True`.
 *
 * MEMORY AND PARALLELISM. Allocates nothing; touches only the two
 * buffers it is given. `operator` is read-only, so concurrent calls
 * on disjoint chunks (even of the same operator) are safe - this is
 * what lets the existing chunked/threaded pipeline call this kernel
 * from multiple worker threads at once with the GIL released, the
 * same way the WHT and coefficient kernels already do.
 */

#ifndef PAULIKIT_GATHER_H
#define PAULIKIT_GATHER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Direct XOR-gather of one chunk from a fully dense (dim, dim)
 * complex128 operator.
 *
 * `operator_data` points to dim*dim complex numbers, row-major,
 * C-contiguous complex128 layout (consecutive real/imaginary double
 * pairs) - exactly what a NumPy (dim, dim) complex128 array exposes,
 * so it can be passed with no copy or conversion. Read only.
 *
 * `dim` must be a power of two >= 1.
 *
 * `chunk_start` and `rows` select the x values `chunk_start` through
 * `chunk_start + rows - 1`; both must satisfy
 * `0 <= chunk_start` and `chunk_start + rows <= dim`.
 *
 * `out_data` points to caller-allocated space for `rows * dim`
 * complex numbers (row-major, same layout as `operator_data`) and is
 * written completely - every one of its `rows * dim` cells is
 * assigned exactly once, so the caller need not pre-zero it.
 */
void paulikit_gather_dense_chunk(
    const double *operator_data,
    int64_t dim,
    int64_t chunk_start,
    int64_t rows,
    double *out_data
);

#ifdef __cplusplus
}
#endif

#endif /* PAULIKIT_GATHER_H */
