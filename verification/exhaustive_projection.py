"""Exhaustive, independent Pauli-coefficient verification via direct projection.

This is a second, differently-derived way to compute a Hamiltonian's
Pauli-decomposition coefficients, used to verify
``paulikit.algorithms.fwht.fwht_pauli_terms`` without depending on it or
re-deriving the same FWHT-based shortcut. It does NOT use
``qml.pauli_decompose`` either: PennyLane's own documented theory
section states its decomposition is O(n * 4**n) regardless of input
sparsity (a docstring claim, not one derived from reading its
implementation - see ``verification/FINDINGS.md``), so its output is
exponentially large in qubit count well before the N~150 scale
paulikit targets. This module instead evaluates the projection
formula directly, for exactly the set of labels paulikit already
claims are nonzero - giving true 100% coverage (every term paulikit
outputs, not a sample) at any N, in time bounded by
``nnz(H) * n_terms`` rather than by ``4**n``.

Math: for a Hermitian or non-Hermitian operator H and a Pauli label
with symplectic bitmasks (x_mask, z_mask) (convention: leftmost
character = qubit 0, matching
``paulikit.algorithms.fwht.pauli_label``'s own docstring), the
label's coefficient is

    c = Tr(H @ P_label^dagger) / dim

computed without ever materializing the full (dim, dim) matrix
P_label, by using that P_label's nonzero pattern sends row r to
column c = r ^ x_mask (the X-part flips exactly the x_mask bits),
with value sign * phase where
    sign  = (-1) ** popcount(r & z_mask)
    phase = (1j) ** popcount(x_mask & z_mask)
So H[r, c] contributes to label (x_mask, z_mask)'s coefficient
exactly when r ^ c == x_mask, i.e. every nonzero H[r, c] contributes
to exactly one label per fixed z_mask... but since we only need
specific (x_mask, z_mask) pairs (the ones paulikit reports), we
instead group H's nonzeros by their r^c value and each queried label
by its x_mask, then match groups via a sorted searchsorted - no
Python-level loop over individual labels or individual H nonzeros.

Iteration history (kept here, not just in memory, per this project's
documentation-of-exploration policy - see FINDINGS.md for the full
writeup):
  v1: sp.kron per-label rebuild               too slow to reach N=150
  v2: per-label Python loop over H's nonzeros  still too slow
  v3: dict-bucketed-by-xor, Python loop/bucket does NOT scale
      (bucket-content loop still Python-level)
  v4: fully vectorized NumPy (this file)       scales correctly,
      see FINDINGS.md for N=50/80/100/150 measurements
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


def _popcount(x: NDArray[np.int64]) -> NDArray[np.int64]:
    """Vectorized popcount for int64 arrays (bit-trick, no Python loop)."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return x & 0x7F


def label_to_masks(label: str) -> tuple[int, int]:
    """Inverse of ``paulikit.algorithms.fwht.pauli_label``.

    Leftmost character = qubit 0, bit position ``n-1-j`` for qubit j -
    matching that function's own documented convention exactly.
    """
    letters = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
    n = len(label)
    x_mask = 0
    z_mask = 0
    for qubit, ch in enumerate(label):
        xj, zj = letters[ch]
        bit = n - 1 - qubit
        x_mask |= xj << bit
        z_mask |= zj << bit
    return x_mask, z_mask


def project_labels(
    operator: sp.spmatrix,
    labels: list[str],
    max_broadcast_elements: int = 50_000_000,
) -> NDArray[np.complexfloating]:
    """Compute Tr(H @ P_label^dagger) / dim for every label, vectorized.

    Args:
        operator: A ``(dim, dim)`` ``scipy.sparse`` matrix (any format;
            converted to COO internally). Real or complex, Hermitian
            or not - the projection formula makes no Hermiticity
            assumption.
        labels: Pauli-string labels to evaluate, e.g. paulikit's own
            ``fwht_pauli_terms(...).keys()``. Every label must have
            length ``log2(dim)``.
        max_broadcast_elements: Memory-bound cap on the size of any
            single ``(n_H_nonzeros_in_group, n_labels_in_group)``
            broadcast array built during projection. A group whose
            full broadcast would exceed this is processed in
            sub-chunks along its member axis instead of all at once -
            same result, bounded peak memory. Found necessary at
            N=150: an unbounded single-shot broadcast for a group with
            many shared-x_mask labels against a dense H nonzero range
            was OOM-killed (observed: process silently killed, host
            swap exhausted, no partial result) before this cap was
            added. 50M complex128 entries is ~800MB per temporary
            array, safely inside a typical dev machine's RAM even with
            a few temporaries alive at once.

    Returns:
        A complex array of coefficients, same order as ``labels``.
    """
    coo = operator.tocoo()
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))

    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    vals = coo.data.astype(np.complex128)
    xor_rc = rows ^ cols

    x_masks = np.empty(len(labels), dtype=np.int64)
    z_masks = np.empty(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        x_masks[i], z_masks[i] = label_to_masks(label)

    # Sort H's nonzeros by r^c so each label's matching entries form a
    # contiguous slice, found via searchsorted - no Python-level loop
    # over individual nonzeros.
    order = np.argsort(xor_rc, kind="stable")
    xor_sorted = xor_rc[order]
    rows_sorted = rows[order]
    vals_sorted = vals[order]

    start_idx = np.searchsorted(xor_sorted, x_masks, side="left")
    end_idx = np.searchsorted(xor_sorted, x_masks, side="right")

    # Group labels that share the same (start, end) slice - i.e. the
    # same x_mask - so the per-group broadcast work happens once per
    # distinct x_mask, not once per label.
    slice_key = start_idx.astype(np.int64) * (len(xor_rc) + 1) + end_idx.astype(
        np.int64
    )
    # np.unique's return order is always (unique, [index], [inverse],
    # [counts]) regardless of which return_* kwargs are passed - index
    # (first-occurrence position per unique value) comes before
    # inverse (group id per original element).
    _, group_starts_ends, group_of = np.unique(
        slice_key, return_index=True, return_inverse=True
    )

    coefficients = np.zeros(len(labels), dtype=np.complex128)
    n_groups = len(group_starts_ends)
    for g in range(n_groups):
        rep = group_starts_ends[g]
        s, e = start_idx[rep], end_idx[rep]
        member_mask = group_of == g
        member_idx = np.nonzero(member_mask)[0]
        if s == e:
            continue  # no H nonzero has this x_mask -> coefficient is 0
        r_group = rows_sorted[s:e]
        v_group = vals_sorted[s:e]
        n_in_group = e - s
        n_members = len(member_idx)

        # Bound peak memory: a single (n_in_group, n_members) broadcast
        # array can be huge if either axis is large (seen at N=150 -
        # see docstring). Sub-chunk along the member axis so no single
        # temporary array exceeds max_broadcast_elements.
        chunk = max(1, max_broadcast_elements // max(1, n_in_group))
        for m_start in range(0, n_members, chunk):
            m_end = min(m_start + chunk, n_members)
            sub_idx = member_idx[m_start:m_end]
            z_sub = z_masks[sub_idx]

            # sign[k, m] = (-1) ** popcount(r_group[k] & z_sub[m])
            and_rz = r_group[:, None] & z_sub[None, :]
            sign = 1 - 2 * (_popcount(and_rz) & 1)  # (n_in_group, n_sub)
            contrib = (v_group[:, None] * sign).sum(axis=0)  # (n_sub,)

            x_sub = x_masks[sub_idx]
            phase_exp = _popcount(x_sub & z_sub) & 3
            phase = 1j ** phase_exp  # (n_sub,)

            coefficients[sub_idx] = contrib * phase / dim

    return coefficients


def verify_terms_streaming(
    operator: sp.spmatrix,
    chunks: Iterable[dict[str, complex] | dict[str, float]],
    atol: float = 1e-9,
) -> dict:
    """Like ``verify_terms``, but consumes an iterator of per-chunk
    term dicts (e.g. ``paulikit.algorithms.fwht.fwht_pauli_terms_iter``)
    instead of one combined dict.

    Required at large N: ``fwht_pauli_terms`` (the dict-returning API)
    cannot complete at N=150 regardless of available RAM (see
    FINDINGS.md section 5), because it re-fuses every chunk into one
    91.65M-entry dict before returning.
    This function never holds more than one chunk's terms (plus
    running summary statistics) at once, matching the streaming
    design's own memory contract.

    Args:
        operator: Same as ``verify_terms``.
        chunks: An iterable of ``label -> coefficient`` dicts, one per
            chunk.
        atol: Absolute-error tolerance for pass/fail.

    Returns:
        Same shape as ``verify_terms``'s return value.
    """
    n_terms = 0
    max_abs_error = 0.0
    sum_abs_error = 0.0
    worst_label = None

    for chunk_terms in chunks:
        if not chunk_terms:
            continue
        labels = list(chunk_terms.keys())
        expected = np.array([chunk_terms[label] for label in labels], dtype=np.complex128)
        computed = project_labels(operator, labels)
        errors = np.abs(computed - expected)

        chunk_max_idx = int(np.argmax(errors))
        if errors[chunk_max_idx] > max_abs_error:
            max_abs_error = float(errors[chunk_max_idx])
            worst_label = labels[chunk_max_idx]

        n_terms += len(labels)
        sum_abs_error += float(errors.sum())

    return {
        "n_terms": n_terms,
        "max_abs_error": max_abs_error,
        "mean_abs_error": (sum_abs_error / n_terms) if n_terms else 0.0,
        "worst_label": worst_label,
        "passed": bool(max_abs_error <= atol),
    }


def verify_terms(
    operator: sp.spmatrix,
    terms: dict[str, complex] | dict[str, float],
    atol: float = 1e-9,
) -> dict:
    """Verify every entry of ``terms`` against the independent projection.

    Args:
        operator: The same ``(dim, dim)`` sparse matrix ``terms`` was
            decomposed from.
        terms: paulikit's own ``label -> coefficient`` output, e.g.
            from ``fwht_pauli_terms``.
        atol: Absolute-error tolerance for pass/fail.

    Returns:
        A dict with ``n_terms``, ``max_abs_error``, ``mean_abs_error``,
        ``passed`` (bool), and ``worst_label`` for diagnostics.
    """
    labels = list(terms.keys())
    expected = np.array([terms[label] for label in labels], dtype=np.complex128)
    computed = project_labels(operator, labels)

    errors = np.abs(computed - expected)
    max_err_idx = int(np.argmax(errors))

    return {
        "n_terms": len(labels),
        "max_abs_error": float(errors[max_err_idx]),
        "mean_abs_error": float(errors.mean()),
        "worst_label": labels[max_err_idx],
        "passed": bool(errors.max() <= atol),
    }
