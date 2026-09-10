"""Minimal, dependency-free Pauli-matrix utilities shared across paulikit.

Deliberately does not depend on PennyLane, Qiskit, or Classiq: this
module is meant to remain usable as an independent check even if the
libraries used to *generate* the fixtures (see
``paulikit.testing.fixtures``) are absent or change behavior.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_PAULI_MATRICES: dict[str, NDArray[np.complexfloating]] = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string_to_matrix(label: str) -> NDArray[np.complexfloating]:
    """Build the dense matrix for a Pauli string label via tensor product.

    Args:
        label: A string of Pauli letters, e.g. ``"IXZ"``. The leftmost
            character is qubit 0, matching the convention used by
            ``paulikit.testing.fixtures.pauli_word_to_label``.

    Returns:
        A ``(2**len(label), 2**len(label))`` complex ``numpy.ndarray``.
    """
    matrix = _PAULI_MATRICES[label[0]]
    for letter in label[1:]:
        matrix = np.kron(matrix, _PAULI_MATRICES[letter])
    return matrix


def reconstruct_from_terms(
    terms: dict[str, float] | dict[str, complex], n_qubits: int
) -> NDArray[np.complexfloating]:
    """Reconstruct a dense operator from a label -> coefficient dict.

    Args:
        terms: Mapping from Pauli-string label to coefficient (real or
            complex - complex coefficients arise when decomposing a
            non-Hermitian operator), as produced by a decomposition
            algorithm (e.g.
            ``paulikit.algorithms.fwht.fwht_pauli_terms``) or stored
            in ``paulikit.testing.fixtures``.
        n_qubits: Number of qubits; every label in ``terms`` must have
            this length.

    Returns:
        A ``(2**n_qubits, 2**n_qubits)`` ``numpy.ndarray`` equal to the
        sum of ``coefficient * pauli_string_to_matrix(label)`` over all
        terms. The dtype is always ``complex128``, even when every
        coefficient was real - take ``.real`` if a real array is
        wanted, having checked the imaginary part is negligible.
    """
    dim = 2**n_qubits
    total = np.zeros((dim, dim), dtype=complex)
    for label, coefficient in terms.items():
        assert len(label) == n_qubits, (
            f"label {label!r} has length {len(label)}, expected {n_qubits}"
        )
        total += coefficient * pauli_string_to_matrix(label)
    return total
