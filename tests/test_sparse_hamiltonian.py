"""Correctness tests for the sparse-Hamiltonian construction/input
path: ``build_hamiltonian(..., sparse=True)``,
``pad_to_power_of_two(..., sparse=True)``, and
``fwht_pauli_coefficients``/``fwht_pauli_terms`` accepting a
``scipy.sparse`` operator directly.

Requires the ``sparse`` extra (scipy); the whole module is skipped if
scipy is not installed.
"""

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

from paulikit.algorithms.fwht import fwht_pauli_coefficients, fwht_pauli_terms
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def _spring_constants(n_oscillators):
    return {
        (i, j): 1.0 + 0.1 * (i + j) for i in range(n_oscillators) for j in range(i, n_oscillators)
    }


def _masses(n_oscillators):
    return [1.0 + 0.05 * i for i in range(n_oscillators)]


@pytest.mark.parametrize("n_oscillators", [2, 5, 10])
def test_build_hamiltonian_sparse_matches_dense(n_oscillators):
    spring_constants = _spring_constants(n_oscillators)
    masses = _masses(n_oscillators)

    dense = build_hamiltonian(n_oscillators, spring_constants, masses)
    sparse = build_hamiltonian(n_oscillators, spring_constants, masses, sparse=True)

    assert sp.issparse(sparse)
    max_error = np.max(np.abs(dense - sparse.toarray()))
    assert max_error == 0.0


@pytest.mark.parametrize("n_oscillators", [2, 5, 10])
def test_pad_to_power_of_two_sparse_matches_dense(n_oscillators):
    spring_constants = _spring_constants(n_oscillators)
    masses = _masses(n_oscillators)

    dense = build_hamiltonian(n_oscillators, spring_constants, masses)
    sparse = build_hamiltonian(n_oscillators, spring_constants, masses, sparse=True)

    padded_dense, n_qubits_dense = pad_to_power_of_two(dense)
    padded_sparse, n_qubits_sparse = pad_to_power_of_two(sparse, sparse=True)

    assert n_qubits_dense == n_qubits_sparse
    assert sp.issparse(padded_sparse)
    max_error = np.max(np.abs(padded_dense - padded_sparse.toarray()))
    assert max_error == 0.0


def test_pad_to_power_of_two_sparse_does_not_mutate_input():
    """csr_matrix.resize() mutates its input in place (verified
    directly while scoping this sparse path) - pad_to_power_of_two
    must not exhibit that behavior, matching its existing dense
    contract."""
    spring_constants = _spring_constants(5)
    masses = _masses(5)
    sparse = build_hamiltonian(5, spring_constants, masses, sparse=True)
    before = sparse.toarray().copy()

    pad_to_power_of_two(sparse, sparse=True)

    assert np.array_equal(sparse.toarray(), before)


@pytest.mark.parametrize("n_oscillators", [2, 5, 10])
def test_fwht_pauli_coefficients_accepts_sparse_operator(n_oscillators):
    spring_constants = _spring_constants(n_oscillators)
    masses = _masses(n_oscillators)

    dense = build_hamiltonian(n_oscillators, spring_constants, masses)
    sparse = build_hamiltonian(n_oscillators, spring_constants, masses, sparse=True)
    padded_dense, _ = pad_to_power_of_two(dense)
    padded_sparse, _ = pad_to_power_of_two(sparse, sparse=True)

    dense_result = fwht_pauli_coefficients(padded_dense)
    sparse_result = fwht_pauli_coefficients(padded_sparse)

    max_error = np.max(np.abs(dense_result - sparse_result))
    assert max_error == 0.0


def test_fwht_pauli_coefficients_sparse_output_mode_matches_with_sparse_input():
    spring_constants = _spring_constants(5)
    masses = _masses(5)
    sparse = build_hamiltonian(5, spring_constants, masses, sparse=True)
    padded_sparse, _ = pad_to_power_of_two(sparse, sparse=True)
    dim = padded_sparse.shape[0]

    active_x, active_coefficients = fwht_pauli_coefficients(padded_sparse, sparse=True)
    full = np.zeros((dim, dim), dtype=complex)
    full[active_x] = active_coefficients

    dense_result = fwht_pauli_coefficients(padded_sparse.toarray())
    assert np.max(np.abs(full - dense_result)) == 0.0


def test_fwht_pauli_coefficients_sparse_input_with_chunking():
    """chunk_size switches fwht_pauli_coefficients to the already-
    thresholded COO (x, z, coefficient) triple return form -
    reconstructed here and compared against the dense result with
    the same atol applied, rather than a raw bit-for-bit comparison
    (which would not hold once near-zero entries are dropped inside
    the chunked path itself)."""
    spring_constants = _spring_constants(5)
    masses = _masses(5)
    sparse = build_hamiltonian(5, spring_constants, masses, sparse=True)
    padded_sparse, _ = pad_to_power_of_two(sparse, sparse=True)
    dim = padded_sparse.shape[0]
    atol = 1e-10

    x_out, z_out, coeff_out = fwht_pauli_coefficients(
        padded_sparse, sparse=True, chunk_size=1, atol=atol
    )
    full = np.zeros((dim, dim), dtype=complex)
    full[x_out, z_out] = coeff_out

    dense_result = fwht_pauli_coefficients(padded_sparse.toarray())
    dense_result = np.where(np.abs(dense_result) > atol, dense_result, 0.0)
    assert np.max(np.abs(full - dense_result)) == 0.0


def test_fwht_pauli_terms_sparse_input_matches_dense():
    spring_constants = _spring_constants(10)
    masses = _masses(10)

    dense = build_hamiltonian(10, spring_constants, masses)
    sparse = build_hamiltonian(10, spring_constants, masses, sparse=True)
    padded_dense, _ = pad_to_power_of_two(dense)
    padded_sparse, _ = pad_to_power_of_two(sparse, sparse=True)

    terms_dense = fwht_pauli_terms(padded_dense)
    terms_sparse = fwht_pauli_terms(padded_sparse)

    assert set(terms_dense) == set(terms_sparse)
    for label, expected in terms_dense.items():
        assert terms_sparse[label] == pytest.approx(expected, abs=1e-9)


def test_build_hamiltonian_sparse_without_scipy_raises_import_error(monkeypatch):
    import paulikit.hamiltonian as hamiltonian_module

    monkeypatch.setattr(hamiltonian_module, "_sp", None)
    with pytest.raises(ImportError, match="paulikit\\[sparse\\]"):
        build_hamiltonian(2, _spring_constants(2), _masses(2), sparse=True)


def test_pad_to_power_of_two_sparse_without_scipy_raises_import_error(monkeypatch):
    import paulikit.hamiltonian as hamiltonian_module

    dense = build_hamiltonian(2, _spring_constants(2), _masses(2))
    monkeypatch.setattr(hamiltonian_module, "_sp", None)
    with pytest.raises(ImportError, match="paulikit\\[sparse\\]"):
        pad_to_power_of_two(dense, sparse=True)
