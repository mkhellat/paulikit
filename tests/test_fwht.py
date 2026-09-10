"""Correctness tests for the original FWHT-based Pauli decomposition.

Tests at two levels:

1. Against a from-scratch, definition-level brute-force decomposition
   (Frobenius inner product against every tensor-product Pauli string,
   built independently of ``paulikit.algorithms.fwht``'s internals) on
   small random Hermitian matrices. This is the same
   derivation-verification approach used while developing the
   algorithm - see that module's docstring - re-run here as an
   automated regression test rather than a one-off manual check.
2. Against ``paulikit.testing.fixtures.ALL_FIXTURES`` (the
   coupled-oscillator Hamiltonians with PennyLane-derived expected
   terms), per that module's own instruction that algorithms be
   tested against those fixtures directly rather than re-deriving
   expected values inline.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    fwht_pauli_coefficients,
    fwht_pauli_terms,
    pauli_label,
)
from paulikit.hamiltonian import pad_to_power_of_two
from paulikit.pauli_utils import pauli_string_to_matrix, reconstruct_from_terms
from paulikit.testing.fixtures import ALL_FIXTURES


def _reference_pauli_matrix(x_mask: int, z_mask: int, n_qubits: int) -> np.ndarray:
    """Definition-level Pauli string matrix, independent of
    ``paulikit.algorithms.fwht``.

    Built directly from the symplectic representation
    P = kron_j( i**(x_j & z_j) * X**x_j * Z**z_j ), matching the same
    convention used to derive that module's formulas (qubit j is bit
    (n_qubits - 1 - j) of the masks).
    """
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    factors = []
    for qubit in range(n_qubits):
        bit = n_qubits - 1 - qubit
        xj = (x_mask >> bit) & 1
        zj = (z_mask >> bit) & 1
        phase = 1j ** (xj & zj)
        factor = phase * np.linalg.matrix_power(pauli_x, xj) @ np.linalg.matrix_power(
            pauli_z, zj
        )
        factors.append(factor)
    matrix = factors[0]
    for factor in factors[1:]:
        matrix = np.kron(matrix, factor)
    return matrix


def _reference_brute_force_decompose(hamiltonian: np.ndarray, n_qubits: int) -> np.ndarray:
    """O(4**n) definition-level decomposition: Frobenius inner product
    against every Pauli string, with no FWHT/shortcut involved."""
    dim = hamiltonian.shape[0]
    coefficients = np.zeros((dim, dim), dtype=complex)
    for x_mask in range(dim):
        for z_mask in range(dim):
            pauli = _reference_pauli_matrix(x_mask, z_mask, n_qubits)
            coefficients[x_mask, z_mask] = (
                np.trace(hamiltonian @ pauli.conj().T) / dim
            )
    return coefficients


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_fwht_matches_brute_force_on_random_hermitian(n_qubits):
    """The fast algorithm must match the definition-level brute force
    exactly (to floating-point precision) on general Hermitian input,
    not just on the structured coupled-oscillator fixtures."""
    rng = np.random.default_rng(seed=42 + n_qubits)
    dim = 2**n_qubits
    real_part = rng.random((dim, dim))
    imag_part = rng.random((dim, dim))
    matrix = real_part + 1j * imag_part
    hamiltonian = matrix + matrix.conj().T  # make Hermitian

    fast = fwht_pauli_coefficients(hamiltonian)
    reference = _reference_brute_force_decompose(hamiltonian, n_qubits)

    max_error = np.max(np.abs(fast - reference))
    assert max_error < 1e-9, f"n_qubits={n_qubits}: max error {max_error}"


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_fwht_matches_fixture_expected_terms(fixture):
    """The fast algorithm's output must match the fixtures' stored,
    independently-generated expected terms exactly."""
    padded = fixture.padded_hamiltonian()
    computed = fwht_pauli_terms(padded)

    assert set(computed) == set(fixture.expected_terms), (
        f"fixture {fixture.name!r}: label set mismatch - "
        f"missing {set(fixture.expected_terms) - set(computed)}, "
        f"extra {set(computed) - set(fixture.expected_terms)}"
    )
    for label, expected_coefficient in fixture.expected_terms.items():
        assert computed[label] == pytest.approx(expected_coefficient, abs=1e-9), (
            f"fixture {fixture.name!r}, term {label!r}: "
            f"got {computed[label]}, expected {expected_coefficient}"
        )


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_fwht_terms_reconstruct_hamiltonian(fixture):
    """The fast algorithm's own output, independently reconstructed via
    pauli_utils (not the fixtures' expected_terms), must reproduce H."""
    padded = fixture.padded_hamiltonian()
    computed = fwht_pauli_terms(padded)
    reconstructed = reconstruct_from_terms(computed, fixture.n_qubits)

    max_error = np.max(np.abs(reconstructed.real - padded))
    assert max_error < 1e-9, f"fixture {fixture.name!r}: max error {max_error}"


def test_pauli_label_round_trips_through_pauli_utils():
    """pauli_label's IXYZ strings must be understood by pauli_utils
    (shared convention across the package) for every single-term case
    at a couple of qubit counts."""
    for n_qubits in [1, 2, 3]:
        dim = 2**n_qubits
        for x_mask in range(dim):
            for z_mask in range(dim):
                label = pauli_label(x_mask, z_mask, n_qubits)
                assert len(label) == n_qubits
                # pauli_utils's matrix, scaled by the same phase used
                # internally, should match the reference construction.
                reference = _reference_pauli_matrix(x_mask, z_mask, n_qubits)
                from_label = pauli_string_to_matrix(label)
                assert np.allclose(reference, from_label), (
                    f"label {label!r} (x={x_mask}, z={z_mask}, "
                    f"n_qubits={n_qubits}) does not match pauli_utils's "
                    "matrix for that label"
                )


def test_rejects_non_power_of_two_dimension():
    with pytest.raises(ValueError):
        fwht_pauli_coefficients(np.eye(5))


def test_rejects_non_square_input():
    with pytest.raises(ValueError):
        fwht_pauli_coefficients(np.zeros((4, 8)))


def test_padded_odd_size_hamiltonian_decomposes_correctly():
    """Exercises the pad_to_power_of_two + fwht_pauli_terms combination
    end to end on a small, non-power-of-two matrix, as used for real
    coupled-oscillator Hamiltonians."""
    rng = np.random.default_rng(7)
    raw = rng.random((5, 5))
    raw = raw + raw.T
    padded, n_qubits = pad_to_power_of_two(raw)
    assert padded.shape == (8, 8)
    assert n_qubits == 3

    terms = fwht_pauli_terms(padded)
    reconstructed = reconstruct_from_terms(terms, n_qubits)
    max_error = np.max(np.abs(reconstructed.real - padded))
    assert max_error < 1e-9


# --- Non-Hermitian operator support -----------------------------------
#
# The Pauli strings (I, X, Y, Z tensor products) span the *entire*
# space of 2**n x 2**n complex matrices, not just the Hermitian ones -
# there is nothing in the Frobenius-inner-product math that requires
# Hermiticity. This matters beyond mathematical completeness: not every
# physically meaningful operator is Hermitian (non-Hermitian effective
# Hamiltonians for open/dissipative systems, PT-symmetric Hamiltonians,
# Liouvillian superoperators, individual non-Hermitian summands of a
# Hermitian total, ...). These tests confirm paulikit decomposes such
# operators correctly rather than only working by accident on the
# Hermitian coupled-oscillator case the fixtures happen to cover.


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_fwht_pauli_coefficients_handles_non_hermitian_matrices(n_qubits):
    """fwht_pauli_coefficients must decompose an arbitrary (non-Hermitian)
    complex matrix exactly, matching the brute-force reference and
    producing genuinely complex coefficients."""
    rng = np.random.default_rng(100 + n_qubits)
    dim = 2**n_qubits
    operator = rng.random((dim, dim)) + 1j * rng.random((dim, dim))
    # deliberately NOT symmetrized/Hermitized

    fast = fwht_pauli_coefficients(operator)
    reference = _reference_brute_force_decompose(operator, n_qubits)

    max_error = np.max(np.abs(fast - reference))
    assert max_error < 1e-9, f"n_qubits={n_qubits}: max error {max_error}"

    # A generic non-Hermitian matrix should actually produce some
    # terms with non-negligible imaginary coefficients - otherwise
    # this test isn't exercising the non-Hermitian path at all.
    assert np.max(np.abs(fast.imag)) > 1e-3


def test_fwht_pauli_coefficients_reconstructs_non_hermitian_matrix():
    """The full coefficient array must reconstruct a non-Hermitian
    operator exactly via pauli_utils (independent reconstruction)."""
    rng = np.random.default_rng(11)
    n_qubits = 2
    dim = 2**n_qubits
    operator = rng.random((dim, dim)) + 1j * rng.random((dim, dim))

    coefficients = fwht_pauli_coefficients(operator)
    reconstructed = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        for z in range(dim):
            c = coefficients[x, z]
            if abs(c) > 1e-12:
                reconstructed += c * pauli_string_to_matrix(pauli_label(x, z, n_qubits))

    max_error = np.max(np.abs(reconstructed - operator))
    assert max_error < 1e-9


def test_fwht_pauli_terms_assume_hermitian_true_rejects_non_hermitian_input():
    """The default (assume_hermitian=True) convenience-wrapper behavior
    must refuse to silently discard imaginary parts of a genuinely
    non-Hermitian operator's coefficients."""
    rng = np.random.default_rng(22)
    n_qubits = 2
    dim = 2**n_qubits
    operator = rng.random((dim, dim)) + 1j * rng.random((dim, dim))

    with pytest.raises(ValueError, match="may not be Hermitian"):
        fwht_pauli_terms(operator)  # assume_hermitian=True is the default

    with pytest.raises(ValueError, match="may not be Hermitian"):
        fwht_pauli_terms(operator, assume_hermitian=True)


def test_fwht_pauli_terms_hermiticity_error_names_a_real_violating_term():
    """The vectorized Hermiticity check must preserve the original
    per-term loop's diagnostic specificity - the error must name an
    actual violating label and its actual imaginary part, not a
    generic "something failed" message. Since the check is now
    vectorized over the whole array and only re-scans on the rare
    violation path (np.nonzero, first match), this specifically
    exercises that fallback path with a real, decomposable operator
    (not a placeholder)."""
    rng = np.random.default_rng(22)
    n_qubits = 2
    dim = 2**n_qubits
    operator = rng.random((dim, dim)) + 1j * rng.random((dim, dim))

    # Independently recompute which term(s) actually violate, using
    # the exact same tolerance formula as the implementation, so the
    # error message can be checked against a value derived the same
    # way production computes it - not just any string.
    coefficients = fwht_pauli_coefficients(operator, sparse=False)
    atol = 1e-10
    violating = np.argwhere(
        np.abs(coefficients.imag) > np.maximum(atol, 1e-6 * np.abs(coefficients))
    )
    assert len(violating) > 0, "test operator must actually violate Hermiticity"
    x0, z0 = violating[0]
    expected_label = pauli_label(int(x0), int(z0), n_qubits)
    expected_imag = coefficients[x0, z0].imag

    with pytest.raises(ValueError) as excinfo:
        fwht_pauli_terms(operator)

    message = str(excinfo.value)
    assert repr(expected_label) in message
    assert repr(expected_imag) in message


def test_build_real_terms_tolerance_floor_uses_full_complex_magnitude():
    """The vectorized check's tolerance floor must be built from
    abs(c) (the full complex magnitude), not abs(c.real) - an easy,
    silently-wrong substitution (caught once already while scoping
    this microbenchmark, corrected 2026-08-31). Note this cannot be
    turned into a pass/fail-verdict regression test: analysis while
    writing this test showed abs(c) and abs(c.real) agree to leading
    order for any term near either floor (their difference is
    O(imag**2 / real), negligible exactly where it would flip a
    verdict), and for terms far from the floor both formulas agree
    the term is a clear violation either way - no single term's
    Hermiticity verdict can distinguish the two formulas. This
    instead pins down the formula directly against the module's own
    source, so a future edit that reintroduces the substitution is
    still caught even though no behavioral test can."""
    import inspect

    from paulikit.algorithms.fwht import _build_real_terms

    source = inspect.getsource(_build_real_terms)
    assert "c_abs = np.abs(coefficient_values)" in source
    assert "np.abs(coefficient_values.real)" not in source


def test_fwht_pauli_terms_assume_hermitian_false_decomposes_non_hermitian_input():
    """With assume_hermitian=False, fwht_pauli_terms must return complex
    coefficients and reconstruct the (non-Hermitian) operator exactly."""
    rng = np.random.default_rng(33)
    n_qubits = 2
    dim = 2**n_qubits
    operator = rng.random((dim, dim)) + 1j * rng.random((dim, dim))

    terms = fwht_pauli_terms(operator, assume_hermitian=False)

    assert any(isinstance(c, complex) and c.imag != 0 for c in terms.values()), (
        "expected at least one term with a non-negligible imaginary "
        "coefficient for a generic non-Hermitian operator"
    )

    reconstructed = reconstruct_from_terms(terms, n_qubits)
    max_error = np.max(np.abs(reconstructed - operator))
    assert max_error < 1e-9


def test_fwht_pauli_terms_assume_hermitian_false_matches_true_on_hermitian_input():
    """For genuinely Hermitian input, both modes must agree (up to the
    float-vs-complex return type): assume_hermitian=False should return
    the same values as assume_hermitian=True, just as complex with a
    zero imaginary part instead of float."""
    for fixture in ALL_FIXTURES:
        padded = fixture.padded_hamiltonian()
        real_terms = fwht_pauli_terms(padded, assume_hermitian=True)
        complex_terms = fwht_pauli_terms(padded, assume_hermitian=False)

        assert set(real_terms) == set(complex_terms)
        for label in real_terms:
            assert complex_terms[label] == pytest.approx(
                real_terms[label], abs=1e-9
            )
            assert abs(complex_terms[label].imag) < 1e-9
