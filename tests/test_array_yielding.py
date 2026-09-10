"""Tests for the array-yielding parallel decomposition API:
parallel_decompose_arrays and terms_from_arrays.

Correctness is anchored on two independent references: round-trip
equivalence with parallel_decompose (same machinery, different output
format) and agreement with fwht_pauli_terms (an independently derived
sequential path), so a bug shared by both parallel paths cannot pass.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    _build_real_terms,
    _check_hermitian_violation,
    _pauli_label_batch,
    fwht_pauli_terms,
    parallel_decompose,
)
from paulikit.testing.fixtures import ALL_FIXTURES


def test_check_hermitian_violation_raises_naming_offending_term():
    # Two terms; only the second has a non-negligible imaginary part.
    x = np.array([0, 1], dtype=np.uint16)
    z = np.array([0, 2], dtype=np.uint16)
    coeff = np.array([1.0 + 0.0j, 2.0 + 0.5j])

    with pytest.raises(ValueError, match="imaginary part"):
        _check_hermitian_violation(coeff, 1e-10, x, z, n_qubits=2)


def test_check_hermitian_violation_message_matches_build_real_terms():
    # The array path has no labels at check time; it must still produce
    # a byte-identical message to the dict path's, by labeling ONLY the
    # single offending term.
    x = np.array([0, 1], dtype=np.uint16)
    z = np.array([0, 2], dtype=np.uint16)
    coeff = np.array([1.0 + 0.0j, 2.0 + 0.5j])
    labels = _pauli_label_batch(x, z, 2)

    with pytest.raises(ValueError) as dict_err:
        _build_real_terms(labels, coeff, 1e-10)
    with pytest.raises(ValueError) as array_err:
        _check_hermitian_violation(coeff, 1e-10, x, z, n_qubits=2)

    assert str(array_err.value) == str(dict_err.value)


def test_check_hermitian_violation_passes_for_hermitian_input():
    x = np.array([0, 1], dtype=np.uint16)
    z = np.array([0, 2], dtype=np.uint16)
    coeff = np.array([1.0 + 0.0j, 2.0 + 0.0j])

    assert _check_hermitian_violation(coeff, 1e-10, x, z, n_qubits=2) is None


def test_terms_from_arrays_matches_build_real_terms():
    from paulikit.algorithms.fwht import terms_from_arrays

    x = np.array([0, 1, 2], dtype=np.uint16)
    z = np.array([0, 2, 1], dtype=np.uint16)
    coeff = np.array([1.5 + 0.0j, -2.0 + 0.0j, 0.25 + 0.0j])

    expected = _build_real_terms(_pauli_label_batch(x, z, 2), coeff, 1e-10)
    assert terms_from_arrays(x, z, coeff, n_qubits=2) == expected


def test_terms_from_arrays_non_hermitian_raises():
    from paulikit.algorithms.fwht import terms_from_arrays

    x = np.array([0], dtype=np.uint16)
    z = np.array([0], dtype=np.uint16)
    coeff = np.array([1.0 + 0.5j])

    with pytest.raises(ValueError, match="imaginary part"):
        terms_from_arrays(x, z, coeff, n_qubits=2)


def test_terms_from_arrays_assume_hermitian_false_keeps_complex():
    from paulikit.algorithms.fwht import terms_from_arrays

    x = np.array([0], dtype=np.uint16)
    z = np.array([0], dtype=np.uint16)
    coeff = np.array([1.0 + 0.5j])

    result = terms_from_arrays(x, z, coeff, n_qubits=2, assume_hermitian=False)
    assert result == {"II": 1.0 + 0.5j}


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.intp])
def test_terms_from_arrays_accepts_any_integer_dtype(dtype):
    # Legacy checkpoints hold intp; new results hold uint16. Both must work.
    from paulikit.algorithms.fwht import terms_from_arrays

    x = np.array([1], dtype=dtype)
    z = np.array([2], dtype=dtype)
    coeff = np.array([1.0 + 0.0j])

    assert terms_from_arrays(x, z, coeff, n_qubits=2) == {"ZX": 1.0}


def test_terms_from_arrays_empty_input_returns_empty_dict():
    from paulikit.algorithms.fwht import terms_from_arrays

    empty_i = np.array([], dtype=np.uint16)
    empty_c = np.array([], dtype=complex)

    assert terms_from_arrays(empty_i, empty_i, empty_c, n_qubits=2) == {}


def _combine_arrays(chunks, n_qubits):
    """Render every yielded chunk and merge, mirroring _combine in
    tests/test_parallel_decompose.py."""
    from paulikit.algorithms.fwht import terms_from_arrays

    combined = {}
    for x, z, coeff in chunks:
        combined.update(terms_from_arrays(x, z, coeff, n_qubits))
    return combined


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_parallel_decompose_arrays_matches_fwht_pauli_terms(fixture, chunk_size):
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = fixture.padded_hamiltonian()
    n_qubits = int(np.log2(padded.shape[0]))
    reference = fwht_pauli_terms(padded)

    combined = _combine_arrays(
        parallel_decompose_arrays(padded, chunk_size=chunk_size, n_workers=2),
        n_qubits,
    )

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_round_trip_equals_parallel_decompose(fixture):
    # The correctness anchor: rendering the array path must reproduce
    # the dict path exactly, term for term.
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = fixture.padded_hamiltonian()
    n_qubits = int(np.log2(padded.shape[0]))

    dict_terms = {}
    for chunk in parallel_decompose(padded, chunk_size=2, n_workers=2):
        dict_terms.update(chunk)
    array_terms = _combine_arrays(
        parallel_decompose_arrays(padded, chunk_size=2, n_workers=2), n_qubits
    )

    assert array_terms == dict_terms


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_parallel_decompose_arrays_no_duplicate_terms(fixture):
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = fixture.padded_hamiltonian()
    seen = set()
    total = 0
    for x, z, _coeff in parallel_decompose_arrays(padded, chunk_size=2, n_workers=2):
        for xi, zi in zip(x.tolist(), z.tolist()):
            seen.add((xi, zi))
            total += 1

    assert total == len(seen), "a term was yielded more than once"


def test_parallel_decompose_arrays_yields_three_arrays_of_equal_length():
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = ALL_FIXTURES[1].padded_hamiltonian()
    for chunk in parallel_decompose_arrays(padded, chunk_size=2, n_workers=2):
        assert len(chunk) == 3
        x, z, coeff = chunk
        assert len(x) == len(z) == len(coeff)
        assert np.iscomplexobj(coeff), "coefficients must stay complex"


def test_parallel_decompose_arrays_non_hermitian_raises():
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    operator = np.zeros((4, 4), dtype=complex)
    operator[0, 0] = 1.0 + 0.5j

    with pytest.raises(ValueError, match="imaginary part"):
        list(parallel_decompose_arrays(operator, chunk_size=2, n_workers=2))


def test_checkpoint_written_by_dict_path_resumes_under_array_path(tmp_path):
    # Both functions must share one on-disk format. Asserted, not assumed.
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = ALL_FIXTURES[1].padded_hamiltonian()
    n_qubits = int(np.log2(padded.shape[0]))
    reference = fwht_pauli_terms(padded)
    checkpoint = tmp_path / "ckpt.jsonl"

    # Consume only the first chunk from the dict path, leaving a
    # partial checkpoint on disk.
    gen = parallel_decompose(padded, chunk_size=2, n_workers=2,
                             checkpoint_path=checkpoint)
    next(gen)
    gen.close()

    combined = _combine_arrays(
        parallel_decompose_arrays(padded, chunk_size=2, n_workers=2,
                                  checkpoint_path=checkpoint),
        n_qubits,
    )

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_checkpoint_written_by_array_path_resumes_under_dict_path(tmp_path):
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = ALL_FIXTURES[1].padded_hamiltonian()
    reference = fwht_pauli_terms(padded)
    checkpoint = tmp_path / "ckpt.jsonl"

    gen = parallel_decompose_arrays(padded, chunk_size=2, n_workers=2,
                                    checkpoint_path=checkpoint)
    next(gen)
    gen.close()

    combined = {}
    for chunk in parallel_decompose(padded, chunk_size=2, n_workers=2,
                                    checkpoint_path=checkpoint):
        combined.update(chunk)

    assert set(combined) == set(reference)


def test_resume_replay_still_checks_hermiticity(tmp_path):
    # The replay path is separate code from the drain loop; it is easy
    # to add the check to one and forget the other.
    from paulikit.algorithms.fwht import (
        _append_checkpoint_frame,
        _append_progress_record,
        parallel_decompose_arrays,
    )

    checkpoint = tmp_path / "ckpt.bin"
    progress = tmp_path / "ckpt.bin.parallel_progress.json"
    # A checkpoint holding one non-Hermitian term. Written as a binary
    # frame rather than a hand-written JSON line because the checkpoint
    # payload is now the chunk-framed binary format; the index dtype is
    # uint16 because _index_dtype_for_dim(4) returns uint16 for this
    # 4x4 operator, and the frame header records that width.
    _append_checkpoint_frame(
        checkpoint, 0,
        np.array([0], dtype=np.uint16),
        np.array([0], dtype=np.uint16),
        np.array([1.0 + 0.5j], dtype=complex),
        np.dtype(np.uint16),
    )
    # The progress marker is now append-only fixed-width records (one
    # 8-byte u64 chunk index per completed chunk), not JSON.
    _append_progress_record(progress, 0)

    operator = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="imaginary part"):
        list(parallel_decompose_arrays(operator, chunk_size=2, n_workers=2,
                                       checkpoint_path=checkpoint))


def test_new_api_listed_in_package_docstring():
    import paulikit

    assert "parallel_decompose" in paulikit.__doc__
    assert "parallel_decompose_arrays" in paulikit.__doc__
    assert "terms_from_arrays" in paulikit.__doc__
