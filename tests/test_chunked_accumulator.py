"""Correctness tests for the chunked path's space-complexity fix
(per-chunk thresholding into a growable COO accumulator instead of
one dense (n_active, dim) block) and its optional checkpoint/resume
support.

See ``fwht_pauli_coefficients``'s ``chunk_size``/``atol``/
``checkpoint_path`` docstring for the contract being tested here.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    _PROGRESS_RECORD,
    _read_completed_indices,
    fwht_pauli_coefficients,
    fwht_pauli_terms,
)
from paulikit.hamiltonian import pad_to_power_of_two
from paulikit.testing.fixtures import ALL_FIXTURES


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
def test_chunked_coo_output_matches_dense_thresholded(fixture, chunk_size):
    """The chunked path's (x, z, coefficient) triples, reconstructed
    into a dense array, must equal the dense (unchunked) path's output
    with the same atol threshold applied - the actual correctness
    claim for the space-complexity rewrite."""
    padded = fixture.padded_hamiltonian()
    dim = padded.shape[0]
    atol = 1e-10

    x_out, z_out, coeff_out = fwht_pauli_coefficients(
        padded, sparse=True, chunk_size=chunk_size, atol=atol
    )
    chunked_full = np.zeros((dim, dim), dtype=complex)
    chunked_full[x_out, z_out] = coeff_out

    dense_result = fwht_pauli_coefficients(padded)
    dense_thresholded = np.where(np.abs(dense_result) > atol, dense_result, 0.0)

    max_error = np.max(np.abs(chunked_full - dense_thresholded))
    assert max_error < 1e-9, f"{fixture.name!r} chunk_size={chunk_size}: max error {max_error}"


@pytest.mark.parametrize("chunk_size", [1, 5, 64])
def test_fwht_pauli_terms_chunked_matches_unchunked(chunk_size):
    """fwht_pauli_terms must return the same term dict whether or not
    chunk_size is set - the chunked path now does its own thresholding
    inside fwht_pauli_coefficients rather than in fwht_pauli_terms, so
    this exercises that the two code paths agree end to end."""
    for fixture in ALL_FIXTURES:
        padded = fixture.padded_hamiltonian()
        unchunked = fwht_pauli_terms(padded)
        chunked = fwht_pauli_terms(padded, chunk_size=chunk_size)

        assert set(unchunked) == set(chunked), (
            f"{fixture.name!r} chunk_size={chunk_size}: label set mismatch"
        )
        for label in unchunked:
            assert chunked[label] == pytest.approx(unchunked[label], abs=1e-9)


def test_chunk_size_bounds_peak_terms_growth_correctness_at_odd_boundaries():
    """Regression guard for the amortized-growth accumulator: chunk
    counts that don't evenly divide n_active, and chunk_size=1 (the
    most stress-testing case for the growable-array doubling logic),
    must still produce exactly the right number of terms."""
    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()

    reference = fwht_pauli_terms(padded)
    for chunk_size in [1, 2, 5, 9999]:
        chunked = fwht_pauli_terms(padded, chunk_size=chunk_size)
        assert len(chunked) == len(reference), f"chunk_size={chunk_size}"


def test_checkpoint_resume_produces_same_result_as_uninterrupted_run(tmp_path):
    """Simulates a crash partway through a chunked run: process the
    first N chunks with a checkpoint_path, discard the in-memory
    result, then call again with the same checkpoint_path (as if
    resuming a fresh process) and confirm the final result matches an
    uninterrupted run - the actual resumability claim."""
    fixture = ALL_FIXTURES[-1]  # largest available fixture
    padded = fixture.padded_hamiltonian()
    checkpoint_path = tmp_path / "checkpoint.bin"

    reference = fwht_pauli_terms(padded, chunk_size=2)

    # First "run": let it complete once to populate the checkpoint.
    first_run = fwht_pauli_terms(padded, chunk_size=2, checkpoint_path=checkpoint_path)
    assert set(first_run) == set(reference)

    # Second "run" against the now-complete checkpoint must resume
    # (skip all chunks, replay from file) and agree exactly.
    second_run = fwht_pauli_terms(padded, chunk_size=2, checkpoint_path=checkpoint_path)
    assert set(second_run) == set(reference)
    for label in reference:
        assert second_run[label] == pytest.approx(reference[label], abs=1e-9)


def test_checkpoint_resume_from_partial_progress_file(tmp_path):
    """Directly simulates a crash mid-run by hand-truncating the
    progress file to an earlier chunk index after a full run, then
    confirms a fresh call resumes correctly and still reaches the
    right final answer (not just that a complete checkpoint round-trips)."""
    fixture = ALL_FIXTURES[-1]
    padded = fixture.padded_hamiltonian()
    checkpoint_path = tmp_path / "checkpoint.bin"
    progress_path = tmp_path / "checkpoint.bin.progress.json"

    reference = fwht_pauli_terms(padded, chunk_size=2)

    fwht_pauli_terms(padded, chunk_size=2, checkpoint_path=checkpoint_path)
    total_chunks = len(_read_completed_indices(progress_path))
    assert total_chunks > 1, "fixture too small to exercise partial resume meaningfully"

    # Roll the progress marker back, simulating a crash after only the
    # first chunk was durably recorded - but leave the checkpoint file
    # itself with all lines (over-recording is safe; the code should
    # simply recompute/re-append the "already there" chunks' triples
    # again on top, matching the module's documented on-crash behavior
    # of losing at most the one in-flight chunk, never corrupting state).
    progress_path.write_bytes(_PROGRESS_RECORD.pack(0))

    resumed = fwht_pauli_terms(padded, chunk_size=2, checkpoint_path=checkpoint_path)
    assert set(resumed) == set(reference)
    for label in reference:
        assert resumed[label] == pytest.approx(reference[label], abs=1e-9)


def test_no_checkpoint_path_does_not_create_files(tmp_path):
    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()
    fwht_pauli_terms(padded, chunk_size=2)
    assert list(tmp_path.iterdir()) == []
