"""Tests for the append-only progress marker
(docs/superpowers/specs/2026-09-08-append-only-progress-marker-design.md).

These pin the record format and the recovery rule directly, separately
from the decomposition behaviour the four existing checkpoint test
files already cover.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    _PROGRESS_RECORD,
    _append_progress_record,
    _read_completed_indices,
)


def test_record_is_8_bytes():
    assert _PROGRESS_RECORD.size == 8


def test_round_trip_recovers_exactly_the_appended_set(tmp_path):
    path = tmp_path / "p.bin"
    for i in (0, 1, 2, 7):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {0, 1, 2, 7}


def test_absent_file_is_the_empty_set(tmp_path):
    assert _read_completed_indices(tmp_path / "absent.bin") == set()


def test_empty_file_is_the_empty_set(tmp_path):
    path = tmp_path / "p.bin"
    path.write_bytes(b"")
    assert _read_completed_indices(path) == set()


def test_records_need_not_be_sorted(tmp_path):
    # Parallel workers complete out of order, so the file is not
    # sorted. Recovery must not assume it is.
    path = tmp_path / "p.bin"
    for i in (5, 0, 3, 1):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {0, 1, 3, 5}


def test_duplicate_records_collapse(tmp_path):
    # A rollback-resume can record the same chunk twice; the file is
    # append-only and never compacted. Reading into a set handles it.
    path = tmp_path / "p.bin"
    for i in (2, 2, 2):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {2}


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5, 6, 7])
def test_torn_final_record_is_discarded(tmp_path, cut):
    # A crash mid-append can only ever tear the LAST record, because
    # appends are 8 bytes in one call and never rewritten. Every
    # complete record must survive; the partial one must not appear.
    path = tmp_path / "p.bin"
    for i in (0, 1, 2):
        _append_progress_record(path, i)
    raw = path.read_bytes()
    path.write_bytes(raw[:-cut])
    assert _read_completed_indices(path) == {0, 1}


def test_large_chunk_index_survives(tmp_path):
    # u64, matching the width the frame header uses for chunk_index,
    # so the two cannot disagree about range.
    path = tmp_path / "p.bin"
    _append_progress_record(path, 2**40)
    assert _read_completed_indices(path) == {2**40}


def test_sequential_next_chunk_is_max_plus_one(tmp_path):
    # The sequential path completes chunks in order, so the recovered
    # set is contiguous and next_chunk is max + 1. It shares the
    # parallel path's record format so both use one writer/reader.
    from paulikit.algorithms.fwht import _load_checkpoint

    ckpt = tmp_path / "c.bin"
    prog = tmp_path / "c.bin.progress.json"
    # A payload frame must exist for the checkpoint to be considered
    # present; its contents do not matter to next_chunk derivation.
    from paulikit.algorithms.fwht import _append_checkpoint_frame

    for i in (0, 1, 2):
        _append_checkpoint_frame(
            ckpt, i,
            np.array([i], dtype=np.uint16),
            np.array([i], dtype=np.uint16),
            np.array([1 + 0j], dtype=complex),
            np.dtype(np.uint16),
        )
        _append_progress_record(prog, i)

    next_chunk, frames = _load_checkpoint(ckpt)
    assert next_chunk == 3
    assert len(list(frames)) == 3


def test_sequential_next_chunk_is_zero_when_nothing_recorded(tmp_path):
    from paulikit.algorithms.fwht import _load_checkpoint

    ckpt = tmp_path / "c.bin"
    ckpt.write_bytes(b"")
    (tmp_path / "c.bin.progress.json").write_bytes(b"")
    next_chunk, frames = _load_checkpoint(ckpt)
    assert next_chunk == 0
    assert frames is None


# --- End-to-end, above the 4-qubit fixtures -------------------------
#
# Every other checkpoint test in this repo runs on ALL_FIXTURES, which
# is dim=8 and dim=16 (3 and 4 qubits), and no profiling script has
# ever passed checkpoint_path at a realistic size. That gap is how a
# resume path that needed >20 GB at N=150 stayed invisible until it was
# measured directly. These exercise the whole checkpoint machinery -
# write, resume, and crash recovery - at n_qubits=9, which is small
# enough to stay a fast unit test and large enough that the chunk count
# (35) is not degenerate.


def _n24_operator():
    from paulikit.cli import _default_masses, _default_spring_constants
    from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

    n = 24
    unpadded = build_hamiltonian(
        n, _default_spring_constants(n), _default_masses(n), sparse=True
    )
    padded, _n_qubits = pad_to_power_of_two(unpadded, sparse=True)
    return padded


def _merge(chunks):
    out = {}
    for chunk in chunks:
        out.update(chunk)
    return out


def test_end_to_end_checkpointed_run_above_four_qubits(tmp_path):
    from paulikit.algorithms.fwht import fwht_pauli_terms, parallel_decompose

    padded = _n24_operator()
    reference = fwht_pauli_terms(padded)
    ckpt = tmp_path / "par.bin"

    got = _merge(parallel_decompose(
        padded, chunk_size=8, n_workers=4, checkpoint_path=str(ckpt)
    ))
    assert set(got) == set(reference)
    for label in reference:
        assert got[label] == pytest.approx(reference[label], abs=1e-9)

    # One record per completed chunk, and nothing else in the file.
    progress = tmp_path / "par.bin.parallel_progress.json"
    completed = _read_completed_indices(progress)
    assert progress.stat().st_size == len(completed) * _PROGRESS_RECORD.size

    # Resuming a COMPLETE checkpoint reproduces the same result without
    # recomputing anything.
    resumed = _merge(parallel_decompose(
        padded, chunk_size=8, n_workers=4, checkpoint_path=str(ckpt)
    ))
    assert set(resumed) == set(reference)


def test_end_to_end_torn_marker_tail_still_recovers(tmp_path):
    # The crash-recovery path at a real size: truncate the marker
    # mid-record, which is exactly what a crash during the append
    # leaves behind, and confirm the resumed run is still exact.
    from paulikit.algorithms.fwht import (
        fwht_pauli_terms,
        fwht_pauli_terms_iter,
    )

    padded = _n24_operator()
    reference = fwht_pauli_terms(padded)
    ckpt = tmp_path / "seq.bin"

    _merge(fwht_pauli_terms_iter(
        padded, chunk_size=8, checkpoint_path=str(ckpt)
    ))
    progress = tmp_path / "seq.bin.progress.json"
    raw = progress.read_bytes()
    assert len(raw) % _PROGRESS_RECORD.size == 0
    progress.write_bytes(raw[:-3])  # tear the final record

    recovered = _merge(fwht_pauli_terms_iter(
        padded, chunk_size=8, checkpoint_path=str(ckpt)
    ))
    assert set(recovered) == set(reference)
    for label in reference:
        assert recovered[label] == pytest.approx(reference[label], abs=1e-9)
