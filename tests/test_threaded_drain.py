"""Tests for the threaded drain (``executor="thread"``).

The threaded path exists because both compiled kernels release the
GIL, so per-chunk work runs genuinely concurrently and no arrays cross
a process boundary. What must be guaranteed is that changing the drain
changes *nothing* observable: same terms, same contract, same
checkpoint format.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    fwht_pauli_coefficients,
    parallel_decompose_arrays,
)
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def _operator(n_oscillators):
    h = build_hamiltonian(
        n_oscillators,
        _default_spring_constants(n_oscillators),
        _default_masses(n_oscillators),
        sparse=True,
    )
    padded, _ = pad_to_power_of_two(h, sparse=True)
    return padded


def _collect(operator, **kwargs):
    xs, zs, cs = [], [], []
    for x, z, c in parallel_decompose_arrays(operator, **kwargs):
        xs.append(x)
        zs.append(z)
        cs.append(c)
    x = np.concatenate(xs) if xs else np.empty(0, dtype=np.intp)
    z = np.concatenate(zs) if zs else np.empty(0, dtype=np.intp)
    c = np.concatenate(cs) if cs else np.empty(0, dtype=complex)
    order = np.lexsort((z, x))
    return x[order], z[order], c[order]


@pytest.mark.parametrize("n_oscillators", [10, 20, 30])
def test_thread_matches_sequential_exactly(n_oscillators):
    """The drain is a scheduling choice, not an arithmetic one, so the
    output must be bit-identical - not merely close."""
    op = _operator(n_oscillators)

    x0, z0, c0 = fwht_pauli_coefficients(
        op, sparse=True, chunk_size=2, atol=1e-9
    )
    order = np.lexsort((z0, x0))
    x0, z0, c0 = x0[order], z0[order], c0[order]

    x1, z1, c1 = _collect(op, chunk_size=2, atol=1e-9, executor="thread")

    assert np.array_equal(x0, x1)
    assert np.array_equal(z0, z1)
    assert np.array_equal(c0, c1)


@pytest.mark.parametrize("n_oscillators", [10, 20, 30])
def test_thread_matches_process_exactly(n_oscillators):
    """The two executors must agree with each other, too."""
    op = _operator(n_oscillators)
    a = _collect(op, chunk_size=2, atol=1e-9, executor="process")
    b = _collect(op, chunk_size=2, atol=1e-9, executor="thread")
    for left, right in zip(a, b):
        assert np.array_equal(left, right)


@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_thread_count_does_not_change_the_answer(n_workers):
    """Chunks are independent, so the worker count is free to vary."""
    op = _operator(20)
    ref = _collect(op, chunk_size=2, atol=1e-9, executor="thread", n_workers=1)
    got = _collect(
        op, chunk_size=2, atol=1e-9, executor="thread", n_workers=n_workers
    )
    for left, right in zip(ref, got):
        assert np.array_equal(left, right)


def test_thread_checkpoint_is_interchangeable_with_process(tmp_path):
    """Checkpoints must stay in one format regardless of drain, so a
    run started under one executor can resume under the other."""
    op = _operator(20)

    proc_ckpt = tmp_path / "proc.ckpt"
    thread_ckpt = tmp_path / "thread.ckpt"
    _collect(op, chunk_size=2, atol=1e-9, executor="process",
             checkpoint_path=proc_ckpt)
    _collect(op, chunk_size=2, atol=1e-9, executor="thread",
             checkpoint_path=thread_ckpt)

    assert proc_ckpt.exists() and thread_ckpt.exists()

    # Resuming from the process-written checkpoint under the thread
    # executor must reproduce the full result.
    resumed = _collect(op, chunk_size=2, atol=1e-9, executor="thread",
                       checkpoint_path=proc_ckpt)
    expected = _collect(op, chunk_size=2, atol=1e-9, executor="process")
    for left, right in zip(expected, resumed):
        assert np.array_equal(left, right)


def test_rejects_unknown_executor():
    op = _operator(10)
    with pytest.raises(ValueError, match="process.*thread"):
        list(parallel_decompose_arrays(op, executor="fork-bomb"))


def test_thread_preserves_hermiticity_check():
    """assume_hermitian must still raise on a non-Hermitian operator,
    from inside the threaded drain."""
    rng = np.random.default_rng(0)
    dim = 8
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    assert not np.allclose(a, a.conj().T)

    with pytest.raises(ValueError):
        list(parallel_decompose_arrays(
            a, chunk_size=2, atol=1e-9, executor="thread",
            assume_hermitian=True,
        ))


def test_thread_leaves_worker_state_restored():
    """The threaded path sets the module-level worker state that the
    process path normally populates per worker. It must restore it, or
    a later call from inside a worker process would see foreign
    state."""
    import paulikit.algorithms.fwht as fwht_module

    before = fwht_module._parallel_worker_state
    _collect(_operator(10), chunk_size=2, atol=1e-9, executor="thread")
    assert fwht_module._parallel_worker_state is before
