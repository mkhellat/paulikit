# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Tests for the threaded drain (``executor="thread"``).

The threaded path exists because both compiled kernels release the
GIL, so per-chunk work runs genuinely concurrently and no arrays cross
a process boundary. What must be guaranteed is that changing the drain
changes *nothing* observable: same terms, same contract, same
checkpoint format.

Several tests also call ``executor="process"`` to cross-check drains.
Those forks can trip CPython 3.12+'s multi-threaded-fork
``DeprecationWarning`` under pytest (OpenBLAS / prior thread pools);
that warning is filtered in ``pyproject.toml``, not ignored ad hoc here.
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


@pytest.mark.parametrize("n_workers", [1, 2, 5, 8])
def test_eager_threads_does_not_change_the_answer(n_workers):
    """eager_threads only changes WHEN worker OS threads are created
    (all up front via a barrier, instead of ThreadPoolExecutor's own
    lazy one-per-submit() spin-up) - never what they compute. Chunks
    are independent regardless of thread-creation timing, so the
    result must be bit-identical to the default (lazy) path at every
    n_workers, including 1 (the single-worker fast path, which never
    constructs a ThreadPoolExecutor at all and so never reaches the
    eager_threads branch) and a count above the 4-core dev machine's
    physical core count (8), to also exercise oversubscription."""
    op = _operator(20)
    lazy = _collect(
        op, chunk_size=2, atol=1e-9, executor="thread", n_workers=n_workers,
        eager_threads=False,
    )
    eager = _collect(
        op, chunk_size=2, atol=1e-9, executor="thread", n_workers=n_workers,
        eager_threads=True,
    )
    for left, right in zip(lazy, eager):
        assert np.array_equal(left, right)


def test_eager_threads_has_no_effect_under_process_executor():
    """eager_threads is read only inside the executor == "thread"
    branch - passing it with executor="process" must be accepted
    (not raise) and change nothing, since ProcessPoolExecutor workers
    are never lazily-vs-eagerly spun up by this code path at all."""
    op = _operator(10)
    without = _collect(
        op, chunk_size=2, atol=1e-9, executor="process", eager_threads=False,
    )
    with_flag = _collect(
        op, chunk_size=2, atol=1e-9, executor="process", eager_threads=True,
    )
    for left, right in zip(without, with_flag):
        assert np.array_equal(left, right)


def test_eager_threads_does_not_deadlock_when_n_workers_is_clamped():
    """max_in_flight's own clamp (fwht.py) can reduce the EFFECTIVE
    n_workers used to construct ThreadPoolExecutor below the value the
    caller requested, when there are fewer chunks than requested
    workers - e.g. a tiny operator with n_workers=8 but only 2 chunks
    total. The eager_threads barrier is sized from that SAME
    already-clamped n_workers (it runs after the clamp, inside the
    executor == "thread" branch), so it must never deadlock waiting
    for more parties than the pool actually has. A tiny N=2 operator
    at chunk_size=2 with the same operator's nonzero count producing
    far fewer than 8 chunks reproduces the clamp."""
    op = _operator(2)
    result = _collect(
        op, chunk_size=2, atol=1e-9, executor="thread", n_workers=8,
        eager_threads=True,
    )
    # Correctness, not just "it returned": must match the unclamped,
    # non-eager reference.
    reference = _collect(
        op, chunk_size=2, atol=1e-9, executor="thread", n_workers=8,
        eager_threads=False,
    )
    for left, right in zip(result, reference):
        assert np.array_equal(left, right)
