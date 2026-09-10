"""Tests for the runtime auto-tuning heuristics
(paulikit.algorithms.autotune) and fwht.auto_decompose's streaming-vs-
dense decision.

Memory/cgroup/cache-detection tests monkeypatch the module's own
internal helpers rather than depending on this machine's actual
hardware/cgroup state - the formulas are what's under test, not any
particular machine's numbers.
"""

import numpy as np
import pytest

from paulikit.algorithms import autotune, fwht
from paulikit.algorithms.fwht import auto_decompose, fwht_pauli_terms
from paulikit.testing.fixtures import ALL_FIXTURES


@pytest.fixture(autouse=True)
def _reset_autotune_caches():
    """autotune's memory/chunk_size results are cached per-process
    (module-level) - reset before and after each test so tests don't
    leak state into each other via the cache."""
    autotune._cached_l2_bytes = autotune._L2_BYTES_UNSET
    autotune._cached_memory_budget_bytes = None
    yield
    autotune._cached_l2_bytes = autotune._L2_BYTES_UNSET
    autotune._cached_memory_budget_bytes = None


def test_available_memory_bytes_uses_meminfo_when_present(monkeypatch):
    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: None)
    assert autotune.available_memory_bytes() == 8 * 1024**3


def test_available_memory_bytes_falls_back_to_posix_when_no_meminfo(monkeypatch):
    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", lambda: None)
    monkeypatch.setattr(autotune, "_posix_available_physical_bytes", lambda: 4 * 1024**3)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: None)
    assert autotune.available_memory_bytes() == 4 * 1024**3


def test_available_memory_bytes_uses_smaller_of_physical_and_cgroup(monkeypatch):
    # The correctness-critical case for shared HPC nodes: a cgroup cap
    # smaller than physically-available memory must win.
    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: 2 * 1024**3)
    assert autotune.available_memory_bytes() == 2 * 1024**3


def test_available_memory_bytes_ignores_larger_cgroup_limit(monkeypatch):
    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", lambda: 2 * 1024**3)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: 16 * 1024**3)
    assert autotune.available_memory_bytes() == 2 * 1024**3


def test_available_memory_bytes_is_cached_per_process(monkeypatch):
    calls = []

    def fake_meminfo():
        calls.append(1)
        return 8 * 1024**3

    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", fake_meminfo)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: None)

    first = autotune.available_memory_bytes()
    second = autotune.available_memory_bytes()
    assert first == second
    assert len(calls) == 1, "expected the underlying probe to run exactly once (cached)"


def test_recommended_chunk_size_uses_declared_size_when_no_probe(monkeypatch):
    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", lambda: 256 * 1024)
    # dim=1024 -> 1024*16 = 16384 bytes/row; 262144 // 16384 = 16
    assert autotune.recommended_chunk_size(dim=1024) == 16


def test_recommended_chunk_size_respects_floor(monkeypatch):
    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", lambda: 1024)  # tiny
    # dim large enough that the cache-driven size would be < floor
    assert autotune.recommended_chunk_size(dim=4096) == autotune._min_chunk_size_floor(4096)


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (256, 8),  # below smallest anchor - clamps to it, no extrapolation
        (512, 8),  # N=25 anchor, exact
        (2048, 8),  # N=50 anchor, exact
        (16384, 2),  # N=150 anchor, exact
        (32768, 1),  # N=200 anchor, exact
        (65536, 1),  # above largest anchor - clamps to it, no extrapolation
    ],
)
def test_min_chunk_size_floor_matches_measured_anchors(dim, expected):
    # Regression test for the real N=25/50/150/200 measurements of
    # chunk-size-floor scale dependence - pins the exact anchor points
    # so a future edit can't silently drift away from measured data
    # without a deliberate, documented decision.
    assert autotune._min_chunk_size_floor(dim) == expected


def test_min_chunk_size_floor_interpolates_monotonically_in_the_unmeasured_gap():
    # The dim=2048..16384 gap has no real measurement - this only pins
    # that the interpolation stays monotonically non-increasing and
    # within the bracketing anchors' range, not any specific
    # unmeasured value.
    dims = [2048, 4096, 8192, 16384]
    floors = [autotune._min_chunk_size_floor(d) for d in dims]
    assert floors == sorted(floors, reverse=True)
    assert floors[0] == 8
    assert floors[-1] == 2
    assert all(2 <= f <= 8 for f in floors)


def test_recommended_chunk_size_falls_back_to_32_when_nothing_works(monkeypatch):
    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", lambda: None)
    assert autotune.recommended_chunk_size(dim=64) == 32


def test_recommended_chunk_size_is_cached_per_process(monkeypatch):
    calls = []

    def fake_declared():
        calls.append(1)
        return 256 * 1024

    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", fake_declared)

    first = autotune.recommended_chunk_size(dim=64)
    second = autotune.recommended_chunk_size(dim=64)
    assert first == second
    assert len(calls) == 1, (
        "the L2-boundary lookup itself must still be cached (dim-"
        "independent, expensive) even though the returned chunk_size "
        "is recomputed per call"
    )


def test_recommended_chunk_size_recomputes_per_dim_despite_l2_cache(monkeypatch):
    # Regression test for a real bug (REVIEW_NOTES.md, found 2026-09-04):
    # recommended_chunk_size(dim) used to cache its FINAL RESULT keyed on
    # nothing, so a process calling it with dim=512 and then dim=16384
    # wrongly got dim=512's answer for both. Only the dim-independent
    # L2-boundary lookup should be cached - the chunk_size math itself
    # must run fresh every call.
    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", lambda: 256 * 1024)

    small_dim_result = autotune.recommended_chunk_size(dim=512)
    large_dim_result = autotune.recommended_chunk_size(dim=16384)
    assert small_dim_result != large_dim_result, (
        "different dims must not silently share a cached chunk_size"
    )
    # 512*16=8192 bytes/row -> 262144//8192 = 32
    assert small_dim_result == 32
    # 16384*16=262144 bytes/row -> 262144//262144 = 1, floored to the
    # N=150 anchor's own floor (2)
    assert large_dim_result == autotune._min_chunk_size_floor(16384)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_auto_decompose_dense_path_matches_fwht_pauli_terms(monkeypatch, fixture):
    # Force the dense path by giving a huge memory budget.
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 2**40)
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    result = auto_decompose(padded)
    assert isinstance(result, dict)
    assert set(result) == set(reference)
    for label in reference:
        assert result[label] == pytest.approx(reference[label], abs=1e-9)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_auto_decompose_streaming_path_matches_fwht_pauli_terms(monkeypatch, fixture):
    # Force the streaming path by giving a near-zero memory budget.
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 1)
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 4)
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    result = auto_decompose(padded)
    assert not isinstance(result, dict)
    combined = {}
    for chunk_dict in result:
        combined.update(chunk_dict)

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_auto_decompose_return_type_is_distinguishable_via_isinstance(monkeypatch):
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 2**40)
    padded = ALL_FIXTURES[0].padded_hamiltonian()
    dense_result = auto_decompose(padded)
    assert isinstance(dense_result, dict)

    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 1)
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 4)
    streaming_result = auto_decompose(padded)
    assert not isinstance(streaming_result, dict)


def test_dense_memory_estimate_regression_at_measured_n150_scale(monkeypatch):
    # Regression test for a real bug found by direct measurement at
    # N=100/N=150 scale: the original dim**2*16 estimate with
    # _DENSE_MEMORY_SAFETY_FRACTION=0.5 underestimated the dense path's
    # real N=150 peak memory usage by 3x+, in the unsafe direction -
    # the real dense path failed to complete under memory caps as high
    # as ~11.4 GiB while the old formula said it would fit in 4 GiB.
    # This pins the fixed formula's behavior at N=150's real dim/budget
    # numbers from that same measurement, so an accidental revert to
    # the old (multiplier=1x, fraction=0.5) combination is caught here
    # rather than only being caught by another expensive real N=150 run.
    dim_n150 = 16384
    real_measured_budget_bytes = 11_447_480_320  # from that remeasurement

    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: real_measured_budget_bytes)

    estimated_dense_bytes = dim_n150 * dim_n150 * 16 * fwht._DENSE_MEMORY_MULTIPLIER
    threshold = real_measured_budget_bytes * fwht._DENSE_MEMORY_SAFETY_FRACTION
    assert estimated_dense_bytes > threshold, (
        "the fixed formula must NOT choose the dense path at N=150's real "
        "dim/budget numbers - the real dense path was measured to fail "
        "even under an ~11.4 GiB cap, well above what this budget offers"
    )


def test_dense_memory_multiplier_and_safety_fraction_are_conservative():
    # Pins the specific constants chosen after the real-measurement fix
    # (see their own comments in fwht.py for the full derivation) -
    # guards against an accidental relaxation back toward the disproven
    # 1x/0.5 combination without a deliberate, documented decision.
    assert fwht._DENSE_MEMORY_MULTIPLIER >= 6.0
    assert fwht._DENSE_MEMORY_SAFETY_FRACTION <= 0.2


def test_recommended_chunk_size_thread_safe_single_underlying_call(monkeypatch):
    # Regression test for the concurrent-cache-population race found
    # while investigating the cache-probe's idempotency: without a
    # lock, multiple threads could all observe an empty
    # _cached_l2_bytes and all invoke the underlying probe/detection
    # concurrently. An artificial delay inside the (mocked) detection
    # widens the race window so this test reliably exercises it rather
    # than depending on timing luck.
    import threading
    import time

    call_count = 0
    call_count_lock = threading.Lock()

    def slow_declared_l2():
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        time.sleep(0.05)  # widen the race window
        return 256 * 1024

    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", slow_declared_l2)

    results = []

    def worker():
        results.append(autotune.recommended_chunk_size(dim=1024))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count == 1, (
        f"expected the underlying detection to run exactly once across "
        f"8 concurrent threads, ran {call_count} times - the cache "
        f"population lock is not preventing the race"
    )
    assert len(set(results)) == 1, "all threads must observe the same result"


def test_available_memory_bytes_thread_safe_single_underlying_call(monkeypatch):
    import threading
    import time

    call_count = 0
    call_count_lock = threading.Lock()

    def slow_meminfo():
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        time.sleep(0.05)
        return 8 * 1024**3

    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", slow_meminfo)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: None)

    results = []

    def worker():
        results.append(autotune.available_memory_bytes())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count == 1, (
        f"expected the underlying detection to run exactly once across "
        f"8 concurrent threads, ran {call_count} times - the cache "
        f"population lock is not preventing the race"
    )
    assert len(set(results)) == 1, "all threads must observe the same result"
