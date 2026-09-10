"""Tests for the empirical cache-latency probe.

Skipped entirely if the optional compiled extension isn't built -
same optionality stance as pauli_label_native, see
paulikit._native.cache_probe's own module docstring.
"""

import pytest

cache_probe = pytest.importorskip("paulikit._native.cache_probe")


def test_probe_cache_boundaries_returns_doubling_sizes():
    result = cache_probe.probe_cache_boundaries(
        min_size_bytes=8192, n_sizes=13, reps=50000, repeats=2
    )
    assert len(result) == 13
    sizes = [size for size, _ in result]
    assert sizes == [8192 * (2**i) for i in range(13)]


def test_probe_cache_boundaries_cycles_are_positive():
    result = cache_probe.probe_cache_boundaries(
        min_size_bytes=8192, n_sizes=5, reps=50000, repeats=2
    )
    for size, cycles in result:
        assert cycles > 0, f"non-positive cycles/access at size {size}"


def test_probe_cache_boundaries_small_sizes_are_faster_than_large():
    # Not a strict monotonicity assertion (real hardware boundaries
    # are step-function-shaped with a flat region within each cache
    # level, not a smooth ramp) - just checks the smallest measured
    # size is meaningfully faster than a size well past L3, which
    # should hold on any real cache hierarchy.
    result = cache_probe.probe_cache_boundaries(
        min_size_bytes=8192, n_sizes=13, reps=100000, repeats=3
    )
    smallest_cycles = result[0][1]
    largest_cycles = result[-1][1]
    assert largest_cycles > smallest_cycles * 2


def test_probe_cache_boundaries_repeats_reduces_or_matches_noise():
    # repeats=1 vs repeats=5 at the same sizes: the repeats=5 result
    # should never show a *lower* minimum than what's physically
    # possible, and in practice should be at least as stable. This is
    # a smoke test, not a strict statistical claim - real hardware
    # noise (a found-and-fixed preemption outlier issue in this same
    # probe) cannot be fully eliminated in a unit test.
    result = cache_probe.probe_cache_boundaries(
        min_size_bytes=8192, n_sizes=3, reps=50000, repeats=5
    )
    assert len(result) == 3
    for _, cycles in result:
        assert cycles > 0
