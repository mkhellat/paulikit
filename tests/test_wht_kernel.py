"""Tests for the compiled Walsh-Hadamard butterfly kernel.

The kernel is an optional build artifact, so every test here either
skips cleanly when it is absent or exercises the pure-Python fallback
directly. The central guarantee is that the two paths agree
*exactly* - the butterfly is a fixed sequence of additions and
subtractions performed in the same order, so agreement is bit-level,
not approximate, and the tests assert equality rather than closeness.
"""

import numpy as np
import pytest

import paulikit.algorithms.fwht as fwht_module
from paulikit.algorithms.fwht import _walsh_hadamard_transform_rows

wht_native = pytest.importorskip(
    "paulikit._native.wht_native",
    reason="compiled WHT kernel not built (-Dwht_kernel=disabled or no Cython)",
)


def _pure_python_wht(array):
    """Run the transform with the compiled path disabled."""
    saved = fwht_module._wht_native
    fwht_module._wht_native = None
    try:
        return _walsh_hadamard_transform_rows(array.copy(), overwrite_input=True)
    finally:
        fwht_module._wht_native = saved


@pytest.mark.parametrize("n_qubits", range(0, 15))
@pytest.mark.parametrize("rows", [1, 2, 3, 8])
def test_kernel_matches_pure_python_exactly(n_qubits, rows):
    """Bit-identical agreement across every dim from 1 to 16384."""
    dim = 1 << n_qubits
    rng = np.random.default_rng(n_qubits * 100 + rows)
    a = rng.standard_normal((rows, dim)) + 1j * rng.standard_normal((rows, dim))

    expected = _pure_python_wht(a)
    got = _walsh_hadamard_transform_rows(a.copy(), overwrite_input=True)

    assert np.array_equal(expected, got), (
        f"compiled and pure-Python paths disagree at dim={dim}, rows={rows}"
    )


def test_transform_is_its_own_inverse_up_to_scale():
    """(H^{tensor n})^2 == 2^n * I - the convention the kernel and the
    phase step jointly rely on. An independent check that does not
    reference the Python implementation at all."""
    dim = 1024
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, dim)) + 1j * rng.standard_normal((2, dim))

    once = _walsh_hadamard_transform_rows(a.copy(), overwrite_input=True)
    twice = _walsh_hadamard_transform_rows(once, overwrite_input=True)

    np.testing.assert_allclose(twice, a * dim, rtol=1e-12, atol=1e-12)


def test_known_hadamard_values():
    """A hand-checkable case: the transform of a basis vector is the
    corresponding row of signs, and of a constant vector is a spike."""
    e0 = np.zeros((1, 4), dtype=complex)
    e0[0, 0] = 1.0
    np.testing.assert_array_equal(
        _walsh_hadamard_transform_rows(e0, overwrite_input=True),
        np.ones((1, 4), dtype=complex),
    )

    ones = np.ones((1, 4), dtype=complex)
    out = _walsh_hadamard_transform_rows(ones, overwrite_input=True)
    np.testing.assert_array_equal(out, np.array([[4, 0, 0, 0]], dtype=complex))


@pytest.mark.parametrize("tile", [1, 2, 4, 16, 256, 1024, 4096])
def test_every_tile_gives_the_same_answer(tile):
    """The tile is a pure cache-blocking parameter: it changes the
    order tiles are visited, never the arithmetic. Any tile must give
    the identical result, which is what makes runtime tuning safe."""
    dim, rows = 4096, 2
    rng = np.random.default_rng(7)
    a = rng.standard_normal((rows, dim)) + 1j * rng.standard_normal((rows, dim))

    reference = _pure_python_wht(a)
    got = a.copy()
    wht_native.wht_rows_inplace(got, tile)

    assert np.array_equal(reference, got), f"tile={tile} changed the result"


def test_tile_is_derived_from_the_machine_not_hardcoded():
    """The kernel sizes its tile from this machine's L1 via sysconf.
    Assert the derived value is sane and a power of two rather than
    asserting a specific number, which would be exactly the
    machine-specific constant this design avoids."""
    for dim in (256, 1024, 4096, 16384):
        tile = wht_native.tile_for_cache(dim)
        assert 1 <= tile <= dim
        assert tile & (tile - 1) == 0, f"tile {tile} is not a power of two"

    # Stable within a process and across calls - no probe noise.
    assert wht_native.tile_for_cache(16384) == wht_native.tile_for_cache(16384)

    fallback = wht_native.fallback_tile()
    assert fallback >= 1 and fallback & (fallback - 1) == 0


def test_zero_tile_means_derive_it():
    """tile=0 is the documented "choose for me" value and must behave
    exactly like the derived tile."""
    dim, rows = 2048, 2
    rng = np.random.default_rng(3)
    a = rng.standard_normal((rows, dim)) + 1j * rng.standard_normal((rows, dim))

    auto = a.copy()
    wht_native.wht_rows_inplace(auto, 0)
    explicit = a.copy()
    wht_native.wht_rows_inplace(explicit, wht_native.tile_for_cache(dim))

    assert np.array_equal(auto, explicit)


def test_operates_in_place_and_returns_the_same_buffer():
    """Bounded memory depends on the kernel not allocating: it must
    mutate the caller's array and hand back that same object."""
    a = np.ones((2, 64), dtype=complex)
    returned = wht_native.wht_rows_inplace(a, 0)
    assert returned is a
    assert a[0, 0] == 64.0  # transformed in place, not left untouched


def test_empty_and_degenerate_inputs():
    """Zero rows and dim=1 are valid no-ops, not errors - a chunk can
    legitimately be empty."""
    empty = np.zeros((0, 16), dtype=complex)
    assert wht_native.wht_rows_inplace(empty, 0).shape == (0, 16)

    single = np.array([[3.0 + 4.0j]], dtype=complex)
    out = wht_native.wht_rows_inplace(single.copy(), 0)
    np.testing.assert_array_equal(out, single)


def test_rejects_inputs_it_cannot_handle():
    """The kernel reads a raw buffer, so it must refuse anything whose
    layout or dtype would make that read wrong - silently accepting
    them would corrupt memory."""
    with pytest.raises(ValueError, match="2-D"):
        wht_native.wht_rows_inplace(np.zeros(8, dtype=complex), 0)

    with pytest.raises(ValueError, match="complex128"):
        wht_native.wht_rows_inplace(np.zeros((2, 8), dtype=np.float64), 0)

    with pytest.raises(ValueError, match="C-contiguous"):
        wht_native.wht_rows_inplace(np.zeros((2, 16), dtype=complex)[:, ::2], 0)

    with pytest.raises(ValueError, match="power of two"):
        wht_native.wht_rows_inplace(np.zeros((2, 6), dtype=complex), 0)

    with pytest.raises(ValueError, match="non-negative"):
        wht_native.wht_rows_inplace(np.zeros((2, 8), dtype=complex), -1)


def test_rows_are_independent():
    """No cross-row coupling: this is what keeps chunks independent
    and the caller safely parallelizable."""
    rng = np.random.default_rng(11)
    dim = 512
    rows = rng.standard_normal((4, dim)) + 1j * rng.standard_normal((4, dim))

    together = _walsh_hadamard_transform_rows(rows.copy(), overwrite_input=True)
    for i in range(4):
        alone = _walsh_hadamard_transform_rows(
            rows[i : i + 1].copy(), overwrite_input=True
        )
        assert np.array_equal(together[i : i + 1], alone)


def test_overwrite_input_false_leaves_caller_array_untouched():
    """The compiled path must honour the same copy semantics as the
    Python one, or callers relying on overwrite_input=False silently
    lose data."""
    a = np.ones((2, 32), dtype=complex)
    original = a.copy()
    out = _walsh_hadamard_transform_rows(a, overwrite_input=False)
    assert np.array_equal(a, original), "input was mutated despite the flag"
    assert not np.array_equal(out, original)
