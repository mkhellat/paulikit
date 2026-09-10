# Package layout

```
src/paulikit/
    __init__.py           Package metadata, public API summary.
    hamiltonian.py         Coupled-oscillator Hamiltonian construction
                            (NumPy implementation, with an optional
                            scipy.sparse output path).
    pauli_utils.py          Minimal, dependency-free Pauli-matrix
                            helpers (label <-> matrix, reconstruction).
    algorithms/
        __init__.py
        fwht.py             The Fast Walsh-Hadamard Transform based
                            decomposition algorithm (
                            more algorithms planned here).
        autotune.py          Runtime auto-tuning for chunk size
                            (against the machine's measured cache
                            hierarchy) and the streaming-vs-dense
                            decision (against available memory).
    testing/
        __init__.py
        fixtures.py         Known-good Hamiltonians and their
                            independently-verified expected Pauli
                            decompositions, for use by any algorithm's
                            tests.
    _native/                Optional compiled extensions used by
                            `algorithms/fwht.py` and `autotune.py`
                            when available, each with a pure-Python or
                            NumPy fallback otherwise — see "Native
                            extension" in the README. Four extensions,
                            gated by three `meson.options` features:
                            `pauli_label_native` (Cython/C++ + oneTBB,
                            option `native`), `wht_native` and
                            `coeffs_native` (Cython/C, option
                            `wht_kernel`), and `cache_probe`
                            (Cython/C, option `cache_probe`).
    cli.py                  Command-line interface wiring the above
                            together into subcommands.
    meson.build             Per-directory Meson build rules (one per
                            subpackage above, plus a top-level
                            `meson.build` and `meson.options` at the
                            repository root of this package).
tests/
    test_fixtures.py        Self-consistency checks for the fixtures.
    test_fwht.py             Correctness tests for algorithms/fwht.py.
    (and further test modules covering autotuning, checkpointing,
    streaming, parallel decomposition, sparse input, and the native
    kernels — see the `tests/` directory for the full list).
```

`hamiltonian.py` and `pauli_utils.py` sit at the package root (not
under `algorithms/`) because they are not algorithm-specific: every
current and planned decomposition algorithm needs the same Hamiltonian
construction and the same Pauli-matrix utilities.


