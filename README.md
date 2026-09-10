# paulikit

Exact Pauli decomposition of arbitrary complex matrices, at scales
where materialising the full coefficient set is the binding
constraint.

Any $2^n \times 2^n$ complex matrix can be written as a weighted sum
over the $4^n$ $n$-qubit Pauli strings. That decomposition is what
turns a Hamiltonian into something a quantum algorithm can consume —
it is the input to linear-combination-of-unitaries (LCU) routines,
to Hamiltonian simulation, and to variational methods. Computing it
is a fast Walsh-Hadamard transform, which is well established and
cheap in theory.

The difficulty is not the transform. It is that the output has $4^n$
entries: at 15 qubits a dense decomposition is over a billion
coefficients, and implementations that build the result in memory
before returning it run out of memory long before they run out of
time.

`paulikit` addresses that directly:

- **Streaming output.** Peak resident memory is bounded by the chunk
  size, not by the term count, so it stays roughly flat as the problem
  grows: measured peak resident set is 72 MiB for a 14-qubit
  decomposition yielding 91,652,096 terms, 89 MiB at 15 qubits
  (326,134,272 terms) and 123 MiB at 16 qubits (1,470,021,632 terms).
  An implementation requiring the caller to hold the dense
  $2^n \times 2^n$ operator needs 4 GiB, 16 GiB and 64 GiB
  respectively for the same three sizes.
- **Exhaustive verification.** Every term is checked individually —
  not sampled — against an independently derived projection oracle.
- **Checkpoint and restart.** A binary chunk-framed checkpoint cheap
  enough to leave permanently enabled, so long decompositions survive
  interruption.
- **Multi-core execution, in the shipped package.** Decomposition
  runs across multiple workers, selectable per call and from the
  command line: a thread pool that runs the compiled kernels
  concurrently, or a process pool. Chunks are independent by
  construction, and chunk sizing is tuned against measured cache
  boundaries. Scaling is characterised under a documented
  measurement protocol rather than asserted.

Hermitian and non-Hermitian input are equally supported and take the
same transform — neither is a degraded path. `assume_hermitian=True`
(the default) additionally returns real rather than complex
coefficients, and *checks* that the input really is Hermitian rather
than trusting it; pass `assume_hermitian=False` for general complex
matrices. Both cases are separately verified.

Requires Python >= 3.10; the only runtime dependency is NumPy.


## Documentation

| | |
|---|---|
| [`docs/installation.md`](docs/installation.md) | Full build and install reference |
| [`docs/tutorial.md`](docs/tutorial.md) | Step-by-step walkthrough |
| [`docs/theory.md`](docs/theory.md) | Mathematical derivation |
| [`docs/background.md`](docs/background.md) | Physical motivation |
| [`docs/non_hermitian.md`](docs/non_hermitian.md) | Non-Hermitian operators |
| [`docs/package_layout.md`](docs/package_layout.md) | Annotated source tree |


## Installation

Not yet published to PyPI. Install from source:

```bash
./configure && make          # creates a venv, generates a Makefile
make check                   # run the test suite
```

`./configure` prints a capability report (compiler, Cython, oneTBB,
cache hierarchy, NumPy's BLAS backend) and generates a Makefile with
the standard GNU targets. The build optionally compiles a native
Cython/C++ kernel; if the toolchain is unavailable it falls back to
pure Python automatically, with a warning the first time that path
runs.

See [`docs/installation.md`](docs/installation.md) for the full
reference, including editable-install sequencing and how to force the
native extension on or off.


## Usage

### Command line

Once installed, the `paulikit` console script is available:

```bash
paulikit --help
paulikit decompose --n-oscillators 4 --show-terms
paulikit benchmark --n-oscillators 2 4 8 16 30
paulikit regenerate-fixtures
```

Run `paulikit <subcommand> --help` for full details on each.

### As a library

```python
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_terms

spring_constants = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
masses = [1.0, 2.0]

H = build_hamiltonian(n_oscillators=2, spring_constants=spring_constants, masses=masses)
H_padded, n_qubits = pad_to_power_of_two(H)

terms = fwht_pauli_terms(H_padded)  # {"IXI": -0.556..., "XII": -0.354..., ...}
```


## Package layout

```
src/paulikit/
    hamiltonian.py      Coupled-oscillator Hamiltonian construction
    pauli_utils.py      Pauli-matrix helpers (label <-> matrix)
    algorithms/fwht.py  The decomposition algorithm
    testing/fixtures.py Known-good operators and expected outputs
    _native/            Optional compiled kernel, pure-Python fallback
    cli.py              Command-line interface
tests/                  Test suite (pytest)
verification/           Exhaustive correctness runs and their artifacts
docs/                   Tutorial, theory, background, installation
```

See [`docs/package_layout.md`](docs/package_layout.md) for the
annotated tree.


## Running the tests

```bash
pytest
```

(from this directory; `pyproject.toml` sets `testpaths = ["tests"]`,
and the package must be installed — `pip install -e ".[test]"` — for
imports to resolve).


## Algorithms implemented

### Fast Walsh-Hadamard Transform (FWHT) — `paulikit.algorithms.fwht`

$O(N^2 \log N)$ for an $N \times N$ matrix — equivalently
$O(n \cdot 4^n)$ for $n$ qubits, since $N = 2^n$. Note the two
symbols differ by an exponential: $n$ counts qubits everywhere else
in this file, $N$ is the matrix side.

Decomposition by Walsh-Hadamard transform is established practice
rather than novel — PennyLane and Classiq both use it, and it is
treated at length in
[Pauli decomposition via the fast Walsh-Hadamard transform](https://iopscience.iop.org/article/10.1088/1367-2630/adb44d).
What is original here is the implementation, not the method: the three
steps (XOR-index gather, Walsh-Hadamard transform, phase-factor
multiplication) were re-derived from the symplectic (X/Z)
representation of Pauli operators and checked against a
definition-level brute-force decomposition before being written in
fast form — see `algorithms/fwht.py`'s module docstring for the
derivation. The contribution of this package is making that
computation memory-bounded, checkpointable, and verified at scale.

Unit-level checks on this algorithm (see `tests/test_fwht.py`; the
whole-package correctness evidence is under **Correctness** below):
- Against a from-scratch brute-force reference on random Hermitian
  matrices ($n=1..4$ qubits): exact match to floating-point
  precision.
- Against `testing.fixtures.ALL_FIXTURES` (real coupled-oscillator
  Hamiltonians at $N=2$, $N=4$): exact label-set and coefficient
  match.

Planned: Tensorized Pauli Decomposition (TPD), PHASE,
and C-ported variants of whichever algorithm profiling identifies as
worth porting — this is why `algorithms/` is a subpackage rather than
a single module.


## Correctness

`paulikit`'s output is verified three ways:

- **Exhaustive projection.** Every term of a decomposition is checked
  individually against an independently derived projection oracle —
  not sampled — up to 91,652,096 terms at 14 qubits. The oracle
  computes $\operatorname{Tr}(H P^{\dagger}) / \text{dim}$ directly
  from the projection formula, so it shares no code path with the
  transform it checks. Artifacts and method:
  [`verification/`](https://codeberg.org/beavernets/paulikit/src/branch/main/verification).
- **Cross-implementation.** Where PennyLane's `qml.pauli_decompose`
  can also run, both agree exactly on term count and on coefficients
  within tolerance. PennyLane is a test-only dependency and is never
  imported by `paulikit.algorithms`.
- **Regression suite.** 234 tests, including crash-recovery and
  checkpoint-format cases.

No performance comparison is published here. Benchmark tables in a
README go stale as either implementation changes, and any figure worth
citing has to be replicated, interleaved, thermally controlled and
tested for significance — which a hand-maintained table cannot
guarantee over time. Measured figures, the protocol behind them, and
the raw data are not published in this repository.


## Status

Alpha. The API is usable and the correctness evidence is strong, but
the version is 0.x and signatures may still change.

Implemented and verified: the FWHT decomposition with a native
label kernel, sparsity-aware coefficients, streaming output with
bounded memory, chunked and parallel execution with cache-aware
auto-tuning, binary checkpoint/restart, and exhaustive verification to
91,652,096 terms at 14 qubits.

Known gaps:

- Prebuilt wheels are not yet published, so the native extension
  remains an optional accelerator rather than a hard requirement.
- CPU pinning and topology detection are Linux-only, with a documented
  fallback elsewhere; the non-Linux paths are not yet exercised in CI.
- A parallel-efficiency step at the 14-to-15 qubit boundary was
  measured under the process-pool drain and remains unexplained; it
  has not been re-characterised since the threaded drain landed.


## License

GPL-3.0-or-later. See [`LICENSE`](https://codeberg.org/beavernets/paulikit/src/branch/main/LICENSE).
