# Tutorial

This page walks through using `paulikit` end to end: building a
Hamiltonian, decomposing it, interpreting and verifying the result,
and using the command-line interface. For the "why" behind each step,
see {doc}`background` and {doc}`theory`; for the full function
signatures, see the {doc}`API reference <api/index>`.

## Installation

```bash
pip install paulikit
```

Or, from a checkout of the repository:

```bash
pip install -e . --no-build-isolation
```

Everything in this tutorial works with just the core install (`numpy`
is the only runtime dependency). The command-line examples below
assume the `paulikit` console script is on your `PATH`, which the
install above sets up automatically.

The install above also compiles a native (Cython/C++) label-generation
kernel if a C++ toolchain and oneTBB are available, for faster
`fwht_pauli_terms` — falling back to pure Python automatically
otherwise. See the project README's "Native extension" section for
details; nothing in this tutorial depends on which path is active.

## 1. Building a Hamiltonian

`paulikit.hamiltonian.build_hamiltonian` constructs the
coupled-oscillator Hamiltonian matrix from physical parameters: a
dictionary of spring constants and a list of masses.

```python
from paulikit.hamiltonian import build_hamiltonian

spring_constants = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
masses = [1.0, 2.0]

H = build_hamiltonian(n_oscillators=2, spring_constants=spring_constants, masses=masses)
```

`spring_constants` maps `(i, j)` with `i <= j` to the spring constant
$k_{ij}$: diagonal entries are each oscillator's own spring constant
(coupling to a fixed wall), off-diagonal entries are the coupling
strength between oscillator `i` and oscillator `j`. `masses[i]` is
oscillator `i`'s mass.

The result is a $5 \times 5$ matrix (for $N=2$, the dimension is
$N + N(N+1)/2$):

```
[[ 0.          0.         -1.          0.         -1.41421356]
 [ 0.          0.          0.         -1.22474487  1.        ]
 [-1.          0.          0.          0.          0.        ]
 [ 0.         -1.22474487  0.          0.          0.        ]
 [-1.41421356  1.          0.          0.          0.        ]]
```

## 2. Padding to a power-of-two dimension

Pauli decomposition requires the matrix dimension to be exactly
$2^n$ for some integer $n$ (the number of qubits). Real
coupled-oscillator Hamiltonians rarely have a power-of-two dimension
already, so pad first:

```python
from paulikit.hamiltonian import pad_to_power_of_two

H_padded, n_qubits = pad_to_power_of_two(H)
# H_padded.shape == (8, 8), n_qubits == 3
```

This zero-pads $H$ into the top-left block of an $8 \times 8$ matrix
(since $\lceil \log_2 5 \rceil = 3$). The padding entries are all
zero, so they contribute nothing to the physics — they exist only to
satisfy the tensor-product structure quantum circuits require.

## 3. Decomposing into Pauli terms

```python
from paulikit.algorithms.fwht import fwht_pauli_terms

terms = fwht_pauli_terms(H_padded)
```

`terms` is a dictionary mapping Pauli-string labels to their
coefficients:

```
{
    'IXI': -0.556186, 'IXZ': 0.056186,
    'XII': -0.353553, 'XIX': 0.250000, 'XIZ': -0.353553,
    'XZI': -0.353553, 'XZX': 0.250000, 'XZZ': -0.353553,
    'YIY': 0.250000, 'YZY': 0.250000,
    'ZXI': -0.556186, 'ZXZ': 0.056186,
}
```

Each label is a string of length `n_qubits`, read left-to-right as
qubit 0, 1, 2, ... — e.g. `'XIZ'` means $X$ on qubit 0, $I$
(identity) on qubit 1, $Z$ on qubit 2, i.e. the operator
$X \otimes I \otimes Z$. Only nonzero terms are included (12 of the
$4^3 = 64$ possible 3-qubit Pauli strings, here); the threshold is
controlled by `fwht_pauli_terms`'s `atol` parameter.

By default (`assume_hermitian=True`), coefficients are returned as
real `float`s, and a `ValueError` is raised if the input wasn't
actually Hermitian (a useful sanity check — see {doc}`non_hermitian`
for when and how to decompose non-Hermitian operators instead).

## 4. Verifying the result

`paulikit.pauli_utils.reconstruct_from_terms` rebuilds the dense
matrix from a term dictionary, useful both as a sanity check and for
programmatically confirming a decomposition is correct:

```python
from paulikit.pauli_utils import reconstruct_from_terms

reconstructed = reconstruct_from_terms(terms, n_qubits)
error = (reconstructed.real - H_padded)
print(abs(error).max())  # 0.0 - exact to floating-point precision
```

This is exactly the check `paulikit`'s own test suite runs against
every fixture (see `tests/test_fwht.py`), and it's good practice to
run it yourself whenever decomposing a new Hamiltonian you haven't
validated before.

## 5. Using the command-line interface

For quick exploration without writing a script, the `paulikit`
console command wraps the same functionality:

```console
$ paulikit decompose --n-oscillators 4 --show-terms
N=4 oscillators, 4 qubits, 16x16 padded Hamiltonian
Decomposition time: ...
Nonzero Pauli terms: 56
  IXII: -0.5470915958155509
  IXIZ: 0.015053558406667805
  IXZI: 0.02983035390312644
  ...
```

(Timing is reported but not reproducible across machines; term count
and coefficient values are the part worth checking against your own
run.)

`paulikit decompose` builds a synthetic Hamiltonian internally (a
fixed, deterministic — not physically calibrated — set of spring
constants and masses that scale with $N$), so it's meant for quickly
checking behavior and timing at a given size, not for physically
meaningful results; use the library API (above) with your own
parameters for real work.

`paulikit benchmark` sweeps multiple $N$ values and reports timing:

```console
$ paulikit benchmark --n-oscillators 2 4 8 16 30
    N  qubits    dim    terms   time (s)
    2       3      8       12        ...
    4       4     16       56        ...
    8       6     64      928        ...
   16       8    256    15360        ...
   30       9    512   112384        ...
```

(Timings vary run to run and by machine; term counts are the part
worth checking against your own run.)

### Decomposing in parallel from the command line

`--parallel` runs the decomposition across multiple workers through
`parallel_decompose_arrays`, which is the path that scales. It needs
`--chunk-size`:

```console
$ paulikit decompose --n-oscillators 150 --chunk-size 2 --parallel
N=150 oscillators, 14 qubits, 16384x16384 padded Hamiltonian
Decomposition time (parallel, executor=auto): ...
Chunks: 5595, nonzero Pauli terms: 91652096
```

Two things this path does differently, both deliberate:

- **The operator is built sparse.** At 15 qubits a dense operator is
  16 GiB and at 16 qubits it is 64 GiB, so densifying would put the
  sizes this path exists for out of reach before the decomposition
  started.
- **Labels are never built.** That per-term Python work is the serial
  cost this API removes, so `--show-terms` reports counts only and
  says so rather than silently ignoring the flag. Use
  `terms_from_arrays` in a script if you need labels for a subset.

`--executor` chooses how chunks are drained:

| value | behaviour |
|---|---|
| `auto` (default) | `thread` when the compiled kernels are available, `process` otherwise |
| `thread` | one process, threads running the kernels concurrently — no pickling |
| `process` | a process pool; pays IPC but does not depend on the kernels being built |

The default is conditional because the right answer flips depending on
build. The compiled kernels release the GIL, so with them threads win
decisively; without them the NumPy fallback holds the GIL and threads
are slower than processes. Deciding per build rather than globally is
what makes a default safe here.

`--n-workers` sets the worker count, defaulting to the number of
distinct physical cores. Hyperthread siblings share execution units
rather than adding independent ones, so they measured worse: on a
4-core machine, adding the 4 hyperthread siblings on top of the 4
physical-core threads cost extra CPU cycles for no further wall-clock
gain.

`paulikit regenerate-fixtures` recomputes the expected Pauli terms
used by the test suite's correctness fixtures, using PennyLane as an
independent oracle — see the {doc}`API reference <api/testing>` for
`paulikit.testing.fixtures` if you're extending the test suite itself
rather than just using the library.

Run `paulikit --help` or `paulikit <subcommand> --help` for full
argument details on any of these.

## 6. Multi-core decomposition and the array-yielding API

> **Read the end of this section before choosing a function for
> speed.** Since the transform moved into compiled C kernels, whether
> the multi-core path beats the sequential one depends on which
> executor and build you're using — see "Which executor, and why it
> matters" below. It is always the right choice for streaming and
> bounded memory, independent of throughput.

For a large Hamiltonian, `paulikit.algorithms.fwht.parallel_decompose`
spreads the FWHT coefficient math for each chunk across a
`ProcessPoolExecutor`, then streams back `dict[str, complex]` chunks
just like `fwht_pauli_terms_iter`:

```python
from paulikit.algorithms.fwht import parallel_decompose

for chunk in parallel_decompose(H_padded):
    ...
```

This is the right tool when you actually want every term's Pauli
label. But it comes with a caveat worth knowing about before you reach
for it purely for speed: building the label string and dict entry for
every term happens in the single parent process, not in the worker
pool, and at large problem sizes that step dominates the function's
own runtime. Measured directly at $N=150$ (91.6 million terms), it is
about 82% of total runtime — a serial fraction that, by Amdahl's law,
caps the achievable speedup for the dict-returning API no matter how
many cores are thrown at the problem. (The array-yielding API added
later removes that ceiling; see below.) This isn't a defect to be
fixed later; it's an inherent cost of returning fully-labeled Python
dicts at that scale, and `parallel_decompose` remains the correct,
supported choice whenever you need those labels.

When you don't need every label — for instance, filtering to the
largest-magnitude terms, or feeding coefficients straight into a
numerical routine that never looks at the Pauli string itself —
`parallel_decompose_arrays` skips that serial labeling step entirely.
It shares `parallel_decompose`'s pool, chunking, auto-tuning, and
checkpoint machinery exactly (checkpoints are even interchangeable
between the two functions — both write the same binary, chunk-framed
format through one shared writer, so a checkpoint started under one
function resumes cleanly under the other), but its drain loop yields
each chunk's raw
`(x, z, coeff)` NumPy arrays — symplectic `x`/`z` bitmasks and
`complex128` coefficients — instead of building labels and a dict from
them:

```python
import numpy as np
from paulikit.algorithms.fwht import parallel_decompose_arrays, terms_from_arrays

n_qubits = int(np.log2(H_padded.shape[0]))
for x, z, coeff in parallel_decompose_arrays(H_padded):
    big = np.abs(coeff) > 1e-3          # keep only what you need
    terms = terms_from_arrays(x[big], z[big], coeff[big], n_qubits)
```

`terms_from_arrays` is the opt-in rendering step: pass it whichever
arrays (or filtered subset of them) you actually want labels for, and
it returns the same `dict[str, float]` (or `dict[str, complex]` if
`assume_hermitian=False`) that `fwht_pauli_terms` and
`parallel_decompose` produce. Because labeling a handful of surviving
terms is cheap regardless of how large the original decomposition was,
this pattern — decompose with `parallel_decompose_arrays`, filter, then
label only the survivors — sidesteps the serial bottleneck rather than
paying it and discarding most of the result.

How much does this actually buy you? Removing per-term Python work
from the drain loop is a real and well-understood fix for a real
serial bottleneck — the label-and-dict path scaled *negatively*,
where adding workers made it slower.

### Which executor, and why it matters

`parallel_decompose_arrays` takes an `executor` argument. The default,
`"auto"`, picks `"thread"` when the compiled kernels are available and
`"process"` otherwise.

That conditional is not hedging: the two builds have opposite winners.
With the compiled kernels present, `"thread"` beats `"process"` by a
wide margin; with the pure-NumPy fallback, `"process"` beats
`"thread"` instead. The mechanism explains why: the kernels release
the GIL, so with them threads run the real work concurrently and pay
no pickling at all. Without them the NumPy fallback holds the GIL
through most of each chunk, so threads serialise *and* add contention.
Neither choice is right in general, which is why it is decided per
build.

At N=150 (91.6M terms), peak RSS stays in the same tens-of-MiB range
across all three paths (sequential chunked, parallel threads, parallel
processes), so the choice among them is a throughput question, not a
memory one. With the compiled kernels present, the parallel-threads
path avoids the process pool's IPC and pickling cost entirely, which
is why it is the path worth reaching for first once the kernels are
built; measure on your own build and machine to confirm the ranking
before relying on it.

### Reaching sizes a dense implementation cannot

The streaming design's real payoff is not the constant factor. Peak
resident memory stays roughly flat as the problem grows, because only
one chunk is live at a time:

| qubits | terms | peak RSS |
|---|---|---|
| 14 | 91,652,096 | 72 MiB |
| 15 | 326,134,272 | 89 MiB |
| 16 | 1,470,021,632 | 123 MiB |

An implementation that requires the caller to hold the dense
$2^n \times 2^n$ operator needs 4 GiB at 14 qubits, 16 GiB at 15 and
64 GiB at 16 — so on a 16 GiB machine the last two are simply out of
reach, regardless of how fast its inner loop is.

Measure your own workload rather than assuming either way — but the
memory profile is a property of the design, not of tuning.
