# Background and Motivation

## The physical problem

The problem this package was built for is simulating the dynamics of a
1-dimensional chain of $N$ coupled classical harmonic oscillators on a
quantum computer, following
[Babbush et al., *Exponential Quantum Speedup in Simulating Coupled
Classical Oscillators*](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.041041).
The physical system — masses connected by springs, each mass also
tethered to a fixed wall — is entirely classical. The quantum algorithm
is a computational tool for evolving that classical system's state
forward in time, not a simulation of anything intrinsically quantum
about the oscillators themselves.

The key move that makes this possible is a change of variables that
turns Newton's equations of motion for the coupled oscillators into a
Schrödinger-like equation:

$$
i \frac{d}{dt} \lvert \psi(t) \rangle = \hat{H} \lvert \psi(t) \rangle,
$$

where $\lvert \psi(t) \rangle$ encodes the system's positions and
velocities as amplitudes of a normalized quantum state, and $\hat{H}$
is a Hermitian matrix built from the masses and spring constants. Once
the problem is in this form, the machinery of quantum Hamiltonian
simulation — evolving $\lvert \psi(0) \rangle$ to $\lvert \psi(t)
\rangle$ via $e^{-i \hat{H} t}$ — applies directly, and that machinery
is what can, in principle, run exponentially faster on a quantum
computer than direct classical numerical integration would.

## Why Pauli decomposition is the bottleneck

"In principle" is doing a lot of work in that last sentence. A quantum
computer doesn't execute $\hat{H}$ directly — it executes sequences of
quantum gates, and the gate set doesn't include "apply an arbitrary
Hermitian matrix." The standard bridge is:

1. Write $\hat{H}$ as a weighted sum of *Pauli strings* — tensor
   products of single-qubit Pauli operators ($I$, $X$, $Y$, $Z$), each
   of which *does* correspond directly to a small, physically
   implementable quantum gate sequence.
2. Approximate $e^{-i \hat{H} t}$ by *Trotterizing*: alternately
   applying $e^{-i P_k t/r}$ for each Pauli term $P_k$, $r$ times, which
   converges to the true evolution as $r \to \infty$ (see
   [Suzuki, *General theory of fractal path integrals with applications
   to many-body theories and statistical physics*](https://arxiv.org/abs/math-ph/0506007v1)
   for the generalized decomposition used here).

Step 1 — the Pauli decomposition — is where this package lives. It has
to happen *before* any quantum circuit can be built, and it is a
**classical, pre-processing** computation: given the numeric matrix
$\hat{H}$, find the coefficients $\alpha_P$ such that
$\hat{H} = \sum_P \alpha_P \, P$.

This is not a minor implementation detail. If the classical
decomposition step scales badly, it doesn't matter how good the
quantum circuit is downstream — you can never build the circuit for
large $N$ in the first place. A "quantum speedup" that requires an
exponential-time classical pre-processing step to set up isn't a
speedup at all for that regime.

## Why a direct symbolic approach doesn't scale

A direct approach computes each Pauli coefficient from its
definition — the Frobenius inner product

$$
\alpha_P = \frac{1}{2^n} \operatorname{Tr}(\hat{H} \, P^\dagger)
$$

evaluated **symbolically**, once per Pauli string. For an $n$-qubit
Hamiltonian there are $4^n$ possible Pauli strings, so this is
$O(4^n)$ symbolic trace evaluations, each itself costing $O(2^n)$
matrix work — before even accounting for how much slower symbolic
arithmetic is than floating point.

Concretely: the coupled-oscillator Hamiltonian for $N$ oscillators is
padded to $n = \lceil \log_2(N^2/2 + 3N/2) \rceil$ qubits. At $N=30$,
that's $n \approx 9$, meaning roughly $4^9 \approx 262{,}000$ symbolic
trace evaluations on $512 \times 512$ symbolic matrices — a workload
that grows purely from the term count, before any per-term cost is
even counted.

## What paulikit changes

`paulikit.algorithms.fwht` replaces the symbolic $O(4^n)$-with-huge-
constant approach with a numeric algorithm that is $O(N^2 \log N)$ for
an $N \times N$ matrix ($N = 2^n$) — see {doc}`theory` for the full
derivation. This is the same asymptotic complexity class as the naive
approach's *term count* ($O(4^n) = O(N^2)$, up to the $\log N$ factor)
but without the crushing constant-factor overhead of symbolic
computation, and with a numerically exact (to floating-point
precision) result. `paulikit` decomposes Hamiltonians well beyond
where a symbolic, per-term approach becomes impractical; see
{doc}`theory` for the complexity argument and the test suite for
concrete N values this package is exercised against.

## Where this fits in the pipeline

Pauli decomposition is necessary but not sufficient for the full
Hamiltonian-simulation pipeline. Downstream of it:

- The decomposed Pauli terms feed into a Trotterization step (e.g.
  Classiq's `suzuki_trotter`), which builds the actual quantum circuit.
- The number of Trotter repetitions needed for a target accuracy is a
  separate question from decomposition speed, and is not addressed by
  this package.
- Decoding the simulation's output state back into physical positions
  and velocities is its own step, with its own subtleties (e.g. sign
  recovery from measurement probabilities, which lose phase
  information), and is likewise not addressed by this package.

`paulikit` is scoped narrowly and deliberately: it solves the
Pauli-decomposition bottleneck, and only that, so it can be developed,
tested, and (if useful) reused independently of the rest of the
simulation pipeline.
