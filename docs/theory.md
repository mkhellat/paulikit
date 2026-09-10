# Mathematical Theory

This page derives, from first principles, the algorithm implemented in
`paulikit.algorithms.fwht`. Nothing here is assumed from the source
paper without re-derivation — this derivation is numerically verified
against a brute-force reference implementation; see
`tests/test_fwht.py::test_fwht_matches_brute_force_on_random_hermitian`
for the automated version of that check.

## 1. Setup: the symplectic representation of Pauli strings

An $n$-qubit Pauli string is a tensor product of single-qubit Pauli
operators, one per qubit, drawn from $\{I, X, Y, Z\}$. There are
$4^n$ such strings for $n$ qubits, and they form an orthogonal basis
(under the Frobenius/Hilbert-Schmidt inner product) for the full
$2^n \times 2^n$ complex vector space — meaning *any* $2^n \times 2^n$
complex matrix, not just Hermitian ones, can be written as a unique
linear combination of them.

Rather than index Pauli strings by an explicit list of letters, it is
useful to use the **symplectic representation**: encode a Pauli
string by two $n$-bit integers $x$ and $z$, where bit $j$ of each
(counting from the most significant bit, so qubit $0$ is the leftmost
bit) determines the single-qubit factor on qubit $j$:

$$
P(x, z) = \bigotimes_{j=0}^{n-1} i^{\,x_j \wedge z_j} \, X^{x_j} Z^{z_j},
$$

where $x_j, z_j \in \{0, 1\}$ are the $j$-th bits of $x$ and $z$, and
$\wedge$ is bitwise AND. Working through the four combinations of
$(x_j, z_j)$:

| $x_j$ | $z_j$ | $i^{x_j \wedge z_j} X^{x_j} Z^{z_j}$ | Pauli |
|-------|-------|----------------------------------------|-------|
| 0     | 0     | $I$                                     | $I$   |
| 1     | 0     | $X$                                     | $X$   |
| 0     | 1     | $Z$                                     | $Z$   |
| 1     | 1     | $i X Z = Y$                             | $Y$   |

(using $XZ = \begin{pmatrix}0&-1\\1&0\end{pmatrix} = -iY$, so
$iXZ = Y$). This is exactly `paulikit.algorithms.fwht.pauli_label`'s
lookup table, and the reason the package's internal representation of
a Pauli string is a pair of integers rather than a string until the
very last step.

## 2. The decomposition coefficient as a matrix element

The coefficient of $P(x, z)$ in the decomposition of a matrix $H$ is
given by the Frobenius inner product:

$$
c(x, z) = \frac{1}{2^n} \operatorname{Tr}\!\left(H \, P(x, z)^\dagger\right).
$$

To turn this trace into something computable without ever
constructing the full $2^n \times 2^n$ Pauli matrix, expand the trace
over the computational basis $\{\lvert q \rangle\}$:

$$
c(x, z) = \frac{1}{2^n} \sum_{p, q} H_{p, q} \, \langle q \rvert P(x, z)^\dagger \lvert p \rangle
        = \frac{1}{2^n} \sum_{p, q} H_{p, q} \, \overline{\langle p \rvert P(x, z) \lvert q \rangle}.
$$

The key simplification comes from working out $\langle p \rvert P(x,
z) \lvert q \rangle$ directly. Since $X \lvert q_j \rangle = \lvert 1
- q_j \rangle$ (bit flip) and $Z \lvert q_j \rangle = (-1)^{q_j}
\lvert q_j \rangle$ (phase flip) act independently on each qubit, and
$X^{x_j} Z^{z_j}$ first applies the phase then the flip:

$$
X^{x_j} Z^{z_j} \lvert q_j \rangle = (-1)^{q_j z_j} \lvert q_j \oplus x_j \rangle.
$$

Tensoring over all $n$ qubits, and folding in the $i^{x_j \wedge z_j}$
phase per qubit (which multiplies out to $i^{\,\mathrm{popcount}(x \wedge z)}$
across all qubits, since $\mathrm{popcount}$ of a bitwise AND counts
exactly the qubits where both $x_j$ and $z_j$ are 1):

$$
P(x, z) \lvert q \rangle = i^{\,\mathrm{popcount}(x \wedge z)} \, (-1)^{\mathrm{popcount}(q \wedge z)} \, \lvert q \oplus x \rangle.
$$

So $\langle p \rvert P(x, z) \lvert q \rangle$ is nonzero (and equal to
the phase above) exactly when $p = q \oplus x$, and zero otherwise.
Substituting back:

$$
c(x, z) = \frac{1}{2^n} \, \overline{i^{\,\mathrm{popcount}(x \wedge z)}} \sum_{q} H_{q \oplus x,\, q} \, (-1)^{\mathrm{popcount}(q \wedge z)}.
$$

This formula is exact for **any** complex matrix $H$ — nothing here
assumed $H$ is Hermitian. (See {doc}`non_hermitian` for what happens,
and why it matters physically, when $H$ isn't Hermitian.)

## 3. Recognizing the Walsh-Hadamard Transform

Fix $x$ and define $g_x(q) = H_{q \oplus x, q}$ — the entries of $H$
lying on the "$x$-th anti-diagonal" (an XOR-shifted diagonal; for
$x=0$ this is the literal diagonal). Then the sum above, as a function
of $z$, is:

$$
\sum_q g_x(q) \, (-1)^{\mathrm{popcount}(q \wedge z)}.
$$

This is precisely the (unnormalized, $\pm 1$) **Walsh-Hadamard
Transform** of the sequence $g_x$: the WHT of a length-$2^n$ sequence
$g$ is $\mathrm{WHT}(g)[z] = \sum_q g(q) \, (-1)^{\mathrm{popcount}(q
\wedge z)}$, and it has a well-known $O(n \cdot 2^n)$ fast algorithm
(the same divide-and-conquer butterfly structure as the FFT, but with
real $\pm 1$ "twiddle factors" instead of complex roots of unity —
this is `paulikit.algorithms.fwht._walsh_hadamard_transform_rows`).

Putting it together, the full algorithm is three steps:

1. **XOR-index gather.** For every $x \in [0, 2^n)$, build $g_x(q) =
   H_{q \oplus x, q}$. This is a data-movement step: reading $2^n$
   specific entries of $H$ for each of $2^n$ values of $x$, i.e.
   $2^n \times 2^n$ reads total, each $O(1)$.
2. **Walsh-Hadamard Transform.** Apply the WHT to each $g_x$
   independently. Each transform costs $O(n \cdot 2^n)$; there are
   $2^n$ of them (one per $x$), for $O(n \cdot 4^n)$ total.
3. **Phase-factor multiplication.** Multiply each transformed value by
   $\overline{i^{\mathrm{popcount}(x \wedge z)}} / 2^n$ — an $O(1)$
   operation per $(x, z)$ pair, $O(4^n)$ total.

## 4. Complexity

Step 2 dominates: $O(n \cdot 4^n)$. Writing $N = 2^n$ for the matrix
dimension, $n = \log_2 N$, so this is:

$$
O(n \cdot 4^n) = O(\log N \cdot N^2) = O(N^2 \log N).
$$

Compare to the naive approach of evaluating $\operatorname{Tr}(H
P(x,z)^\dagger)$ directly for each of the $4^n = N^2$ Pauli strings,
each trace costing $O(N)$ multiply-adds if done densely: that's
already $O(N^3)$ just for the arithmetic, *before* accounting for the
much larger constant factor of doing it symbolically (as the tutorial
notebook's SymPy-based `decompose_to_pauli_terms` does) rather than
numerically. The FWHT approach is both asymptotically better and has
a dramatically smaller constant.

## 5. Worked example: $N=2$ oscillators by hand

To make this concrete, work through one term of the actual $N=2$
coupled-oscillator fixture used in this package's tests
(`paulikit.testing.fixtures.FIXTURE_N2`). The (unpadded) Hamiltonian,
for spring constants $k_{00}=1, k_{01}=2, k_{11}=3$ and masses
$m_0=1, m_1=2$, is:

$$
H = \begin{pmatrix}
0 & 0 & -1 & 0 & -\sqrt{2} \\
0 & 0 & 0 & -\sqrt{3/2} & 1 \\
-1 & 0 & 0 & 0 & 0 \\
0 & -\sqrt{3/2} & 0 & 0 & 0 \\
-\sqrt{2} & 1 & 0 & 0 & 0
\end{pmatrix},
$$

padded with zeros to $8 \times 8$ ($n = 3$ qubits, since $\lceil
\log_2 5 \rceil = 3$).

**Target term:** the Pauli string "XII" (X on qubit 0, identity on
qubits 1 and 2). In the symplectic representation this is $x=4$ (binary
$100$, X on the leftmost/qubit-0 bit), $z=0$.

**Step 1 (gather):** $g_4(q) = H_{q \oplus 4,\, q}$ for $q = 0,
\ldots, 7$. Since $q \oplus 4$ flips the top bit of $q$, this reads
the entries connecting index $q$ to index $q+4$ (for $q < 4$) — which
is exactly the padded Hamiltonian's block linking the original
$\{0,1\}$ rows to the original $\{4\}$ column (recall the unpadded
matrix is $5\times5$; row/column index 4 is the last row/column
above). Concretely:

$$
g_4 = (-\sqrt{2},\ 0,\ 0,\ 0,\ -\sqrt{2},\ 0,\ 0,\ 0).
$$

**Step 2 (WHT):** applying the length-8 Walsh-Hadamard Transform to
$g_4$ gives (only the first four output values are nonzero, since
$g_4$ is nonzero only at $q \in \{0, 4\}$, which have the same value
$-\sqrt2$ and are related by the top bit — a pattern that WHT maps to
a "flat" block of outputs):

$$
\mathrm{WHT}(g_4) = (-2\sqrt2,\ -2\sqrt2,\ -2\sqrt2,\ -2\sqrt2,\ 0,\ 0,\ 0,\ 0).
$$

**Step 3 (phase multiply):** for $z = 0$, $x \wedge z = 0$, so the
phase factor is $i^0 = 1$, and:

$$
c(4, 0) = \frac{1}{8} \cdot (-2\sqrt2) = -\frac{\sqrt2}{4} = -\frac{1}{2\sqrt2} \approx -0.35355339\ldots
$$

This matches `FIXTURE_N2.expected_terms["XII"] ==
-0.3535533905932738` exactly (up to floating-point rounding). The
same mechanical process, run for every $(x, z)$ pair, produces all 12
nonzero terms of the fixture — see
`tests/test_fwht.py::test_fwht_matches_fixture_expected_terms` for the
automated, full-fixture version of this check.

## References

- [Babbush et al., *Exponential Quantum Speedup in Simulating Coupled
  Classical Oscillators*](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.041041) —
  the physical problem motivating this package; see {doc}`background`.
- [*Pauli decomposition via the fast Walsh-Hadamard transform*](https://iopscience.iop.org/article/10.1088/1367-2630/adb44d) —
  the published algorithm this implementation is based on (re-derived
  independently above, not transcribed).
- [Suzuki, *General theory of fractal path integrals with applications
  to many-body theories and statistical physics*](https://arxiv.org/abs/math-ph/0506007v1) —
  the Trotterization method applied downstream of this package's
  output (outside this package's scope).
