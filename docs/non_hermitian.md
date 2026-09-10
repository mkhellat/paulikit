# Non-Hermitian Operators

## Why this matters

It is tempting to assume "Hamiltonian" implies "Hermitian" — in
elementary quantum mechanics, Hermiticity is exactly what guarantees
real energy eigenvalues and unitary (probability-preserving) time
evolution. But this assumption is not universal, and treating it as a
hard requirement would make `paulikit` unable to decompose several
classes of operators that show up in real research:

- **Open/dissipative quantum systems.** A system that exchanges energy
  or probability with an environment (e.g. an atom that can spontaneously
  emit a photon) is often modeled with a **non-Hermitian effective
  Hamiltonian** $H_{\text{eff}} = H_0 - \tfrac{i}{2}\Gamma$, where $H_0$
  is the ordinary (Hermitian) Hamiltonian and $\Gamma$ is a positive
  semi-definite matrix of decay rates. The imaginary part encodes
  probability *leaving* the subspace being modeled — the system's
  norm decays over time, exactly reflecting population lost to
  emission, decay, or measurement back-action. This is the standard
  approach in quantum optics and open-system dynamics (see, e.g., the
  quantum-trajectory / Monte Carlo wavefunction method).
- **$\mathcal{PT}$-symmetric Hamiltonians.** A body of work (starting
  with Bender & Boettcher, 1998) studies Hamiltonians that are not
  Hermitian but are symmetric under combined parity ($\mathcal{P}$)
  and time-reversal ($\mathcal{T}$) operations, and shows that such
  Hamiltonians can still have entirely real spectra under certain
  conditions — a genuinely non-Hermitian but physically sensible
  framework, actively used to model gain/loss-balanced systems.
- **Liouvillian superoperators.** The generator of open-system dynamics
  in the Lindblad master-equation formalism, $\mathcal{L}$, acts on
  density matrices rather than state vectors and is not Hermitian in
  the ordinary sense (though it has its own structural constraints).
  Decomposing $\mathcal{L}$ into a Pauli-type basis is a natural
  building block for simulating open-system dynamics on a quantum
  computer.
- **Individual summands of a Hermitian total.** Even when the
  *physical* operator of interest is Hermitian, it's often useful to
  decompose it as a sum of pieces that are individually not — e.g.
  splitting a Hamiltonian into "forward" and "backward" hopping terms
  for a Trotterization scheme that treats them differently. Each piece
  needs its own decomposition, and there's no guarantee each piece is
  separately Hermitian.

## What actually requires Hermiticity — nothing, mathematically

As derived in {doc}`theory`, the coefficient formula

$$
c(x, z) = \frac{1}{2^n} \, \overline{i^{\,\mathrm{popcount}(x \wedge z)}} \sum_{q} H_{q \oplus x,\, q} \, (-1)^{\mathrm{popcount}(q \wedge z)}
$$

was derived directly from the definition of the Frobenius inner
product and the action of $X^{x_j}Z^{z_j}$ on basis states — at no
point did the derivation use $H = H^\dagger$. The Pauli strings
$P(x,z)$ span the *entire* $2^n$-dimensional-squared space of complex
matrices (they are, up to normalization, an orthonormal basis for it),
so **every** complex matrix — Hermitian or not, even non-normal,
non-diagonalizable, anything — has a unique, well-defined Pauli
decomposition. `fwht_pauli_coefficients` computes it correctly in all
cases; this was verified directly against a from-scratch brute-force
reference on random non-Hermitian matrices (see
`tests/test_fwht.py::test_fwht_pauli_coefficients_handles_non_hermitian_matrices`).

What *is* special about Hermitian input is that the resulting
coefficients are guaranteed **real**. This follows from a short
argument: $H^\dagger = H$ implies $c(x,z) = \overline{c(x,z)}$ whenever
$P(x,z)$ is itself Hermitian, and every Pauli string *is* Hermitian
(since $I, X, Y, Z$ are each individually Hermitian, and a tensor
product of Hermitian operators is Hermitian). So for Hermitian $H$,
every coefficient equals its own complex conjugate, i.e. is real.

## How `fwht_pauli_terms` exposes this

`paulikit.algorithms.fwht.fwht_pauli_terms` has an `assume_hermitian`
parameter (default `True`, matching this package's primary use case
of real-symmetric coupled-oscillator Hamiltonians):

- `assume_hermitian=True` (default): asserts every coefficient's
  imaginary part is negligible (raising `ValueError` if not — a
  useful sanity check that catches bugs where a matrix was expected
  to be Hermitian but isn't) and returns real (`float`) coefficients.
- `assume_hermitian=False`: makes no such assumption, returns complex
  coefficients as-is, and works correctly for arbitrary operators.

## Worked example: a non-Hermitian effective Hamiltonian

Consider a single-qubit open-system effective Hamiltonian
$H_{\text{eff}} = H_0 - \tfrac{i}{2}\Gamma$ with:

$$
H_0 = \begin{pmatrix} 1 & 0.5 \\ 0.5 & -1 \end{pmatrix} = 0.5\,X + Z,
\qquad
\Gamma = \begin{pmatrix} 0.2 & 0 \\ 0 & 0.6 \end{pmatrix} = 0.4\,I - 0.2\,Z,
$$

so that:

$$
H_{\text{eff}} = 0.5\,X + Z - \frac{i}{2}(0.4\,I - 0.2\,Z)
              = -0.2i\,I + 0.5\,X + (1 + 0.1i)\,Z.
$$

Running this through `paulikit`:

```python
import numpy as np
from paulikit.algorithms.fwht import fwht_pauli_terms
from paulikit.pauli_utils import reconstruct_from_terms

H0 = np.array([[1.0, 0.5], [0.5, -1.0]])
Gamma = np.array([[0.2, 0.0], [0.0, 0.6]])
H_eff = H0 - 1j * Gamma / 2

terms = fwht_pauli_terms(H_eff, assume_hermitian=False)
# {'I': -0.2j, 'X': (0.5+0j), 'Z': (1+0.1j)}

reconstructed = reconstruct_from_terms(terms, n_qubits=1)
np.max(np.abs(reconstructed - H_eff))  # ~1.4e-17, i.e. exact
```

matching the hand-derived decomposition above exactly: the identity
term's coefficient is purely imaginary ($-0.2i$, entirely from the
decay), the $X$ term's coefficient is purely real ($0.5$, entirely
from $H_0$), and the $Z$ term's coefficient is complex ($1 + 0.1i$),
mixing a real contribution from $H_0$ with an imaginary contribution
from the decay term $\Gamma$'s own $Z$-component.

Calling `fwht_pauli_terms(H_eff)` (the default,
`assume_hermitian=True`) on this same matrix correctly raises
`ValueError`, since $H_{\text{eff}}$ is genuinely not Hermitian — this
is the intended behavior, not a limitation: it catches the mistake of
assuming Hermiticity where none exists, rather than silently
discarding the physically meaningful imaginary parts.
