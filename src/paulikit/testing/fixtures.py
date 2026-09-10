"""Correctness fixtures for Pauli decomposition, independent of any
algorithm implementation under test.

These fixtures exist so that any Pauli-decomposition algorithm added
to ``paulikit.algorithms`` can be validated against known-good data,
without depending on the implementation under test to also generate
its own "ground truth."

How this data was produced
---------------------------
Each fixture's Hamiltonian is built with
``paulikit.hamiltonian.build_hamiltonian``. The *expected*
decomposition was generated with PennyLane's ``qml.pauli_decompose``,
used strictly as an independent oracle - PennyLane is a
development-only dependency and no algorithm in this package imports
it. Call ``generate_fixture_data()`` to regenerate the constants
below if the Hamiltonian construction ever changes.

The constants are stored here, in version control, rather than
regenerated on every run: that keeps them reviewable in a diff, and
keeps this module importable and usable with PennyLane absent.

They are not meant to be taken on trust. Reconstructing H from the
stored terms and comparing against a freshly built Hamiltonian
reproduces it to machine epsilon, and ``padded_hamiltonian`` raises
if a fixture's recorded qubit count no longer matches what the
Hamiltonian construction produces - so a stale constant surfaces as
an error rather than as a quietly weakened check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


@dataclass(frozen=True)
class DecompositionFixture:
    """A Hamiltonian and its known-correct Pauli decomposition.

    Attributes:
        name: Short identifier, e.g. "N=2".
        n_oscillators: Number of coupled oscillators the Hamiltonian
            models.
        spring_constants: The ``(i, j) -> k_ij`` mapping used to build
            the Hamiltonian.
        masses: The per-oscillator masses used to build the
            Hamiltonian.
        n_qubits: Number of qubits the padded Hamiltonian spans.
        expected_terms: Mapping from Pauli-string label (e.g. "IXZ",
            read left-to-right as qubit 0, 1, 2, ...) to its real
            coefficient in the decomposition. Only nonzero terms
            (``abs(coefficient) > 1e-10``) are included.
    """

    name: str
    n_oscillators: int
    spring_constants: dict[tuple[int, int], float]
    masses: list[float]
    n_qubits: int
    expected_terms: dict[str, float] = field(default_factory=dict)

    def hamiltonian(self) -> NDArray[np.floating]:
        """Rebuild the (unpadded) Hamiltonian matrix for this fixture."""
        return build_hamiltonian(
            self.n_oscillators, self.spring_constants, self.masses
        )

    def padded_hamiltonian(self) -> NDArray[np.floating]:
        """Rebuild the zero-padded, power-of-two-dimensioned Hamiltonian."""
        padded, n_qubits = pad_to_power_of_two(self.hamiltonian())
        # Deliberately not an assert: this guard is the only thing
        # standing between a stale fixture and a silently wrong
        # "expected" answer, and asserts vanish under python -O.
        if n_qubits != self.n_qubits:
            raise ValueError(
                f"fixture {self.name!r} recorded n_qubits="
                f"{self.n_qubits} but rebuilding now gives "
                f"{n_qubits} - fixture is stale, regenerate with "
                "generate_fixture_data()"
            )
        return padded


N2_SPRING_CONSTANTS = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
N2_MASSES = [1.0, 2.0]

N4_SPRING_CONSTANTS = {
    (0, 0): 1.0, (0, 1): 2.0, (0, 2): 2.5, (0, 3): 3.0,
    (1, 1): 1.5, (1, 2): 2.2, (1, 3): 1.8,
    (2, 2): 2.0, (2, 3): 2.7,
    (3, 3): 3.3,
}
N4_MASSES = [1.0, 2.0, 1.5, 2.5]


FIXTURE_N2 = DecompositionFixture(
    name="N=2",
    n_oscillators=2,
    spring_constants=N2_SPRING_CONSTANTS,
    masses=N2_MASSES,
    n_qubits=3,
    expected_terms={
        "IXI": -0.5561862178478972,
        "IXZ": 0.056186217847897235,
        "XII": -0.3535533905932738,
        "XIX": 0.25,
        "XIZ": -0.3535533905932738,
        "XZI": -0.3535533905932738,
        "XZX": 0.25,
        "XZZ": -0.3535533905932738,
        "YIY": 0.25,
        "YZY": 0.25,
        "ZXI": -0.5561862178478972,
        "ZXZ": 0.056186217847897235,
    },
)

FIXTURE_N4 = DecompositionFixture(
    name="N=4",
    n_oscillators=4,
    spring_constants=N4_SPRING_CONSTANTS,
    masses=N4_MASSES,
    n_qubits=4,
    expected_terms={
        "IXII": -0.521204808933912,
        "IXIZ": -0.0174703256609009,
        "IXZI": 0.05469845798780232,
        "IXZZ": -0.016023323392989453,
        "XIII": -0.03702244670290003,
        "XIIX": -0.07264235376052372,
        "XIIZ": -0.31653094389037373,
        "XIXI": -0.0854052449248407,
        "XIXX": -0.03231185325344296,
        "XIXZ": -0.3476074569673786,
        "XIYY": 0.3384980711013402,
        "XIZI": -0.03702244670290003,
        "XIZX": -0.07264235376052372,
        "XIZZ": -0.31653094389037373,
        "XXII": -0.02246822282970609,
        "XXIX": -0.08681054680812308,
        "XXIZ": -0.31294197379526234,
        "XXZI": -0.02246822282970609,
        "XXZX": -0.08681054680812308,
        "XXZZ": -0.31294197379526234,
        "XYIY": 0.32398137132075155,
        "XYZY": 0.32398137132075155,
        "XZII": -0.03702244670290003,
        "XZIX": -0.07264235376052372,
        "XZIZ": -0.31653094389037373,
        "XZXI": -0.0854052449248407,
        "XZXX": -0.03231185325344296,
        "XZXZ": -0.3476074569673786,
        "XZYY": 0.3384980711013402,
        "XZZI": -0.03702244670290003,
        "XZZX": -0.07264235376052372,
        "XZZZ": -0.31653094389037373,
        "YIIY": 0.32264235376052375,
        "YIXY": 0.3384980711013402,
        "YIYI": 0.0854052449248407,
        "YIYX": 0.03231185325344296,
        "YIYZ": 0.3476074569673786,
        "YIZY": 0.32264235376052375,
        "YXIY": 0.32398137132075155,
        "YXZY": 0.32398137132075155,
        "YYII": 0.02246822282970609,
        "YYIX": 0.08681054680812308,
        "YYIZ": 0.31294197379526234,
        "YYZI": 0.02246822282970609,
        "YYZX": 0.08681054680812308,
        "YYZZ": 0.31294197379526234,
        "YZIY": 0.32264235376052375,
        "YZXY": 0.3384980711013402,
        "YZYI": 0.0854052449248407,
        "YZYX": 0.03231185325344296,
        "YZYZ": 0.3476074569673786,
        "YZZY": 0.32264235376052375,
        "ZXII": -0.521204808933912,
        "ZXIZ": -0.0174703256609009,
        "ZXZI": 0.05469845798780232,
        "ZXZZ": -0.016023323392989453,
    },
)

ALL_FIXTURES = [FIXTURE_N2, FIXTURE_N4]


def pauli_word_to_label(pauli_word, n_qubits: int) -> str:
    """Convert a PennyLane ``PauliWord`` to a fixed-width label string.

    Args:
        pauli_word: A ``pennylane.pauli.PauliWord``-like mapping from
            wire index to single-letter Pauli operator name.
        n_qubits: Total number of qubits/wires, used to fill in
            implicit identities.

    Returns:
        A string of length ``n_qubits``, e.g. ``"IXZ"`` for identity
        on wire 0, X on wire 1, Z on wire 2.
    """
    return "".join(pauli_word.get(w, "I") for w in range(n_qubits))


def generate_fixture_data() -> None:
    """Regenerate the ``expected_terms`` dicts above from scratch.

    Uses PennyLane's ``qml.pauli_decompose`` as an independent oracle.
    Call this by hand if the Hamiltonian construction changes and the
    constants above need updating - it is never called automatically,
    so the stored fixtures stay fixed and reviewable in version
    control rather than silently regenerating on every run. Requires
    PennyLane to be installed (a development-only dependency of this
    package, not required to use ``paulikit.algorithms`` itself).
    """
    import pennylane as qml

    for fixture in ALL_FIXTURES:
        padded = fixture.padded_hamiltonian()
        pauli_sentence = qml.pauli_decompose(padded, pauli=True)
        terms = {
            pauli_word_to_label(pauli_word, fixture.n_qubits):
                float(np.real(c))
            for pauli_word, c in pauli_sentence.items()
            if abs(c) > 1e-10
        }
        print(f"# {fixture.name}: {len(terms)} terms")
        for label in sorted(terms):
            print(f"    {label!r}: {terms[label]!r},")
        print()
