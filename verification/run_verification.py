#!/usr/bin/env python3
"""Single reproducible entry point for exhaustive correctness verification.

Every number quoted in FINDINGS.md or memory MUST come from a JSON
file this script produced - no ad-hoc shell heredocs. Rerunning the
exact command recorded in a result file's "command" field must
reproduce the same result (mod wall-clock timing noise).

Usage:
    python run_verification.py --n 150 --hermitian
    python run_verification.py --n 20 --hermitian --against pennylane
    python run_verification.py --n 30 --non-hermitian
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import scipy.sparse as sp

from exhaustive_projection import verify_terms, verify_terms_streaming
from paulikit.algorithms.fwht import fwht_pauli_terms, fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def synthetic_spring_constants(n_oscillators: int) -> dict[tuple[int, int], float]:
    """Same generator as tests/test_benchmark_reference.py, reused
    verbatim so results are comparable to that suite's own runs."""
    return {
        (i, j): 1.0 + 0.1 * (i + j)
        for i in range(n_oscillators)
        for j in range(i, n_oscillators)
    }


def synthetic_masses(n_oscillators: int) -> list[float]:
    return [1.0 + 0.05 * i for i in range(n_oscillators)]


def make_non_hermitian(padded: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic, reproducible non-Hermitian perturbation: adds an
    antisymmetric imaginary component on the same sparsity pattern as
    the real Hamiltonian, so nnz stays comparable."""
    rng = np.random.default_rng(seed)
    mask = padded != 0
    perturbation = np.zeros_like(padded, dtype=complex)
    rows, cols = np.nonzero(np.triu(mask, k=1))
    values = rng.uniform(0.01, 0.05, size=len(rows)) * 1j
    perturbation[rows, cols] = values
    perturbation[cols, rows] = -values  # antisymmetric -> non-Hermitian
    return padded.astype(complex) + perturbation


def get_versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": sp.__version__ if hasattr(sp, "__version__") else __import__("scipy").__version__,
    }
    try:
        import pennylane as qml

        versions["pennylane"] = qml.__version__
    except ImportError:
        versions["pennylane"] = None
    return versions


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="Number of oscillators")
    herm_group = parser.add_mutually_exclusive_group(required=True)
    herm_group.add_argument("--hermitian", action="store_true")
    herm_group.add_argument("--non-hermitian", action="store_true")
    parser.add_argument(
        "--against",
        choices=["projection", "pennylane", "both"],
        default="projection",
        help="Oracle(s) to verify against. 'pennylane' only tractable for small N.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for non-Hermitian perturbation")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--timeout-note", type=str, default=None, help="Free-text note, e.g. wall-clock budget used")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Passed to fwht_pauli_terms/fwht_pauli_terms_iter. 256 is "
            "the established convention."
        ),
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Use fwht_pauli_terms_iter + verify_terms_streaming instead "
            "of fwht_pauli_terms + verify_terms - REQUIRED at N=150: "
            "fwht_pauli_terms (dict-returning, even with chunk_size set) "
            "is documented to OOM at N=150 regardless of RAM (it re-fuses "
            "all ~91.65M chunked terms into one dict before returning - "
            "see this project's own verification/FINDINGS.md §5 for "
            "the OOM hit while developing this script). Requires "
            "--chunk-size. "
            "Only supports --against projection (PennyLane needs the "
            "full dict; use --streaming only at N too large for PennyLane "
            "anyway)."
        ),
    )
    args = parser.parse_args()

    if args.streaming and args.chunk_size is None:
        parser.error("--streaming requires --chunk-size")
    if args.streaming and args.against != "projection":
        parser.error("--streaming only supports --against projection")

    command = " ".join(["python", "run_verification.py"] + sys.argv[1:])

    hermitian = args.hermitian
    k = synthetic_spring_constants(args.n)
    m = synthetic_masses(args.n)

    # Build sparse end-to-end at large N (avoids ever materializing the
    # O(dim**2) dense Hamiltonian - see build_hamiltonian's own
    # docstring: ~4GiB at N=150 despite 0.034% density). Use dense only
    # when scipy's sparse path isn't needed (non-Hermitian perturbation
    # below is easiest to apply densely, and small N doesn't need it).
    if hermitian:
        H_sparse = build_hamiltonian(args.n, k, m, sparse=True)
        padded_sparse, n_qubits = pad_to_power_of_two(H_sparse, sparse=True)
        padded_sparse = padded_sparse.tocsr()
    else:
        H = build_hamiltonian(args.n, k, m)
        padded, n_qubits = pad_to_power_of_two(H)
        padded = make_non_hermitian(padded, args.seed)
        padded_sparse = sp.csr_matrix(padded)

    result: dict = {
        "command": command,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
        "versions": get_versions(),
        "machine": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "n_oscillators": args.n,
        "n_qubits": n_qubits,
        "dim": padded_sparse.shape[0],
        "hermitian": hermitian,
        "seed": args.seed if not hermitian else None,
        "nnz": int(padded_sparse.nnz),
        "atol": args.atol,
        "timeout_note": args.timeout_note,
        "streaming": args.streaming,
    }

    if args.streaming:
        # Never materialize the combined term dict - see --streaming's
        # help text for why fwht_pauli_terms cannot be used at this
        # scale regardless of RAM.
        t0 = time.perf_counter()
        chunk_iter = fwht_pauli_terms_iter(
            padded_sparse, chunk_size=args.chunk_size, assume_hermitian=hermitian
        )
        proj_result = verify_terms_streaming(padded_sparse, chunk_iter, atol=args.atol)
        proj_time = time.perf_counter() - t0
        paulikit_time = proj_time  # decomposition and verification are interleaved per chunk
        result["paulikit_time_s"] = None  # not separable from verification time in streaming mode
        result["paulikit_n_terms"] = proj_result["n_terms"]
        result["projection_verification"] = {
            **proj_result,
            "wall_time_s": proj_time,
            "us_per_term": (proj_time / proj_result["n_terms"]) * 1e6 if proj_result["n_terms"] else 0.0,
        }
    else:
        t0 = time.perf_counter()
        paulikit_terms = fwht_pauli_terms(
            padded_sparse, assume_hermitian=hermitian, chunk_size=args.chunk_size
        )
        paulikit_time = time.perf_counter() - t0
        result["paulikit_time_s"] = paulikit_time
        result["paulikit_n_terms"] = len(paulikit_terms)

        if args.against in ("projection", "both"):
            t0 = time.perf_counter()
            proj_result = verify_terms(padded_sparse, paulikit_terms, atol=args.atol)
            proj_time = time.perf_counter() - t0
            result["projection_verification"] = {
                **proj_result,
                "wall_time_s": proj_time,
                "us_per_term": (proj_time / proj_result["n_terms"]) * 1e6,
            }

    if args.against in ("pennylane", "both"):
        import pennylane as qml

        t0 = time.perf_counter()
        pl_result = qml.pauli_decompose(padded_sparse, check_hermitian=not (not hermitian))
        pl_time = time.perf_counter() - t0
        pl_coeffs, pl_ops = pl_result.terms()

        from paulikit.algorithms.fwht import pauli_label

        pl_terms = {}
        for coeff, op in zip(pl_coeffs, pl_ops):
            wire_map = {w: w for w in range(n_qubits)}
            label_chars = ["I"] * n_qubits
            pauli_rep = op.pauli_rep
            if pauli_rep is not None:
                ((pw, _coeff),) = pauli_rep.items()
                for wire, letter in pw.items():
                    label_chars[wire] = letter
            label = "".join(label_chars)
            pl_terms[label] = complex(coeff)

        matched = 0
        max_diff = 0.0
        for label, c in paulikit_terms.items():
            if label in pl_terms:
                matched += 1
                max_diff = max(max_diff, abs(complex(c) - pl_terms[label]))

        result["pennylane_verification"] = {
            "wall_time_s": pl_time,
            "n_terms": len(pl_terms),
            "matched_labels": matched,
            "paulikit_n_terms": len(paulikit_terms),
            "max_abs_diff_on_matched": max_diff,
            "term_count_equal": len(pl_terms) == len(paulikit_terms),
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    herm_tag = "hermitian" if hermitian else "nonhermitian"
    fname = f"N{args.n}_{herm_tag}_{time.strftime('%Y%m%d')}.json"
    out_path = RESULTS_DIR / fname
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
