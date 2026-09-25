# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, the public API may change between minor
versions; see the README's Status section.

## [Unreleased]

### Added

- Threaded drain for `parallel_decompose_arrays` (`executor="thread"` /
  `"auto"`), with physical-core pinning, capacity-weighted slices,
  work-stealing, and optional eager thread spin-up.
- Compiled gather and Hermiticity-check kernels under `wht_kernel`,
  plus runtime-dispatched x86-64-v3 twins of the WHT and coefficient
  kernels.
- GitHub Actions workflow that builds manylinux wheels (x86_64 /
  aarch64, CPython 3.10–3.13) and an sdist on tag / manual dispatch.

### Changed

- README Usage leads with the measured sparse/dense fastest paths;
  Sphinx tutorial, installation, and native docs match the multi-kernel
  surface; `--chunk-size` help no longer claims auto-tuning is absent
  from the library APIs.

### Fixed

- `make check` no longer dumps expected autotune/fork warnings from
  intentional no-probe and process-pool tests.

### Removed

- Stale Status/CHANGELOG claim of an unexplained 14-to-15 qubit
  parallel-efficiency step under the process-pool drain.

## [0.1.0] - 2026-09-10

First release.

### Added

- **Pauli decomposition by fast Walsh-Hadamard transform.**
  `paulikit.algorithms.fwht` decomposes any $2^n \times 2^n$ complex matrix
  into its Pauli-basis coefficients in $O(n \cdot 4^n)$. Hermitian and
  non-Hermitian input take the same transform; `assume_hermitian=True`
  (the default) returns real coefficients and verifies the input
  really is Hermitian rather than assuming it.

- **Bounded memory.** The active-x set is chunked and each chunk is
  thresholded and emitted before the next is computed, so peak
  resident memory is set by `chunk_size` rather than by the size of
  the result. Sparse input is kept sparse throughout.

- **Streaming entry points.** `fwht_pauli_terms_iter` and
  `parallel_decompose_arrays` yield per chunk, so a caller can consume
  and discard results that would not fit in memory if accumulated.

- **Multi-core execution.** `parallel_decompose` and
  `parallel_decompose_arrays` distribute independent chunks across a
  process pool or a thread pool. Worker count is taken from the CPUs
  actually available to the process rather than from `os.cpu_count()`,
  so a cgroup- or cpuset-restricted machine is not oversubscribed.

- **Checkpoint and restart.** A binary chunk-framed checkpoint format
  cheap enough to leave permanently enabled. Frames carry their own
  chunk index, so completed work replays on resume regardless of the
  order a pool returns it in, and a torn tail means exactly "that
  chunk was not recorded".

- **Optional compiled kernels.** Under three meson options:
  `wht_kernel` (Walsh–Hadamard butterfly, fused coefficients, dense
  gather, Hermiticity check, with optional x86-64-v3 twins), `native`
  (Pauli label kernel with a oneTBB-parallel variant), and
  `cache_probe` (empirical cache-latency probe). All are optional: the
  package is correct and installable with no C toolchain, and reports
  once when a fallback path is taken.

- **Cache-aware auto-tuning.** `paulikit.algorithms.autotune` sizes
  chunks against measured cache boundaries, interpolating between
  measured anchor points rather than applying a closed-form fit.

- **Command-line interface.** `paulikit decompose`, `paulikit
  benchmark` and `paulikit regenerate-fixtures`.

- **Exhaustive verification.** Every term of a decomposition checked
  individually against an independently derived projection oracle -
  not sampled - up to 91,652,096 terms at 14 qubits, Hermitian and
  non-Hermitian. Artifacts and method under `verification/`.

- **Test suite.** 323 default tests (8 further slow benchmark
  comparisons excluded by default).

- **Documentation.** Installation, tutorial, theory, background,
  non-Hermitian operators, annotated source tree, and an API
  reference built from the docstrings.

### Known limitations

- Prebuilt wheels are not yet published on PyPI, so the compiled
  extensions remain optional accelerators rather than hard
  requirements. CI builds manylinux wheels on tag / manual dispatch.
- CPU pinning and topology detection are Linux-only, with a documented
  fallback elsewhere; the non-Linux paths are not exercised in CI.

[Unreleased]: https://codeberg.org/beavernets/paulikit/compare/v0.1.0...HEAD
[0.1.0]: https://codeberg.org/beavernets/paulikit/releases/tag/v0.1.0
