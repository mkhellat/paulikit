# Installation

Full installation reference. For the common case, the README's
Installation section is enough; this document covers the build
system's requirements, the editable-install sequencing, and the
optional native extension in detail.

paulikit is built with [meson-python](https://mesonbuild.com/meson-python/)
(the same build backend NumPy and SciPy use), and optionally compiles a
native (Cython/C++) `pauli_label` kernel, which materially reduces
label-generation cost — see [Native extension](#native-extension)
below.

**Recommended for development:** run `./configure` from this
directory. It creates/reuses a dedicated venv (default
`~/.venvs/paulikit`, deliberately outside the source tree — meson
rejects an absolute in-tree numpy include path if the venv lives
inside the project root), prints an itemized capability/environment
diagnostic report (compiler, Cython, TBB, cache hierarchy, NumPy's
BLAS backend, etc.; run `./configure --help` for the full list), and
generates a `Makefile` with the standard GNU set of
targets (`all`/`build`/`install`/`install-strip`/`installdirs`/
`check`/`test`/`installcheck`/`uninstall`/`docs`/`dist`/`TAGS`/
`clean`/`mostlyclean`/`distclean`/`maintainer-clean`/`report`) that
handle the `--no-build-isolation` editable-install sequencing below
correctly and automatically:

```bash
./configure                 # accepts --prefix/--docdir/--srcdir/VAR=value
                             # and the standard GNU dir-var options too, see --help
make                         # = make all = make build (editable install)
make check                   # run the test suite
```

If you'd rather not use `./configure`, the manual sequence it
automates is:

```bash
pip install numpy meson-python cython ninja
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required for editable installs: without it,
NumPy's include path gets baked in from a throwaway build-isolation
environment that goes stale on later rebuilds (this is NumPy's own
documented practice for meson-python editable installs, not a
paulikit-specific quirk) — and `numpy`/`meson-python`/`cython`/`ninja`
must already be installed in the target environment first, since
`--no-build-isolation` means pip won't fetch them into a throwaway
env for you the way it normally would. A regular, non-editable
`pip install .` does not need any of this.

With test/profiling dependencies:

```bash
pip install -e ".[test]" --no-build-isolation   # pytest, PennyLane (for fixture regeneration/reference checks), scipy
pip install -e ".[dev]" --no-build-isolation    # the above, plus Cython and ruff
pip install -e ".[sparse]" --no-build-isolation # just scipy, for build_hamiltonian(sparse=True) / pad_to_power_of_two(sparse=True)
```

Requires Python >= 3.10. Runtime dependencies are just `numpy` — the
core algorithms have no dependency on PennyLane, Qiskit, or Classiq;
those are only used in the `test`/`dev` extras, for generating and
cross-checking correctness fixtures. `scipy` is likewise optional:
only needed for the `sparse=True` path on `build_hamiltonian`,
`pad_to_power_of_two`, and (as an input type)
`fwht_pauli_coefficients`/`fwht_pauli_terms` -
calling any of those with `sparse=True` (or passing a
`scipy.sparse` operator directly) without `scipy` installed raises a
clear `ImportError` naming the `sparse` extra, rather than silently
falling back to the dense path. Building from source always
requires `meson-python`, `Cython`, and `numpy` (PEP 517/518's
`[build-system] requires` has no conditional mechanism), but both are
pure-Python-installable — no C toolchain is needed just to build the
pure-Python parts of the package.


## Native extension

By default (`-Dnative=auto`) the build compiles
`paulikit._native.pauli_label_native`, a Cython/C++ port of the
per-term label-generation kernel, if a C++ toolchain and
[oneTBB](https://github.com/oneapi-src/oneTBB) are available. If they
aren't, the build falls back to pure Python automatically — no error,
just slower label generation, with a one-time `UserWarning` the first
time the fallback path actually runs.

To force the behavior explicitly:

```bash
# Fail the build if the native extension can't be compiled:
pip install -e . --no-build-isolation --config-settings=setup-args="-Dnative=enabled"

# Force pure-Python-only, even if a toolchain is available:
pip install -e . --no-build-isolation --config-settings=setup-args="-Dnative=disabled"
```

The native extension is currently an optional, best-effort
accelerator, not a hard requirement — paulikit has no prebuilt-wheel
CI yet, so requiring a C++ toolchain for every `pip install` would be
too heavy a default. This is a deliberate, temporary trade-off, not
a permanent architecture decision. Migrating to prebuilt wheels (so
the extension can
become a hard requirement, matching the NumPy/SciPy model) is tracked
as a near-term goal, not indefinitely deferred.


