# Contributing to paulikit

Thank you for your interest. This document covers how to build the
package, how to run its tests, and what a change is expected to carry
with it.

## Where the project lives

Development happens at
[codeberg.org/beavernets/paulikit](https://codeberg.org/beavernets/paulikit),
mirrored to
[github.com/mkhellat/paulikit](https://github.com/mkhellat/paulikit).
Issues and pull requests are welcome at either.

## Building

The package uses [meson-python](https://mesonbuild.com/meson-python/),
the same build backend as NumPy and SciPy. A `configure` script wraps
the setup:

```bash
./configure          # creates a venv, reports what the toolchain offers
make                 # editable install with the correct sequencing
make check           # run the test suite
```

`./configure` prints an itemised capability report - compiler, Cython,
oneTBB, cache hierarchy, NumPy's BLAS backend - and then generates a
Makefile with the standard GNU targets. Run `./configure --help` for
the options.

If you would rather not use it, `docs/installation.md` gives the
manual sequence. The one thing that matters is `--no-build-isolation`
on editable installs; without it NumPy's include path is baked in from
a throwaway environment and goes stale on the next rebuild.

## Running the tests

```bash
make check                       # or: pytest
make test-native                 # the standalone C and C++ kernel self-tests
```

`pyproject.toml` sets `addopts = "-m 'not slow'"`, so the eight
benchmark comparisons are skipped by default. To run them, override
that rather than adding to it:

```bash
pytest -o addopts="" -m slow
```

The suite passes with no compiled extensions and no development
dependencies installed - a few tests skip in that configuration, which
is deliberate. That is the same environment the pure-Python fallback
exists to serve, so it needs to stay green there.

## What a change should carry

- **A test.** A bug fix should come with a test that fails without it.
  A new code path should come with tests that cover it.
- **The reason, not just the change.** Comments and docstrings here
  record why something is the way it is, including approaches that
  were measured and rejected. That is the most expensive information
  in the codebase and the easiest to lose.
- **A measurement, if the claim is about performance.** Wall-clock
  time on a laptop is not evidence: CPU frequency scaling alone moves
  it by a factor of two. Prefer instruction and cycle counts
  (`perf stat -e instructions:u,cycles:u`), report the problem size,
  and replicate before quoting a ratio.
- **Documentation, if behaviour changed.** Including the docstring, so
  the API reference stays correct.

## Style

`ruff` is configured in `pyproject.toml` (line length 92) and is part
of the `dev` extra. Most of the codebase is wrapped tighter than that,
at 79, and new code should follow the file it lives in.

Commit messages follow the GNU convention: a short imperative subject
line, a blank line, then a body explaining *why* the change is being
made, wrapped at 72 characters. The existing history is the reference.

## Reporting a bug

Please include the paulikit version, the Python version, the operating
system, whether the compiled extensions were built (`./configure`
reports this), and the smallest input that reproduces the problem.

For a decomposition that produces a wrong result, the most useful
report includes the operator - or the code that builds it - and the
term that disagrees.

## Licence

paulikit is GPL-3.0-or-later. By contributing you agree that your
contribution is licensed under the same terms.
