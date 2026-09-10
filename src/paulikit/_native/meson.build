# The compiled fast path for paulikit.algorithms.fwht.fwht_pauli_terms.
# Optional (see meson.options's "native" feature) so paulikit stays
# pip-installable without a C++ toolchain or oneTBB; fwht.py falls
# back to pure Python with a visible warning if this module isn't
# built. The optional/fallback model is a deliberate temporary
# compromise, not the eventual target: the NumPy/SciPy model, where
# the extension is always required and shipped as prebuilt wheels, is
# the goal once wheel-building CI exists.

native_opt = get_option('native')

# have_cython / have_cpp come from add_languages() in the top-level
# meson.build. Guarding on them matters: meson.get_compiler('cython')
# is a hard error when the language was never successfully added, so
# calling it unguarded defeats the whole auto-fallback.
if not native_opt.disabled() and have_cython and have_cpp
  cy = meson.get_compiler('cython')
  tbb_dep = dependency('tbb', required: native_opt)
  cython_ok = cy.version().version_compare('>=3.0')

  if tbb_dep.found() and cython_ok
    np_include_path = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    py.extension_module(
      'pauli_label_native',
      [
        'pauli_label_native.pyx',
        'pauli_label.c',
        'pauli_label_parallel.cpp',
      ],
      include_directories: include_directories('.', np_include_path, is_system: true),
      dependencies: [tbb_dep],
      override_options: ['cython_language=cpp'],
      install: true,
      subdir: 'paulikit/_native',
    )
  elif native_opt.enabled()
    error('native extension explicitly requested (-Dnative=enabled) but ' +
          'oneTBB (found: @0@) or Cython >= 3.0 (found: @1@) is missing'.format(
            tbb_dep.found(), cython_ok))
  else
    message('paulikit: native extension not built (oneTBB or Cython >= 3.0 ' +
            'not found) - falling back to pure Python at import time')
  endif
else
  message('paulikit: native extension disabled via -Dnative=disabled')
endif

# The Walsh-Hadamard butterfly kernel - the compiled fast path for
# the transform itself, as opposed to the label generation above.
# Deliberately independent of the `native` option: this needs only a C
# compiler and Cython, with no C++ and no oneTBB, so a machine that
# cannot build pauli_label_native can still get the kernel that
# carries most of the per-chunk work. fwht.py falls back to the
# pure-Python butterfly if it is absent.
wht_opt = get_option('wht_kernel')

if not wht_opt.disabled() and have_cython and have_c
  cy3 = meson.get_compiler('cython')
  cython_ok3 = cy3.version().version_compare('>=3.0')

  if cython_ok3
    np_include_wht = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    py.extension_module(
      'wht_native',
      [
        'wht_native.pyx',
        'wht.c',
      ],
      include_directories: include_directories('.', np_include_wht, is_system: true),
      install: true,
      subdir: 'paulikit/_native',
    )
  elif wht_opt.enabled()
    error('WHT kernel explicitly requested (-Dwht_kernel=enabled) but ' +
          'Cython >= 3.0 was not found')
  else
    message('paulikit: wht_native not built (Cython >= 3.0 not found) - ' +
            'falling back to the pure-Python butterfly at import time')
  endif
else
  message('paulikit: wht_native disabled via -Dwht_kernel=disabled')
endif

# The fused coefficient kernel - phase factor,
# scaling, thresholding and index emission in one pass. Same
# C-and-Cython-only dependency set as the WHT kernel above, and shares
# its option: the two are a pair (transform, then emit) and there is
# no case for building one without the other.
if not wht_opt.disabled() and have_cython and have_c
  cy4 = meson.get_compiler('cython')

  if cy4.version().version_compare('>=3.0')
    np_include_coeffs = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    py.extension_module(
      'coeffs_native',
      [
        'coeffs_native.pyx',
        'coeffs.c',
      ],
      include_directories: include_directories('.', np_include_coeffs, is_system: true),
      install: true,
      subdir: 'paulikit/_native',
    )
  endif
endif

# The empirical cache-latency probe - standalone,
# Cython-only, deliberately independent of the native/oneTBB option
# above (the probe has no relationship to oneTBB; see cache_probe.pyx's
# own docstring for why bundling them would be wrong).
cache_probe_opt = get_option('cache_probe')

if not cache_probe_opt.disabled() and have_cython and have_c
  cy2 = meson.get_compiler('cython')
  cython_ok2 = cy2.version().version_compare('>=3.0')

  if cython_ok2
    py.extension_module(
      'cache_probe',
      [
        'cache_probe.pyx',
        'cache_probe.c',
      ],
      include_directories: include_directories('.'),
      install: true,
      subdir: 'paulikit/_native',
    )
  elif cache_probe_opt.enabled()
    error('cache_probe extension explicitly requested (-Dcache_probe=enabled) but ' +
          'Cython >= 3.0 was not found')
  else
    message('paulikit: cache_probe extension not built (Cython >= 3.0 not found) - ' +
            'falling back to declared-size cache detection at runtime')
  endif
else
  message('paulikit: cache_probe extension disabled via -Dcache_probe=disabled')
endif

# The __init__.py for the _native package itself (plain Python,
# always installed regardless of whether the extension built).
py.install_sources(
  '__init__.py',
  subdir: 'paulikit/_native',
)
