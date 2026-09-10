# Languages are declared OPTIONAL and detected below, not required
# here. Listing 'c', 'cpp' and 'cython' in project() makes every one of
# them mandatory at configure time, so a machine without a C++ compiler
# or without Cython fails the build outright - before meson.options's
# native=auto fallback is ever consulted. That contradicted the
# documented behaviour ("falls back to pure Python silently") and broke
# CI on runners that have no toolchain. The pure-Python path must stay
# installable with no compiler at all.
project(
  'paulikit',
  version: '0.1.0',
  license: 'GPL-3.0-or-later',
  meson_version: '>=1.1.0',
  default_options: ['cpp_std=c++17'],
)

have_c = add_languages('c', required: false, native: false)
have_cpp = add_languages('cpp', required: false, native: false)
have_cython = add_languages('cython', required: false, native: false)

py_mod = import('python')
py = py_mod.find_installation(pure: false)

subdir('src/paulikit')
