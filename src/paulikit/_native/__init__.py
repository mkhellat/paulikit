# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Internal package: the optional compiled extensions.

Six extensions may be built here, gated by three feature options in
``meson.options``::

    wht_kernel   ``wht_native``         the Walsh-Hadamard butterfly
                 ``wht_native_v3``      same, compiled -march=x86-64-v3
                 ``coeffs_native``      phase, scaling, thresholding
                                        and index emission
                 ``coeffs_native_v3``   same, compiled -march=x86-64-v3
    native       ``pauli_label_native`` Pauli label strings
    cache_probe  ``cache_probe``        empirical cache sizing

Which are present depends on the toolchain available at build time.
``wht_native``, ``coeffs_native`` and ``cache_probe`` need only a C
compiler and Cython. ``pauli_label_native`` additionally needs C++
and oneTBB, so it is the one most likely to be absent from a prebuilt
wheel.

The ``_v3`` modules are the SAME source as their non-``_v3``
counterpart, compiled a second time under ``-march=x86-64-v3``
(AVX2 + BMI2 + FMA + POPCNT) when the build toolchain accepts that
flag. ``fwht.py`` picks between a module and its ``_v3`` twin at
IMPORT time, based on a runtime check of the CPU actually running the
process (not the machine that built the wheel) - never at build time,
so a wheel built on a newer CPU stays correct (just slower) on an
older one it later runs on.

Nothing outside ``paulikit.algorithms`` should import these directly.
Callers go through the public API, which handles a missing extension
by falling back - to NumPy for the kernels, and to declared cache
sizes for the probe.
"""
