"""Internal package: the optional compiled extensions.

Four extensions may be built here, gated by three feature options in
``meson.options``:

    wht_kernel   ``wht_native``         the Walsh-Hadamard butterfly
                 ``coeffs_native``      phase, scaling, thresholding
                                        and index emission
    native       ``pauli_label_native`` Pauli label strings
    cache_probe  ``cache_probe``        empirical cache sizing

Which are present depends on the toolchain available at build time.
``wht_native``, ``coeffs_native`` and ``cache_probe`` need only a C
compiler and Cython. ``pauli_label_native`` additionally needs C++
and oneTBB, so it is the one most likely to be absent from a prebuilt
wheel.

Nothing outside ``paulikit.algorithms`` should import these directly.
Callers go through the public API, which handles a missing extension
by falling back - to NumPy for the kernels, and to declared cache
sizes for the probe.
"""
