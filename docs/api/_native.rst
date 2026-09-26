paulikit._native
=====================

.. automodule:: paulikit._native
   :members:
   :undoc-members:
   :show-inheritance:

Optional compiled extensions (Cython/C and Cython/C++) used by
``paulikit.algorithms.fwht`` and ``paulikit.algorithms.autotune`` when
available, with pure-Python / NumPy / declared-cache fallbacks
otherwise. Three meson feature options gate them:

* ``wht_kernel`` — ``wht_native`` (and optional ``wht_native_v3``),
  ``coeffs_native`` (and optional ``coeffs_native_v3``),
  ``gather_native``, ``hermitian_check_native``
* ``native`` — serial ``pauli_label_native`` (C + Cython); optional
  ``pauli_label_parallel_native`` (C++ + oneTBB)
* ``cache_probe`` — ``cache_probe``

See :doc:`../installation` for build details and the rationale behind
keeping the extensions optional.

Autodoc note: this page only renders members for modules that were
compiled at doc-build time. With a feature disabled, the corresponding
extension modules do not exist and that part of the page is empty
aside from this note.
