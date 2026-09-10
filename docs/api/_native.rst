paulikit._native
=====================

.. automodule:: paulikit._native
   :members:
   :undoc-members:
   :show-inheritance:

Optional compiled extension (Cython/C++) used by
``paulikit.algorithms.fwht`` when available, with a pure-Python
fallback otherwise. See :doc:`../installation` for build details and
the rationale behind keeping the extension optional.

Autodoc note: this page only renders members if the extension was
compiled at doc-build time (``-Dnative=auto`` or ``enabled``). With
``-Dnative=disabled``, ``paulikit._native.pauli_label_native`` doesn't
exist and this page will be empty aside from this note.
