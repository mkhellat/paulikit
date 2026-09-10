# cython: language_level=3
"""Cython binding for the empirical cache-latency probe.

Standalone extension, deliberately NOT part of ``pauli_label_native``
- gated only on Cython >= 3.0, with no oneTBB dependency (the probe
has no relationship to oneTBB; bundling them would wrongly make
chunk_size auto-tuning unavailable whenever oneTBB specifically is
missing). See meson.build's ``cache_probe`` extension entry (sibling
to ``pauli_label_native``'s ``native`` option).

This extension is OPTIONAL, same packaging stance as
``pauli_label_native``: paulikit must remain pip-installable without a
C++ toolchain. If this module fails to build or import, callers fall
back to declared-size cache detection - see
``paulikit.algorithms.fwht``'s chunk_size auto-tuner.
"""

from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

cdef extern from "cache_probe.h":
    ctypedef struct cache_probe_sample:
        size_t buffer_size_bytes
        double cycles_per_access

    size_t c_cache_probe_run "cache_probe_run"(
        size_t min_size_bytes,
        size_t n_sizes,
        int64_t reps,
        int repeats,
        cache_probe_sample *out,
    )


def probe_cache_boundaries(
    size_t min_size_bytes = 8192,
    size_t n_sizes = 13,
    int64_t reps = 300000,
    int repeats = 3,
):
    """Runs the pointer-chase probe, returns a list of
    ``(buffer_size_bytes, cycles_per_access)`` tuples, one per buffer
    size actually measured (doubling from ``min_size_bytes``; may be
    fewer than ``n_sizes`` entries if a large allocation fails - see
    ``cache_probe_run``'s docstring in cache_probe.h). Units are raw
    hardware cycles, not calibrated to wall-clock time - callers
    should only compare *ratios* between rows (see cache_probe.h for
    why cycle counters, not clock_gettime, are used).

    Raises ``RuntimeError`` if this architecture has no known
    cycle-counter instruction (``cycles_per_access`` would be all
    zeros, meaningless) - callers should fall back to declared-size
    cache detection in that case.

    Defaults match ``configure --probe-cache-latency``'s own probe
    (8 KiB start, 13 sizes -> 32 MiB ceiling, 300000 timed reps).
    """
    cdef cache_probe_sample *samples = <cache_probe_sample *>malloc(
        n_sizes * sizeof(cache_probe_sample)
    )
    if samples is NULL:
        raise MemoryError("could not allocate cache-probe sample buffer")

    cdef size_t written
    cdef size_t i
    try:
        written = c_cache_probe_run(min_size_bytes, n_sizes, reps, repeats, samples)
        result = []
        for i in range(written):
            result.append((samples[i].buffer_size_bytes, samples[i].cycles_per_access))
        if result and all(cycles == 0.0 for _, cycles in result):
            raise RuntimeError(
                "no hardware cycle-counter instruction available on this "
                "architecture - cache_probe results would be meaningless"
            )
        return result
    finally:
        free(samples)
