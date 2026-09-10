"""Runtime auto-tuning heuristics for ``chunk_size`` and the
streaming-vs-dense decision.

Two independent pieces, both designed explicitly for correctness on
shared HPC nodes (not just workstations) per direct user instruction:

- ``recommended_chunk_size`` - targets a per-chunk working-set size
  that fits within a cache boundary, found via an *empirical*
  pointer-chase probe (``paulikit._native.cache_probe``) rather than
  any declared cache-topology source (``lscpu``/``/sys`` parsing was
  found ambiguous). Falls back to declared-size detection if the
  compiled probe extension isn't available.
- ``available_memory_bytes`` - the memory budget the streaming-vs-dense
  decision (``fwht.auto_decompose``) is based on: the minimum of
  physically-available memory (``/proc/meminfo``'s ``MemAvailable``,
  falling back to the POSIX ``SC_AVPHYS_PAGES`` figure) and any
  cgroup memory cap (v2 then v1) - the cgroup check is the
  correctness-critical piece on a shared/multi-tenant HPC node, where
  a job is frequently capped below the node's physical RAM.

Both results are cached per-process (module-level, lazy) - not
persisted to disk, since a persistent cache would be stale the moment
the process runs on different hardware (a real risk in HPC/container
contexts, not hypothetical).

**Thread safety**: both caches are populated under a lock
(``_cache_lock``), so concurrent callers from multiple threads in one
process (e.g. ``auto_decompose`` invoked from several worker threads
at once - a real pattern under heavy/aggressive parallelism, not just
multi-process HPC use) cannot race to call the underlying probe/memory
detection more than once. Without this, two threads could both
observe an empty cache and both invoke the cache-latency probe
concurrently - a different, untested failure mode (simultaneous
CPU/cache contention between two probes) than the sequential-pollution
bug the probe's own warm-up logic is designed against (see
``cache_probe_idempotency_investigation_findings.md``) - closed here
by construction rather than left as a residual risk. Does not protect
against separate *processes* racing (each process has its own
independent cache; that pattern was checked and found safe on its own
- see the same findings doc - since each process's own first, single
probe call was independently confirmed reliable).
"""

from __future__ import annotations

import math
import os
import threading
import warnings

try:
    from paulikit._native import cache_probe as _cache_probe
except ImportError:
    _cache_probe = None

_WARNED_NO_CACHE_PROBE = False

# Module-level, per-process caches (see module docstring for why not
# persisted to disk) and the lock guarding their population (see
# module docstring's "Thread safety" section).
_cache_lock = threading.Lock()
_cached_l2_bytes: int = -1  # see _L2_BYTES_UNSET below
_cached_memory_budget_bytes: int | None = None


def _warn_no_cache_probe() -> None:
    global _WARNED_NO_CACHE_PROBE
    if not _WARNED_NO_CACHE_PROBE:
        warnings.warn(
            "paulikit's compiled cache-latency probe is not available "
            "(built with -Dcache_probe=disabled, or Cython >= 3.0 was "
            "missing at build time) - falling back to declared cache-size "
            "detection for chunk_size auto-tuning, which is less reliable "
            "(declared sizes can be ambiguous, e.g. aggregate-vs-per-"
            "core). Rebuild paulikit with Cython >= 3.0 available to "
            "get the empirical probe.",
            stacklevel=3,
        )
        _WARNED_NO_CACHE_PROBE = True


def _declared_l2_size_bytes() -> int | None:
    """Best-effort per-core L2 size from ``/sys`` (Linux only).

    Returns ``None`` if unavailable (non-Linux, sysfs not mounted,
    permission denied, etc.) - callers must have their own final
    fallback. Deliberately does NOT use ``lscpu``: it was found that
    `lscpu`'s reported cache sizes are aggregated across cores, not
    the per-core figure a single-threaded chunk computation actually
    sees.
    """
    cache_dir = "/sys/devices/system/cpu/cpu0/cache"
    try:
        entries = os.listdir(cache_dir)
    except OSError:
        return None

    for entry in entries:
        index_dir = os.path.join(cache_dir, entry)
        level_path = os.path.join(index_dir, "level")
        type_path = os.path.join(index_dir, "type")
        size_path = os.path.join(index_dir, "size")
        try:
            with open(level_path) as f:
                level = f.read().strip()
            with open(type_path) as f:
                cache_type = f.read().strip()
            if level != "2" or cache_type not in ("Unified", "Data"):
                continue
            with open(size_path) as f:
                size_str = f.read().strip()
            if size_str.endswith("K"):
                return int(size_str[:-1]) * 1024
            if size_str.endswith("M"):
                return int(size_str[:-1]) * 1024 * 1024
            return int(size_str)
        except (OSError, ValueError):
            continue
    return None


def _detect_l2_boundary_bytes_via_probe() -> int | None:
    """Runs the empirical cache probe, returns the buffer size at the
    first ratio jump exceeding 1.3x (matching ``configure
    --probe-cache-latency``'s own boundary-detection threshold) past
    the first jump - i.e. the L1/L2 boundary is expected to be the
    first jump, L2/L3 the second; this returns the *second* jump's
    lower-side buffer size (the largest size that still behaves like
    L2, used as the L2 capacity estimate). Returns ``None`` if fewer
    than two boundary jumps are detected (unusual/flat hierarchy - not
    trusted enough to build a heuristic on).
    """
    samples = _cache_probe.probe_cache_boundaries()
    if len(samples) < 3:
        return None

    boundary_indices = []
    for i in range(1, len(samples)):
        prev_cycles = samples[i - 1][1]
        cur_cycles = samples[i][1]
        if prev_cycles <= 0:
            continue
        ratio = cur_cycles / prev_cycles
        if ratio > 1.3:
            boundary_indices.append(i)

    if len(boundary_indices) < 2:
        return None

    # samples[boundary_indices[1] - 1] is the last size that still
    # behaved like the level below the second detected boundary - our
    # L2 capacity estimate.
    return samples[boundary_indices[1] - 1][0]


# Real, measured (dim, best-chunk_size) anchor points.
# Best chunk_size decreases monotonically as dim grows (fixed
# per-chunk overhead dominates at small dim/total-work; at large dim
# even chunk_size=1 has enough work per chunk to amortize that
# overhead, and the cache-locality target itself shifts from L2
# toward L3). The relationship does NOT resolve to one clean
# closed-form fit to these 4 points alone (verified: neither a fixed
# constant nor a simple chunk_size = K/dim fits both ends, including
# an unfilled dim=2048..16384 gap) - piecewise log-log interpolation
# between real anchors is the honest choice here, not a fabricated
# formula extrapolated from sparse data.
_FLOOR_ANCHORS_DIM_TO_CHUNK_SIZE: tuple[tuple[int, int], ...] = (
    (512, 8),  # N=25 - measured best, chunk_size=2 was ~57% slower
    (2048, 8),  # N=50 - measured best, chunk_size=2 was ~12% slower
    (16384, 2),  # N=150 - measured best, ~11% faster than the old floor of 8
    (32768, 1),  # N=200 - measured best, ~22% faster than the old floor of 8
)


def _min_chunk_size_floor(dim: int) -> int:
    """Dim-dependent floor below which chunk_size hurts more than it
    helps, derived from real measurement at 4 anchor dims (see
    ``_FLOOR_ANCHORS_DIM_TO_CHUNK_SIZE``), not a single static guess.

    Below the smallest measured dim, or above the largest, clamps to
    that anchor's value rather than extrapolating past measured data.
    Between anchors, interpolates log-linearly in both dim and
    chunk_size (both anchor columns span roughly an order of magnitude
    each) - a reasoned interpolation of real endpoints, not a fit to
    an assumed closed form; the dim=2048..16384 gap between anchors is
    real and unfilled, so this is a deliberate compromise for that
    range, not a fully re-verified value.
    """
    anchors = _FLOOR_ANCHORS_DIM_TO_CHUNK_SIZE
    if dim <= anchors[0][0]:
        return anchors[0][1]
    if dim >= anchors[-1][0]:
        return anchors[-1][1]

    for (dim_lo, cs_lo), (dim_hi, cs_hi) in zip(anchors, anchors[1:]):
        if dim_lo <= dim <= dim_hi:
            if cs_lo == cs_hi:
                return cs_lo
            log_dim_lo, log_dim_hi = math.log(dim_lo), math.log(dim_hi)
            log_cs_lo, log_cs_hi = math.log(cs_lo), math.log(cs_hi)
            t = (math.log(dim) - log_dim_lo) / (log_dim_hi - log_dim_lo)
            interpolated = math.exp(log_cs_lo + t * (log_cs_hi - log_cs_lo))
            return max(1, round(interpolated))

    # Unreachable given the clamps above, but keeps the function total.
    return anchors[-1][1]


_L2_BYTES_UNSET = -1  # sentinel: probe/declared-size lookup not yet attempted


def _l2_bytes_or_none() -> int | None:
    """Empirically-measured or declared L2 boundary in bytes, cached
    per-process after the first call (dim-independent, safe to cache
    for the whole process lifetime - see module docstring's "Thread
    safety" section: population is lock-guarded, so concurrent callers
    cannot race to invoke the underlying probe more than once).
    Returns ``None`` if neither the probe nor ``/sys`` could determine
    it (that ``None`` result is itself cached, not re-attempted).
    """
    global _cached_l2_bytes
    if _cached_l2_bytes != _L2_BYTES_UNSET:
        return _cached_l2_bytes

    with _cache_lock:
        if _cached_l2_bytes != _L2_BYTES_UNSET:
            return _cached_l2_bytes

        l2_bytes: int | None = None
        if _cache_probe is not None:
            try:
                l2_bytes = _detect_l2_boundary_bytes_via_probe()
            except RuntimeError:
                l2_bytes = None  # no cycle-counter instruction - fall back
        else:
            _warn_no_cache_probe()

        if l2_bytes is None:
            l2_bytes = _declared_l2_size_bytes()

        _cached_l2_bytes = l2_bytes
        return _cached_l2_bytes


def recommended_chunk_size(dim: int) -> int:
    """Auto-computed ``chunk_size`` for a given operator dimension.

    Targets a per-chunk working-set size (``chunk_size * dim * 16``
    bytes - one row of ``dim`` complex128 entries per active row,
    matching ``fwht_pauli_coefficients``'s own accounting) that fits
    within the empirically-measured L2 cache boundary, subject to
    ``_min_chunk_size_floor()``'s lower bound.

    The L2-boundary lookup (dim-independent) is cached per-process
    after its first call, but the ``chunk_size`` computed from it is
    recomputed fresh on every call, since it depends on ``dim`` -
    caching the final result itself would silently reuse the first
    call's ``dim`` forever (a real bug found and fixed 2026-09-04, see
    REVIEW_NOTES.md: a process computing ``recommended_chunk_size(512)``
    then ``recommended_chunk_size(16384)`` would wrongly get the same
    answer for both).
    """
    l2_bytes = _l2_bytes_or_none()

    if l2_bytes is None:
        # Neither the probe nor /sys worked (non-Linux without the
        # compiled probe, e.g.) - conservative fixed fallback,
        # matches the smallest value validated as safe across
        # N=25/50/100 (chunk_size=32).
        return 32

    bytes_per_row = dim * 16  # complex128
    return max(_min_chunk_size_floor(dim), l2_bytes // max(bytes_per_row, 1))


def _read_meminfo_available_bytes() -> int | None:
    """Linux ``/proc/meminfo``'s ``MemAvailable`` field, in bytes.

    This is the same figure ``free -h``'s "available" column reports -
    correctly counts reclaimable page cache/buffers as usable, unlike
    the POSIX ``SC_AVPHYS_PAGES`` figure (confirmed on the development
    machine: ``SC_AVPHYS_PAGES`` read ~1.5 GiB "free" while
    ``MemAvailable`` correctly read ~10 GiB "available"). Returns
    ``None`` if ``/proc/meminfo`` doesn't exist or lacks this field
    (non-Linux).
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    # Format: "MemAvailable:   12345678 kB"
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _posix_available_physical_bytes() -> int | None:
    """POSIX-standard fallback: ``SC_AVPHYS_PAGES * SC_PAGESIZE``.

    Reports *free* physical memory, not *available* (does not count
    reclaimable cache as usable) - more conservative than
    ``_read_meminfo_available_bytes`` where that is available. Kept as
    the fallback for non-Linux/no-``/proc`` systems, using only
    POSIX-standard APIs (no third-party dependency, no shelling out -
    per explicit instruction).
    """
    try:
        avphys = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGESIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if avphys < 0 or page_size < 0:
        return None
    return avphys * page_size


def _cgroup_memory_limit_bytes() -> int | None:
    """Cgroup-imposed memory cap, if any - the correctness-critical
    check for shared HPC nodes, where a scheduler (Slurm/PBS) commonly
    caps a job below the node's physical RAM via a cgroup. Checks v2
    first, falls back to v1. Returns ``None`` if no limit is set
    (unlimited) or cgroups aren't present (not an HPC/container
    context, or cgroups not delegated to this process).
    """
    # cgroup v2: a single unified value, "max" means unlimited.
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            value = f.read().strip()
        if value != "max":
            return int(value)
        return None
    except (OSError, ValueError):
        pass

    # cgroup v1 fallback: often a huge sentinel (e.g.
    # 9223372036854771712) when effectively unset - only trust it if
    # it's smaller than total physical memory, otherwise treat as
    # unlimited.
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            value = int(f.read().strip())
        try:
            total_phys = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGESIZE")
        except (ValueError, OSError, AttributeError):
            total_phys = None
        if total_phys is not None and value >= total_phys:
            return None  # sentinel value, not a real limit
        return value
    except (OSError, ValueError):
        return None


def available_memory_bytes() -> int:
    """The memory budget for the streaming-vs-dense auto-decision.

    The minimum of (a) currently-available physical memory
    (``/proc/meminfo``'s ``MemAvailable``, falling back to POSIX
    ``SC_AVPHYS_PAGES``) and (b) any cgroup memory cap (v2 then v1) -
    whichever is smaller and present. Using only the physical figure
    would be actively unsafe on a shared HPC node where a job is
    cgroup-capped well below the node's full RAM.

    Cached per-process after the first call - see module docstring
    (including its "Thread safety" section: population is
    lock-guarded).
    """
    global _cached_memory_budget_bytes
    if _cached_memory_budget_bytes is not None:
        return _cached_memory_budget_bytes

    with _cache_lock:
        if _cached_memory_budget_bytes is not None:
            return _cached_memory_budget_bytes

        physical = _read_meminfo_available_bytes()
        if physical is None:
            physical = _posix_available_physical_bytes()
        if physical is None:
            # Nothing worked (unusual platform) - a conservative fixed
            # fallback rather than raising, so auto_decompose()
            # degrades to "always stream" rather than crashing.
            physical = 512 * 1024 * 1024

        cgroup_limit = _cgroup_memory_limit_bytes()
        budget = physical if cgroup_limit is None else min(physical, cgroup_limit)

        _cached_memory_budget_bytes = budget
        return budget


def per_worker_memory_budget_bytes(n_workers: int) -> int:
    """Correctness fix for using ``available_memory_bytes()`` under
    multi-process parallelism.

    ``available_memory_bytes()`` answers "how much memory can *this
    process* safely use" - correct for a single lone process, its
    original design, but if reused unchanged as each of
    ``n_workers`` *concurrent* worker processes' own individual
    budget, the *sum* of what they might use is ``n_workers`` times
    the actual node/cgroup limit - a real, silent OOM risk under
    parallelism that the original memory-budget work was never
    exposed to (a single process there never shared its budget with
    concurrent siblings). This divides the one shared budget evenly
    across workers instead - conservative (assumes every worker peaks
    simultaneously, which is the safe assumption, not the average
    case) but correctness-critical on a shared/multi-tenant node, the
    same priority the original cgroup-awareness work was built for.

    Args:
        n_workers: Number of concurrent worker processes that will
            share the budget returned by ``available_memory_bytes()``.
            Must be >= 1.

    Returns:
        ``available_memory_bytes() // n_workers`` - each worker's own
        safe ceiling, not the whole-node figure.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    return available_memory_bytes() // n_workers
