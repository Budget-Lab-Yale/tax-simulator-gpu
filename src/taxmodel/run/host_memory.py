"""Host memory accounting, and returning freed memory to the operating system.

On a machine with a graphics card the memory question is about the card. On a
laptop there is one pool of memory and several claimants on it: the population
read from disk, the XLA compiler while it builds a program, the compiled
programs while they run, every scenario's resident arrays, and the finished
answers waiting to be written out. Whether a run finishes on a 16 GB machine
is decided by the sum.

Two claimants free large amounts of memory and then do not return it to the
operating system, because holding it makes their next allocation faster:

- Arrow's memory pool, which the Parquet reader that reads the population
  draws on. It keeps the buffers a finished read used.
- The C library's heap, which the XLA compiler allocates from. Building a
  projection year takes gigabytes of compiler working memory, and when the
  compile is done the C library keeps that space for the next caller.

This module reports what is held, asks for it back, and on Linux sets the C
library's allocation thresholds so that the largest holder never builds up.

Phase ledger
------------
Set ``TAXMODEL_MEMORY_LEDGER=on`` and the run prints a line at each named
point: the process's resident memory, Arrow's pool, the total bytes of live
JAX arrays, and what a trim of the C heap and Arrow's pool gives back. The
lines are for measurement only; nothing the model computes reads them.

The operating system's own memory figures come from a different place on each
system: `/proc/self/status` on Linux, `GetProcessMemoryInfo` on Windows, and
`resource.getrusage` elsewhere, which reports the peak alone. A figure a
system does not offer prints as `n/a`.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
from pathlib import Path

import pyarrow as pa

from taxmodel.run import hardware

LINUX = sys.platform.startswith("linux")
WINDOWS = sys.platform == "win32"

LEDGER_VARIABLE = "TAXMODEL_MEMORY_LEDGER"
PYTHON_HEAP_VARIABLE = "TAXMODEL_MEMORY_PYTHON_HEAP"
TRIM_EVERY_YEAR_VARIABLE = "TAXMODEL_MEMORY_TRIM_EVERY_YEAR"

GIGABYTE = 1024**3


def ledger_is_on() -> bool:
    """Whether the run was asked to print the phase ledger."""
    return os.environ.get(LEDGER_VARIABLE, "").strip().lower() in (
        "on",
        "1",
        "yes",
        "true",
    )


def trim_after_every_year() -> bool:
    """Whether the year loop should return freed memory after every year.

    The compiled year's temporaries are allocated and freed through the C
    library, which by default keeps the space for the next call. Over thirty
    years the space it keeps grows. This asks for it back each year.
    """
    return os.environ.get(TRIM_EVERY_YEAR_VARIABLE, "").strip().lower() in (
        "on",
        "1",
        "yes",
        "true",
    )


def python_heap_is_traced() -> bool:
    """Whether the run was asked to trace Python's own allocations."""
    return os.environ.get(PYTHON_HEAP_VARIABLE, "").strip().lower() in (
        "on",
        "1",
        "yes",
        "true",
    )


def start_tracing_the_python_heap() -> None:
    """Begin recording where Python-level allocations come from, if asked.

    This covers memory allocated through Python's allocator, which includes
    numpy and pandas buffers but not what a C library such as XLA or Arrow
    allocates for itself. It roughly doubles the cost of every allocation, so
    it is off unless asked for.
    """
    if not python_heap_is_traced():
        return
    import tracemalloc

    tracemalloc.start(1)


def report_the_python_heap(phase: str, how_many: int = 15) -> None:
    """Print the largest Python-level allocations, grouped by source line."""
    if not python_heap_is_traced():
        return
    import tracemalloc

    if not tracemalloc.is_tracing():
        return
    snapshot = tracemalloc.take_snapshot()
    traced, _ = tracemalloc.get_traced_memory()
    print(
        f"python heap at {phase}: {traced / GIGABYTE:.2f} GB traced, "
        f"largest {how_many} sources:",
        flush=True,
    )
    for statistic in snapshot.statistics("lineno")[:how_many]:
        print(f"    {statistic.size / GIGABYTE:6.3f} GB  {statistic}", flush=True)


def _proc_status_kilobytes(field: str) -> float:
    """One VmRSS-style field of /proc/self/status, in kilobytes."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return float(line.split()[1])
    except OSError:
        pass
    return float("nan")


class _WindowsMemoryCounters(ctypes.Structure):
    """The PROCESS_MEMORY_COUNTERS structure GetProcessMemoryInfo fills in.

    The two fields the ledger reads are the working set, which is the Windows
    name for resident memory, and its high-water mark. The rest are declared
    because the structure has to be the size Windows expects.
    """

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_working_set_gigabytes() -> tuple[float, float]:
    """This process's working set and its high-water mark, from Windows.

    Both are missing rather than wrong if the call fails.
    """
    missing = (float("nan"), float("nan"))
    try:
        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        psapi = ctypes.WinDLL("psapi")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return missing
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsMemoryCounters),
        ctypes.c_uint32,
    ]
    counters = _WindowsMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    filled = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not filled:
        return missing
    return (
        counters.WorkingSetSize / GIGABYTE,
        counters.PeakWorkingSetSize / GIGABYTE,
    )


def _getrusage_peak_gigabytes() -> float:
    """The peak resident memory this process has reached, from getrusage.

    macOS reports the peak in bytes and Linux in kilobytes; every other system
    the model might run on is treated as Linux.
    """
    import resource

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / GIGABYTE if sys.platform == "darwin" else peak / 1024**2


def resident_gigabytes() -> float:
    """How much memory this process holds, as the operating system counts it.

    macOS has no cheap figure for this, so it comes back missing there and the
    ledger prints the peak alone.
    """
    if LINUX:
        return _proc_status_kilobytes("VmRSS") / 1024**2
    if WINDOWS:
        return _windows_working_set_gigabytes()[0]
    return float("nan")


def peak_resident_gigabytes() -> float:
    """The highest resident memory this process has reached."""
    if LINUX:
        return _proc_status_kilobytes("VmHWM") / 1024**2
    if WINDOWS:
        return _windows_working_set_gigabytes()[1]
    return _getrusage_peak_gigabytes()


def arrow_pool_gigabytes() -> tuple[float, float]:
    """Arrow's memory pool: bytes it currently holds, and its own high-water mark."""
    pool = pa.default_memory_pool()
    return pool.bytes_allocated() / GIGABYTE, pool.max_memory() / GIGABYTE


def live_jax_gigabytes() -> tuple[float, int]:
    """The total size of every JAX array alive in this process, and how many there are.

    On the CPU backend a JAX array's buffer is ordinary host memory, so this is
    the part of the process's resident memory the model's own arrays account
    for.
    """
    import jax

    total = 0
    count = 0
    for array in jax.live_arrays():
        # A deleted or donated buffer no longer reports a size.
        if array.is_deleted():
            continue
        total += array.nbytes
        count += 1
    return total / GIGABYTE, count


def host_array_gigabytes(tree: object) -> float:
    """The total size of the plain NumPy arrays in a pytree.

    Arrays held on the host do not appear in ``jax.live_arrays``, so the two
    figures together account for the model's own arrays.
    """
    import jax
    import numpy as np

    total = 0
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, np.ndarray):
            total += leaf.nbytes
    return total / GIGABYTE


def note_host_holders(named: dict[str, object]) -> None:
    """Print the size of each named pytree of host arrays, if the ledger is on."""
    if not ledger_is_on():
        return
    for name, tree in named.items():
        print(
            f"memory ledger | host arrays: {name:<30s} "
            f"{host_array_gigabytes(tree):6.3f} GB",
            flush=True,
        )


def give_back_free_memory() -> None:
    """Return to the operating system what Arrow and the C library are not using.

    Costs a few milliseconds and changes no number the run reports. Only the
    GNU C library offers `malloc_trim`, so off Linux this releases Arrow's
    pool and nothing else.
    """
    pa.default_memory_pool().release_unused()
    if not LINUX:
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# glibc's parameter numbers for `mallopt`, from its malloc.h.
M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3
M_MMAP_MAX = -4

# Blocks of this size and over are served by their own mmap, and the ceiling
# on how many such blocks may be outstanding at once is raised well above the
# few thousand a run needs.
LARGE_BLOCK_BYTES = 128 * 1024
MMAP_BLOCK_LIMIT = 1_000_000

ALLOCATION_THRESHOLD_NOTE = (
    "the C library will serve blocks of 128 KB and over with mmap, so that "
    "freeing one returns it to the operating system"
)

_allocation_thresholds_are_set = False


def set_the_c_library_allocation_thresholds() -> str:
    """Ask the C library to give large freed blocks back to the operating system.

    A block the C library serves from its heap is kept for the next caller
    when it is freed; a block it serves with its own mmap is returned to the
    operating system at once. The compiled projection year allocates and frees
    gigabytes of temporaries in each year of the window, and the kept space
    accumulates over the window: the same thirty-year two-scenario run peaks
    at 26.0 GB of resident memory without this setting and 13.8 GB with it,
    computing identical numbers and taking about a tenth longer.

    Applies on Linux, and only where there is no graphics card, since a run on
    a card holds its arrays on the card. `taxmodel/__init__.py` calls this
    before JAX is imported, because the C library keeps the thresholds it was
    already using for memory it has already taken. Returns a line for the
    run's log, or an empty string where the setting does not apply; calling it
    again changes nothing and returns the same line.
    """
    global _allocation_thresholds_are_set
    if not LINUX:
        return ""
    if hardware.graphics_card_is_present() is not False:
        return ""
    if _allocation_thresholds_are_set:
        return ALLOCATION_THRESHOLD_NOTE
    try:
        mallopt = ctypes.CDLL("libc.so.6").mallopt
    except (OSError, AttributeError):
        return ""
    mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
    mallopt.restype = ctypes.c_int
    settings = (
        (M_MMAP_THRESHOLD, LARGE_BLOCK_BYTES),
        (M_TRIM_THRESHOLD, LARGE_BLOCK_BYTES),
        (M_MMAP_MAX, MMAP_BLOCK_LIMIT),
    )
    for parameter, value in settings:
        if mallopt(parameter, value) != 1:
            return (
                "the C library would not take the allocation thresholds; this "
                "run will hold more memory than it needs"
            )
    _allocation_thresholds_are_set = True
    return ALLOCATION_THRESHOLD_NOTE


def _figure(gigabytes: float, width: int = 6) -> str:
    """One memory figure for the ledger, or `n/a` where the system offers none."""
    if math.isnan(gigabytes):
        return "n/a".rjust(width)
    return f"{gigabytes:{width}.2f}"


def note(phase: str, *, trim: bool = False) -> None:
    """Print one line of the phase ledger, if the ledger was asked for.

    ``trim`` also asks the C library's heap and Arrow's pool to give back what
    they are not using, and reports how much resident memory that freed. Use
    it at points where something large has just finished, such as a compile.
    Trimming happens only while the ledger is on, so that a run with the
    ledger off measures the peak the model reaches on its own.
    """
    if not ledger_is_on():
        return
    resident = resident_gigabytes()
    peak = peak_resident_gigabytes()
    arrow_held, arrow_peak = arrow_pool_gigabytes()
    jax_bytes, jax_count = live_jax_gigabytes()
    line = (
        f"memory ledger | {phase:<44s} | resident {_figure(resident)} GB "
        f"| peak {_figure(peak)} GB | live JAX arrays {jax_bytes:6.2f} GB "
        f"in {jax_count:6d} | Arrow pool {arrow_held:5.2f} GB "
        f"(its peak {arrow_peak:5.2f} GB)"
    )
    if trim:
        report_the_python_heap(phase)
        give_back_free_memory()
        after = resident_gigabytes()
        line += f" | trim gave back {_figure(resident - after, width=5)} GB"
    print(line, flush=True)


__all__ = [
    "ALLOCATION_THRESHOLD_NOTE",
    "LEDGER_VARIABLE",
    "PYTHON_HEAP_VARIABLE",
    "arrow_pool_gigabytes",
    "give_back_free_memory",
    "host_array_gigabytes",
    "ledger_is_on",
    "live_jax_gigabytes",
    "note",
    "note_host_holders",
    "peak_resident_gigabytes",
    "python_heap_is_traced",
    "report_the_python_heap",
    "resident_gigabytes",
    "set_the_c_library_allocation_thresholds",
    "start_tracing_the_python_heap",
    "trim_after_every_year",
]
