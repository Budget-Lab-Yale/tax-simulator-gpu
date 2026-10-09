"""The on-disk cache of compiled programs.

XLA writes the programs it compiles into a directory and reads them back in a
later process. Compiling the year program is the largest fixed cost of a run,
and a warm cache cuts it by roughly three quarters. The cache is on by default.

A cache miss only costs the compile it would have saved. The cache key is built
from the compiled program's text, the compiler version, and the target device,
so changed code, an upgraded JAX, or a different card always recompiles. A
stale entry sits unread until it is evicted.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import jax

# The default cache location, under `~/.cache` or `XDG_CACHE_HOME`. Compiled
# programs are machine-specific build products and stay out of the repository
# and out of run directories.
CACHE_SUBDIRECTORY = Path("tax-simulator-gpu") / "xla"

# The size at which XLA starts evicting the least recently read entries. A
# ten-scenario run's programs are about 6 megabytes, so two gigabytes holds
# hundreds of versions of the model.
#
# JAX enforces the limit only when the `filelock` package is installed, because
# evicting entries another process may be reading needs a lock. Asking for a
# limit without the package makes every cache read and write fail with a
# warning. The limit is therefore set only when the package is present, and the
# directory grows without bound otherwise. Deleting the directory is always safe.
CACHE_SIZE_LIMIT_BYTES = 2 * 1024**3

# Programs that compiled faster than this are not stored. The year program and
# the report programs clear this easily; the many tiny programs a run also
# builds do not.
MINIMUM_COMPILE_SECONDS_TO_STORE = 2.0

# Environment variable holding a cache path to use instead of the default, or
# the word "off" for no cache.
OVERRIDE_VARIABLE = "TAXMODEL_COMPILE_CACHE"


def size_limit_can_be_enforced() -> bool:
    """Whether `filelock` is installed. JAX caps the cache only when it is."""
    return importlib.util.find_spec("filelock") is not None


def default_cache_directory() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / CACHE_SUBDIRECTORY


def switch_on_compile_cache() -> str:
    """Point XLA at the cache directory and return the line to print about it.

    Call this before anything in the run compiles. Setting
    `TAXMODEL_COMPILE_CACHE` to a path uses that path instead, and setting it
    to "off" runs with no cache.
    """
    override = os.environ.get(OVERRIDE_VARIABLE, "").strip()
    if override.lower() == "off":
        jax.config.update("jax_enable_compilation_cache", False)
        return (
            f"compile cache: off, because {OVERRIDE_VARIABLE}=off. "
            "Every program in this run compiles from scratch."
        )

    if override:
        directory = Path(override).expanduser()
    elif os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        # A job script that named its own cache directory keeps it.
        directory = Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).expanduser()
    else:
        directory = default_cache_directory()

    directory.mkdir(parents=True, exist_ok=True)
    capped = size_limit_can_be_enforced()
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    if capped:
        jax.config.update("jax_compilation_cache_max_size", CACHE_SIZE_LIMIT_BYTES)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        MINIMUM_COMPILE_SECONDS_TO_STORE,
    )
    # No lower bound on the size of a stored program. Compile time alone decides
    # what is stored.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    if capped:
        gigabytes = CACHE_SIZE_LIMIT_BYTES / 1024**3
        how_much = f"holding up to {gigabytes:.0f} GB of compiled programs"
    else:
        how_much = "growing by a few MB per version of the model, with no cap"
    return (
        f"compile cache: on, in {directory}, {how_much}. Set "
        f"{OVERRIDE_VARIABLE} to another path, or to 'off', to change that. "
        "A miss only costs the compile it would have saved: the cache keys on "
        "the program's own text, so changed code always recompiles."
    )
