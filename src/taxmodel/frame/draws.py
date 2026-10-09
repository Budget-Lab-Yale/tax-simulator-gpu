"""Keyed random draws: each record's random numbers, derived from the run seed,
the column's name and the record's id.

A record's draw for one column depends only on those three inputs
(ARCHITECTURE D8), so adding or removing a random column leaves every other
column's draws unchanged, and a record's draw does not depend on its position
in the file.

The bridge (`frame/bridge.py`) calls this module once per Tax-Data vintage
and stores the draws as Parquet columns, which a production run reads.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# The random columns the model reads. R's unused `r.behavior3` column is not
# created.
DRAW_COLUMNS: tuple[str, ...] = (
    "r.bus_loss",  # excess business loss limitation eligibility
    "r.cdctc_takeup",  # child and dependent care credit takeup
    "r.salt_workaround",  # participation in the state pass-through entity tax
    "r.new_car",  # new-versus-used imputation for the auto loan deduction
    "r.behavior1",  # spare draw for behavioral modules
    "r.behavior2",  # spare draw for behavioral modules
    "r.eitc_precert",  # EITC pre-certification check
)

# The default run seed.
DEFAULT_SEED = 76


# ---------------------------------------------------------------------------
# Keyed random draws
# ---------------------------------------------------------------------------

# The standard mixing multipliers of the MurmurHash3 64-bit finalizer, which
# turns three integers into one well-spread 64-bit number in a few vectorized
# operations.
_MIX_A = np.uint64(0xFF51AFD7ED558CCD)
_MIX_B = np.uint64(0xC4CEB9FE1A85EC53)
_SHIFT = np.uint64(33)

# A double has 53 bits of mantissa, so a uniform draw uses the top 53 bits of
# the hash and scales by 2**-53.
_MANTISSA_SHIFT = np.uint64(11)
_TWO_POW_53 = np.float64(1.0 / 9007199254740992.0)


def _mix64(x: np.ndarray) -> np.ndarray:
    """Scramble 64-bit integers so that neighboring inputs land far apart."""
    x = np.asarray(x, dtype=np.uint64).copy()
    x ^= x >> _SHIFT
    x *= _MIX_A
    x ^= x >> _SHIFT
    x *= _MIX_B
    x ^= x >> _SHIFT
    return x


def _name_key(seed: int, column: str) -> np.uint64:
    """One 64-bit number standing for this run's seed and this column's name.

    The column name enters through its bytes, so two columns never share a
    key and renaming a column changes its draws. Arithmetic is done in Python
    integers and truncated to 64 bits at each step, as the hash specifies.
    """
    mask = (1 << 64) - 1
    # The odd constant is the 64-bit golden-ratio increment the same family
    # of hashes uses to step from one input to the next.
    step = 0x9E3779B97F4A7C15
    key = (int(seed) * step) & mask
    for byte in column.encode("utf-8"):
        key = int(_mix64(np.uint64(key ^ byte))[()])
        key = (key + step) & mask
    return np.uint64(key)


def draw_uniform(ids: np.ndarray, seed: int, column: str) -> np.ndarray:
    """Uniform draws on [0, 1), one per id, for one named column.

    The same (seed, column, id) always gives the same number, on any machine
    and in any order, and no other column's numbers depend on this one.
    """
    key = _name_key(seed, column)
    hashed = _mix64(np.asarray(ids, dtype=np.int64).astype(np.uint64) + key)
    hashed = _mix64(hashed ^ key)
    return (hashed >> _MANTISSA_SHIFT).astype(np.float64) * _TWO_POW_53


def keyed_draws(
    ids: Sequence[int] | np.ndarray,
    seed: int = DEFAULT_SEED,
    columns: Sequence[str] = DRAW_COLUMNS,
) -> pd.DataFrame:
    """Every random column the model reads, uniform on [0, 1), derived from ids alone."""
    ids = np.asarray(ids, dtype=np.int64)
    out = pd.DataFrame({"id": ids})
    for column in columns:
        out[column] = draw_uniform(ids, seed, column)
    return out
