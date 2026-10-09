"""Things every test package in the suite needs: where the fixtures live, and
the skip for the parts of the model that read a mounted publication.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from taxmodel.config.interfaces import interface_root
from taxmodel.reporting.revenue import CellDifference, channel_name
from taxmodel.scenario.provenance import read_compiled_npz

REPO_ROOT = Path(__file__).resolve().parent.parent

# The golden law fixtures checked into the repository: law parameters and
# published price and wage series, no microdata.
GOLDEN_LAW_FIXTURES = REPO_ROOT / "tests" / "golden" / "fixtures" / "law"

# The golden index table. Tests that compile law pin it explicitly: they are
# about the engine, not about deriving growth rates, and the default source --
# the pinned Macro-Projections publication -- only exists on the cluster.
# tests/unit/law/test_index_table.py holds the derivation to this same table.
INDEX_TABLE = GOLDEN_LAW_FIXTURES / "indexes.csv"

# The scenario configuration tree this repository ships.
CONFIG_ROOT = REPO_ROOT / "config" / "scenarios"

# The frozen compiled baseline law, which several test packages bind against.
COMPILED_BASELINE = GOLDEN_LAW_FIXTURES / "compiled_baseline.npz"


@pytest.fixture(scope="session")
def compiled_baseline() -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    """The frozen baseline law: its value arrays and its year axis."""
    return read_compiled_npz(COMPILED_BASELINE)


@pytest.fixture(scope="module")
def macro_root() -> Path:
    """The pinned Macro-Projections publication, or skip if it is not mounted.

    The publication supplies the price level and the Treasury rate. It lives on
    the cluster, so tests that read it cannot run on a laptop.
    """
    root = Path(interface_root("Macro-Projections"))
    if not root.is_dir():
        pytest.skip(f"Macro-Projections publication not mounted at {root}")
    return root


def stacked_table_row(
    table, provision: str, channel: CellDifference
) -> np.ndarray:
    """One row of a stacked table as a year series, by provision and channel.

    The published table is a header and rows of cells. Only tests need to pick
    a single row out of it by name, so the lookup lives here rather than on
    the table the run writes.
    """
    wanted = (channel_name(channel), provision)
    for row in table.rows:
        if (row[0], row[1]) == wanted:
            return np.asarray(row[2:], dtype=np.float64)
    raise KeyError(
        f"the table has no row for provision {provision!r} and channel "
        f"{channel_name(channel)}"
    )
