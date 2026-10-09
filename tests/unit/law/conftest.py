"""Fixtures for the law compiler tests.

Most tests build a small invented law tree in a temp directory and check one
rule at a time. A few read this repository's own law tree under
config/scenarios.

The golden CSVs under tests/golden/fixtures/law are in the repo and carry no
microdata: they are law parameters and published price and wage series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from taxmodel.law import IndexTable

# The years the tests of the real law tree cover.
LAW_TREE_YEARS = tuple(range(2025, 2037))


@dataclass
class LawTree:
    """A small invented law tree, shaped like the real one."""

    root: Path

    def write(self, layer: str, parameter: str, body: dict) -> Path:
        folder = (
            self.root / "tax_law" / "default"
            if layer == "default"
            else self.root / "tax_law" / "alternatives" / layer
        )
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{parameter}.yaml"
        path.write_text(yaml.safe_dump(body, sort_keys=False))
        return path


@pytest.fixture
def law_tree(tmp_path: Path) -> LawTree:
    return LawTree(root=tmp_path / "config" / "scenarios")


@pytest.fixture
def flat_index() -> IndexTable:
    """Prices growing 2% a year and wages 4%, from 1980 to 2100.

    Round numbers on purpose: an indexed value under this table can be worked
    out by hand, so a test that fails says which of base value, ratio and
    rounding is wrong.
    """
    rows = []
    for year in range(1980, 2101):
        rows.append(("cpi", year, 0.02))
        rows.append(("chained_cpi", year, 0.02))
        rows.append(("awi", year, 0.04))
    return IndexTable.from_rows(rows)
