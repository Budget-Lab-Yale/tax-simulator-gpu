"""Small hand-built populations for the frame tests.

Nothing here comes from Tax-Data. Each fixture is a handful of records
written out in full so the expected answer can be worked out by hand and
stated in the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxmodel.frame import steps_jax

# Every balance-sheet column the estate and wealth steps read. The tests set
# one or two of them per record and leave the rest at zero.
BALANCE_SHEET = list(steps_jax.ESTATE_ASSET_COLS) + list(steps_jax.ESTATE_DEBT_COLS)

# The cost basis of each asset class the capital-gains model tracks. Tax-Data
# publishes these next to the market values in BALANCE_SHEET.
CAPITAL_GAINS_BASIS = list(steps_jax.CAPITAL_GAINS_BASIS_COLS)

# The income columns the SALT workaround reads.
SALT_INCOME_COLS = [
    "wages_base1",
    "wages_base2",
    "tips1",
    "tips2",
    "ot1",
    "ot2",
    "trad_contr_er1",
    "trad_contr_er2",
    "txbl_int",
    "exempt_int",
    "div_ord",
    "div_pref",
    "state_ref",
    "txbl_ira_dist",
    "gross_pens_dist",
    "kg_st",
    "kg_lt",
    "other_gains",
    "alimony",
    "sole_prop1",
    "sole_prop2",
    "farm1",
    "farm2",
    "gross_ss",
    "ui",
    "other_inc",
    "part_active",
    "part_passive",
    "part_active_loss",
    "part_passive_loss",
    "part_179",
    "scorp_active",
    "scorp_passive",
    "scorp_active_loss",
    "scorp_passive_loss",
    "scorp_179",
]


def make_frame(records: list[dict]) -> pd.DataFrame:
    """A frame with every column the frame steps read, zero unless stated."""
    defaults: dict[str, float] = {name: 0.0 for name in SALT_INCOME_COLS}
    defaults.update({name: 0.0 for name in BALANCE_SHEET})
    defaults.update({name: 0.0 for name in CAPITAL_GAINS_BASIS})
    defaults.update(
        {
            "weight": 1.0,
            "filing_status": 1,
            "age1": 40,
            "age2": np.nan,
            # Tax-Data leaves the divorce year empty for anyone with no
            # divorce on record, which is nearly everyone.
            "male1": 0,
            "divorce_year": np.nan,
            "q_death1": 0.0,
            "q_death2": 0.0,
            "salt_inc_sales": 0.0,
            "sstb_part": np.nan,
            "sstb_scorp": np.nan,
            "kg_lt_years_held": np.nan,
            "kg_lt_basis": 0.0,
            "r.salt_workaround": 0.5,
            "item.salt_limit": 10_000.0,
            "amt.exempt": 100_000.0,
            "item.salt_workaround_allowed_part": 1.0,
            "item.salt_workaround_allowed_sstb": 0.0,
            "item.salt_workaround_allowed_non_sstb": 0.0,
        }
    )
    rows = []
    for record in records:
        row = dict(defaults)
        row.update(record)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def index_series() -> pd.DataFrame:
    """A price series that grows 10 percent a year from 2020 through 2030.

    Both published series carry the same growth, so the 2017 splice between
    CPI-U and chained CPI-U does not muddy a test that is about the ratio.
    """
    years = list(range(2010, 2031))
    rows = []
    for series in ("cpi", "chained_cpi"):
        for i, year in enumerate(years):
            rows.append(
                {"series": series, "year": year, "growth": np.nan if i == 0 else 0.10}
            )
    return pd.DataFrame(rows)


@pytest.fixture
def law_table() -> pd.DataFrame:
    """A two-year law table with one row per filing status."""
    rows = []
    for year in (2026, 2027):
        for filing_status in (1, 2, 3, 4):
            rows.append(
                {
                    "year": year,
                    "filing_status": filing_status,
                    "filing.repeal_hoh": 0,
                    "item.salt_limit": 10_000.0 * (2 if filing_status == 2 else 1),
                    "amt.exempt": 100_000.0,
                }
            )
    return pd.DataFrame(rows)
