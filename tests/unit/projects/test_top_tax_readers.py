"""Unit tests for the top-tax readers of the distribution table."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from taxmodel.reporting import distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "projects" / "top_tax"))
import readers

YEAR = 2027
GROUPS = ("Quintile 5", "Negative income", "Top 1%")
SCENARIO = "stack_ref"

# Made-up tax dollars for the Top 1% group, by tax and world. The total row is
# their sum.
TOP_ONE_TAX = {
    "income_tax": {"baseline": 900e9, "static": 1_000e9, "conventional": 950e9},
    "payroll": {"baseline": 60e9, "static": 60e9, "conventional": 58e9},
    "estate": {"baseline": 20e9, "static": 30e9, "conventional": 25e9},
    "deemed_at_death": {"baseline": 0.0, "static": 40e9, "conventional": 35e9},
    "wealth": {"baseline": 0.0, "static": 0.0, "conventional": 0.0},
    "corporate": {"baseline": 200e9, "static": 250e9, "conventional": 245e9},
}
TOP_ONE_INCOME = {
    "baseline": {"cash": 5_000e9, "accrual": 8_000e9},
    "static": {"cash": 5_000e9, "accrual": 8_000e9},
    "conventional": {"cash": 4_900e9, "accrual": 7_900e9},
}
TOP_ONE_UNITS = 1.8e6
TOP_ONE_CUTOFF = 800_000.0

# The asset-spread version of the table carries every number at this multiple
# of the income-spread version, so a reader that mixes the two is caught.
ASSET_SPREAD_MULTIPLE = 3.0


def one_row(group, tax, spread=distribution.CORPORATE_SPREAD_INCOME):
    """One row of the table. Groups other than the Top 1% get the same numbers
    at a tenth of the size; the Negative income group has negative income."""
    scale = 1.0 if group == "Top 1%" else 0.1
    if spread == distribution.CORPORATE_SPREAD_ASSETS:
        scale *= ASSET_SPREAD_MULTIPLE
    taxes = distribution.TAXES if tax == distribution.TAX_TOTAL else (tax,)
    row = dict.fromkeys(distribution.table_columns(), 0.0)
    row.update(
        group_dimension="Income", group=group, tax=tax, corporate_spread=spread,
        year=YEAR,
    )
    row["n_tax_units"] = TOP_ONE_UNITS * scale
    row["income_cutoff"] = TOP_ONE_CUTOFF * scale
    for world in distribution.LEVEL_WORLDS:
        dollars = scale * sum(TOP_ONE_TAX[t][world] for t in taxes)
        row[f"{world}_tax_dollars"] = dollars
        for measure in distribution.INCOME_MEASURES:
            income = scale * TOP_ONE_INCOME[world][measure]
            if group == "Negative income":
                income = -income
            row[f"{world}_{measure}_income_dollars"] = income
            row[f"{world}_{measure}_etr"] = (
                dollars / income if income > 0 else float("nan"))
    return row


@pytest.fixture
def run_dir(tmp_path):
    rows = [
        one_row(group, tax, spread)
        for spread in distribution.CORPORATE_SPREADS
        for group in GROUPS
        for tax in distribution.TAX_ROWS
    ]
    # A second dimension, to check that the reader keeps to the one asked for.
    other = one_row("Top 1%", distribution.TAX_TOTAL)
    other["group_dimension"] = "Net worth"
    other["n_tax_units"] = 1.0
    rows.append(other)
    folder = tmp_path / "scenarios" / SCENARIO
    folder.mkdir(parents=True)
    pd.DataFrame(rows, columns=list(distribution.table_columns())).to_parquet(
        folder / readers.DISTRIBUTION_FILE, index=False
    )
    return tmp_path


def test_the_reader_returns_one_row_per_group_and_income_definition(run_dir):
    rows = readers.etr_rows(run_dir, SCENARIO, years=[YEAR])
    # The Negative income group is left out.
    expected = {
        (group, definition)
        for group in ("Quintile 5", "Top 1%")
        for definition in readers.ETR_INCOME_DEFS
    }
    assert {(r["group"], r["income_definition"]) for r in rows} == expected
    assert len(rows) == len(expected)


@pytest.mark.parametrize("leg", ["static", "conventional"])
def test_the_top_one_percent_levels_and_rates_match_the_table(run_dir, leg):
    rows = readers.etr_rows(run_dir, SCENARIO, reform_leg=leg, years=[YEAR])
    by_definition = {r["income_definition"]: r for r in rows if r["group"] == "Top 1%"}
    for definition, measure in readers.INCOME_MEASURE_OF.items():
        r = by_definition[definition]
        assert r["n_tax_units"] == TOP_ONE_UNITS
        for side, world in (("baseline", "baseline"), ("reform", leg)):
            tax = sum(TOP_ONE_TAX[t][world] for t in distribution.TAXES)
            income = TOP_ONE_INCOME[world][measure]
            assert r[f"tax_{side}"] == pytest.approx(tax / 1e9, abs=0.01)
            assert r[f"income_{side}"] == pytest.approx(income / 1e9, abs=0.01)
            assert r[f"etr_{side}"] == pytest.approx(tax / income)
            for comp, tax_row in readers.ETR_COMP_TAXES.items():
                assert r[f"etr_{comp}_{side}"] == pytest.approx(
                    TOP_ONE_TAX[tax_row][world] / income
                )
            components = sum(r[f"etr_{comp}_{side}"] for comp in readers.ETR_COMPS)
            assert components == pytest.approx(r[f"etr_{side}"])


def test_asking_for_the_asset_spread_returns_the_asset_rows(run_dir):
    rows = readers.etr_rows(
        run_dir, SCENARIO, years=[YEAR],
        corporate_spread=distribution.CORPORATE_SPREAD_ASSETS,
    )
    top = [r for r in rows if r["group"] == "Top 1%"]
    assert len(top) == len(readers.ETR_INCOME_DEFS)
    for r in top:
        assert r["n_tax_units"] == TOP_ONE_UNITS * ASSET_SPREAD_MULTIPLE
    cutoffs = readers.income_cutoffs(
        run_dir, SCENARIO, YEAR,
        corporate_spread=distribution.CORPORATE_SPREAD_ASSETS,
    )
    assert cutoffs["Top 1%"] == TOP_ONE_CUTOFF * ASSET_SPREAD_MULTIPLE


def test_the_income_cutoffs_come_from_the_total_rows(run_dir):
    cutoffs = readers.income_cutoffs(run_dir, SCENARIO, YEAR)
    assert cutoffs["Top 1%"] == TOP_ONE_CUTOFF
    assert set(cutoffs) == set(GROUPS)


def test_a_table_without_the_level_columns_is_refused(tmp_path):
    folder = tmp_path / "scenarios" / SCENARIO
    folder.mkdir(parents=True)
    pd.DataFrame(
        {"year": [YEAR], "group_dimension": ["Income"], "group": ["Top 1%"],
         "income_cutoff": [1.0]}
    ).to_parquet(folder / readers.DISTRIBUTION_FILE, index=False)
    with pytest.raises(ValueError, match="no column"):
        readers.etr_rows(tmp_path, SCENARIO, years=[YEAR])
