"""The four tax-specific totals files: form 1040, payroll, estate, wealth.

Each file is IRS-style summary statistics for one tax: the income and
deduction concepts that go into it, the intermediate figures, and the
liability that comes out. One row per world per year, one column per
variable. They report a run's calendar-year liability; the receipts files
report fiscal-year collections.

The four files and the record universe each one counts:

* `form1040.csv`, the individual income tax, filers only. A filer is a
  household the model says files a return, including one that files only
  because a refundable credit is worth money to it. Since a reform can change
  who files, the flag is each world's own.
* `payroll.csv`, payroll tax, everyone. Non-filers earn wages and owe FICA.
* `estate.csv`, the estate tax, everyone.
* `wealth.csv`, the wealth tax, everyone.

Counts are weighted tax units and dollars are dollars, with no scaling to
millions or billions. This matches the rest of the model's reporting.

Inputs from the run
-------------------
These builders do no household arithmetic. A run reduces its households once
and hands over, for every world and every year, a weighted sum and a weighted
count of records with a nonzero value for each result column, plus the four
counts of units that need a frame column rather than a result column. That is
`WorldTotals` below, and `projection.tax_totals` maps each evaluated
(household frame, tax law) cell to one of them.

The record universe is a property of the column, so the run has to know which
universe each column belongs to before it reduces. `FILERS_ONLY_COLUMNS` and
`EVERY_RECORD_COLUMNS` below say which is which.

The estate tax cannot be a plain weighted sum of a liability column. Each
record contributes its chance of dying this year times a blend of two
liabilities (one where the decedent's spouse left an unused exclusion amount
to carry over, and one where there is none). The count of taxable estates
blends the two branches' taxable-or-not indicators, because a record taxable
in one branch and not the other is a fraction of a taxable estate. The run
computes `est_tax_exp` and `est_returns` per record, already blended, and
hands them over as columns that can be added up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from taxmodel.years import world_names

Array = np.ndarray

# One number per run year, in the run's year order.
YearSeries = Any


@dataclass(frozen=True)
class WorldTotals:
    """One world's weighted annual totals for one scenario.

    `sums` and `nonzero_counts` are keyed by result column; each value is one
    number per run year. A sum is the weighted total of that column over the
    column's own record universe; a nonzero count is the weighted number of
    records in that universe whose value is not zero.

    The four unit counts are separate because none of them is a result
    column: they are counted off the household frame and the model's own
    filing flag.

    * `n_tax_units` — every record, filing or not.
    * `n_returns` — records the model says file a return.
    * `n_returns_dep` — returns filed by someone claimed as a dependent on
      someone else's return.
    * `n_dep` — dependents claimed on filed returns.
    """

    sums: Mapping[str, YearSeries]
    nonzero_counts: Mapping[str, YearSeries]
    n_tax_units: YearSeries
    n_returns: YearSeries
    n_returns_dep: YearSeries
    n_dep: YearSeries


# ---------------------------------------------------------------------------
# Contents of each file
# ---------------------------------------------------------------------------

# The individual income tax variables. Each contributes two columns to
# `form1040.csv`: the weighted number of returns with a nonzero value, and the
# weighted dollar total.
FORM_1040_VARIABLES: tuple[str, ...] = (
    # Wages and labor income
    "wages",
    "wages1",
    "wages2",
    "tips",
    "ot",
    # Investment income
    "txbl_int",
    "exempt_int",
    "div_ord",
    "div_pref",
    "txbl_ira_dist",
    "txbl_pens_dist",
    "txbl_kg",
    "kg_pref",
    # Other income items
    "state_ref",
    "alimony",
    # Business income
    "sole_prop",
    "part_scorp",
    "part_scorp_loss",
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
    "excess_bus_loss",
    "net_rent",
    "net_estate",
    "sch_e",
    "farm",
    # Social Security, unemployment, and the income subtotal
    "gross_ss",
    "txbl_ss",
    "ui",
    "other_inc",
    "gross_inc",
    # Above-the-line deductions
    "auto_int_exp",
    "auto_int_ded",
    "sl_int_ded",
    "char_above_ded",
    "above_ded",
    # Adjusted gross income and the deduction taken
    "agi",
    "ded",
    "itemizing",
    "std_ded",
    "med_item_ded",
    "salt_item_ded",
    "mort_int_item_ded",
    "inv_int_item_ded",
    "int_item_ded",
    "char_item_ded",
    "casualty_item_ded",
    "misc_item_ded",
    "other_item_ded",
    "item_ded_ex_limits",
    "item_ded",
    "pe_ded",
    "qbi_ded",
    "tip_ded",
    "ot_ded",
    "senior_ded",
    # Taxable income and tax before credits
    "txbl_inc",
    "liab_ord",
    "liab_pref",
    "liab_1250",
    "liab_collect",
    "amt_gross_inc",
    "amt_txbl_inc",
    "liab_amt",
    "excess_ptc",
    "liab_bc",
    # Nonrefundable credits
    "ftc",
    "cdctc_nonref",
    "ed_nonref",
    "savers_nonref",
    "old_cred",
    "ctc_nonref",
    "nonref",
    # Refundable credits
    "ctc_ref",
    "ed_ref",
    "net_ptc",
    "eitc",
    "wage_subsidy1",
    "wage_subsidy2",
    "rebate",
    "cdctc_ref",
    "savers_ref",
    "ref",
    "ref_iit",
    "ref_other",
    "refund",
    # Final liability
    "liab_niit",
    "liab_surtax",
    "liab_iit",
    "liab_iit_net",
    # The payment-timing series the receipts files read
    "pmt_iit_nonwithheld",
    "pmt_iit_withheld",
    "pmt_refund_nonwithheld",
    "pmt_refund_withheld",
    # The corporate tax change from business-entity shifting, reported on
    # this file because the receipts step reads it here
    "corp_tax_change",
)

# `itemizing` is a 0/1 flag saying the return itemized. The count of returns
# with a nonzero flag is the number of itemizers. Summing the flag would repeat
# that count under a dollar heading, so only the count is published.
FORM_1040_COUNT_ONLY_VARIABLES: frozenset[str] = frozenset({"itemizing"})

# Payroll tax variables. Each contributes a count column and a dollar column
# to `payroll.csv`.
#
# `pmt_pr_nonwithheld` is zero for every record, because no payroll tax is paid
# at filing, and `pmt_pr_withheld` equals `liab_pr`. The receipts timing model
# looks these columns up by name.
PAYROLL_VARIABLES: tuple[str, ...] = (
    "gross_wages1",
    "gross_wages2",
    "gross_wages",
    "se1",
    "se2",
    "se",
    "liab_fica",
    "liab_seca",
    "liab_seca_er",
    "liab_oasdi",
    "liab_hi",
    "liab_add_med",
    "liab_pr_ee",
    "liab_pr_er",
    "liab_pr",
    "pmt_pr_nonwithheld",
    "pmt_pr_withheld",
)

# The estate tax's two published figures, computed per record and already
# blended over the mortality and unused-exclusion branches (see the module
# docstring). `est_tax_exp` is expected estate tax in dollars; `est_returns`
# is the expected number of taxable estates.
ESTATE_TAX_EXPECTED = "est_tax_exp"
ESTATE_RETURNS_EXPECTED = "est_returns"

# The wealth tax liability column. The file reports its weighted total as
# `wealth_tax` and the weighted number of records owing any wealth tax as
# `wealth_returns`.
WEALTH_LIABILITY = "liab_wealth"

# The flag marking a return the IRS could plausibly pre-fill: the household
# files, does not itemize, and has nothing on the return but wages and Social
# Security. The flag is only ever set on a filer.
SIMPLE_FILER = "simple_filer"

# The record universe each column is reduced over. The run needs this before
# it reduces.
FILERS_ONLY_COLUMNS: tuple[str, ...] = (*FORM_1040_VARIABLES, SIMPLE_FILER)
EVERY_RECORD_COLUMNS: tuple[str, ...] = (
    *PAYROLL_VARIABLES,
    ESTATE_TAX_EXPECTED,
    ESTATE_RETURNS_EXPECTED,
    WEALTH_LIABILITY,
)


def form_1040_columns() -> list[str]:
    """The columns of `form1040.csv`, in order."""
    columns = [
        "world",
        "year",
        "n_tax_units",
        "n_returns",
        "n_returns_dep",
        "n_dep",
        "n_simple_filers",
    ]
    for name in FORM_1040_VARIABLES:
        columns.append(f"n_{name}")
        if name not in FORM_1040_COUNT_ONLY_VARIABLES:
            columns.append(name)
    return columns


def payroll_columns() -> list[str]:
    """The columns of `payroll.csv`, in order."""
    columns = ["world", "year", "n_tax_units"]
    for name in PAYROLL_VARIABLES:
        columns.append(f"n_{name}")
        columns.append(name)
    return columns


# The columns of `estate.csv` and of `wealth.csv`, in order.
ESTATE_COLUMNS: tuple[str, ...] = (
    "world",
    "year",
    ESTATE_TAX_EXPECTED,
    ESTATE_RETURNS_EXPECTED,
)
WEALTH_COLUMNS: tuple[str, ...] = ("world", "year", "wealth_tax", "wealth_returns")


# ---------------------------------------------------------------------------
# Building the four tables
# ---------------------------------------------------------------------------


def _evaluated_worlds(
    projection: Any,
) -> list[tuple[str, WorldTotals]]:
    """The worlds this run totaled, named, in the fixed order the worlds run in.

    A world the run did not total has no row in the file.
    """
    named = []
    for world, cell in world_names.WORLD_CELLS.items():
        totals = projection.tax_totals.get(cell)
        if totals is not None:
            named.append((world, totals))
    return named


def _series(totals: Mapping[str, YearSeries], name: str, what: str) -> Array:
    """One column's per-year numbers, with a useful error when it is absent."""
    try:
        return np.asarray(totals[name], dtype=np.float64)
    except KeyError as error:
        raise KeyError(
            f"{what} needs the '{name}' column and the run did not total it"
        ) from error


def _rows(
    projection: Any,
    columns: Sequence[str],
    fill: Any,
) -> pd.DataFrame:
    """Assemble one file: `fill(world totals)` gives every column but world and year."""
    years = [int(year) for year in projection.years]
    rows: list[dict[str, Any]] = []
    for world, totals in _evaluated_worlds(projection):
        values = fill(totals)
        for position, year in enumerate(years):
            row: dict[str, Any] = {"world": world, "year": year}
            row.update(
                {name: float(series[position]) for name, series in values.items()}
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def form_1040_table(projection: Any) -> pd.DataFrame:
    """`form1040.csv`: individual income tax summary statistics, filers only.

    Every dollar figure and every count of returns here is over filers as the
    model determines them. The five unit counts at the front say how many
    units and returns the run had.
    """
    what = "form1040.csv"

    def fill(totals: WorldTotals) -> dict[str, Array]:
        values: dict[str, Array] = {
            "n_tax_units": np.asarray(totals.n_tax_units, dtype=np.float64),
            "n_returns": np.asarray(totals.n_returns, dtype=np.float64),
            "n_returns_dep": np.asarray(totals.n_returns_dep, dtype=np.float64),
            "n_dep": np.asarray(totals.n_dep, dtype=np.float64),
            "n_simple_filers": _series(totals.sums, SIMPLE_FILER, what),
        }
        for name in FORM_1040_VARIABLES:
            values[f"n_{name}"] = _series(totals.nonzero_counts, name, what)
            if name not in FORM_1040_COUNT_ONLY_VARIABLES:
                values[name] = _series(totals.sums, name, what)
        return values

    return _rows(projection, form_1040_columns(), fill)


def payroll_table(projection: Any) -> pd.DataFrame:
    """`payroll.csv`: payroll tax summary statistics over every record.

    No filing restriction: a household with no filing requirement still owes
    FICA on its wages.
    """
    what = "payroll.csv"

    def fill(totals: WorldTotals) -> dict[str, Array]:
        values: dict[str, Array] = {
            "n_tax_units": np.asarray(totals.n_tax_units, dtype=np.float64),
        }
        for name in PAYROLL_VARIABLES:
            values[f"n_{name}"] = _series(totals.nonzero_counts, name, what)
            values[name] = _series(totals.sums, name, what)
        return values

    return _rows(projection, payroll_columns(), fill)


def estate_table(projection: Any) -> pd.DataFrame:
    """`estate.csv`: expected estate tax and expected taxable estates.

    Both figures are expectations rather than realized amounts, because no
    record is known to die: each contributes its mortality probability times
    its liability, blended over the two unused-exclusion branches.
    """
    what = "estate.csv"

    def fill(totals: WorldTotals) -> dict[str, Array]:
        return {
            ESTATE_TAX_EXPECTED: _series(totals.sums, ESTATE_TAX_EXPECTED, what),
            ESTATE_RETURNS_EXPECTED: _series(
                totals.sums, ESTATE_RETURNS_EXPECTED, what
            ),
        }

    return _rows(projection, ESTATE_COLUMNS, fill)


def wealth_table(projection: Any) -> pd.DataFrame:
    """`wealth.csv`: wealth tax collected and the number of units owing it.

    A wealth tax falls on the living every year, so there is no mortality to
    apply: the total is the weighted sum of liability and the count is the
    weighted number of units owing anything. Both are zero under baseline law,
    which has no wealth tax.
    """
    what = "wealth.csv"

    def fill(totals: WorldTotals) -> dict[str, Array]:
        return {
            "wealth_tax": _series(totals.sums, WEALTH_LIABILITY, what),
            "wealth_returns": _series(totals.nonzero_counts, WEALTH_LIABILITY, what),
        }

    return _rows(projection, WEALTH_COLUMNS, fill)


def files_and_tables(projection: Any) -> list[tuple[str, pd.DataFrame]]:
    """The four totals files a scenario writes, each with its file name."""
    return [
        ("form1040.csv", form_1040_table(projection)),
        ("payroll.csv", payroll_table(projection)),
        ("estate.csv", estate_table(projection)),
        ("wealth.csv", wealth_table(projection)),
    ]


def every_totalled_column() -> tuple[str, ...]:
    """Every result column the four files need the run to total.

    A run reducing its households once reads this to know what to reduce.
    """
    return (*FILERS_ONLY_COLUMNS, *EVERY_RECORD_COLUMNS)
