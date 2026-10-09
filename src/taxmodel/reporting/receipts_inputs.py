"""The three published files the receipts calculation reads.

The federal revenue picture receipts assembles is this model's own figures
laid alongside levels the Congressional Budget Office projects, plus the
revenue effects of policies other models score. Those come from three
publications in the lab's shared model-data tree, each pinned to a version and
a vintage. This module opens them.

    Macro-Projections        CBO's baseline revenue levels and fiscal-year GDP
    Off-Model-Estimates      reform effects scored outside this model
    Cost-Recovery-Simulator  the depreciation and expensing score

All three are stated on a fiscal-year basis. The calendar-year to fiscal-year
conversion applies only to the model's own liability and is in `receipts.py`.
The one conversion made here is billions of dollars into dollars.

A run stops rather than publish a budget total with a hole in it. The pinned
Cost-Recovery-Simulator vintage carries rows through 2096 but leaves the score
itself blank from 2057 on. Every reader below checks the years it was asked
for and, when a year is not covered, stops the run with a message naming the
publication, the column and the years.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from taxmodel.config.interfaces import DEFAULT_CONFIG_ROOT, interface_root
from taxmodel.law.indexes import read_macro_spliced

Array = np.ndarray

# The three publications state dollars in billions. This model states dollars in
# dollars, so every figure read here is multiplied by this on the way in.
BILLIONS = 1e9

# What receipts takes from CBO's projections. Every one is a fiscal-year series
# in billions of nominal dollars except that `gdp_fy` is fiscal-year nominal
# gross domestic product, which is the denominator of the share-of-GDP measure.
#
# The same file also publishes `rev`, `rev_iit` and `rev_payroll`, and receipts
# does not read them: the individual income tax and payroll tax levels this
# model reports are its own, computed from the microdata, not CBO's.
MACRO_PROJECTIONS_COLUMNS = (
    "rev_corp",  # corporate income tax receipts
    "rev_estate",  # estate and gift tax receipts
    "rev_excise",  # excise tax receipts
    "rev_customs",  # customs duties
    "rev_misc",  # miscellaneous receipts, chiefly Federal Reserve remittances
    "gdp_fy",  # fiscal-year nominal gross domestic product
)

# What receipts takes from the hand-assembled off-model scores. All are
# fiscal-year changes from baseline law, in billions of dollars.
#
# The file's `estate` column is not read. This model computes the estate tax on
# its own microdata, and that figure supersedes any off-model score of the same
# tax; reading both would count an estate reform twice.
OFF_MODEL_COLUMNS = (
    "individual",  # individual income tax
    "payroll",  # payroll tax
    "corporate",  # corporate tax, including the modeled behavioral response
    "corporate_static",  # corporate tax, the purely mechanical change
)

# The cost-recovery simulator publishes the change in receipts caused by the
# scenario's depreciation and expensing policy, fiscal-year, in billions.
COST_RECOVERY_COLUMNS = ("delta",)

# The Cost-Recovery-Simulator keeps its published figures in a subfolder, unlike
# the other three publications, which put theirs at the top of the scenario
# folder.
COST_RECOVERY_SUBFOLDER = "deltas"


class ReceiptsInputError(Exception):
    """An upstream publication cannot serve the years or columns receipts needs."""


def _year_indexed(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """One publication's table, indexed by year.

    Where a year appears twice the first row is kept, so for a year CBO has
    both completed and projected, the completed figure is used.
    """
    if "year" not in frame.columns:
        raise ReceiptsInputError(f"{source} has no 'year' column")
    frame = frame.drop_duplicates(subset="year", keep="first").copy()
    frame["year"] = frame["year"].astype(int)
    return frame.set_index("year")


def _columns_for_the_run_years(
    frame: pd.DataFrame,
    source: str,
    columns: Sequence[str],
    years: Sequence[int],
) -> dict[str, Array]:
    """The named columns for exactly the run's years, in the run's year order.

    A column the publication does not have, a year it has no row for, and a
    year whose value is blank are all reasons to stop the run. All of them are
    collected before raising so that a reader sees everything wrong with the
    vintage at once.
    """
    table = _year_indexed(frame, source)
    complaints: list[str] = []
    series: dict[str, Array] = {}

    for column in columns:
        if column not in table.columns:
            complaints.append(f"  {column}: the publication does not have this column")
            continue
        values = np.full(len(years), np.nan)
        no_row: list[int] = []
        blank: list[int] = []
        for position, year in enumerate(years):
            if year not in table.index:
                no_row.append(year)
                continue
            values[position] = pd.to_numeric(table.at[year, column], errors="coerce")
            if not np.isfinite(values[position]):
                blank.append(year)
        if no_row:
            complaints.append(f"  {column}: no row for {_year_list(no_row)}")
        if blank:
            complaints.append(f"  {column}: the value is blank for {_year_list(blank)}")
        series[column] = values

    if complaints:
        raise ReceiptsInputError(
            f"{source} does not cover the years this run simulates "
            f"({_year_list(list(years))}):\n" + "\n".join(complaints) + "\n"
            "Receipts will not publish a budget total with a hole in it. "
            "Either shorten the run's years or pin a vintage that "
            "covers them."
        )
    return series


def _year_list(years: Sequence[int]) -> str:
    """A run of consecutive years written as a range, anything else listed."""
    years = [int(year) for year in years]
    if len(years) > 2 and years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}-{years[-1]}"
    return ", ".join(str(year) for year in years)


def _read_billions(
    path: Path, columns: Sequence[str], years: Sequence[int], what: str
) -> dict[str, Array]:
    """One published fiscal-year table, read in billions and given in dollars.

    ``what`` names the publication in the message raised when the file is not
    where the pinned vintage says it is.
    """
    if not path.is_file():
        raise ReceiptsInputError(f"there is no {what} file at {path}")
    values = _columns_for_the_run_years(pd.read_csv(path), str(path), columns, years)
    return {name: series * BILLIONS for name, series in values.items()}


def read_macro_projections(root: str | Path, years: Sequence[int]) -> dict[str, Array]:
    """CBO's baseline revenue levels and fiscal-year GDP, in dollars.

    History and projection are two files that do not overlap, read as one table
    the same way the law's index tables read them.
    """
    frame = read_macro_spliced(root)
    source = f"Macro-Projections at {root} (historical.csv and projections.csv)"
    values = _columns_for_the_run_years(
        frame, source, MACRO_PROJECTIONS_COLUMNS, years
    )
    return {name: series * BILLIONS for name, series in values.items()}


def read_off_model_estimates(
    root: str | Path, years: Sequence[int]
) -> dict[str, Array]:
    """The reform effects scored outside this model, in dollars.

    A vintage without the `corporate_static` column stops the run. That column
    is the purely mechanical corporate change, which the model's mechanical
    worlds report. The `corporate` column includes the behavioral corporate
    response, which does not belong in a world without behavior.
    """
    path = Path(root) / "revenues.csv"
    # Read the header alone, so that an old vintage gets a message saying why
    # the column matters. A file that is not there at all raises its own
    # message in the read that follows.
    if path.is_file() and "corporate_static" not in pd.read_csv(path, nrows=0).columns:
        raise ReceiptsInputError(
            f"{path} has no 'corporate_static' column, so it cannot say what the "
            f"purely mechanical corporate change is. That column arrived with "
            f"version 5 of this interface; pin a version 5 vintage rather than "
            f"letting the mechanical worlds report the behavioral corporate "
            f"response."
        )
    return _read_billions(path, OFF_MODEL_COLUMNS, years, "off-model estimates")


def read_cost_recovery(root: str | Path, years: Sequence[int]) -> dict[str, Array]:
    """The depreciation and expensing score, in dollars."""
    path = Path(root) / COST_RECOVERY_SUBFOLDER / "revenues.csv"
    return _read_billions(path, COST_RECOVERY_COLUMNS, years, "cost-recovery")


@dataclass(frozen=True)
class ReceiptsInputs:
    """Everything receipts reads from outside this model, for one run's years.

    Every series has one entry per year, in the order `years` gives them, and
    every dollar figure is in dollars on a fiscal-year basis.
    """

    years: tuple[int, ...]
    cbo_levels: dict[str, Array]
    off_model: dict[str, Array]
    cost_recovery: dict[str, Array]


def load_receipts_inputs(
    years: Sequence[int],
    economy: str | None = None,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
) -> ReceiptsInputs:
    """Open all three publications at the vintages this scenario pins.

    `economy` names the economy alternative the scenario ran, which holds the
    scenario's interface pins. Choosing a different off-model score for a
    reform means naming an economy alternative that pins a different
    Off-Model-Estimates scenario.
    """
    years = tuple(int(year) for year in years)

    def root(interface: str) -> Path:
        return interface_root(interface, economy=economy, config_root=config_root)

    return ReceiptsInputs(
        years=years,
        cbo_levels=read_macro_projections(root("Macro-Projections"), years),
        off_model=read_off_model_estimates(root("Off-Model-Estimates"), years),
        cost_recovery=read_cost_recovery(root("Cost-Recovery-Simulator"), years),
    )
