"""Federal receipts by fiscal year.

Everything upstream of this module is calendar-year tax liability on tax units.
The federal budget is stated in fiscal years and in money collected, so this
module converts calendar year to fiscal year and liability to collections, then
assembles the federal revenue picture by putting what this model computes
beside what CBO projects and what other models score.

Levels come from CBO and changes come from the model, with nothing scaled after
the fact to hit a target. Where the model simulates a tax in levels (the
individual income tax, the payroll tax, refundable credits, and a wealth tax if
a scenario has one) the model's own level is reported. Where it does not
(corporate, estate, excise, customs, miscellaneous) CBO's projection is the
reported level and the model contributes only the change a reform causes.

Calendar year to fiscal year
----------------------------
The federal fiscal year runs October through September, so fiscal year *t*
overlaps nine months of calendar year *t* and three months of calendar year
*t−1*. Taxes withheld from pay, or paid in quarterly estimated payments, arrive
evenly across the calendar year, so fiscal year *t* collects three quarters of
calendar year *t*'s withholding and one quarter of calendar year *t−1*'s. Taxes
settled up at filing are booked in the fiscal year carrying the same label as
the calendar year of liability. The estate tax is booked one year late, because
Form 706 is due nine months after death. A wealth tax is also booked one year
late, because it is paid at filing in the spring after the year its net worth
was measured.

The first simulated year takes the prior calendar year's withholding as zero.
The first year's income and payroll tax levels are therefore only three
quarters of a year of withholding and should not be read as a level. Changes
from baseline are unaffected, because the baseline world has the same partial
first year.

The receipts have two known defects.

- Income tax settled at filing is booked in the fiscal year of liability, even
  though the filing season falls in the next fiscal year. This is in
  `model_receipts_by_world`, where the nonwithheld term is added unshifted.
- The corporate change from taxpayers shifting income between the individual
  and corporate sectors counts filing households only, because it passes
  through the totals machinery that multiplies by the filing flag.
  The convention is set where `corp_tax_change` is totaled; this module names
  it in `MODEL_COLUMNS` and does not undo it.

Pieces added to every world
---------------------------
CBO's projected levels go into every world, since they describe the economy rather than a reform. So does the
depreciation and expensing figure from the Cost-Recovery-Simulator, which is
added to every world's corporate line, baseline included. Both legs of a
revenue estimate must carry the same figure; otherwise the whole cost-recovery
figure would appear in every revenue estimate as though the reform had caused
it.

What that figure represents is an open question: the folder the model reads is
named `baseline` but holds the Cost-Recovery-Simulator's reform run for the
enacted cost-recovery provisions of the 2025 reconciliation act, while the CBO
projection beside it already includes those provisions.

Pieces added to the reform worlds only
--------------------------------------
The off-model scores are changes a reform causes, so this module adds them to
the reform worlds and not to the baseline world. All five worlds of one
scenario share one economy leg and therefore one set of pins, and adding an
off-model reform score to the baseline world as well would cancel it out of
every revenue estimate.

The same cancellation would happen to a cost-recovery reform: a scenario
pinning a different Cost-Recovery-Simulator run would carry that run's figure
in all five worlds and the revenue estimate would net it to zero. Every
scenario configured today pins the same cost-recovery run. A scenario that
reforms depreciation needs the baseline world pointed at a baseline
cost-recovery pin, which is a change to how interfaces are resolved.

Receipts are published in dollars, the same unit as every other dollar this
model publishes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from taxmodel.reporting.receipts_inputs import ReceiptsInputs
from taxmodel.years import world_names
from taxmodel.years.world_names import BASELINE_WORLD

Array = np.ndarray

# The share of a calendar year's withholding that fiscal year t collects: the
# nine months of calendar year t that fall inside it, out of twelve.
WITHHELD_SHARE_FROM_THE_SAME_CALENDAR_YEAR = 0.75
# And the share it collects from the previous calendar year: the October to
# December quarter of calendar year t-1, which is the first quarter of fiscal
# year t.
WITHHELD_SHARE_FROM_THE_PRIOR_CALENDAR_YEAR = 0.25

# The calendar-year totals receipts needs from the model, one weighted sum per
# world per year, in dollars. Which households each is summed over is a
# property of the totals that arrive here: income tax quantities count filing
# households only, using the model's filing flag; payroll and wealth count
# everybody. The corporate shifting figure is also counted over filers only,
# which is the defect described in the module docstring.
MODEL_COLUMNS: tuple[str, ...] = (
    "pmt_iit_withheld",  # income tax withheld from pay or paid quarterly
    "pmt_iit_nonwithheld",  # income tax settled up at filing
    "pmt_refund_withheld",  # refundable credits paid out in advance during the year
    "pmt_refund_nonwithheld",  # refundable credits paid out at filing
    "liab_pr",  # payroll tax, all of it treated as withheld
    "corp_tax_change",  # corporate tax change from income shifting between sectors
    "est_tax_exp",  # expected estate tax of that year's decedents
    "liab_wealth",  # wealth tax assessed on that year's net worth
)

# The seven published revenue lines. The refundable credit line is an outlay,
# so it is subtracted from the budget total.
REVENUE_LINES: tuple[str, ...] = (
    "revenues_payroll_tax",
    "revenues_income_tax",
    "outlays_tax_credits",
    "revenues_corp_tax",
    "revenues_estate_tax",
    "revenues_wealth_tax",
    "revenues_other",
)

OUTLAY_LINE = "outlays_tax_credits"

# The budget total, published so that a reader does not have to know which of
# the seven lines is an outlay.
TOTAL_LINE = "total_budget_effect"


# --------------------------------------------------------------------------
# Calendar year to fiscal year
# --------------------------------------------------------------------------


def withheld_on_a_fiscal_year_basis(calendar_year_series: Array) -> Array:
    """Three quarters of this calendar year's withholding plus a quarter of last's.

    The first simulated year has no prior year, so its prior-year withholding is
    taken as zero, leaving a partial first year in the levels. See the module
    docstring.
    """
    values = np.asarray(calendar_year_series, dtype=np.float64)
    prior_year = np.concatenate(([0.0], values[:-1]))
    return (
        WITHHELD_SHARE_FROM_THE_SAME_CALENDAR_YEAR * values
        + WITHHELD_SHARE_FROM_THE_PRIOR_CALENDAR_YEAR * prior_year
    )


def collected_the_following_year(calendar_year_series: Array) -> Array:
    """A tax assessed on calendar year *t* and collected in fiscal year *t+1*.

    The estate tax and a wealth tax are both like this. The first simulated year
    collects nothing, because the year that would have been assessed was not
    simulated.
    """
    values = np.asarray(calendar_year_series, dtype=np.float64)
    return np.concatenate(([0.0], values[:-1]))


# --------------------------------------------------------------------------
# The model's contribution by world
# --------------------------------------------------------------------------


def missing_model_columns(projection: Any) -> list[str]:
    """The calendar-year totals receipts needs that the projection did not total.

    The run directory uses this to decide whether it can write a receipts file
    at all, since a line built from a column that was never totaled would report
    a zero as though it were a measurement.
    """
    absent: set[str] = set()
    for totals in projection.totals.values():
        absent |= {name for name in MODEL_COLUMNS if name not in totals}
    return sorted(absent)


def model_totals_by_world(projection: Any) -> dict[str, dict[str, Array]]:
    """One projection's calendar-year totals, keyed by the world's reported name."""
    return {
        world_names.world_name(*cell): {
            name: np.asarray(totals[name], dtype=np.float64) for name in MODEL_COLUMNS
        }
        for cell, totals in projection.totals.items()
    }


def off_model_corporate_column(world: str) -> str:
    """Which corporate score a world reads out of the off-model estimates file.

    Only the conventional world contains households and firms responding to the
    reform, so only it reads the corporate figure that includes the modeled
    corporate behavioral response. Every other world reads the purely mechanical
    corporate change.
    """
    return "corporate" if world == "conventional" else "corporate_static"


def model_receipts_by_world(
    model_totals: Mapping[str, Array],
    baseline_totals: Mapping[str, Array],
) -> dict[str, Array]:
    """The fiscal-year collections this model itself accounts for, for one world.

    `baseline_totals` is the baseline world's calendar-year totals, needed for
    the estate line: CBO's estate level already reflects baseline law, so the
    model adds the change a reform makes.
    """
    income_tax = (
        withheld_on_a_fiscal_year_basis(model_totals["pmt_iit_withheld"])
        # Income tax settled at filing is booked in the fiscal year of
        # liability, though filing season falls in the next one.
        + np.asarray(model_totals["pmt_iit_nonwithheld"], dtype=np.float64)
    )
    tax_credit_outlays = withheld_on_a_fiscal_year_basis(
        model_totals["pmt_refund_withheld"]
    ) + np.asarray(model_totals["pmt_refund_nonwithheld"], dtype=np.float64)
    return {
        "income_tax": income_tax,
        "payroll_tax": withheld_on_a_fiscal_year_basis(model_totals["liab_pr"]),
        "tax_credit_outlays": tax_credit_outlays,
        "corporate_shifting_change": withheld_on_a_fiscal_year_basis(
            model_totals["corp_tax_change"]
        ),
        "estate_change": collected_the_following_year(
            np.asarray(model_totals["est_tax_exp"], dtype=np.float64)
            - np.asarray(baseline_totals["est_tax_exp"], dtype=np.float64)
        ),
        "wealth_tax": collected_the_following_year(model_totals["liab_wealth"]),
    }


def revenue_lines_by_world(
    world: str,
    model_totals: Mapping[str, Array],
    baseline_totals: Mapping[str, Array],
    inputs: ReceiptsInputs,
    corporate_rate_change: Array | None = None,
) -> dict[str, Array]:
    """The seven published revenue lines, plus the budget total, for one world.

    `corporate_rate_change` is what a change in the corporate statutory rate
    raises or loses in this world, in dollars, one entry per run year. It is
    computed from an elasticity of the corporate tax base in
    `years/corporate/scorer.py`, which `scenario/prepare.prepare_scenario`
    calls before the run starts. It differs by world because only the
    conventional world lets the corporate tax base erode as the rate rises. A
    scenario that leaves the corporate rate alone passes nothing.
    """
    collected = model_receipts_by_world(model_totals, baseline_totals)
    n_years = len(inputs.years)
    if corporate_rate_change is None:
        corporate_rate_change = np.zeros(n_years)

    # A reform's off-model score belongs to the reform worlds. See the module
    # docstring.
    reform = 0.0 if world == BASELINE_WORLD else 1.0
    off_model = inputs.off_model
    corporate_off_model = off_model[off_model_corporate_column(world)]

    lines = {
        "revenues_payroll_tax": collected["payroll_tax"]
        + reform * off_model["payroll"],
        "revenues_income_tax": collected["income_tax"]
        + reform * off_model["individual"],
        OUTLAY_LINE: collected["tax_credit_outlays"],
        "revenues_corp_tax": (
            inputs.cbo_levels["rev_corp"]
            + collected["corporate_shifting_change"]
            # The depreciation score restates CBO's corporate level, so it goes
            # into every world including the baseline one. See the module
            # docstring.
            + inputs.cost_recovery["delta"]
            + np.asarray(corporate_rate_change, dtype=np.float64)
            + reform * corporate_off_model
        ),
        "revenues_estate_tax": inputs.cbo_levels["rev_estate"]
        + collected["estate_change"],
        "revenues_wealth_tax": collected["wealth_tax"],
        "revenues_other": (
            inputs.cbo_levels["rev_excise"]
            + inputs.cbo_levels["rev_customs"]
            + inputs.cbo_levels["rev_misc"]
        ),
    }
    lines[TOTAL_LINE] = budget_total(lines)
    return lines


def budget_total(lines: Mapping[str, Array]) -> Array:
    """The seven lines added up, with the refundable credit outlay subtracted."""
    total = np.zeros(len(next(iter(lines.values()))), dtype=np.float64)
    for name in REVENUE_LINES:
        values = np.asarray(lines[name], dtype=np.float64)
        total = total - values if name == OUTLAY_LINE else total + values
    return total


def receipts_by_world(
    projection: Any,
    inputs: ReceiptsInputs,
    corporate_rate_score_by_world: Mapping[str, Array] | None = None,
) -> dict[str, dict[str, Array]]:
    """Every world's revenue lines for one scenario, keyed by world then line.

    `corporate_rate_score_by_world` holds each world's corporate statutory rate
    score, computed before the run started; see `revenue_lines_by_world`. A
    world missing from it is scored at nothing.
    """
    model_totals = model_totals_by_world(projection)
    if BASELINE_WORLD not in model_totals:
        raise KeyError(
            f"the projection has no '{BASELINE_WORLD}' world, so there is nothing "
            f"to measure the estate change or the revenue estimates against"
        )
    baseline_totals = model_totals[BASELINE_WORLD]
    scores = corporate_rate_score_by_world or {}
    return {
        world: revenue_lines_by_world(
            world, totals, baseline_totals, inputs, scores.get(world)
        )
        for world, totals in model_totals.items()
    }


# --------------------------------------------------------------------------
# The two published files
# --------------------------------------------------------------------------


def receipts_table(
    lines_by_world: Mapping[str, Mapping[str, Array]], inputs: ReceiptsInputs
) -> pd.DataFrame:
    """Federal receipts by fiscal year, world and revenue line, in dollars.

    One row per year, world and line, with the budget total as a line of its
    own. The first year is a partial year for the withheld lines; see the module
    docstring. `lines_by_world` is what `receipts_by_world` returned, which the
    caller works out once and hands to both published files.
    """
    years = list(inputs.years)
    rows = [
        {
            "year": int(year),
            "world": world,
            "line": line,
            "dollars": float(value),
        }
        for world, lines in lines_by_world.items()
        for line, values in lines.items()
        for year, value in zip(years, values, strict=True)
    ]
    return pd.DataFrame(rows, columns=["year", "world", "line", "dollars"])


def revenue_estimates_table(
    lines_by_world: Mapping[str, Mapping[str, Array]], inputs: ReceiptsInputs
) -> pd.DataFrame:
    """Each world's annual receipts less the baseline world's, line by line.

    Two measures:

    - `dollars`, the plain difference;
    - `share_of_gdp`, the difference divided by fiscal-year GDP.

    `lines_by_world` is what `receipts_by_world` returned, which the caller
    works out once and hands to both published files.
    """
    years = list(inputs.years)
    baseline_lines = lines_by_world[BASELINE_WORLD]
    gdp = inputs.cbo_levels["gdp_fy"]

    rows: list[dict[str, Any]] = []
    for world, lines in lines_by_world.items():
        if world == BASELINE_WORLD:
            continue
        for line, values in lines.items():
            dollars_by_year = np.asarray(values, dtype=np.float64) - np.asarray(
                baseline_lines[line], dtype=np.float64
            )
            rows.extend(
                {
                    "year": int(year),
                    "world": world,
                    "line": line,
                    "dollars": float(dollars),
                    "share_of_gdp": float(share),
                }
                for year, dollars, share in zip(
                    years, dollars_by_year, dollars_by_year / gdp, strict=True
                )
            )
    return pd.DataFrame(
        rows, columns=["year", "world", "line", "dollars", "share_of_gdp"]
    )
