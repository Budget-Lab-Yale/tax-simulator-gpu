#!/usr/bin/env python3
"""
Income above the top groups' own thresholds, before and after federal tax.

Report table 1. For the top 1, 0.1 and 0.01 percent of tax units, on two
income measures, in one year under current law, the table gives the group's
income threshold, the number of tax units in the group, the group's total
income, the income above the threshold, the federal tax on that
above-threshold income, what is left after that tax, and each dollar amount as
a share of calendar-year GDP.

The two measures are adjusted gross income, and adjusted gross income with
realized gains replaced by accrued gains (defined at `accrual_income` below).
Each group is ranked on its own measure, and the thresholds are cut over tax
units that are not dependents.

Tax on the above-threshold income
---------------------------------
Federal tax here is individual income tax after refundable credits plus
payroll tax. The tax on a record's above-threshold income is measured by
removing that income. Every positive income flow and every deduction on the
record (and, on the accrual measure, every positive accrued gain) is
multiplied by one factor between zero and one, chosen so that the record's
measured income equals the threshold. The record is then taxed again under
current law, and the fall in its tax is the tax on the removed income.

Scaling every flow by the same factor keeps the record's mix of income, so the
income left below the threshold is taxed as it would be for that taxpayer at
the threshold. Deductions scale with income; the standard deduction is set by
law and does not scale. Losses do not scale, since shrinking a loss raises
income.

The factor is found for each record by bisection, with every record in the
same calculator call. A record whose income is still above the threshold with
every positive flow removed cannot be brought down to it. Such a record is
given a factor of zero, and the share of above-threshold income held by such
records is reported as `unreachable_share`.

This is not part of the model run and writes nothing into a run directory. It
builds the baseline frame for the year the way the model's baseline sweep does
(`years/one_year.baseline_year_step`) and calls the model's tax calculator on
the device, so it runs as a Slurm GPU job:

    sbatch jobs/above_threshold_income.sbatch <checkout> <output csv> [year]

or, on a GPU node,

    python projects/top_tax/above_threshold_income.py --output PATH \\
        [--year 2027] [--population BRIDGE_ROOT]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readers

from taxmodel.frame.population import prepare_population
from taxmodel.run.plan import population_root_for
from taxmodel.scenario import prepare
from taxmodel.scenario.provenance import DEFAULT_CONFIG_ROOT
from taxmodel.years import one_year

# The checkout this script belongs to. The model's configuration paths are
# relative to it.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The year the report's table 1 is drawn for.
TABLE_YEAR = 2027

# The top groups, as shares of non-dependent tax units.
TOP_SHARES = (0.01, 0.001, 0.0001)

# The two income measures, in the order the table lists them.
MEASURES = ("agi", "accrual")

# Bisection settings for the scale factor: at most 26 halvings, stopping early
# once every record's measured income is within one part in ten million of the
# threshold.
BISECTION_ITERATIONS = 26
BISECTION_TOLERANCE = 1e-7

# A record counts as unable to reach the threshold when, with every positive
# flow removed, its income still exceeds the threshold by more than one part in
# a billion. The margin keeps a record that lands exactly on the threshold at a
# factor of zero from being counted.
UNREACHABLE_MARGIN = 1e-9

DOLLARS_PER_BILLION = 1e9
UNITS_PER_MILLION = 1e6
PERCENT = 100.0

# Calendar-year nominal GDP in billions of dollars, read from the baseline
# Macro-Projections file the model is pinned to (readers.MACRO, vintage
# 2026022522). Calendar year, because every income measure here is a
# calendar-year amount.
GDP_COLUMN = "gdp"


# ---------------------------------------------------------------------------
# The columns that scale
# ---------------------------------------------------------------------------
#
# Tax-Data's overlapping wage totals are split into non-overlapping pieces as
# the population is read (`frame/prepare.OVERLAPPING_COLUMNS`), so:
#
#   wages, wages1, wages2   -> wages_base1, wages_base2 (each earner's pay
#                              other than tips and overtime), which with tips1,
#                              tips2, ot1 and ot2 make up the wage line
#   ot, tips                -> ot1, ot2, tips1, tips2
#   sole_prop               -> sole_prop1, sole_prop2
#   farm                    -> farm1, farm2
#   part_se                 -> part_se1, part_se2
#
# Tax-Data's `nols` (net operating losses carried forward) is not in the
# population (`frame/prepare.NEVER_READ_COLUMNS`) and no calculator reads it,
# so it is left out.

# Every income flow, ordinary and preferentially taxed. Sub-components scale
# with their totals, and long-term gain basis with the gain.
INCOME_COLUMNS = (
    "wages_base1", "wages_base2", "ot1", "ot2", "tips1", "tips2",
    "txbl_int", "exempt_int", "div_ord", "div_pref", "state_ref", "alimony",
    "sole_prop1", "sole_prop2",
    "kg_st", "kg_lt", "kg_lt_basis", "kg_1250", "kg_collect", "other_gains",
    "txbl_ira_dist", "gross_pens_dist", "txbl_pens_dist",
    "part_active", "part_active_loss", "part_passive", "part_passive_loss",
    "part_179", "part_se1", "part_se2",
    "scorp_active", "scorp_active_loss", "scorp_passive", "scorp_passive_loss",
    "scorp_179", "rent", "rent_loss", "estate", "estate_loss",
    "farm1", "farm2", "ui", "gross_ss", "other_inc",
)

# Deductions and the expenses behind them. Mortgage balances scale with the
# interest on them.
DEDUCTION_COLUMNS = (
    "ed_exp", "hsa_contr", "trad_contr_ira", "sl_int_ded", "keogh_contr",
    "se_health", "early_penalty", "alimony_exp", "tuition_ded", "dpad",
    "char_cash", "char_noncash", "other_above_ded", "auto_int_exp",
    "med_exp", "salt_inc_sales", "salt_prop", "salt_pers",
    "first_mort_bal", "second_mort_bal", "first_mort_int", "second_mort_int",
    "inv_int_exp", "casualty_exp", "job_exp", "tax_prep_exp", "other_misc_exp",
    "other_item_exp",
)

# Accrued gains, which are untaxed in the year they accrue and enter only the
# accrual measure.
ACCRUAL_COLUMNS = (
    "accruals.equities", "accruals.pass_throughs", "accruals.primary_home",
    "accruals.other_home", "accruals.re_fund", "accruals.dc", "accruals.trusts",
)

# Retirement balances, for the defined-contribution share of pensions.
DEFINED_CONTRIBUTION_BALANCE = "value.dc"
DEFINED_BENEFIT_BALANCE = "value.db"

# Calculator results read here.
AGI = "agi"
TAXABLE_GAINS = "txbl_kg"
INCOME_TAX = "liab_iit_net"
PAYROLL_TAX = "liab_pr"


def scaled_columns(measure: str) -> tuple[str, ...]:
    """The columns scaled down for one measure.

    Accrued gains are part of the accrual measure and scale with it. They are
    not in adjusted gross income and are left alone on that measure.
    """
    if measure == "agi":
        return INCOME_COLUMNS + DEDUCTION_COLUMNS
    return INCOME_COLUMNS + DEDUCTION_COLUMNS + ACCRUAL_COLUMNS


def check_frame_columns(frame_columns: Sequence[str]) -> None:
    """Stop if the frame lacks any column this script scales or reads."""
    needed = (
        INCOME_COLUMNS + DEDUCTION_COLUMNS + ACCRUAL_COLUMNS
        + (DEFINED_CONTRIBUTION_BALANCE, DEFINED_BENEFIT_BALANCE, "weight",
           "dep_status")
    )
    missing = sorted(set(needed) - set(frame_columns))
    if missing:
        raise KeyError(
            "the household frame is missing columns the above-threshold "
            "table scales or reads: " + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Income measures and tax
# ---------------------------------------------------------------------------


def _zero_if_missing(value):
    return jnp.nan_to_num(jnp.asarray(value, dtype=jnp.float64), nan=0.0)


def accrual_income(frame, taxed):
    """Adjusted gross income with realized gains replaced by accrued gains.

    The gains taken out are those adjusted gross income contains: taxable
    capital gains after the capital loss limit and any exclusion, plus capital
    gain distributions. The seven accrued-gain columns go in. Retirement
    income that would then be counted twice comes out: the taxable IRA
    distribution, and the defined-contribution share of gross pension
    distributions (the household's defined-contribution balance over its
    defined-contribution plus defined-benefit balances, zero with no
    balances). Defined-benefit pensions have no accrual column and stay in.
    """
    dc = _zero_if_missing(frame[DEFINED_CONTRIBUTION_BALANCE])
    balances = dc + _zero_if_missing(frame[DEFINED_BENEFIT_BALANCE])
    has_balances = balances > 0.0
    dc_share = jnp.where(
        has_balances, dc / jnp.where(has_balances, balances, 1.0), 0.0
    )
    accruals = sum(_zero_if_missing(frame[name]) for name in ACCRUAL_COLUMNS)
    return (
        _zero_if_missing(taxed[AGI])
        - _zero_if_missing(taxed[TAXABLE_GAINS])
        - _zero_if_missing(frame["other_gains"])
        + accruals
        - _zero_if_missing(frame["txbl_ira_dist"])
        - _zero_if_missing(frame["gross_pens_dist"]) * dc_share
    )


def measured_income(frame, taxed, measure: str):
    if measure == "agi":
        return _zero_if_missing(taxed[AGI])
    return accrual_income(frame, taxed)


def federal_tax(taxed):
    """Individual income tax after refundable credits, plus payroll tax."""
    return _zero_if_missing(taxed[INCOME_TAX]) + _zero_if_missing(taxed[PAYROLL_TAX])


def make_pricer(tax_calculator, measure: str):
    """A compiled function from (frame, law, factor) to (income, federal tax).

    Each record's positive amounts in the scaled columns are multiplied by its
    factor; zeros and negative amounts are left as they are.
    """
    columns = scaled_columns(measure)

    def price(frame, law, factor):
        cut = dict(frame)
        for name in columns:
            value = _zero_if_missing(frame[name])
            cut[name] = jnp.where(value > 0.0, value * factor, value)
        taxed = tax_calculator(cut, law)
        return measured_income(cut, taxed, measure), federal_tax(taxed)

    return jax.jit(price)


# ---------------------------------------------------------------------------
# Threshold and bisection
# ---------------------------------------------------------------------------


def threshold_for_share(income, weight, eligible, share: float) -> float:
    """The income of the last unit inside the top `share` of eligible units.

    Eligible units are sorted from highest income down. The threshold is the
    income of the last unit whose cumulative weight, counting all units above
    it, is still below `share` of the eligible total; it is the first unit's
    income if no unit is.
    """
    eligible = np.asarray(eligible)
    income = np.asarray(income)[eligible]
    weight = np.asarray(weight)[eligible]
    order = np.argsort(-income, kind="stable")
    cumulative = np.cumsum(weight[order])
    position = max(1, int(np.sum(cumulative < weight.sum() * share)))
    return float(income[order][position - 1])


def solver_gap(income, active, threshold: float) -> float:
    """The largest distance of an active record from the threshold, relative to it."""
    if not bool(jnp.any(active)):
        return 0.0
    distance = jnp.where(active, jnp.abs(income - threshold), 0.0)
    return float(jnp.max(distance) / threshold)


def bisect_scale_factors(
    income_at: Callable[[jnp.ndarray], jnp.ndarray],
    active,
    unreachable,
    threshold: float,
    iterations: int = BISECTION_ITERATIONS,
    tolerance: float = BISECTION_TOLERANCE,
):
    """Solve each active record's factor so that its income equals the threshold.

    `income_at` maps a vector of factors, one per record, to each record's
    income, and must rise with the factor. Active records search between zero
    and one. Unreachable records get zero and every other record keeps one.

    Returns the factors and the number of halvings taken.
    """
    active = jnp.asarray(active)
    unreachable = jnp.asarray(unreachable)
    lo = jnp.where(active | unreachable, 0.0, 1.0)
    hi = jnp.where(unreachable, 0.0, 1.0)
    halvings = 0
    for _ in range(iterations):
        halvings += 1
        mid = (lo + hi) / 2.0
        income = income_at(mid)
        too_high = active & (income > threshold)
        hi = jnp.where(too_high, mid, hi)
        lo = jnp.where(active & ~too_high, mid, lo)
        if solver_gap(income, active, threshold) < tolerance:
            break
    return (lo + hi) / 2.0, halvings


# ---------------------------------------------------------------------------
# One group on one measure
# ---------------------------------------------------------------------------


def one_group(pricer, frame, law, income, tax, weight, adult, share, gdp):
    """One row of the table: one top group on one measure.

    `income` and `tax` are each record's measured income and federal tax
    under current law with nothing removed.
    """
    n_records = income.shape[0]
    threshold = threshold_for_share(income, weight, adult, share)
    in_group = adult & (income >= threshold)
    above_threshold = (income > threshold) & (weight > 0.0)

    income_with_nothing, _ = pricer(frame, law, jnp.zeros(n_records))
    unreachable = above_threshold & (
        income_with_nothing > threshold * (1.0 + UNREACHABLE_MARGIN)
    )
    active = above_threshold & ~unreachable

    factor, halvings = bisect_scale_factors(
        lambda f: pricer(frame, law, f)[0], active, unreachable, threshold
    )
    income_cut, tax_cut = pricer(frame, law, factor)

    excess = jnp.maximum(0.0, income - threshold)
    total_b = float(jnp.sum(jnp.where(in_group, weight * income, 0.0))) / DOLLARS_PER_BILLION
    above_b = float(jnp.sum(weight * excess)) / DOLLARS_PER_BILLION
    tax_on_above_b = float(jnp.sum(weight * (tax - tax_cut))) / DOLLARS_PER_BILLION
    unreachable_b = (
        float(jnp.sum(jnp.where(unreachable, weight * excess, 0.0))) / DOLLARS_PER_BILLION
    )
    row = {
        "group": f"Top {share * PERCENT:g}%",
        "threshold": threshold,
        "n_units_M": float(jnp.sum(jnp.where(in_group, weight, 0.0))) / UNITS_PER_MILLION,
        "total_income_B": total_b,
        "above_B": above_b,
        "tax_on_above_B": tax_on_above_b,
        "after_tax_above_B": above_b - tax_on_above_b,
        "rate_on_above": PERCENT * tax_on_above_b / above_b,
        "total_income_pct_gdp": PERCENT * total_b / gdp,
        "above_pct_gdp": PERCENT * above_b / gdp,
        "after_tax_above_pct_gdp": PERCENT * (above_b - tax_on_above_b) / gdp,
        "solver_gap": solver_gap(income_cut, active, threshold),
        "unreachable_share": unreachable_b / above_b,
    }
    return row, halvings


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def gdp_billions(year: int, macro_path: str = readers.MACRO) -> float:
    """Calendar-year nominal GDP for one year, in billions of dollars."""
    gdp = readers.year_map(readers.read_csv(macro_path), GDP_COLUMN)
    if year not in gdp:
        raise KeyError(f"{macro_path} has no {GDP_COLUMN} for {year}")
    return gdp[year]


def baseline_frame(population_root: Path, year: int):
    """The year's household frame under current law, as the baseline sweep builds it.

    Returns the frame, the law read for each household, and the model's tax
    calculator.
    """
    population = prepare_population(population_root, (year,))
    prepared = prepare.prepare_scenario(population, (year,))
    raw = jax.tree.map(
        lambda value: value[0],
        one_year.population_arrays_for_years(population, (year,)),
    )
    law = one_year.law_read_for_each_household(
        jax.tree.map(lambda value: value[0], prepared.baseline_laws),
        raw["filing_status"],
    )
    build = jax.jit(
        lambda raw, law, year: prepared.frame_builder(
            raw, law, year, apply_adjusters=False
        )
    )
    frame = dict(build(raw, law, jnp.asarray(year, dtype=jnp.float64)))
    # Slots holding no record this year carry zero weight.
    frame["weight"] = jnp.where(raw["valid"], _zero_if_missing(frame["weight"]), 0.0)
    return frame, law, prepared.tax_calculator


COLUMNS = (
    "year", "gdp_B", "measure", "group", "threshold", "n_units_M",
    "total_income_B", "above_B", "tax_on_above_B", "after_tax_above_B",
    "rate_on_above", "total_income_pct_gdp", "above_pct_gdp",
    "after_tax_above_pct_gdp", "solver_gap", "unreachable_share",
)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", required=True, help="CSV file to write")
    ap.add_argument("--year", type=int, default=TABLE_YEAR)
    ap.add_argument(
        "--population",
        help="Tax-Data bridge folder. By default, the one the default economy "
        "leg pins, found the way a model run finds it",
    )
    args = ap.parse_args(argv)
    output = Path(args.output).resolve()
    population = Path(args.population).resolve() if args.population else None

    # The model's configuration paths are relative to the checkout.
    os.chdir(REPO_ROOT)
    if population is None:
        population = population_root_for(DEFAULT_CONFIG_ROOT, None)
    print(f"population: {population}")
    print(f"year: {args.year}")
    print(f"device: {jax.devices()[0]}")

    started = time.perf_counter()
    gdp = gdp_billions(args.year)
    print(f"calendar-year GDP: {gdp:,.2f} billion, from {readers.MACRO}")

    frame, law, tax_calculator = baseline_frame(population, args.year)
    check_frame_columns(list(frame))
    print(f"frame built: {time.perf_counter() - started:.1f} s")

    weight = frame["weight"]
    adult = (_zero_if_missing(frame["dep_status"]) == 0.0) & (weight > 0.0)

    rows = []
    for measure in MEASURES:
        pricer = make_pricer(tax_calculator, measure)
        income, tax = pricer(frame, law, jnp.ones(weight.shape[0]))
        for share in TOP_SHARES:
            case_started = time.perf_counter()
            row, halvings = one_group(
                pricer, frame, law, income, tax, weight, adult, share, gdp
            )
            print(
                f"{measure} {row['group']}: threshold {row['threshold']:,.0f}, "
                f"rate on above-threshold income {row['rate_on_above']:.2f}%, "
                f"{halvings} halvings, solver gap {row['solver_gap']:.2e}, "
                f"{time.perf_counter() - case_started:.1f} s"
            )
            rows.append({"year": args.year, "gdp_B": gdp, "measure": measure, **row})

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table)
    print(f"wrote {output}")
    print(f"total time: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
