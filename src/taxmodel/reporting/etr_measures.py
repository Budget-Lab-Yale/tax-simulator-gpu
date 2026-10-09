"""Per-household income, tax and corporate bases for the distribution table.

These are formed inside the compiled year, from one world's tax results and
the household frame that world's tax was calculated on. The distribution
table is formed from them in `reporting/device_sums.distribution_table_for_year`.

Haig-Simons income
------------------
Expanded income counts capital gains when they are realized. Haig-Simons
income counts them as they accrue. Starting from expanded income:

* the three realized-gain lines come out: short-term gains, long-term gains
  and capital gain distributions;
* the seven accrued-gain columns Tax-Data carries go in: equities,
  pass-through businesses, the primary home, other homes, real estate funds,
  defined-contribution retirement balances and trusts;
* retirement income that would then be counted twice comes out: the whole
  taxable individual retirement account distribution, and the
  defined-contribution share of gross pension distributions. That share is the
  household's defined-contribution balance over its defined-contribution plus
  defined-benefit balances, and zero for a household with no balances.
  Defined-benefit pensions have no accrual column, so they stay in.

Nothing is floored, so a household whose portfolio lost value has negative
Haig-Simons income.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp

from taxmodel.reporting import corporate_spread
from taxmodel.reporting.estate_allocator_jax import DEEMED_GAINS_COLUMN
from taxmodel.years.corporate import burden_jax as corporate_burden

Array = jnp.ndarray

# The evaluated worlds whose measures the distribution table reads, in the
# order of the world axis of `device_sums.ReportColumns.world_measures`.
MEASURED_WORLDS: tuple[str, ...] = ("baseline", "first_order", "conventional")

# The tax results the table reads. A calculator that produces none of these
# (the loop tests stand a simple function in for the calculator chain) forms
# no table.
EXPANDED_INCOME_COLUMN = "expanded_inc"
RESULT_COLUMNS: tuple[str, ...] = (
    EXPANDED_INCOME_COLUMN,
    "liab_iit_net",
    "liab_pr",
    "liab_wealth",
    # The positive half of netted partnership and S corporation income, part
    # of the corporate spread's business income.
    "part_scorp",
)

# The seven accrued-gain columns Tax-Data carries.
ACCRUAL_COLUMNS: tuple[str, ...] = (
    "accruals.equities",
    "accruals.pass_throughs",
    "accruals.primary_home",
    "accruals.other_home",
    "accruals.re_fund",
    "accruals.dc",
    "accruals.trusts",
)

# Retirement balances: defined-contribution, which has an accrual column, and
# defined-benefit, which does not.
DEFINED_CONTRIBUTION_BALANCE_COLUMN = "value.dc"
DEFINED_BENEFIT_BALANCE_COLUMN = "value.db"

# Short-term gains, long-term gains and capital gain distributions.
REALIZED_GAIN_COLUMNS: tuple[str, ...] = ("kg_st", "kg_lt", "other_gains")

IRA_DISTRIBUTION_COLUMN = "txbl_ira_dist"
PENSION_DISTRIBUTION_COLUMN = "gross_pens_dist"

HAIG_SIMONS_FRAME_COLUMNS: tuple[str, ...] = (
    *ACCRUAL_COLUMNS,
    DEFINED_CONTRIBUTION_BALANCE_COLUMN,
    DEFINED_BENEFIT_BALANCE_COLUMN,
    *REALIZED_GAIN_COLUMNS,
    IRA_DISTRIBUTION_COLUMN,
    PENSION_DISTRIBUTION_COLUMN,
)

# The frame columns the corporate spread's labor and capital income are built
# from. Wages are
# each earner's base pay, tips and overtime.
WAGE_COLUMNS: tuple[str, ...] = (
    "wages_base1",
    "tips1",
    "ot1",
    "wages_base2",
    "tips2",
    "ot2",
)
SELF_EMPLOYMENT_AND_FARM_COLUMNS: tuple[str, ...] = (
    "sole_prop1",
    "sole_prop2",
    "farm1",
    "farm2",
)
FINANCIAL_INCOME_COLUMNS: tuple[str, ...] = (
    "txbl_int",
    "exempt_int",
    "div_ord",
    "div_pref",
    "kg_st",
    "kg_lt",
)
CORPORATE_BASE_FRAME_COLUMNS: tuple[str, ...] = (
    *WAGE_COLUMNS,
    *SELF_EMPLOYMENT_AND_FARM_COLUMNS,
    *FINANCIAL_INCOME_COLUMNS,
)

# The per-world measures `world_measures` returns.
WORLD_MEASURES: tuple[str, ...] = (
    "expanded_income",
    "haig_simons_income",
    "income_tax",
    "payroll_tax",
    "wealth_tax",
)


def forms_the_table(cell: Mapping[str, Array]) -> bool:
    """Whether this world's tax results carry what the table reads."""
    return all(name in cell for name in RESULT_COLUMNS)


def _require(columns: Mapping[str, Array], names: Sequence[str], where: str) -> None:
    missing = [name for name in names if name not in columns]
    if missing:
        raise KeyError(
            f"the distribution table reads {', '.join(missing)} from {where}, "
            "which does not carry them"
        )


def _added_up_reading_blanks_as_zero(
    columns: Mapping[str, Array], names: Sequence[str], template: Array
) -> Array:
    """The named columns added up, with a missing value read as zero."""
    total = jnp.zeros_like(template)
    for name in names:
        total = total + jnp.nan_to_num(
            jnp.asarray(columns[name], dtype=jnp.float64), nan=0.0
        )
    return total


def haig_simons_income(cell: Mapping[str, Array], frame: Mapping[str, Array]) -> Array:
    """Haig-Simons income for each household, as the module docstring defines it."""
    _require(frame, HAIG_SIMONS_FRAME_COLUMNS, "the household frame")
    expanded_income = jnp.asarray(cell[EXPANDED_INCOME_COLUMN], dtype=jnp.float64)
    accruals = _added_up_reading_blanks_as_zero(frame, ACCRUAL_COLUMNS, expanded_income)
    realized_gains = _added_up_reading_blanks_as_zero(
        frame, REALIZED_GAIN_COLUMNS, expanded_income
    )
    defined_contribution = _added_up_reading_blanks_as_zero(
        frame, (DEFINED_CONTRIBUTION_BALANCE_COLUMN,), expanded_income
    )
    retirement_balances = defined_contribution + _added_up_reading_blanks_as_zero(
        frame, (DEFINED_BENEFIT_BALANCE_COLUMN,), expanded_income
    )
    has_balances = retirement_balances > 0.0
    defined_contribution_share = jnp.where(
        has_balances,
        defined_contribution / jnp.where(has_balances, retirement_balances, 1.0),
        0.0,
    )
    return (
        expanded_income
        - realized_gains
        + accruals
        - jnp.asarray(frame[IRA_DISTRIBUTION_COLUMN], dtype=jnp.float64)
        - jnp.asarray(frame[PENSION_DISTRIBUTION_COLUMN], dtype=jnp.float64)
        * defined_contribution_share
    )


def income_tax(cell: Mapping[str, Array]) -> Array:
    """Income tax after refundable credits, less any tax on gains deemed realized at death.

    The tax on gains deemed realized at death is charged to heirs as part of
    the tax owed at death, so it is taken out of the income tax here.
    """
    tax = jnp.asarray(cell["liab_iit_net"], dtype=jnp.float64)
    deemed = cell.get(DEEMED_GAINS_COLUMN)
    if deemed is None:
        return tax
    return tax - jnp.asarray(deemed, dtype=jnp.float64)


def corporate_bases(
    cell: Mapping[str, Array], frame: Mapping[str, Array]
) -> tuple[Array, Array]:
    """The labor and capital income the corporate tax is spread over."""
    _require(frame, CORPORATE_BASE_FRAME_COLUMNS, "the household frame")
    template = jnp.asarray(frame[WAGE_COLUMNS[0]], dtype=jnp.float64)
    wages = _added_up_reading_blanks_as_zero(frame, WAGE_COLUMNS, template)
    business_income = _added_up_reading_blanks_as_zero(
        frame, SELF_EMPLOYMENT_AND_FARM_COLUMNS, template
    ) + _added_up_reading_blanks_as_zero(cell, ("part_scorp",), template)
    financial_income = _added_up_reading_blanks_as_zero(
        frame, FINANCIAL_INCOME_COLUMNS, template
    )
    return corporate_burden.capital_income_base(
        wages, business_income, financial_income
    )


def baseline_corporate_bases(
    cell: Mapping[str, Array], frame: Mapping[str, Array]
) -> dict[str, Array]:
    """The baseline world's amounts the corporate tax is spread over.

    Every world's corporate tax is spread over these, under the names in
    `corporate_spread.BASE_NAMES`.
    """
    return corporate_spread.baseline_bases(*corporate_bases(cell, frame), frame)


def world_measures(
    cell: Mapping[str, Array], frame: Mapping[str, Array]
) -> dict[str, Array]:
    """One world's per-household measures, under the names in `WORLD_MEASURES`."""
    return {
        "expanded_income": jnp.asarray(cell[EXPANDED_INCOME_COLUMN], dtype=jnp.float64),
        "haig_simons_income": haig_simons_income(cell, frame),
        "income_tax": income_tax(cell),
        "payroll_tax": jnp.asarray(cell["liab_pr"], dtype=jnp.float64),
        "wealth_tax": jnp.asarray(cell["liab_wealth"], dtype=jnp.float64),
    }

