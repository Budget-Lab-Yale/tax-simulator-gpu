"""The corporate income tax spread over households for the distribution table.

The distribution table counts the corporate income tax top-down: a national
dollar figure for each reported world is split between the above-normal
return, the normal return to capital and labor by the rules in
`years/corporate/burden_jax.py`, and the share borne by foreign owners is
dropped. The national figures are known before the run starts, so they are
built on the host in scenario preparation and handed to the distribution
table's compiled program one year at a time.

Every world's figure is spread over the same bases, the baseline world's, so
a household's corporate tax differs between worlds only because the national
figure does. The bases are totaled over the
households the distribution table counts, so returns filed by dependents bear
none of the tax. Three household amounts come out of the spread:

    C_b          the household's share of current-law corporate tax;
    dC_static    its share of the reform's static corporate change;
    dC_conv      its share of the reform's corporate change including the
                 corporate behavioral response.

The corporate tax a household bears is C_b in the baseline, C_b + dC_static in
the static world and C_b + dC_conv in the conventional world. The change in
cost recovery is spread with all of it on the normal return. It is the same in
every world, so it is part of C_b and never of a change.

The table is reported under two spreads (`distribution.CORPORATE_SPREADS`),
which share the labor base and differ in the two capital bases:

    income   capital income is both the above-normal base and the
             normal-capital base (`years/corporate/burden_jax.py` defines
             labor and capital income);
    assets   the above-normal base is the household's corporate equity: its
             equities, defined-contribution balances, trusts and real estate
             funds, each times the share of it that is corporate equity
             (`corp.asset_exposure_*`). The normal-capital base is its net
             capital: every asset other than homes, less every debt other
             than mortgages, plus the structure share of net home equity
             (`distribution.housing_structure_share` times home values less
             mortgages), floored at zero. Residential land is left out of
             the normal-capital base because it cannot move to other uses.

Both spreads read the baseline world's balance sheet, which no corporate
markdown or wealth shortfall has touched.

This module also names the frame columns that record what the model's own
record-level corporate and employer-payroll steps took out of income. The
conventional world's pretax income adds them back, because the table counts
the corporate burden top-down instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.wealth import avoidance as wealth_avoidance
from taxmodel.frame import steps_jax
from taxmodel.reporting import distribution, receipts
from taxmodel.reporting.receipts_inputs import ReceiptsInputs
from taxmodel.years import income_adjustments, world_names
from taxmodel.years.corporate import burden_jax

Array = jnp.ndarray

# The simulated worlds whose corporate figures become the static and the
# conventional figures.
STATIC_WORLD = "first_order"
CONVENTIONAL_WORLD = "conventional"


# The balance-sheet columns that hold corporate equity, each with the field of
# `CorporateDollars` holding the share of it that is corporate equity.
EXPOSURE_FIELD_BY_COLUMN: dict[str, str] = {
    "value.equities": "equities_exposure",
    "value.dc": "dc_exposure",
    "value.trusts": "trusts_exposure",
    "value.re_fund": "re_fund_exposure",
}

# Owner-occupied homes and the mortgages against them. The asset spread counts
# only the structure share of the equity between them.
HOUSING_ASSET_COLUMNS: tuple[str, ...] = ("value.primary_home", "value.other_home")
HOUSING_DEBT_COLUMNS: tuple[str, ...] = ("value.primary_mortgage", "value.other_mortgage")
NON_HOUSING_ASSET_COLUMNS: tuple[str, ...] = tuple(
    name for name in steps_jax.ESTATE_ASSET_COLS if name not in HOUSING_ASSET_COLUMNS
)
NON_MORTGAGE_DEBT_COLUMNS: tuple[str, ...] = tuple(
    name for name in steps_jax.ESTATE_DEBT_COLS if name not in HOUSING_DEBT_COLUMNS
)

# The baseline world's per-household amounts both spreads are built from, the
# names of `baseline_bases`. Labor and capital income are
# `burden_jax.capital_income_base`'s; the four exposed asset columns are
# market values; the last two are the balance sheet netted as the asset spread
# needs it.
LABOR_INCOME = "labor_income"
CAPITAL_INCOME = "capital_income"
NET_NON_HOUSING_ASSETS = "non_housing_assets_less_non_mortgage_debts"
NET_HOME_EQUITY = "home_values_less_mortgages"
BASE_NAMES: tuple[str, ...] = (
    LABOR_INCOME,
    CAPITAL_INCOME,
    *EXPOSURE_FIELD_BY_COLUMN,
    NET_NON_HOUSING_ASSETS,
    NET_HOME_EQUITY,
)


class CorporateDollars(NamedTuple):
    """The national corporate figures the distribution table spreads.

    Every field is shaped [year], one entry per projection year, in dollars
    for the four dollar figures. For one year of one scenario the same type
    holds scalars.

    `baseline` is CBO's corporate receipts (`rev_corp`) plus the baseline
    world's statutory rate score, which is zero. `static` adds to CBO's
    receipts the first_order world's rate score and the off-model mechanical
    corporate score (`corporate_static`). `conventional` adds the conventional
    world's rate score and the off-model score that includes the corporate
    behavioral response (`corporate`). `cost_recovery` is the cost-recovery
    change in corporate receipts, the same in every world.

    The remaining fields are the economy leg's incidence assumptions, repeated
    for every year so that the whole record can be cut one year at a time:
    `normal_return_share` (`corp.sigma_n`), `foreign_share`
    (`distribution.corp_foreign_share`), the four shares of an asset that are
    corporate equity (`corp.asset_exposure_*`, named in
    `EXPOSURE_FIELD_BY_COLUMN`), and `housing_structure_share`
    (`distribution.housing_structure_share`), the share of net home equity
    that is structures rather than land. The last five are read only by the
    asset spread.
    """

    baseline: Array
    static: Array
    conventional: Array
    cost_recovery: Array
    normal_return_share: Array
    foreign_share: Array
    equities_exposure: Array
    dc_exposure: Array
    trusts_exposure: Array
    re_fund_exposure: Array
    housing_structure_share: Array


class Incidence(NamedTuple):
    """The economy leg's corporate incidence assumptions, for one scenario.

    `asset_exposure` maps each column of `EXPOSURE_FIELD_BY_COLUMN` to the
    share of it that is corporate equity.
    """

    normal_return_share: float
    foreign_share: float
    asset_exposure: Mapping[str, float]
    housing_structure_share: float


def no_corporate_tax(n_years: int) -> CorporateDollars:
    """Figures of zero, for a scenario whose economy leg states no incidence."""
    zeros = np.zeros(n_years, dtype=np.float64)
    return CorporateDollars(*(zeros for _field in CorporateDollars._fields))


def from_receipts_inputs(
    inputs: ReceiptsInputs,
    rate_score_by_world: Mapping[str, np.ndarray],
    incidence: Incidence,
) -> CorporateDollars:
    """The national figures, from the receipts inputs and the rate score.

    `inputs` holds the run's years, in dollars. `rate_score_by_world` is the
    revenue from the scenario's change in the corporate statutory rate, one
    series per world, zero for a scenario that leaves the rate alone.
    """
    n_years = len(inputs.years)
    cbo_receipts = np.asarray(inputs.cbo_levels["rev_corp"], dtype=np.float64)

    def rate_score(world: str) -> np.ndarray:
        return np.asarray(rate_score_by_world[world], dtype=np.float64)

    def off_model_score(world: str) -> np.ndarray:
        return np.asarray(
            inputs.off_model[receipts.off_model_corporate_column(world)],
            dtype=np.float64,
        )

    def every_year(value: float) -> np.ndarray:
        return np.full(n_years, float(value), dtype=np.float64)

    return CorporateDollars(
        baseline=cbo_receipts + rate_score(world_names.BASELINE_WORLD),
        static=cbo_receipts + rate_score(STATIC_WORLD) + off_model_score(STATIC_WORLD),
        conventional=cbo_receipts
        + rate_score(CONVENTIONAL_WORLD)
        + off_model_score(CONVENTIONAL_WORLD),
        cost_recovery=np.asarray(inputs.cost_recovery["delta"], dtype=np.float64),
        normal_return_share=every_year(incidence.normal_return_share),
        foreign_share=every_year(incidence.foreign_share),
        **{
            field: every_year(incidence.asset_exposure[column])
            for column, field in EXPOSURE_FIELD_BY_COLUMN.items()
        },
        housing_structure_share=every_year(incidence.housing_structure_share),
    )


def _added_up_reading_blanks_as_zero(
    frame: Mapping[str, Array], names: tuple[str, ...], template: Array
) -> Array:
    total = jnp.zeros_like(template)
    for name in names:
        total = total + jnp.nan_to_num(jnp.asarray(frame[name], dtype=jnp.float64), nan=0.0)
    return total


def baseline_bases(
    labor_income: Array, capital_income: Array, frame: Mapping[str, Array]
) -> dict[str, Array]:
    """The baseline world's amounts both spreads are built from, under `BASE_NAMES`.

    `frame` is the baseline world's household frame; a missing balance-sheet
    entry counts as zero.
    """
    missing = [
        name
        for name in (*steps_jax.ESTATE_ASSET_COLS, *steps_jax.ESTATE_DEBT_COLS)
        if name not in frame
    ]
    if missing:
        raise KeyError(
            "the distribution table's asset spread reads "
            f"{', '.join(missing)} from the household frame, which does not "
            "carry them"
        )
    template = jnp.asarray(labor_income, dtype=jnp.float64)
    bases = {
        LABOR_INCOME: template,
        CAPITAL_INCOME: jnp.asarray(capital_income, dtype=jnp.float64),
    }
    for column in EXPOSURE_FIELD_BY_COLUMN:
        bases[column] = _added_up_reading_blanks_as_zero(frame, (column,), template)
    bases[NET_NON_HOUSING_ASSETS] = _added_up_reading_blanks_as_zero(
        frame, NON_HOUSING_ASSET_COLUMNS, template
    ) - _added_up_reading_blanks_as_zero(frame, NON_MORTGAGE_DEBT_COLUMNS, template)
    bases[NET_HOME_EQUITY] = _added_up_reading_blanks_as_zero(
        frame, HOUSING_ASSET_COLUMNS, template
    ) - _added_up_reading_blanks_as_zero(frame, HOUSING_DEBT_COLUMNS, template)
    return bases


def capital_bases(
    spread: str, dollars: CorporateDollars, bases: Mapping[str, Array]
) -> tuple[Array, Array]:
    """The above-normal base and the normal-capital base of one spread.

    `bases` is `baseline_bases`'s result and `dollars` one year's figures.
    """
    if spread == distribution.CORPORATE_SPREAD_INCOME:
        return bases[CAPITAL_INCOME], bases[CAPITAL_INCOME]
    if spread != distribution.CORPORATE_SPREAD_ASSETS:
        raise ValueError(f"no corporate spread is called {spread!r}")
    corporate_equity = sum(
        bases[column] * getattr(dollars, field)
        for column, field in EXPOSURE_FIELD_BY_COLUMN.items()
    )
    # A household's net capital is floored at zero, because a household cannot
    # hold a negative share of a national tax.
    net_capital = jnp.maximum(
        0.0,
        bases[NET_NON_HOUSING_ASSETS]
        + dollars.housing_structure_share * bases[NET_HOME_EQUITY],
    )
    return corporate_equity, net_capital


def _spread(
    amount: Array,
    normal_return_share: Array,
    dollars: CorporateDollars,
    weight: Array,
    labor_base: Array,
    above_normal_base: Array,
    normal_capital_base: Array,
) -> Array:
    """One national figure spread over households by the three-way split."""
    return burden_jax.allocate_corporate_dollars(
        amount,
        normal_return_share,
        dollars.foreign_share,
        weight,
        labor_base,
        above_normal_base,
        normal_capital_base,
    )


def current_law_share(
    dollars: CorporateDollars,
    weight: Array,
    labor_base: Array,
    above_normal_base: Array,
    normal_capital_base: Array,
) -> Array:
    """C_b: each household's share of current-law corporate tax, in dollars.

    `dollars` is one year's figures. The bases are the baseline world's, one
    value per household, and `weight` is the household weight they are totaled
    with.
    """
    bases = (weight, labor_base, above_normal_base, normal_capital_base)
    return _spread(
        dollars.baseline, dollars.normal_return_share, dollars, *bases
    ) + _spread(
        dollars.cost_recovery,
        burden_jax.COST_RECOVERY_NORMAL_RETURN_SHARE,
        dollars,
        *bases,
    )


def reform_change_share(
    reform_dollars: Array,
    dollars: CorporateDollars,
    weight: Array,
    labor_base: Array,
    above_normal_base: Array,
    normal_capital_base: Array,
) -> Array:
    """A household's share of one reform world's corporate change, in dollars.

    `reform_dollars` is that world's national figure for the year
    (`dollars.static` or `dollars.conventional`); the change is it less
    `dollars.baseline`. The bases are the baseline world's, as for C_b.
    """
    return _spread(
        reform_dollars - dollars.baseline,
        dollars.normal_return_share,
        dollars,
        weight,
        labor_base,
        above_normal_base,
        normal_capital_base,
    )


class HouseholdCorporateTax(NamedTuple):
    """One spread's corporate amounts per household, in dollars."""

    current_law: Array
    static_change: Array
    conventional_change: Array


def household_corporate_tax(
    spread: str,
    dollars: CorporateDollars,
    weight: Array,
    bases: Mapping[str, Array],
) -> HouseholdCorporateTax:
    """C_b, dC_static and dC_conv per household under one spread.

    `bases` is `baseline_bases`'s result, and `weight` the household weight
    the bases are totaled with.
    """
    labor = bases[LABOR_INCOME]
    above_normal, normal_capital = capital_bases(spread, dollars, bases)
    spread_over = (weight, labor, above_normal, normal_capital)
    return HouseholdCorporateTax(
        current_law=current_law_share(dollars, *spread_over),
        static_change=reform_change_share(dollars.static, dollars, *spread_over),
        conventional_change=reform_change_share(
            dollars.conventional, dollars, *spread_over
        ),
    )


# What the model's record-level corporate and employer-payroll steps changed
# in a household's conventional-world income, in dollars, signed as the step
# changed income (negative for a tax increase). These are the add-backs,
# under the names below.
#
#   corporate_flows      dividends, interest, rent and pass-through income
#   wages                cash wages, from the employer-payroll step
#   retirement           taxable IRA distributions plus gross pension
#                        distributions, as expanded income counts them
#   retirement_accrual   the part of that change Haig-Simons income counts:
#                        the defined-benefit share of the gross pension
#                        change (Haig-Simons income leaves out IRA
#                        distributions and the defined-contribution share of
#                        pensions)
#   gains                short- and long-term realized gains, from the
#                        sale-volume factor and the equity markdown
#
# Each is the step's change as it stands in the final conventional frame:
# the change the step recorded, scaled by every later step that scales the
# same income (the wealth haircut, income evasion and wealth-tax
# concealment). Later steps that move income from one form to another
# (conversion, entity shifting) do not scale it.
CORPORATE_FLOWS_CHANGE = "corporate_flows"
WAGE_CHANGE = "wages"
RETIREMENT_CHANGE = "retirement"
RETIREMENT_CHANGE_ACCRUAL = "retirement_accrual"
GAINS_CHANGE = "gains"

# The add-backs to each income measure. Haig-Simons income counts accrued
# gains in place of realized gains, so the change in realized gains is not
# added back to it, and it counts only the defined-benefit share of the
# pension change. The equity markdown also lowers the defined-contribution
# share of pensions, which raises Haig-Simons income; that part of the change
# is not added back, a known small error.
CASH_ADD_BACKS: tuple[str, ...] = (
    CORPORATE_FLOWS_CHANGE,
    WAGE_CHANGE,
    RETIREMENT_CHANGE,
    GAINS_CHANGE,
)
ACCRUAL_ADD_BACKS: tuple[str, ...] = (
    CORPORATE_FLOWS_CHANGE,
    WAGE_CHANGE,
    RETIREMENT_CHANGE_ACCRUAL,
)
ADD_BACK_COLUMNS: tuple[str, ...] = (
    CORPORATE_FLOWS_CHANGE,
    WAGE_CHANGE,
    RETIREMENT_CHANGE,
    RETIREMENT_CHANGE_ACCRUAL,
    GAINS_CHANGE,
)

# The employer-payroll step's change in cash wages. No later step of the
# conventional world scales wages.
WAGE_CHANGE_COLUMN = "pr_dY_exog"

# The share of the household's assets the wealth haircut removed this year,
# written on the frame by `years/wealth_state.apply_carried_stock`. Capital
# income and retirement distributions are scaled by one minus it; pass-through
# income by one minus the economy leg's pass-through capital weight times it.
HAIRCUT_FRACTION_COLUMN = "wealth_haircut"

# How the wealth haircut scales each piece of the corporate step's change
# (`years/income_adjustments.CORPORATE_CHANGE_PIECES`): as capital income, or
# as pass-through income.
_SCALED_AS_PASS_THROUGH = frozenset(
    {
        "corp_dY.part_active",
        "corp_dY.part_passive",
        "corp_dY.scorp_active",
        "corp_dY.scorp_passive",
        "corp_dY.sole_prop",
        "corp_dY.farm",
        "corp_dY.pass_through_losses",
    }
)

# The pieces income evasion scales, each with the factor column the evasion
# module leaves on the record and the columns whose positive total decides
# whether the module scaled that income (`behavior/evasion/debacker.py`).
_EVASION: dict[str, tuple[str, tuple[str, ...]]] = {
    "corp_dY.part_active": ("evasion_g_pt", ("part_active",)),
    "corp_dY.part_passive": ("evasion_g_pt", ("part_passive",)),
    "corp_dY.scorp_active": ("evasion_g_pt", ("scorp_active",)),
    "corp_dY.scorp_passive": ("evasion_g_pt", ("scorp_passive",)),
    "corp_dY.sole_prop": ("evasion_g_schc", ("sole_prop1", "sole_prop2")),
    "corp_dY.farm": ("evasion_g_schc", ("farm1", "farm2")),
    "corp_dY.rent": ("evasion_g_rent", ("rent",)),
}


def _concealment(piece: str) -> tuple[str, tuple[str, ...]] | None:
    """The concealed-share column and gate columns for a piece, or None.

    Read off the wealth-avoidance module's own table of the income each class
    of concealed wealth takes with it.
    """
    columns = {name for name, _sign in income_adjustments.CORPORATE_CHANGE_PIECES[piece]}
    for gate, scaled, kind in wealth_avoidance.CONCEALED_INCOME_GROUPS:
        if columns <= set(scaled):
            share = (
                wealth_avoidance.CONCEALED_SHARE_MARKETABLE_COL
                if kind == wealth_avoidance.MARKETABLE
                else wealth_avoidance.CONCEALED_SHARE_CLOSELY_HELD_COL
            )
            return share, gate
    return None


def _gated_scale(frame: Mapping[str, Array], factor: Array, gate: tuple[str, ...]) -> Array:
    """The factor a step applied to income it scales only when positive.

    The test is made on the final frame. A step that scales positive income by
    a positive factor keeps it positive, so the final total is positive
    exactly when the step saw positive income. A factor of zero leaves a
    total of zero, which is read as scaled.
    """
    template = jnp.asarray(factor, dtype=jnp.float64)
    total = sum(bs.column(frame, name, template) for name in gate)
    scaled = (total > 0.0) | ((factor == 0.0) & (total == 0.0))
    return jnp.where(scaled, factor, 1.0)


def _remaining(
    frame: Mapping[str, Array], piece: str, pass_through_weight: Array
) -> Array:
    """One piece of the corporate step's change, as it stands in the frame."""
    change = bs.column(frame, piece)
    fraction = bs.column(frame, HAIRCUT_FRACTION_COLUMN, change)
    weight = pass_through_weight if piece in _SCALED_AS_PASS_THROUGH else 1.0
    scale = 1.0 - weight * fraction
    if piece in _EVASION:
        factor_column, gate = _EVASION[piece]
        factor = (
            bs.column(frame, factor_column)
            if factor_column in frame
            else jnp.full_like(change, bs.NO_EVASION)
        )
        scale = scale * _gated_scale(frame, factor, gate)
    concealment = _concealment(piece)
    if concealment is not None and concealment[0] in frame:
        share_column, gate = concealment
        scale = scale * _gated_scale(frame, 1.0 - bs.column(frame, share_column), gate)
    return change * scale


def add_backs(
    frame: Mapping[str, Array], pass_through_weight: Array
) -> dict[str, Array]:
    """The add-backs for each household, under the names in `ADD_BACK_COLUMNS`.

    `frame` is the final conventional frame, and `pass_through_weight` the
    capital share of pass-through income the wealth haircut scales
    (`wealth.cap_flows_pt_weight`). A frame the corporate step did not run on
    gives zeros.
    """
    template = jnp.asarray(frame["weight"], dtype=jnp.float64)
    zeros = jnp.zeros_like(template)
    if income_adjustments.RETIREMENT_CHANGE_COLUMN not in frame:
        wages = bs.column(frame, WAGE_CHANGE_COLUMN, template)
        return {
            name: (wages if name == WAGE_CHANGE else zeros)
            for name in ADD_BACK_COLUMNS
        }
    kept = 1.0 - bs.column(frame, HAIRCUT_FRACTION_COLUMN, template)
    flows = sum(
        _remaining(frame, piece, pass_through_weight)
        for piece in income_adjustments.FLOW_CHANGE_PIECES
    )
    gains_in_the_step = sum(
        bs.column(frame, piece) for piece in income_adjustments.GAIN_CHANGE_PIECES
    )
    # The rest of the gains change was recorded after every scaling step:
    # the markdown's share of realized gains by the gains applier (already
    # scaled for concealment) and the sale-volume factor, applied last.
    gains_recorded_later = (
        bs.column(frame, income_adjustments.GAINS_CHANGE_COLUMN) - gains_in_the_step
    )
    gains = gains_recorded_later + sum(
        _remaining(frame, piece, pass_through_weight)
        for piece in income_adjustments.GAIN_CHANGE_PIECES
    )
    return {
        CORPORATE_FLOWS_CHANGE: flows,
        WAGE_CHANGE: bs.column(frame, WAGE_CHANGE_COLUMN, template),
        RETIREMENT_CHANGE: kept
        * bs.column(frame, income_adjustments.RETIREMENT_CHANGE_COLUMN),
        RETIREMENT_CHANGE_ACCRUAL: kept
        * bs.column(frame, income_adjustments.RETIREMENT_CHANGE_ACCRUAL_COLUMN),
        GAINS_CHANGE: gains,
    }



def income_taken_out(add_backs: Mapping[str, Array]) -> tuple[Array, Array]:
    """What the record-level steps took out of cash and of accrual income.

    Positive for a tax increase: the add-backs are signed as the steps
    changed income, so the amount to add back is minus their sum.
    """
    cash = -sum(add_backs[name] for name in CASH_ADD_BACKS)
    accrual = -sum(add_backs[name] for name in ACCRUAL_ADD_BACKS)
    return cash, accrual
