"""Spreading the year's age-cell answers over the households in those cells.

The schedule solver and the stock recurrence write one cell table per year.
This module turns the year's cell table into per-household numbers.  Every
quantity it writes was decided in the cell table.

Three things change for a household, and one does not.

1. **The rate channel.**  A household with realized gains scales them by its
   cell's rate factor, the cell's scenario turnover over its baseline turnover.
   A household with a realized loss is left alone: the model is about whether
   to sell an appreciated asset.

2. **The lock-in channel.**  The cell's extra realizations — what the change in
   the cell's stock of gain adds to or takes off this year's sales — are spread
   over the cell's households.  Half of the cell's amount follows realized
   gains and half follows holdings (the ``allocation_weight_on_holdings``
   parameter); a cell with no realizations allocates entirely by holdings, and
   one holding nothing allocates entirely by realizations.

3. **Gains deemed realized at death.**  What would land on the final return if
   the household died this year: gain summed across asset classes at the
   regime's realized-at-death share, net of the §121 exclusion on a primary
   home and of the death-gain exclusion, scaled by the cell's deemed factor and
   by the share of the estate not left to charity.  Multiplying by the
   household's own mortality gives the expected amount, so death enters as an
   expectation and not as a random draw.

``kg_lt`` does not change: no deemed death gain enters it, so it stays the
living taxpayer's return and the marginal rate on gains measures the
during-life margin.  The tax on death gains is priced separately, by running
the calculator again on the same households with the deemed gain added to
their realized gains (``years/worlds.py``).

The applier also attaches the two equivalent tax rates — the expected present
value of tax on a dollar of gain left unrealized, under baseline law and under
the reform — because entity shifting reads them and runs after this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs
from taxmodel.years.wealth_state import age_cohort

from .stepper import cell_table_column, extra_realizations_from_stock_change
from .stock_inputs import KG_AGE_MIN

# The gross estate each household would leave.  A decedent's chance of leaving
# appreciated property to charity is a function of it.  `frame/steps_jax.py`
# writes it as part of the estate tax's mortality step, adding up the
# balance-sheet asset columns.  It is defined here, not in the pre-pass, so
# that both modules can read it without an import cycle.
ESTATE_GROSS_COLUMN = "estate_economic_gross"

Array = jnp.ndarray
Records = Mapping[str, Array]

# The asset classes whose gain a decedent can be deemed to realize, in the order
# the regime's realized-at-death shares are given in.
DEEMED_ASSET_CLASSES = (
    "equities",
    "pass_throughs",
    "primary_home",
    "other_home",
    "re_fund",
)

# The frame carries an asset's market value as `value.equities` and the owner's
# cost basis as `basis.equities`, and so on for each class above.
VALUE_COLUMN_PREFIX = "value."
BASIS_COLUMN_PREFIX = "basis."

# The household columns this module writes.  All are new columns except
# `kg_lt`, the realized long-term gain the tax calculation uses.
LOCKIN_COLUMN = "kg_lockin"
DEEMED_FULL_COLUMN = "kg_deemed_full"
DEEMED_FULL_OWN_EXCLUSION_COLUMN = "kg_deemed_full_own_exclusion"
DEEMED_FULL_MARRIED_EXCLUSION_COLUMN = "kg_deemed_full_married_exclusion"
DEEMED_EXPECTED_COLUMN = "kg_deemed"
EQUIVALENT_TAX_RATE_BASELINE_COLUMN = "equivalent_tax_rate_baseline"
EQUIVALENT_TAX_RATE_SCENARIO_COLUMN = "equivalent_tax_rate_scenario"

# The change in the household's realized long-term gains caused by the
# corporate equity markdown: its share of the cell's extra realizations less
# its share of what they would have been with no markdown debit. Negative when
# the markdown lowers sales. Only the distribution table reads it, through
# `corp_dY_gains` (`years/worlds.py`); the wealth avoidance module conceals
# the same share of it as of the gains it is part of.
CORPORATE_MARKDOWN_GAINS_COLUMN = "kg_corporate_markdown"

# The household's own probability of dying this year, and its probability of
# being a widow or widower at death.  A frame without the widowhood column is
# treated as having none.
#
# A data caveat about widows: the basis the SCF reports for a widowed household
# is the couple's original purchase cost, not the stepped-up value the survivor
# legally received when the first spouse died.  Under current law the data
# overstates a widow's taxable gain and the model treats her as more locked in
# than she is; under a reform where spouses inherit each other's basis, the
# data is right.
MORTALITY_COLUMN = "m_household"
WIDOWHOOD_COLUMN = "p_widow"

# The household's stock of unrealized gain.  The lock-in allocation follows it
# for the part of the allocation that goes by holdings.
UNREALIZED_GAIN_STOCK_COLUMN = "unrealized_gain_stock"

# Filing statuses of an unmarried filer.  Their death-gain exclusion has two
# branches — their own amount, and the married amount they would have if
# widowed — which are priced separately and blended by the widowhood
# probability (``years/worlds.the_deemed_death_tax``).
UNMARRIED_FILING_STATUSES = (1, 4)

# Regime codes for the two treatments under which a decedent's gain is still
# taxable, and so eligible for the death-gain exclusion: carryover basis (1) and
# deemed realization at death (2).  Step-up (0) leaves nothing to exclude.
TAXABLE_AT_DEATH_REGIME_CODES = (1, 2)


DEFAULTS: dict[str, Any] = {
    # How the cell's extra realizations are spread within the cell: 0 spreads
    # them by realized gains, 1 by holdings, and 0.5 splits the difference.
    "allocation_weight_on_holdings": 0.5,
    # The fraction of market value at which a decedent's pass-through
    # interests are valued for the tax at death.  The other asset classes are
    # valued at market value.  The model uses the estate tax's fitted
    # pass-through valuation factor (``rho_pt`` in
    # config/calibrations/estate/valuation_fit.yaml).
    "pass_through_valuation_factor": 1.0,
    # The share of each asset class's gain a decedent is deemed to realize under
    # this year's treatment of gains at death, in DEEMED_ASSET_CLASSES order.
    # It comes from the year's regime rather than from this module.
    "realized_at_death_share": (0.0, 0.0, 0.0, 0.0, 0.0),
    # Which treatment each asset class is under, in the same order, using
    # TAXABLE_AT_DEATH_REGIME_CODES.  Decides which classes the death-gain
    # exclusion is spread across.
    "regime_code": (0, 0, 0, 0, 0),
    # The §121 exclusion on the gain from a primary home, and the exclusion on
    # a decedent's gain, both in dollars.  Either may be one number for everyone
    # or one number per household.  `death_gain_exclusion_married` is the amount
    # a filer would have if married at death, which is the second branch for an
    # unmarried filer.
    "section_121_exclusion": 0.0,
    "death_gain_exclusion": 0.0,
    "death_gain_exclusion_married": 0.0,
    # The year's cell table, shaped [age cell, field] over
    # stepper.CELL_TABLE_FIELDS.  The runner slices it out of the stock
    # recurrence's stack of years and passes it in.
    "cell_table": None,
    # The base year's price level over this year's, and the coefficients of the
    # two logits, which together give a decedent's own chance of leaving
    # appreciated property to charity.  Both belong to the run: the deflator is
    # macro data and the coefficients are calibration.  With no settings
    # supplied, the share falls back to the cell table's.
    "charitable_deflator": 1.0,
    "charitable_settings": None,
}


MANIFEST: dict[str, Any] = {
    "name": "kg_applier",
    # The function in this file that computes the response.
    "entry_point": "apply_cell_table",
    # The applier has no zero response.  A scenario that runs the gains model
    # gets the applier at the front of the behavior chain, with its solved cell
    # table; a scenario that does not is given no applier at all.
    "zero_response": None,
    "family": "kg_dynamics",
    "reads": (
        "kg_lt",
        "age1",
        "age2",
        "filing_status",
        UNREALIZED_GAIN_STOCK_COLUMN,
        MORTALITY_COLUMN,
        # The decedent's own gross estate, which sizes their chance of leaving
        # appreciated property to charity.
        ESTATE_GROSS_COLUMN,
        *(VALUE_COLUMN_PREFIX + name for name in DEEMED_ASSET_CLASSES),
        *(BASIS_COLUMN_PREFIX + name for name in DEEMED_ASSET_CLASSES),
    ),
    "writes": (
        "kg_lt",
        LOCKIN_COLUMN,
        DEEMED_FULL_COLUMN,
        DEEMED_FULL_OWN_EXCLUSION_COLUMN,
        DEEMED_FULL_MARRIED_EXCLUSION_COLUMN,
        DEEMED_EXPECTED_COLUMN,
        EQUIVALENT_TAX_RATE_BASELINE_COLUMN,
        EQUIVALENT_TAX_RATE_SCENARIO_COLUMN,
        CORPORATE_MARKDOWN_GAINS_COLUMN,
    ),
    # The cell table carries every number the applier reads, so it asks the
    # pricing pass for nothing.
    "prices": (),
    "discrete_prices": (),
    "draws": (),
}


# -----------------------------------------------------------------------------
# The chance a decedent leaves appreciated property to charity
# -----------------------------------------------------------------------------
# The pre-pass runs this per age cell and the applier per household.  It is
# defined here, like the gross-estate column above, to keep the imports between
# this module and the pre-pass one-way.  The coefficients are read out of the
# settings file by ``prepass.read_charitable_bequest_settings``.


@dataclass(frozen=True)
class CharitableBequestSettings:
    """The two-part logit for charitable bequests, and the year it was fit in.

    A decedent's estate goes to charity in two steps: whether anything is left
    to charity at all (the extensive part) and what share of the estate that is
    (the intensive part).  Both are logits in the log of the gross estate,
    measured in millions of dollars of ``base_year``, and the model uses their
    product as the share of a dollar of gain that escapes tax through charity.

    The four coefficients and the base year come from
    ``config/calibrations/kg/settings.yaml``, which records where they were
    estimated.
    """

    extensive_intercept: float
    extensive_log_estate_slope: float
    intensive_intercept: float
    intensive_log_estate_slope: float
    base_year: int


def charitable_bequest_probability(
    estate_gross: Array,
    deflator_to_base_year: Array | float,
    settings: CharitableBequestSettings,
) -> Array:
    """The share of a decedent's gain that goes to charity, household by household.

    The gross estate is deflated to the base year the coefficients were fit in
    and expressed in millions of dollars, then run through both logits; the
    answer is their product.  A household with no estate at all gets zero.

    ``deflator_to_base_year`` is the base year's CPI-U over this year's, so
    that a 2035 estate is measured in the same dollars a 2026 one is.
    """

    estate = jnp.asarray(estate_gross, dtype=jnp.float64)
    dollars_per_million = 1e6
    in_base_year_millions = estate * deflator_to_base_year / dollars_per_million

    has_estate = in_base_year_millions > 0.0
    # Take the log only where there is an estate.  The substituted value keeps
    # the logarithm of a non-positive number out of the array, which matters
    # under automatic differentiation as well as at run time.
    safe_estate = jnp.where(has_estate, in_base_year_millions, 1.0)
    log_estate = jnp.log(safe_estate)

    extensive = _logistic(
        settings.extensive_intercept + settings.extensive_log_estate_slope * log_estate
    )
    intensive = _logistic(
        settings.intensive_intercept + settings.intensive_log_estate_slope * log_estate
    )
    return jnp.where(has_estate, extensive * intensive, 0.0)


def _logistic(x: Array) -> Array:
    """The logistic function."""

    return 1.0 / (1.0 + jnp.exp(-x))


def cell_row_for_household(records: Records, n_ages: int) -> Array:
    """Each household's row in the cell table.

    The cell is the filer's age, or the older spouse's on a joint return, with
    everyone at or above the top of the grid pooled into the last cell.  The
    model's grid runs from 18 to 80; the length comes from the cell table it is
    being read for, so the tests can run the same code on a handful of cells.
    """

    ages = jnp.arange(KG_AGE_MIN, KG_AGE_MIN + n_ages)
    return age_cohort(records, ages) - KG_AGE_MIN


def gain_by_class(records: Records, value_factor: Array | None = None) -> Array:
    """Each household's unrealized gain by asset class, shaped [household, class].

    The gain is value less basis, floored at zero, household by household.
    ``value_factor``, one number per class in :data:`DEEMED_ASSET_CLASSES`
    order, scales each class's value before the basis comes off; the tax at
    death uses it to value pass-through interests below market value.  Without
    it the gain is at market value.
    """

    template = jnp.asarray(records["kg_lt"], dtype=jnp.float64)
    if value_factor is None:
        value_factor = jnp.ones(len(DEEMED_ASSET_CLASSES), dtype=jnp.float64)
    value_factor = jnp.asarray(value_factor, dtype=jnp.float64)
    columns = []
    for index, name in enumerate(DEEMED_ASSET_CLASSES):
        value = bs.column(records, VALUE_COLUMN_PREFIX + name, template)
        basis = bs.column(records, BASIS_COLUMN_PREFIX + name, template)
        columns.append(jnp.maximum(0.0, value_factor[index] * value - basis))
    return jnp.stack(columns, axis=-1)


def value_factor_at_death(pass_through_valuation_factor: Array) -> Array:
    """The fraction of market value each asset class is valued at for the tax
    at death: the pass-through factor for pass-through interests, and one for
    every other class.
    """

    pass_throughs = DEEMED_ASSET_CLASSES.index("pass_throughs")
    return (
        jnp.ones(len(DEEMED_ASSET_CLASSES), dtype=jnp.float64)
        .at[pass_throughs]
        .set(jnp.asarray(pass_through_valuation_factor, dtype=jnp.float64))
    )


def gain_after_the_home_sale_exclusion(
    stakes: Array, section_121_exclusion: Array
) -> Array:
    """Each asset class's gain after §121 comes off the primary home.

    The exclusion is for gain on a principal residence, so it comes off that
    one class and leaves the other four alone, and it cannot take the home's
    gain below zero.

    ``stakes`` is [household, asset class] in :data:`DEEMED_ASSET_CLASSES`
    order.
    """

    primary_home = DEEMED_ASSET_CLASSES.index("primary_home")
    exclusion = jnp.asarray(section_121_exclusion, dtype=jnp.float64)
    return stakes.at[:, primary_home].set(
        jnp.maximum(0.0, stakes[:, primary_home] - exclusion)
    )


def taxable_shares_if_widowed_or_not(
    stakes: Array,
    regime_code: Array,
    per_decedent: Array,
    per_decedent_if_widowed: Array,
    filing_status: Array,
) -> tuple[Array, Array]:
    """The share of the gain still taxable at death, counted both ways.

    Two shares per household: the one left after the decedent's own
    per-decedent exclusion, and the one left after the larger amount a
    surviving spouse can claim.  Only an unmarried filer can be the survivor
    claiming it, so for a married filer the second is the first again.  The
    caller blends the two by the chance of being widowed at death.
    """

    is_unmarried = jnp.isin(
        jnp.asarray(filing_status), jnp.asarray(UNMARRIED_FILING_STATUSES)
    )
    own = jnp.asarray(per_decedent, dtype=jnp.float64)
    if_widowed = jnp.where(
        is_unmarried, jnp.asarray(per_decedent_if_widowed, dtype=jnp.float64), own
    )
    return (
        taxable_share_after_death_gain_exclusion(stakes, regime_code, own),
        taxable_share_after_death_gain_exclusion(stakes, regime_code, if_widowed),
    )


def taxable_share_after_death_gain_exclusion(
    stakes: Array,
    regime_code: Array,
    exclusion: Array,
) -> Array:
    """The share of a decedent's gain the death-gain exclusion leaves taxable.

    The exclusion takes the decedent's first dollars of gain away pro-rata
    across the asset classes whose treatment still taxes the gain — carryover
    basis or deemed realization — after the §121 exclusion.  Classes that get a stepped-up basis have nothing to exclude and
    stay out of the pool.
    """

    eligible = jnp.isin(
        jnp.asarray(regime_code), jnp.asarray(TAXABLE_AT_DEATH_REGIME_CODES)
    )
    pool = jnp.sum(jnp.where(eligible, stakes, 0.0), axis=-1)
    holds_something = pool > 0
    safe_pool = jnp.where(holds_something, pool, 1.0)
    return jnp.where(
        holds_something, jnp.maximum(0.0, pool - exclusion) / safe_pool, 0.0
    )


def apply_cell_table(
    records: Records,
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, Any],
    draws: Mapping[str, Array] | None = None,
) -> dict[str, Array]:
    """The gains applier, in the behavior module contract's signature.

    Prices and draws go unread.  The cell table was solved on the host before
    this execution; this module spreads it over households, so the gains
    households realize and the gains the stock loses are the same gains.  The unused
    arguments stay in the signature because every behavior module has the same
    one.

    Returns the columns listed in ``MANIFEST["writes"]``.
    """

    del prices_reform, prices_baseline, draws

    settings = {**DEFAULTS, **params}
    cell_table = settings["cell_table"]
    if cell_table is None:
        raise ValueError(
            "the gains applier needs the year's cell table; the runner slices "
            "it out of the stock recurrence's result and passes it as the "
            "`cell_table` parameter"
        )
    cell_table = jnp.asarray(cell_table, dtype=jnp.float64)

    row = cell_row_for_household(records, cell_table.shape[0])

    def cell(field: str) -> Array:
        return cell_table_column(cell_table, field)[row]

    rate_factor = cell("rate_factor")
    extra_realizations = cell("extra_realizations")
    deemed_factor = cell("deemed_factor")
    baseline_realizations = cell("baseline_realizations")
    baseline_stock = cell("baseline_stock")
    equivalent_tax_rate_baseline = cell("equivalent_tax_rate_baseline")
    equivalent_tax_rate_scenario = cell("equivalent_tax_rate_scenario")

    kg_lt = jnp.asarray(records["kg_lt"], dtype=jnp.float64)
    gain_stock = bs.column(records, UNREALIZED_GAIN_STOCK_COLUMN, kg_lt)

    # The share of a decedent's gain that goes to charity is worked out for the
    # household itself, from its own gross estate.  The age cell's average
    # share is used in the stock recurrence (`behavior/kg/stepper.py`).
    charity_settings = settings.get("charitable_settings")
    if charity_settings is None:
        charitable_bequest_share = cell("charitable_bequest_share")
    else:
        charitable_bequest_share = charitable_bequest_probability(
            bs.column(records, ESTATE_GROSS_COLUMN, kg_lt),
            settings["charitable_deflator"],
            charity_settings,
        )

    # Each share adds to one within a cell, so any blend of them does too.  A
    # cell that realized nothing allocates entirely by holdings, and one that
    # holds nothing allocates entirely by realizations.
    cell_realized_something = baseline_realizations > 0
    cell_holds_something = baseline_stock > 0
    share_of_realizations = jnp.maximum(kg_lt, 0.0) / jnp.where(
        cell_realized_something, baseline_realizations, 1.0
    )
    share_of_holdings = gain_stock / jnp.where(
        cell_holds_something, baseline_stock, 1.0
    )
    by_realizations = jnp.where(
        cell_realized_something,
        share_of_realizations,
        jnp.where(cell_holds_something, share_of_holdings, 0.0),
    )
    by_holdings = jnp.where(
        cell_holds_something,
        share_of_holdings,
        jnp.where(cell_realized_something, share_of_realizations, 0.0),
    )
    weight_on_holdings = jnp.asarray(
        settings["allocation_weight_on_holdings"], dtype=jnp.float64
    )
    allocation = (1.0 - weight_on_holdings) * by_realizations + (
        weight_on_holdings * by_holdings
    )
    lockin = extra_realizations * allocation

    # The cell's extra realizations with and without the markdown debit, from
    # the formula the stored `extra_realizations` was built with.
    def extra_realizations_with_debit(debit: Array) -> Array:
        return extra_realizations_from_stock_change(
            cell("scenario_realization_rate"),
            cell("stock_change_entering"),
            debit,
            baseline_stock,
        )

    markdown_gains_change = (
        extra_realizations_with_debit(cell("corporate_gain_debit"))
        - extra_realizations_with_debit(jnp.zeros_like(baseline_stock))
    ) * allocation

    stakes = gain_after_the_home_sale_exclusion(
        gain_by_class(
            records, value_factor_at_death(settings["pass_through_valuation_factor"])
        ),
        settings["section_121_exclusion"],
    )

    realized_at_death = jnp.asarray(
        settings["realized_at_death_share"], dtype=jnp.float64
    )
    taxable_stake = stakes @ realized_at_death

    taxable_after_own, taxable_after_married = taxable_shares_if_widowed_or_not(
        stakes,
        jnp.asarray(settings["regime_code"]),
        settings["death_gain_exclusion"],
        settings["death_gain_exclusion_married"],
        records["filing_status"],
    )

    # Appreciated property left to charity is never taxed at death, so the
    # charitable bequest share comes off the gain before anything else.
    not_left_to_charity = 1.0 - charitable_bequest_share
    deemed_own = not_left_to_charity * deemed_factor * taxable_stake * taxable_after_own
    deemed_married = (
        not_left_to_charity * deemed_factor * taxable_stake * taxable_after_married
    )
    widowhood = bs.column(records, WIDOWHOOD_COLUMN, kg_lt)
    deemed_full = widowhood * deemed_married + (1.0 - widowhood) * deemed_own
    mortality = bs.column(records, MORTALITY_COLUMN, kg_lt)

    return {
        "kg_lt": jnp.where(kg_lt > 0, kg_lt * rate_factor, kg_lt) + lockin,
        LOCKIN_COLUMN: lockin,
        DEEMED_FULL_OWN_EXCLUSION_COLUMN: deemed_own,
        DEEMED_FULL_MARRIED_EXCLUSION_COLUMN: deemed_married,
        DEEMED_FULL_COLUMN: deemed_full,
        DEEMED_EXPECTED_COLUMN: mortality * deemed_full,
        EQUIVALENT_TAX_RATE_BASELINE_COLUMN: equivalent_tax_rate_baseline,
        EQUIVALENT_TAX_RATE_SCENARIO_COLUMN: equivalent_tax_rate_scenario,
        CORPORATE_MARKDOWN_GAINS_COLUMN: markdown_gains_change,
    }
