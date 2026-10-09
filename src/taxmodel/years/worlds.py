"""The five worlds of one simulated year, in the order the model evaluates them.

One call computes one calendar year. The first part computes the raw,
income-flow, and mechanical frames, then advances the mechanical wealth stock
once this year's burden is known. The second part applies household behavior
to the conventional frame and advances a separate conventional wealth stock.
Baseline tax results are computed outside this function and passed in.

The caller supplies the frame builder and the behavior step. A production run
uses `frame/build_frame.make_frame_builder` for the frame and the composed
chain of behavior modules from `behavior/loader.py`, plus the capital-gains
builders in `behavior/gains/builders.py` for a scenario that runs the gains model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import jax.numpy as jnp
from jax.tree_util import register_dataclass

from taxmodel.reporting import estate_allocator_jax
from taxmodel.scenario import law_by_year
from taxmodel.years import burden as burden_module
from taxmodel.years import income_adjustments, wealth_history, wealth_state
from taxmodel.years.corporate.channel import (
    BASELINE_CORPORATE_RATE_COLUMN,
    CORPORATE_RATE_COLUMN,
)

Array = jnp.ndarray
Records = Mapping[str, Array]
TaxResults = Mapping[str, Array]
Prices = Mapping[str, Array]
# A frame builder turns one year's raw population arrays, that year's bound
# law, and the scalar calendar year into the frame the calculators read. The
# keyword ``apply_adjusters`` is false for the raw frame and true for the
# flows frame; the true branch applies the corporate and employer-payroll
# adjustments and supplies ``corp_dY_exog``, ``pr_dY_exog`` and the other
# income-change columns `reporting/corporate_spread.py` names. Neither
# branch applies the wealth haircut or household behavior. The set and shapes
# of output keys are the same in every projection year, because the callable
# runs inside a compiled program. `frame/build_frame.make_frame_builder` makes
# the production one.
FrameBuilder = Callable[..., dict[str, Array]]
TaxCalculator = Callable[[Records, object], dict[str, Array]]
# A price calculator takes a frame and a bound law and returns that frame's
# marginal tax rates under that law, as derivatives of the calculators.
PriceCalculator = Callable[[Records, object], dict[str, Array]]
# The two above in one call: this year's liabilities and marginal rates for
# the same households under the same law, from one run of the calculators.
TaxesAndPrices = Callable[
    [Records, object], tuple[dict[str, Array], dict[str, Array]]
]
# The behavior step: the year's households, the year's capital-gains tables
# (None for a scenario without the gains model), both laws' prices, and this
# scenario's behavior parameters. The gains tables and the parameters are
# passed as data each year, so scenarios with the same array shapes share one
# compiled program (years/gains_solve.py, behavior/parameters.py).
BehaviorBuilder = Callable[[Records, Any, Prices, Prices, Any], dict[str, Array]]


@dataclass(frozen=True)
class SimulationState:
    """The two wealth shortfalls carried between years.

    Each field is an age-by-percentile matrix in dollars. The mechanical
    matrix belongs to the first sweep and the conventional matrix to the
    second. Nothing else is carried between years; the percentile cutoffs are
    formed from each year's pre-haircut frame.
    """

    mechanical_wealth: Array
    conventional_wealth: Array


register_dataclass(
    SimulationState,
    data_fields=[field.name for field in fields(SimulationState)],
    meta_fields=[],
)


@dataclass(frozen=True)
class AnnualResults:
    """One year's five worlds, and the prices a behavioral response reads.

    The first five fields hold the tax results for the five worlds:
    `baseline`, `first_order`, `mechanical_ex_saving`, `mechanical` and
    `conventional`. `years/world_names.py` defines each world. The frame
    fields hold the household information each world's tax was calculated on:
    `mechanical_frame` is the frame behind the `mechanical` results.

    ``mechanical_wealth_channel`` is the first sweep's wealth-channel state
    for the year, in the names ``reporting/wealth_diagnostics.py`` reads. The
    reporting layer's ``WealthShortfallHistory`` needs the cell rates and the
    per-household haircut; ``years/wealth_history.py`` names the pieces.
    """

    baseline: dict[str, Array]
    first_order: dict[str, Array]
    mechanical_ex_saving: dict[str, Array]
    mechanical: dict[str, Array]
    conventional: dict[str, Array]
    mechanical_frame: dict[str, Array]
    mechanical_prices_reform: dict[str, Array]
    mechanical_prices_baseline: dict[str, Array]
    mechanical_wealth_channel: dict[str, Array]
    # The household frame each reform world's tax was calculated on. The
    # published totals report income amounts that are frame columns (wages,
    # dividends, Social Security), read from each world's own frame because
    # the employer-payroll adjustment changes wages and the wealth haircut
    # changes capital income. The baseline sweep runs before the year loop and
    # sums its own totals there.
    raw_frame: dict[str, Array]
    flows_frame: dict[str, Array]
    behavior_frame: dict[str, Array]
    # Inputs to the capital-gains model's host-side solve: the age cells'
    # weighted rate sums and the conversion channel's per-household columns.
    # Each is named for the law and the frame it was measured on
    # (`behavior/gains/builders.py` defines the names). Empty for a scenario without
    # the gains model.
    gains_rate_exports: dict[str, Array]


register_dataclass(
    AnnualResults,
    data_fields=[field.name for field in fields(AnnualResults)],
    meta_fields=[],
)


# The result columns the income tax on gains deemed realized at death is
# added to. It is an income tax bill on a decedent's final return, so it is
# added to the income tax owed and to the income tax owed net of refundable
# credits, and on the payment-timing side it is settled at filing rather than
# withheld through the year.
DEEMED_DEATH_TAX_LANDS_IN = ("liab_iit", "liab_iit_net", "pmt_iit_nonwithheld")

# The frozen gains pass holds turnover at baseline and reads no prices. Every
# behavior builder takes two sets of prices, so this pass is passed empty
# ones.
NO_PRICES: dict[str, Array] = {}


def with_baseline_corporate_rate(frame: Records, baseline_law: Any) -> Records:
    """Add baseline law's corporate rate to the frame the behavior step sees.

    The entity shifting response compares the total tax on a dollar earned in
    corporate form against the tax on the same dollar earned through a
    pass-through, under the reform and under baseline law. The baseline side
    needs baseline law's statutory corporate rate, which the frame does not
    otherwise carry because it is built under the law being scored.

    A law compiled without a corporate rate (a cut-down test law) gets nothing
    added, and the module falls back to its own parameters.
    """
    columns = getattr(baseline_law, "frame_columns", None)
    if not columns or CORPORATE_RATE_COLUMN not in columns:
        return frame
    # The law is held four rows deep, one per filing status. The corporate rate
    # is gathered by status like every other law column.
    filing_index = jnp.asarray(frame["filing_status"], dtype=jnp.int32) - 1
    rate = jnp.asarray(columns[CORPORATE_RATE_COLUMN])
    return {
        **frame,
        BASELINE_CORPORATE_RATE_COLUMN: (
            rate if rate.ndim == 0 else rate[filing_index]
        ),
    }


# The frame column the estate calculation reads to take §2053's deduction for
# the income tax a decedent's final return owes. It holds the tax conditional
# on death, not the expected amount, because the estate calculation is already
# conditional on death.
ESTATE_DEDUCTION_COLUMN = "estate_income_tax_ded"


# The income tax the deemed gain is priced against: what a decedent's final
# return owes net of refundable credits.
DEEMED_DEATH_LIABILITY_FIELD = "liab_iit_net"


def deemed_death_gain_by_evaluation_point(frame: Records) -> tuple[Array, ...]:
    """The deemed gain each extra evaluation point adds to the return.

    Empty for a frame with no deemed gain, which is every scenario without the
    capital-gains model.

    Where the law states an exclusion on a decedent's gain there are two
    points: the exclusion the filer has, and the married amount it would have
    if widowed at death. The two are blended by the chance of widowhood after
    the tax on each is known. A frame that carries only the whole deemed gain
    (no exclusion to spread, or the unit tests' input) has one point.
    """
    from taxmodel.behavior.kg import applier

    if applier.DEEMED_FULL_COLUMN not in frame:
        return ()
    own = frame.get(applier.DEEMED_FULL_OWN_EXCLUSION_COLUMN)
    married = frame.get(applier.DEEMED_FULL_MARRIED_EXCLUSION_COLUMN)
    widowhood = frame.get(applier.WIDOWHOOD_COLUMN)
    if own is None or married is None or widowhood is None:
        return (jnp.asarray(frame[applier.DEEMED_FULL_COLUMN], dtype=jnp.float64),)
    return (
        jnp.asarray(own, dtype=jnp.float64),
        jnp.asarray(married, dtype=jnp.float64),
    )


def the_deemed_death_tax(
    results: TaxResults,
    liability_with_deemed_gain: Sequence[Array],
    frame: Records,
) -> tuple[Array, Array]:
    """The tax on the deemed gain, from the liability at each evaluation point.

    ``results`` is the household's own tax results, and
    ``liability_with_deemed_gain`` holds one liability array per evaluation
    point: what the same household would owe with that point's deemed gain
    added to its realized long-term gains.

    The first return value is the expected amount: the tax the final return
    would owe on the deemed gain, times the household's chance of dying this
    year. The second is the conditional amount, which the estate deducts under
    §2053; the estate calculation is already conditional on death.

    Both are floored at zero. The deemed gain is itself floored at zero per
    asset class (`behavior/kg/applier.py`), so adding it to a return cannot
    lower the income tax, and a negative amount here is rounding noise between
    the two calculator calls.

    With two evaluation points the two branches are the death-gain exclusion
    the filer has and the married amount it would have if widowed, blended by
    the chance of widowhood.
    """
    from taxmodel.behavior.kg import applier

    living = jnp.asarray(results[DEEMED_DEATH_LIABILITY_FIELD], dtype=jnp.float64)

    def tax_at_point(point: int) -> Array:
        return jnp.asarray(
            liability_with_deemed_gain[point], dtype=jnp.float64
        ) - living

    if len(liability_with_deemed_gain) == 1:
        conditional = tax_at_point(0)
    else:
        with_own_exclusion = tax_at_point(0)
        with_married_exclusion = tax_at_point(1)
        widowhood = jnp.asarray(
            frame[applier.WIDOWHOOD_COLUMN], dtype=jnp.float64
        )
        conditional = (
            widowhood * with_married_exclusion
            + (1.0 - widowhood) * with_own_exclusion
        )

    conditional = jnp.maximum(conditional, 0.0)
    mortality = jnp.asarray(
        frame.get(applier.MORTALITY_COLUMN, jnp.zeros_like(living)),
        dtype=jnp.float64,
    )
    return mortality * conditional, conditional


def fold_in_the_deemed_death_tax(
    results: TaxResults,
    frame: Records,
    law: object,
    tax_calculator: TaxCalculator,
    expected_tax: Array,
    estate_deduction: Array,
) -> dict[str, Array]:
    """Add the deemed-death tax to the year's results and recompute the estate.

    The marginal tax rates a behavioral response reads are anchored on the
    surviving taxpayer's liability: they come from the calculator run on the
    households as they are, without the deemed gain.

    The wealth channel does not see the deemed tax. This runs before the
    year's burden is formed, and ``years/burden.py`` subtracts ``liab_deemed``
    from the during-life tax it passes to the channel, because the heirs carry
    that tax through the estate allocation.
    """
    folded = dict(results)
    for name in DEEMED_DEATH_TAX_LANDS_IN:
        if name in folded:
            folded[name] = jnp.asarray(folded[name]) + expected_tax

    # The estate deducts that income tax under §2053 before its own tax is
    # computed, so the estate figures are recomputed with the deduction in
    # place. The estate tax reads frame columns only, so the compiler drops
    # the rest of this second calculator run.
    with_the_deduction = tax_calculator(
        {**frame, ESTATE_DEDUCTION_COLUMN: estate_deduction}, law
    )
    for name in estate_allocator_jax.ESTATE_RESULT_COLUMNS:
        if name in folded and name in with_the_deduction:
            folded[name] = with_the_deduction[name]

    folded[estate_allocator_jax.DEEMED_GAINS_COLUMN] = expected_tax
    return folded


def with_a_zero_deemed_death_tax(results: TaxResults) -> dict[str, Array]:
    """The same results, with a zero column for tax on gains deemed realized at death.

    Baseline law keeps the step-up in basis, so the baseline sweep has no
    deemed gain, but a scenario that runs the capital-gains model has the
    column in its other four worlds and the five worlds' results are
    differenced against each other.
    """
    folded = dict(results)
    if estate_allocator_jax.DEEMED_GAINS_COLUMN not in folded:
        folded[estate_allocator_jax.DEEMED_GAINS_COLUMN] = jnp.zeros_like(
            jnp.asarray(next(iter(folded.values())))
        )
    return folded


def _outside_income(records: Records, name: str) -> Array | None:
    return None if name not in records else jnp.asarray(records[name])


def _advance(
    stock: Array,
    frame: Records,
    assignment: wealth_state.CellAssignment,
    results: TaxResults,
    baseline: TaxResults,
    profile: wealth_state.WealthProfileYear,
) -> tuple[Array, Array]:
    """Next year's shortfall, and the fresh inflow that entered it."""
    weighted_burden = burden_module.burden(
        results,
        baseline,
        _outside_income(frame, "corp_dY_exog"),
        _outside_income(frame, "pr_dY_exog"),
        jnp.asarray(frame["weight"]),
    )
    return wealth_state.advance_stock(stock, weighted_burden, assignment, profile)


def annual_step(
    raw_arrays: Records,
    reform_law: object,
    baseline_law: object,
    baseline_results: TaxResults,
    state: SimulationState,
    profile: wealth_state.WealthProfileYear,
    year: Array,
    frame_builder: FrameBuilder,
    tax_calculator: TaxCalculator,
    price_calculator: PriceCalculator,
    behavior_builder: BehaviorBuilder,
    taxes_and_prices: TaxesAndPrices,
    frozen_gains_builder: BehaviorBuilder | None = None,
    gains_year: Any = None,
    runs_the_gains_model: bool = False,
    gains_rate_pricers: Any = None,
    gains_cell_weights: Any = None,
    behavior_parameters: Any = None,
) -> tuple[SimulationState, AnnualResults]:
    """Compute one year: apply the carried wealth shortfalls, then advance them.

    ``baseline_results`` is the baseline sweep for this year, computed
    separately. The incoming state is D[t]; it changes this year's mechanical
    and behavior frames before tax is calculated. This year's two burdens are
    added after those calculations, producing D[t+1].

    ``taxes_and_prices`` computes the mechanical frame's liabilities and
    marginal rates under the reform's law from one run of the calculators.

    ``gains_year`` is the year's capital-gains tables (the cell tables and the
    year's law on gains at death), passed as data so that one compiled
    program serves every gains scenario of the same shape. It is passed to
    the two builders and read nowhere else here.

    ``gains_rate_pricers`` measures the rates a gains scenario's host-side
    solve reads (`behavior/gains/builders.py` builds it), and ``gains_cell_weights``
    is this year's slice of the dollars each household brings to its age
    cell, the weights those rates are averaged over before they leave the
    compiled program. When the pricers are given, the year also prices the
    reform's gains rate on the mechanical frame before the frozen gains pass,
    which is the rate the retiming overlay reads.
    """
    # The conventional world's corporate incidence comes from the corporate
    # revenue change with the tax base eroding (``reform_law.corporate``); the
    # two mechanical worlds' comes from the change with the base fixed. The
    # mechanical law differs from the reform's law only in that.
    mechanical_law = law_by_year.for_the_mechanical_worlds(reform_law)

    def apply_corporate_gain_quantity_margin(
        frame: Records, law: object
    ) -> dict[str, Array]:
        """Apply the corporate sale-volume margin after the gains applier."""
        corporate = getattr(law, "corporate", None)
        gain_quantity_factor = getattr(corporate, "gain_quantity_factor", None)
        if not runs_the_gains_model or gain_quantity_factor is None:
            return dict(frame)
        return income_adjustments.apply_corporate_gain_quantity_margin(
            frame, gain_quantity_factor
        )

    def taxed(frame: Records, evaluate) -> tuple[dict[str, Array], ...]:
        """One world's results, including the tax on gains deemed at death.

        ``evaluate(frame, law)`` returns a tuple of dictionaries: that world's
        tax results first, then the mechanical world's marginal rates when
        that world asks for both in one call.

        Where the frame carries a deemed gain, the calculator runs again once
        per evaluation point, on the same households with that point's deemed
        gain added to their realized long-term gains, and the tax on the
        deemed gain is the difference in liability. Only the liability column
        of those extra runs is kept. A law that keeps the step-up in basis
        deems a gain of zero. Where the frame carries no deemed gain (every
        scenario without the capital-gains model), the calculator runs once.
        """
        outputs = evaluate(frame, reform_law)
        added_gain = deemed_death_gain_by_evaluation_point(frame)
        if not added_gain:
            return outputs
        realized_gain = jnp.asarray(frame["kg_lt"], dtype=jnp.float64)
        liability_with_deemed_gain = tuple(
            evaluate({**frame, "kg_lt": realized_gain + added}, reform_law)[0][
                DEEMED_DEATH_LIABILITY_FIELD
            ]
            for added in added_gain
        )
        expected_tax, estate_deduction = the_deemed_death_tax(
            outputs[0], liability_with_deemed_gain, frame
        )
        return (
            fold_in_the_deemed_death_tax(
                outputs[0],
                frame,
                reform_law,
                tax_calculator,
                expected_tax,
                estate_deduction,
            ),
            *outputs[1:],
        )

    def just_the_taxes(frame: Records, law: object) -> tuple[dict[str, Array]]:
        return (tax_calculator(frame, law),)

    # The three worlds below the conventional one carry the frozen gains pass:
    # the treatment of gains at death changes while realization behavior is
    # held at baseline. That is law arithmetic, not a behavioral response, so
    # it is included in the first-order world. The first-order world's pass
    # leaves out the corporate markdown of share prices; the two mechanical
    # worlds' pass subtracts it from the stock of unrealized gain. A scenario
    # with no gains model has no frozen pass (the builder is None) and these
    # three frames are the frames the frame builder made.
    raw_frame = frame_builder(raw_arrays, reform_law, year, apply_adjusters=False)
    if frozen_gains_builder is not None:
        raw_frame = frozen_gains_builder(
            raw_frame,
            gains_year,
            NO_PRICES,
            NO_PRICES,
            behavior_parameters,
            with_corporate_markdown=False,
        )
    (first_order,) = taxed(raw_frame, just_the_taxes)

    # The flows frame is built twice, once from each corporate incidence path.
    # The mechanical worlds read the fixed-base frame with the frozen pass
    # applied. The conventional sweep below starts from the eroded-base frame
    # without the frozen pass: the behavioral cell table already includes the
    # death treatment, the routing of decedents' gains to heirs included, so a
    # conventional world built on the frozen frame would count it twice.
    bare_flows_frame = frame_builder(
        raw_arrays, mechanical_law, year, apply_adjusters=True
    )
    conventional_flows_frame = frame_builder(
        raw_arrays, reform_law, year, apply_adjusters=True
    )

    def with_frozen_gains(frame: Records) -> dict[str, Array]:
        """The frozen gains pass with the corporate markdown, then the
        sale-volume margin.

        The corporate channel changes realized gains in two ways. The markdown
        of share prices lowers the gain on each share sold; the mechanical
        frozen table carries it. The cut in payouts lowers the number of
        shares the company buys back; the quantity margin scales sales for it.
        Gains at death are unaffected by the quantity margin. Neither is
        applied to the first-order frame.
        """
        if frozen_gains_builder is None:
            return dict(frame)
        frame = frozen_gains_builder(
            frame,
            gains_year,
            NO_PRICES,
            NO_PRICES,
            behavior_parameters,
            with_corporate_markdown=True,
        )
        return apply_corporate_gain_quantity_margin(frame, mechanical_law)

    # The mechanical-ex-saving world carries no wealth shortfall. The
    # mechanical world applies the shortfall carried into this year first and
    # runs the frozen gains pass on the drawn-down balance sheet, in the same
    # order as the conventional world, so the gain deemed realized at death is
    # measured after the drawdown.
    flows_frame = with_frozen_gains(bare_flows_frame)
    (mechanical_ex_saving,) = taxed(flows_frame, just_the_taxes)
    mechanical_assignment = wealth_state.assign_cells(bare_flows_frame, profile)
    drawn_down_frame = wealth_state.apply_carried_stock(
        bare_flows_frame, state.mechanical_wealth, mechanical_assignment, profile
    )
    mechanical_frame = with_frozen_gains(drawn_down_frame)
    mechanical, mechanical_prices_reform = taxed(mechanical_frame, taxes_and_prices)
    mechanical_prices_baseline = price_calculator(mechanical_frame, baseline_law)
    next_mechanical, mechanical_cell_inflow = _advance(
        state.mechanical_wealth,
        mechanical_frame,
        mechanical_assignment,
        mechanical,
        baseline_results,
        profile,
    )
    mechanical_wealth_channel = wealth_history.year_wealth_channel_state(
        mechanical_frame,
        mechanical_assignment,
        mechanical_cell_inflow,
        profile,
    )

    # The second sweep starts from the eroded-base flows frame (before any
    # frozen gains pass) and its own state. Its behavior chain applies the
    # behavioral cell table, which already contains the frozen pass's law
    # arithmetic.
    conventional_assignment = wealth_state.assign_cells(
        conventional_flows_frame, profile
    )
    conventional_frame = wealth_state.apply_carried_stock(
        conventional_flows_frame,
        state.conventional_wealth,
        conventional_assignment,
        profile,
    )
    behavior_frame = behavior_builder(
        with_baseline_corporate_rate(conventional_frame, baseline_law),
        gains_year,
        mechanical_prices_reform,
        mechanical_prices_baseline,
        behavior_parameters,
    )
    # The sale-volume margin is applied after the conventional gains applier
    # and the behavioral modules have written realized gains. Both corporate
    # changes to realized gains are added to `corp_dY_gains`, which only the
    # distribution table reads.
    behavior_frame = income_adjustments.record_corporate_markdown_gains(
        behavior_frame
    )
    behavior_frame = apply_corporate_gain_quantity_margin(behavior_frame, reform_law)
    # The tax on gains deemed realized at death is priced after the prices
    # above were measured, so the rates a behavioral response reads are
    # anchored on the surviving taxpayer's liability.
    (conventional,) = taxed(behavior_frame, just_the_taxes)

    # The rates the capital-gains model's host-side solve reads out of this
    # year, summed into age cells before they leave. Two frames carry them:
    # the mechanical frame under both laws, and the same frame before the
    # frozen gains pass under the reform's law alone, which feeds the retiming
    # overlay.
    gains_rate_exports: dict[str, Array] = {}
    if gains_rate_pricers is not None:
        pre_injection_frame = drawn_down_frame
        gains_rate_exports = gains_rate_pricers.year_exports(
            cell_weights=gains_cell_weights,
            mechanical_frame=mechanical_frame,
            mechanical_prices_reform=mechanical_prices_reform,
            mechanical_prices_baseline=mechanical_prices_baseline,
            pre_injection_frame=pre_injection_frame,
            reform_law=reform_law,
            mechanical_results=mechanical,
        )
    next_conventional, _conventional_cell_inflow = _advance(
        state.conventional_wealth,
        behavior_frame,
        conventional_assignment,
        conventional,
        baseline_results,
        profile,
    )

    return (
        SimulationState(next_mechanical, next_conventional),
        AnnualResults(
            baseline=dict(baseline_results),
            first_order=first_order,
            mechanical_ex_saving=mechanical_ex_saving,
            mechanical=mechanical,
            conventional=conventional,
            mechanical_frame=mechanical_frame,
            mechanical_prices_reform=mechanical_prices_reform,
            mechanical_prices_baseline=mechanical_prices_baseline,
            mechanical_wealth_channel=mechanical_wealth_channel,
            raw_frame=raw_frame,
            flows_frame=flows_frame,
            behavior_frame=behavior_frame,
            gains_rate_exports=gains_rate_exports,
        ),
    )
