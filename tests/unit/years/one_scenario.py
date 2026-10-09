"""Run one scenario through the batch runner, for tests of the year loop.

Production runs many scenarios side by side through `years/loop.run_scenarios`.
A test of the year loop usually has one scenario and placeholder
calculators, and no interest in the batching machinery; this builds the batch
of one for it and hands back that one lane's results. The placeholder
frame builders beside it are for the same tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from taxmodel.reporting import assemble, heir_inputs
from taxmodel.reporting import detail as detail_output
from taxmodel.years import loop, one_year, wealth_state, worlds

Array = Any


def passthrough_frame_builder(
    raw_arrays: Any,
    law: object,
    year: Array,
    *,
    apply_adjusters: bool,
) -> dict[str, Array]:
    """A frame builder that applies no frame rules: it copies the raw arrays.

    Tests of the year loop's plumbing use this instead of the production
    builder (`frame/build_frame.make_frame_builder`), whose rules need columns
    a placeholder population does not carry. `worlds.FrameBuilder` says what
    the production builder must do.
    """
    del law, year, apply_adjusters
    return dict(raw_arrays)


def make_passthrough_frame_builder(
    adjuster: Callable[[Any, object, Array], dict[str, Array]] | None = None,
) -> worlds.FrameBuilder:
    """The copying builder above, with an optional adjuster for the flows frame."""

    def build(
        raw_arrays: Any, law: object, year: Array, *, apply_adjusters: bool
    ) -> dict[str, Array]:
        frame = dict(raw_arrays)
        if apply_adjusters and adjuster is not None:
            frame = adjuster(frame, law, year)
        return frame

    return build


def unchanged_frame(
    records: Any,
    gains_year: Any,
    reform_prices: Any,
    baseline_prices: Any,
    behavior_parameters: Any = None,
) -> dict[str, Array]:
    """A behavior step that changes nothing, for tests with placeholder calculators.

    The production behavior step is the pinned module chain
    (`behavior/loader.plain_behavior_builder`), whose modules read columns a
    placeholder calculator does not produce. Tests of the year loop's plumbing
    use this instead, so what they test is the loop and not the modules.
    """
    del gains_year, reform_prices, baseline_prices, behavior_parameters
    return dict(records)


def taxes_then_prices(
    tax_calculator: worlds.TaxCalculator, price_calculator: worlds.PriceCalculator
) -> worlds.TaxesAndPrices:
    """Tax and price the mechanical frame in two calls, for placeholder calculators.

    Production's version (`scenario/prepare.make_calculators`) runs the
    calculators once and takes the rates as derivatives of that same run. A
    test with placeholder calculators has no derivatives to share, so it asks
    for the two separately.
    """

    def both(frame, law):
        return (tax_calculator(frame, law), price_calculator(frame, law))

    return both


def run_one_scenario(
    population: Any,
    reform_laws: object,
    baseline_laws: object,
    wealth_profiles: wealth_state.WealthProfileYear,
    *,
    years: Sequence[int] = one_year.DEFAULT_YEARS,
    frame_builder: worlds.FrameBuilder = passthrough_frame_builder,
    tax_calculator: worlds.TaxCalculator | None = None,
    price_calculator: worlds.PriceCalculator | None = None,
    behavior_builder: worlds.BehaviorBuilder = unchanged_frame,
    initial_state: worlds.SimulationState | None = None,
    detail_request: detail_output.DetailRequest | None = None,
    heir_structure: heir_inputs.HeirStructure | None = None,
    taxes_and_prices: worlds.TaxesAndPrices | None = None,
    frozen_gains_builder: worlds.BehaviorBuilder | None = None,
    gains_tables: object = None,
    runs_the_gains_model: bool = False,
    gains_rate_pricers: object = None,
    gains_cell_weights: object = None,
    behavior_parameters: object = None,
) -> assemble.SimulationRun:
    """One scenario over the projection window, as a batch of one lane.

    ``reform_laws`` and ``baseline_laws`` are bound annual laws already stacked
    on their first axis. The arguments mean what they mean on
    `loop.ScenarioProgram` and `loop.BatchScenario`.
    """
    requested_years = one_year.check_years(years)
    if wealth_profiles.r_total.shape[0] != len(requested_years):
        raise ValueError(
            "wealth profiles have "
            f"{wealth_profiles.r_total.shape[0]} years, expected "
            f"{len(requested_years)}"
        )
    if tax_calculator is None or price_calculator is None:
        raise ValueError(
            "a simulation needs its tax calculator and its price calculator; "
            "production builds them with scenario/prepare.make_calculators"
        )

    scenario = loop.BatchScenario(
        name="the one scenario",
        program=loop.ScenarioProgram(
            # One scenario has nothing to be grouped with, so the label that
            # decides which scenarios share a compiled program says only that.
            switch_fingerprint="a batch of one scenario",
            frame_builder=frame_builder,
            tax_calculator=tax_calculator,
            price_calculator=price_calculator,
            behavior_builder=behavior_builder,
            detail_request=detail_request,
            taxes_and_prices=(
                taxes_then_prices(tax_calculator, price_calculator)
                if taxes_and_prices is None
                else taxes_and_prices
            ),
            frozen_gains_builder=frozen_gains_builder,
            runs_the_gains_model=runs_the_gains_model,
            gains_rate_pricers=gains_rate_pricers,
        ),
        reform_laws=reform_laws,
        baseline_laws=baseline_laws,
        wealth_profiles=wealth_profiles,
        initial_state=initial_state,
        gains_tables=gains_tables,
        behavior_parameters=behavior_parameters,
    )
    batched = loop.run_scenarios(
        population,
        [scenario],
        scenarios_per_program=1,
        years=requested_years,
        heir_structure=heir_structure,
        gains_cell_weights=gains_cell_weights,
    )
    return batched.runs[scenario.name]
