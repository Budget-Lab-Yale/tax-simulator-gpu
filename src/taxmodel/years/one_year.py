"""One simulated year of one scenario.

`one_year_step` runs the baseline sweep, then the four reform worlds
(`worlds.annual_step`: first_order, mechanical_ex_saving, mechanical,
conventional, then the advance of the wealth stocks), and reduces them to
the columns the run reports. A plain Python loop in `years/loop.py` drives
the years, so each compiled program holds one year.

The report sums taken inside the compiled year (population totals, the
heirs' share of the taxes at death, the distribution table, and the
marginal-rate table's accumulators) are in `reporting/device_sums.py`. This
module holds the year step and the host-side helpers that shape a run's
inputs for it. `reporting/assemble.py` turns a finished run into tables.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from taxmodel.frame import prepare, steps_jax
from taxmodel.frame.population import PopulationSource
from taxmodel.reporting import detail as detail_output
from taxmodel.reporting import device_sums, heir_inputs
from taxmodel.reporting.device_sums import ReportColumns
from taxmodel.taxes import law_for_calculators
from taxmodel.years import wealth_arithmetic, wealth_state, worlds

if TYPE_CHECKING:
    # Only for the annotation below; `years/loop.py` imports this module.
    from taxmodel.years.loop import ScenarioProgram

Array = jnp.ndarray

DEFAULT_FIRST_YEAR = 2026
DEFAULT_LAST_YEAR = 2055
DEFAULT_YEARS = tuple(range(DEFAULT_FIRST_YEAR, DEFAULT_LAST_YEAR + 1))


def law_read_for_each_household(law: object, filing_status: Array) -> object:
    """Read the law parameters for each household's filing status under this law.

    The status is the one this law puts the household in, which can differ
    from the status on the frame: see `filing_status_the_law_is_read_for`.

    A law object (``law_for_calculators.YearLaw``, or the
    ``law_by_year.AnnualLaw`` that carries one) holds each parameter as its
    four filing-status values and has a ``for_records`` method that reads them
    for a population. The runner tests pass a plain array in place of the law;
    an array is returned unchanged.
    """
    read = getattr(law, "for_records", None)
    if read is None:
        return law
    return read(filing_status_the_law_is_read_for(law, filing_status))


def filing_status_the_law_is_read_for(
    law: object, filing_status: Array
) -> Array:
    """The filing status each household is taxed under this law.

    A law that repeals head-of-household filing status taxes those filers as
    single filers: the single standard deduction, the single rate brackets,
    and every other single-filer parameter. The recode happens before the law
    is read for each household, so that the repeal changes the parameters and
    not only the frame's filing-status column.

    A law object with no wide law table behind it (the runner tests pass a
    plain array in place of a law) repeals nothing, and the raw status is used.
    """
    columns = getattr(law, "frame_columns", None)
    if not columns:
        return filing_status
    repeal_hoh = steps_jax.head_of_household_repeal_switch(columns)
    return steps_jax.filing_status_under_law(filing_status, repeal_hoh)


def stack_wealth_profiles(
    saving_shares: Array,
    transition: Array,
    r_total: Array,
    *,
    fmax: float | Array,
    pass_through_weight: float | Array,
) -> wealth_state.WealthProfileYear:
    """Repeat fixed profile tables and attach one total return per run year."""
    saving_shares = jnp.asarray(saving_shares, dtype=jnp.float64)
    transition = jnp.asarray(transition, dtype=jnp.float64)
    r_total = jnp.asarray(r_total, dtype=jnp.float64)
    if saving_shares.ndim != 2:
        raise ValueError(
            "saving shares must have age on rows and percentiles on columns"
        )
    n_ages, n_bins = saving_shares.shape
    expected_ages = len(wealth_arithmetic.age_grid())
    if n_ages != expected_ages:
        raise ValueError(
            f"saving shares have {n_ages} ages; the wealth grid has {expected_ages}"
        )
    if transition.shape != (n_bins, n_bins):
        raise ValueError(
            f"the transition is {transition.shape}, expected {(n_bins, n_bins)}"
        )
    if r_total.ndim != 1:
        raise ValueError("r_total must have one value per run year")
    n_years = r_total.shape[0]
    repeat = lambda value: jnp.broadcast_to(value, (n_years, *value.shape))
    ages = jnp.asarray(wealth_arithmetic.age_grid())
    aging = wealth_arithmetic.build_aging_matrix(n_ages)
    return wealth_state.WealthProfileYear(
        saving_shares=repeat(saving_shares),
        transition=repeat(transition),
        r_total=r_total,
        ages=repeat(ages),
        aging=repeat(aging),
        fmax=jnp.broadcast_to(jnp.asarray(fmax), (n_years,)),
        pass_through_weight=jnp.broadcast_to(
            jnp.asarray(pass_through_weight), (n_years,)
        ),
    )


def population_arrays_for_years(
    population: PopulationSource, years: Sequence[int]
) -> dict[str, Array]:
    """The population block for the requested years, as device arrays.

    Every scenario reads the same block, so a batch of scenarios loads it once.
    The population is asked for one year at a time and the years are stacked
    here, so a run whose population is read from disk holds one year's columns
    at a time.

    Whole-number columns are copied to the accelerator in their 16-bit storage
    form and widened to float64 there. The widening is exact.
    """
    years = tuple(int(year) for year in years)
    year_slices = [population.year(year) for year in years]
    valid = np.stack([one.valid for one in year_slices])
    n_years = len(years)
    n_slots = population.universe.n_slots

    def stacked(name: str) -> np.ndarray:
        """One value column for every requested year, in its stored type."""
        return np.stack([one.values[name] for one in year_slices])

    arrays: dict[str, Array] = {}
    for name in population.field_names:
        if name == "id":
            value = np.broadcast_to(population.universe.ids, (n_years, n_slots))
        elif name == "year":
            value = np.broadcast_to(
                np.asarray(years, dtype=np.float64)[:, None], (n_years, n_slots)
            )
        elif name in population.draws:
            value = np.where(valid, population.draws[name][None, :], 0.0)
        elif prepare.is_narrow_whole_number_column(name):
            narrow = jnp.asarray(stacked(name))
            arrays[name] = jnp.where(
                narrow == prepare.MISSING_WHOLE_NUMBER,
                jnp.nan,
                narrow.astype(jnp.float64),
            )
            continue
        else:
            value = stacked(name)
        arrays[name] = jnp.asarray(value)
    arrays["valid"] = jnp.asarray(valid)
    # Law parameters are read by filing-status code inside the compiled
    # program, so a code outside 1 through 4 would read the wrong row of every
    # law table. Checked once here, on the host.
    law_for_calculators.check_filing_status_codes(stacked("filing_status"))
    return arrays


class DeviceOutputs(NamedTuple):
    """One scenario's results for the whole projection, before host-side assembly.

    Cut out of its batch on the host and passed to
    `reporting/assemble.assemble_simulation_run`. Every entry has the
    projection years on its first axis.
    """

    totals_by_cell: dict[str, dict[str, Array]]
    mtr_table_sums_by_year: dict[str, Array] | None
    wealth_channel_history: dict[str, Array]
    states_after_year: worlds.SimulationState
    # The per-household columns a detail request asked for, every projection
    # year; the host drops the years the request did not want. Empty when
    # nothing was requested.
    detail_by_year: dict[tuple[str, str], Array]
    # The heir allocation's balancing totals each year, one
    # [year, world, ladder, total] array under the key `estate_allocation`.
    # The two ladders are the two taxes charged at death, in
    # `TAX_AT_DEATH_LADDERS` order, and the totals are named by
    # `reporting/estate_allocator_jax.DIAGNOSTIC_NAMES`. The host checks them
    # after the run. Empty when the run allocated no tax at death.
    estate_allocation_by_year: dict[str, Array]
    # Inputs to the capital-gains model's host-side solve, every projection
    # year: the age cells' weighted rate sums and the conversion channel's
    # per-household columns. Empty on runs without the gains model.
    gains_rate_exports_by_year: dict[str, Array]
    # The distribution table, [year, corporate spread, tax row, group,
    # column] as `reporting/device_sums.distribution_table_for_year` lays out
    # each year. None when the calculator produced none of the columns the
    # table reads.
    distribution_table_by_year: Array | None = None


def zero_initial_state(
    wealth_profiles: wealth_state.WealthProfileYear,
) -> worlds.SimulationState:
    """Zero wealth shortfalls, shaped like the profile's age-by-percentile grid."""
    state_shape = wealth_profiles.saving_shares.shape[1:]
    zero = jnp.zeros(state_shape, dtype=jnp.float64)
    return worlds.SimulationState(zero, zero)


class BaselineYearOutputs(NamedTuple):
    """One year of baseline law and the inputs the reform worlds take from it.

    The baseline sweep is current law with no reform and no behavioral
    response. It runs first because the rest of the year needs four of its
    results: the household amounts the distribution groups are cut on, the
    rankings every world is reported on, the baseline mortality, and the
    baseline world's income and tax measures for the distribution table,
    which are formed here because this is where the baseline frame exists.
    """

    tax_result: Mapping[str, Array]
    distribution_households: dict[str, Array]
    rank_labels: dict[str, Array]
    reported_columns: dict[str, Array]
    mortality: dict[str, Array]
    world_measures: dict[str, Array]
    corporate_bases: dict[str, Array]


class YearOutputs(NamedTuple):
    """One projection year's results for all five worlds.

    The driver stacks these to add the year axis.

    The year produces per-household columns and the wealth state; the driver
    forms every reported table from those columns in a separate program as
    soon as the year is done, then drops the columns.
    """

    # Each world's per-household columns, selected by
    # :func:`reported_columns_for_totals`. The driver passes them to
    # :func:`population_totals_and_counts` and keeps only the totals.
    reported_columns_by_cell: dict[str, dict[str, Array]]
    # The columns the three report programs read; see :class:`ReportColumns`.
    report_columns: ReportColumns
    wealth_channel_history: dict[str, Array]
    state_after_year: worlds.SimulationState
    detail: dict[tuple[str, str], Array]
    gains_rate_exports: dict[str, Array]


def baseline_year_step(
    program: ScenarioProgram,
    raw: Mapping[str, Array],
    baseline_law: object,
    year: Array,
    heir: Mapping[str, Array],
) -> BaselineYearOutputs:
    """Run one year under baseline law and collect the inputs the reform worlds need."""
    law = law_read_for_each_household(baseline_law, raw["filing_status"])
    frame = program.frame_builder(raw, law, year, apply_adjusters=False)
    # Baseline law keeps the step-up in basis, so there is no deemed death
    # gain here. When the scenario runs the gains model the column is written
    # as zeros, so that every world's results have the same columns as the
    # baseline they are differenced against. Otherwise nothing is written and
    # the heir allocation skips its deemed-gains branch.
    tax_result = program.tax_calculator(frame, law)
    if program.runs_the_gains_model:
        tax_result = worlds.with_a_zero_deemed_death_tax(tax_result)
    households = device_sums.distribution_households(raw, frame, tax_result)
    return BaselineYearOutputs(
        tax_result=tax_result,
        distribution_households=households,
        rank_labels=device_sums.baseline_rank_labels(households, heir),
        # The baseline world's reported columns are selected here, while its
        # frame exists.
        reported_columns=device_sums.reported_columns_for_totals(tax_result, raw, frame),
        # The baseline chances of dying, which the heir allocation needs. They
        # are on this sweep's frame, not on the results passed to the reform
        # step.
        mortality={
            name: frame[name]
            for name in (
                device_sums.ESTATE_MORTALITY_COLUMN,
                device_sums.DEEMED_GAINS_MORTALITY_COLUMN,
            )
            if name in frame
        },
        world_measures=device_sums.baseline_world_measures(tax_result, frame),
        corporate_bases=device_sums.baseline_corporate_bases(tax_result, frame),
    )


def reform_year_step(
    program: ScenarioProgram,
    state: worlds.SimulationState,
    raw: Mapping[str, Array],
    reform_law: object,
    baseline_law: object,
    baseline: BaselineYearOutputs,
    heir: Mapping[str, Array],
    profile: wealth_state.WealthProfileYear,
    year: Array,
    gains_year: object,
    gains_cell_weights_year: object,
    behavior_parameters: object,
) -> tuple[worlds.SimulationState, YearOutputs]:
    """One projection year's four reform worlds, reduced to the columns the run reports.

    The outputs are the new wealth state, the columns the population totals
    are formed from, the columns the three report programs read, and any
    per-household columns the run asked for.
    """
    # Each law is read for the filing status that law puts the household in.
    # Under a reform repealing head-of-household status those filers are read
    # as single filers, while baseline law still reads them as heads of
    # household.
    reform_law = law_read_for_each_household(reform_law, raw["filing_status"])
    baseline_law = law_read_for_each_household(baseline_law, raw["filing_status"])
    next_state, year_results = worlds.annual_step(
        raw,
        reform_law,
        baseline_law,
        baseline.tax_result,
        state,
        profile,
        year,
        program.frame_builder,
        program.tax_calculator,
        program.price_calculator,
        program.behavior_builder,
        program.taxes_and_prices,
        program.frozen_gains_builder,
        gains_year=gains_year,
        runs_the_gains_model=program.runs_the_gains_model,
        gains_rate_pricers=program.gains_rate_pricers,
        gains_cell_weights=gains_cell_weights_year,
        behavior_parameters=behavior_parameters,
    )
    # The columns the published totals are formed from, selected while the
    # per-household results still exist. The baseline world's were selected in
    # the sweep above. The summing happens in the year's second compiled
    # program (:func:`population_totals_and_counts`).
    reported_columns = {
        world: device_sums.reported_columns_for_totals(
            getattr(year_results, world),
            raw,
            getattr(year_results, device_sums.FRAME_FIELD_BY_WORLD[world]),
        )
        for world in device_sums.REFORM_WORLDS
    }
    # The requested per-household columns for this year. The years the
    # request did not ask for are dropped on the host afterwards.
    requested_detail = (
        {}
        if program.detail_request is None
        else detail_output.one_year(
            program.detail_request,
            device_sums.detail_columns_available(program.detail_request, year_results),
        )
    )
    return next_state, YearOutputs(
        reported_columns_by_cell={
            device_sums.BASELINE_WORLD: baseline.reported_columns,
            **reported_columns,
        },
        report_columns=device_sums.report_columns_for_the_year(
            baseline, heir, year_results, profile.pass_through_weight
        ),
        wealth_channel_history=year_results.mechanical_wealth_channel,
        state_after_year=next_state,
        detail=requested_detail,
        gains_rate_exports=year_results.gains_rate_exports,
    )


def one_year_step(
    program: ScenarioProgram,
    state: worlds.SimulationState,
    raw: Mapping[str, Array],
    reform_law: object,
    baseline_law: object,
    heir: Mapping[str, Array],
    profile: wealth_state.WealthProfileYear,
    year: Array,
    gains_year: object,
    gains_cell_weights_year: object,
    behavior_parameters: object,
) -> tuple[worlds.SimulationState, YearOutputs]:
    """One projection year: the baseline sweep, then the four reform worlds.

    This is the first and largest of the five programs a projection year
    compiles. It holds one calendar year, and the same compiled program serves
    every year of every scenario in the run. The other four programs are the
    population totals and the three report programs, each of which reads the
    columns this one produces.
    """
    baseline = baseline_year_step(program, raw, baseline_law, year, heir)
    return reform_year_step(
        program,
        state,
        raw,
        reform_law,
        baseline_law,
        baseline,
        heir,
        profile,
        year,
        gains_year,
        gains_cell_weights_year,
        behavior_parameters,
    )


def heir_arrays(
    structure: heir_inputs.HeirStructure | None, years: Sequence[int]
) -> dict[str, Array]:
    """The heir structure for the run's years, as device arrays.

    Bequests are measured before tax, so the heir structure does not vary by
    scenario and every scenario reads the same block. An empty result means
    the run has no heir structure and the distribution table reports no tax at
    death.
    """
    if structure is None:
        return {}
    wanted = tuple(int(year) for year in years)
    rows = [structure.years.index(year) for year in wanted]
    return {
        "probability_of_inheriting": jnp.asarray(
            structure.probability_of_inheriting[rows]
        ),
        "inheritance": jnp.asarray(structure.inheritance[rows]),
    }


def check_years(years: Sequence[int]) -> tuple[int, ...]:
    """Check that the requested projection years are consecutive and ascending."""
    requested = tuple(int(year) for year in years)
    if not requested:
        raise ValueError("a simulation needs at least one year")
    if len(set(requested)) != len(requested):
        raise ValueError("simulation years cannot repeat")
    # The wealth state carries from each year to the next. A gap or a reversal
    # would apply one year's aging and growth across a longer or a backward
    # span and drop the skipped years' burden from the stock.
    if any(later - earlier != 1 for earlier, later in itertools.pairwise(requested)):
        raise ValueError(
            "simulation years must be consecutive and ascending; the wealth "
            f"state carries year to year and cannot jump: {requested}"
        )
    return requested
