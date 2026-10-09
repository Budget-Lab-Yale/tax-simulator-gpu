"""The wired projection: real frame steps, real law, real calculator chain.

Everything here runs on three hand-written households over two years, with the
law compiled from the repository's own configuration tree. The point is not the
dollar amounts -- the households are made up -- but that the pieces fit: that
the adapter builds the frames the annual step's contract describes, that the
runner still satisfies the timing properties the placeholder frame builder
proved, and that the wealth bathtub's history comes back in the shape the
diagnostics tables require.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from taxmodel.frame import build_frame, prepare, steps_jax
from taxmodel.frame.population import Population, population_from_frames
from taxmodel.reporting import distribution, etr_measures, wealth_diagnostics
from taxmodel.reporting.assemble import assemble_wealth_history
from taxmodel.scenario import law_by_year, provenance
from taxmodel.scenario.prepare import make_calculators, stacked_annual_laws
from taxmodel.taxes import statutory_order
from taxmodel.years import one_year, wealth_arithmetic, world_names
from tests.tolerances import assert_agrees_to_the_cent
from tests.unit.years import grid_differences
from tests.unit.years.one_scenario import run_one_scenario

YEARS = (2026, 2027)
# The two income changes from outside the tax system that the flows branch
# adds: the corporate incidence path and the employer-payroll wage change.
FLOWS_DOLLAR_COLUMNS = ("corp_dY_exog", "pr_dY_exog")
N_AGES = len(wealth_arithmetic.age_grid())
# Four percentile bins keep the cell grid small enough to read in a failure
# message while still splitting the three households apart.
N_BINS = 4


@pytest.fixture(autouse=True)
def release_compiled_programs_after_each_test():
    """Do not retain a separate full projection program for every test."""
    yield
    jax.clear_caches()

# The three households. Each is written out on purpose: one wage earner with a
# small balance sheet, one older couple holding most of the wealth, and one
# household with pass-through income and a state tax bill, which is what makes
# the state-tax workaround step do something.
#
# All three file a return, and each says so. 1040 quantities are counted
# for filers only, so a household that leaves the filing flag at zero
# contributes nothing to income tax however much it owes, and a test asking
# whether a reform moved income tax would compare zero with zero.
HOUSEHOLDS: tuple[dict[str, float], ...] = (
    {
        "id": 1,
        "filing_status": 1,
        "filer": 1.0,
        "age1": 35.0,
        "weight": 1_000.0,
        "wages_base1": 60_000.0,
        "value.cash": 20_000.0,
    },
    {
        "id": 2,
        "filing_status": 2,
        "filer": 1.0,
        "age1": 68.0,
        "age2": 66.0,
        "weight": 500.0,
        "div_ord": 40_000.0,
        "div_pref": 60_000.0,
        "gross_ss": 30_000.0,
        "value.equities": 4_000_000.0,
        "value.primary_home": 900_000.0,
        "value.primary_mortgage": 100_000.0,
        "q_death1": 0.02,
        "q_death2": 0.01,
    },
    {
        "id": 3,
        "filing_status": 4,
        "filer": 1.0,
        "age1": 50.0,
        "weight": 800.0,
        "n_dep": 2.0,
        "n_dep_eitc": 2.0,
        "dep_age1": 10.0,
        "dep_age2": 14.0,
        "dep_ssn1": 1.0,
        "dep_ssn2": 1.0,
        "dep_ctc1": 1.0,
        "dep_ctc2": 1.0,
        "part_active": 200_000.0,
        "sstb_part": 0.0,
        "salt_inc_sales": 25_000.0,
        "value.pass_throughs": 1_500_000.0,
    },
)


def _every_column() -> tuple[str, ...]:
    """Every record column the frame steps and the calculator chain read."""
    seen: list[str] = []
    for name in (
        *build_frame.FRAME_STEP_RECORD_COLUMNS,
        *statutory_order.RECORD_COLUMNS,
        *statutory_order.ESTATE_RECORD_COLUMNS,
        "kg_lt_basis",
        "kg_1250",
        "kg_collect",
        "rent",
        "rent_loss",
        "estate",
        "estate_loss",
        "txbl_pens_dist",
        # The population columns the distribution table's Haig-Simons income reads.
        *etr_measures.HAIG_SIMONS_FRAME_COLUMNS,
        *prepare.DRAW_COLUMNS,
    ):
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def synthetic_population() -> Population:
    """The three households in both years, on one set of stable slots."""
    columns = _every_column()
    frames: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        rows = []
        for household in HOUSEHOLDS:
            row = {name: 0.0 for name in columns}
            # A missing spouse age and a missing holding period both mean
            # something to the frame steps, so they start missing rather than
            # at zero.
            row["age2"] = np.nan
            row["sstb_part"] = np.nan
            row["sstb_scorp"] = np.nan
            row["kg_lt_years_held"] = np.nan
            # The random draws are the same for a household in every year.
            for draw in prepare.DRAW_COLUMNS:
                row[draw] = 0.5
            row.update(household)
            rows.append(row)
        frames[year] = pd.DataFrame(rows)
    universe = prepare.build_universe(
        {year: np.array([h["id"] for h in HOUSEHOLDS], dtype=np.int64) for year in YEARS}
    )
    return population_from_frames(frames, universe)


@pytest.fixture(scope="module")
def scenario() -> provenance.Scenario:
    """Baseline law compiled over the two test years, from the config tree."""
    return provenance.load_scenario(YEARS)


@pytest.fixture(scope="module")
def population() -> Population:
    return synthetic_population()


def wealth_profiles(saving_share: float, return_rate: float = 0.04):
    """A flat saving surface and a persistent percentile transition."""
    return one_year.stack_wealth_profiles(
        jnp.full((N_AGES, N_BINS), saving_share),
        jnp.eye(N_BINS),
        jnp.full(len(YEARS), return_rate),
        fmax=0.9,
        pass_through_weight=0.2,
    )


def wire(scenario, reform: provenance.Scenario | None = None):
    """The adapter, the two stacked laws, and the calculators, as scenario/prepare wires them."""
    reform_scenario = scenario if reform is None else reform
    reform_series, baseline_series = stacked_annual_laws(
        reform_scenario, scenario, YEARS
    )
    # The behavioral prices are left out: these tests score no behavioral
    # response, and the gradient passes are what would make them expensive.
    taxes, _prices, _shared, _double_run = make_calculators(
        reform_series.by_year, YEARS, scenario.estate_params
    )

    def prices(records, law):
        del records, law
        return {}

    builder = build_frame.make_frame_builder(
        scenario.indexes, scenario.cluster_death_weight_cap
    )
    return builder, reform_series.stacked, baseline_series.stacked, taxes, prices


def run(scenario, population, saving_share: float, reform=None):
    builder, reform_laws, baseline_laws, taxes, prices = wire(scenario, reform)
    return run_one_scenario(
        population,
        reform_laws,
        baseline_laws,
        wealth_profiles(saving_share),
        years=YEARS,
        frame_builder=builder,
        tax_calculator=taxes,
        price_calculator=prices,
    )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def test_the_raw_branch_is_exactly_the_jax_frame_steps(scenario, population):
    builder, _reform, _baseline, _taxes, _prices = wire(scenario)
    raw = one_year.population_arrays_for_years(population, YEARS)
    first_year = {name: value[0] for name, value in raw.items()}
    law = stacked_annual_laws(scenario, scenario, YEARS)[0].by_year[0].for_records(
        first_year["filing_status"]
    )

    through_the_adapter = builder(
        first_year, law, jnp.asarray(float(YEARS[0])), apply_adjusters=False
    )
    index_years, cpi_levels = law_by_year.cpi_level_series(scenario.indexes)
    directly = steps_jax.build_world_frame(
        first_year,
        jnp.asarray(float(YEARS[0])),
        law.frame_columns,
        jnp.asarray(index_years),
        jnp.asarray(cpi_levels),
        jnp.asarray(scenario.cluster_death_weight_cap),
        jnp.asarray(build_frame.DONOR_CLUSTER_GROSS_ESTATE_FLOOR),
        jnp.asarray(first_year["valid"], dtype=bool),
    )

    assert through_the_adapter.keys() == directly.keys()
    for name, value in directly.items():
        np.testing.assert_array_equal(
            np.asarray(through_the_adapter[name]), np.asarray(value), err_msg=name
        )


def test_the_raw_branch_carries_no_income_adjustment(scenario, population):
    builder, _reform, _baseline, _taxes, _prices = wire(scenario)
    raw = one_year.population_arrays_for_years(population, YEARS)
    first_year = {name: value[0] for name, value in raw.items()}
    law = stacked_annual_laws(scenario, scenario, YEARS)[0].by_year[0].for_records(
        first_year["filing_status"]
    )

    frame = builder(first_year, law, jnp.asarray(2026.0), apply_adjusters=False)
    for name in FLOWS_DOLLAR_COLUMNS:
        assert name not in frame


def test_the_flows_branch_carries_both_dollar_columns_exactly_once(
    scenario, population
):
    builder, _reform, _baseline, _taxes, _prices = wire(scenario)
    raw = one_year.population_arrays_for_years(population, YEARS)
    first_year = {name: value[0] for name, value in raw.items()}
    law = stacked_annual_laws(scenario, scenario, YEARS)[0].by_year[0].for_records(
        first_year["filing_status"]
    )

    frame = builder(first_year, law, jnp.asarray(2026.0), apply_adjusters=True)
    for name in FLOWS_DOLLAR_COLUMNS:
        assert name in frame
        assert np.asarray(frame[name]).shape == (len(HOUSEHOLDS),)

    # Both adjusters refuse a second application, which is what "exactly once"
    # is enforced by. Running the branch again on its own output must stop.
    with pytest.raises(ValueError):
        build_frame.apply_reform_income_adjustments(frame, law)


def test_the_wealth_channel_columns_survive_the_adapter(scenario, population):
    builder, _reform, _baseline, _taxes, _prices = wire(scenario)
    raw = one_year.population_arrays_for_years(population, YEARS)
    first_year = {name: value[0] for name, value in raw.items()}
    law = stacked_annual_laws(scenario, scenario, YEARS)[0].by_year[0].for_records(
        first_year["filing_status"]
    )

    frame = builder(first_year, law, jnp.asarray(2026.0), apply_adjusters=True)
    for name in (
        "net_worth",
        "weight",
        "age1",
        "age2",
        "filing_status",
        "dep_status",
        *steps_jax.ESTATE_ASSET_COLS,
        *steps_jax.ESTATE_DEBT_COLS,
    ):
        assert name in frame, name


def test_the_validity_mask_is_honored(scenario):
    """A slot holding no household this year gets no estate mortality."""
    columns = _every_column()
    rows = []
    for household in HOUSEHOLDS[:2]:
        row = {name: 0.0 for name in columns}
        row["age2"] = np.nan
        row["sstb_part"] = np.nan
        row["sstb_scorp"] = np.nan
        row["kg_lt_years_held"] = np.nan
        for draw in prepare.DRAW_COLUMNS:
            row[draw] = 0.5
        row.update(household)
        rows.append(row)
    # The second household is present in 2026 only.
    frames = {
        2026: pd.DataFrame(rows),
        2027: pd.DataFrame(rows[:1]),
    }
    universe = prepare.build_universe(
        {2026: np.array([1, 2], dtype=np.int64), 2027: np.array([1], dtype=np.int64)}
    )
    population = population_from_frames(frames, universe)

    builder, _reform, _baseline, _taxes, _prices = wire(scenario)
    raw = one_year.population_arrays_for_years(population, YEARS)
    second_year = {name: value[1] for name, value in raw.items()}
    law = stacked_annual_laws(scenario, scenario, YEARS)[0].by_year[0].for_records(
        second_year["filing_status"]
    )
    frame = builder(second_year, law, jnp.asarray(2027.0), apply_adjusters=False)

    empty_slot = int(np.flatnonzero(~np.asarray(second_year["valid"]))[0])
    assert float(np.asarray(frame["estate_m"])[empty_slot]) == 0.0
    assert float(np.asarray(frame["net_worth"])[empty_slot]) == 0.0


# ---------------------------------------------------------------------------
# The wired runner's timing properties
# ---------------------------------------------------------------------------


def test_a_null_reform_leaves_every_channel_at_zero(scenario, population):
    """Baseline law as its own reform: no channel moves and the tub stays empty."""
    simulation = run(scenario, population, saving_share=0.5)

    np.testing.assert_array_equal(
        np.asarray(simulation.states_after_year.mechanical_wealth),
        np.zeros((len(YEARS), N_AGES, N_BINS)),
    )
    np.testing.assert_array_equal(
        np.asarray(simulation.states_after_year.conventional_wealth),
        np.zeros((len(YEARS), N_AGES, N_BINS)),
    )
    for year in YEARS:
        baseline = simulation.grid.get(world_names.Frame.RAW, world_names.Law.BASELINE, year)
        for frame in (
            world_names.Frame.RAW,
            world_names.Frame.FLOWS,
            world_names.Frame.MECHANICAL,
            world_names.Frame.BEHAVIOR,
        ):
            reform = simulation.grid.get(frame, world_names.Law.REFORM, year)
            for column, value in baseline.items():
                np.testing.assert_allclose(
                    np.asarray(reform[column]),
                    np.asarray(value),
                    rtol=0,
                    atol=0,
                    err_msg=f"{frame.value} {column} in {year}",
                )


def test_the_first_year_mechanical_frame_equals_the_flows_frame(scenario, population):
    """Nothing is carried into the first year, so the haircut has nothing to apply."""
    simulation = run(scenario, population, saving_share=0.5)
    first_year = YEARS[0]
    flows = simulation.grid.get(world_names.Frame.FLOWS, world_names.Law.REFORM, first_year)
    mechanical = simulation.grid.get(
        world_names.Frame.MECHANICAL, world_names.Law.REFORM, first_year
    )
    for column, value in flows.items():
        np.testing.assert_allclose(
            np.asarray(mechanical[column]), np.asarray(value), err_msg=column
        )


def test_the_four_channels_telescope_to_the_whole_reform_effect(scenario, population):
    simulation = run(scenario, population, saving_share=0.5)
    for year in YEARS:
        pieces = (
            grid_differences.first_order_minus_baseline(simulation.grid, year),
            grid_differences.mechanical_ex_saving_minus_first_order(simulation.grid, year),
            grid_differences.mechanical_minus_mechanical_ex_saving(simulation.grid, year),
            grid_differences.conventional_minus_mechanical(simulation.grid, year),
        )
        whole = grid_differences.conventional_minus_baseline(simulation.grid, year)
        for column, value in whole.items():
            added_up = sum(
                (piece[column] for piece in pieces), jnp.zeros_like(value)
            )
            np.testing.assert_allclose(added_up, value, err_msg=column)


def test_a_zero_saving_share_leaves_the_tub_empty(scenario, population):
    simulation = run(scenario, population, saving_share=0.0)
    np.testing.assert_array_equal(
        np.asarray(simulation.states_after_year.mechanical_wealth[-1]),
        np.zeros((N_AGES, N_BINS)),
    )


# ---------------------------------------------------------------------------
# The wealth-shortfall history
# ---------------------------------------------------------------------------


def test_the_history_satisfies_the_diagnostics_contract(scenario, population):
    simulation = run(scenario, population, saving_share=0.5)
    history = assemble_wealth_history(simulation, haircut_cap=0.9)

    history.check_shapes()
    assert history.cell_shortfall.shape == (len(YEARS), N_AGES, N_BINS)
    assert history.allocated_dollars.shape == (len(YEARS), len(HOUSEHOLDS))


def test_the_history_shortfall_is_the_state_the_runner_carried(scenario, population):
    simulation = run(scenario, population, saving_share=0.5)
    history = assemble_wealth_history(simulation, haircut_cap=0.9)

    np.testing.assert_array_equal(
        history.cell_shortfall,
        np.asarray(simulation.states_after_year.mechanical_wealth),
    )
    totals = wealth_diagnostics.shortfall_totals_by_year(history)
    np.testing.assert_allclose(
        totals["total_shortfall_dollars"],
        np.asarray(simulation.states_after_year.mechanical_wealth).sum(axis=(1, 2)),
    )


# ---------------------------------------------------------------------------
# The distribution table's groups
# ---------------------------------------------------------------------------


def test_the_distribution_groups_partition_the_population(scenario, population):
    """The quintiles and the negative group of each ranked cut hold every household once.

    Over those six groups, the weighted count is the whole population's, and
    the baseline payroll tax adds up to the published payroll total.
    """
    simulation = run(scenario, population, saving_share=0.5)
    table = simulation.distribution_table
    assert table.shape == (
        len(YEARS),
        len(distribution.CORPORATE_SPREADS),
        len(distribution.TAX_ROWS),
        len(distribution.GROUPS),
        len(distribution.value_column_names()),
    )
    column = distribution.value_column_names().index
    payroll = distribution.TAX_ROWS.index("payroll")
    total_weight = sum(household["weight"] for household in HOUSEHOLDS)
    negative_label = {distribution.DIMENSION_NET_WORTH: distribution.LABEL_NEGATIVE_NET_WORTH}
    for dimension in (
        distribution.DIMENSION_INCOME,
        distribution.DIMENSION_ACCRUAL_INCOME,
        distribution.DIMENSION_AGI,
        distribution.DIMENSION_NET_WORTH,
    ):
        labels = (
            *distribution.QUINTILE_LABELS,
            negative_label.get(dimension, distribution.LABEL_NEGATIVE_INCOME),
        )
        rows = [distribution.GROUPS.index((dimension, label)) for label in labels]
        for year_index, year in enumerate(YEARS):
            published = simulation.grid.get(
                world_names.Frame.RAW, world_names.Law.BASELINE, year
            )
            for spread in range(len(distribution.CORPORATE_SPREADS)):
                groups = table[year_index, spread, payroll][rows]
                assert groups[:, column("n_tax_units")].sum() == pytest.approx(
                    total_weight
                )
                assert_agrees_to_the_cent(
                    groups[:, column("baseline_tax_dollars")].sum(),
                    float(np.asarray(published["liab_pr"])[0]),
                    f"payroll tax over the {dimension} cut against the published "
                    "total",
                )
