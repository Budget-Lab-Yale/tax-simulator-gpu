"""One scenario over a tiny fixed Population block, a year at a time.

The runner reports population totals, not households, so every check here is
written on a total. The two test households carry a weight of one, which makes
a total the sum over the two of them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.frame.population import Population, stored_values
from taxmodel.frame.prepare import RecordUniverse
from taxmodel.reporting import device_sums, distribution, mtr_table
from taxmodel.taxes import marginal_rates
from taxmodel.years import one_year, wealth_arithmetic, world_names, worlds
from tests.unit.years import grid_differences
from tests.unit.years.one_scenario import (
    make_passthrough_frame_builder,
    run_one_scenario,
    taxes_then_prices,
    unchanged_frame,
)

YEARS = (2026, 2027)
N_AGES = len(wealth_arithmetic.age_grid())
N_BINS = 2


def population() -> Population:
    ids = np.array([1, 2], dtype=np.int64)
    universe = RecordUniverse(ids, YEARS, np.ones((len(YEARS), len(ids)), dtype=bool))
    shape = (len(YEARS), len(ids))
    values = {
        "weight": np.ones(shape),
        "age1": np.full(shape, 40.0),
        "age2": np.zeros(shape),
        "filing_status": np.ones(shape),
        "dep_status": np.zeros(shape),
        "n_dep": np.zeros(shape),
        # Both households file, so these totals count everyone; the
        # filers-only rule for 1040 amounts still applies.
        "filer": np.ones(shape),
        "dep_age1": np.full(shape, np.nan),
        "dep_age2": np.full(shape, np.nan),
        "dep_age3": np.full(shape, np.nan),
        "income": np.zeros(shape),
        "net_worth": np.broadcast_to(np.array([100.0, 200.0]), shape).copy(),
        "value.cash": np.broadcast_to(np.array([100.0, 200.0]), shape).copy(),
    }
    # The average-marginal-rate table weights each rate by the dollars of the
    # thing it prices, so the frame needs a column for every priced dollar.
    # Reading the list off `PRICED_DOLLARS` means widening it does not have to
    # be repeated here.
    for priced_dollar in marginal_rates.PRICED_DOLLARS:
        values[priced_dollar] = np.zeros(shape)
    values["wages_base1"] = np.broadcast_to(np.array([10.0, 20.0]), shape).copy()
    return Population(
        universe=universe,
        field_names=("id", "year", *values),
        values=stored_values(values),
        draws={},
        dropped_text_columns=(),
    )


def taxes(frame, law):
    base = jnp.asarray(frame["income"]) + jnp.asarray(frame["value.cash"])
    liability = jnp.asarray(law) * base
    zero = jnp.zeros_like(liability)
    return {
        "liab_iit_net": liability,
        "liab_pr": zero,
        "liab_wealth": zero,
        "expanded_inc": base,
        "agi": jnp.asarray(frame["income"]),
        # The totals reduction reads the filing flag out of the world's own
        # results and counts the returns the IRS could pre-fill,
        # so the placeholder calculator produces both. Both test households
        # file, and neither is simple enough to pre-fill.
        "filer_recomputed": jnp.ones_like(liability),
        "simple_filer": zero,
    }


def prices(frame, law):
    """A stand-in price calculator: one named rate, so the wiring is visible."""
    del law
    return {"mtr_wages_base1": jnp.zeros_like(frame["weight"])}


def marginal_prices(frame, law):
    """Rates with distinct law sides so the MTR table has known answers."""
    value = jnp.asarray(law) * jnp.ones_like(frame["weight"])
    return {
        name.removeprefix("baseline_law_"): value
        for name in marginal_rates.rate_columns(marginal_rates.BASELINE_LAW)
    }


def adjust_income(frame, law, year):
    del year
    out = dict(frame)
    out["income"] = out["income"] + (jnp.asarray(law) - 1.0) * 10.0
    return out


FRAME_BUILDER = make_passthrough_frame_builder(adjust_income)


def profiles(saving_share: float):
    return one_year.stack_wealth_profiles(
        jnp.full((N_AGES, N_BINS), saving_share),
        jnp.eye(N_BINS),
        jnp.zeros(len(YEARS)),
        fmax=0.9,
        pass_through_weight=0.2,
    )


def run(reform_law, saving_share: float):
    return run_one_scenario(
        population(),
        jnp.asarray(reform_law),
        jnp.ones(len(YEARS)),
        profiles(saving_share),
        years=YEARS,
        frame_builder=FRAME_BUILDER,
        tax_calculator=taxes,
        price_calculator=prices,
    )


def assert_zero_state_path(simulation):
    np.testing.assert_array_equal(
        simulation.states_after_year.mechanical_wealth,
        np.zeros((len(YEARS), N_AGES, N_BINS)),
    )
    np.testing.assert_array_equal(
        simulation.states_after_year.conventional_wealth,
        np.zeros((len(YEARS), N_AGES, N_BINS)),
    )


def test_no_reform_and_zero_saving_share_are_both_fixed_points():
    no_reform = run([1.0, 1.0], saving_share=0.5)
    no_saving = run([1.1, 1.1], saving_share=0.0)

    assert_zero_state_path(no_reform)
    assert_zero_state_path(no_saving)
    for year in YEARS:
        baseline = no_reform.grid.get(world_names.Frame.RAW, world_names.Law.BASELINE, year)
        for frame in (
            world_names.Frame.RAW,
            world_names.Frame.FLOWS,
            world_names.Frame.MECHANICAL,
            world_names.Frame.BEHAVIOR,
        ):
            np.testing.assert_array_equal(
                no_reform.grid.get(frame, world_names.Law.REFORM, year)["liab_iit_net"],
                baseline["liab_iit_net"],
            )
        np.testing.assert_array_equal(
            no_saving.grid.get(world_names.Frame.MECHANICAL, world_names.Law.REFORM, year)[
                "liab_iit_net"
            ],
            no_saving.grid.get(world_names.Frame.FLOWS, world_names.Law.REFORM, year)[
                "liab_iit_net"
            ],
        )
        np.testing.assert_array_equal(
            no_saving.grid.get(world_names.Frame.BEHAVIOR, world_names.Law.REFORM, year)[
                "liab_iit_net"
            ],
            no_saving.grid.get(world_names.Frame.MECHANICAL, world_names.Law.REFORM, year)[
                "liab_iit_net"
            ],
        )


def test_five_channel_differences_telescope_to_the_total():
    simulation = run([1.1, 1.1], saving_share=0.5)

    for year in YEARS:
        pieces = (
            grid_differences.first_order_minus_baseline(simulation.grid, year),
            grid_differences.mechanical_ex_saving_minus_first_order(simulation.grid, year),
            grid_differences.mechanical_minus_mechanical_ex_saving(simulation.grid, year),
            grid_differences.conventional_minus_mechanical(simulation.grid, year),
        )
        complete = grid_differences.conventional_minus_baseline(simulation.grid, year)
        for name in complete:
            total = sum(
                (piece[name] for piece in pieces), jnp.zeros_like(complete[name])
            )
            np.testing.assert_allclose(total, complete[name])


def test_the_runner_keeps_no_household_results():
    simulation = run([1.1, 1.1], saving_share=0.5)

    for year in YEARS:
        cell = simulation.grid.get(world_names.Frame.MECHANICAL, world_names.Law.REFORM, year)
        for name, value in cell.items():
            assert np.asarray(value).shape == (1,), name


def test_runner_stacks_the_same_mtr_sums_as_each_mechanical_year():
    """The MTR reduction uses fixed baseline quintiles on the priced frame."""
    pop = population()
    reform_laws = jnp.array([1.1, 1.2])
    baseline_laws = jnp.ones(len(YEARS))
    wealth_profiles = profiles(0.0)
    simulation = run_one_scenario(
        pop,
        reform_laws,
        baseline_laws,
        wealth_profiles,
        years=YEARS,
        frame_builder=FRAME_BUILDER,
        tax_calculator=taxes,
        price_calculator=marginal_prices,
    )

    raw_by_year = one_year.population_arrays_for_years(pop, YEARS)
    zero = jnp.zeros((N_AGES, N_BINS))
    state = worlds.SimulationState(zero, zero)
    expected_sums = []
    for index, year in enumerate(YEARS):
        raw = jax.tree.map(lambda value, i=index: value[i], raw_by_year)
        profile = jax.tree.map(lambda value, i=index: value[i], wealth_profiles)
        baseline_frame = FRAME_BUILDER(
            raw, baseline_laws[index], year, apply_adjusters=False
        )
        baseline_result = taxes(baseline_frame, baseline_laws[index])
        households = device_sums.distribution_households(
            raw, baseline_frame, baseline_result
        )
        labels = device_sums.baseline_rank_labels(households, {})
        state, year_results = worlds.annual_step(
            raw,
            reform_laws[index],
            baseline_laws[index],
            baseline_result,
            state,
            profile,
            jnp.asarray(year, dtype=jnp.float64),
            FRAME_BUILDER,
            taxes,
            marginal_prices,
            unchanged_frame,
            taxes_then_prices(taxes, marginal_prices),
        )
        rates = {
            **marginal_rates.under_law(
                year_results.mechanical_prices_baseline,
                marginal_rates.BASELINE_LAW,
            ),
            **marginal_rates.under_law(
                year_results.mechanical_prices_reform,
                marginal_rates.REFORM_LAW,
            ),
        }
        expected_sums.append(
            mtr_table.mtr_table_sums(
                rates,
                year_results.mechanical_frame,
                device_sums._income_quintile_labels(labels["household_income"]),
                year_results.mechanical_frame["weight"],
                device_sums.MTR_GROUPS,
            )
        )

    assert simulation.mtr_table_sums is not None
    for name in expected_sums[0]:
        expected = jnp.stack([year_sums[name] for year_sums in expected_sums])
        np.testing.assert_array_equal(simulation.mtr_table_sums[name], expected)

    actual_rows = mtr_table.mtr_table(
        simulation.mtr_table_sums,
        years=YEARS,
        group_names=distribution.QUINTILE_LABELS,
    )
    expected_rows = mtr_table.mtr_table(
        {
            name: jnp.stack([year_sums[name] for year_sums in expected_sums])
            for name in expected_sums[0]
        },
        years=YEARS,
        group_names=distribution.QUINTILE_LABELS,
    )
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        assert actual.keys() == expected.keys()
        for name in actual:
            if isinstance(actual[name], float) and np.isnan(actual[name]):
                assert np.isnan(expected[name])
            else:
                assert actual[name] == expected[name]


COUNTING_YEARS = (2026,)

# Five households whose counts can be worked out on paper. Weights are round
# numbers so every expected count below is arithmetic anyone can check.
COUNTING_WEIGHT = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
# The model's own filing flag: the third and fifth households do not file.
COUNTING_FILER_RECOMPUTED = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
# The second and fifth households are themselves claimed as dependents on
# someone else's return.
COUNTING_DEP_STATUS = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
# Dependents claimed: the third household claims two, the fourth three.
COUNTING_N_DEP = np.array([0.0, 0.0, 2.0, 3.0, 0.0])
# Returns the IRS could plausibly pre-fill, among the households that file.
COUNTING_SIMPLE_FILER = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
# Income tax owed. The second household owes nothing, so it is not counted;
# the third and fifth do not file, so neither their dollars nor their counts
# reach the published income-tax totals.
COUNTING_INCOME_TAX = np.array([5.0, 0.0, 7.0, -3.0, 9.0])
# Payroll tax, which is owed whether or not a return is filed, so every
# household with a nonzero amount is counted.
COUNTING_PAYROLL_TAX = np.array([1.0, 0.0, 2.0, 0.0, 3.0])


def counting_population() -> Population:
    """One year of five households, with every count workable by hand."""
    ids = np.arange(1, len(COUNTING_WEIGHT) + 1, dtype=np.int64)
    universe = RecordUniverse(ids, COUNTING_YEARS, np.ones((1, len(ids)), dtype=bool))
    row = lambda values: np.asarray(values, dtype=np.float64)[None, :]
    zeros = row(np.zeros(len(ids)))
    values = {
        "weight": row(COUNTING_WEIGHT),
        "age1": row(np.full(len(ids), 40.0)),
        "age2": zeros,
        "filing_status": row(np.ones(len(ids))),
        "dep_status": row(COUNTING_DEP_STATUS),
        "n_dep": row(COUNTING_N_DEP),
        # Tax-Data says every household files. The model's own flag, produced
        # by the calculator below, says two of them do not, and the published
        # totals follow the model. Setting the two apart here is
        # what makes this test tell them apart.
        "filer": row(np.ones(len(ids))),
        "dep_age1": row(np.full(len(ids), np.nan)),
        "dep_age2": row(np.full(len(ids), np.nan)),
        "dep_age3": row(np.full(len(ids), np.nan)),
        "income": zeros,
        "net_worth": zeros,
        "value.cash": zeros,
    }
    return Population(
        universe=universe,
        field_names=("id", "year", *values),
        values=stored_values(values),
        draws={},
        dropped_text_columns=(),
    )


def counting_taxes(frame, law):
    """Fixed per-household amounts, so the expected totals are known exactly."""
    del law
    shape = jnp.shape(frame["weight"])
    fixed = lambda values: jnp.broadcast_to(jnp.asarray(values), shape)
    return {
        "liab_iit_net": fixed(COUNTING_INCOME_TAX),
        "liab_pr": fixed(COUNTING_PAYROLL_TAX),
        "liab_wealth": jnp.zeros(shape),
        "expanded_inc": jnp.zeros(shape),
        "agi": jnp.zeros(shape),
        "filer_recomputed": fixed(COUNTING_FILER_RECOMPUTED),
        "simple_filer": fixed(COUNTING_SIMPLE_FILER),
    }


def test_the_counts_are_what_five_households_add_up_to_by_hand():
    """Every count the published totals files report, on paper and in the run.

    The five households above were chosen so that the filers-only rule, the
    everyone rule for payroll tax, and the difference between Tax-Data's
    filing flag and the model's own all change the answer.
    """
    simulation = run_one_scenario(
        counting_population(),
        jnp.ones(len(COUNTING_YEARS)),
        jnp.ones(len(COUNTING_YEARS)),
        one_year.stack_wealth_profiles(
            jnp.zeros((N_AGES, N_BINS)),
            jnp.eye(N_BINS),
            jnp.zeros(len(COUNTING_YEARS)),
            fmax=0.9,
            pass_through_weight=0.2,
        ),
        years=COUNTING_YEARS,
        frame_builder=FRAME_BUILDER,
        tax_calculator=counting_taxes,
        price_calculator=prices,
    )
    cell = simulation.grid.get(
        world_names.Frame.RAW, world_names.Law.BASELINE, COUNTING_YEARS[0]
    )

    expected = {
        # All five households.
        "n_tax_units": 10.0 + 20.0 + 30.0 + 40.0 + 50.0,
        # The three the model says file.
        "n_returns": 10.0 + 20.0 + 40.0,
        # Of those three, only the second is claimed as someone's dependent.
        "n_returns_dep": 20.0,
        # Dependents claimed on the three filed returns: the fourth household's
        # three. The third household's two do not count, because it does not
        # file.
        "n_dep": 40.0 * 3.0,
        # The first and fourth households.
        "n_simple_filers": 10.0 + 40.0,
        # Filers with a nonzero income tax: the first and the fourth. The
        # second owes nothing; the third and fifth do not file.
        "n_liab_iit_net": 10.0 + 40.0,
        "liab_iit_net": 10.0 * 5.0 + 40.0 * -3.0,
        # Payroll tax counts everyone, filing or not: the first, third and
        # fifth households owe some.
        "n_liab_pr": 10.0 + 30.0 + 50.0,
        "liab_pr": 10.0 * 1.0 + 30.0 * 2.0 + 50.0 * 3.0,
        # Nobody owes a wealth tax, so nobody is counted.
        "n_liab_wealth": 0.0,
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(cell[name], np.array([value]), err_msg=name)


def test_a_gap_or_reversal_in_the_year_list_is_refused():
    """The wealth state carries year to year; the runner must not let a
    request like (2026, 2028) silently apply one year's aging across two."""
    for bad_years in ((2026, 2028), (2027, 2026)):
        try:
            run_one_scenario(
                population(),
                jnp.asarray(0.1),
                jnp.ones(len(bad_years)),
                profiles(0.5),
                years=bad_years,
                frame_builder=FRAME_BUILDER,
                tax_calculator=taxes,
                price_calculator=prices,
            )
        except ValueError as error:
            assert "consecutive" in str(error)
        else:
            raise AssertionError(f"years {bad_years} were accepted")


def test_expected_estate_tax_blends_the_two_unused_exclusion_branches():
    """Expected estate tax blends the liability with and without a deceased
    spouse's unused exclusion, weighted by the chance of each.

    Each household contributes its chance of dying times that blend. Four
    households cover the cases that matter: taxable in both branches, taxable
    in neither, taxable only without the carried-over exclusion, and taxable
    only with it.
    """
    cell = {
        "estate_p_dsue": jnp.asarray([0.0, 0.25, 0.5, 1.0]),
        "liab_estate_dsue": jnp.asarray([100.0, 0.0, 0.0, 80.0]),
        "liab_estate_nodsue": jnp.asarray([120.0, 0.0, 60.0, 0.0]),
    }
    mortality = jnp.asarray([0.01, 0.02, 0.03, 0.04])

    blended = device_sums._expected_estate_columns(cell, mortality)

    # Worked by hand, chance of dying times the blended liability:
    # 0.01 x 120, zero, 0.03 x (0.5 x 60), and 0.04 x 80.
    np.testing.assert_allclose(
        np.asarray(blended["est_tax_exp"]), [1.2, 0.0, 0.9, 3.2]
    )
    # The expected count of taxable estates. The third household is taxable
    # only when there is no exclusion to carry over, which is half its chance,
    # so it counts as half of its three-in-a-hundred chance of dying.
    np.testing.assert_allclose(
        np.asarray(blended["est_returns"]), [0.01, 0.0, 0.5 * 0.03, 0.04]
    )
