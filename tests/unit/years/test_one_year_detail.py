"""Asking a run for per-household detail, on the two-household test block.

Three things have to be true, and each has a test here:

* asking for detail changes no reported number -- the totals, the distribution
  tables, and the wealth-bathtub path all come back bitwise identical to the
  same run without a request;
* the detail columns re-add to the run's own totals: the weight times the
  column, summed over the exported rows, is the number the run reported;
* the household id column is a real key, so a join back to the Tax-Data
  inputs loses no rows.

The population, the placeholder calculators, and the wealth profiles are the
ones `test_one_year.py` uses, so this reads as the same run with one argument
added.
"""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest

from taxmodel.reporting import detail, device_sums
from taxmodel.years import world_names
from tests.unit.reporting.test_detail import weighted_totals
from tests.unit.years.one_scenario import run_one_scenario

from .test_one_year import (
    FRAME_BUILDER,
    YEARS,
    marginal_prices,
    population,
    prices,
    profiles,
    taxes,
)

# The two columns the placeholder calculator produces that mean something as a
# household amount: the income tax it owes and the income the distribution
# tables rank on.
DETAIL_COLUMNS = ("liab_iit_net", "expanded_inc")


def run(detail_request=None, price_calculator=prices):
    return run_one_scenario(
        population(),
        np.asarray([1.1, 1.1]),
        np.ones(len(YEARS)),
        profiles(0.5),
        years=YEARS,
        frame_builder=FRAME_BUILDER,
        tax_calculator=taxes,
        price_calculator=price_calculator,
        detail_request=detail_request,
    )


WORLDS_WITH_NUMBERS = world_names.WORLD_NAMES


def every_world_with_numbers(columns=DETAIL_COLUMNS, years=YEARS):
    return detail.DetailRequest.same_columns_from_each_world(
        columns=columns, worlds=WORLDS_WITH_NUMBERS, years=years
    )


def test_asking_for_detail_changes_no_reported_number():
    without = run()
    with_detail = run(every_world_with_numbers())

    for year in YEARS:
        for _field, frame, law in device_sums.EVALUATED_CELLS:
            plain = without.grid.get(frame, law, year)
            asked = with_detail.grid.get(frame, law, year)
            assert plain.keys() == asked.keys()
            for column, value in plain.items():
                np.testing.assert_array_equal(value, asked[column], err_msg=column)
    if without.distribution_table is None:
        assert with_detail.distribution_table is None
    else:
        np.testing.assert_array_equal(
            without.distribution_table, with_detail.distribution_table
        )
    for plain_value, asked_value in zip(
        jax.tree.leaves(without.states_after_year),
        jax.tree.leaves(with_detail.states_after_year),
        strict=True,
    ):
        np.testing.assert_array_equal(plain_value, asked_value)
    for name, plain_value in without.wealth_channel_history.items():
        np.testing.assert_array_equal(
            np.asarray(plain_value),
            np.asarray(with_detail.wealth_channel_history[name]),
            err_msg=name,
        )
    assert without.detail is None


def test_the_detail_columns_re_add_to_the_runs_own_totals():
    simulation = run(every_world_with_numbers())

    for world in WORLDS_WITH_NUMBERS:
        frame, law = world_names.world_cell(world)
        re_added = weighted_totals(simulation.detail, world)
        for index, year in enumerate(YEARS):
            reported = simulation.grid.get(frame, law, year)
            for column in DETAIL_COLUMNS:
                assert float(re_added[column][index]) == float(
                    np.asarray(reported[column])[0]
                ), f"{world} {column} {year}"


def test_a_join_back_to_the_household_ids_loses_no_rows(tmp_path):
    """The Tax-Data side of the join is stood in for by the population's own ids."""
    simulation = run(every_world_with_numbers())
    written = detail.write_parquet(simulation.detail, tmp_path)
    assert [file.world for file in written] == list(WORLDS_WITH_NUMBERS)

    inputs = pd.DataFrame({"household_id": population().universe.ids})
    for file in written:
        exported = pd.read_parquet(file.path)
        joined = exported.merge(
            inputs, on="household_id", how="left", indicator=True, validate="many_to_one"
        )
        assert len(joined) == len(exported)
        assert (joined["_merge"] == "both").all()
        # Every household in the inputs is present in every exported year.
        for year in YEARS:
            assert set(exported.loc[exported["year"] == year, "household_id"]) == set(
                inputs["household_id"]
            )


def test_only_the_years_asked_for_are_written_but_every_year_is_computed():
    one_year = run(every_world_with_numbers(years=(YEARS[1],)))
    every_year = run(every_world_with_numbers())

    assert one_year.detail.years == (YEARS[1],)
    assert every_year.detail.years == YEARS
    # The kept year holds the same numbers either way, which is what says the
    # compiled program ran the same years in both cases and the filtering
    # happened afterwards on the host.
    for key, column in one_year.detail.columns.items():
        np.testing.assert_array_equal(
            column, every_year.detail.columns[key][1:2], err_msg=str(key)
        )


def test_the_priced_rates_come_back_from_the_mechanical_world():
    request = detail.DetailRequest(
        columns_by_world={"mechanical": ("baseline_law_mtr_kg_lt", "liab_iit_net")},
        years=YEARS,
    )
    simulation = run(request, price_calculator=marginal_prices)

    rates = np.asarray(
        simulation.detail.columns[("mechanical", "baseline_law_mtr_kg_lt")]
    )
    # The placeholder price calculator returns the law's own number as every
    # rate, and the baseline law is 1.0 in both years.
    np.testing.assert_array_equal(rates, np.ones_like(rates))


def test_asking_for_a_rate_a_run_does_not_price_fails_saying_so():
    request = detail.DetailRequest(
        columns_by_world={"mechanical": ("reform_law_mtr_kg_lt",)}, years=YEARS
    )
    # The plain placeholder price calculator returns one rate on wages and
    # nothing else, so this run never priced a dollar of capital gain.
    with pytest.raises(KeyError) as raised:
        run(request)
    assert "reform_law_mtr_kg_lt" in str(raised.value)


def test_the_detail_carries_the_weight_and_the_validity_the_run_used():
    simulation = run(every_world_with_numbers())
    block = population()

    np.testing.assert_array_equal(
        simulation.detail.weight,
        np.asarray([block.values["weight"][block.universe.year_index(y)] for y in YEARS]),
    )
    np.testing.assert_array_equal(
        simulation.detail.valid,
        np.asarray([block.universe.valid[block.universe.year_index(y)] for y in YEARS]),
    )
