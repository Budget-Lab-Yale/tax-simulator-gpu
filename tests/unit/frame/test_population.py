"""Stable population slots, annual fields, random draws, and fill values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxmodel.frame import population, prepare
from tests.unit.frame import population_checks


def universe() -> prepare.RecordUniverse:
    """Three records with unequal windows over two years."""
    return prepare.build_universe({2026: np.array([3, 1]), 2027: np.array([2, 3])})


def annual_frames() -> dict[int, pd.DataFrame]:
    """Two small annual frames with one draw shared across years.

    `filing_status` and `age2` are both stored as narrow whole numbers, and
    `age2` is missing for the household with no second spouse, which is what
    exercises the integer standing for a missing value.
    """
    return {
        2026: pd.DataFrame(
            {
                "income": [300.0, np.nan],
                "id": [3, 1],
                "filing_status": [2, 1],
                "age2": [64.0, np.nan],
                "weight": [30.0, 10.0],
                "r.bus_loss": [0.3, 0.1],
                "forbes_name": [None, "Named household"],
            }
        ),
        2027: pd.DataFrame(
            {
                "income": [200.0, 310.0],
                "id": [2, 3],
                "filing_status": [1, 2],
                "age2": [np.nan, 65.0],
                "weight": [20.0, 31.0],
                "r.bus_loss": [0.2, 0.3],
                "forbes_name": [None, None],
            }
        ),
    }


def expected_slice(
    frame: pd.DataFrame, record_universe: prepare.RecordUniverse, year: int
) -> pd.DataFrame:
    """One year on the run's slots, absent records filled as the calculators expect."""
    numeric = frame.drop(columns=["forbes_name"])
    padded = pd.DataFrame({"id": record_universe.ids}).merge(
        numeric, on="id", how="left", validate="one_to_one"
    )
    valid = np.isin(record_universe.ids, frame["id"].to_numpy())
    for name in padded.columns:
        if name == "id":
            continue
        fill = population.FILING_SINGLE if name == "filing_status" else 0.0
        padded[name] = padded[name].where(valid, fill)
    padded["year"] = int(year)
    padded["valid"] = valid
    return padded


def test_population_places_each_record_in_its_slot_with_the_stated_fills():
    record_universe = universe()
    frames = annual_frames()
    prepared = population.population_from_frames(frames, record_universe)

    for year in record_universe.years:
        expected = expected_slice(frames[year], record_universe, year)
        actual = population_checks.year_frame(prepared, year)
        assert list(actual.columns) == list(expected.columns)
        for name in actual.columns:
            np.testing.assert_array_equal(
                actual[name].to_numpy(), expected[name].to_numpy(), err_msg=name
            )


def test_unequal_year_windows_keep_one_slot_for_each_record():
    prepared = population.population_from_frames(annual_frames(), universe())
    first = population_checks.year_frame(prepared, 2026)
    second = population_checks.year_frame(prepared, 2027)

    assert list(first["id"]) == list(second["id"]) == [1, 2, 3]
    assert list(first["valid"]) == [True, False, True]
    assert list(second["valid"]) == [False, True, True]


def test_shuffling_annual_records_changes_no_stored_value():
    frames = annual_frames()
    shuffled = {
        year: frame.sample(frac=1.0, random_state=year).reset_index(drop=True)
        for year, frame in frames.items()
    }
    ordered_population = population.population_from_frames(frames, universe())
    shuffled_population = population.population_from_frames(shuffled, universe())
    differences = population_checks.population_differences(
        ordered_population, shuffled_population
    )
    assert differences == []


def test_invalid_slots_receive_safe_values_and_present_missing_values_remain():
    frames = annual_frames()
    frames[2026].loc[frames[2026]["id"] == 1, "weight"] = np.nan
    prepared = population.population_from_frames(frames, universe())
    first = population_checks.year_frame(prepared, 2026).set_index("id")

    assert first.loc[2, "filing_status"] == population.FILING_SINGLE
    assert first.loc[2, "weight"] == 0.0
    assert first.loc[2, "income"] == 0.0
    assert first.loc[2, "r.bus_loss"] == 0.0
    assert first.loc[1, "weight"] == 0.0
    assert np.isnan(first.loc[1, "income"])
    assert prepared.dropped_text_columns == ("forbes_name",)


def test_a_whole_number_column_is_stored_narrowly_and_read_back_exactly():
    prepared = population.population_from_frames(annual_frames(), universe())

    assert prepared.values["age2"].dtype == prepare.NARROW_WHOLE_NUMBER_TYPE
    assert prepared.values["income"].dtype == np.float64

    # Record 1 has no second spouse in 2026, so its age comes back missing;
    # record 3's comes back as the age it was given. A slot holding no record
    # this year gets the same zero every other field gets.
    first = population_checks.year_frame(prepared, 2026).set_index("id")
    assert first["age2"].dtype == np.float64
    assert np.isnan(first.loc[1, "age2"])
    assert first.loc[3, "age2"] == 64.0
    assert first.loc[2, "age2"] == 0.0


def test_a_whole_number_column_holding_a_fraction_stops_the_read():
    frames = annual_frames()
    frames[2026].loc[frames[2026]["id"] == 3, "age2"] = 64.5
    with pytest.raises(ValueError, match="age2 is stored as a whole number"):
        population.population_from_frames(frames, universe())


def test_draws_are_stored_once_per_slot_and_returned_in_slot_order():
    prepared = population.population_from_frames(annual_frames(), universe())
    draws = population_checks.draws_frame(prepared)
    assert list(draws["id"]) == [1, 2, 3]
    assert list(draws["r.bus_loss"]) == [0.1, 0.2, 0.3]
    assert prepared.draws["r.bus_loss"].shape == (3,)


def test_a_draw_cannot_change_for_the_same_record_between_years():
    frames = annual_frames()
    frames[2027].loc[frames[2027]["id"] == 3, "r.bus_loss"] = 0.4
    with pytest.raises(ValueError, match="differs between years"):
        population.population_from_frames(frames, universe())
