"""The record universe: unequal year windows, stable slots, permutation
invariance, and the per-year slices written to disk.

The record universe is described in ARCHITECTURE D17. These tests use ids
written out by hand rather than Tax-Data, so what should happen is visible in
the test itself.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from taxmodel.frame import population, prepare
from tests.unit.frame import population_checks


def universe_from(year_ids: dict[int, list[int]]) -> prepare.RecordUniverse:
    return prepare.build_universe({y: np.array(ids) for y, ids in year_ids.items()})


class TestUnequalYearWindows:
    """A record present in some years and absent in others."""

    def test_universe_is_the_union_over_years(self):
        # Three records with three different windows: one all the way
        # through, one that only exists 2026-2030, one that appears in 2031.
        universe = universe_from(
            {
                2026: [1, 2],
                2030: [1, 2],
                2031: [1, 3],
            }
        )
        assert list(universe.ids) == [1, 2, 3]

    def test_a_record_absent_in_a_year_is_masked_not_dropped(self):
        universe = universe_from({2026: [1, 2], 2031: [1, 3]})
        assert list(universe.valid_in(2026)) == [True, True, False]
        assert list(universe.valid_in(2031)) == [True, False, True]
        # Every year has the same number of slots, which is the whole point.
        assert universe.valid.shape == (2, 3)

    def test_a_record_keeps_its_slot_in_every_year(self):
        universe = universe_from({2026: [1, 2], 2031: [1, 3]})
        assert list(universe.slots_of([1, 3])) == [0, 2]
        # Slot 1 is record 2 in both years, even though 2026 is the only year
        # it exists in.
        assert universe.ids[1] == 2

    def test_padding_zeroes_the_weight_of_an_absent_record(self):
        universe = universe_from({2026: [1, 2], 2031: [1, 3]})
        year_frame = pd.DataFrame({"id": [1, 3], "weight": [100.0, 200.0]})
        padded = population_checks.year_frame(
            population.population_from_frames(
                {
                    2026: pd.DataFrame({"id": [1, 2], "weight": [1.0, 1.0]}),
                    2031: year_frame,
                },
                universe,
            ),
            2031,
        )
        assert list(padded["id"]) == [1, 2, 3]
        assert list(padded["valid"]) == [True, False, True]
        assert list(padded["weight"]) == [100.0, 0.0, 200.0]
        # A weighted total needs no special case for the padded slot.
        assert padded["weight"].sum() == 300.0

    def test_a_record_outside_the_universe_is_an_error(self):
        universe = universe_from({2026: [1, 2]})
        stranger = pd.DataFrame({"id": [1, 99], "weight": [1.0, 1.0]})
        with pytest.raises(ValueError, match="not members of the record universe"):
            population.population_from_frames({2026: stranger}, universe)

    def test_a_year_outside_the_run_is_an_error(self):
        universe = universe_from({2026: [1, 2]})
        with pytest.raises(KeyError, match="not in this run"):
            universe.valid_in(2040)

class TestPermutationInvariance:
    """Shuffling the input records changes nothing downstream."""

    def test_slot_order_does_not_depend_on_file_order(self):
        forward = universe_from({2026: [5, 1, 9], 2027: [9, 5]})
        shuffled = universe_from({2026: [9, 5, 1], 2027: [5, 9]})
        assert list(forward.ids) == list(shuffled.ids)
        assert (forward.valid == shuffled.valid).all()

    def test_year_order_does_not_matter(self):
        forward = universe_from({2026: [1, 2], 2027: [2, 3]})
        backward = prepare.build_universe(
            {2027: np.array([3, 2]), 2026: np.array([2, 1])}
        )
        assert list(forward.ids) == list(backward.ids)
        assert (forward.valid == backward.valid).all()

    def test_padding_puts_shuffled_records_in_the_same_slots(self):
        universe = universe_from({2026: [1, 2, 3]})
        ordered = pd.DataFrame({"id": [1, 2, 3], "weight": [10.0, 20.0, 30.0]})
        shuffled = ordered.iloc[[2, 0, 1]].reset_index(drop=True)
        left = population_checks.year_frame(
            population.population_from_frames({2026: ordered}, universe), 2026
        )
        right = population_checks.year_frame(
            population.population_from_frames({2026: shuffled}, universe), 2026
        )
        pd.testing.assert_frame_equal(left, right)

    def test_a_records_draws_do_not_depend_on_its_position(self):
        forward = prepare.keyed_draws([7, 3, 11])
        shuffled = prepare.keyed_draws([11, 7, 3])
        for column in prepare.DRAW_COLUMNS:
            by_id = dict(zip(forward["id"], forward[column], strict=True))
            for record_id, value in zip(shuffled["id"], shuffled[column], strict=True):
                assert by_id[record_id] == value


class TestKeyedDraws:
    """Draws keyed on (seed, column name, id), per ARCHITECTURE D8."""

    def test_the_same_inputs_always_give_the_same_draws(self):
        first = prepare.keyed_draws([1, 2, 3], seed=76)
        second = prepare.keyed_draws([1, 2, 3], seed=76)
        pd.testing.assert_frame_equal(first, second)

    def test_adding_a_column_does_not_move_the_existing_draws(self):
        before = prepare.keyed_draws([1, 2, 3], columns=("r.bus_loss", "r.new_car"))
        after = prepare.keyed_draws(
            [1, 2, 3], columns=("r.bus_loss", "r.brand_new", "r.new_car")
        )
        pd.testing.assert_series_equal(before["r.bus_loss"], after["r.bus_loss"])
        pd.testing.assert_series_equal(before["r.new_car"], after["r.new_car"])

    def test_removing_a_column_does_not_move_the_existing_draws(self):
        full = prepare.keyed_draws([1, 2, 3])
        without_spare = prepare.keyed_draws(
            [1, 2, 3], columns=tuple(c for c in prepare.DRAW_COLUMNS if c != "r.behavior2")
        )
        for column in without_spare.columns:
            pd.testing.assert_series_equal(full[column], without_spare[column])

    def test_adding_a_record_does_not_move_anyone_elses_draws(self):
        small = prepare.keyed_draws([1, 2])
        larger = prepare.keyed_draws([1, 2, 3])
        pd.testing.assert_frame_equal(small, larger.iloc[:2].reset_index(drop=True))

    def test_two_columns_never_share_a_draw(self):
        draws = prepare.keyed_draws(list(range(200)))
        assert not (draws["r.bus_loss"] == draws["r.cdctc_takeup"]).any()

    def test_a_different_seed_gives_different_draws(self):
        assert not np.array_equal(
            prepare.keyed_draws([1, 2, 3], seed=76)["r.bus_loss"].to_numpy(),
            prepare.keyed_draws([1, 2, 3], seed=77)["r.bus_loss"].to_numpy(),
        )

    def test_uniform_draws_are_on_the_unit_interval(self):
        draws = prepare.keyed_draws(list(range(5000)))
        for column in prepare.DRAW_COLUMNS:
            values = draws[column].to_numpy()
            assert values.min() >= 0.0
            assert values.max() < 1.0
            # A uniform draw over 5,000 records should average near a half.
            assert abs(values.mean() - 0.5) < 0.02


class TestSlices:
    """Per-year frames that all share one slot layout (ARCHITECTURE D17)."""

    def test_every_year_padded_has_the_same_slots(self, tmp_path):
        universe = universe_from({2026: [1, 2], 2027: [2, 3]})
        prepared = population.population_from_frames(
            {
                2026: pd.DataFrame({"id": [1, 2], "weight": [1.0, 2.0]}),
                2027: pd.DataFrame({"id": [2, 3], "weight": [3.0, 4.0]}),
            },
            universe,
        )
        first = population_checks.year_frame(prepared, 2026)
        second = population_checks.year_frame(prepared, 2027)
        assert list(first["id"]) == list(second["id"]) == [1, 2, 3]
        assert list(first["valid"]) == [True, True, False]
        assert list(second["valid"]) == [False, True, True]
        # Slot 1 is record 2 in both years.
        assert first.loc[1, "weight"] == 2.0
        assert second.loc[1, "weight"] == 3.0


def tax_data_year(records: list[dict]) -> pd.DataFrame:
    """A stand-in for one Tax-Data year, written the way Tax-Data writes it.

    Every earner column named here exists in the real file, and so do the
    household totals `wages`, `sole_prop` and `farm`, which repeat what the
    earner columns already say.
    """
    earner_columns = [
        "wages1", "wages2", "tips1", "tips2", "ot1", "ot2",
        "sole_prop1", "sole_prop2", "farm1", "farm2",
    ]
    frame = pd.DataFrame(records).fillna(0.0)
    for name in ["id", *earner_columns]:
        if name not in frame.columns:
            frame[name] = 0.0
    frame["wages"] = frame["wages1"] + frame["wages2"]
    frame["sole_prop"] = frame["sole_prop1"] + frame["sole_prop2"]
    frame["farm"] = frame["farm1"] + frame["farm2"]
    frame["weight"] = 1000.0
    return frame


class TestSplittingOverlappingWages:
    """Tax-Data's double-counted columns become pieces that stand alone."""

    def test_base_pay_is_wages_less_that_earners_tips_and_overtime(self):
        year = tax_data_year(
            [
                {
                    "id": 1,
                    "wages1": 50_000.0, "tips1": 4_000.0, "ot1": 1_500.0,
                    "wages2": 30_000.0, "tips2": 0.0, "ot2": 2_000.0,
                }
            ]
        )
        split = prepare.split_overlapping_wages(year, 2026)
        assert split.loc[0, "wages_base1"] == 50_000.0 - 4_000.0 - 1_500.0
        assert split.loc[0, "wages_base2"] == 30_000.0 - 2_000.0
        # Tips and overtime are untouched: they were always their own columns.
        assert split.loc[0, "tips1"] == 4_000.0
        assert split.loc[0, "ot2"] == 2_000.0

    def test_the_overlapping_columns_are_gone(self):
        year = tax_data_year([{"id": 1, "wages1": 50_000.0, "sole_prop1": 900.0}])
        split = prepare.split_overlapping_wages(year, 2026)
        for name in ("wages", "wages1", "wages2", "sole_prop", "farm"):
            assert name not in split.columns

    def test_the_pieces_add_back_up_to_what_tax_data_reported(self):
        # This is why the household totals can be dropped: adding the pieces
        # up gives back exactly the totals the file carried, so no calculator
        # loses anything by composing them itself.
        year = tax_data_year(
            [
                {
                    "id": 1,
                    "wages1": 61_234.56, "tips1": 3_210.98, "ot1": 456.78,
                    "wages2": 12_345.67, "tips2": 89.10, "ot2": 1_112.13,
                    "sole_prop1": 4_321.09, "sole_prop2": -876.54,
                    "farm1": -2_468.10, "farm2": 1_357.911,
                },
                {"id": 2},
            ]
        )
        split = prepare.split_overlapping_wages(year, 2026)
        household_wages = (
            split["wages_base1"] + split["tips1"] + split["ot1"]
            + split["wages_base2"] + split["tips2"] + split["ot2"]
        )
        np.testing.assert_allclose(household_wages, year["wages"], rtol=1e-12)
        np.testing.assert_allclose(
            split["sole_prop1"] + split["sole_prop2"], year["sole_prop"]
        )
        np.testing.assert_allclose(split["farm1"] + split["farm2"], year["farm"])

    def test_an_earner_with_more_tips_than_wages_keeps_the_negative_amount(self):
        # About two dozen records a year report tips and overtime adding up to
        # more than the earner's wages, which is a Tax-Data problem. The
        # negative amount is carried and the run is warned.
        year = tax_data_year(
            [
                {"id": 77, "wages1": 1_000.0, "tips1": 900.0, "ot1": 400.0},
                {"id": 88, "wages1": 5_000.0, "tips1": 100.0},
            ]
        )
        with pytest.warns(UserWarning) as warnings_raised:
            split = prepare.split_overlapping_wages(year, 2026)
        assert split.loc[0, "wages_base1"] == -300.0
        message = str(warnings_raised[0].message)
        assert "1 records" in message
        assert "77" in message
        assert "88" not in message

    def test_a_clean_year_warns_about_nothing(self):
        year = tax_data_year(
            [{"id": 1, "wages1": 10_000.0, "tips1": 500.0, "ot1": 100.0}]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            prepare.split_overlapping_wages(year, 2026)

    def test_reading_a_year_of_tax_data_splits_it(self, tmp_path):
        # The split happens where Tax-Data enters the model, so a caller who
        # reads a year never has to remember to do it.
        year = tax_data_year(
            [{"id": 1, "wages1": 20_000.0, "tips1": 2_000.0, "ot1": 500.0}]
        )
        year.to_csv(tmp_path / "tax_units_2026.csv", index=False)
        frame = prepare.read_tax_data_year_frame(tmp_path, 2026)
        assert frame.loc[0, "wages_base1"] == 17_500.0
        assert "wages" not in frame.columns
