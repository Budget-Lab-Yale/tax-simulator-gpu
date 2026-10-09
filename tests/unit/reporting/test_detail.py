"""The detail request: what may be asked for, and what the files look like.

The run-level checks -- that asking for detail changes no total, and that the
detail re-adds to the run's own totals -- live in
`tests/unit/years/test_one_year_detail.py`, because they need a run. This module is
about the request object, the refusals, and the Parquet files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxmodel.reporting import detail
from taxmodel.taxes import marginal_rates, statutory_order
from taxmodel.years import world_names


def weighted_totals(
    household_detail: detail.HouseholdDetail, world: str
) -> dict[str, np.ndarray]:
    """Re-add one world's detail to population totals, one number per year.

    This is the check that the detail files say the same thing the run's own
    totals say: the weight times the column, added up over every exported row.
    """
    if world not in household_detail.worlds:
        raise KeyError(f"the run's detail holds no {world!r} world")
    return {
        name: np.sum(
            household_detail.weight
            * np.asarray(household_detail.columns[(world, name)]),
            axis=1,
        )
        for name in household_detail.column_names(world)
    }

YEARS = (2026, 2027, 2028)


def a_detail(years=YEARS) -> detail.HouseholdDetail:
    """Two households over three years, with numbers that are easy to add up."""
    n_years, n_slots = len(years), 2
    shape = (n_years, n_slots)
    return detail.HouseholdDetail(
        years=tuple(years),
        household_id=np.broadcast_to(np.array([11, 22]), shape).copy(),
        weight=np.broadcast_to(np.array([2.0, 3.0]), shape).copy(),
        valid=np.ones(shape, dtype=bool),
        columns={
            ("mechanical", "liab_iit_net"): np.arange(
                n_years * n_slots, dtype=np.float64
            ).reshape(shape),
            ("baseline", "liab_iit_net"): np.zeros(shape),
        },
    )


# ---------------------------------------------------------------------------
# The vocabulary and its refusals
# ---------------------------------------------------------------------------


def test_every_world_offers_the_calculators_results():
    for world in world_names.WORLD_NAMES:
        offered = detail.vocabulary(world)
        assert "liab_iit_net" in offered
        assert "expanded_inc" in offered
        assert set(statutory_order.CALCULATED_COLUMNS) <= set(offered)


def test_the_priced_rates_are_offered_from_the_mechanical_world_only():
    mechanical = set(detail.vocabulary("mechanical"))
    assert set(marginal_rates.MARGINAL_RATE_COLUMNS) <= mechanical
    # The twenty-seven priced dollars, each priced two ways --
    # income tax alone and income tax plus the employee's payroll tax -- under
    # each of the two laws.
    assert len(marginal_rates.MARGINAL_RATE_COLUMNS) == 27 * 2 * 2
    for world in world_names.WORLD_NAMES:
        if world != "mechanical":
            assert not set(marginal_rates.MARGINAL_RATE_COLUMNS) & set(
                detail.vocabulary(world)
            )


def test_an_input_column_is_refused_and_told_where_to_get_it():
    with pytest.raises(KeyError) as raised:
        detail.DetailRequest(
            columns_by_world={"baseline": ("wages_base1",)}, years=(2026,)
        )
    message = str(raised.value)
    assert "wages_base1" in message
    assert "Tax-Data" in message


def test_a_name_the_model_does_not_calculate_is_refused_by_name():
    with pytest.raises(KeyError) as raised:
        detail.DetailRequest(
            columns_by_world={"baseline": ("liab_iit_net", "liab_iit_nett")},
            years=(2026,),
        )
    message = str(raised.value)
    assert "liab_iit_nett" in message
    assert "not a column the model calculates" in message


def test_the_wealth_and_estate_rates_on_the_balance_sheet_can_be_asked_for():
    # The wealth tax's and the estate tax's cost per dollar of each kind of
    # asset and each kind of debt. They replaced the two rates the wealth
    # channel used to measure by re-running a calculator on a bumped frame,
    # which were never exportable.
    request = detail.DetailRequest(
        columns_by_world={
            "mechanical": (
                "reform_law_mtr_wealth_tax_on_value.equities",
                "baseline_law_mtr_estate_tax_on_value.primary_mortgage",
            )
        },
        years=(2026,),
    )
    assert request.columns_by_world["mechanical"] == (
        "reform_law_mtr_wealth_tax_on_value.equities",
        "baseline_law_mtr_estate_tax_on_value.primary_mortgage",
    )


def test_a_rate_asked_for_from_the_wrong_world_says_which_world_has_it():
    with pytest.raises(KeyError) as raised:
        detail.DetailRequest(
            columns_by_world={"baseline": ("reform_law_mtr_kg_lt",)}, years=(2026,)
        )
    assert "mechanical" in str(raised.value)


def test_an_unknown_world_is_refused_with_the_five_names():
    with pytest.raises(KeyError) as raised:
        detail.DetailRequest(
            columns_by_world={"static": ("liab_iit_net",)}, years=(2026,)
        )
    message = str(raised.value)
    assert "static" in message
    for world in world_names.WORLD_NAMES:
        assert world in message


def test_the_model_has_five_worlds():
    assert world_names.WORLD_NAMES == (
        "baseline",
        "first_order",
        "mechanical_ex_saving",
        "mechanical",
        "conventional",
    )


def test_an_empty_or_repetitive_request_is_refused():
    with pytest.raises(ValueError):
        detail.DetailRequest(columns_by_world={}, years=(2026,))
    with pytest.raises(ValueError):
        detail.DetailRequest(columns_by_world={"baseline": ()}, years=(2026,))
    with pytest.raises(ValueError):
        detail.DetailRequest(columns_by_world={"baseline": ("agi",)}, years=())
    with pytest.raises(ValueError):
        detail.DetailRequest(
            columns_by_world={"baseline": ("agi", "agi")}, years=(2026,)
        )
    with pytest.raises(ValueError):
        detail.DetailRequest(
            columns_by_world={"baseline": ("agi",)}, years=(2026, 2026)
        )


def test_the_common_case_constructor_asks_each_world_for_the_same_columns():
    request = detail.DetailRequest.same_columns_from_each_world(
        columns=("agi", "liab_iit_net"),
        worlds=("mechanical", "baseline"),
        years=(2027, 2026),
    )
    assert request.years == (2026, 2027)
    # The worlds come back in the grid's order, whatever order they were asked
    # in.
    assert request.worlds == ("baseline", "mechanical")
    assert request.keys() == (
        ("baseline", "agi"),
        ("baseline", "liab_iit_net"),
        ("mechanical", "agi"),
        ("mechanical", "liab_iit_net"),
    )


# ---------------------------------------------------------------------------
# Keeping the years asked for
# ---------------------------------------------------------------------------


def test_only_the_years_asked_for_come_out():
    request = detail.DetailRequest(
        columns_by_world={"mechanical": ("liab_iit_net",)}, years=(2028,)
    )
    every_year = np.arange(len(YEARS) * 2, dtype=np.float64).reshape(len(YEARS), 2)
    kept = detail.assemble(
        request,
        YEARS,
        household_id=np.broadcast_to(np.array([11, 22]), (len(YEARS), 2)),
        weight=np.ones((len(YEARS), 2)),
        valid=np.ones((len(YEARS), 2), dtype=bool),
        columns_by_year={("mechanical", "liab_iit_net"): every_year},
    )
    assert kept.years == (2028,)
    np.testing.assert_array_equal(
        kept.columns[("mechanical", "liab_iit_net")], every_year[2:3]
    )


def test_a_year_the_run_did_not_simulate_is_refused():
    request = detail.DetailRequest(
        columns_by_world={"mechanical": ("liab_iit_net",)}, years=(2040,)
    )
    with pytest.raises(ValueError) as raised:
        detail.assemble(
            request,
            YEARS,
            household_id=np.zeros((len(YEARS), 2)),
            weight=np.ones((len(YEARS), 2)),
            valid=np.ones((len(YEARS), 2), dtype=bool),
            columns_by_year={
                ("mechanical", "liab_iit_net"): np.zeros((len(YEARS), 2))
            },
        )
    assert "2040" in str(raised.value)


# ---------------------------------------------------------------------------
# The files
# ---------------------------------------------------------------------------


def test_one_parquet_file_per_world_named_for_the_world(tmp_path):
    written = detail.write_parquet(a_detail(), tmp_path)

    assert [file.world for file in written] == ["baseline", "mechanical"]
    assert {file.path.name for file in written} == {
        "baseline.parquet",
        "mechanical.parquet",
    }
    for file in written:
        assert file.path.exists()
        assert file.years == YEARS
        assert file.columns == ("liab_iit_net",)
        # Two households in each of three years.
        assert file.n_rows == 6


def test_a_detail_file_carries_the_household_id_the_year_and_the_weight(tmp_path):
    written = detail.write_parquet(a_detail(), tmp_path)
    mechanical = next(file for file in written if file.world == "mechanical")

    frame = pd.read_parquet(mechanical.path)
    assert list(frame.columns) == [
        "household_id",
        "year",
        "weight",
        "liab_iit_net",
        "valid",
    ]
    assert sorted(set(frame["household_id"])) == [11, 22]
    assert sorted(set(frame["year"])) == list(YEARS)
    # Every household id appears once per year, so a join to the Tax-Data
    # bridge on (household_id, year) has one row to match each way.
    assert not frame.duplicated(subset=["household_id", "year"]).any()


def test_the_calculated_columns_are_written_alphabetically():
    """A stable column order, because the compiled loop's own order is sorted."""
    shape = (len(YEARS), 2)
    detail_out = detail.HouseholdDetail(
        years=YEARS,
        household_id=np.broadcast_to(np.array([11, 22]), shape).copy(),
        weight=np.ones(shape),
        valid=np.ones(shape, dtype=bool),
        columns={
            ("mechanical", "liab_iit_net"): np.zeros(shape),
            ("mechanical", "agi"): np.zeros(shape),
        },
    )
    assert detail_out.column_names("mechanical") == ("agi", "liab_iit_net")
    assert list(detail.table(detail_out, "mechanical").columns) == [
        "household_id",
        "year",
        "weight",
        "agi",
        "liab_iit_net",
        "valid",
    ]


def test_the_weighted_total_of_a_detail_column_is_a_plain_sum():
    detail_out = a_detail()
    totals = weighted_totals(detail_out, "mechanical")
    column = detail_out.columns[("mechanical", "liab_iit_net")]
    np.testing.assert_array_equal(
        totals["liab_iit_net"], np.sum(detail_out.weight * column, axis=1)
    )


def test_a_world_is_built_only_when_it_is_asked_for(tmp_path, monkeypatch):
    """One detail table in memory at a time, not one per world.

    Each of these tables holds a row per household per year -- about a
    gigabyte over a thirty-year run -- so building every world's before the
    first one is written would multiply what a detailed run needs.
    """
    built = []
    original = detail.table

    def note_which_world(household_detail, world):
        built.append(world)
        return original(household_detail, world)

    monkeypatch.setattr(detail, "table", note_which_world)

    worlds = iter(detail.tables_to_write(a_detail(), tmp_path))
    assert built == []
    next(worlds)
    assert built == ["baseline"]
    next(worlds)
    assert built == ["baseline", "mechanical"]
