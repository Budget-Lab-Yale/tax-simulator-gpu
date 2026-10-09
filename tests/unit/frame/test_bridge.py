"""The Tax-Data bridge: converting a vintage once and reading it back.

Nothing here touches real Tax-Data. Each test writes a handful of records to
a CSV in the shape Tax-Data publishes — overlapping wage columns and all —
converts them, and checks that what the model reads out is exactly what it
would have read had it parsed the text itself. The real vintage is held to
the same standard in a job (tests/bridge_byte_identity.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxmodel.frame import bridge, population, prepare
from tests.unit.frame import population_checks

# One household with tips and overtime inside its wages, one with a second
# earner, one that reports more tips and overtime than wages (a Tax-Data
# defect), and one that only exists in the later year.
RECORDS_2026 = [
    {"id": 11, "wages1": 50_000.0, "tips1": 4_000.0, "ot1": 1_000.0, "weight": 100.0},
    {
        "id": 12,
        "wages1": 80_000.0,
        "wages2": 30_000.0,
        "tips2": 2_000.0,
        "sole_prop1": 5_000.0,
        "weight": 250.0,
    },
    {"id": 13, "wages1": 1_000.0, "tips1": 900.0, "ot1": 400.0, "weight": 75.0},
]
# The cost basis of the two asset classes household 12 holds. Tax-Data
# publishes a basis next to the market value for the five classes the
# capital-gains model tracks; the bridge has to carry them through unchanged,
# because the model differences them to get the stock of unrealized gain.
BASIS_2026 = {
    12: {
        "value.equities": 2_000_000.0,
        "basis.equities": 750_000.25,
        "value.pass_throughs": 500_000.0,
        "basis.pass_throughs": 500_000.0,
    },
}
RECORDS_2027 = [
    {"id": 11, "wages1": 52_000.0, "tips1": 4_100.0, "ot1": 1_100.0, "weight": 101.0},
    {"id": 13, "wages1": 1_100.0, "tips1": 950.0, "ot1": 450.0, "weight": 76.0},
    {"id": 14, "wages1": 20_000.0, "weight": 40.0},
]

# Every column `read_tax_data_year_frame` has to see, whether or not a record
# uses it. The household totals are the per-earner amounts added up, which is
# how Tax-Data writes them.
NUMERIC_COLUMNS = (
    "wages1",
    "wages2",
    "tips1",
    "tips2",
    "ot1",
    "ot2",
    "sole_prop1",
    "sole_prop2",
    "farm1",
    "farm2",
    "weight",
    # Filing status: 1 is Single and 2 is Married filing jointly. Written for
    # every record because the year loop reads the law by filing status and
    # stops on a code outside 1 through 4.
    "filing_status",
    # The two asset classes this file gives a basis to. Listed so that a
    # record which does not hold them still has the columns, at zero.
    "value.equities",
    "basis.equities",
    "value.pass_throughs",
    "basis.pass_throughs",
)


def write_tax_data_year(root, year: int, records: list[dict]) -> None:
    """One year of Tax-Data as it is published: text, with overlapping columns."""
    rows = []
    for record in records:
        row = {name: 0.0 for name in NUMERIC_COLUMNS}
        row["id"] = record["id"]
        row.update(record)
        row["wages"] = row["wages1"] + row["wages2"]
        row["sole_prop"] = row["sole_prop1"] + row["sole_prop2"]
        row["farm"] = row["farm1"] + row["farm2"]
        row["filing_status"] = 2 if row["wages2"] else 1
        # A text column and a true/false column with blanks, because those are
        # the two shapes that survive a Parquet round trip least obviously.
        for name, amount in BASIS_2026.get(row["id"], {}).items():
            row[name] = amount
        row["forbes_name"] = "A named billionaire" if row["id"] == 12 else None
        row["sstb_part"] = True if row["sole_prop1"] else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / f"tax_units_{year}.csv", index=False)


@pytest.fixture
def tax_data(tmp_path):
    """A two-year Tax-Data vintage in the published CSV shape."""
    root = tmp_path / "Tax-Data"
    root.mkdir()
    write_tax_data_year(root, 2026, RECORDS_2026)
    write_tax_data_year(root, 2027, RECORDS_2027)
    return root


@pytest.fixture
def converted(tax_data, tmp_path):
    """That vintage, run through the bridge."""
    out = tmp_path / "Tax-Data-Bridge"
    bridge.convert_vintage(tax_data, out)
    return out


def test_the_bridge_converts_every_year_the_vintage_publishes(tax_data, converted):
    assert bridge.available_years(tax_data) == (2026, 2027)
    assert prepare.bridge_year_path(converted, 2026).exists()
    assert prepare.bridge_year_path(converted, 2027).exists()


def test_a_converted_year_is_what_parsing_the_text_gives(tax_data, converted):
    parsed = prepare.read_tax_data_year_frame(tax_data, 2026)
    parsed = parsed.drop(
        columns=[c for c in prepare.NEVER_READ_COLUMNS if c in parsed.columns]
    )
    parsed = parsed.merge(
        prepare.keyed_draws(parsed["id"].to_numpy()), on="id", how="left"
    )
    from_bridge = population_checks.read_bridge_year_frame(converted, 2026)
    assert population_checks.frame_differences(parsed, from_bridge) == []


def test_the_whole_prepared_population_is_identical_either_way(tax_data, converted):
    years = [2026, 2027]
    from_text = population.prepare_population(tax_data, years)
    from_bridge = population.prepare_population(converted, years)

    assert list(from_text.universe.ids) == list(from_bridge.universe.ids)
    assert (from_text.universe.valid == from_bridge.universe.valid).all()
    assert (
        population_checks.frame_differences(population_checks.draws_frame(from_text), population_checks.draws_frame(from_bridge))
        == []
    )
    assert population_checks.population_differences(from_text, from_bridge) == []


def test_the_wage_split_leaves_no_column_counting_the_same_dollar_twice(converted):
    frame = population_checks.read_bridge_year_frame(converted, 2026)
    for retired in prepare.OVERLAPPING_COLUMNS:
        assert retired not in frame.columns
    # The columns nothing in the model reads are not written either.
    for unread in prepare.NEVER_READ_COLUMNS:
        assert unread not in frame.columns
    household_11 = frame[frame["id"] == 11].iloc[0]
    # 50,000 of pay, of which 4,000 is tips and 1,000 is overtime.
    assert household_11["wages_base1"] == 45_000.0
    assert household_11["tips1"] == 4_000.0
    assert household_11["ot1"] == 1_000.0


def test_negative_base_wages_are_carried_through_and_named_in_the_manifest(converted):
    frame = population_checks.read_bridge_year_frame(converted, 2026)
    # Household 13 reports 1,000 of wages containing 900 of tips and 400 of
    # overtime, which is impossible; Tax-Data says it anyway and the model
    # taxes what Tax-Data says, so base pay comes through at minus 300.
    assert frame.loc[frame["id"] == 13, "wages_base1"].iloc[0] == -300.0

    manifest = prepare.read_bridge_manifest(converted)
    entry = manifest["years"][2026]
    assert entry["records"] == 3
    assert entry["negative_base_wages_earner_1"] == 1
    assert entry["negative_base_wages_earner_2"] == 0
    assert entry["records_with_negative_base_wages"] == 1
    assert entry["negative_base_wage_ids"]["earner_1"] == [13]


def test_the_manifest_says_what_was_converted_and_by_what_code(tax_data, converted):
    manifest = prepare.read_bridge_manifest(converted)
    assert manifest["source_vintage"] == str(tax_data)
    assert manifest["random_draw_seed"] == prepare.DEFAULT_SEED
    assert manifest["random_draw_columns"] == list(prepare.DRAW_COLUMNS)
    assert manifest["bridge_code"]["module"] == "taxmodel.frame.bridge"
    assert set(manifest["years"]) == {2026, 2027}


def test_the_form_of_a_population_folder_is_read_off_the_folder(
    tax_data, converted, tmp_path
):
    assert prepare.input_form(tax_data) == prepare.TAX_DATA_CSV_INPUT
    assert prepare.input_form(converted) == prepare.BRIDGE_INPUT
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="neither a bridge manifest"):
        prepare.input_form(empty)


def test_a_run_cannot_ask_for_draws_the_conversion_did_not_make(converted):
    with pytest.raises(ValueError, match="converted with seed"):
        population.prepare_population(converted, [2026], seed=prepare.DEFAULT_SEED + 1)


# ---------------------------------------------------------------------------
# Reading one year at a time instead of holding every year at once
# ---------------------------------------------------------------------------


def test_a_year_read_from_disk_matches_the_year_held_in_the_block(converted):
    years = [2026, 2027]
    block = population.population_from_bridge(converted, years)
    from_disk = population.population_from_bridge_by_year(converted, years)

    assert from_disk.universe.years == block.universe.years
    assert from_disk.universe.ids.tobytes() == block.universe.ids.tobytes()
    assert from_disk.universe.valid.tobytes() == block.universe.valid.tobytes()
    assert from_disk.field_names == block.field_names
    assert from_disk.dropped_text_columns == block.dropped_text_columns
    for name, draw in block.draws.items():
        assert from_disk.draws[name].tobytes() == draw.tobytes()

    for year in years:
        held = block.year(year)
        read = from_disk.year(year)
        assert read.valid.tobytes() == held.valid.tobytes()
        assert set(read.values) == set(held.values)
        for name, values in held.values.items():
            assert read.values[name].dtype == values.dtype
            assert read.values[name].tobytes() == values.tobytes(), name


def test_a_year_read_from_disk_can_be_limited_to_the_columns_asked_for(converted):
    years = [2026, 2027]
    block = population.population_from_bridge(converted, years)
    from_disk = population.population_from_bridge_by_year(converted, years)

    read = from_disk.year(2027, columns=("weight", "filing_status"))
    assert set(read.values) == {"weight", "filing_status"}
    held = block.year(2027)
    for name in ("weight", "filing_status"):
        assert read.values[name].tobytes() == held.values[name].tobytes()


def test_the_year_loops_own_input_is_the_same_read_either_way(converted):
    from taxmodel.years import one_year

    years = [2026, 2027]
    block = population.population_from_bridge(converted, years)
    from_disk = population.population_from_bridge_by_year(converted, years)

    held = one_year.population_arrays_for_years(block, years)
    read = one_year.population_arrays_for_years(from_disk, years)
    assert set(read) == set(held)
    for name, values in held.items():
        assert np.asarray(read[name]).tobytes() == np.asarray(values).tobytes(), name


def test_a_run_reading_year_by_year_also_checks_the_conversion_seed(converted):
    with pytest.raises(ValueError, match="converted with seed"):
        population.prepare_population(
            converted,
            [2026],
            seed=prepare.DEFAULT_SEED + 1,
            read_every_year_up_front=False,
        )


def test_a_run_records_which_vintage_its_population_came_from(tax_data, converted):
    facts = prepare.population_provenance(converted)
    assert facts["population_form"] == prepare.BRIDGE_INPUT
    assert facts["source_vintage"] == str(tax_data)
    assert facts["random_draw_seed"] == prepare.DEFAULT_SEED


def test_the_bridge_carries_the_per_asset_cost_basis_through_unchanged(converted):
    """The stock of unrealized gain is value less basis, so basis has to survive.

    Exact equality, not a tolerance: converting a vintage changes the
    container the numbers sit in and nothing else.
    """
    frame = population_checks.read_bridge_year_frame(converted, 2026)
    household_12 = frame[frame["id"] == 12].iloc[0]
    for name, amount in BASIS_2026[12].items():
        assert household_12[name] == amount
    # A record that reports no balance sheet at all still gets the columns,
    # at zero, so every year's frame has the same shape.
    household_11 = frame[frame["id"] == 11].iloc[0]
    assert household_11["basis.equities"] == 0.0
