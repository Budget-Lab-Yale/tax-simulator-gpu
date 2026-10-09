"""Compiling a law tree into arrays."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from taxmodel.config.law_layers import resolve_law
from taxmodel.law import MAX_BRACKETS, compile_law
from taxmodel.law.compile import units_of
from taxmodel.law.errors import LawError
from tests.conftest import CONFIG_ROOT

YEARS = tuple(range(2025, 2031))


@pytest.fixture
def simple(law_tree):
    """One file with a mapper, an indexed schedule, and a plain switch."""
    law_tree.write(
        "default",
        "ord",
        {
            "indexation_defaults": {
                "i_measure": {"2000": "cpi"},
                "i_base_year": [2020, 2020, 2020],
                "i_direction": [-1, -1, -1],
                "i_increment": [25, 25, 25],
            },
            "filing_status_mapper": {
                "brackets": {
                    "1": "brackets_single",
                    "2": "brackets_married",
                    "3": "brackets_married / 2",
                    "4": "brackets_single",
                }
            },
            "rates": {"value": [0.1, 0.22, 0.37]},
            "brackets_single": {
                "value": [0, 10000, 200000],
                "i_measure": "default",
                "i_base_year": "default",
                "i_direction": "default",
                "i_increment": "default",
            },
            "brackets_married": {"value": [0, 20000, 400000]},
            "repeal_hoh": {"value": 0},
        },
    )
    return law_tree


def test_shapes_are_year_by_filing_status_by_slot(simple, flat_index):
    law = compile_law(simple.root, YEARS, flat_index)
    assert law.values["ord.rates"].shape == (len(YEARS), 4, MAX_BRACKETS)
    assert law.years == YEARS


def test_a_mapper_input_is_not_a_compiled_field(simple, flat_index):
    """`brackets_single` fed the mapper; only `brackets` survives."""
    law = compile_law(simple.root, YEARS, flat_index)
    assert "ord.brackets" in law.schema
    assert "ord.brackets_single" not in law.schema
    assert law.schema["ord.brackets"].from_mapper
    assert law.schema["ord.brackets"].varies_by_filing_status


def test_a_parameter_with_no_mapper_is_the_same_for_every_status(simple, flat_index):
    law = compile_law(simple.root, YEARS, flat_index)
    rates = law.values["ord.rates"]
    for status in range(4):
        assert rates[0, status, :3].tolist() == [0.1, 0.22, 0.37]
    assert not law.schema["ord.rates"].varies_by_filing_status


def test_the_separate_return_gets_half_the_joint_brackets(simple, flat_index):
    law = compile_law(simple.root, YEARS, flat_index)
    brackets = law.values["ord.brackets"]
    joint = brackets[0, 1, :3]
    separate = brackets[0, 2, :3]
    assert (separate * 2).tolist() == joint.tolist()


def test_padding_is_nan_and_the_mask_says_where_the_law_stops(simple, flat_index):
    law = compile_law(simple.root, YEARS, flat_index)
    rates = law.values["ord.rates"]
    active = law.active["ord.rates"]
    assert active[0, 0, :3].all()
    assert not active[0, 0, 3:].any()
    assert np.isnan(rates[0, 0, 3:]).all()
    assert law.schema["ord.rates"].slots_used == 3


def test_a_year_before_the_law_says_anything_is_inactive(law_tree, flat_index):
    law_tree.write("default", "ctc", {"value": {"value": {"2028": 2000}}})
    law = compile_law(law_tree.root, YEARS, flat_index)
    active = law.active["ctc.value"]
    assert not active[YEARS.index(2027), 0, 0]
    assert active[YEARS.index(2028), 0, 0]


def test_an_indexed_bracket_grows_and_is_rounded_down(simple, flat_index):
    """2% a year from a 2020 base, floored to the next $25 below."""
    law = compile_law(simple.root, YEARS, flat_index)
    single_2026 = law.values["ord.brackets"][YEARS.index(2026), 0, 1]
    assert single_2026 == math.floor(10000 * 1.02**5 / 25) * 25


def test_switches_and_counts_are_not_differentiable(simple, flat_index):
    law = compile_law(simple.root, YEARS, flat_index)
    assert not law.schema["ord.repeal_hoh"].differentiable
    assert not law.differentiable_mask("ord.repeal_hoh").any()
    assert law.schema["ord.rates"].differentiable
    assert law.differentiable_mask("ord.rates")[0, 0, :3].all()


def test_an_unbounded_limit_is_active_but_out_of_the_gradient(law_tree, flat_index):
    """`0 * Inf` in an adjoint is the bug class the mask exists to prevent."""
    law_tree.write("default", "ctc", {"po_range": {"value": ["Inf", "Inf"]}})
    law = compile_law(law_tree.root, YEARS, flat_index)
    assert law.active["ctc.po_range"][0, 0, 0]
    assert not law.differentiable_mask("ctc.po_range")[0, 0, 0]


def test_a_schedule_past_the_compiled_maximum_is_refused(law_tree, flat_index):
    law_tree.write(
        "default", "estate", {"rates": {"value": [0.1] * (MAX_BRACKETS + 1)}}
    )
    with pytest.raises(LawError, match="past the compiled maximum"):
        compile_law(law_tree.root, YEARS, flat_index)


def test_units_come_from_the_declared_rules():
    assert units_of("rates") == "rate"
    assert units_of("rate1") == "rate"
    assert units_of("po_rate_0") == "rate"
    assert units_of("kg_top_rate_addon") == "rate"
    assert units_of("div_top_rate_addon") == "rate"
    assert units_of("po_thresh_married") == "dollars"
    assert units_of("med_floor_agi") == "share of AGI"
    assert units_of("old_age_limit") == "count"
    assert units_of("alimony_repeal_year") == "calendar year"
    assert units_of("repeal_hoh") == "indicator"
    assert units_of("kg_death_regime_equities") == "regime code"


# -- the written table ------------------------------------------------------


def test_the_csv_is_long_in_year_and_filing_status(simple, flat_index):
    law = compile_law(simple.root, YEARS, flat_index)
    header, rows = law.to_frame()
    assert header[:2] == ["year", "filing_status"]
    assert len(rows) == len(YEARS) * 4
    assert rows[0][:2] == [2025, 1]
    assert rows[1][:2] == [2025, 2]


def test_a_scalar_field_has_no_column_index_and_a_schedule_does(simple, flat_index):
    law = compile_law(simple.root, YEARS, flat_index)
    columns = law.column_names()
    assert "ord.repeal_hoh" in columns
    assert "ord.rates1" in columns and "ord.rates3" in columns
    assert "ord.rates" not in columns


def test_the_frozen_arrays_round_trip(simple, flat_index, tmp_path):
    law = compile_law(simple.root, YEARS, flat_index)
    law.save_npz(tmp_path / "law.npz")
    loaded = np.load(tmp_path / "law.npz")
    assert loaded["years"].tolist() == list(YEARS)
    np.testing.assert_array_equal(
        loaded["value/ord.rates"], law.values["ord.rates"], strict=True
    )


# -- this repository's law tree -----------------------------------------------


def test_every_law_folder_in_the_tree_resolves():
    """Every reform folder under tax_law/alternatives resolves without an error."""
    alternatives = os.path.join(CONFIG_ROOT, "tax_law", "alternatives")
    layers = [
        os.path.relpath(folder, alternatives)
        for folder, _, files in os.walk(alternatives)
        if folder != alternatives and any(f.endswith(".yaml") for f in files)
    ]
    assert layers
    for layer in sorted(layers):
        resolve_law(CONFIG_ROOT, layer)
