"""Reading a parameter file: carried-forward years, `Inf`, and `NA`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from taxmodel.law.errors import LawError
from taxmodel.law.parse import (
    FIRST_PARSE_YEAR,
    carry_forward,
    parse_inf,
    parse_na,
    parse_parameter,
    parse_subparameter,
    parse_years,
)

YEARS = tuple(range(2014, 2031))


def one(block: dict, name: str = "thing", defaults=None):
    return parse_subparameter(block, "param", name, defaults, YEARS)


def test_parse_window_starts_in_2014_whatever_the_run():
    """The law is parsed from 2014 forward so indexation has a year to lag into."""
    assert parse_years(2030)[0] == FIRST_PARSE_YEAR
    assert parse_years(2030)[-1] == 2030
    with pytest.raises(LawError):
        parse_years(2010)


# -- years stated only when they change -------------------------------------


def test_a_value_holds_until_the_next_stated_year():
    sub = one({"value": {"2014": 1000, "2018": 2000, "2026": 1000}})
    values = dict(zip(YEARS, sub.base_value[:, 0]))
    assert values[2017] == 1000
    assert values[2018] == 2000
    assert values[2025] == 2000
    assert values[2026] == 1000
    assert values[2030] == 1000


def test_nothing_is_carried_backward():
    """A parameter that starts in 2021 has no value in 2020, not a zero."""
    sub = one({"value": {"2021": 3600}})
    values = dict(zip(YEARS, sub.base_value[:, 0]))
    assert math.isnan(values[2020])
    assert values[2021] == 3600


def test_a_scalar_holds_for_every_year():
    sub = one({"value": 0.9235})
    assert np.all(sub.base_value == 0.9235)
    assert sub.base_value.shape == (len(YEARS), 1)


def test_carry_forward_leaves_a_gap_before_the_first_stated_year():
    assert carry_forward({"2016": 5}, (2014, 2015, 2016, 2017)) == [None, None, 5, 5]


# -- 'Inf' and 'NA' ---------------------------------------------------------


def test_inf_is_a_limit_that_never_binds():
    """A refundability cap of Inf is a fully refundable credit, not a missing one."""
    sub = one({"value": {"2014": "Inf", "2018": 1400}})
    values = dict(zip(YEARS, sub.base_value[:, 0]))
    assert values[2014] == math.inf
    assert values[2018] == 1400


def test_inf_in_one_slot_makes_the_whole_schedule_numeric():
    sub = one({"value": {"2018": [200000, "Inf"]}})
    assert sub.base_value[YEARS.index(2018)].tolist() == [200000.0, math.inf]


def test_parse_inf_leaves_ordinary_numbers_alone():
    assert parse_inf([0, 9525, 38700]) == [0, 9525, 38700]
    assert parse_inf({"2018": [0, "Inf"]}) == {"2018": [0.0, math.inf]}


def test_na_in_an_indexation_field_freezes_that_slot_alone():
    """The donut hole: the taxable maximum is indexed, the bracket above it is not."""
    sub = one(
        {
            "value": [0, 60600, 400000],
            "i_measure": {"1991": "awi"},
            "i_base_year": [1991, 1991, "NA"],
            "i_direction": [1, 1, "NA"],
            "i_increment": [300, 300, "NA"],
        }
    )
    assert sub.i_base_year[0].tolist()[:2] == [1991.0, 1991.0]
    assert math.isnan(sub.i_base_year[0, 2])
    assert math.isnan(sub.i_increment[0, 2])


def test_parse_na_coerces_the_whole_field():
    assert parse_na([1991, "NA"])[0] == 1991.0
    assert math.isnan(parse_na([1991, "NA"])[1])


def test_a_value_that_is_neither_a_number_nor_a_sentinel_is_refused():
    with pytest.raises(LawError, match="neither a number"):
        one({"value": "chained_cpi"})


# -- indexation fields ------------------------------------------------------


def test_no_measure_means_no_indexation_whatever_else_the_block_says():
    """A reform that drops i_measure has un-indexed the parameter (lint's job)."""
    sub = one({"value": 40000, "i_base_year": 2017, "i_direction": -1, "i_increment": 100})
    assert not sub.indexed
    assert sub.i_base_year is None


def test_default_pulls_the_files_house_rule():
    defaults = {
        "i_measure": {"1987": "cpi", "2017": "chained_cpi"},
        "i_base_year": 2017,
        "i_direction": -1,
        "i_increment": 100,
    }
    sub = one(
        {
            "value": 1400,
            "i_measure": "default",
            "i_base_year": "default",
            "i_direction": "default",
            "i_increment": 25,
        },
        defaults=defaults,
    )
    assert sub.measure[2016] == "cpi"
    assert sub.measure[2018] == "chained_cpi"
    assert sub.i_base_year[0, 0] == 2017
    assert sub.i_increment[0, 0] == 25


def test_indexation_fields_out_of_order_are_refused():
    """They are paired with the defaults by position, not by name."""
    defaults = {
        "i_measure": {"2017": "cpi"},
        "i_base_year": 2017,
        "i_direction": -1,
        "i_increment": 100,
    }
    with pytest.raises(LawError, match="by position"):
        one(
            {
                "value": 1400,
                "i_measure": "default",
                "i_direction": "default",
                "i_base_year": "default",
                "i_increment": "default",
            },
            defaults=defaults,
        )


def test_a_measure_with_no_start_year_is_refused():
    with pytest.raises(LawError, match="no start year"):
        one(
            {
                "value": 1400,
                "i_measure": "chained_cpi",
                "i_base_year": 2017,
                "i_direction": -1,
                "i_increment": 100,
            }
        )


def test_a_measure_written_out_is_a_freeze():
    sub = one(
        {
            "value": 1400,
            "i_measure": {"2014": "cpi", "2020": "NA"},
            "i_base_year": 2017,
            "i_direction": -1,
            "i_increment": 100,
        }
    )
    assert sub.measure[2019] == "cpi"
    assert sub.measure[2020] is None
    assert sub.measure[2025] is None


def test_remove_indexation_is_a_declaration_not_a_field():
    """It says the un-indexing was meant; it carries nothing into the law."""
    sub = one({"value": 40000, "remove_indexation": True})
    assert not sub.indexed
    assert sub.base_value[0, 0] == 40000


# -- whole files ------------------------------------------------------------


def test_reserved_keys_are_not_subparameters():
    parsed = parse_parameter(
        {
            "indexation_defaults": {"i_measure": {"2017": "cpi"}},
            "filing_status_mapper": {"brackets": {"1": "brackets_single"}},
            "brackets_single": {"value": [0, 9525]},
        },
        "ord",
        YEARS,
    )
    assert set(parsed) == {"brackets_single"}


def test_a_schedule_that_gains_a_slot_is_ragged_by_year():
    """Slot 2 exists from 2026; before then the law says nothing about it."""
    sub = one({"value": {"2014": [0.1], "2026": [0.1, 0.35]}})
    assert sub.base_value.shape[1] == 2
    assert math.isnan(sub.base_value[YEARS.index(2025), 1])
    assert sub.base_value[YEARS.index(2026), 1] == 0.35
