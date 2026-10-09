"""The filing-status mapper language, and every mapper in the law tree."""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import yaml

from taxmodel.law import mappers
from taxmodel.law.errors import LawError
from tests.conftest import CONFIG_ROOT


def test_a_bare_name_is_the_subparameter_itself():
    assert mappers.evaluate("brackets_single", {"brackets_single": 9525.0}) == 9525.0


def test_the_separate_return_gets_half_the_joint_schedule():
    joint = np.array([0.0, 19050.0, 77400.0])
    got = mappers.evaluate("brackets_married / 2", {"brackets_married": joint})
    assert got.tolist() == [0.0, 9525.0, 38700.0]


def test_the_joint_threshold_is_the_single_one_doubled():
    assert mappers.evaluate("po_thresh_single * 2", {"po_thresh_single": 200000.0}) == 400000.0


def test_a_marriage_adjustment_adds_a_second_subparameter():
    got = mappers.evaluate(
        "po_thresh_0 + po_thresh_marriage_adj",
        {"po_thresh_0": 10000.0, "po_thresh_marriage_adj": 7000.0},
    )
    assert got == 17000.0


def test_multiplication_binds_tighter_than_addition():
    assert mappers.evaluate("a + b * 2", {"a": 1.0, "b": 3.0}) == 7.0


def test_parentheses_override_precedence():
    assert mappers.evaluate("(a + b) * 2", {"a": 1.0, "b": 3.0}) == 8.0


def test_subtraction_and_a_leading_minus():
    assert mappers.evaluate("a - b", {"a": 5.0, "b": 2.0}) == 3.0
    assert mappers.evaluate("-a", {"a": 5.0}) == -5.0


def test_a_zero_multiplier_zeroes_the_schedule():
    """`sl_limit_single * 0`: the separate return cannot claim the deduction."""
    assert mappers.evaluate("sl_limit_single * 0", {"sl_limit_single": 2500.0}) == 0.0


def test_infinite_operands_survive_the_arithmetic():
    """A phaseout range of Inf halved is still a phaseout that never ends."""
    assert mappers.evaluate("po_range_married / 2", {"po_range_married": np.inf}) == np.inf


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "max(a, b)",
        "a ** 2",
        "a if b else c",
        "a % 2",
        "a; b",
        "a[0]",
        "a.value",
        "a & b",
    ],
)
def test_the_language_is_the_language_and_nothing_else(expression):
    with pytest.raises(LawError):
        mappers.evaluate(expression, {"a": 1.0, "b": 2.0, "c": 3.0})


def test_an_unknown_name_names_the_file_it_is_missing_from():
    with pytest.raises(LawError, match="not a subparameter"):
        mappers.evaluate("brackets_widow", {"brackets_single": 1.0})


def test_an_unclosed_parenthesis_is_refused():
    with pytest.raises(LawError, match="unclosed"):
        mappers.evaluate("(a + b", {"a": 1.0, "b": 2.0})


def test_an_operand_is_read_with_or_without_spaces_around_the_operator():
    assert mappers.identifiers("brackets_married / 2") == {"brackets_married"}
    assert mappers.identifiers("brackets_married/2") == {"brackets_married"}


# -- every mapper in the law tree -------------------------------------------


def test_every_mapper_in_the_law_tree_evaluates():
    """Walk the real law tree and evaluate each mapper against unit operands.

    Unit operands make the result a statement about the arithmetic, not the
    law: a mapper reading `brackets_married / 2` returns 0.5, and any mapper
    the grammar cannot read raises instead. This is the cross-check that the
    restricted language covers the tree, not just the files the tests picked.
    """
    files = sorted(
        glob.glob(os.path.join(CONFIG_ROOT, "tax_law", "**", "*.yaml"), recursive=True)
    )
    assert len(files) > 1000, "expected the full law tree"

    seen = 0
    distinct: set[str] = set()
    for path in files:
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            continue
        mapper = raw.get("filing_status_mapper") or {}
        subparams = {
            name: 1.0
            for name, block in raw.items()
            if name not in ("indexation_defaults", "filing_status_mapper")
        }
        # A reform file may restate the mapper while leaving its operands in the
        # baseline file. The layering supplies those at compile time; here the
        # question is only whether the grammar reads the expression.
        for by_status in mapper.values():
            for expression in by_status.values():
                for name in mappers.identifiers(expression):
                    subparams.setdefault(name, 1.0)

        for new_name, by_status in mapper.items():
            for status, expression in by_status.items():
                result = mappers.evaluate(expression, subparams)
                assert np.isfinite(result), (path, new_name, status, expression)
                seen += 1
                distinct.add(str(expression))

    # The tree held 2,184 mapper expressions in September 2026. The bound
    # guards against a glob that finds too little, not a count to maintain.
    assert seen > 2000, f"only walked {seen} mappers"
    assert len(distinct) > 100
