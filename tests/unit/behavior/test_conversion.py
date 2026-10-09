"""The in-year conversion module: it clamps, it moves salary, it never re-prices."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from taxmodel.behavior.conversion.sigma import (
    CLAMPED_SHARE_COLUMN,
    DEFAULTS,
    MANIFEST,
    apply_conversions,
)
from tests.tolerances import CENT


@pytest.fixture
def households():
    """Three households with salary and business income, in round numbers."""

    return {
        "wages_base1": jnp.asarray([500_000.0, 200_000.0, 0.0]),
        "wages_base2": jnp.asarray([100_000.0, 0.0, 0.0]),
        "tips1": jnp.asarray([10_000.0, 0.0, 0.0]),
        "ot1": jnp.asarray([5_000.0, 0.0, 0.0]),
        "part_active": jnp.asarray([400_000.0, 0.0, 100_000.0]),
        "part_se1": jnp.asarray([300_000.0, 0.0, 0.0]),
        "part_se2": jnp.asarray([100_000.0, 0.0, 0.0]),
        "scorp_active": jnp.asarray([0.0, 150_000.0, 0.0]),
        "sole_prop1": jnp.asarray([0.0, 0.0, 60_000.0]),
        "sole_prop2": jnp.asarray([0.0, 0.0, 40_000.0]),
    }


def conversions(rows):
    return jnp.asarray(rows, dtype=jnp.float64)


def test_base_wages_fall_by_the_conversion_and_tips_and_overtime_do_not(households):
    """What is converted is salary, so only base pay moves."""

    asked = conversions(
        [
            [50_000.0, 20_000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    changed = apply_conversions(households, {}, {}, {"conversions": asked})

    assert changed["wages_base1"][0] == pytest.approx(450_000.0, abs=CENT)
    assert changed["wages_base2"][0] == pytest.approx(80_000.0, abs=CENT)
    assert "tips1" not in changed
    assert "ot1" not in changed
    # Pay is conserved in the sense the module claims: what the households give
    # up in base wages is exactly what the pre-pass converted.
    given_up = (
        households["wages_base1"][0]
        - changed["wages_base1"][0]
        + households["wages_base2"][0]
        - changed["wages_base2"][0]
    )
    assert given_up == pytest.approx(70_000.0, abs=CENT)


def test_a_leg_cannot_give_up_more_than_it_holds(households):
    """The live frame's say: a household with no wages converts no wages."""

    asked = conversions(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [80_000.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    changed = apply_conversions(households, {}, {}, {"conversions": asked})

    assert changed["wages_base1"][2] == pytest.approx(0.0, abs=CENT)
    # The whole request was clamped away, and the module says so.
    assert changed[CLAMPED_SHARE_COLUMN][2] == pytest.approx(1.0)
    assert changed[CLAMPED_SHARE_COLUMN][0] == pytest.approx(0.0)


def test_the_earner_splits_move_with_the_leg_they_belong_to(households):
    """Partnership income comes down and its two earners' shares come with it."""

    asked = conversions(
        [
            [0.0, 0.0, 100_000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 50_000.0],
        ]
    )
    changed = apply_conversions(households, {}, {}, {"conversions": asked})

    # A quarter of $400,000 of active partnership income was converted, so the
    # two self-employment companions come down by a quarter each.
    assert changed["part_active"][0] == pytest.approx(300_000.0, abs=CENT)
    assert changed["part_se1"][0] == pytest.approx(225_000.0, abs=CENT)
    assert changed["part_se2"][0] == pytest.approx(75_000.0, abs=CENT)

    # Sole proprietorship is reported per earner, so its leg is the two added
    # up and each earner's share comes down proportionally: half of each.
    assert changed["sole_prop1"][2] == pytest.approx(30_000.0, abs=CENT)
    assert changed["sole_prop2"][2] == pytest.approx(20_000.0, abs=CENT)


def test_the_module_asks_the_pricing_pass_for_nothing(households):
    """The conversion was priced in the pre-pass; the rates here go unread.

    The declaration says so, and the arithmetic backs it up: handing the module
    two wildly different sets of prices changes nothing it returns.
    """

    assert MANIFEST["prices"] == ()
    assert MANIFEST["discrete_prices"] == ()

    asked = conversions(
        [
            [50_000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    quiet = apply_conversions(households, {}, {}, {"conversions": asked})
    loud = apply_conversions(
        households,
        {"mtr_kg_lt": jnp.asarray([0.9, 0.9, 0.9])},
        {"mtr_kg_lt": jnp.zeros(3)},
        {"conversions": asked},
    )
    for name, values in quiet.items():
        assert loud[name] == pytest.approx(values)


def test_no_conversions_means_no_change(households):
    """A scenario the pre-pass converted nothing for gets its frame back."""

    changed = apply_conversions(households, {}, {}, DEFAULTS)
    for name, values in changed.items():
        if name == CLAMPED_SHARE_COLUMN:
            assert values == pytest.approx(jnp.zeros(3))
        else:
            assert values == pytest.approx(households[name])
