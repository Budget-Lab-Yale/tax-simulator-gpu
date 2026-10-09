"""Entity shifting: no price change means no response, and the corporate side adds up."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from taxmodel.behavior.entity_shifting.pearce_prisinzano import (
    CORPORATE_TAX_CHANGE_COLUMN,
    DEFAULTS,
    PASS_THROUGH_NEGATIVE_COLUMNS,
    PASS_THROUGH_POSITIVE_COLUMNS,
    shift_entity_form,
)
from tests.tolerances import CENT


@pytest.fixture
def households():
    """Two households with business income, one of them with none."""

    zeros = jnp.zeros(2)
    return {
        "kg_lt": jnp.asarray([100_000.0, 0.0]),
        "part_active": jnp.asarray([400_000.0, 0.0]),
        "part_passive": jnp.asarray([100_000.0, 0.0]),
        "part_active_loss": zeros,
        "part_passive_loss": zeros,
        "part_179": zeros,
        "scorp_active": jnp.asarray([300_000.0, 0.0]),
        "scorp_passive": zeros,
        "scorp_active_loss": zeros,
        "scorp_passive_loss": zeros,
        "scorp_179": zeros,
        "sole_prop1": jnp.asarray([150_000.0, 0.0]),
        "sole_prop2": jnp.asarray([50_000.0, 0.0]),
        "part_se1": jnp.asarray([300_000.0, 0.0]),
        "part_se2": jnp.asarray([100_000.0, 0.0]),
        "equivalent_tax_rate_baseline": jnp.asarray([0.15, 0.15]),
        "equivalent_tax_rate_scenario": jnp.asarray([0.15, 0.15]),
    }


def prices(gains, pass_through):
    return {
        "mtr_kg_lt": jnp.asarray(gains, dtype=jnp.float64),
        "mtr_part_active": jnp.asarray(pass_through, dtype=jnp.float64),
    }


def test_no_price_change_means_no_response(households):
    """A reform that moves none of a module's prices moves nothing."""

    same = prices([0.238, 0.0], [0.37, 0.0])
    params = {**DEFAULTS, "corp.rate": 0.21, "corp.rate_baseline": 0.21}
    changed = shift_entity_form(households, same, same, params)

    assert changed[CORPORATE_TAX_CHANGE_COLUMN] == pytest.approx(jnp.zeros(2), abs=CENT)
    assert changed["part_active"] == pytest.approx(households["part_active"], abs=CENT)
    assert changed["kg_lt"] == pytest.approx(households["kg_lt"], abs=CENT)
    assert changed["part_se1"] == pytest.approx(households["part_se1"], abs=CENT)


def test_a_lower_corporate_rate_moves_income_out_of_pass_through_form(households):
    """Cutting the corporate rate makes corporate form cheaper, so income leaves.

    The household's whole pass-through basis is $1,000,000: $400,000 active and
    $100,000 passive partnership income, $300,000 active S corporation income
    and $200,000 of sole proprietorship income across the two earners.  A
    corporate rate cut from 35 to 21 percent, with the shareholder's rates held
    where they are, makes the corporate price fall, the differential move
    against pass-through, and a negative amount shift — income moving into
    corporate form.
    """

    same = prices([0.238, 0.0], [0.37, 0.0])
    params = {
        **DEFAULTS,
        "corp.rate": 0.21,
        "corp.rate_baseline": 0.35,
    }
    changed = shift_entity_form(households, same, same, params)

    shareholder_price = 0.45 * 0.238 + 0.55 * 0.15
    change_in_differential = (0.21 - 0.35) * (1.0 - shareholder_price)
    share_shifted = (0.3788 / 0.6) * change_in_differential
    amount_shifted = 1_000_000.0 * share_shifted

    assert share_shifted < 0
    # Every leg shrinks by the same percentage.
    assert changed["part_active"][0] == pytest.approx(
        400_000.0 * (1.0 + share_shifted), abs=CENT
    )
    assert changed["scorp_active"][0] == pytest.approx(
        300_000.0 * (1.0 + share_shifted), abs=CENT
    )
    # Corporate revenue rises by the new corporate income times the new rate.
    assert changed[CORPORATE_TAX_CHANGE_COLUMN][0] == pytest.approx(
        -amount_shifted * 0.21, abs=CENT
    )
    assert changed[CORPORATE_TAX_CHANGE_COLUMN][0] > 0


def test_retained_earnings_are_priced_off_the_equivalent_tax_rate(households):
    """The retained share is priced off the gains model's equivalent tax rate.

    Nothing else moves here. The corporate rate is 21 percent under both laws
    and both marginal rates are unchanged, so the only thing the reform does is
    raise the tax on a dollar of corporate earnings held inside the company
    from 15 to 30 percent. That makes corporate form more expensive and pushes
    income back into pass-through form.
    """

    reform = dict(households)
    reform["equivalent_tax_rate_scenario"] = jnp.asarray([0.30, 0.30])

    same = prices([0.238, 0.0], [0.37, 0.0])
    params = {**DEFAULTS, "corp.rate": 0.21, "corp.rate_baseline": 0.21}
    changed = shift_entity_form(reform, same, same, params)

    # The corporate price rises by the after-corporate-tax part of the increase
    # in the shareholder's price, and 55 percent of corporate earnings are the
    # retained share that this rate applies to.
    change_in_corporate_price = (1.0 - 0.21) * 0.55 * (0.30 - 0.15)
    share_shifted = (0.3788 / 0.6) * change_in_corporate_price
    amount_shifted = 1_000_000.0 * share_shifted

    assert share_shifted > 0
    assert changed["part_active"][0] == pytest.approx(
        400_000.0 * (1.0 + share_shifted), abs=CENT
    )
    # Income leaving corporate form takes corporate revenue down with it.
    assert changed[CORPORATE_TAX_CHANGE_COLUMN][0] == pytest.approx(
        -amount_shifted * 0.21, abs=CENT
    )
    assert changed[CORPORATE_TAX_CHANGE_COLUMN][0] < 0


def change_in_pass_through_income(before, after):
    """The change in net pass-through income, losses counted negatively."""

    change = 0.0
    for name in PASS_THROUGH_POSITIVE_COLUMNS:
        change = change + jnp.nan_to_num(after[name]) - jnp.nan_to_num(before[name])
    for name in PASS_THROUGH_NEGATIVE_COLUMNS:
        change = change - jnp.nan_to_num(after[name]) + jnp.nan_to_num(before[name])
    return change


def test_changes_across_pass_through_columns_add_up_to_the_amount_shifted():
    """The shifted amount is spread across every column, losses included.

    The first household's income is mostly S corporation and sole
    proprietorship income, with a small active partnership leg and losses on
    several legs. Income leaves pass-through form, and no column changes sign.
    The second household's losses exceed its income, so it shifts nothing.
    """

    zeros = jnp.zeros(2)
    households = {
        "kg_lt": jnp.asarray([50_000.0, 0.0]),
        "part_active": jnp.asarray([10_000.0, 5_000.0]),
        "part_passive": jnp.asarray([20_000.0, 0.0]),
        "part_active_loss": jnp.asarray([4_000.0, 0.0]),
        "part_passive_loss": zeros,
        "part_179": jnp.asarray([1_000.0, 0.0]),
        "scorp_active": jnp.asarray([500_000.0, 0.0]),
        "scorp_passive": jnp.asarray([60_000.0, 0.0]),
        "scorp_active_loss": jnp.asarray([30_000.0, 50_000.0]),
        "scorp_passive_loss": jnp.asarray([5_000.0, 0.0]),
        "scorp_179": jnp.asarray([2_000.0, 0.0]),
        "sole_prop1": jnp.asarray([200_000.0, 0.0]),
        "sole_prop2": jnp.asarray([40_000.0, 0.0]),
        "part_se1": jnp.asarray([8_000.0, 5_000.0]),
        "part_se2": zeros,
        "equivalent_tax_rate_baseline": jnp.asarray([0.15, 0.15]),
        "equivalent_tax_rate_scenario": jnp.asarray([0.15, 0.15]),
    }
    same = prices([0.238, 0.238], [0.37, 0.37])
    params = {**DEFAULTS, "corp.rate": 0.21, "corp.rate_baseline": 0.35}
    changed = shift_entity_form(households, same, same, params)

    net_income = 10_000 + 20_000 - 4_000 - 1_000 + 500_000 + 60_000
    net_income += -30_000 - 5_000 - 2_000 + 200_000 + 40_000
    shareholder_price = 0.45 * 0.238 + 0.55 * 0.15
    share_shifted = (0.3788 / 0.6) * (0.21 - 0.35) * (1.0 - shareholder_price)
    amount_shifted = net_income * share_shifted

    change = change_in_pass_through_income(households, changed)
    assert float(change[0]) == pytest.approx(amount_shifted, abs=CENT)
    assert float(change[1]) == pytest.approx(0.0, abs=CENT)
    assert changed[CORPORATE_TAX_CHANGE_COLUMN][0] == pytest.approx(
        -amount_shifted * 0.21, abs=CENT
    )
    for name in (*PASS_THROUGH_POSITIVE_COLUMNS, *PASS_THROUGH_NEGATIVE_COLUMNS):
        assert float(changed[name][0]) >= 0.0, f"{name} changed sign"
    # The partnership self-employment earnings move by the same percentage.
    assert changed["part_se1"][0] == pytest.approx(
        8_000.0 * (1.0 + share_shifted), abs=CENT
    )
    assert changed["part_se1"][1] == pytest.approx(5_000.0, abs=CENT)


def test_a_blank_column_stays_blank_and_the_others_carry_the_shift(households):
    """A blank pass-through column is read as zero and goes back out blank.

    The first household has a blank active partnership entry but real S
    corporation income. Its other columns are scaled, and their changes still
    add up to the amount shifted, so no shifted income is lost.
    """

    blank = dict(households)
    blank["part_active"] = jnp.asarray([jnp.nan, jnp.nan])

    same = prices([0.238, 0.0], [0.37, 0.0])
    params = {
        **DEFAULTS,
        "corp.rate": 0.21,
        "corp.rate_baseline": 0.35,
    }
    changed = shift_entity_form(blank, same, same, params)

    # The first household's pass-through income is $600,000 once the blank
    # reads as zero: $100,000 passive partnership, $300,000 active S
    # corporation, $200,000 sole proprietorship.
    shareholder_price = 0.45 * 0.238 + 0.55 * 0.15
    change_in_differential = (0.21 - 0.35) * (1.0 - shareholder_price)
    share_shifted = (0.3788 / 0.6) * change_in_differential
    amount_shifted = 600_000.0 * share_shifted

    assert bool(jnp.all(jnp.isnan(changed["part_active"])))
    change = change_in_pass_through_income(blank, changed)
    assert float(change[0]) == pytest.approx(amount_shifted, abs=CENT)
    assert changed[CORPORATE_TAX_CHANGE_COLUMN][0] == pytest.approx(
        -amount_shifted * 0.21, abs=CENT
    )
