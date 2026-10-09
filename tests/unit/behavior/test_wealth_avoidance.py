"""Wealth-tax avoidance: the response, the hidden ledger, and the derivative."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.kg.applier import CORPORATE_MARKDOWN_GAINS_COLUMN
from taxmodel.behavior.wealth.avoidance import (
    CLOSELY_HELD_ELASTICITY,
    CONCEALED_INCOME_GROUPS,
    DEFAULTS,
    PUBLIC_ELASTICITY,
    avoided_and_concealed_shares,
    hidden_ledger,
    wealth_avoidance,
)
from tests.tolerances import CENT
from tests.unit.behavior.avoidance_fixtures import flat_prices, households, only

# One household with a real balance sheet: $5,000,000 of marketable wealth,
# $3,000,000 of closely held wealth, and $900,000 of debt, throwing off
# interest, dividends, a realized gain, partnership income and rent.
RICH_HOUSEHOLD = dict(
    count=1,
    **{
        "value.equities": 3_000_000.0,
        "value.bonds": 1_500_000.0,
        "value.cash": 500_000.0,
        "value.pass_throughs": 2_000_000.0,
        "value.primary_home": 1_000_000.0,
        "value.primary_mortgage": 700_000.0,
        "value.credit_cards": 200_000.0,
        "txbl_int": 60_000.0,
        "div_ord": 25_000.0,
        "div_pref": 75_000.0,
        "kg_lt": 400_000.0,
        "part_active": 250_000.0,
        "part_se1": 150_000.0,
        "part_se2": 100_000.0,
        "sole_prop1": 90_000.0,
        "sole_prop2": 10_000.0,
        "rent": 40_000.0,
    },
)


def test_no_change_in_the_wealth_tax_rate_changes_nothing_at_all():
    # Both laws charge the same 3% rate, so there is nothing new to avoid.
    records = households(**RICH_HOUSEHOLD)
    prices = flat_prices("wealth_tax", 0.03, 1)

    changed = wealth_avoidance(records, prices, prices, DEFAULTS)

    assets = np.asarray(
        bs.total(records, bs.MARKETABLE_ASSET_COLS)
        + bs.total(records, bs.CLOSELY_HELD_ASSET_COLS)
    )
    debts = np.asarray(bs.total(records, bs.DEBT_COLS))
    np.testing.assert_array_equal(changed["net_worth"], assets - debts)
    np.testing.assert_array_equal(changed["wealth_c_pub"], np.zeros(1))
    np.testing.assert_array_equal(changed["wealth_c_priv"], np.zeros(1))
    for name in ("txbl_int", "kg_lt", "part_active", "sole_prop1", "rent"):
        np.testing.assert_array_equal(changed[name], records[name])


def test_reported_plus_hidden_equals_what_was_there_before():
    # A 2% wealth tax where baseline law has none, on the millionaire above.
    records = households(**RICH_HOUSEHOLD)
    reform = flat_prices("wealth_tax", 0.02, 1)
    baseline = flat_prices("wealth_tax", 0.0, 1)

    ledger = hidden_ledger(records, reform, baseline, DEFAULTS)

    # Marketable wealth: what is still reported, what was concealed, and what
    # was only assessed lower, add back up to what the household owns.
    np.testing.assert_allclose(
        ledger["reported_marketable_wealth"]
        + ledger["concealed_marketable_wealth"]
        + ledger["gamed_marketable_wealth"],
        ledger["marketable_wealth"],
        atol=CENT,
    )
    # Closely held wealth has the fourth piece: assets pulled out because the
    # income they produce was evaded.
    np.testing.assert_allclose(
        ledger["reported_closely_held_wealth"]
        + ledger["concealed_closely_held_wealth"]
        + ledger["gamed_closely_held_wealth"]
        + ledger["closely_held_wealth_pulled_out_by_evasion"],
        ledger["closely_held_wealth"],
        atol=CENT,
    )
    # And the reported base the wealth tax is charged on is exactly the two
    # reported classes less the debts, which nobody avoids.
    np.testing.assert_allclose(
        ledger["reported_net_worth"],
        ledger["reported_marketable_wealth"]
        + ledger["reported_closely_held_wealth"]
        - ledger["debts"],
        atol=CENT,
    )

    # Every income leg: what is still reported plus what was hidden with the
    # asset behind it equals the leg before the module ran.
    changed = wealth_avoidance(records, reform, baseline, DEFAULTS)
    for _gate, scaled_cols, _kind in CONCEALED_INCOME_GROUPS:
        for name in scaled_cols:
            np.testing.assert_allclose(
                changed[name] + ledger[f"hidden_{name}"],
                records[name],
                atol=CENT,
            )

    # The hidden income has to be consistent with the hidden assets: the same
    # concealed share applies to the income as to the wealth that produced it.
    shares = avoided_and_concealed_shares(records, reform, baseline, DEFAULTS)
    np.testing.assert_allclose(
        ledger["hidden_txbl_int"],
        records["txbl_int"] * shares["concealed_marketable"],
        atol=CENT,
    )
    np.testing.assert_allclose(
        ledger["hidden_rent"],
        records["rent"] * shares["concealed_closely_held"],
        atol=CENT,
    )


def test_a_household_reporting_a_loss_keeps_the_loss_intact():
    records = households(1, **{"value.equities": 1_000_000.0, "kg_lt": -50_000.0})
    reform = flat_prices("wealth_tax", 0.02, 1)
    baseline = flat_prices("wealth_tax", 0.0, 1)

    changed = wealth_avoidance(records, reform, baseline, DEFAULTS)

    np.testing.assert_array_equal(changed["kg_lt"], records["kg_lt"])


def test_closely_held_wealth_conceals_more_than_marketable_wealth():
    # The same 3% rate on both classes. Closely held wealth responds at the
    # larger elasticity, and even after only half of its response counts as
    # concealment, more of it goes out of sight.
    records = households(
        1, **{"value.equities": 1_000_000.0, "value.pass_throughs": 1_000_000.0}
    )
    reform = flat_prices("wealth_tax", 0.03, 1)
    baseline = flat_prices("wealth_tax", 0.0, 1)

    shares = avoided_and_concealed_shares(records, reform, baseline, DEFAULTS)

    assert only(shares["avoided_closely_held"]) > only(shares["avoided_marketable"])
    assert only(shares["concealed_closely_held"]) > only(shares["concealed_marketable"])
    # The default elasticities, spelled out: a 3% rate erodes about 19% of
    # reported marketable wealth and about 40% of reported closely held.
    np.testing.assert_allclose(
        only(shares["avoided_marketable"]), 1.0 - math.exp(0.03 * PUBLIC_ELASTICITY)
    )
    np.testing.assert_allclose(
        only(shares["avoided_closely_held"]),
        1.0 - math.exp(0.03 * CLOSELY_HELD_ELASTICITY),
    )

    # And a household holding only closely held wealth reports a smaller share
    # of it than one holding only marketable wealth, at the same rate.
    only_marketable = households(1, **{"value.equities": 2_000_000.0})
    only_closely_held = households(1, **{"value.pass_throughs": 2_000_000.0})
    marketable_reported = only(
        wealth_avoidance(only_marketable, reform, baseline, DEFAULTS)["net_worth"]
    )
    closely_held_reported = only(
        wealth_avoidance(only_closely_held, reform, baseline, DEFAULTS)["net_worth"]
    )
    assert closely_held_reported < marketable_reported


def test_evaded_income_pulls_the_matching_business_out_of_reported_wealth():
    # The evasion module left this household reporting 90% of its partnership
    # income, and that is its only closely held income, so a tenth of its
    # closely held wealth stops being reported on top of the price response.
    records = households(
        1,
        **{
            "value.pass_throughs": 1_000_000.0,
            "part_active": 200_000.0,
            "evasion_g_pt": 0.9,
        },
    )
    reform = flat_prices("wealth_tax", 0.01, 1)
    baseline = flat_prices("wealth_tax", 0.0, 1)

    ledger = hidden_ledger(records, reform, baseline, DEFAULTS)

    np.testing.assert_allclose(only(ledger["evaded_income_share"]), 0.1)
    expected_reported = (
        1_000_000.0 * math.exp(0.01 * CLOSELY_HELD_ELASTICITY) * (1.0 - 0.1)
    )
    np.testing.assert_allclose(
        only(ledger["reported_closely_held_wealth"]), expected_reported, atol=CENT
    )


def test_the_derivative_of_the_reported_base_is_the_response_at_the_priced_rate():
    # The price is a fact about the law, measured before any module ran, so a
    # dollar more of an asset changes the reported base by the share of that
    # class of wealth still being reported. Nothing re-prices.
    records = households(
        1,
        **{
            "value.equities": 4_000_000.0,
            "value.pass_throughs": 2_000_000.0,
            "value.primary_mortgage": 500_000.0,
            "part_active": 100_000.0,
            "evasion_g_pt": 0.8,
        },
    )
    reform = flat_prices("wealth_tax", 0.02, 1)
    baseline = flat_prices("wealth_tax", 0.0, 1)

    def reported_base(equities, pass_throughs, mortgage):
        moved = dict(records)
        moved["value.equities"] = equities
        moved["value.pass_throughs"] = pass_throughs
        moved["value.primary_mortgage"] = mortgage
        return jnp.sum(wealth_avoidance(moved, reform, baseline, DEFAULTS)["net_worth"])

    slopes = jax.grad(reported_base, argnums=(0, 1, 2))(
        jnp.asarray(records["value.equities"]),
        jnp.asarray(records["value.pass_throughs"]),
        jnp.asarray(records["value.primary_mortgage"]),
    )
    on_equities, on_pass_throughs, on_mortgage = (only(s) for s in slopes)

    assert all(math.isfinite(s) for s in (on_equities, on_pass_throughs, on_mortgage))
    np.testing.assert_allclose(on_equities, math.exp(0.02 * PUBLIC_ELASTICITY))
    # The closely held slope carries the evasion link as well: a fifth of this
    # household's partnership income is hidden, so a fifth of what is left
    # after the price response never reaches the reported base either.
    np.testing.assert_allclose(
        on_pass_throughs, math.exp(0.02 * CLOSELY_HELD_ELASTICITY) * (1.0 - 0.2)
    )
    # A dollar more of mortgage debt takes a dollar off the reported base:
    # debts are not avoided.
    np.testing.assert_allclose(on_mortgage, -1.0)


def test_the_markdown_gains_change_is_concealed_with_the_gains():
    """The markdown's share of realized gains hides at the rate the gains do.

    The first household reports a gain, so the reported change is scaled like
    its gain. The second reports a loss, so neither moves.
    """
    records = households(
        2,
        **{
            "value.equities": 1_000_000.0,
            "kg_lt": [400_000.0, -50_000.0],
            CORPORATE_MARKDOWN_GAINS_COLUMN: [-10_000.0, -2_000.0],
        },
    )
    reform = flat_prices("wealth_tax", 0.02, 2)
    baseline = flat_prices("wealth_tax", 0.0, 2)

    changed = wealth_avoidance(records, reform, baseline, DEFAULTS)

    reported_share = changed["kg_lt"][0] / records["kg_lt"][0]
    assert reported_share < 1.0
    np.testing.assert_allclose(
        changed[CORPORATE_MARKDOWN_GAINS_COLUMN],
        [-10_000.0 * reported_share, -2_000.0],
        atol=CENT,
    )
