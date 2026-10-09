"""Estate-tax avoidance: the 0.16 reporting response and what it stacks on."""

import numpy as np

from taxmodel.behavior.estate.avoidance import (
    CONCEALED_FRACTION_FLOOR,
    concealed_fraction_parts,
    estate_avoidance,
)
from taxmodel.behavior.estate.avoidance import (
    DEFAULTS as ESTATE_DEFAULTS,
)
from taxmodel.behavior.wealth.avoidance import DEFAULTS as WEALTH_DEFAULTS
from taxmodel.behavior.wealth.avoidance import wealth_avoidance
from tests.tolerances import CENT
from tests.unit.behavior.avoidance_fixtures import flat_prices, households, only

# A $2,000,000 estate, half marketable and half closely held, so a share of
# the estate translates into dollars without any weighting.
ESTATE = 2_000_000.0
HALF_AND_HALF = dict(
    count=1,
    **{"value.equities": 1_000_000.0, "value.pass_throughs": 1_000_000.0},
)


def test_unchanged_estate_law_and_neutral_upstream_factors_conceal_nothing():
    # Both laws charge the same 40% marginal estate rate, the wealth module
    # concealed nothing, and the evasion module hid nothing.
    records = households(**HALF_AND_HALF)
    prices = flat_prices("estate_tax", 0.40, 1)

    changed = estate_avoidance(records, prices, prices, ESTATE_DEFAULTS)

    np.testing.assert_array_equal(changed["estate_concealed_frac"], np.zeros(1))


def test_the_own_rate_response_matches_the_worked_example():
    # Kopczuk and Slemrod's elasticity applied to a top rate going from 40% to
    # 55%: the share of the estate still reported is
    #     (0.45 / 0.60) ^ 0.16 = 0.955014140702901
    # so 4.4986% of the estate stops being reported -- $89,971.72 of a
    # $2,000,000 estate. Nothing is carried in from the other two modules.
    records = households(**HALF_AND_HALF)
    reform = flat_prices("estate_tax", 0.55, 1)
    baseline = flat_prices("estate_tax", 0.40, 1)

    parts = concealed_fraction_parts(records, reform, baseline, ESTATE_DEFAULTS)

    np.testing.assert_allclose(
        only(parts["still_reported_after_own_rate"]), 0.9550141407029011
    )
    np.testing.assert_allclose(only(parts["carried_in"]), 0.0)
    np.testing.assert_allclose(
        only(parts["concealed_fraction"]) * ESTATE, 89_971.71859, atol=CENT
    )


def test_the_three_contributions_stack_on_what_is_still_reported():
    # The wealth module concealed 10% of this household's marketable wealth
    # and 20% of its closely held wealth. The evasion module left it reporting
    # 90% of its partnership income, which is all the closely held income it
    # has, so a tenth of the closely held wealth still visible after
    # concealment drops out of the estate as well:
    #     closely held out of sight = 0.20 + 0.80 * 0.10 = 0.28
    #     carried in = (0.10 * 1,000,000 + 0.28 * 1,000,000) / 2,000,000 = 0.19
    # and the estate's own response applies to the 81% still reported:
    #     concealed = 1 - 0.81 * 0.955014140702901 = 0.2264385460
    # which is $452,877.09 of the $2,000,000 estate.
    records = households(
        **HALF_AND_HALF,
        wealth_c_pub=0.10, wealth_c_priv=0.20, part_active=300_000.0, evasion_g_pt=0.90,
    )
    reform = flat_prices("estate_tax", 0.55, 1)
    baseline = flat_prices("estate_tax", 0.40, 1)

    parts = concealed_fraction_parts(records, reform, baseline, ESTATE_DEFAULTS)

    np.testing.assert_allclose(only(parts["evaded_income_share"]), 0.10)
    np.testing.assert_allclose(only(parts["closely_held_out_of_sight_share"]), 0.28)
    np.testing.assert_allclose(only(parts["carried_in"]), 0.19)
    np.testing.assert_allclose(
        only(parts["concealed_fraction"]) * ESTATE, 452_877.09206, atol=CENT
    )


def test_wealth_concealment_reaches_the_estate_through_the_two_shares():
    # The two modules run in the chain's order: wealth avoidance writes the
    # two concealed shares, the runner puts them on the frame, estate
    # avoidance reads them. Estate law itself does not change
    # here, so everything in the answer came from the wealth tax.
    records = households(**HALF_AND_HALF)
    wealth_reform = flat_prices("wealth_tax", 0.03, 1)
    wealth_baseline = flat_prices("wealth_tax", 0.0, 1)
    estate_prices = flat_prices("estate_tax", 0.40, 1)

    after_wealth = dict(records)
    after_wealth.update(wealth_avoidance(records, wealth_reform, wealth_baseline, WEALTH_DEFAULTS))
    parts = concealed_fraction_parts(after_wealth, estate_prices, estate_prices, ESTATE_DEFAULTS)

    concealed_marketable = only(after_wealth["wealth_c_pub"])
    concealed_closely_held = only(after_wealth["wealth_c_priv"])
    assert concealed_marketable > 0.0 and concealed_closely_held > 0.0
    np.testing.assert_allclose(only(parts["still_reported_after_own_rate"]), 1.0)
    np.testing.assert_allclose(
        only(parts["concealed_fraction"]) * ESTATE,
        (concealed_marketable + concealed_closely_held) * 1_000_000.0,
        atol=CENT,
    )


def test_a_rate_cut_brings_hidden_estate_back_into_view():
    # Cutting the top rate from 40% to 20% makes the concealed fraction
    # negative: estate that was not being reported reappears.
    records = households(**HALF_AND_HALF)
    reform = flat_prices("estate_tax", 0.20, 1)
    baseline = flat_prices("estate_tax", 0.40, 1)

    concealed = only(
        estate_avoidance(records, reform, baseline, ESTATE_DEFAULTS)["estate_concealed_frac"]
    )

    assert concealed < 0.0
    assert concealed >= CONCEALED_FRACTION_FLOOR


def test_a_household_with_no_balance_sheet_conceals_nothing():
    records = households(1)
    reform = flat_prices("estate_tax", 0.55, 1)
    baseline = flat_prices("estate_tax", 0.40, 1)

    parts = concealed_fraction_parts(records, reform, baseline, ESTATE_DEFAULTS)

    # No estate means no rate on it either, so the own-rate response is one
    # and the concealed fraction is zero rather than undefined.
    np.testing.assert_array_equal(np.asarray(parts["carried_in"]), np.zeros(1))
    np.testing.assert_array_equal(np.asarray(parts["concealed_fraction"]), np.zeros(1))
