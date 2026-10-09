"""The tips module: no reform, no response; pay conserved; who flips; the cap.

The fixtures are written out by hand. Dollar comparisons are to a cent.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from taxmodel.behavior.tips import fringe_low
from tests.tolerances import CENT

# The exemption takes effect in 2025 and the fixture year is 2028, so three
# years of 3 percent nominal wage growth have accumulated.
FIXTURE_YEAR = 2028
YEARS_SINCE_REFORM = 3
# The fixture years use the $25,000 statutory limit on the tips deduction.
FIXTURE_TIPS_DEDUCTION_LIMIT = 25_000.0


def household_fixture(draw1: list[float], draw2: list[float]) -> dict:
    """Five households in the year the fixture runs.

    Record 0 is a single tipped worker. Record 1 is a joint return with a
    tipped first earner and an untipped second earner. Record 2 is a single
    untipped wage earner. Record 3 is a joint return with two untipped
    earners. Record 4 is a single earner with no pay at all.
    """
    columns = {
        "year": [FIXTURE_YEAR] * 5,
        "below.tip_ded_limit": [FIXTURE_TIPS_DEDUCTION_LIMIT] * 5,
        "weight": [100.0, 100.0, 100.0, 100.0, 100.0],
        "filing_status": [1, 2, 1, 2, 1],
        "wages_base1": [20000.0, 30000.0, 40000.0, 50000.0, 0.0],
        "wages_base2": [0.0, 45000.0, 0.0, 60000.0, 0.0],
        "tips1": [5000.0, 6000.0, 0.0, 0.0, 0.0],
        "tips2": [0.0, 0.0, 0.0, 0.0, 0.0],
        "ot1": [1000.0, 0.0, 2000.0, 0.0, 0.0],
        "ot2": [0.0, 0.0, 0.0, 3000.0, 0.0],
        "r.behavior1": draw1,
        "r.behavior2": draw2,
    }
    records = {
        name: jnp.asarray(values, dtype=jnp.float64)
        for name, values in columns.items()
        if not name.startswith("r.")
    }
    draws = {
        "r.behavior1": jnp.asarray(draw1, dtype=jnp.float64),
        "r.behavior2": jnp.asarray(draw2, dtype=jnp.float64),
    }
    return records, draws


def rates(tips1: float, tips2: float) -> dict[str, jnp.ndarray]:
    """Each earner's marginal rate on a tipped dollar under one law.

    The names are the ones the pricing pass writes and the module reads.
    """
    return {
        "mtr_tips1": jnp.full(5, tips1, dtype=jnp.float64),
        "mtr_tips2": jnp.full(5, tips2, dtype=jnp.float64),
    }


def probability_of_becoming_tipped(records, reform, baseline, settings) -> float:
    """Recompute the extensive-margin probability the way the module states it.

    This repeats the arithmetic in plain NumPy so the test pins the formula,
    not the code that implements it.
    """
    weight = np.asarray(records["weight"])
    tips1 = np.asarray(records["tips1"])
    tips2 = np.asarray(records["tips2"])
    pay1 = np.asarray(records["wages_base1"]) + tips1 + np.asarray(records["ot1"])
    pay2 = np.asarray(records["wages_base2"]) + tips2 + np.asarray(records["ot2"])
    household_tips = tips1 + tips2
    household_pay = pay1 + pay2

    rate_change1 = np.asarray(reform["mtr_tips1"]) - np.asarray(baseline["mtr_tips1"])
    rate_change2 = np.asarray(reform["mtr_tips2"]) - np.asarray(baseline["mtr_tips2"])
    tips_weight = np.concatenate([weight * tips1, weight * tips2])
    average_rate_change = np.sum(
        np.concatenate([rate_change1, rate_change2]) * tips_weight
    ) / np.sum(tips_weight)

    new_tips_needed = (
        settings["e_extensive"] * average_rate_change * np.sum(weight * household_tips)
    )
    tipped = household_tips > 0
    share = np.sum(weight[tipped] * household_tips[tipped] / household_pay[tipped]) / (
        np.sum(weight[tipped])
    )
    untipped = (household_tips <= 0) & (household_pay > 0)
    untipped_pay = np.sum(weight[untipped] * household_pay[untipped])
    return float((new_tips_needed / untipped_pay) / share), float(share)


class TestNoReformMeansNoResponse:
    def test_identical_rates_move_no_pay_at_all(self):
        records, draws = household_fixture([0.1] * 5, [0.1] * 5)
        prices = rates(0.30, 0.30)

        changed = fringe_low.tips(records, prices, prices, fringe_low.DEFAULTS, draws)

        for name in ("tips1", "tips2", "wages_base1", "wages_base2"):
            assert np.array_equal(np.asarray(changed[name]), np.asarray(records[name]))
        assert np.array_equal(
            np.asarray(changed["tips_wage_floor_binding"]), np.zeros(5)
        )


class TestTheDeductionLimit:
    def test_zero_limit_blocks_an_ordinary_rate_reform_but_positive_limit_keeps_response(
        self,
    ):
        records, draws = household_fixture([0.0] * 5, [0.0] * 5)
        reform, baseline = rates(0.20, 0.20), rates(0.30, 0.30)

        # With no tips deduction, the marginal rate on tips moves with the
        # ordinary rate. That ordinary-rate change must not relabel pay.
        no_deduction_records = dict(records)
        no_deduction_records["below.tip_ded_limit"] = jnp.zeros(5)
        no_deduction_response = fringe_low.tips(
            no_deduction_records, reform, baseline, fringe_low.DEFAULTS, draws
        )
        for name in ("tips1", "tips2", "wages_base1", "wages_base2"):
            assert np.array_equal(
                np.asarray(no_deduction_response[name]),
                np.asarray(no_deduction_records[name]),
            )
        assert np.array_equal(
            np.asarray(no_deduction_response["tips_wage_floor_binding"]),
            np.zeros(5),
        )

        # With the positive limit in the fixture, the module responds:
        # tipped workers gain at the intensive margin
        # and the low-draw untipped worker becomes tipped.
        response = fringe_low.tips(records, reform, baseline, fringe_low.DEFAULTS, draws)
        capped_growth = 1.03**YEARS_SINCE_REFORM
        assert abs(float(response["tips1"][0]) - 5000.0 * capped_growth) < CENT
        probability, share = probability_of_becoming_tipped(
            records, reform, baseline, fringe_low.DEFAULTS
        )
        assert 0.0 < probability < 1.0
        pay1 = float(records["wages_base1"][2] + records["ot1"][2])
        assert abs(float(response["tips1"][2]) - pay1 * share) < CENT


class TestPayIsConserved:
    def test_every_earner_keeps_the_same_total_pay(self):
        # Draws of 0.0 put every eligible earner over the extensive margin, so
        # both margins are active. Base wages here are large enough that
        # nobody's base wages run out.
        records, draws = household_fixture([0.0] * 5, [0.0] * 5)
        changed = fringe_low.tips(
            records, rates(0.0, 0.0), rates(0.30, 0.30), fringe_low.DEFAULTS, draws
        )

        assert not np.any(np.asarray(changed["tips_wage_floor_binding"]))
        for earner in ("1", "2"):
            before = np.asarray(records[f"wages_base{earner}"]) + np.asarray(
                records[f"tips{earner}"]
            )
            after = np.asarray(changed[f"wages_base{earner}"]) + np.asarray(
                changed[f"tips{earner}"]
            )
            assert np.all(np.abs(after - before) < CENT)

    def test_tips_actually_rose(self):
        records, draws = household_fixture([0.0] * 5, [0.0] * 5)
        changed = fringe_low.tips(
            records, rates(0.0, 0.0), rates(0.30, 0.30), fringe_low.DEFAULTS, draws
        )
        assert float(changed["tips1"][0]) > float(records["tips1"][0])
        assert float(changed["tips1"][2]) > 0.0

    def test_base_wages_stop_at_zero_and_the_household_is_flagged(self):
        # A tipped worker with almost no base pay left to relabel: $50 of base
        # wages against $5,000 of tips growing 3 percent.
        columns, draws = household_fixture([0.9] * 5, [0.9] * 5)
        columns = dict(columns)
        columns["wages_base1"] = jnp.asarray(
            [50.0, 30000.0, 40000.0, 50000.0, 0.0], dtype=jnp.float64
        )
        changed = fringe_low.tips(
            columns, rates(0.0, 0.0), rates(0.30, 0.30), fringe_low.DEFAULTS, draws
        )
        assert float(changed["wages_base1"][0]) == 0.0
        assert float(changed["tips_wage_floor_binding"][0]) == 1.0
        # The tips gain is capped at the base wages available, so total pay
        # is conserved even where the floor binds.
        before = 50.0 + float(columns["tips1"][0])
        after = float(changed["wages_base1"][0]) + float(changed["tips1"][0])
        assert abs(after - before) < CENT


class TestTheExtensiveMargin:
    def test_exactly_the_earners_whose_draw_is_under_the_probability_flip(self):
        settings = dict(fringe_low.DEFAULTS)
        reform, baseline = rates(0.0, 0.0), rates(0.30, 0.30)
        # Draws chosen after seeing the probability below: two eligible
        # earners sit under it and two sit over it.
        probe_records, _ = household_fixture([0.5] * 5, [0.5] * 5)
        probability, share = probability_of_becoming_tipped(
            probe_records, reform, baseline, settings
        )
        assert 0.0 < probability < 1.0

        low, high = probability / 2.0, min(1.0, probability * 2.0)
        draw1 = [high, high, low, high, low]
        draw2 = [high, low, high, high, high]
        records, draws = household_fixture(draw1, draw2)
        changed = fringe_low.tips(records, reform, baseline, fringe_low.DEFAULTS, draws)

        pay1 = (
            np.asarray(records["wages_base1"])
            + np.asarray(records["tips1"])
            + np.asarray(records["ot1"])
        )
        pay2 = (
            np.asarray(records["wages_base2"])
            + np.asarray(records["tips2"])
            + np.asarray(records["ot2"])
        )
        # Record 2's first earner flips (untipped, has pay, low draw).
        assert abs(float(changed["tips1"][2]) - pay1[2] * share) < CENT
        # Record 1's second earner flips: untipped, on a joint return, low
        # draw.
        assert abs(float(changed["tips2"][1]) - pay2[1] * share) < CENT
        # Record 4's first earner has a low draw but no pay, so nothing
        # happens.
        assert float(changed["tips1"][4]) == 0.0
        # Record 3's earners both drew above the probability.
        assert float(changed["tips1"][3]) == 0.0
        assert float(changed["tips2"][3]) == 0.0

    def test_a_second_earner_on_a_single_return_never_flips(self):
        records, draws = household_fixture([0.0] * 5, [0.0] * 5)
        changed = fringe_low.tips(
            records, rates(0.0, 0.0), rates(0.30, 0.30), fringe_low.DEFAULTS, draws
        )
        for single_return in (0, 2, 4):
            assert float(changed["tips2"][single_return]) == 0.0


class TestTheWageGrowthCap:
    def test_a_large_rate_cut_is_capped_at_cumulative_wage_growth(self):
        # The uncapped intensive response to a 30-point cut would be
        # 1 + 1.05 x 0.30 = 1.315, well above 1.03^3 = 1.092727.
        records, draws = household_fixture([0.9] * 5, [0.9] * 5)
        changed = fringe_low.tips(
            records, rates(0.0, 0.0), rates(0.30, 0.30), fringe_low.DEFAULTS, draws
        )

        capped = 1.03**YEARS_SINCE_REFORM
        assert abs(float(changed["tips1"][0]) - 5000.0 * capped) < CENT

    def test_a_small_rate_cut_is_not_capped(self):
        # A 1-point cut gives 1 + 1.05 x 0.01 = 1.0105, under the cap.
        records, draws = household_fixture([0.9] * 5, [0.9] * 5)
        changed = fringe_low.tips(
            records, rates(0.29, 0.29), rates(0.30, 0.30), fringe_low.DEFAULTS, draws
        )
        assert abs(float(changed["tips1"][0]) - 5000.0 * 1.0105) < CENT
