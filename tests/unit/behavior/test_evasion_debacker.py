"""The noncompliance module: no reform, no response; and one worked example.

The fixtures are written out by hand, so what should happen is visible in the
test itself. Dollar comparisons are to a cent.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.evasion import debacker
from tests.tolerances import CENT

INCOME_COLUMNS = (
    "sole_prop1",
    "sole_prop2",
    "farm1",
    "farm2",
    "part_active",
    "part_se1",
    "part_se2",
    "part_passive",
    "scorp_active",
    "scorp_passive",
    "rent",
)


def household_fixture() -> dict[str, jnp.ndarray]:
    """Four households.

    Record 0 is a sole proprietor with a farm on the side and some rent.
    Record 1 is a working partner in a partnership and a passive S corporation
    shareholder. Record 2 has proprietorship income on one earner and a larger
    loss on the other, so the household total is negative. Record 3 has no
    business income at all.
    """
    columns = {
        "sole_prop1": [1000.0, 0.0, 1000.0, 0.0],
        "sole_prop2": [500.0, 0.0, -1200.0, 0.0],
        "farm1": [200.0, 0.0, 0.0, 0.0],
        "farm2": [0.0, 0.0, 0.0, 0.0],
        "part_active": [0.0, 4000.0, 0.0, 0.0],
        "part_se1": [0.0, 3000.0, 0.0, 0.0],
        "part_se2": [0.0, 1000.0, 0.0, 0.0],
        "part_passive": [0.0, 250.0, 0.0, 0.0],
        "scorp_active": [0.0, 0.0, 0.0, 0.0],
        "scorp_passive": [0.0, 800.0, 0.0, 0.0],
        "rent": [900.0, -300.0, 0.0, 0.0],
    }
    return {name: jnp.asarray(values) for name, values in columns.items()}


def rates(sole_prop1: float, part_active: float, rent: float) -> dict[str, jnp.ndarray]:
    """The three marginal rates the module prices off, under one law.

    The names are the ones the pricing pass writes and the module reads.
    """
    return {
        "mtr_sole_prop1": jnp.full(4, sole_prop1),
        "mtr_part_active": jnp.full(4, part_active),
        "mtr_rent": jnp.full(4, rent),
    }


class TestNoReformMeansNoResponse:
    def test_identical_rates_leave_every_column_alone(self):
        records = household_fixture()
        prices = rates(0.32, 0.35, 0.28)

        changed = debacker.evasion(records, prices, prices, debacker.DEFAULTS, {})

        for name in INCOME_COLUMNS:
            assert np.array_equal(np.asarray(changed[name]), np.asarray(records[name]))

    def test_identical_rates_leave_the_concealment_factors_at_one(self):
        records = household_fixture()
        prices = rates(0.32, 0.35, 0.28)

        changed = debacker.evasion(records, prices, prices, debacker.DEFAULTS, {})

        for name in bs.EVASION_FACTOR_COLS:
            assert np.array_equal(np.asarray(changed[name]), np.ones(4))


class TestTheWorkedExample:
    """A rate rise from 30 to 40 percent on every group.

    The factor for Schedule C and farm income is
    1 + 0.046 x (0.60/0.70 - 1) = 0.9934285714..., so $1,000 of reported sole
    proprietorship income becomes $993.43. The same arithmetic with the
    pass-through elasticity of 0.052 gives 0.9925714286, and with the rent
    elasticity of 0.040 gives 0.9942857143.
    """

    def setup_method(self):
        self.records = household_fixture()
        self.changed = debacker.evasion(
            self.records, rates(0.40, 0.40, 0.40), rates(0.30, 0.30, 0.30), debacker.DEFAULTS, {}
        )

    def test_sole_proprietorship_income_falls_to_the_hand_computed_dollar(self):
        assert abs(float(self.changed["sole_prop1"][0]) - 993.43) < CENT
        assert abs(float(self.changed["sole_prop2"][0]) - 496.71) < CENT
        assert abs(float(self.changed["farm1"][0]) - 198.69) < CENT

    def test_partnership_income_and_its_self_employment_shares_move_together(self):
        assert abs(float(self.changed["part_active"][1]) - 3970.29) < CENT
        assert abs(float(self.changed["part_se1"][1]) - 2977.71) < CENT
        assert abs(float(self.changed["part_se2"][1]) - 992.57) < CENT
        assert abs(float(self.changed["part_passive"][1]) - 248.14) < CENT
        assert abs(float(self.changed["scorp_passive"][1]) - 794.06) < CENT

    def test_rent_uses_its_own_elasticity(self):
        assert abs(float(self.changed["rent"][0]) - 894.86) < CENT

    def test_losses_and_negative_household_totals_are_left_alone(self):
        # Record 1's rent is a loss, and record 2's proprietorship income is
        # negative once the two earners are added up. Neither responds.
        assert float(self.changed["rent"][1]) == -300.0
        assert float(self.changed["sole_prop1"][2]) == 1000.0
        assert float(self.changed["sole_prop2"][2]) == -1200.0

    def test_the_factors_are_written_for_every_household(self):
        for name in bs.EVASION_FACTOR_COLS:
            assert np.asarray(self.changed[name]).shape == (4,)
        assert abs(float(self.changed["evasion_g_schc"][3]) - 0.99342857) < 1e-6


class TestTheGuards:
    def test_a_baseline_rate_of_one_gives_no_response(self):
        records = household_fixture()
        changed = debacker.evasion(
            records, rates(0.40, 0.40, 0.40), rates(1.0, 1.0, 1.0), debacker.DEFAULTS, {}
        )
        for name in bs.EVASION_FACTOR_COLS:
            assert np.array_equal(np.asarray(changed[name]), np.ones(4))
        assert float(changed["sole_prop1"][0]) == 1000.0

    def test_the_clamp_holds_the_factor_between_zero_and_two(self):
        records = household_fixture()
        # A large top-end multiplier and a big rate cut would otherwise send
        # reported income far above double.
        changed = debacker.evasion(
            records,
            rates(0.01, 0.01, 0.01),
            rates(0.95, 0.95, 0.95),
            {**debacker.DEFAULTS, "top_end_multiplier": 50.0},
            {},
        )
        for name in bs.EVASION_FACTOR_COLS:
            factors = np.asarray(changed[name])
            assert np.all(factors >= 0.0)
            assert np.all(factors <= 2.0)
        assert float(changed["sole_prop1"][0]) == 2000.0
