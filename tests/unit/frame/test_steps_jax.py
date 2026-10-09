"""The JAX annual frame-preparation steps."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.frame import steps_jax

from .conftest import make_frame


def _arrays(frame):
    """Convert the numeric columns of a small test frame to JAX arrays."""
    return {
        name: jnp.asarray(column.to_numpy())
        for name, column in frame.items()
        if column.dtype.kind in "biufc"
    }


class TestHeadOfHouseholdAndLawJoin:

    def test_the_recode_and_gather_are_traced_together(self):
        frame = {
            "filing_status": jnp.asarray([4, 2, 1]),
            "weight": jnp.ones(3),
        }
        law = {"filing.repeal_hoh": jnp.asarray(1), "limit": jnp.arange(4.0)}
        valid = jnp.asarray([True, True, False])

        @jax.jit
        def run(fields, repeal, annual_law, mask):
            recoded = steps_jax.recode_head_of_household(fields, repeal, mask)
            return steps_jax.join_law(recoded, annual_law)

        got = run(frame, jnp.asarray(1), law, valid)
        np.testing.assert_array_equal(got["filing_status"], [1, 2, 1])
        np.testing.assert_array_equal(got["limit"], [0.0, 1.0, 0.0])
        assert int(got["filing.repeal_hoh"]) == 1

    def test_invalid_slots_keep_single_status(self):
        frame = {"filing_status": jnp.asarray([4, 1])}
        got = steps_jax.recode_head_of_household(
            frame, jnp.asarray(1), jnp.asarray([True, False])
        )
        np.testing.assert_array_equal(got["filing_status"], [1, 1])


class TestSaltWorkaround:

    def test_invalid_slots_do_not_elect(self):
        frame = make_frame(
            [
                {
                    "id": 1,
                    "part_active": 100.0,
                    "salt_inc_sales": 1_000.0,
                    "sstb_part": 0.0,
                    "item.salt_workaround_allowed_non_sstb": 1.0,
                },
                {
                    "id": 2,
                    "part_active": 100.0,
                    "salt_inc_sales": 1_000.0,
                    "sstb_part": 0.0,
                    "item.salt_workaround_allowed_non_sstb": 1.0,
                },
            ]
        )
        got = steps_jax.apply_salt_workaround(
            _arrays(frame), jnp.asarray([True, False])
        )
        assert float(got["salt_workaround_part"][0]) == 1_000.0
        assert float(got["salt_workaround_part"][1]) == 0.0
        assert float(got["salt_inc_sales"][1]) == 0.0


class TestSocialSecurity:

    def test_invalid_slots_have_no_benefit(self):
        frame = make_frame(
            [
                {"id": 1, "age1": 64, "gross_ss": 10_000.0},
                {"id": 2, "age1": 64, "gross_ss": 10_000.0},
            ]
        )
        got = steps_jax.zero_social_security_in_padded_rows(
            _arrays(frame), jnp.asarray([True, False])
        )
        np.testing.assert_array_equal(got["gross_ss"], [10_000.0, 0.0])


class TestCapitalGainsPriceRatio:

    def test_invalid_slots_receive_a_ratio_of_one(self):
        frame = make_frame(
            [
                {"id": 1, "kg_lt": 1_000.0, "kg_lt_years_held": 2.0},
                {"id": 2, "kg_lt": 1_000.0, "kg_lt_years_held": 2.0},
            ]
        )
        got = steps_jax.calc_kg_cpi_ratio(
            _arrays(frame),
            2026,
            jnp.asarray([2024, 2025, 2026]),
            jnp.asarray([1.0, 1.1, 1.21]),
            jnp.asarray([True, False]),
        )
        np.testing.assert_allclose(got["kg_lt_cpi_ratio"], [1.21, 1.0])

    def test_a_gain_with_no_holding_period_cannot_look_valid(self):
        frame = make_frame([{"id": 1, "kg_lt": 1_000.0, "kg_lt_years_held": np.nan}])
        got = jax.jit(steps_jax.calc_kg_cpi_ratio)(
            _arrays(frame),
            jnp.asarray(2026),
            jnp.asarray([2025, 2026]),
            jnp.asarray([1.0, 1.1]),
            jnp.asarray([True]),
        )
        assert np.isnan(np.asarray(got["kg_lt_cpi_ratio"])[0])


class TestEstateMortality:

    def test_grouping_is_independent_of_record_order(self):
        frame = make_frame(
            [
                {
                    "id": 3,
                    "value.equities": 10_000_001.0,
                    "q_death1": 0.5,
                    "weight": 1_000.0,
                },
                {
                    "id": 1,
                    "value.equities": 10_000_000.0,
                    "q_death1": 0.5,
                    "weight": 1_000.0,
                },
                {
                    "id": 2,
                    "value.equities": 10_000_000.0,
                    "q_death1": 0.5,
                    "weight": 1_000.0,
                },
            ]
        )
        got = steps_jax.calc_estate_mortality(
            _arrays(frame), 300.0, 5_000_000.0, jnp.ones(3, dtype=bool)
        )
        np.testing.assert_allclose(got["estate_m"], [0.5, 0.15, 0.15])

    def test_invalid_slots_do_not_join_a_cluster(self):
        frame = make_frame(
            [
                {
                    "id": record_id,
                    "value.equities": 10_000_000.0,
                    "q_death1": 0.5,
                    "weight": 1_000.0,
                }
                for record_id in (1, 2, 3)
            ]
        )
        got = jax.jit(steps_jax.calc_estate_mortality)(
            _arrays(frame),
            jnp.asarray(300.0),
            jnp.asarray(5_000_000.0),
            jnp.asarray([True, True, False]),
        )
        np.testing.assert_allclose(got["estate_m"], [0.15, 0.15, 0.0])

    def test_gross_estates_are_rounded_half_to_even_before_clustering(self):
        # Rounded half to even, 9,999,999.5 and 10,000,000.5 both become
        # 10,000,000 and form one cluster, whose 1,000 expected deaths are
        # scaled to the cap of 300. 10,000,001.5 rounds to 10,000,002 and
        # stays on its own, unscaled.
        frame = make_frame(
            [
                {
                    "id": record_id,
                    "value.equities": gross,
                    "q_death1": 0.5,
                    "weight": 1_000.0,
                }
                for record_id, gross in (
                    (1, 9_999_999.5),
                    (2, 10_000_000.5),
                    (3, 10_000_001.5),
                )
            ]
        )
        got = steps_jax.calc_estate_mortality(
            _arrays(frame), 300.0, 5_000_000.0, jnp.ones(3, dtype=bool)
        )
        np.testing.assert_allclose(got["estate_m"], [0.15, 0.15, 0.5])


class TestNetWorth:

    def test_invalid_slots_receive_zero_net_worth(self):
        frame = make_frame(
            [
                {"id": 1, "value.equities": 500_000.0},
                {"id": 2, "value.equities": 500_000.0},
            ]
        )
        got = steps_jax.add_net_worth(_arrays(frame), jnp.asarray([True, False]))
        np.testing.assert_array_equal(got["net_worth"], [500_000.0, 0.0])
