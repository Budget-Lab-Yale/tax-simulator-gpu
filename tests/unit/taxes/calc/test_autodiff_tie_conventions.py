"""What JAX does with a derivative exactly at a corner, pinned as facts.

The model reads marginal tax rates off a gradient taken through the whole
calculator chain. Every threshold in the tax code -- a bracket edge, the start
or end of a phase-out, a cap, a floor -- is a corner: the slope just below it
differs from the slope just above it, and exactly on it there is no single
answer. Which of the two slopes autodiff reports at an exact tie is decided by
nothing but how the expression was written. The model's convention: at an exact threshold the household is treated as having crossed it
in the direction of more income. In code that means each corner is written so
the tie lands wherever one more dollar of income would put the household --
which is `>=` for an amount that grows with income and `>` for an amount, such
as a credit being phased out, that shrinks as income rises.

That convention rests on the facts measured below. They are true of JAX 0.7.0
and are not promised by any interface; a toolchain upgrade that changed one of
them would silently move marginal rates at every corner. This is the test that
fails first if that happens.

`floored_at` and `capped_at` in `taxmodel.taxes.calc._corners` are the two spellings
the model actually uses, so they are checked here alongside the raw ones.
"""

import jax
import jax.numpy as jnp
import pytest

from taxmodel.taxes.calc._corners import capped_at, floored_at

EDGE = 5.0


def slope_at(f, x):
    """The derivative JAX reports for `f` at `x`."""
    return float(jax.grad(f)(jnp.float64(x)))


class TestTheAveragingSpellings:
    """`maximum`, `minimum` and `clip` split the difference at a tie.

    Half of the slope below plus half of the slope above is a marginal rate no
    household ever faces, which is why these are not used on a path a gradient
    runs through.
    """

    def test_maximum_averages_the_two_slopes(self):
        assert slope_at(lambda x: jnp.maximum(x, EDGE), EDGE) == 0.5
        assert slope_at(lambda x: jnp.maximum(EDGE, x), EDGE) == 0.5

    def test_minimum_averages_the_two_slopes(self):
        assert slope_at(lambda x: jnp.minimum(x, EDGE), EDGE) == 0.5
        assert slope_at(lambda x: jnp.minimum(EDGE, x), EDGE) == 0.5

    def test_clip_averages_at_both_of_its_edges(self):
        def clipped(x):
            return jnp.clip(x, EDGE, 2 * EDGE)

        assert slope_at(clipped, EDGE) == 0.5
        assert slope_at(clipped, 2 * EDGE) == 0.5

    def test_absolute_value_reports_the_upper_slope_at_zero(self):
        assert slope_at(jnp.abs, 0.0) == 1.0

    def test_relu_reports_the_lower_slope_at_zero(self):
        assert slope_at(jax.nn.relu, 0.0) == 0.0


class TestTheChosenSpelling:
    """A `where` puts the tie wherever its comparison puts it, and nowhere else."""

    def test_strictly_greater_stands_on_the_branch_below(self):
        assert slope_at(lambda x: jnp.where(x > EDGE, x, EDGE), EDGE) == 0.0

    def test_greater_or_equal_stands_on_the_branch_above(self):
        assert slope_at(lambda x: jnp.where(x >= EDGE, x, EDGE), EDGE) == 1.0

    def test_a_rising_amount_exactly_at_its_floor_counts_as_above_it(self):
        # The sloped branch: one more dollar of income is one more dollar out.
        assert slope_at(lambda x: floored_at(x, EDGE), EDGE) == 1.0
        assert slope_at(lambda x: floored_at(x, 0.0), 0.0) == 1.0

    def test_a_rising_amount_exactly_at_its_cap_counts_as_capped(self):
        # The flat branch: one more dollar of income changes nothing.
        assert slope_at(lambda x: capped_at(x, EDGE), EDGE) == 0.0

    def test_a_falling_amount_exactly_at_its_floor_has_already_reached_it(self):
        # A credit that is exactly used up: one more dollar of income leaves it
        # at zero, so the phaseout rate is not part of the marginal rate here.
        def falling(x):
            return floored_at(x, EDGE, sinks_against_the_limit=True)

        assert slope_at(falling, EDGE) == 0.0
        assert slope_at(falling, EDGE + 1.0) == 1.0

    def test_a_falling_amount_exactly_at_its_cap_is_on_its_way_back_under(self):
        def falling(x):
            return capped_at(x, EDGE, sinks_against_the_limit=True)

        assert slope_at(falling, EDGE) == 1.0
        assert slope_at(falling, EDGE + 1.0) == 0.0

    @pytest.mark.parametrize("amount", [EDGE - 1.0, EDGE + 1.0])
    def test_away_from_the_edge_the_spellings_agree(self, amount):
        assert slope_at(lambda x: floored_at(x, EDGE), amount) == slope_at(
            lambda x: jnp.maximum(x, EDGE), amount
        )
        assert slope_at(lambda x: capped_at(x, EDGE), amount) == slope_at(
            lambda x: jnp.minimum(x, EDGE), amount
        )


class TestTheValuesThemselves:
    """Changing the spelling was allowed to change a slope, never a number."""

    @pytest.mark.parametrize("amount", [-1.0, 0.0, EDGE - 0.5, EDGE, EDGE + 0.5, 1e9])
    def test_floored_at_returns_what_maximum_returns(self, amount):
        assert float(floored_at(jnp.float64(amount), EDGE)) == float(
            jnp.maximum(jnp.float64(amount), EDGE)
        )

    @pytest.mark.parametrize("amount", [-1.0, 0.0, EDGE - 0.5, EDGE, EDGE + 0.5, 1e9])
    def test_capped_at_returns_what_minimum_returns(self, amount):
        assert float(capped_at(jnp.float64(amount), EDGE)) == float(
            jnp.minimum(jnp.float64(amount), EDGE)
        )

    @pytest.mark.parametrize("amount", [-1.0, 0.0, EDGE - 0.5, EDGE, EDGE + 0.5, 1e9])
    def test_the_falling_spellings_return_the_same_numbers_too(self, amount):
        # Only the slope at a tie moves; the numbers a run produces do not.
        x = jnp.float64(amount)
        assert float(floored_at(x, EDGE, sinks_against_the_limit=True)) == float(
            jnp.maximum(x, EDGE)
        )
        assert float(capped_at(x, EDGE, sinks_against_the_limit=True)) == float(
            jnp.minimum(x, EDGE)
        )

    def test_an_infinite_edge_behaves_the_same_way(self):
        # An unlimited cap and an absent bracket both arrive as infinity.
        assert float(capped_at(jnp.float64(7.0), jnp.inf)) == 7.0
        assert float(floored_at(jnp.float64(7.0), -jnp.inf)) == 7.0
        assert float(floored_at(jnp.float64(-jnp.inf), 0.0)) == 0.0

    def test_a_nan_amount_is_the_one_difference(self):
        # `jnp.maximum` propagates a NaN and these do not, which is why a law
        # array with NaN-padded slots goes through `live_slots` first.
        assert jnp.isnan(jnp.maximum(jnp.float64(jnp.nan), 0.0))
        assert float(floored_at(jnp.float64(jnp.nan), 0.0)) == 0.0
