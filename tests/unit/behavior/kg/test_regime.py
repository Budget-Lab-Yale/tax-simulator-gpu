"""What happens to a decedent's gain, averaged over an age cell.

Each case is a single age cell whose gains are split between two asset classes,
so the weighted average can be read off by hand.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from taxmodel.behavior.kg import regime

# The five tracked classes, in order: equities, pass-throughs, primary home,
# other home, real estate funds.
STEP_UP_EVERYTHING = (0, 0, 0, 0, 0)
CARRYOVER_EVERYTHING = (1, 1, 1, 1, 1)
DEEMED_EVERYTHING = (2, 2, 2, 2, 2)

# One cell holding six hundred dollars of gain in equities and four hundred in
# a primary home, with nothing excluded.
ONE_CELL = jnp.asarray([[600.0, 0.0, 400.0, 0.0, 0.0]])


def _death_shares(whole, at_stake, regime_codes):
    """The three death shares, from regime codes as a year's law gives them."""
    return regime.death_shares_from_regime_weights(
        whole, at_stake, jnp.asarray(regime.what_becomes_of_a_dollar(regime_codes))
    )


def test_current_law_forgives_every_dollar_a_decedent_holds():
    """A stepped-up basis on every class: nothing is taxed, ever."""

    forgiven, routed, taxed = _death_shares(
        ONE_CELL, ONE_CELL, STEP_UP_EVERYTHING
    )

    assert float(forgiven[0]) == pytest.approx(1.0)
    assert float(routed[0]) == 0.0
    assert float(taxed[0]) == 0.0


def test_deemed_realization_taxes_every_dollar_on_the_final_return():
    """Nothing is forgiven and nothing moves to the heir."""

    forgiven, _routed, taxed = _death_shares(
        ONE_CELL, ONE_CELL, DEEMED_EVERYTHING
    )

    assert float(forgiven[0]) == 0.0
    assert float(taxed[0]) == pytest.approx(1.0)


def test_carryover_moves_every_dollar_to_the_heir():
    """Nothing is collected in the year of death; the gain keeps its basis."""

    forgiven, routed, taxed = _death_shares(
        ONE_CELL, ONE_CELL, CARRYOVER_EVERYTHING
    )

    assert float(routed[0]) == pytest.approx(1.0)
    assert float(forgiven[0]) == 0.0
    assert float(taxed[0]) == 0.0


def test_a_cell_averages_its_classes_by_their_share_of_its_gain():
    """Equities deemed realized, the house stepped up: six-tenths and four."""

    equities_deemed_house_stepped_up = (2, 0, 0, 0, 0)

    forgiven, routed, taxed = _death_shares(
        ONE_CELL, ONE_CELL, equities_deemed_house_stepped_up
    )

    assert float(taxed[0]) == pytest.approx(0.6)
    assert float(forgiven[0]) == pytest.approx(0.4)
    assert float(routed[0]) == 0.0


def test_an_exclusion_forgives_its_dollars_rather_than_routing_them():
    """The excluded gain must not be counted twice.

    An exclusion works like a step-up in basis up to its cap, so the dollars
    below it are forgiven whatever regime the class is under. Here half the
    cell's gain is excluded, so half is routed to heirs and half is forgiven,
    and the three shares add to one.
    """

    whole = jnp.asarray([[1000.0, 0.0, 0.0, 0.0, 0.0]])
    after_the_exclusion = jnp.asarray([[500.0, 0.0, 0.0, 0.0, 0.0]])

    forgiven, routed, taxed = _death_shares(
        whole, after_the_exclusion, CARRYOVER_EVERYTHING
    )

    assert float(routed[0]) == pytest.approx(0.5)
    assert float(forgiven[0]) == pytest.approx(0.5)
    assert float(taxed[0]) == 0.0
    assert float(forgiven[0] + routed[0] + taxed[0]) == pytest.approx(1.0)


def test_a_cell_holding_no_gain_gets_zeros_and_no_missing_numbers():
    """An empty cell must not divide by its own zero."""

    empty = jnp.zeros((1, 5))

    shares = _death_shares(
        empty, empty, DEEMED_EVERYTHING
    )

    for share in shares:
        assert float(share[0]) == 0.0


def test_an_unknown_treatment_of_gains_at_death_stops_the_run():
    """Rather than being read as one of the three."""

    with pytest.raises(ValueError, match="death regime 7"):
        _death_shares(
            ONE_CELL, ONE_CELL, (7, 0, 0, 0, 0)
        )


def test_the_applier_is_told_which_classes_a_decedent_realizes():
    """One share per class: the share that goes on the final return."""

    shares = regime.realized_at_death_share_by_class((2, 1, 0, 0, 2))

    assert shares == (1.0, 0.0, 0.0, 0.0, 1.0)
