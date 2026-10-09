"""The heir allocation on a population with the awkward cases in it.

`reporting/estate_allocator_jax.py` spreads the estate tax and the tax on
gains deemed realized at death over heirs by ranking estates against
inheritances. This file checks that it balances on a population messier than
the hand-worked four households of `test_estate_allocator.py`, and that
walking all five worlds in one batched pass gives what walking them one at a
time gives.

The population is a hundred households with a spread of estates and
inheritance prospects, built from a fixed random seed so that the
awkward cases are in it -- households certain not to inherit, a household
whose published chance of inheriting is above one, estates with nothing left
to distribute -- rather than only the tidy ones.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.reporting import estate_allocator_jax
from tests.tolerances import CENT
from tests.unit.reporting.conftest import named

N_HOUSEHOLDS = 100


def a_population() -> dict[str, np.ndarray]:
    draw = np.random.default_rng(20260804)
    weight = np.round(draw.uniform(1.0, 500.0, N_HOUSEHOLDS), 3)
    chance_of_inheriting = np.clip(draw.uniform(-0.3, 0.9, N_HOUSEHOLDS), 0.0, None)
    # One household stands to inherit from more than one estate, which is what
    # a published chance above one means. The chance is kept and the non-heir
    # copy, whose weight would go negative, is dropped.
    chance_of_inheriting[7] = 2.03
    inheritance = np.round(np.exp(draw.normal(13.5, 2.0, N_HOUSEHOLDS)), 2)
    chance_of_dying = np.clip(draw.uniform(-0.02, 0.05, N_HOUSEHOLDS), 0.0, None)
    distributable = np.round(np.exp(draw.normal(13.0, 2.5, N_HOUSEHOLDS)), 2)
    # Two estates owe tax but have nothing left to distribute, which the
    # allocation cannot pin to any bequest dollars.
    distributable[np.array([3, 44])] = 0.0
    estate_tax = np.where(
        distributable > 400_000.0, np.round(0.4 * (distributable - 400_000.0), 2), 0.0
    )
    estate_tax[np.array([3, 44])] = 5_000.0
    return {
        "record_id": np.arange(N_HOUSEHOLDS, dtype=np.int64),
        "weight": weight,
        "chance_of_dying": chance_of_dying,
        "chance_of_unused_spousal_exemption": draw.uniform(0.0, 1.0, N_HOUSEHOLDS),
        "estate_tax_with_it": estate_tax,
        "estate_tax_without_it": np.round(estate_tax * 1.2, 2),
        "distributable_estate": distributable,
        "probability_of_inheriting": chance_of_inheriting,
        "inheritance": inheritance,
    }


def test_the_estate_tax_balances_on_a_hundred_awkward_households():
    """Every dollar of estate tax either reaches an heir or is reported.

    The two estates that owe tax and have nothing left to distribute cannot be
    pinned to any bequest dollars, so their tax is reported as unallocated
    rather than spread over heirs. Everything else adds up to the cent.
    """
    people = a_population()
    compiled, diagnostics = estate_allocator_jax.allocate_estate_tax(
        weight=people["weight"],
        mortality_probability=people["chance_of_dying"],
        probability_of_unused_spousal_exemption=people[
            "chance_of_unused_spousal_exemption"
        ],
        liability_with_unused_spousal_exemption=people["estate_tax_with_it"],
        liability_without_unused_spousal_exemption=people["estate_tax_without_it"],
        distributable_estate=people["distributable_estate"],
        probability_of_inheriting=people["probability_of_inheriting"],
        inheritance=people["inheritance"],
        record_id=people["record_id"],
    )
    totals = named(diagnostics)

    # Every dollar that went onto the line comes off it again.
    assert totals["allocated_dollars"] == pytest.approx(
        totals["tax_dollars"], abs=CENT
    )
    # The heirs bear what was allocated: each household's tax if it inherits,
    # weighted by the population it stands for and its chance of inheriting.
    borne = np.sum(
        np.asarray(compiled)
        * people["weight"]
        * people["probability_of_inheriting"]
    )
    assert borne == pytest.approx(totals["allocated_dollars"], abs=CENT)

    # Slots 3 and 44 owe estate tax on an estate with nothing to distribute.
    # Each puts two entries on the line, one for each answer to whether a
    # deceased spouse's unused exemption is available, and the death weight is
    # split between them by the chance of each.
    with_nothing_to_leave = np.array([3, 44])
    death_weight = (people["weight"] * people["chance_of_dying"])[with_nothing_to_leave]
    with_the_exemption = people["chance_of_unused_spousal_exemption"][
        with_nothing_to_leave
    ]
    stranded = np.sum(
        death_weight
        * (
            with_the_exemption * people["estate_tax_with_it"][with_nothing_to_leave]
            + (1.0 - with_the_exemption)
            * people["estate_tax_without_it"][with_nothing_to_leave]
        )
    )
    assert totals["unallocated_dollars"] == pytest.approx(stranded, abs=CENT)


def test_the_deemed_gains_line_holds_every_dying_estate():
    """The tax on gains deemed realized at death takes the other convention.

    Every dying estate goes on this line, whatever it owes, because an estate
    with no unrealized gains can be any size and leaving it off would push the
    taxed bequest dollars onto larger heirs. Estates owing nothing and estates
    whose tax comes out a shade below zero, which is the rounding noise of the
    two calculator calls the tax is the difference of, are both on it. The
    bequest dollars the line carries are what show the convention.
    """
    people = a_population()
    expected_tax = (
        people["chance_of_dying"] * np.round(0.2 * people["distributable_estate"], 2)
    )
    # Two dying estates come out a hundredth of a cent below zero.
    rounding_noise = np.array([5, 60])
    expected_tax[rounding_noise] = -1e-14
    # Two dying estates hold no unrealized gains and so owe nothing at death.
    # They are large enough to matter, and they stay on the line.
    estates_with_no_gains = np.array([10, 11])
    expected_tax[estates_with_no_gains] = 0.0
    compiled, diagnostics = estate_allocator_jax.allocate_deemed_gains_tax(
        weight=people["weight"],
        mortality_probability=people["chance_of_dying"],
        distributable_estate=people["distributable_estate"],
        expected_deemed_gains_tax=expected_tax,
        probability_of_inheriting=people["probability_of_inheriting"],
        inheritance=people["inheritance"],
        record_id=people["record_id"],
    )
    totals = named(diagnostics)

    death_weight = people["weight"] * people["chance_of_dying"]
    on_the_line = (death_weight > 0.0) & (people["distributable_estate"] > 0.0)
    assert totals["bequest_dollars"] == pytest.approx(
        np.sum(death_weight[on_the_line] * people["distributable_estate"][on_the_line]),
        abs=CENT,
    )
    # The estates owing nothing and the two carrying rounding noise are part of
    # that total: they hold their place between the heirs and the estates that
    # do owe.
    assert on_the_line[estates_with_no_gains].all()
    assert on_the_line[rounding_noise].all()

    assert totals["allocated_dollars"] == pytest.approx(
        totals["tax_dollars"], abs=CENT
    )
    borne = np.sum(
        np.asarray(compiled)
        * people["weight"]
        * people["probability_of_inheriting"]
    )
    assert borne == pytest.approx(totals["allocated_dollars"], abs=CENT)


def test_all_five_worlds_in_one_pass_match_one_world_at_a_time():
    """The batched allocation against the one-line one it replaced.

    A projection year spreads two taxes at death over heirs in each of five
    worlds, and `allocate_taxes_at_death` ranks all ten lines of estates in one
    sorted array rather than ten. This is the check that batching them changes
    no answer. The five worlds are given slightly different chances of dying so
    that a mistake mixing two lines up would show.
    """
    people = a_population()
    n_worlds = 5
    per_world = lambda values: jnp.stack(
        [jnp.asarray(values) * (1.0 + world / 100.0) for world in range(n_worlds)]
    )
    mortality = per_world(people["chance_of_dying"])
    deemed_tax = per_world(
        people["chance_of_dying"] * np.round(0.2 * people["distributable_estate"], 2)
    )
    estate_tax, deemed_gains_tax, diagnostics = (
        estate_allocator_jax.allocate_taxes_at_death(
            weight=people["weight"],
            record_id=people["record_id"],
            mortality_probability=mortality,
            probability_of_unused_spousal_exemption=per_world(
                people["chance_of_unused_spousal_exemption"]
            ),
            liability_with_unused_spousal_exemption=per_world(
                people["estate_tax_with_it"]
            ),
            liability_without_unused_spousal_exemption=per_world(
                people["estate_tax_without_it"]
            ),
            distributable_estate=per_world(people["distributable_estate"]),
            expected_deemed_gains_tax=deemed_tax,
            deemed_gains_mortality_probability=mortality,
            probability_of_inheriting=people["probability_of_inheriting"],
            inheritance=people["inheritance"],
        )
    )
    assert estate_tax.shape == (n_worlds, N_HOUSEHOLDS)
    assert diagnostics.shape == (
        n_worlds,
        2,
        len(estate_allocator_jax.DIAGNOSTIC_NAMES),
    )

    for world in range(n_worlds):
        one_at_a_time, its_diagnostics = estate_allocator_jax.allocate_estate_tax(
            weight=people["weight"],
            mortality_probability=mortality[world],
            probability_of_unused_spousal_exemption=per_world(
                people["chance_of_unused_spousal_exemption"]
            )[world],
            liability_with_unused_spousal_exemption=per_world(
                people["estate_tax_with_it"]
            )[world],
            liability_without_unused_spousal_exemption=per_world(
                people["estate_tax_without_it"]
            )[world],
            distributable_estate=per_world(people["distributable_estate"])[world],
            probability_of_inheriting=people["probability_of_inheriting"],
            inheritance=people["inheritance"],
            record_id=people["record_id"],
        )
        assert np.asarray(estate_tax[world]) == pytest.approx(
            np.asarray(one_at_a_time), abs=CENT
        )
        assert np.asarray(diagnostics[world, 0]) == pytest.approx(
            np.asarray(its_diagnostics), abs=CENT
        )

        deemed_one_at_a_time, _ = estate_allocator_jax.allocate_deemed_gains_tax(
            weight=people["weight"],
            mortality_probability=mortality[world],
            distributable_estate=per_world(people["distributable_estate"])[world],
            expected_deemed_gains_tax=deemed_tax[world],
            probability_of_inheriting=people["probability_of_inheriting"],
            inheritance=people["inheritance"],
            record_id=people["record_id"],
        )
        assert np.asarray(deemed_gains_tax[world]) == pytest.approx(
            np.asarray(deemed_one_at_a_time), abs=CENT
        )
