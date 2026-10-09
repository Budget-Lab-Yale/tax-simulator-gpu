"""The heir allocator, on a population small enough to match estates by hand.

Four households. Two of them can die this year and leave an estate; two of
them can inherit. Everything is chosen so the matching can be done on paper.

The decedents, both married so there is no unused-spousal-exemption branch to
split (the two liabilities are equal and the probability of the branch is
zero):

    slot  weight  chance of dying  distributable estate  estate tax if it dies
    0         10             0.01               1,000              400
    1         10             0.01                 500               50

Weighting by the chance of dying, slot 0 puts 10 x 0.01 x 1,000 = 100 bequest
dollars on the line carrying 10 x 0.01 x 400 = 40 tax dollars, and slot 1 puts
50 bequest dollars carrying 5 tax dollars. Slot 0 has the larger estate, so it
comes first: the cumulative-tax curve runs 0 to 40 over the first 100 bequest
dollars, then 40 to 45 over the next 50, and stops at 150 bequest dollars
carrying 45 tax dollars in all.

The heirs:

    slot  weight  chance of inheriting  inheritance
    2         10                   0.5          100
    3         10                   0.1           50

Slot 2 has the larger inheritance and comes first, with 10 x 0.5 x 100 = 500
inheritance dollars, and slot 3 brings 10 x 0.1 x 50 = 50. Slot 2's 500
dollars already run past the 150 bequest dollars on the line, so slot 2
absorbs the whole 45 dollars of tax and slot 3 gets nothing. Slot 2's tax
conditional on inheriting is 45 / (10 x 0.5) = 9 dollars, which on a 100
dollar inheritance is a 9 percent average rate.

The allocator runs inside the compiled year, so it cannot warn or stop; it
reports five totals instead, and `reporting/assemble.py` reads them once the run is
back on the host. The tests below therefore read the diagnostics where the
old host-side allocator raised or warned.
"""

from __future__ import annotations

import numpy as np
import pytest

from taxmodel.reporting import estate_allocator_jax
from tests.tolerances import CENT
from tests.unit.reporting.conftest import named

WEIGHT = np.array([10.0, 10.0, 10.0, 10.0])
RECORD_ID = np.arange(4, dtype=np.int64)
CHANCE_OF_DYING = np.array([0.01, 0.01, 0.0, 0.0])
DISTRIBUTABLE_ESTATE = np.array([1_000.0, 500.0, 0.0, 0.0])
ESTATE_TAX_IF_IT_DIES = np.array([400.0, 50.0, 0.0, 0.0])
# Both decedents are married, so no unused spousal exemption is in play and
# the two branches of the estate calculation carry the same liability.
CHANCE_OF_UNUSED_SPOUSAL_EXEMPTION = np.zeros(4)

CHANCE_OF_INHERITING = np.array([0.0, 0.0, 0.5, 0.1])
INHERITANCE = np.array([0.0, 0.0, 100.0, 50.0])

TOTAL_ESTATE_TAX = 45.0


def allocate_the_estate_tax(**changes):
    """The four-household estate tax allocation, with any argument replaced."""
    arguments = {
        "weight": WEIGHT,
        "mortality_probability": CHANCE_OF_DYING,
        "probability_of_unused_spousal_exemption": (
            CHANCE_OF_UNUSED_SPOUSAL_EXEMPTION
        ),
        "liability_with_unused_spousal_exemption": ESTATE_TAX_IF_IT_DIES,
        "liability_without_unused_spousal_exemption": ESTATE_TAX_IF_IT_DIES,
        "distributable_estate": DISTRIBUTABLE_ESTATE,
        "probability_of_inheriting": CHANCE_OF_INHERITING,
        "inheritance": INHERITANCE,
        "record_id": RECORD_ID,
    }
    per_heir, diagnostics = estate_allocator_jax.allocate_estate_tax(
        **{**arguments, **changes}
    )
    return np.asarray(per_heir), named(diagnostics)


def test_the_largest_heir_absorbs_the_taxed_bequests():
    """The heir with the larger inheritance takes all the tax; the other none."""
    per_heir, _diagnostics = allocate_the_estate_tax()
    assert per_heir == pytest.approx([0.0, 0.0, 9.0, 0.0])


def test_a_smaller_heir_picks_up_the_stretch_the_larger_one_leaves():
    """Shrink the larger heir so its dollars stop part-way along the estates.

    The estates are the ones described at the top: 100 bequest dollars taxed at
    40 cents on the dollar, then 50 more taxed at 10 cents, 150 in all carrying
    45 dollars of tax. Slot 2 keeps the larger inheritance, 24 dollars, and
    with a weight of ten and a one-in-two chance of inheriting brings 120
    inheritance dollars. Those cover the first 100 bequest dollars, worth 40 in
    tax, and 20 dollars of the second stretch, worth 2 more, so slot 2 bears 42
    dollars, or 42/5 = 8.40 for each household that inherits. Slot 3's 40
    dollars pick up the last 30 taxed bequest dollars, worth 3 dollars, or
    3/2 = 1.50 each, and its remaining 10 dollars sit past the end of the
    taxed estates and are untaxed.
    """
    per_heir, _diagnostics = allocate_the_estate_tax(
        probability_of_inheriting=np.array([0.0, 0.0, 0.5, 0.2]),
        inheritance=np.array([0.0, 0.0, 24.0, 20.0]),
    )
    assert per_heir == pytest.approx([0.0, 0.0, 8.4, 1.5])


def test_the_allocated_tax_adds_up_to_the_tax_owed():
    """Every dollar of expected estate tax reaches an heir, exactly."""
    per_heir, diagnostics = allocate_the_estate_tax()
    allocated = np.sum(per_heir * WEIGHT * CHANCE_OF_INHERITING)
    assert allocated == pytest.approx(TOTAL_ESTATE_TAX, abs=CENT)
    assert diagnostics["allocated_dollars"] == pytest.approx(
        TOTAL_ESTATE_TAX, abs=CENT
    )
    assert diagnostics["tax_dollars"] == pytest.approx(TOTAL_ESTATE_TAX, abs=CENT)
    assert diagnostics["bequest_dollars"] == pytest.approx(150.0, abs=CENT)
    assert diagnostics["inheritance_dollars"] == pytest.approx(550.0, abs=CENT)


def test_the_adding_up_holds_on_a_thousand_random_estates():
    """The identity is a property of the matching, not of the four households."""
    generator = np.random.default_rng(20260803)
    n_households = 1_000
    weight = generator.uniform(1.0, 500.0, n_households)
    chance_of_dying = generator.uniform(0.0, 0.05, n_households)
    estate = generator.uniform(0.0, 5e6, n_households)
    tax = estate * generator.uniform(0.0, 0.4, n_households)
    chance_of_inheriting = generator.uniform(0.0, 0.3, n_households)
    inheritance = generator.uniform(0.0, 2e6, n_households)

    _per_heir, diagnostics = estate_allocator_jax.allocate_estate_tax(
        weight=weight,
        mortality_probability=chance_of_dying,
        probability_of_unused_spousal_exemption=generator.uniform(
            0.0, 1.0, n_households
        ),
        liability_with_unused_spousal_exemption=tax,
        liability_without_unused_spousal_exemption=tax * 1.2,
        distributable_estate=estate,
        probability_of_inheriting=chance_of_inheriting,
        inheritance=inheritance,
        record_id=np.arange(n_households, dtype=np.int64),
    )
    totals = named(diagnostics)
    # The heir data has to have room for the estates, or the shortfall below
    # would be exhaustion rather than a failure of the identity.
    assert totals["bequest_dollars"] <= totals["inheritance_dollars"]
    assert totals["allocated_dollars"] == pytest.approx(
        totals["tax_dollars"], abs=CENT
    )


def test_more_taxed_bequests_than_inheritances_is_refused():
    """Rates are never scaled up to absorb estates the heir data cannot receive.

    Multiply every estate by a thousand and the taxed bequest dollars run past
    the end of the heir data. The allocator does not stretch the heirs to cover
    them: the estates past the crossing are matched to nobody, so less tax is
    allocated than is owed, and the two totals that show it are reported. The
    stop itself is on the host, in `reporting/assemble._check_one_ladder`, which turns
    exactly this pair of totals into an error for the estate tax's own ladder.
    """
    _per_heir, diagnostics = allocate_the_estate_tax(
        distributable_estate=DISTRIBUTABLE_ESTATE * 1_000.0,
    )
    assert diagnostics["bequest_dollars"] > diagnostics["inheritance_dollars"]
    assert diagnostics["allocated_dollars"] < diagnostics["tax_dollars"] - CENT


def test_tax_on_estates_with_nothing_to_distribute_stays_unallocated():
    """An estate that owes tax but leaves nothing has no bequests to match."""
    _per_heir, diagnostics = allocate_the_estate_tax(
        # Slot 1 owes 50 dollars of estate tax on lifetime gifts alone and
        # has nothing left to leave.
        distributable_estate=np.array([1_000.0, 0.0, 0.0, 0.0]),
    )
    # Slot 0's 40 dollars are allocated; slot 1's 5 are not.
    assert diagnostics["allocated_dollars"] == pytest.approx(40.0, abs=CENT)
    assert diagnostics["unallocated_dollars"] == pytest.approx(5.0, abs=CENT)


def test_the_deemed_gains_tax_keeps_untaxed_estates_on_the_line():
    """An estate with no gains still stands between heirs and the taxed ones.

    Slot 0 and slot 1 both die with probability 0.01 and both leave estates, so
    both are on the line: 100 bequest dollars from slot 0 and 50 from slot 1.
    Only slot 0 owes tax on deemed gains, 40 dollars of it in expectation, so
    the cumulative-tax curve rises to 40 over the first 100 dollars and stays
    flat over the next 50. Slot 2's inheritance dollars cover the whole line
    and it takes all 40.
    """
    per_heir, diagnostics = estate_allocator_jax.allocate_deemed_gains_tax(
        weight=WEIGHT,
        mortality_probability=CHANCE_OF_DYING,
        distributable_estate=DISTRIBUTABLE_ESTATE,
        # 10 x 0.01 x 400 = 40 dollars expected, so the conditional liability
        # the allocator recovers is 400.
        expected_deemed_gains_tax=np.array([4.0, 0.0, 0.0, 0.0]),
        probability_of_inheriting=CHANCE_OF_INHERITING,
        inheritance=INHERITANCE,
        record_id=RECORD_ID,
    )
    totals = named(diagnostics)
    # Both estates are on the line, the untaxed one included: 100 bequest
    # dollars from slot 0 and 50 from slot 1.
    assert totals["bequest_dollars"] == pytest.approx(150.0, abs=CENT)
    assert np.asarray(per_heir) == pytest.approx([0.0, 0.0, 8.0, 0.0])
    assert totals["allocated_dollars"] == pytest.approx(40.0, abs=CENT)


def test_a_rounding_sized_negative_tax_keeps_an_estate_on_the_line():
    """A deemed-gains tax of minus a hundredth of a cent is still on the line.

    Pricing the deemed gain takes the difference of two calculator runs, so an
    estate with a deemed gain of exactly zero can come out owing a fraction of
    a cent below zero. That estate belongs on the line like any other: its
    bequest dollars stand between heirs and the estates that do owe tax.
    """
    per_heir, diagnostics = estate_allocator_jax.allocate_deemed_gains_tax(
        weight=WEIGHT,
        mortality_probability=CHANCE_OF_DYING,
        distributable_estate=DISTRIBUTABLE_ESTATE,
        expected_deemed_gains_tax=np.array([4.0, -1e-14, 0.0, 0.0]),
        probability_of_inheriting=CHANCE_OF_INHERITING,
        inheritance=INHERITANCE,
        record_id=RECORD_ID,
    )
    totals = named(diagnostics)
    # Slot 1's 50 bequest dollars are counted, as they are when its tax is
    # exactly zero in the test above.
    assert totals["bequest_dollars"] == pytest.approx(150.0, abs=CENT)
    assert np.asarray(per_heir) == pytest.approx([0.0, 0.0, 8.0, 0.0])
    assert totals["allocated_dollars"] == pytest.approx(40.0, abs=CENT)


def test_a_world_with_no_tax_at_death_allocates_nothing():
    """Repeal the estate tax and every heir bears nothing, without complaint."""
    per_heir, diagnostics = allocate_the_estate_tax(
        liability_with_unused_spousal_exemption=np.zeros(4),
        liability_without_unused_spousal_exemption=np.zeros(4),
    )
    assert per_heir == pytest.approx(np.zeros(4))
    assert diagnostics["tax_dollars"] == 0.0


def test_the_deemed_gains_tax_uses_the_chance_of_dying_it_was_formed_with():
    """The deemed-gains line reads the capital-gains model's chance of dying.

    The estate tax and the capital-gains model can give one household
    different chances of dying, as for a joint return on which the second
    spouse's chance is recorded as zero. Here slot 0's estate-tax chance is
    0.005 and its capital-gains chance is 0.01. The expected deemed-gains tax
    was formed with the second: 0.01 x 400 = 4 dollars per household, or 40
    dollars over its weight of ten.

    At that chance of dying, slot 0 is one estate owing 400, putting 100
    bequest dollars on the line carrying 40 dollars of tax, followed by slot
    1's 50 untaxed bequest dollars. The heirs are made smaller so that where
    the tax lands depends on that: slot 2 brings 10 x 0.06 x 100 = 60
    inheritance dollars and slot 3 brings 10 x 0.1 x 50 = 50. Slot 2 covers
    the first 60 of slot 0's 100 bequest dollars and bears 24 dollars, 40 for
    each household that inherits; slot 3 covers the other 40 and bears 16.

    Dividing by the estate tax's chance of dying instead would put half as
    many slot-0 deaths on the line, each owing 800, and slot 2 would bear all
    40 dollars.
    """
    estate_tax_chance_of_dying = np.array([0.005, 0.01, 0.0, 0.0])
    expected_deemed_gains_tax = np.array([4.0, 0.0, 0.0, 0.0])
    probability_of_inheriting = np.array([0.0, 0.0, 0.06, 0.1])
    one_world = lambda values: np.asarray(values)[None, :]
    _estate_tax, deemed_gains_tax, diagnostics = (
        estate_allocator_jax.allocate_taxes_at_death(
            weight=WEIGHT,
            record_id=RECORD_ID,
            mortality_probability=one_world(estate_tax_chance_of_dying),
            probability_of_unused_spousal_exemption=one_world(
                CHANCE_OF_UNUSED_SPOUSAL_EXEMPTION
            ),
            liability_with_unused_spousal_exemption=one_world(np.zeros(4)),
            liability_without_unused_spousal_exemption=one_world(np.zeros(4)),
            distributable_estate=one_world(DISTRIBUTABLE_ESTATE),
            expected_deemed_gains_tax=one_world(expected_deemed_gains_tax),
            deemed_gains_mortality_probability=one_world(CHANCE_OF_DYING),
            probability_of_inheriting=probability_of_inheriting,
            inheritance=INHERITANCE,
        )
    )

    line = estate_allocator_jax._deemed_gains_line(
        weight=WEIGHT,
        mortality_probability=CHANCE_OF_DYING,
        distributable_estate=DISTRIBUTABLE_ESTATE,
        expected_deemed_gains_tax=expected_deemed_gains_tax,
        record_id=RECORD_ID,
        padded_to_the_estate_line=False,
    )
    # Each dying estate owes its own deemed-gains tax, 400 for slot 0, at the
    # capital-gains model's chance of dying.
    assert np.asarray(line.liability) == pytest.approx([400.0, 0.0, 0.0, 0.0])
    assert np.asarray(line.death_weight) == pytest.approx([0.1, 0.1, 0.0, 0.0])

    deemed = named(diagnostics[0, 1])
    assert deemed["bequest_dollars"] == pytest.approx(150.0, abs=CENT)
    assert np.asarray(deemed_gains_tax[0]) == pytest.approx([0.0, 0.0, 40.0, 16.0])
    total_assigned = np.sum(
        np.asarray(deemed_gains_tax[0]) * WEIGHT * probability_of_inheriting
    )
    assert total_assigned == pytest.approx(40.0, abs=CENT)
    assert deemed["tax_dollars"] == pytest.approx(40.0, abs=CENT)
