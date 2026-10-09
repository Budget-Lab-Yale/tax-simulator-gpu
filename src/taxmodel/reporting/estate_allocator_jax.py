"""Passing the tax owed at death through to the heirs who bear it.

The estate tax is paid out of the estate, but the people who are worse off for
it are the heirs, who receive that much less. The same is true of the income
tax on gains deemed realized at death under a reform that repeals step-up in
basis. So a distribution table that wants to say who bears the reform's cost
has to move both taxes off the decedent and onto heirs. The rest of the model
already assumes this happens: ``years/burden.py`` leaves tax on deemed gains out
of the household's during-life burden, on the understanding that it comes back
here.

Nobody observes which household inherits from which estate. The upstream
Estate-Tax-Distribution model supplies, per household, the probability that
the household receives an inheritance in the year and how much it would
receive. The allocation assumes that bigger inheritances come from bigger
estates. Estates are lined up largest first and heirs are lined up largest
inheritance first, and the two lines are matched dollar for dollar. An heir
whose inheritance dollars sit opposite estates paying an average of 30 cents
of tax per dollar of bequest is charged 30 percent of the inheritance. Heirs
whose dollars sit past the end of the taxed estates pay nothing.

The matching runs inside the compiled year, because a household's estate tax
never leaves the device.

Two details of the matching change the answer.

*On the estate side a single decedent puts two entries on the line.* Whether
a surviving spouse's unused exemption is available decides whether a single
filer's estate is taxable at all near the exemption, and averaging the two
branches together would smear the tax rate across estates that do and do not
owe. Each branch carries the share of the death probability that belongs to
it. A married decedent has no such branch and the two entries coincide.

*On the deemed-gains side estates owing nothing stay on the line.* An estate
with no unrealized gains can be any size, so dropping it would push the taxed
bequest dollars onto larger heirs and leave smaller heirs untaxed. On the
estate-tax side, an estate that owes estate tax but has nothing distributable
(debts wipe out the estate, and the add-back of lifetime gifts alone exceeds
the exemption) carries tax with no bequest dollars to match it to. That tax is
left unallocated and reported in the diagnostics.

The heir's income is the distribution table's business: whenever the table
charges heirs the tax at death it also adds the expected inheritance to heir
income, including the income it ranks on.

One projection year has ten lines to walk: each of the five worlds spreads the
estate tax over heirs and spreads the tax on gains deemed realized at death
over them, and each of those is its own ranking of estates against
inheritances. `allocate_taxes_at_death` stacks the lines into one array with a
line axis and does the arithmetic in a single batched sort. What a household
inherits does not depend on which tax is being spread over it, so the heirs'
ranking is worked out once and every line reads it.

Running inside a compiled program has two consequences for the code.

*Entries are zeroed and sorted to the end instead of filtered out.* A compiled
program cannot change an array's length, so an estate that does not belong on
the line (no death weight, no tax, or nothing left to distribute) keeps its
slot with zero bequest dollars and zero tax dollars, and the sort puts every
such slot after the ones that count. Cumulative bequests and cumulative tax
are flat across the tail, so an heir whose dollars run into it picks up
nothing.

*The checks report numbers instead of raising.* The allocation warns when tax
cannot be matched to any heir and stops when it does not add up or when more
bequest dollars are taxed than there are inheritance dollars to receive them.
Neither is available inside a traced function, so the five totals those
checks need come back as an array and the host reads them after the run, in
`reporting/assemble.check_estate_allocation`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray

# The columns the estate calculation produces for each household, which the
# estate side of the allocation needs. A run without the estate measurement
# bridge produces none of them and no allocation happens.
ESTATE_RESULT_COLUMNS = (
    "liab_estate_dsue",
    "liab_estate_nodsue",
    "estate_p_dsue",
    "estate_distributable",
)

# The expected income tax on gains deemed realized at death, reported beside
# the income-tax columns it was folded into so a reader can see how much of
# the reform's revenue is this. A scenario that runs the capital-gains model
# carries it in all five worlds: the four reform
# worlds compute it, and the baseline world, which keeps the step-up in basis
# and so owes nothing, carries a column of zeros so that the five can be
# differenced against each other. A scenario without the gains model does not
# carry the column at all, and it is read as zero when absent.
DEEMED_GAINS_COLUMN = "liab_deemed"

# The totals one allocation reports, in the order of the diagnostics array.
# `unallocated_dollars` is tax sitting on estates with nothing to distribute,
# which has no bequest dollars to be matched to.
DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "bequest_dollars",
    "inheritance_dollars",
    "tax_dollars",
    "allocated_dollars",
    "unallocated_dollars",
)


class _LineOfEstates(NamedTuple):
    """Estates waiting to be ranked and matched against inheritances.

    Every field holds one entry per slot on the line, and a stack of lines
    carries the line on a leading axis. ``belongs_on_the_line`` says which
    slots hold an estate that counts; the rest are sorted past the end and
    pick nothing up.
    """

    death_weight: Array
    liability: Array
    distributable_estate: Array
    identifier: Array
    order_within_estate: Array
    belongs_on_the_line: Array


def _estate_tax_line(
    *,
    weight: Array,
    mortality_probability: Array,
    probability_of_unused_spousal_exemption: Array,
    liability_with_unused_spousal_exemption: Array,
    liability_without_unused_spousal_exemption: Array,
    distributable_estate: Array,
    record_id: Array,
) -> _LineOfEstates:
    """The estate tax's line: two entries per decedent, one per exemption answer.

    Each decedent puts two entries on the line, one for each answer to whether
    a deceased spouse's unused exemption is available, carrying the share of
    the death probability that belongs to it. An estate owing nothing is left
    off the line, because the line is about which estates pay.
    """
    death_weight = jnp.asarray(weight, dtype=jnp.float64) * jnp.asarray(
        mortality_probability, dtype=jnp.float64
    )
    with_carryover = jnp.asarray(
        probability_of_unused_spousal_exemption, dtype=jnp.float64
    )
    branch_death_weight = jnp.concatenate(
        [death_weight * with_carryover, death_weight * (1.0 - with_carryover)],
        axis=-1,
    )
    branch_liability = jnp.concatenate(
        [
            jnp.asarray(liability_with_unused_spousal_exemption, dtype=jnp.float64),
            jnp.asarray(liability_without_unused_spousal_exemption, dtype=jnp.float64),
        ],
        axis=-1,
    )
    estate = jnp.asarray(distributable_estate, dtype=jnp.float64)
    identifier = jnp.asarray(record_id)
    # The entry for an available spousal exemption sorts first between the two
    # entries of one estate.
    order_within_estate = jnp.concatenate(
        [
            jnp.zeros(identifier.shape, dtype=jnp.int64),
            jnp.ones(identifier.shape, dtype=jnp.int64),
        ],
        axis=-1,
    )
    return _LineOfEstates(
        death_weight=branch_death_weight,
        liability=branch_liability,
        distributable_estate=jnp.concatenate([estate, estate], axis=-1),
        identifier=jnp.concatenate([identifier, identifier], axis=-1),
        order_within_estate=order_within_estate,
        belongs_on_the_line=(branch_death_weight > 0.0) & (branch_liability > 0.0),
    )


def _deemed_gains_line(
    *,
    weight: Array,
    mortality_probability: Array,
    distributable_estate: Array,
    expected_deemed_gains_tax: Array,
    record_id: Array,
    padded_to_the_estate_line: bool,
) -> _LineOfEstates:
    """The deemed-gains line: one entry per decedent.

    ``expected_deemed_gains_tax`` is already an expectation over dying this
    year, so dividing by the chance of dying recovers what the estate would owe.
    ``mortality_probability`` is therefore the chance of dying the expected tax
    was formed with, the frame's ``m_household``.
    Every dying estate is on the line, including one owing nothing. An estate
    with no unrealized gain can be any size, so leaving it off would move its
    bequest dollars onto the wrong heirs. This is unlike the estate tax's line,
    which is about which estates pay.

    ``padded_to_the_estate_line`` adds a second, empty half so this line is the
    same length as the estate tax's and the two can be stacked and sorted
    together. The added slots belong on no line, so they sort past the end and
    change no answer, in the same way as an estate that does not count.
    """
    weight = jnp.asarray(weight, dtype=jnp.float64)
    mortality_probability = jnp.asarray(mortality_probability, dtype=jnp.float64)
    expected_tax = jnp.asarray(expected_deemed_gains_tax, dtype=jnp.float64)
    death_weight = weight * mortality_probability
    dies = death_weight > 0.0
    liability = jnp.where(
        dies, expected_tax / jnp.where(dies, mortality_probability, 1.0), 0.0
    )
    estate = jnp.asarray(distributable_estate, dtype=jnp.float64)
    identifier = jnp.asarray(record_id)
    line = _LineOfEstates(
        death_weight=death_weight,
        liability=liability,
        distributable_estate=estate,
        identifier=identifier,
        order_within_estate=jnp.zeros(identifier.shape, dtype=jnp.int64),
        belongs_on_the_line=dies,
    )
    if not padded_to_the_estate_line:
        return line
    return _LineOfEstates(
        death_weight=jnp.concatenate([death_weight, jnp.zeros_like(death_weight)], -1),
        liability=jnp.concatenate([liability, jnp.zeros_like(liability)], axis=-1),
        distributable_estate=jnp.concatenate([estate, estate], axis=-1),
        identifier=jnp.concatenate([identifier, identifier], axis=-1),
        order_within_estate=jnp.concatenate(
            [line.order_within_estate, line.order_within_estate], axis=-1
        ),
        belongs_on_the_line=jnp.concatenate(
            [line.belongs_on_the_line, jnp.zeros_like(line.belongs_on_the_line)],
            axis=-1,
        ),
    )


def allocate_estate_tax(
    *,
    weight: Array,
    mortality_probability: Array,
    probability_of_unused_spousal_exemption: Array,
    liability_with_unused_spousal_exemption: Array,
    liability_without_unused_spousal_exemption: Array,
    distributable_estate: Array,
    probability_of_inheriting: Array,
    inheritance: Array,
    record_id: Array,
) -> tuple[Array, Array]:
    """Spread one world's expected estate tax over the heirs who bear it.

    Every argument is one value per household. ``weight`` is the population
    weight, ``mortality_probability`` the chance the household dies this year,
    and the two liabilities are what the estate would owe with and without a
    deceased spouse's unused exemption available, with
    ``probability_of_unused_spousal_exemption`` the chance of the first.
    ``distributable_estate`` is what the estate has to leave. ``record_id``
    breaks ties between estates of exactly equal size, so that the answer does
    not depend on the order households happen to sit in.

    Two things come back. The tax each household bears if it inherits, one
    dollar amount per household in the order the households were given, which
    multiplied by the probability of inheriting is the expected amount; and
    the diagnostics array named by `DIAGNOSTIC_NAMES`.

    The model calls `allocate_taxes_at_death`; this one-line form is what the
    tests compare the batched match against.
    """
    line = _estate_tax_line(
        weight=weight,
        mortality_probability=mortality_probability,
        probability_of_unused_spousal_exemption=(
            probability_of_unused_spousal_exemption
        ),
        liability_with_unused_spousal_exemption=(
            liability_with_unused_spousal_exemption
        ),
        liability_without_unused_spousal_exemption=(
            liability_without_unused_spousal_exemption
        ),
        distributable_estate=distributable_estate,
        record_id=record_id,
    )
    return _one_line(
        line,
        weight=weight,
        probability_of_inheriting=probability_of_inheriting,
        inheritance=inheritance,
        identifier_of_household=record_id,
    )


def allocate_deemed_gains_tax(
    *,
    weight: Array,
    mortality_probability: Array,
    distributable_estate: Array,
    expected_deemed_gains_tax: Array,
    probability_of_inheriting: Array,
    inheritance: Array,
    record_id: Array,
) -> tuple[Array, Array]:
    """Spread the income tax on gains deemed realized at death over the heirs.

    Returns the same pair of arrays as `allocate_estate_tax`, for the other
    tax charged at death. ``mortality_probability`` is the chance of
    dying ``expected_deemed_gains_tax`` was formed with. `_deemed_gains_line` says how this line differs.
    Like `allocate_estate_tax`, it is used by the tests.
    """
    line = _deemed_gains_line(
        weight=weight,
        mortality_probability=mortality_probability,
        distributable_estate=distributable_estate,
        expected_deemed_gains_tax=expected_deemed_gains_tax,
        record_id=record_id,
        padded_to_the_estate_line=False,
    )
    return _one_line(
        line,
        weight=weight,
        probability_of_inheriting=probability_of_inheriting,
        inheritance=inheritance,
        identifier_of_household=record_id,
    )


def allocate_taxes_at_death(
    *,
    weight: Array,
    record_id: Array,
    mortality_probability: Array,
    probability_of_unused_spousal_exemption: Array,
    liability_with_unused_spousal_exemption: Array,
    liability_without_unused_spousal_exemption: Array,
    distributable_estate: Array,
    expected_deemed_gains_tax: Array | None,
    deemed_gains_mortality_probability: Array | None,
    probability_of_inheriting: Array,
    inheritance: Array,
) -> tuple[Array, Array, Array]:
    """Both taxes charged at death, in every world, in one batched match.

    Each per-world argument carries the worlds on its first axis and the
    households on its second; ``weight``, ``record_id`` and the two heir
    columns are the same in every world and carry households only.

    Three arrays come back. The estate tax each household bears if it inherits
    and the tax on gains deemed realized at death it bears, each one row per
    world, and the diagnostics shaped [world, tax charged at death, total].
    Position 0 of the middle axis is the estate tax's own line and position 1
    the deemed-gains line, and the totals are named by `DIAGNOSTIC_NAMES`.

    Each line uses the chance of dying its tax was formed with. The estate
    tax's line uses ``mortality_probability``, the estate calculation's
    ``estate_m``. The deemed-gains line uses
    ``deemed_gains_mortality_probability``, the capital-gains model's
    ``m_household``. Both carry the same cap on expected deaths in clusters of
    replicated donor records; they differ for a joint return on which the
    second spouse's chance of dying is recorded as zero.

    ``expected_deemed_gains_tax`` and ``deemed_gains_mortality_probability``
    are None for a scenario that keeps the step-up in basis at death, and every
    deemed-gains answer is then zero.
    """
    n_worlds = jnp.asarray(distributable_estate).shape[0]
    if expected_deemed_gains_tax is None:
        no_deemed_gains = jnp.zeros_like(
            jnp.asarray(distributable_estate, dtype=jnp.float64)
        )
        deemed_gains_tax = no_deemed_gains
        deemed_gains_mortality = no_deemed_gains
    else:
        deemed_gains_tax = expected_deemed_gains_tax
        deemed_gains_mortality = deemed_gains_mortality_probability
    estate_line = _estate_tax_line(
        weight=weight,
        mortality_probability=mortality_probability,
        probability_of_unused_spousal_exemption=(
            probability_of_unused_spousal_exemption
        ),
        liability_with_unused_spousal_exemption=(
            liability_with_unused_spousal_exemption
        ),
        liability_without_unused_spousal_exemption=(
            liability_without_unused_spousal_exemption
        ),
        distributable_estate=distributable_estate,
        record_id=jnp.broadcast_to(
            jnp.asarray(record_id), jnp.asarray(distributable_estate).shape
        ),
    )
    deemed_line = _deemed_gains_line(
        weight=weight,
        mortality_probability=deemed_gains_mortality,
        distributable_estate=distributable_estate,
        expected_deemed_gains_tax=deemed_gains_tax,
        record_id=jnp.broadcast_to(
            jnp.asarray(record_id), jnp.asarray(distributable_estate).shape
        ),
        padded_to_the_estate_line=True,
    )
    # The two lines of each world sit next to each other, so the answers come
    # apart again into a world axis and a tax-charged-at-death axis.
    both = jax.tree.map(
        lambda estate, deemed: jnp.concatenate(
            [estate[:, None, :], deemed[:, None, :]], axis=1
        ).reshape(2 * n_worlds, -1),
        estate_line,
        deemed_line,
    )
    per_heir, diagnostics = _match_lines_to_heirs(
        both,
        weight=weight,
        probability_of_inheriting=probability_of_inheriting,
        inheritance=inheritance,
        identifier_of_household=record_id,
    )
    per_heir = per_heir.reshape(n_worlds, 2, -1)
    return (
        per_heir[:, 0],
        per_heir[:, 1],
        diagnostics.reshape(n_worlds, 2, len(DIAGNOSTIC_NAMES)),
    )


def _one_line(
    line: _LineOfEstates,
    *,
    weight: Array,
    probability_of_inheriting: Array,
    inheritance: Array,
    identifier_of_household: Array,
) -> tuple[Array, Array]:
    """Match a single line, by running the batched match over a stack of one."""
    stacked = jax.tree.map(lambda values: values[None, :], line)
    per_heir, diagnostics = _match_lines_to_heirs(
        stacked,
        weight=weight,
        probability_of_inheriting=probability_of_inheriting,
        inheritance=inheritance,
        identifier_of_household=identifier_of_household,
    )
    return per_heir[0], diagnostics[0]


def _match_lines_to_heirs(
    line: _LineOfEstates,
    *,
    weight: Array,
    probability_of_inheriting: Array,
    inheritance: Array,
    identifier_of_household: Array,
) -> tuple[Array, Array]:
    """The matching itself: the biggest estates against the biggest inheritances.

    Every field of ``line`` carries the lines on its first axis, and every line
    is ranked and matched in the same operations. What comes back is one row of
    per-household tax and one row of diagnostics per line.
    """
    probability_of_inheriting = jnp.asarray(
        probability_of_inheriting, dtype=jnp.float64
    )
    inheritance = jnp.asarray(inheritance, dtype=jnp.float64)
    weight = jnp.asarray(weight, dtype=jnp.float64)

    # Tax on an estate with nothing to distribute has no bequest dollars to be
    # matched against, so it stays unallocated and is reported.
    nothing_to_distribute = line.distributable_estate <= 0.0
    unallocated = jnp.sum(
        jnp.where(
            line.belongs_on_the_line & nothing_to_distribute,
            line.death_weight * line.liability,
            0.0,
        ),
        axis=-1,
    )
    on_the_line = line.belongs_on_the_line & ~nothing_to_distribute

    bequest_dollars = jnp.where(
        on_the_line, line.death_weight * line.distributable_estate, 0.0
    )
    tax_dollars = jnp.where(on_the_line, line.death_weight * line.liability, 0.0)

    # Largest estate first, ties broken by household and then by which spousal
    # exemption branch it is, with every entry that is not on the line pushed
    # past the end so the cumulative curves are flat across them; one sort
    # serves every line at once.
    order = jnp.lexsort(
        (
            line.order_within_estate,
            line.identifier,
            -jnp.where(on_the_line, line.distributable_estate, 0.0),
            (~on_the_line).astype(jnp.int8),
        ),
        axis=-1,
    )
    bequest_in_order = jnp.take_along_axis(bequest_dollars, order, axis=-1)
    tax_in_order = jnp.take_along_axis(tax_dollars, order, axis=-1)
    nothing_yet = jnp.zeros((bequest_in_order.shape[0], 1), dtype=jnp.float64)
    cumulative_bequests = jnp.concatenate(
        [nothing_yet, jnp.cumsum(bequest_in_order, axis=-1)], axis=-1
    )
    cumulative_tax = jnp.concatenate(
        [nothing_yet, jnp.cumsum(tax_in_order, axis=-1)], axis=-1
    )

    # Largest inheritance first, ties broken by household. An heir's dollars are
    # what it expects to receive across the whole population it stands for. This
    # ranking does not depend on which tax is being spread, so it is worked out
    # once and every line above reads it.
    heir_order = jnp.lexsort((identifier_of_household, -inheritance))
    inheritance_dollars = (weight * probability_of_inheriting * inheritance)[heir_order]
    running_inheritance = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(inheritance_dollars)]
    )
    heir_position = jnp.minimum(
        running_inheritance[None, :], cumulative_bequests[:, -1:]
    )
    tax_at_position = jax.vmap(_tax_at)(
        heir_position,
        cumulative_bequests,
        cumulative_tax,
        bequest_in_order,
        tax_in_order,
    )
    tax_dollars_per_heir = jnp.diff(tax_at_position, axis=-1)

    heir_share = (weight * probability_of_inheriting)[heir_order]
    per_heir_in_order = jnp.where(
        inheritance_dollars[None, :] > 0.0,
        tax_dollars_per_heir
        / jnp.where(heir_share > 0.0, heir_share, 1.0)[None, :],
        0.0,
    )
    per_heir = (
        jnp.zeros(per_heir_in_order.shape, dtype=jnp.float64)
        .at[:, heir_order]
        .set(per_heir_in_order)
    )

    inheritance_total = jnp.sum(weight * probability_of_inheriting * inheritance)
    diagnostics = jnp.stack(
        [
            jnp.sum(bequest_dollars, axis=-1),
            jnp.broadcast_to(inheritance_total, unallocated.shape),
            cumulative_tax[:, -1],
            jnp.sum(tax_dollars_per_heir, axis=-1),
            unallocated,
        ],
        axis=-1,
    )
    return per_heir, diagnostics


def _tax_at(
    position: Array,
    cumulative_bequests: Array,
    cumulative_tax: Array,
    bequest_in_order: Array,
    tax_in_order: Array,
) -> Array:
    """Cumulative tax as a function of cumulative bequest dollars.

    Within one estate the curve is a straight line whose slope is that estate's
    average tax per dollar of bequest, so reading the curve at a position means
    finding which estate that dollar belongs to and adding the part of that
    estate's tax the dollars before the position account for.

    This is written out instead of using an interpolation routine because the
    array carries repeated positions (every entry that is not on the line sits
    at the same place at the end), and an interpolation routine handed a
    zero-width interval divides zero by zero.
    """
    last_estate = cumulative_bequests.shape[0] - 2
    # Searching from the right puts a position sitting exactly on a boundary at
    # the end of the run of estates that share it, where the curve has already
    # risen by all of their tax.
    estate = jnp.clip(
        jnp.searchsorted(cumulative_bequests, position, side="right") - 1,
        0,
        last_estate,
    )
    dollars_into_the_estate = position - cumulative_bequests[estate]
    bequest_of_estate = bequest_in_order[estate]
    rate = jnp.where(
        bequest_of_estate > 0.0,
        tax_in_order[estate]
        / jnp.where(bequest_of_estate > 0.0, bequest_of_estate, 1.0),
        0.0,
    )
    return cumulative_tax[estate] + rate * dollars_into_the_estate


__all__ = [
    "DEEMED_GAINS_COLUMN",
    "DIAGNOSTIC_NAMES",
    "ESTATE_RESULT_COLUMNS",
    "allocate_taxes_at_death",
]
