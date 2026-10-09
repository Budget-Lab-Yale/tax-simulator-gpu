"""Households added up into the age cells the capital-gains model runs on.

The rest of the model does not know about households.  It works on 63 age
cells, one for each age from 18 to 80, and every year it reads what those
cells hold: the stock of unrealized gain, the gains realized under baseline
law, the mortality the dollars face, the chance a decedent leaves the gain to
charity, and the marginal tax rates the cell's dollars pay.  This module reads the household
microdata and produces those numbers.  A household appears on the way into the
capital-gains model only here; after this the model works on age cells, until
the applier spreads the answers back out.

Six numbers come out of a year's households, all of them weighted sums within
an age cell:

1. **The stock of unrealized gain** — market value less cost basis over the
   five tracked asset classes, each class floored at zero.
2. **Realizations** — long-term gains realized under baseline law, counting
   only positive amounts, so that turnover can never be negative.
3. **Turnover** — realizations over the stock.  That ratio is what the data
   measures, and because selling is pro-rata out of a well-mixed portfolio it
   is also the share of the cell's asset value sold in the year, which is the
   quantity the selling choice picks.  A cell that holds gains but sells
   nothing takes the whole population's turnover instead, because otherwise
   young heirs receiving inherited gains under carryover would hold them
   forever.
4. **Mortality, two ways** — the plain rate per household and the rate
   weighted by holdings.  The recurrence uses the weighted one: within an age
   cell the households most likely to die hold the least, so a per-person rate
   overstates the flow of gains at death by a factor of about 2.7.
5. **The charitable bequest share** — the chance a dollar of a decedent's gain
   goes to charity and is never taxed, from a two-part logit on the size of the
   estate.
6. **The marginal tax rate on realized gains**, averaged within the cell over
   the dollars it applies to, which is not the same as averaging over
   households.

The first five come from the population data alone and are worked out once
per run, in a compiled program of their own
(``many_scenarios._population_cells_program``).  The sixth cannot be, because
those rates only exist inside the compiled year loop where the tax calculators
run, so the averaging of the rates is done there as well: the compiled program adds each household's rate
into its age cell and hands back the cell's weighted sums, a few hundred
numbers a year instead of one per household ("Averaging the rates into cells"
below), and the host divides the sums it is given.

A household falls in the cell for the filer's age, or for the older spouse's
age on a joint return, with everyone 80 and over pooled into the top cell.

The cell's gain by asset class comes out twice, once whole and once net of the
two exclusions a decedent's gain gets (the §121 exclusion on a primary home
and the per-decedent exclusion on gains at death).  The mix of death
regimes needs both, because a dollar forgiven by a stepped-up basis is measured
against the whole gain while a dollar routed to an heir or taxed at death is
measured against what the exclusions leave.  Both nettings are done household
by household before anything is added up, which is where the exclusion's floor
at zero belongs: a cell whose average basis sits above the cap would otherwise
come out at no gain even though the households with the least basis still have
one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import jax.numpy as jnp
import yaml
from jax.tree_util import register_dataclass

from taxmodel.behavior import balance_sheet as bs
from taxmodel.frame.steps_jax import FILING_JOINT

from .applier import (
    DEEMED_ASSET_CLASSES,
    ESTATE_GROSS_COLUMN,
    TAXABLE_AT_DEATH_REGIME_CODES,
    VALUE_COLUMN_PREFIX,
    WIDOWHOOD_COLUMN,
    CharitableBequestSettings,
    cell_row_for_household,
    charitable_bequest_probability,
    gain_after_the_home_sale_exclusion,
    gain_by_class,
    taxable_shares_if_widowed_or_not,
)
from .stock_inputs import KG_AGE_MIN, KG_N_AGES, BaselineCells

Array = jnp.ndarray
Records = Mapping[str, Array]

# A marginal tax rate on realized gains lies between zero and one.  A record
# measuring outside that range is dropped from the cell average, numerator and
# denominator together, rather than averaged in: the rate is a one-dollar
# derivative of a statute with discrete steps in it, so the dollar that carries
# a household across a credit phase-out block measures at something that is not
# a rate.
MARGINAL_RATE_RANGE = (0.0, 1.0)

# How far past either end a measured rate can sit and still count as sitting
# on the end itself.  The measured rate is two tax bills differenced by one
# dollar, so a household whose true rate is exactly zero can come out of that
# subtraction a few ten-billionths below it.  Without the band, that rounding
# decides whether the household's dollars are counted in its age cell at all.
# The band is far above double-precision rounding on the largest tax bill in
# the file, about $10 billion, and far below any rate the statute produces.
MARGINAL_RATE_BOUNDARY_NOISE = 1e-5


# ---------------------------------------------------------------------------
# Reading the charitable bequest coefficients
# ---------------------------------------------------------------------------
# The logit they feed, ``charitable_bequest_probability``, lives in the
# applier, which runs it per household; the pre-pass runs it per cell.


# The gains model's settings file, and the names the charitable bequest
# coefficients go by in it. Every reader of that file in the tree starts from
# this path.
DEFAULT_SETTINGS_FILE = Path("config/calibrations/kg/settings.yaml")
_CHARITY_SETTING_NAMES = (
    ("extensive_intercept", "char_extensive_intercept"),
    ("extensive_log_estate_slope", "char_extensive_ln_slope"),
    ("intensive_intercept", "char_intensive_intercept"),
    ("intensive_log_estate_slope", "char_intensive_ln_slope"),
    ("base_year", "char_base_year"),
)


def read_charitable_bequest_settings(
    path: Path | str = DEFAULT_SETTINGS_FILE,
) -> CharitableBequestSettings:
    """Read the charitable bequest coefficients out of the settings file."""

    body = yaml.safe_load(Path(path).read_text()) or {}
    values = {}
    for field_name, setting_name in _CHARITY_SETTING_NAMES:
        entry = body.get(setting_name)
        if entry is None:
            raise ValueError(
                f"{path} has no entry for {setting_name}, which the capital-gains "
                "pre-pass needs to work out the chance a decedent leaves "
                "appreciated property to charity"
            )
        values[field_name] = entry["value"] if isinstance(entry, Mapping) else entry
    values["base_year"] = int(values["base_year"])
    return CharitableBequestSettings(**values)


# ---------------------------------------------------------------------------
# The gain a decedent still has at stake, after the exclusions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeathGainExclusions:
    """What the law lets a decedent's gain escape, for one year.

    ``section_121`` is the exclusion on the gain from selling a primary home,
    ``per_decedent`` the exclusion on a decedent's gain generally, and
    ``per_decedent_if_widowed`` the amount an unmarried filer would have if
    they were a surviving spouse at death.  Each may be one number for
    everyone or one number per household, since the law states them by filing
    status.

    ``regime_code_by_class`` says which treatment each of the five tracked
    asset classes is under, in :data:`~.applier.DEEMED_ASSET_CLASSES` order:
    0 for a stepped-up basis, 1 for carryover basis, 2 for deemed realization
    at death.  It decides which classes the per-decedent exclusion is spread
    across, since a class whose gain is forgiven outright has nothing to
    exclude.
    """

    section_121: Array | float = 0.0
    per_decedent: Array | float = 0.0
    per_decedent_if_widowed: Array | float = 0.0
    regime_code_by_class: tuple[int, ...] = (0, 0, 0, 0, 0)


def _gain_at_stake_at_death_by_class(
    records: Records,
    exclusions: DeathGainExclusions,
) -> Array:
    """Each household's gain still at stake at death, by asset class.

    Two exclusions come off before anything else happens to a decedent's
    gain.  The §121 exclusion takes the first quarter of a million dollars (a
    half million for a couple) of the gain on a primary home.  The per-decedent
    exclusion then takes its first dollars pro-rata across the classes whose
    treatment still taxes the gain (carryover basis or deemed realization),
    since a class that gets a stepped-up basis has nothing left to exclude.

    Both are modeled as a step-up in basis up to the cap, so the gain below the
    cap never enters either the stock routed to heirs or the stock taxed at
    death.  An unmarried filer has two amounts, their own and the larger one a
    surviving spouse could claim, and the two are blended by the chance of
    being widowed at death.

    Shaped [household, asset class].
    """

    stakes = gain_after_the_home_sale_exclusion(
        gain_by_class(records), exclusions.section_121
    )

    regime_code = jnp.asarray(exclusions.regime_code_by_class)
    share_if_not_widowed, share_if_widowed = taxable_shares_if_widowed_or_not(
        stakes,
        regime_code,
        exclusions.per_decedent,
        exclusions.per_decedent_if_widowed,
        records["filing_status"],
    )
    widowhood = jnp.nan_to_num(
        jnp.asarray(
            records.get(WIDOWHOOD_COLUMN, jnp.zeros(stakes.shape[0])),
            dtype=jnp.float64,
        )
    )
    surviving = widowhood * share_if_widowed + (1.0 - widowhood) * share_if_not_widowed

    # Only the classes the exclusion is spread across are cut back by it.  A
    # class whose gain is forgiven at death keeps its whole stake, and the
    # forgiven share is measured against that.
    eligible = jnp.isin(regime_code, jnp.asarray(TAXABLE_AT_DEATH_REGIME_CODES))
    return jnp.where(eligible, stakes * surviving[:, None], stakes)


# ---------------------------------------------------------------------------
# Allocation of a decedent's gains
# ---------------------------------------------------------------------------


# The distribution of heir ages, weighted by dollars inherited, measured off the
# 2022 Survey of Consumer Finances.  Rebuilt when the survey vintage changes.
DEFAULT_HEIR_DISTRIBUTION_FILE = Path("resources/heir_distribution_scf2022.csv")


def read_heir_age_distribution(n_ages: int = KG_N_AGES) -> Array:
    """The share of inherited dollars going to each age on the cell grid.

    ``stock_inputs.heir_operator`` turns this into the matrix that routes a
    decedent's gains to the ages of the heirs.  The file's ages must be exactly
    the model's grid, in order, because the shares are read positionally.
    """

    ages: list[int] = []
    shares: list[float] = []
    path = DEFAULT_HEIR_DISTRIBUTION_FILE
    text = path.read_text().splitlines()
    header = [name.strip() for name in text[0].split(",")]
    if header[:2] != ["age", "share"]:
        raise ValueError(
            f"{path} should have an age column and a share column, not {header}"
        )
    for line in text[1:]:
        if not line.strip():
            continue
        age, share = line.split(",")[:2]
        ages.append(int(age))
        shares.append(float(share))

    expected = list(range(KG_AGE_MIN, KG_AGE_MIN + n_ages))
    if ages != expected:
        raise ValueError(
            f"{path} covers ages {ages[0]} to {ages[-1]}, but the stock of gains "
            f"is tracked over ages {expected[0]} to {expected[-1]}"
        )
    if any(share < 0 for share in shares):
        raise ValueError(f"{path} has a negative share of inherited dollars in it")
    return jnp.asarray(shares, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# One year's age cells
# ---------------------------------------------------------------------------


def _sum_by_cell(row: Array, amount: Array, n_ages: int) -> Array:
    """Add each household's amount into its age cell."""

    return jnp.zeros(n_ages, dtype=jnp.float64).at[row].add(amount)


@dataclass(frozen=True)
class PopulationAgeCells:
    """The part of a year's age cells that no scenario can change.

    Every field is one number per age cell, except ``gain_by_asset_class``,
    which carries a second axis over :data:`~.applier.DEEMED_ASSET_CLASSES`.
    Ten scenarios run on one population, so these are worked out once for the
    run and handed to every scenario's solve. :class:`ScenarioAgeCells` holds
    the fields the scenario's law moves.

    ``mortality_gain_weighted`` is the numerator of the holdings-weighted
    mortality rate, not the rate itself: the sum over the cell's households of
    mortality times holdings.  The recurrence divides it by the cell's stock
    when it needs the rate, and applies the fallback for a cell holding nothing
    at the same point.

    ``gain_by_asset_class`` is the cell's whole gain in each class.

    ``tracked_asset_value`` is the market value of everything the cell holds
    in the five tracked classes, gains and basis together.  The experimental
    embedded-gain-ratio channel divides the cell's stock of gains by it; the
    ratio says how much taxable gain a dollar of sale brings into income.
    """

    stock: Array
    realizations: Array
    turnover: Array
    mortality_headcount: Array
    mortality_gain_weighted: Array
    charitable_bequest_share: Array
    gain_by_asset_class: Array
    tracked_asset_value: Array


@dataclass(frozen=True)
class ScenarioAgeCells:
    """The two fields of a year's age cells that the scenario's law decides.

    ``gain_at_stake_by_asset_class`` is the cell's gain in each class after
    the §121 exclusion on a primary home and the per-decedent exclusion on
    gains at death.  The mix of death regimes reads it together with
    :attr:`PopulationAgeCells.gain_by_asset_class`.

    ``corporate_exposed_value`` is the market value of corporate equity
    standing behind the cell's gains.  The corporate channel marks that value
    down when the corporate rate changes, and the debit comes off the cell's
    stock of gain dollar for dollar, because basis does not move.
    """

    gain_at_stake_by_asset_class: Array
    corporate_exposed_value: Array


def _cell_sums_by_class(
    row: Array, weight: Array, per_household: Array, n_ages: int
) -> Array:
    """One cell total for each of the five tracked asset classes."""
    return jnp.stack(
        [
            _sum_by_cell(row, weight * per_household[:, i], n_ages)
            for i in range(len(DEEMED_ASSET_CLASSES))
        ],
        axis=-1,
    )


def population_age_cells(
    records: Records,
    *,
    deflator_to_base_year: Array | float,
    charity: CharitableBequestSettings,
    n_ages: int = KG_N_AGES,
) -> PopulationAgeCells:
    """Add one year's households up into the age-cell facts no reform changes.

    The households are the whole year's population, weighted.  Nobody is
    excluded: gains at death never enter ``kg_lt``, so these are the numbers a
    living taxpayer reports and no decedents need taking out.  Nothing here
    reads the law.
    """

    row = cell_row_for_household(records, n_ages)
    weight = jnp.asarray(records["weight"], dtype=jnp.float64)
    mortality = jnp.asarray(records["m_household"], dtype=jnp.float64)

    gain_by_asset_class = gain_by_class(records)
    gain_stock = jnp.sum(gain_by_asset_class, axis=-1)

    # The market value standing behind those gains, summed over the same five
    # classes.  A missing value reads as holding nothing of the class, the
    # same reading the corporate-exposure sum makes.
    asset_value = jnp.zeros_like(gain_stock)
    for name in DEEMED_ASSET_CLASSES:
        asset_value = asset_value + jnp.nan_to_num(
            jnp.asarray(records[VALUE_COLUMN_PREFIX + name], dtype=jnp.float64)
        )

    # Realized losses do not count, because a household with a realized loss is
    # not selling an appreciated asset and letting its loss in would make a
    # cell's turnover negative.
    realized = jnp.maximum(jnp.asarray(records["kg_lt"], dtype=jnp.float64), 0.0)

    charity_share = charitable_bequest_probability(
        jnp.asarray(records[ESTATE_GROSS_COLUMN], dtype=jnp.float64),
        deflator_to_base_year,
        charity,
    )

    stock = _sum_by_cell(row, weight * gain_stock, n_ages)
    realizations = _sum_by_cell(row, weight * realized, n_ages)
    households = _sum_by_cell(row, weight, n_ages)
    deaths = _sum_by_cell(row, weight * mortality, n_ages)
    mortality_gain_weighted = _sum_by_cell(row, weight * mortality * gain_stock, n_ages)
    # The charity share is averaged over the dollars of gain that die, because
    # what it decides is how much of the gain at death escapes tax.
    charity_numerator = _sum_by_cell(
        row, weight * mortality * gain_stock * charity_share, n_ages
    )

    return PopulationAgeCells(
        stock=stock,
        realizations=realizations,
        turnover=_turnover(stock, realizations),
        mortality_headcount=bs.divide_where_positive(deaths, households),
        mortality_gain_weighted=mortality_gain_weighted,
        charitable_bequest_share=bs.divide_where_positive(
            charity_numerator, mortality_gain_weighted
        ),
        gain_by_asset_class=_cell_sums_by_class(row, weight, gain_by_asset_class, n_ages),
        tracked_asset_value=_sum_by_cell(row, weight * asset_value, n_ages),
    )


def scenario_age_cells(
    records: Records,
    *,
    exclusions: DeathGainExclusions,
    corporate_asset_exposure: Mapping[str, float] | Array | None = None,
    n_ages: int = KG_N_AGES,
) -> ScenarioAgeCells:
    """The two age-cell fields one scenario's law decides, for one year.

    ``exclusions`` is the year's law on what a decedent's gain escapes.

    ``corporate_asset_exposure`` says how much of each asset class is corporate
    equity.  Only the five classes the gains model tracks count, because a
    household's retirement balances are not followed by the gains model, so a
    markdown of their value never reaches the stock of gain.  Left out, the
    corporate exposure comes back as zero, as a run with no corporate channel
    needs.  It may be the mapping the corporate channel reads, keyed
    ``value.<class>``, or the same five shares already laid out as an array in
    :data:`~.applier.DEEMED_ASSET_CLASSES` order.  The array form lets many
    scenarios' exposures enter one compiled program as data.
    """

    row = cell_row_for_household(records, n_ages)
    weight = jnp.asarray(records["weight"], dtype=jnp.float64)

    at_stake_by_class = _gain_at_stake_at_death_by_class(records, exclusions)

    if corporate_asset_exposure is None:
        shares: list[Array | float] = [0.0] * len(DEEMED_ASSET_CLASSES)
    elif isinstance(corporate_asset_exposure, Mapping):
        shares = [
            corporate_asset_exposure.get(f"value.{name}", 0.0)
            for name in DEEMED_ASSET_CLASSES
        ]
    else:
        shares = [corporate_asset_exposure[i] for i in range(len(DEEMED_ASSET_CLASSES))]

    exposed = jnp.zeros(at_stake_by_class.shape[0], dtype=jnp.float64)
    for position, name in enumerate(DEEMED_ASSET_CLASSES):
        share = shares[position]
        # A share the caller left at a plain zero contributes nothing, so
        # skipping it leaves the arithmetic of a single-scenario run unchanged.
        # A share whose value is not known until the compiled program runs is
        # always added.
        if isinstance(share, float) and share == 0.0:
            continue
        value = jnp.nan_to_num(jnp.asarray(records[f"value.{name}"], dtype=jnp.float64))
        exposed = exposed + share * value

    return ScenarioAgeCells(
        gain_at_stake_by_asset_class=_cell_sums_by_class(
            row, weight, at_stake_by_class, n_ages
        ),
        corporate_exposed_value=_sum_by_cell(row, weight * exposed, n_ages),
    )


def _turnover(stock: Array, realizations: Array) -> Array:
    """The share of its portfolio a cell sells in a year.

    Measured as realizations over the stock of gains, which is the ratio the
    data pins down; pro-rata selling makes it the share of asset value sold as
    well.

    A cell that holds gains but sells none of them takes the whole population's
    turnover instead of zero.  Without that, gains inherited into a young age
    cell under carryover basis would sit there forever, because the cell has no
    observed selling behaviour of its own.  The population figure is taken over
    the cells that do sell something, so that the cells being filled in do not
    pull it toward zero.
    """

    own_turnover = bs.divide_where_positive(realizations, stock)
    sells_something = realizations > 0.0
    population_turnover = bs.divide_where_positive(
        jnp.sum(jnp.where(sells_something, realizations, 0.0)),
        jnp.sum(jnp.where(sells_something, stock, 0.0)),
    )
    return jnp.where(
        (stock > 0.0) & ~sells_something, population_turnover, own_turnover
    )


# ---------------------------------------------------------------------------
# Averaging the rates into cells, inside the compiled program
# ---------------------------------------------------------------------------

# Every rate the selling choice reads is a cell average of the same shape:
# a sum over the cell's households of weight times dollars times rate, over a
# sum of weight times dollars.  Both sums are formed inside the compiled year
# loop, where the rates are measured, and only they come back to the host.
# What the averaging needs to know about a household is only its cell and its
# dollars, and those come from the population data, so they are built once on
# the host and handed to the compiled program as data.


def _measures_as_a_rate(marginal_rate: Array) -> Array:
    """Which households' measured rate on gains can be read as a rate at all.

    The comparison at each end allows `MARGINAL_RATE_BOUNDARY_NOISE` of slack,
    so a household whose true rate sits exactly on the boundary is counted
    however the compiled arithmetic happened to round its one-dollar tax
    difference.  See the constant for the padding bug this closes.
    """

    low, high = MARGINAL_RATE_RANGE
    rate = jnp.asarray(marginal_rate, dtype=jnp.float64)
    return (
        jnp.isfinite(rate)
        & (rate >= low - MARGINAL_RATE_BOUNDARY_NOISE)
        & (rate <= high + MARGINAL_RATE_BOUNDARY_NOISE)
    )


@dataclass(frozen=True)
class CellWeights:
    """The dollars each household brings to its age cell's rate averages.

    A cell's rate is an average over dollars, not over households, so the only
    facts the averaging needs about a household are which cell it falls in and
    how many dollars of each kind it carries: the long-term gains it realizes
    and the unrealized gain it holds, each already multiplied by the
    household's population weight.

    All three come from the population block and the year's data alone (nothing
    simulated, and no law), so they are the same for every scenario in a batch
    and the same for both executions of a gains scenario.  They are built once
    on the host and handed to the compiled program as data.

    Stacked on a leading year axis on the way in; the year loop slices out the
    year it is on.
    """

    cell_row: Array
    weighted_realizations: Array
    weighted_stock: Array


register_dataclass(
    CellWeights,
    data_fields=[field.name for field in fields(CellWeights)],
    meta_fields=[],
)


def cell_weights_from_records(
    records: Records, *, n_ages: int = KG_N_AGES
) -> CellWeights:
    """One year's households, reduced to what the rate averaging weights by.

    Only positive realizations count, the same as everywhere else in the
    pre-pass: a household with a realized loss is not selling an appreciated
    asset.
    """

    weight = jnp.asarray(records["weight"], dtype=jnp.float64)
    realized = jnp.maximum(jnp.asarray(records["kg_lt"], dtype=jnp.float64), 0.0)
    gain_stock = jnp.sum(gain_by_class(records), axis=-1)
    return CellWeights(
        cell_row=cell_row_for_household(records, n_ages).astype(jnp.int32),
        weighted_realizations=weight * realized,
        weighted_stock=weight * gain_stock,
    )


def joint_share_of_top_cell_gain(
    records: Records, *, n_ages: int = KG_N_AGES
) -> Array:
    """The share of the top age cell's unrealized gain held on joint returns.

    One number for the year, weighted by survey weight times the household's
    stock of unrealized gain, which is the weighting the gain-weighted
    mortality uses.  The mortality tail above the top-coded age reads it
    (:func:`~.assemble.mortality_above_the_top_code_for_tax_units`): a joint
    return's gain passes on only when both spouses have died, and the life
    table gives one person's chance of dying.

    A cell holding no gain gets zero, which leaves the tail at the life
    table's own rates.
    """

    weights = cell_weights_from_records(records, n_ages=n_ages)
    on_a_joint_return = jnp.asarray(records["filing_status"]) == FILING_JOINT
    dollars = _sum_by_cell(weights.cell_row, weights.weighted_stock, n_ages)
    joint_dollars = _sum_by_cell(
        weights.cell_row,
        jnp.where(on_a_joint_return, weights.weighted_stock, 0.0),
        n_ages,
    )
    return bs.divide_where_positive(joint_dollars[-1], dollars[-1])


def cell_weights_by_year(
    records_by_year: Sequence[Records], *, n_ages: int = KG_N_AGES
) -> CellWeights:
    """Every run year's cell weights, stacked on a leading year axis."""

    each_year = [
        cell_weights_from_records(records, n_ages=n_ages)
        for records in records_by_year
    ]
    return CellWeights(
        cell_row=jnp.stack([year.cell_row for year in each_year]),
        weighted_realizations=jnp.stack(
            [year.weighted_realizations for year in each_year]
        ),
        weighted_stock=jnp.stack([year.weighted_stock for year in each_year]),
    )


# The four sums, in the order they are named. The compiled program writes them
# under these names and the host reads them back under the same ones, so they
# are spelled out once, here.
MARGINAL_RATE_CELL_SUM_FIELDS: tuple[str, ...] = (
    "rate_times_realizations",
    "realizations",
    "rate_times_holdings",
    "holdings",
)


@dataclass(frozen=True)
class MarginalRateCellSums:
    """A cell's weighted sums for one measured marginal rate on realized gains.

    Four numbers per age cell.  ``rate_times_realizations`` over
    ``realizations`` is the average taken over the dollars the cell sells, and
    ``rate_times_holdings`` over ``holdings`` the average over the dollars it
    holds; :meth:`average` says which of the two a cell gets.

    Both denominators count only the dollars of households whose measured rate
    reads as a rate at all (:data:`MARGINAL_RATE_RANGE`), so a household
    dropped from a numerator is dropped from its denominator with it.
    """

    rate_times_realizations: Array
    realizations: Array
    rate_times_holdings: Array
    holdings: Array

    def average(self) -> Array:
        """The cell's rate: over the dollars it sells, or the ones it holds.

        The average is taken over realized dollars, because the rate is meant
        to be the price faced by the gains that actually get sold, which is
        what an elasticity of realizations is an elasticity with respect to.
        A cell that sells nothing falls back to averaging over holdings, which
        is what happens to young cells holding inherited gains, and a cell
        with neither gets zero.
        """

        by_realizations = bs.divide_where_positive(
            self.rate_times_realizations, self.realizations
        )
        by_holdings = bs.divide_where_positive(self.rate_times_holdings, self.holdings)
        return jnp.where(self.realizations > 0.0, by_realizations, by_holdings)

    def named(self, leg: str) -> dict[str, Array]:
        """The four sums under the names they carry out of the compiled program.

        ``leg`` says which law on which frame the rate was measured under.
        The leg names are defined in ``behavior/gains/builders.py``
        (``REFORM_LEG`` and the others).
        """

        return {
            f"{leg}_{field}": getattr(self, field)
            for field in MARGINAL_RATE_CELL_SUM_FIELDS
        }

    @classmethod
    def from_named(
        cls, leg: str, exports: Mapping[str, Array]
    ) -> MarginalRateCellSums:
        """Read one leg's four sums back out of what the program exported."""

        return cls(
            **{
                field: jnp.asarray(exports[f"{leg}_{field}"])
                for field in MARGINAL_RATE_CELL_SUM_FIELDS
            }
        )


def _dollars_whose_rate_counts(
    weights: CellWeights, marginal_rate: Array
) -> tuple[Array, Array]:
    """The realized and the held dollars a cell's rate average is taken over.

    A household whose measured rate cannot be read as a rate leaves both the
    numerator and the denominator, so its dollars are zeroed out of both here.
    """

    counted = jnp.where(_measures_as_a_rate(marginal_rate), 1.0, 0.0)
    return (
        weights.weighted_realizations * counted,
        weights.weighted_stock * counted,
    )


def marginal_rate_cell_sums(
    weights: CellWeights,
    marginal_rate: Array,
    *,
    n_ages: int = KG_N_AGES,
) -> MarginalRateCellSums:
    """Add one measured rate on realized gains into the age cells.

    Households whose measured rate is not a rate leave both the numerator and
    the denominator.
    """

    row = weights.cell_row
    rate = jnp.nan_to_num(jnp.asarray(marginal_rate, dtype=jnp.float64))
    realizations, holdings = _dollars_whose_rate_counts(weights, marginal_rate)
    return MarginalRateCellSums(
        rate_times_realizations=_sum_by_cell(row, realizations * rate, n_ages),
        realizations=_sum_by_cell(row, realizations, n_ages),
        rate_times_holdings=_sum_by_cell(row, holdings * rate, n_ages),
        holdings=_sum_by_cell(row, holdings, n_ages),
    )


# ---------------------------------------------------------------------------
# The whole pre-pass, every year at once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepassResult:
    """Everything the pre-pass produces, for every year of the run.

    ``cells`` is what the stock recurrence reads, laid out year by age because
    the recurrence scans over years.  The rate matrices are laid out age by
    year, because that is the shape the schedule solver's ``ScheduleInputs``
    takes.  ``gain_by_asset_class`` is [year, age, class] and
    ``corporate_exposed_value`` [year, age].

    The rate matrices are on the stock's own 18-to-80 grid.  The realization
    choice is solved over a longer grid that runs to 119, where the life tables
    reach certain death; ``inputs.extend_to_choice_grid`` stretches these onto
    it, holding each rate at its age-80 value.  The caller does that step, since
    mortality supplies a tail of its own.
    """

    cells: BaselineCells
    marginal_rate_baseline: Array
    marginal_rate_reform: Array
    marginal_rate_reform_law_only: Array
    gain_by_asset_class: Array
    tracked_asset_value: Array
    gain_at_stake_by_asset_class: Array
    corporate_exposed_value: Array
