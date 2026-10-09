"""Report sums taken on the device inside the compiled year.

A year's per-household columns, five worlds' results for every household,
never leave the GPU: thirty years of them would be about 35 GB. The sums each
published table needs are taken here, inside the compiled year, and only the
sums come back to the host. Four families of sums are formed: the population
totals and counts the four totals files report, the heirs' share of the taxes
charged at death, the distribution table, and the marginal-rate table's two
dollar-weighted accumulators.

The scenario loop (`years/loop.py`) calls the year loop, which calls
`years/one_year.one_year_step` once per year; that step computes the worlds
and then calls into here for the sums. Once the years are done,
`reporting/assemble.py` arranges the sums into tables on the host.

The first half of this module picks out and adds up the population totals;
the second half forms the report columns (`ReportColumns`) and the three
report programs that read them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from taxmodel.behavior.kg import applier
from taxmodel.frame import steps_jax
from taxmodel.reporting import (
    corporate_spread,
    distribution,
    distribution_jax,
    estate_allocator_jax,
    etr_measures,
    mtr_table,
)
from taxmodel.reporting import detail as detail_output
from taxmodel.reporting import tax_totals as tax_totals_output
from taxmodel.taxes import marginal_rates, statutory_order
from taxmodel.years import world_names, worlds

if TYPE_CHECKING:
    # Only for an annotation. `years/one_year.py` imports this module, so
    # importing it back at run time would be a circle.
    from taxmodel.years.one_year import BaselineYearOutputs

Array = jnp.ndarray

# The marginal-rate table reports the five income quintiles. Its labels come
# from the baseline's finer income ranking, so both tables use the same cuts.
MTR_GROUPS = len(distribution.QUINTILE_LABELS)
_RANK_GROUPS_PER_MTR_GROUP = distribution.RANK_GROUPS // MTR_GROUPS
_MTR_PRICE_NAMES = tuple(
    name.removeprefix(f"{marginal_rates.BASELINE_LAW}_")
    for name in marginal_rates.rate_columns(marginal_rates.BASELINE_LAW)
)
# The frame columns whose dollars weight each rate the marginal-rate table
# reports. Every priced dollar is named for the frame column it moves and is
# weighted by that column's dollars. The table is formed after the year has
# run, so these columns have to leave the year with the rates.
_MTR_BASE_COLUMNS: tuple[str, ...] = tuple(marginal_rates.PRICED_DOLLARS)
# The household weight the marginal-rate table adds up with, stored in the
# same dictionary as the base columns under a name no priced dollar uses.
_MTR_WEIGHT = "weight"


# The five worlds the annual step evaluates, each with its (household frame,
# tax law) cell, in the order the result grid stores them. The names and cells
# are defined in years/world_names.py. Each world's results sit in the field of
# `AnnualResults` with the world's own name.
EVALUATED_CELLS: tuple[tuple[str, world_names.Frame, world_names.Law], ...] = tuple(
    (world, frame, law) for world, (frame, law) in world_names.WORLD_CELLS.items()
)

# The world the baseline sweep evaluates, which runs before the year loop and
# adds up its own totals there, and the four the year loop evaluates.
BASELINE_WORLD = EVALUATED_CELLS[0][0]
REFORM_WORLDS = tuple(world for world, _frame, _law in EVALUATED_CELLS[1:])

# The household frame each of those four worlds calculated its tax on. The
# published totals report income amounts from the frame, so each world's
# totals are added up over its own.
FRAME_FIELD_BY_WORLD = {
    "first_order": "raw_frame",
    "mechanical_ex_saving": "flows_frame",
    "mechanical": "mechanical_frame",
    "conventional": "behavior_frame",
}


# Columns totaled over every household, filer or not. Payroll tax is owed
# through withholding whether or not a return is filed, and the wealth tax
# is assessed on net worth, not on a 1040. Everything else follows the
# 1040-form convention below.
_TOTALED_OVER_EVERYONE = (
    "liab_pr",
    "liab_fica",
    "liab_seca",
    "liab_oasdi",
    "liab_hi",
    "liab_add_med",
    "liab_wealth",
)

# The record universe each published column belongs to, as the published
# files define it.
_EVERY_RECORD_COLUMNS = frozenset(tax_totals_output.EVERY_RECORD_COLUMNS)
_FILERS_ONLY_COLUMNS = frozenset(tax_totals_output.FILERS_ONLY_COLUMNS)


def _counted_over_everyone(name: str) -> bool:
    """Whether this column is added up over every household or over filers only.

    `reporting/tax_totals.py` settles the question for every column the four
    published files report: the payroll file covers everyone, including the
    wages and self-employment earnings the payroll tax is charged on, and the
    1040 file covers filers. The prefix rule above answers for the columns no
    published file names, since the run totals every result column.
    """
    if name in _EVERY_RECORD_COLUMNS:
        return True
    if name in _FILERS_ONLY_COLUMNS:
        return False
    return name.startswith(_TOTALED_OVER_EVERYONE)


# The two estate-tax figures the published totals report, formed per household
# before anything is added up. Both are expectations: nobody is known to die,
# so each household contributes its chance of dying this year times what its
# estate would owe.
EXPECTED_ESTATE_TAX = tax_totals_output.ESTATE_TAX_EXPECTED
EXPECTED_TAXABLE_ESTATES = tax_totals_output.ESTATE_RETURNS_EXPECTED

# The mortality column the frame produces. The blend below needs it and three
# of the estate calculator's columns; a run made without the estate
# measurement bridge has none of them and reports no estate figures.
ESTATE_MORTALITY_COLUMN = "estate_m"

# The capital-gains model's chance of dying this year, which the expected tax
# on gains deemed realized at death was formed with. The deemed-gains heir
# allocation uses it in place of `ESTATE_MORTALITY_COLUMN`.
DEEMED_GAINS_MORTALITY_COLUMN = applier.MORTALITY_COLUMN

# The two taxes charged at death, each spread over heirs by its own rank match
# on its own ladder of estates, in the order the diagnostics array holds them.
# The estate tax's ladder holds only estates that owe estate tax; the deemed
# gains ladder holds every distributable estate, because an estate with no
# unrealized gain can be any size, so it is much the longer of the two.
TAX_AT_DEATH_LADDERS: tuple[str, ...] = (
    "the estate tax",
    "the tax on gains deemed realized at death",
)


def _expected_estate_columns(
    cell: Mapping[str, Array], mortality: Array | None
) -> dict[str, Array]:
    """Expected estate tax and expected taxable estates, per household.

    A household that dies this year owes one of two amounts: what the estate
    owes when a predeceased spouse left an unused exclusion amount to carry
    over, and what it owes when there is none. The estate calculator computes
    both and the probability of the first, so the expected tax is the two
    liabilities blended and then multiplied by the chance of dying.

    The count of taxable estates blends the two branches' taxable-or-not
    answers. A household taxable in one branch and not in the other is a
    fraction of a taxable estate. Both figures are formed here, per household,
    where the branches still exist.

    Nothing comes back when the run has no estate tax to report: a run made
    without the estate measurement bridge, or a test whose frame builder does
    not compute mortality.
    """
    blend_reads = ("estate_p_dsue", "liab_estate_dsue", "liab_estate_nodsue")
    if mortality is None or any(name not in cell for name in blend_reads):
        return {}
    mortality = jnp.asarray(mortality)
    carryover_probability = jnp.asarray(cell["estate_p_dsue"])
    with_carryover = jnp.asarray(cell["liab_estate_dsue"])
    without_carryover = jnp.asarray(cell["liab_estate_nodsue"])
    expected_liability = (
        carryover_probability * with_carryover
        + (1.0 - carryover_probability) * without_carryover
    )
    expected_taxable = carryover_probability * (with_carryover > 0) + (
        1.0 - carryover_probability
    ) * (without_carryover > 0)
    return {
        EXPECTED_ESTATE_TAX: mortality * expected_liability,
        EXPECTED_TAXABLE_ESTATES: mortality * expected_taxable,
    }


# Seven amounts the published totals files report as one household figure,
# which this model keeps split between the two earners. None of the pieces
# overlaps another.
REPORTED_AS_ONE_HOUSEHOLD_FIGURE: dict[str, tuple[str, ...]] = {
    # An earner's W-2 wages are base pay plus tips plus overtime.
    "wages1": ("wages_base1", "tips1", "ot1"),
    "wages2": ("wages_base2", "tips2", "ot2"),
    "wages": ("wages_base1", "tips1", "ot1", "wages_base2", "tips2", "ot2"),
    "tips": ("tips1", "tips2"),
    "ot": ("ot1", "ot2"),
    "sole_prop": ("sole_prop1", "sole_prop2"),
    "farm": ("farm1", "farm2"),
}


def _added_up_over_the_two_earners(
    columns: Mapping[str, Array],
) -> dict[str, Array]:
    """The seven household figures above, where the pieces are all present."""
    added: dict[str, Array] = {}
    for name, pieces in REPORTED_AS_ONE_HOUSEHOLD_FIGURE.items():
        if any(piece not in columns for piece in pieces):
            continue
        total = columns[pieces[0]]
        for piece in pieces[1:]:
            total = total + columns[piece]
        added[name] = total
    return added


# The columns the four totals files report that the calculators do not create:
# income the household brings to the return (wages, interest, Social Security),
# read from the frame. Each world's own frame is read, because the
# employer-payroll adjustment changes wages and the wealth haircut changes
# capital income.
_FRAME_COLUMNS_THE_FILES_REPORT: tuple[str, ...] = tuple(
    name
    for name in tax_totals_output.every_totalled_column()
    if name not in statutory_order.CALCULATED_COLUMNS
    and name not in REPORTED_AS_ONE_HOUSEHOLD_FIGURE
    and name not in (EXPECTED_ESTATE_TAX, EXPECTED_TAXABLE_ESTATES)
)


# The counts of tax units the published totals files report, one per year,
# under the names `reporting/tax_totals.py` uses. They are weighted counts of
# households, not millions.
_UNIT_COUNT_NAMES = (
    "n_tax_units",
    "n_returns",
    "n_returns_dep",
    "n_dep",
    "n_simple_filers",
)


def _unit_counts(
    cell: Mapping[str, Array],
    population: Mapping[str, Array],
    weight: Array,
    filing_weight: Array,
) -> dict[str, Array]:
    """The weighted counts of tax units and returns, for one year of one world.

    ``cell`` holds this world's per-household columns and ``population`` the
    household facts that come from the data rather than from a calculator.
    """
    missing = [name for name in ("dep_status", "n_dep") if name not in population]
    if missing:
        raise KeyError(
            "the population must carry "
            f"{' and '.join(missing)}: the published totals report how many "
            "returns are filed by dependents and how many dependents are "
            "claimed on filed returns"
        )
    if "simple_filer" not in cell:
        raise KeyError(
            "the tax calculator must produce 'simple_filer': the published "
            "totals report how many returns the IRS could plausibly pre-fill"
        )
    return {
        # Every household in the population, whether or not it files.
        "n_tax_units": jnp.sum(weight, keepdims=True),
        # Households that file a return.
        "n_returns": jnp.sum(filing_weight, keepdims=True),
        # Returns filed by someone who is claimed as a dependent on another
        # return.
        "n_returns_dep": jnp.sum(
            filing_weight * jnp.asarray(population["dep_status"]), keepdims=True
        ),
        # Dependents claimed on filed returns.
        "n_dep": jnp.sum(
            filing_weight * jnp.asarray(population["n_dep"]), keepdims=True
        ),
        # Non-itemizing filers whose income is wages and Social Security only.
        # The flag already requires the household to file, so the filing flag
        # is not applied a second time.
        "n_simple_filers": jnp.sum(
            weight * jnp.asarray(cell["simple_filer"]), keepdims=True
        ),
    }


def _reported_columns(
    cell: Mapping[str, Array], frame: Mapping[str, Array] | None
) -> dict[str, Array]:
    """Everything one world reports, per household: its results and its frame.

    A published totals file reports both. `agi` and `liab_iit_net` are results;
    `wages` and `gross_ss` and `txbl_int` are income the household brought to
    the return, read from the frame. Seven of the figures the files report as
    one household number are added up from the two earners' columns here, and
    the two expected estate figures are formed here too, because the mortality
    they need is a frame column.

    A result wins over a frame column of the same name.
    """
    if frame is None:
        return dict(cell)
    reported = {
        name: frame[name]
        for name in _FRAME_COLUMNS_THE_FILES_REPORT
        if name in frame
    }
    reported.update(_added_up_over_the_two_earners(frame))
    reported.update(
        _expected_estate_columns(cell, frame.get(ESTATE_MORTALITY_COLUMN))
    )
    reported.update(cell)
    return reported


# The population columns the totals need that no calculator produces: the
# household's sampling weight, whether it is claimed as a dependent on someone
# else's return, and how many dependents it claims.
POPULATION_COLUMNS_THE_TOTALS_NEED = ("weight", "dep_status", "n_dep")


def population_columns_for_totals(
    raw: Mapping[str, Array],
) -> dict[str, Array]:
    """The three population columns the totals program reads.

    The totals are formed in their own compiled program, which reads only
    these three population columns.
    """
    return {
        name: jnp.asarray(raw[name])
        for name in POPULATION_COLUMNS_THE_TOTALS_NEED
        if name in raw
    }


def reported_columns_for_totals(
    cell: Mapping[str, Array],
    raw: Mapping[str, Array],
    frame: Mapping[str, Array] | None = None,
) -> dict[str, Array]:
    """One world's per-household columns, ready to be added up into totals.

    This part of the published totals happens inside the year program, where a
    household's own numbers still exist: picking out which columns are
    reported, adding the two earners' pieces together, and forming the two
    expected estate figures. `_reported_columns` above says where each column
    comes from. Nothing is weighted or added up here;
    :func:`population_totals_and_counts` does that in its own compiled program.

    A column that is not per-household is left out.
    """
    weight = jnp.asarray(raw["weight"])
    if "filer_recomputed" not in cell:
        raise KeyError(
            "the tax calculator must produce 'filer_recomputed': published "
            "totals count 1040 amounts for filers only, "
            "so a run without it would silently report a different revenue "
            "concept"
        )
    reported = _reported_columns(cell, frame)
    per_household: dict[str, Array] = {}
    for name, value in reported.items():
        value = jnp.asarray(value)
        if value.shape != weight.shape:
            continue
        count_name = f"n_{name}"
        if count_name in _UNIT_COUNT_NAMES or count_name in reported:
            raise ValueError(
                f"the count of nonzero {name} values would be stored as "
                f"{count_name}, which already names something else"
            )
        per_household[name] = value
    return per_household


def _totals_for_one_world(
    reported: Mapping[str, Array],
    population: Mapping[str, Array],
) -> dict[str, Array]:
    """One year of one world's per-household columns as totals and counts.

    Two numbers come back for each reported column: the weighted sum of the
    column, under its own name, and the weighted count of households whose
    value is not zero, under that name with an `n_` in front. Downstream
    writers look them up by these names. The five counts of tax units and
    returns are added on top.

    Income-tax amounts and their counts cover filers only: the model computes
    a liability for households that would not file a return, but no such
    household remits it. Payroll and wealth taxes are owed regardless of
    filing and count everyone. The two expected estate figures count everyone
    too, since an estate owes tax whether or not the decedent filed a 1040.

    Who counts as a filer is the model's own recomputed flag, read from this
    world's own results. A household the refundable-credit rule flips into
    filing is a filer, and because a reform can change who files, each world
    answers the question for itself.

    Every answer is kept as a one-element array in the result grid. Reporting
    reads its single population total directly.

    The sums are two matrix products per counting weight. One weighted sum per
    column would be some 1,500 separate whole-population reductions, each
    compiled as its own kernel, and those reductions were about half of a
    year's compile time.
    """
    weight = jnp.asarray(population["weight"])
    filing_weight = weight * jnp.asarray(reported["filer_recomputed"])
    totals: dict[str, Array] = {}
    for counted_over_everyone, counting_weight in (
        (True, weight),
        (False, filing_weight),
    ):
        # Sorted so the block's rows are in the same order however the column
        # dictionary was built or passed across a compiled program's boundary.
        names = [
            name
            for name in sorted(reported)
            if _counted_over_everyone(name) == counted_over_everyone
        ]
        if not names:
            continue
        # Cast to float64 before the product, as a per-column weighted sum
        # would by multiplying each column by the float64 weight.
        block = jnp.stack(
            [jnp.asarray(reported[name], dtype=jnp.float64) for name in names]
        )
        sums = block @ counting_weight
        counts = (block != 0.0).astype(jnp.float64) @ counting_weight
        for position, name in enumerate(names):
            totals[name] = sums[position : position + 1]
            totals[f"n_{name}"] = counts[position : position + 1]
    totals.update(_unit_counts(reported, population, weight, filing_weight))
    return totals


def population_totals_and_counts(
    reported_by_cell: Mapping[str, Mapping[str, Array]],
    population: Mapping[str, Array],
) -> dict[str, dict[str, Array]]:
    """Every world's published population totals for one projection year.

    This is the second of the two programs a projection year compiles. Its
    inputs are the finished per-household columns the year program produced,
    so it is a handful of matrix products over arrays that already exist, and
    it compiles in seconds.

    It is a separate program because inside the year graph either form is
    slow to compile: a weighted sum per column is some 1,500 reduction
    kernels, and stacking the columns there forces roughly 150 calculator
    results into existence as arrays the year would otherwise never
    materialize. As its own program the stack's inputs are parameters.
    """
    return {
        world: _totals_for_one_world(reported, population)
        for world, reported in reported_by_cell.items()
    }


def distribution_households(
    raw: Mapping[str, Array],
    baseline_frame: Mapping[str, Array],
    baseline_result: Mapping[str, Array],
) -> dict[str, Array]:
    """The baseline household amounts that define distribution groups.

    The older spouse supplies age on a joint return. A parent has at least one
    recorded dependent younger than 18. Expanded income and AGI come from the
    baseline tax calculation, while net worth comes from the baseline frame
    before any wealth shortfall is applied.
    """
    filing_status = jnp.asarray(raw["filing_status"])
    age1 = jnp.asarray(raw["age1"])
    age2 = jnp.asarray(raw["age2"])
    age = jnp.where(
        filing_status == steps_jax.FILING_JOINT, jnp.maximum(age1, age2), age1
    )
    dependent_ages = jnp.stack(
        [jnp.asarray(raw[f"dep_age{number}"]) for number in range(1, 4)], axis=-1
    )
    has_child_under_18 = jnp.any(
        ~jnp.isnan(dependent_ages) & (dependent_ages < 18.0), axis=-1
    )
    return {
        "weight": jnp.asarray(raw["weight"]),
        "age": age,
        "has_child_under_18": has_child_under_18,
        "income": jnp.asarray(baseline_result["expanded_inc"]),
        "agi": jnp.asarray(baseline_result["agi"]),
        "net_worth": jnp.asarray(baseline_frame["net_worth"]),
        "valid": jnp.asarray(raw["valid"]),
        "is_dependent_return": jnp.asarray(raw["dep_status"], dtype=bool),
    }


def _reporting_population(
    households: Mapping[str, Array],
    heir: Mapping[str, Array],
) -> dict[str, Array]:
    """The households the distribution table reports on.

    With no heir structure this is the run's households, with the two flags the
    table reads (the slot holds a household this year, and the return was not
    filed by someone else's dependent) combined into one.

    With an heir structure every household appears twice, once as an heir and
    once as a non-heir; `reporting/distribution_jax.py` explains the split.
    """
    keep = jnp.asarray(households["valid"], dtype=bool) & ~jnp.asarray(
        households["is_dependent_return"], dtype=bool
    )
    if not heir:
        plain = {
            name: value
            for name, value in households.items()
            if name not in ("valid", "is_dependent_return")
        }
        plain["valid"] = keep
        return plain
    return distribution_jax.heir_and_non_heir_copies(
        dict(households), heir["probability_of_inheriting"]
    )


def baseline_rank_labels(
    households: Mapping[str, Array],
    heir: Mapping[str, Array],
) -> dict[str, Array]:
    """Rank the baseline amounts once before the reform worlds are run.

    `agi` and `net_worth` rank the reporting population (the households split
    into heir and non-heir copies when there is an heir structure), and the
    distribution table's AGI and Net worth cuts are cut on them. Neither
    amount picks up a bequest.

    `household_income` ranks the undivided households on expanded income. The
    marginal-rate table reports quintiles of it, and that table has one row
    per household rather than one per copy.
    """
    copies = _reporting_population(households, heir)
    keep = jnp.asarray(copies["valid"], dtype=bool)
    weight = jnp.asarray(copies["weight"])
    return {
        "agi": distribution_jax.baseline_rank_groups(
            jnp.asarray(copies["agi"]), weight, keep
        ),
        "net_worth": distribution_jax.baseline_rank_groups(
            jnp.asarray(copies["net_worth"]), weight, keep
        ),
        "household_income": distribution_jax.baseline_rank_groups(
            jnp.asarray(households["income"]),
            jnp.asarray(households["weight"]),
            jnp.asarray(households["valid"], dtype=bool)
            & ~jnp.asarray(households["is_dependent_return"], dtype=bool),
        ),
    }


def _for_the_heir_copy_only(
    values: Array | None, households_are_split: bool, n_households: int
) -> Array:
    """An amount only an heir has, laid out over the reporting copies.

    On a split population the heir copy carries the amount and the non-heir
    copy carries zero. With no heir structure, or with nothing to place, the
    answer is zeros.
    """
    zeros = jnp.zeros(n_households, dtype=jnp.float64)
    if not households_are_split:
        return zeros
    heir_copy = zeros if values is None else jnp.asarray(values, dtype=jnp.float64)
    return jnp.concatenate([heir_copy, zeros])


def _for_every_copy(values: Array, households_are_split: bool) -> Array:
    """One value per reporting copy, from one value per household."""
    values = jnp.asarray(values, dtype=jnp.float64)
    return jnp.concatenate([values, values]) if households_are_split else values


def _income_quintile_labels(income_rank_labels: Array) -> Array:
    """Turn baseline income ranks into the five labels the MTR table reports."""
    income_rank_labels = jnp.asarray(income_rank_labels)
    return jnp.where(
        income_rank_labels >= 0,
        income_rank_labels // _RANK_GROUPS_PER_MTR_GROUP,
        -1,
    )


class ReportColumns(NamedTuple):
    """One year's finished per-household columns, for the report programs.

    Three families of reported numbers are formed after the year has run,
    each in its own small compiled program: the heirs' share of the taxes
    charged at death, the distribution table, and the marginal-rate table.
    These are their inputs. Nothing here is weighted or added up.

    Three of these dictionaries are empty when the run cannot produce the
    family they feed (a run whose calculator computes no estate columns, no
    marginal-rate prices, or none of the income and tax columns the
    distribution table reads). The program in question then returns nothing.

    Attributes:
        households: The baseline household amounts the reported groups are cut
            on (`distribution_households`).
        rank_labels: The baseline rankings every world is reported on
            (`baseline_rank_labels`).
        at_death: The estate calculator's four columns and each world's chance
            of dying, one row per world in `EVALUATED_CELLS` order, which the heir allocation matches estates to heirs with,
            plus the expected tax on gains deemed realized at death where a
            scenario charges one.
        mtr_rates: The 108 marginal rates that table reports: 27 priced
            dollars, under each of the two definitions of which taxes are in
            the price, under each of the two laws.
        mtr_records: The mechanical frame's columns holding the dollars each
            rate is charged on, and the household weight under `_MTR_WEIGHT`.
        world_measures: The distribution table's per-household income and
            tax amounts: each name in `etr_measures.WORLD_MEASURES`, with one
            row per world in `etr_measures.MEASURED_WORLDS` order.
        corporate_bases: The baseline world's per-household amounts every
            world's corporate tax is spread over, under the names in
            `corporate_spread.BASE_NAMES`.
        add_backs: What the record-level corporate and employer-payroll steps
            changed in each conventional-world household's income, as it
            stands in the final conventional frame, one array per name in
            `corporate_spread.ADD_BACK_COLUMNS`.
    """

    households: dict[str, Array]
    rank_labels: dict[str, Array]
    at_death: dict[str, Array]
    mtr_rates: dict[str, Array]
    mtr_records: dict[str, Array]
    world_measures: dict[str, Array]
    corporate_bases: dict[str, Array]
    add_backs: dict[str, Array]


def population_columns_for_reports(
    raw: Mapping[str, Array],
) -> dict[str, Array]:
    """The population columns the report programs read.

    Both are for the heir allocation: the household's sampling weight, and its
    identifier, which breaks ties in the ranking of estates.
    """
    return {
        name: jnp.asarray(raw[name])
        for name in ("weight", "id")
        if name in raw
    }


# The empty value a year's report columns are replaced by once the three
# report programs have read them, so the driver does not carry them to the end
# of the window.
NO_REPORT_COLUMNS = ReportColumns(
    households={},
    rank_labels={},
    at_death={},
    mtr_rates={},
    mtr_records={},
    world_measures={},
    corporate_bases={},
    add_backs={},
)


def _stacked_over_worlds(values: Sequence[Array]) -> Array:
    """One array per world, laid out with the worlds on the first axis."""
    return jnp.stack([jnp.asarray(value, dtype=jnp.float64) for value in values])


def report_columns_for_the_year(
    baseline: BaselineYearOutputs,
    heir: Mapping[str, Array],
    year_results: worlds.AnnualResults,
    pass_through_weight: Array,
) -> ReportColumns:
    """Pick out what the report programs read, inside the year.

    This part of the reporting happens in the year program, where a
    household's own numbers still exist. It only picks columns out; adding up
    happens in the report programs.

    Each world reads its chances of dying off its own frame. The numbers are
    the same in every world, because the frame steps compute mortality before
    the reform income adjustments and before the wealth haircut.

    `pass_through_weight` is the capital share of pass-through income the
    wealth haircut scaled this year, which the add-backs read
    (`corporate_spread.add_backs`).
    """
    cells = [getattr(year_results, world) for world, _frame, _law in EVALUATED_CELLS]
    mortality_by_world = [baseline.mortality] + [
        getattr(year_results, FRAME_FIELD_BY_WORLD[world])
        for world in REFORM_WORLDS
    ]
    return ReportColumns(
        households=baseline.distribution_households,
        rank_labels=baseline.rank_labels,
        at_death=_at_death_columns(cells, mortality_by_world, heir),
        mtr_rates=_mtr_rate_columns(year_results),
        mtr_records=_mtr_record_columns(year_results),
        world_measures=_world_measure_columns(baseline.world_measures, year_results),
        corporate_bases=baseline.corporate_bases,
        add_backs=corporate_spread.add_backs(
            year_results.behavior_frame, pass_through_weight
        ),
    )


# The frame each reform world the distribution table reads was calculated on.
# The baseline world's measures are formed in the baseline sweep, where its
# frame exists.
_REFORM_WORLD_FRAMES = {
    world: FRAME_FIELD_BY_WORLD[world]
    for world in etr_measures.MEASURED_WORLDS
    if world != BASELINE_WORLD
}


def baseline_world_measures(
    tax_result: Mapping[str, Array],
    frame: Mapping[str, Array],
) -> dict[str, Array]:
    """The baseline world's distribution-table measures, formed in the baseline sweep.

    Nothing comes back when the calculator does not produce the columns the
    table reads (the loop tests stand a simple function in for the calculator
    chain), and the run then writes no distribution table.
    """
    if not etr_measures.forms_the_table(tax_result):
        return {}
    return etr_measures.world_measures(tax_result, frame)


def baseline_corporate_bases(
    tax_result: Mapping[str, Array],
    frame: Mapping[str, Array],
) -> dict[str, Array]:
    """The baseline world's corporate bases, formed in the baseline sweep.

    Empty whenever `baseline_world_measures` is.
    """
    if not etr_measures.forms_the_table(tax_result):
        return {}
    return etr_measures.baseline_corporate_bases(tax_result, frame)


def _world_measure_columns(
    baseline_measures: Mapping[str, Array],
    year_results: worlds.AnnualResults,
) -> dict[str, Array]:
    """The distribution table's per-household amounts, one row per world."""
    if not baseline_measures:
        return {}
    by_world = [baseline_measures] + [
        etr_measures.world_measures(
            getattr(year_results, world), getattr(year_results, frame_field)
        )
        for world, frame_field in _REFORM_WORLD_FRAMES.items()
    ]
    return {
        name: _stacked_over_worlds([measures[name] for measures in by_world])
        for name in etr_measures.WORLD_MEASURES
    }


def _at_death_columns(
    cells: Sequence[Mapping[str, Array]],
    mortality_by_world: Sequence[Mapping[str, Array]],
    heir: Mapping[str, Array],
) -> dict[str, Array]:
    """The columns the heir allocation reads, one row per world.

    ``mortality_by_world`` holds, for each world, a mapping that carries the
    estate tax's chance of dying and, in a scenario that runs the gains model,
    the capital-gains model's chance of dying.

    Nothing comes back when the run has no heir structure or was made without
    the estate measurement bridge, and the distribution table then reports no
    tax at death.
    """
    if not heir:
        return {}
    if any(
        column not in cell
        for cell in cells
        for column in estate_allocator_jax.ESTATE_RESULT_COLUMNS
    ):
        return {}
    if any(
        ESTATE_MORTALITY_COLUMN not in mortality for mortality in mortality_by_world
    ):
        return {}
    columns = {
        name: _stacked_over_worlds([cell[name] for cell in cells])
        for name in estate_allocator_jax.ESTATE_RESULT_COLUMNS
    }
    columns[ESTATE_MORTALITY_COLUMN] = _stacked_over_worlds(
        [mortality[ESTATE_MORTALITY_COLUMN] for mortality in mortality_by_world]
    )
    # A deemed-realization scenario's worlds carry an expected tax on gains
    # realized at death. It reaches heirs through the same rank match as the
    # estate tax, and `years/burden.py` leaves it out of the during-life
    # burden. A world that does not carry the tax column owes nothing, and a
    # world without the capital-gains mortality column has no one dying on the
    # deemed-gains line.
    deemed = estate_allocator_jax.DEEMED_GAINS_COLUMN
    if any(deemed in cell for cell in cells):

        def column_or_zeros(values: Mapping[str, Array], name: str) -> Array:
            if name in values:
                return values[name]
            return jnp.zeros_like(
                jnp.asarray(cells[0]["estate_distributable"], dtype=jnp.float64)
            )

        columns[deemed] = _stacked_over_worlds(
            [column_or_zeros(cell, deemed) for cell in cells]
        )
        columns[DEEMED_GAINS_MORTALITY_COLUMN] = _stacked_over_worlds(
            [
                column_or_zeros(mortality, DEEMED_GAINS_MORTALITY_COLUMN)
                for mortality in mortality_by_world
            ]
        )
    return columns


def _priced_the_marginal_rates(year_results: worlds.AnnualResults) -> bool:
    """Whether this run priced the marginal rates the table is built from.

    A run that feeds no behavioral response leaves the gradient passes out, so
    there are no rates. Both column pickers below then return nothing, and no
    marginal-rate table is written.
    """
    return all(
        name in year_results.mechanical_prices_reform
        and name in year_results.mechanical_prices_baseline
        for name in _MTR_PRICE_NAMES
    )


def _mtr_rate_columns(
    year_results: worlds.AnnualResults,
) -> dict[str, Array]:
    """The marginal rates the table reports, under both laws."""
    if not _priced_the_marginal_rates(year_results):
        return {}
    rates = {
        **marginal_rates.under_law(
            year_results.mechanical_prices_baseline, marginal_rates.BASELINE_LAW
        ),
        **marginal_rates.under_law(
            year_results.mechanical_prices_reform, marginal_rates.REFORM_LAW
        ),
    }
    # Only the rates the table reports are returned. A run may price more than
    # it reports.
    return {name: jnp.asarray(rates[name]) for name in mtr_table.RATE_NAMES}


def _mtr_record_columns(
    year_results: worlds.AnnualResults,
) -> dict[str, Array]:
    """The dollars each marginal rate is charged on, off the mechanical frame."""
    if not _priced_the_marginal_rates(year_results):
        return {}
    frame = year_results.mechanical_frame
    records = {name: jnp.asarray(frame[name]) for name in _MTR_BASE_COLUMNS}
    records[_MTR_WEIGHT] = jnp.asarray(frame[_MTR_WEIGHT])
    return records


def tax_at_death_for_year(
    columns: ReportColumns,
    heir: Mapping[str, Array],
    raw: Mapping[str, Array],
) -> dict[str, Array]:
    """What each world's taxes owed at death cost the heirs, per household.

    The first of the three report programs, and the one the distribution
    table depends on. Three arrays come back under the keys `estate_tax`,
    `deemed_gains_tax` and `estate_allocation`, each with one row per
    evaluated world in `EVALUATED_CELLS` order. The two taxes are kept apart
    because the distribution table reports each as its own tax.

    Each tax is spread over heirs by its own rank match, on its own ladder of
    estates, and the two ladders are not the same length: the estate tax's
    ladder holds only the estates that owe estate tax, while the deemed-gains
    ladder holds every distributable estate. The diagnostics therefore carry a
    ladder axis, so a reader can tell which ladder ran out of heirs.
    Position 0 is the estate tax's ladder
    and position 1 the deemed-gains ladder, named by `TAX_AT_DEATH_LADDERS`.

    Nothing comes back when the year produced no estate columns to allocate.
    """
    at_death = columns.at_death
    if not at_death:
        return {}
    estate_tax, deemed_gains_tax, diagnostics = (
        estate_allocator_jax.allocate_taxes_at_death(
            weight=jnp.asarray(raw["weight"]),
            record_id=jnp.asarray(raw["id"]),
            mortality_probability=at_death[ESTATE_MORTALITY_COLUMN],
            probability_of_unused_spousal_exemption=at_death["estate_p_dsue"],
            liability_with_unused_spousal_exemption=at_death["liab_estate_dsue"],
            liability_without_unused_spousal_exemption=at_death[
                "liab_estate_nodsue"
            ],
            distributable_estate=at_death["estate_distributable"],
            expected_deemed_gains_tax=at_death.get(
                estate_allocator_jax.DEEMED_GAINS_COLUMN
            ),
            deemed_gains_mortality_probability=at_death.get(
                DEEMED_GAINS_MORTALITY_COLUMN
            ),
            probability_of_inheriting=heir["probability_of_inheriting"],
            inheritance=heir["inheritance"],
        )
    )
    return {
        "estate_tax": estate_tax,
        "deemed_gains_tax": deemed_gains_tax,
        "estate_allocation": diagnostics,
    }


# Where each evaluated world sits in the per-world arrays of the heir
# allocation, which carry all five worlds.
_POSITION_AMONG_EVALUATED_CELLS = {
    world: position for position, (world, _frame, _law) in enumerate(EVALUATED_CELLS)
}


def distribution_table_for_year(
    columns: ReportColumns,
    heir: Mapping[str, Array],
    at_death: Mapping[str, Array],
    corporate: corporate_spread.CorporateDollars,
) -> Array | dict:
    """The distribution table for one projection year, under both corporate spreads.

    ``at_death`` is what :func:`tax_at_death_for_year` worked out, or nothing
    when the run allocated no tax at death, in which case both taxes at death
    are zero. ``corporate`` is this year's national corporate figures for
    this scenario, each field a scalar.

    The table reports three worlds, in `distribution.LEVEL_WORLDS` order. Each
    reads its income and individual taxes from the measures of one evaluated
    world (`ReportColumns.world_measures`): the baseline from the baseline,
    the static world from first_order, the conventional world from
    conventional. The corporate tax a household bears is its share of
    current-law corporate tax (C_b) in the baseline, C_b plus its share of the
    reform's static corporate change in the static world, and C_b plus its
    share of the reform's conventional corporate change in the conventional
    world, all spread over the baseline world's bases
    (`ReportColumns.corporate_bases`) of the households the table counts. Each world's pretax income adds C_b and
    the bequest to the world's own income measure; the conventional world's
    also adds back what the record-level corporate and employer-payroll steps
    took out of income.

    The table is formed once for each spread in `distribution.CORPORATE_SPREADS`,
    with that spread's corporate amounts in both the taxes and the pretax
    incomes. Everything else is the same in the two versions.

    Returns [spread, tax row, group, column], each version laid out as
    `distribution_jax.distribution_table` lays it out, or nothing when the
    year formed no such measures.
    """
    measures = columns.world_measures
    if not measures:
        return {}
    households = columns.households
    split = bool(heir)
    n_households = len(jnp.asarray(households["weight"]))
    copies = _reporting_population(households, heir)
    # The corporate bases are totaled over the undivided households the table
    # counts: returns filed by dependents and empty slots hold no base and
    # bear no corporate tax.
    counted = jnp.asarray(households["valid"], dtype=bool) & ~jnp.asarray(
        households["is_dependent_return"], dtype=bool
    )
    household_weight = jnp.asarray(households["weight"], dtype=jnp.float64)
    corporate_bases = {
        name: jnp.where(counted, jnp.asarray(value, dtype=jnp.float64), 0.0)
        for name, value in columns.corporate_bases.items()
    }
    bequest = _for_the_heir_copy_only(heir.get("inheritance"), split, n_households)

    def this_world(name: str, world: str) -> Array:
        return measures[name][etr_measures.MEASURED_WORLDS.index(world)]

    cash_taken_out, accrual_taken_out = corporate_spread.income_taken_out(
        columns.add_backs
    )
    zeros = jnp.zeros(n_households, dtype=jnp.float64)
    # Each reported world: the evaluated world it reads, which of the spread's
    # reform changes its households bear, and what is added back to its cash
    # and accrual income.
    reported = (
        ("baseline", None, zeros, zeros),
        ("first_order", "static_change", zeros, zeros),
        ("conventional", "conventional_change", cash_taken_out, accrual_taken_out),
    )

    def inputs_for_one_spread(spread: str) -> tuple[Array, Array, Array]:
        """Pretax cash income, pretax accrual income and taxes under one spread."""
        shares = corporate_spread.household_corporate_tax(
            spread, corporate, household_weight, corporate_bases
        )
        current_law = shares.current_law
        cash_income, accrual_income, tax = [], [], []
        for world, change, cash_added_back, accrual_added_back in reported:
            cell = _POSITION_AMONG_EVALUATED_CELLS[world]
            corporate_tax = (
                current_law if change is None else current_law + getattr(shares, change)
            )

            def heirs_bear(name: str, cell: int = cell) -> Array:
                return _for_the_heir_copy_only(
                    at_death[name][cell] if at_death else None, split, n_households
                )

            cash_income.append(
                _for_every_copy(
                    this_world("expanded_income", world) + current_law + cash_added_back,
                    split,
                )
                + bequest
            )
            accrual_income.append(
                _for_every_copy(
                    this_world("haig_simons_income", world)
                    + current_law
                    + accrual_added_back,
                    split,
                )
                + bequest
            )
            # In `distribution.TAXES` order.
            tax.append(
                jnp.stack(
                    [
                        _for_every_copy(this_world("income_tax", world), split),
                        _for_every_copy(this_world("payroll_tax", world), split),
                        heirs_bear("estate_tax"),
                        heirs_bear("deemed_gains_tax"),
                        _for_every_copy(this_world("wealth_tax", world), split),
                        _for_every_copy(corporate_tax, split),
                    ]
                )
            )
        return jnp.stack(cash_income), jnp.stack(accrual_income), jnp.stack(tax)

    by_spread = [inputs_for_one_spread(spread) for spread in distribution.CORPORATE_SPREADS]
    cash_income, accrual_income, tax = (
        jnp.stack(parts) for parts in zip(*by_spread, strict=True)
    )
    rank_labels = columns.rank_labels

    def table_for_one_spread(cash: Array, accrual: Array, taxes: Array) -> Array:
        return distribution_jax.distribution_table(
            weight=copies["weight"],
            valid=copies["valid"],
            age=copies["age"],
            has_child_under_18=copies["has_child_under_18"],
            agi=copies["agi"],
            net_worth=copies["net_worth"],
            agi_rank_groups=rank_labels["agi"],
            net_worth_rank_groups=rank_labels["net_worth"],
            pretax_cash_income=cash,
            pretax_accrual_income=accrual,
            tax=taxes,
        )

    # The spread axis leads; the population and its baseline rankings are
    # shared by both versions.
    return jax.vmap(table_for_one_spread)(cash_income, accrual_income, tax)


def mtr_table_sums_for_year(columns: ReportColumns) -> dict[str, Array]:
    """The marginal-rate table's two accumulators for one projection year.

    Nothing comes back when the run priced no marginal rates.
    """
    if not columns.mtr_rates:
        return {}
    return mtr_table.mtr_table_sums(
        columns.mtr_rates,
        columns.mtr_records,
        _income_quintile_labels(columns.rank_labels["household_income"]),
        columns.mtr_records[_MTR_WEIGHT],
        MTR_GROUPS,
    )


def detail_columns_available(
    request: detail_output.DetailRequest,
    year_results: worlds.AnnualResults,
) -> dict[str, dict[str, Array]]:
    """What each requested world computed this year, ready to be read from.

    A world's columns are the calculator's results for that cell. The
    mechanical world also offers the marginal tax rates, which are measured on
    the mechanical frame under both laws, so those are added under the names
    that say which law each rate came from.
    """
    available: dict[str, dict[str, Array]] = {
        world: dict(getattr(year_results, world))
        for world in request.worlds
    }
    mechanical = available.get(detail_output.MARGINAL_RATE_WORLD)
    if mechanical is not None:
        mechanical.update(
            marginal_rates.under_law(
                year_results.mechanical_prices_baseline, marginal_rates.BASELINE_LAW
            )
        )
        mechanical.update(
            marginal_rates.under_law(
                year_results.mechanical_prices_reform, marginal_rates.REFORM_LAW
            )
        )
    return available


def _put_yearly_totals(
    grid: world_names.ResultGrid,
    totals_by_cell: Mapping[str, dict[str, Array]],
    years: tuple[int, ...],
) -> None:
    """Lay the year-stacked totals into the grid, one year at a time.

    The totals arrive as device arrays, one per column per cell, each carrying
    all the years. They are copied to host memory whole before being sliced by
    year, so there is one transfer per column instead of one per column per
    year.
    """
    on_host = jax.device_get(dict(totals_by_cell))
    for index, year in enumerate(years):
        for name, frame, law in EVALUATED_CELLS:
            grid.put(
                frame,
                law,
                year,
                {column: value[index] for column, value in on_host[name].items()},
            )
