"""The distribution table, on a small made-up population with heirs.

The population is random but seeded. Some households are dependents' returns
or empty slots, which the table leaves out; some have negative income; a few
have a net worth of a billion dollars or more; most have some chance of
inheriting. Each check below is an accounting identity the table must satisfy,
compared to the cent on dollar amounts, under both corporate spreads.
"""

from __future__ import annotations

import numpy as np
import pytest

from taxmodel.reporting import (
    corporate_spread,
    device_sums,
    distribution,
    etr_measures,
)
from taxmodel.years.corporate import burden_jax
from tests.tolerances import CENT, assert_agrees_to_the_cent

N_HOUSEHOLDS = 80

# The three reported worlds read these evaluated worlds.
READS = {"baseline": "baseline", "static": "first_order", "conventional": "conventional"}

# Made-up national corporate figures for one year, in dollars, and the
# incidence assumptions of the economy leg.
BASELINE_CORPORATE = 4e8
STATIC_CORPORATE = 4.6e8
CONVENTIONAL_CORPORATE = 4.4e8
COST_RECOVERY = -3e7
NORMAL_RETURN_SHARE = 0.375
FOREIGN_SHARE = 0.4
EXPOSURE = {"equities": 1.0, "dc": 0.55, "trusts": 0.5, "re_fund": 0.3}
HOUSING_STRUCTURE_SHARE = 0.7

# The balance sheet, written out here so that the hand computation of the
# asset spread does not borrow the code's own lists.
NON_HOUSING_ASSETS = (
    "value.cash",
    "value.equities",
    "value.bonds",
    "value.dc",
    "value.db",
    "value.life_ins",
    "value.annuities",
    "value.trusts",
    "value.other_fin",
    "value.pass_throughs",
    "value.re_fund",
    "value.other_nonfin",
)
HOMES = ("value.primary_home", "value.other_home")
MORTGAGES = ("value.primary_mortgage", "value.other_mortgage")
NON_MORTGAGE_DEBTS = (
    "value.credit_lines",
    "value.credit_cards",
    "value.installment_debt",
    "value.other_debt",
)

# Rates are ratios of sums in the millions; the per-tax rates are compared
# with the total rate to this many parts in a million million.
RATE_TOLERANCE = 1e-12

# Each of the six per-tax averages and the total average is rounded to the
# nearest five dollars, so their sums can differ by up to half of five dollars
# for each of the seven.
AVERAGE_ROUNDING_GAP = distribution.AVERAGE_ROUNDING_DOLLARS / 2 * len(
    distribution.TAX_ROWS
)

SPREADS = distribution.CORPORATE_SPREADS


def corporate_dollars(
    static: float = STATIC_CORPORATE, conventional: float = CONVENTIONAL_CORPORATE
) -> corporate_spread.CorporateDollars:
    return corporate_spread.CorporateDollars(
        baseline=np.float64(BASELINE_CORPORATE),
        static=np.float64(static),
        conventional=np.float64(conventional),
        cost_recovery=np.float64(COST_RECOVERY),
        normal_return_share=np.float64(NORMAL_RETURN_SHARE),
        foreign_share=np.float64(FOREIGN_SHARE),
        equities_exposure=np.float64(EXPOSURE["equities"]),
        dc_exposure=np.float64(EXPOSURE["dc"]),
        trusts_exposure=np.float64(EXPOSURE["trusts"]),
        re_fund_exposure=np.float64(EXPOSURE["re_fund"]),
        housing_structure_share=np.float64(HOUSING_STRUCTURE_SHARE),
    )


def made_up_balance_sheet(rng, n: int) -> dict[str, np.ndarray]:
    """Balance-sheet columns, with some households deep in debt."""
    frame = {name: rng.lognormal(10.0, 2.0, n) for name in NON_HOUSING_ASSETS}
    for name in HOMES:
        frame[name] = np.where(rng.random(n) < 0.6, rng.lognormal(12.5, 0.8, n), 0.0)
    for name in MORTGAGES:
        frame[name] = frame[HOMES[MORTGAGES.index(name)]] * rng.uniform(0.0, 1.2, n)
    for name in NON_MORTGAGE_DEBTS:
        frame[name] = rng.lognormal(9.0, 1.5, n)
    # A few households whose debts exceed every asset, so their net capital is
    # floored at zero, and one missing entry, read as zero.
    frame["value.other_debt"][10:14] = 5e7
    frame["value.cash"][20] = np.nan
    return frame


def made_up_year(seed: int = 3, add_backs_are_zero: bool = False):
    """Report columns, heir structure, taxes at death and the balance sheet.

    Returns (columns, heir, at_death, balance_sheet). The world measures carry
    one row per world in `etr_measures.MEASURED_WORLDS` order; the taxes at
    death one row per evaluated world. The corporate bases are formed from the
    balance sheet by the code under test.
    """
    rng = np.random.default_rng(seed)
    n = N_HOUSEHOLDS
    weight = rng.uniform(50.0, 2_000.0, n)
    expanded = rng.lognormal(11.0, 1.5, n)
    expanded[:5] = -rng.uniform(1_000.0, 50_000.0, 5)
    net_worth = rng.lognormal(12.0, 2.0, n)
    net_worth[5:8] = rng.uniform(1e9, 5e9, 3)
    net_worth[8] = -40_000.0
    households = {
        "weight": weight,
        "age": rng.integers(18, 90, n).astype(np.float64),
        "has_child_under_18": rng.random(n) < 0.3,
        "income": expanded,
        "agi": expanded * rng.uniform(0.8, 1.0, n),
        "net_worth": net_worth,
        "valid": np.arange(n) != n - 1,
        "is_dependent_return": np.arange(n) == n - 2,
    }
    heir = {
        "probability_of_inheriting": np.where(
            rng.random(n) < 0.2, 0.0, rng.uniform(0.0, 0.3, n)
        ),
        "inheritance": rng.lognormal(11.0, 1.5, n),
    }
    n_worlds = len(etr_measures.MEASURED_WORLDS)

    def by_world(values: np.ndarray, spread: float) -> np.ndarray:
        return np.stack(
            [values * rng.uniform(1.0 - spread, 1.0 + spread, n) for _ in range(n_worlds)]
        )

    expanded_by_world = np.stack(
        [expanded] + [expanded * rng.uniform(0.95, 1.05, n) for _ in range(n_worlds - 1)]
    )
    measures = {
        "expanded_income": expanded_by_world,
        "haig_simons_income": expanded_by_world + rng.normal(0.0, 20_000.0, (n_worlds, n)),
        "income_tax": by_world(np.maximum(expanded, 0.0) * 0.15, 0.1),
        "payroll_tax": by_world(np.maximum(expanded, 0.0) * 0.07, 0.05),
        "wealth_tax": by_world(np.maximum(net_worth - 5e7, 0.0) * 0.02, 0.1),
    }
    balance_sheet = made_up_balance_sheet(rng, n)
    bases = {
        name: np.asarray(value)
        for name, value in corporate_spread.baseline_bases(
            np.maximum(expanded, 0.0) * 0.6,
            np.maximum(expanded, 0.0) * 0.2 * rng.uniform(0.5, 1.5, n),
            balance_sheet,
        ).items()
    }
    n_cells = len(device_sums.EVALUATED_CELLS)
    at_death = {
        "estate_tax": rng.uniform(0.0, 30_000.0, (n_cells, n)),
        "deemed_gains_tax": rng.uniform(0.0, 5_000.0, (n_cells, n)),
    }
    add_backs = {
        name: (np.zeros(n) if add_backs_are_zero else rng.normal(-500.0, 300.0, n))
        for name in corporate_spread.ADD_BACK_COLUMNS
    }
    rank_labels = device_sums.baseline_rank_labels(households, heir)
    columns = device_sums.NO_REPORT_COLUMNS._replace(
        households=households,
        rank_labels=rank_labels,
        world_measures=measures,
        corporate_bases=bases,
        add_backs=add_backs,
    )
    return columns, heir, at_death, balance_sheet


def host_spread(amount, normal_return_share, weight, labor, above_normal, normal):
    """The three-way corporate split, in NumPy, per household."""
    half = burden_jax.LABOR_SHARE_OF_NORMAL_RETURN
    return (
        (1.0 - normal_return_share)
        * amount
        * (1.0 - FOREIGN_SHARE)
        * above_normal
        / np.sum(above_normal * weight)
        + normal_return_share * amount * half * labor / np.sum(labor * weight)
        + normal_return_share
        * amount
        * half
        * (1.0 - FOREIGN_SHARE)
        * normal
        / np.sum(normal * weight)
    )


def host_capital_bases(columns, balance_sheet, spread: str):
    """The above-normal and normal-capital bases, worked out by hand."""
    if spread == distribution.CORPORATE_SPREAD_INCOME:
        capital = columns.corporate_bases[corporate_spread.CAPITAL_INCOME]
        return capital, capital
    value = {name: np.nan_to_num(column) for name, column in balance_sheet.items()}
    corporate_equity = sum(
        value[f"value.{name}"] * exposure for name, exposure in EXPOSURE.items()
    )
    net_home_equity = sum(value[name] for name in HOMES) - sum(
        value[name] for name in MORTGAGES
    )
    net_capital = np.maximum(
        0.0,
        sum(value[name] for name in NON_HOUSING_ASSETS)
        - sum(value[name] for name in NON_MORTGAGE_DEBTS)
        + HOUSING_STRUCTURE_SHARE * net_home_equity,
    )
    return corporate_equity, net_capital


def host_corporate_tax(columns, balance_sheet, spread: str, world: str) -> np.ndarray:
    """A household's corporate tax in one reported world, worked out by hand.

    It is the world's national figure spread over the baseline bases, plus the
    cost-recovery change spread with all of it on the normal return. The bases
    are totaled over the households the table counts; the others bear nothing.
    """
    households = columns.households
    counted = households["valid"] & ~households["is_dependent_return"]
    weight = households["weight"]
    labor = np.where(counted, columns.corporate_bases[corporate_spread.LABOR_INCOME], 0.0)
    above_normal, normal = (
        np.where(counted, base, 0.0)
        for base in host_capital_bases(columns, balance_sheet, spread)
    )
    amount = {
        "baseline": BASELINE_CORPORATE,
        "static": STATIC_CORPORATE,
        "conventional": CONVENTIONAL_CORPORATE,
    }[world]
    bases = (weight, labor, above_normal, normal)
    return host_spread(amount, NORMAL_RETURN_SHARE, *bases) + host_spread(
        COST_RECOVERY, burden_jax.COST_RECOVERY_NORMAL_RETURN_SHARE, *bases
    )


def kept_copy_weights(columns, heir) -> tuple[np.ndarray, np.ndarray]:
    """The heir copy's and the non-heir copy's weight, zero where not counted."""
    households = columns.households
    kept = households["valid"] & ~households["is_dependent_return"]
    chance = heir["probability_of_inheriting"]
    return (
        np.where(kept, households["weight"] * chance, 0.0),
        np.where(kept, households["weight"] * (1.0 - chance), 0.0),
    )


def column(name: str) -> int:
    return distribution.value_column_names().index(name)


def row(tax: str) -> int:
    return distribution.TAX_ROWS.index(tax)


OVERALL = distribution.GROUPS.index((distribution.DIMENSION_OVERALL,) * 2)


def household_taxes(columns, at_death, world: str, corporate: np.ndarray):
    """Each tax per household in one reported world: (heir copy, non-heir copy).

    Each is shaped [tax, household] in `distribution.TAXES` order.
    """
    measures = columns.world_measures
    position = etr_measures.MEASURED_WORLDS.index(READS[world])
    cell = [name for name, _frame, _law in device_sums.EVALUATED_CELLS].index(
        READS[world]
    )
    during_life = [
        measures["income_tax"][position],
        measures["payroll_tax"][position],
    ]
    zeros = np.zeros(N_HOUSEHOLDS)
    heir_copy = np.stack(
        during_life
        + [at_death["estate_tax"][cell], at_death["deemed_gains_tax"][cell]]
        + [measures["wealth_tax"][position], corporate]
    )
    non_heir_copy = np.stack(
        during_life + [zeros, zeros] + [measures["wealth_tax"][position], corporate]
    )
    return heir_copy, non_heir_copy


@pytest.fixture(scope="module")
def year():
    columns, heir, at_death, balance_sheet = made_up_year()
    table = np.asarray(
        device_sums.distribution_table_for_year(
            columns, heir, at_death, corporate_dollars()
        )
    )
    return columns, heir, at_death, balance_sheet, table


def test_the_table_has_its_layout_shape_and_a_billionaire_group(year):
    columns, heir, _at_death, _balance_sheet, table = year
    assert table.shape == (
        len(SPREADS),
        len(distribution.TAX_ROWS),
        len(distribution.GROUPS),
        len(distribution.value_column_names()),
    )
    billionaires = distribution.GROUPS.index(
        (distribution.DIMENSION_NET_WORTH, distribution.LABEL_BILLIONAIRES)
    )
    heir_weight, non_heir_weight = kept_copy_weights(columns, heir)
    rich = columns.households["net_worth"] >= distribution.BILLIONAIRE_NET_WORTH_DOLLARS
    for spread in range(len(SPREADS)):
        assert_agrees_to_the_cent(
            table[spread, row("total"), billionaires, column("n_tax_units")],
            np.sum((heir_weight + non_heir_weight)[rich]),
            "the billionaire group counts every kept copy worth a billion dollars",
        )


def test_the_asset_spread_matches_a_hand_computation_of_its_three_legs(year):
    columns, _heir, _at_death, balance_sheet, _table = year
    households = columns.households
    counted = households["valid"] & ~households["is_dependent_return"]
    # The distribution table hands the spread only the counted households' bases.
    shares = corporate_spread.household_corporate_tax(
        distribution.CORPORATE_SPREAD_ASSETS,
        corporate_dollars(),
        households["weight"],
        {
            name: np.where(counted, base, 0.0)
            for name, base in columns.corporate_bases.items()
        },
    )
    current_law = host_corporate_tax(
        columns, balance_sheet, distribution.CORPORATE_SPREAD_ASSETS, "baseline"
    )
    assert_agrees_to_the_cent(
        shares.current_law,
        current_law,
        "each household's current-law corporate tax under the asset spread",
    )
    for world, change in (("static", "static_change"), ("conventional", "conventional_change")):
        assert_agrees_to_the_cent(
            getattr(shares, change),
            host_corporate_tax(
                columns, balance_sheet, distribution.CORPORATE_SPREAD_ASSETS, world
            )
            - current_law,
            f"each household's share of the {world} corporate change under the "
            "asset spread",
        )


@pytest.mark.parametrize("spread", SPREADS)
def test_each_world_s_corporate_tax_is_its_figure_spread_over_the_baseline_bases(
    year, spread
):
    columns, heir, _at_death, balance_sheet, table = year
    heir_weight, non_heir_weight = kept_copy_weights(columns, heir)
    corporate = table[SPREADS.index(spread), row("corporate"), OVERALL]
    for world in distribution.LEVEL_WORLDS:
        assert_agrees_to_the_cent(
            corporate[column(f"{world}_tax_dollars")],
            np.sum(
                (heir_weight + non_heir_weight)
                * host_corporate_tax(columns, balance_sheet, spread, world)
            ),
            f"the {world} world's corporate tax under the {spread} spread is its "
            "national figure spread over the baseline bases",
        )


def test_the_overall_corporate_tax_is_the_domestic_share_of_the_national_figure(year):
    """Households bear the whole national figure less the foreign-borne part
    of the two capital legs, under either spread."""
    *_, table = year
    s, f = NORMAL_RETURN_SHARE, FOREIGN_SHARE
    half = burden_jax.LABOR_SHARE_OF_NORMAL_RETURN
    rate_type_share = (1.0 - s) * (1.0 - f) + s * half + s * half * (1.0 - f)
    cost_recovery_share = half + half * (1.0 - f)
    expected = BASELINE_CORPORATE * rate_type_share + COST_RECOVERY * cost_recovery_share
    overall = [
        table[SPREADS.index(spread), row("corporate"), OVERALL, column("baseline_tax_dollars")]
        for spread in SPREADS
    ]
    for spread, dollars in zip(SPREADS, overall, strict=True):
        assert_agrees_to_the_cent(
            dollars,
            expected,
            f"the Overall baseline corporate tax under the {spread} spread is the "
            "domestic share of the national figure",
        )
    assert_agrees_to_the_cent(
        overall[0], overall[1], "the Overall corporate tax is the same under both spreads"
    )


@pytest.mark.parametrize("spread", SPREADS)
def test_the_overall_net_change_is_the_static_less_baseline_tax(year, spread):
    columns, heir, at_death, balance_sheet, table = year
    heir_weight, non_heir_weight = kept_copy_weights(columns, heir)
    current_law = host_corporate_tax(columns, balance_sheet, spread, "baseline")
    static = host_corporate_tax(columns, balance_sheet, spread, "static")
    baseline_heir, baseline_non_heir = household_taxes(
        columns, at_death, "baseline", current_law
    )
    static_heir, static_non_heir = household_taxes(columns, at_death, "static", static)
    exact = np.sum(heir_weight * (static_heir - baseline_heir)) + np.sum(
        non_heir_weight * (static_non_heir - baseline_non_heir)
    )
    overall = table[SPREADS.index(spread), row("total"), OVERALL]
    assert_agrees_to_the_cent(
        overall[column("static_tax_dollars")] - overall[column("baseline_tax_dollars")],
        exact,
        "the Overall static less baseline tax equals first_order less baseline "
        "tax plus the static corporate share",
    )
    rounded = np.sum(
        heir_weight * np.round(static_heir - baseline_heir)
    ) + np.sum(non_heir_weight * np.round(static_non_heir - baseline_non_heir))
    assert_agrees_to_the_cent(
        overall[column("net_change")],
        rounded,
        "the Overall net change adds up each copy's tax changes rounded to the dollar",
    )


def test_with_no_corporate_reform_the_corporate_row_does_not_change():
    columns, heir, at_death, balance_sheet = made_up_year(add_backs_are_zero=True)
    table = np.asarray(
        device_sums.distribution_table_for_year(
            columns,
            heir,
            at_death,
            corporate_dollars(static=BASELINE_CORPORATE, conventional=BASELINE_CORPORATE),
        )
    )
    heir_weight, non_heir_weight = kept_copy_weights(columns, heir)
    conventional = etr_measures.MEASURED_WORLDS.index("conventional")
    for spread in SPREADS:
        corporate = table[SPREADS.index(spread), row("corporate")]
        for world in ("static", "conventional"):
            assert_agrees_to_the_cent(
                corporate[:, column(f"{world}_tax_dollars")],
                corporate[:, column("baseline_tax_dollars")],
                f"with no corporate reform the {world} corporate tax is the "
                f"baseline's under the {spread} spread",
            )
        current_law = host_corporate_tax(columns, balance_sheet, spread, "baseline")
        for measure, name in (("cash", "expanded_income"), ("accrual", "haig_simons_income")):
            own = columns.world_measures[name][conventional]
            expected = np.sum(
                heir_weight * (own + heir["inheritance"] + current_law)
            ) + np.sum(non_heir_weight * (own + current_law))
            assert_agrees_to_the_cent(
                table[
                    SPREADS.index(spread),
                    row("total"),
                    OVERALL,
                    column(f"conventional_{measure}_income_dollars"),
                ],
                expected,
                f"with no add-backs, conventional pretax {measure} income is the "
                f"world's own income plus the bequest plus C_b under the {spread} "
                "spread",
            )


def _assert_rows_add_up(table: np.ndarray, name: str, tolerance: float) -> None:
    per_tax = table[1:, :, column(name)]
    total = table[row("total"), :, column(name)]
    finite = np.isfinite(total)
    np.testing.assert_array_equal(np.all(np.isfinite(per_tax), axis=0), finite)
    gap = np.abs(np.sum(per_tax[:, finite], axis=0) - total[finite])
    assert np.max(gap, initial=0.0) <= tolerance, (
        f"the per-tax rows of {name} differ from the total row by {np.max(gap)}"
    )


@pytest.mark.parametrize("spread", SPREADS)
def test_the_per_tax_rows_add_up_to_the_total_row(year, spread):
    *_, table = year
    one_version = table[SPREADS.index(spread)]
    _assert_rows_add_up(one_version, "average_tax_change", AVERAGE_ROUNDING_GAP)
    _assert_rows_add_up(one_version, "net_change", CENT)
    _assert_rows_add_up(one_version, "percent_change_in_after_tax_income", RATE_TOLERANCE)
    for world in distribution.LEVEL_WORLDS:
        _assert_rows_add_up(one_version, f"{world}_tax_dollars", CENT)
        for measure in distribution.INCOME_MEASURES:
            _assert_rows_add_up(one_version, f"{world}_{measure}_etr", RATE_TOLERANCE)
