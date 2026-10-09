"""Tests for the dollar-weighted average marginal tax rate table.

Every population here is small enough that the weighted products are written
out in the comments and checked against the number the code produces. The rate
values are made up; nothing here runs the tax calculator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.reporting import mtr_table
from taxmodel.taxes import marginal_rates
from tests.unit.reporting.host_ranking import quantile_groups


def _rates(**named: float | list[float]) -> dict[str, jnp.ndarray]:
    """Every rate column set to zero, then the named ones set to what is given.

    A test that cares about the wage rate under the reform's law says so by
    name and leaves the other thirty-one rates at zero, so the table's arithmetic
    on the rate it is testing is not hidden among the others.
    """
    n_records = 0
    for value in named.values():
        n_records = max(n_records, len(np.atleast_1d(value)))
    rates = {
        name: jnp.zeros(n_records, dtype=jnp.float64)
        for name in marginal_rates.MARGINAL_RATE_COLUMNS
    }
    for name, value in named.items():
        if name not in rates:
            raise AssertionError(f"{name} is not one of the stored rate columns")
        rates[name] = jnp.asarray(
            np.broadcast_to(np.asarray(value, dtype=np.float64), (n_records,))
        )
    return rates


def _frame(**columns: list[float]) -> dict[str, jnp.ndarray]:
    """A frame holding zero dollars in every column a priced dollar can move."""
    n_records = max(len(value) for value in columns.values())
    frame = {
        priced_dollar: jnp.zeros(n_records, dtype=jnp.float64)
        for priced_dollar in marginal_rates.PRICED_DOLLARS
    }
    for name, value in columns.items():
        frame[name] = jnp.asarray(np.asarray(value, dtype=np.float64))
    return frame


def _row(rows, *, group, priced_dollar, ending=""):
    """The one row of the table for a group, a priced dollar and a tax definition."""
    wanted = mtr_table.TAXES_INCLUDED_LABELS[ending]
    found = [
        row
        for row in rows
        if row["group"] == group
        and row["priced_dollar"] == priced_dollar
        and row["taxes_included"] == wanted
    ]
    assert len(found) == 1, f"expected one row, found {len(found)}"
    return found[0]


def _one_year(sums):
    """One year's sums with the year axis the host-side table expects."""
    return {name: jnp.asarray(value)[None, ...] for name, value in sums.items()}


def test_three_households_wage_rate_weighted_by_wage_dollars():
    """The table's central claim, with every product done by hand.

    Three households, all in one group, with wages and a marginal rate on the
    primary earner's wages:

        household   weight   wages    wage MTR
        A               2      100        0.10
        B               1      900        0.30
        C               5        0        0.40

    Weighted wage dollars: 2*100 + 1*900 + 5*0 = 1100.
    Weighted rate times dollars: 2*100*0.10 + 1*900*0.30 + 0 = 20 + 270 = 290.
    The rate on the average wage dollar is 290 / 1100 = 0.263636...

    The population-weighted average of the same three rates would be
    (2*0.10 + 1*0.30 + 5*0.40) / 8 = 2.5 / 8 = 0.3125, which is a different
    question -- household C faces the highest rate but earns no wages -- and the
    test checks the table does not report it.
    """
    frame = _frame(wages_base1=[100.0, 900.0, 0.0])
    rates = _rates(reform_law_mtr_wages_base1=[0.10, 0.30, 0.40])
    weight = jnp.asarray([2.0, 1.0, 5.0])
    labels = jnp.asarray([0, 0, 0])

    sums = mtr_table.mtr_table_sums(rates, frame, labels, weight, n_groups=1)
    rows = mtr_table.mtr_table(_one_year(sums), years=[2026])

    row = _row(rows, group="Group 1", priced_dollar="wages_base1")
    assert row["year"] == 2026
    assert row["base_dollars"] == pytest.approx(1100.0)
    assert row["average_marginal_rate_reform_law"] == pytest.approx(290.0 / 1100.0)
    assert row["average_marginal_rate_reform_law"] != pytest.approx(0.3125)


def test_baseline_and_reform_rates_differ_and_the_change_is_their_difference():
    """Both laws are reported off the same dollars, and the change is the gap.

    Two households, one group, weight 1 each:

        household   wages   baseline MTR   reform MTR
        A             200           0.20         0.25
        B             800           0.35         0.30

    Weighted wage dollars: 200 + 800 = 1000, the same under both laws because
    only the law changes between the two rate sets.
    Baseline: (200*0.20 + 800*0.35) / 1000 = (40 + 280) / 1000 = 0.32.
    Reform:   (200*0.25 + 800*0.30) / 1000 = (50 + 240) / 1000 = 0.29.
    The change is 0.29 - 0.32 = -0.03: a cut in the rate on the average wage
    dollar, even though the household with the lower wages saw its rate rise.
    """
    frame = _frame(wages_base1=[200.0, 800.0])
    rates = _rates(
        baseline_law_mtr_wages_base1=[0.20, 0.35],
        reform_law_mtr_wages_base1=[0.25, 0.30],
    )
    weight = jnp.ones(2)
    labels = jnp.asarray([0, 0])

    sums = mtr_table.mtr_table_sums(rates, frame, labels, weight, n_groups=1)
    rows = mtr_table.mtr_table(_one_year(sums))

    row = _row(rows, group="Group 1", priced_dollar="wages_base1")
    assert row["base_dollars"] == pytest.approx(1000.0)
    assert row["average_marginal_rate_baseline_law"] == pytest.approx(0.32)
    assert row["average_marginal_rate_reform_law"] == pytest.approx(0.29)
    assert row["change_in_average_marginal_rate"] == pytest.approx(-0.03)


def test_group_with_no_dollars_of_the_thing_reports_not_a_number():
    """A group that earns none of the priced dollars has no rate to report.

    Two groups. Group 1 holds a household with 500 dollars of rent; group 2
    holds a household with no rent at all but a marginal rate on rent of 0.40.
    Group 2's denominator is zero, so it reports not-a-number rather than
    dividing zero by zero or reporting a rate of zero, which a reader would
    take to mean rent is untaxed for that group.
    """
    frame = _frame(rent=[500.0, 0.0])
    rates = _rates(reform_law_mtr_rent=[0.25, 0.40])
    weight = jnp.ones(2)
    labels = jnp.asarray([0, 1])

    sums = mtr_table.mtr_table_sums(rates, frame, labels, weight, n_groups=2)
    rows = mtr_table.mtr_table(_one_year(sums))

    has_rent = _row(rows, group="Group 1", priced_dollar="rent")
    assert has_rent["average_marginal_rate_reform_law"] == pytest.approx(0.25)

    no_rent = _row(rows, group="Group 2", priced_dollar="rent")
    assert no_rent["base_dollars"] == 0.0
    assert np.isnan(no_rent["average_marginal_rate_reform_law"])
    assert np.isnan(no_rent["average_marginal_rate_baseline_law"])
    assert np.isnan(no_rent["change_in_average_marginal_rate"])


def test_a_rental_loss_brings_no_weight():
    """The flagged convention: a household with a loss earned no dollar.

    Two households in one group, weight 1 each. One has 1000 dollars of rent
    at a marginal rate of 0.30; the other has a rental loss of 400 dollars at a
    marginal rate of 0.10.

    Under the convention this table adopts, the loss is floored at zero: the
    denominator is 1000 and the reported rate is 0.30, the rate on the only
    rental dollars anyone in the group earned.

    Had the loss stayed in as a negative weight the denominator would have been
    1000 - 400 = 600 and the numerator 1000*0.30 - 400*0.10 = 260, giving 0.4333
    -- higher than either household's rate.
    """
    frame = _frame(rent=[1000.0, -400.0])
    rates = _rates(reform_law_mtr_rent=[0.30, 0.10])
    weight = jnp.ones(2)
    labels = jnp.asarray([0, 0])

    sums = mtr_table.mtr_table_sums(rates, frame, labels, weight, n_groups=1)
    rows = mtr_table.mtr_table(_one_year(sums))

    row = _row(rows, group="Group 1", priced_dollar="rent")
    assert row["base_dollars"] == pytest.approx(1000.0)
    assert row["average_marginal_rate_reform_law"] == pytest.approx(0.30)
    assert row["average_marginal_rate_reform_law"] != pytest.approx(260.0 / 600.0)


def test_the_general_partners_dollar_is_weighted_by_partnership_income_only():
    """``part_se1`` is the same dollar seen again, so it is not in the base.

    One household in one group, weight 1: 1000 dollars of ordinary partnership
    income on Schedule E, which the general-partner rate also treats as 1000
    dollars of self-employment earnings, and a rate of 0.44 on that dollar.

    The base is 1000, not 2000, so the reported rate is the household's own
    0.44. Summing both columns the priced dollar moves would have halved it.
    """
    assert marginal_rates.PRICED_DOLLARS["part_active"] == {
        "part_active": 1.0,
        "part_se1": 1.0,
    }
    frame = _frame(part_active=[1000.0])
    frame["part_se1"] = jnp.asarray([1000.0])
    rates = _rates(reform_law_mtr_part_active=[0.44])

    sums = mtr_table.mtr_table_sums(
        rates, frame, jnp.asarray([0]), jnp.ones(1), n_groups=1
    )
    rows = mtr_table.mtr_table(_one_year(sums))

    row = _row(rows, group="Group 1", priced_dollar="part_active")
    assert row["base_dollars"] == pytest.approx(1000.0)
    assert row["average_marginal_rate_reform_law"] == pytest.approx(0.44)


def test_ungrouped_records_are_kept_and_the_overall_row_holds_everybody():
    """Group rows plus the no-group row add up to Overall.

    Three households, weight 1 each, one group. The third has no group -- a
    label of -1, which is what a household with negative baseline income gets.

        household   group   wages   wage MTR
        A               0     100        0.10
        B               0     300        0.20
        C              -1     600        0.50

    Group 1: (100*0.10 + 300*0.20) / 400 = 70 / 400 = 0.175.
    No group: 600*0.50 / 600 = 0.50.
    Overall: (10 + 60 + 300) / (100 + 300 + 600) = 370 / 1000 = 0.37.
    """
    frame = _frame(wages_base1=[100.0, 300.0, 600.0])
    rates = _rates(reform_law_mtr_wages_base1=[0.10, 0.20, 0.50])
    labels = jnp.asarray([0, 0, -1])

    sums = mtr_table.mtr_table_sums(rates, frame, labels, jnp.ones(3), n_groups=1)
    rows = mtr_table.mtr_table(_one_year(sums))

    grouped = _row(rows, group="Group 1", priced_dollar="wages_base1")
    ungrouped = _row(rows, group=mtr_table.NO_GROUP_LABEL, priced_dollar="wages_base1")
    overall = _row(rows, group=mtr_table.OVERALL_LABEL, priced_dollar="wages_base1")

    assert grouped["base_dollars"] == pytest.approx(400.0)
    assert grouped["average_marginal_rate_reform_law"] == pytest.approx(0.175)
    assert ungrouped["base_dollars"] == pytest.approx(600.0)
    assert ungrouped["average_marginal_rate_reform_law"] == pytest.approx(0.50)
    assert overall["base_dollars"] == pytest.approx(1000.0)
    assert overall["average_marginal_rate_reform_law"] == pytest.approx(0.37)
    assert overall["base_dollars"] == pytest.approx(
        grouped["base_dollars"] + ungrouped["base_dollars"]
    )


def test_an_empty_slot_carries_no_weight_and_changes_nothing():
    """A slot holding no household this year has a weight of zero.

    Two live households and one padding slot whose weight is zero and whose
    wage figure is left-over placeholder data. The padding slot has no group,
    so it lands in the no-group column, where its zero weight makes it
    invisible: the no-group row has no dollars and reports not-a-number.
    """
    frame = _frame(wages_base1=[100.0, 100.0, 999_999.0])
    rates = _rates(reform_law_mtr_wages_base1=[0.20, 0.40, 0.99])
    weight = jnp.asarray([1.0, 1.0, 0.0])
    labels = jnp.asarray([0, 0, -1])

    sums = mtr_table.mtr_table_sums(rates, frame, labels, weight, n_groups=1)
    rows = mtr_table.mtr_table(_one_year(sums))

    grouped = _row(rows, group="Group 1", priced_dollar="wages_base1")
    overall = _row(rows, group=mtr_table.OVERALL_LABEL, priced_dollar="wages_base1")
    ungrouped = _row(rows, group=mtr_table.NO_GROUP_LABEL, priced_dollar="wages_base1")

    assert grouped["average_marginal_rate_reform_law"] == pytest.approx(0.30)
    assert overall["average_marginal_rate_reform_law"] == pytest.approx(0.30)
    assert np.isnan(ungrouped["average_marginal_rate_reform_law"])


def test_two_years_add_up_before_the_division():
    """Adding the two accumulators across years and dividing once is exact.

    One group, one household, weight 1. Its wages and marginal rate differ
    between the two years:

        year   wages   wage MTR   dollars   rate times dollars
        1        100       0.10       100                   10
        2        900       0.30       900                  270

    Each year on its own reports its own rate, 0.10 and 0.30. The two years
    taken together report 280 / 1000 = 0.28 -- the rate on the average wage
    dollar earned across the two years, which is not the average of 0.10 and
    0.30 (0.20), because the second year has nine times the dollars.
    """
    first = mtr_table.mtr_table_sums(
        _rates(reform_law_mtr_wages_base1=[0.10]),
        _frame(wages_base1=[100.0]),
        jnp.asarray([0]),
        jnp.ones(1),
        n_groups=1,
    )
    second = mtr_table.mtr_table_sums(
        _rates(reform_law_mtr_wages_base1=[0.30]),
        _frame(wages_base1=[900.0]),
        jnp.asarray([0]),
        jnp.ones(1),
        n_groups=1,
    )
    # What a scan over two years hands back: each accumulator stacked on a
    # leading year axis.
    stacked = {name: jnp.stack([first[name], second[name]]) for name in first}

    per_year = mtr_table.mtr_table(stacked, years=[2026, 2027])
    assert _row(
        [row for row in per_year if row["year"] == 2026],
        group="Group 1",
        priced_dollar="wages_base1",
    )["average_marginal_rate_reform_law"] == pytest.approx(0.10)
    assert _row(
        [row for row in per_year if row["year"] == 2027],
        group="Group 1",
        priced_dollar="wages_base1",
    )["average_marginal_rate_reform_law"] == pytest.approx(0.30)

    both_years = {
        name: jnp.sum(value, axis=0, keepdims=True) for name, value in stacked.items()
    }
    combined = mtr_table.mtr_table(both_years)
    row = _row(combined, group="Group 1", priced_dollar="wages_base1")
    assert row["base_dollars"] == pytest.approx(1000.0)
    assert row["average_marginal_rate_reform_law"] == pytest.approx(0.28)
    assert row["average_marginal_rate_reform_law"] != pytest.approx(0.20)


def test_group_labels_come_from_ranking_baseline_income():
    """The seam: labels are quantile groups of baseline income, made outside.

    Four households of equal weight, baseline incomes 0, 10, 20 and 30 dollars,
    cut into two groups. A household with exactly zero income is ranked, which
    is the reporting convention, so the bottom half is the households with 0 and
    10 dollars of income.

    Each household's wages and wage rate:

        income   wages   wage MTR   group
             0     100        0.10       bottom half
            10     300        0.20       bottom half
            20     500        0.30       top half
            30     700        0.40       top half

    Bottom half: (100*0.10 + 300*0.20) / 400 = 70 / 400 = 0.175.
    Top half:    (500*0.30 + 700*0.40) / 1200 = 430 / 1200 = 0.358333...
    """
    weight = np.ones(4)
    baseline_income = np.array([0.0, 10.0, 20.0, 30.0])
    labels = quantile_groups(baseline_income, weight, 2)
    assert list(labels) == [0, 0, 1, 1]

    frame = _frame(wages_base1=[100.0, 300.0, 500.0, 700.0])
    rates = _rates(reform_law_mtr_wages_base1=[0.10, 0.20, 0.30, 0.40])

    sums = mtr_table.mtr_table_sums(
        rates, frame, jnp.asarray(labels), jnp.asarray(weight), n_groups=2
    )
    rows = mtr_table.mtr_table(
        _one_year(sums), years=[2030], group_names=("Bottom half", "Top half")
    )

    bottom = _row(rows, group="Bottom half", priced_dollar="wages_base1")
    top = _row(rows, group="Top half", priced_dollar="wages_base1")
    assert bottom["average_marginal_rate_reform_law"] == pytest.approx(0.175)
    assert top["average_marginal_rate_reform_law"] == pytest.approx(430.0 / 1200.0)


def test_every_priced_dollar_and_both_tax_definitions_get_a_row():
    """The table reports all eight priced dollars under both price definitions."""
    frame = _frame(wages_base1=[100.0])
    rates = _rates(reform_law_mtr_wages_base1=[0.2])
    sums = mtr_table.mtr_table_sums(
        rates, frame, jnp.asarray([0]), jnp.ones(1), n_groups=3
    )
    rows = mtr_table.mtr_table(_one_year(sums))

    n_priced_dollars = len(marginal_rates.PRICED_DOLLARS)
    n_price_definitions = len(marginal_rates.PRICES)
    # Three groups, the no-group column, and the Overall row.
    n_reported_groups = 3 + 2
    assert len(rows) == n_reported_groups * n_priced_dollars * n_price_definitions
    assert {row["priced_dollar"] for row in rows} == set(marginal_rates.PRICED_DOLLARS)
    assert {row["taxes_included"] for row in rows} == set(
        mtr_table.TAXES_INCLUDED_LABELS.values()
    )
    assert set(rows[0]) == set(mtr_table.TABLE_COLUMNS)


def test_the_payroll_price_is_reported_separately_from_the_income_tax_price():
    """The two definitions of which taxes are in the price share one denominator.

    One household, one group, weight 1, 1000 dollars of wages. The income tax
    alone prices the next wage dollar at 0.22; adding the employee's payroll tax
    -- the employee's half of FICA and the additional Medicare tax -- brings it
    to 0.2965. Both are weighted by the same 1000 wage dollars.
    """
    frame = _frame(wages_base1=[1000.0])
    rates = _rates(
        reform_law_mtr_wages_base1=[0.22],
        reform_law_mtr_wages_base1_plus_employee_payroll=[0.2965],
    )
    sums = mtr_table.mtr_table_sums(
        rates, frame, jnp.asarray([0]), jnp.ones(1), n_groups=1
    )
    rows = mtr_table.mtr_table(_one_year(sums))

    income_tax_only = _row(rows, group="Group 1", priced_dollar="wages_base1")
    with_payroll = _row(
        rows,
        group="Group 1",
        priced_dollar="wages_base1",
        ending="_plus_employee_payroll",
    )
    assert income_tax_only["average_marginal_rate_reform_law"] == pytest.approx(0.22)
    assert with_payroll["average_marginal_rate_reform_law"] == pytest.approx(0.2965)
    assert with_payroll["base_dollars"] == income_tax_only["base_dollars"]


def test_the_reduction_runs_inside_a_compiled_program():
    """The reduction traces, which is what lets it live in the year loop.

    The group labels and the number of groups are fixed before compiling, as the
    wiring contract requires: the labels are a plain array argument and the group
    count is a Python integer baked into the shapes.
    """
    frame = _frame(wages_base1=[100.0, 900.0])
    rates = _rates(reform_law_mtr_wages_base1=[0.10, 0.30])
    labels = jnp.asarray([0, 1])

    compiled = jax.jit(
        lambda rate_values, records, weight: mtr_table.mtr_table_sums(
            rate_values, records, labels, weight, n_groups=2
        )
    )
    sums = compiled(rates, frame, jnp.ones(2))
    rows = mtr_table.mtr_table(_one_year(sums))

    assert _row(rows, group="Group 1", priced_dollar="wages_base1")[
        "average_marginal_rate_reform_law"
    ] == pytest.approx(0.10)
    assert _row(rows, group="Group 2", priced_dollar="wages_base1")[
        "average_marginal_rate_reform_law"
    ] == pytest.approx(0.30)


def test_misshapen_or_mislabelled_input_is_refused():
    """The host-side table says what is wrong rather than reporting a wrong number."""
    frame = _frame(wages_base1=[100.0])
    rates = _rates(reform_law_mtr_wages_base1=[0.2])
    sums = mtr_table.mtr_table_sums(
        rates, frame, jnp.asarray([0]), jnp.ones(1), n_groups=2
    )

    with pytest.raises(ValueError, match="years, groups"):
        mtr_table.mtr_table(sums)
    with pytest.raises(ValueError, match="2 years"):
        mtr_table.mtr_table(_one_year(sums), years=[2026, 2027])
    with pytest.raises(ValueError, match="names for them"):
        mtr_table.mtr_table(_one_year(sums), group_names=("only one",))
    with pytest.raises(ValueError, match="at least one group"):
        mtr_table.mtr_table_sums(rates, frame, jnp.asarray([0]), jnp.ones(1), 0)
