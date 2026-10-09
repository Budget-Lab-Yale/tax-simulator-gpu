"""Synthetic tests for the mechanical wealth bathtub's kernels.

Every population here is a dozen records or fewer with round numbers, so the
expected answers can be worked out on paper.  The percentile count is 4 rather
than the model's 100 for the same reason; it is a configuration value
everywhere, never a constant in the code.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.years.wealth_arithmetic import (
    SINKHORN_TOL,
    WEALTH_DYN_AGE_MAX,
    WEALTH_DYN_AGE_MIN,
    WEALTH_DYN_G_FLOOR,
    ProfileSpec,
    age_grid,
    apply_haircut,
    apply_percentile_transition,
    assign_within_age_bin,
    build_aging_matrix,
    cell_rows,
    cohort_recurrence_step,
    growth_factor,
    haircut_fractions,
    load_saving_shares,
    load_transition_matrix,
    parse_profile_spec,
    rake_to_doubly_stochastic,
    resolve_profile,
    sum_by_cell,
    within_age_cutoffs,
)

# A three-age grid and four percentiles, small enough to check by hand.
AGES = np.array([40, 41, 42])
N_BINS = 4

# The capital share of business income under the model's 20/80 capital-labor
# split; config/scenarios/economy/default/wealth.yaml, cap_flows_pt_weight.
PASS_THROUGH_WEIGHT = 0.2


# -----------------------------------------------------------------------------
# The age grid
# -----------------------------------------------------------------------------


def test_the_age_grid_runs_from_18_to_the_tax_data_top_code():
    assert WEALTH_DYN_AGE_MIN == 18
    assert WEALTH_DYN_AGE_MAX == 80
    grid = age_grid()
    assert grid[0] == 18
    assert grid[-1] == 80
    assert len(grid) == 63


# -----------------------------------------------------------------------------
# Percentile cut points
# -----------------------------------------------------------------------------


def test_the_cut_point_is_the_first_value_whose_cumulative_weight_reaches_the_share():
    # Age 40 holds four records of equal weight worth 10, 20, 30 and 40.  The
    # cumulative shares of weight are 0.25, 0.50, 0.75 and 1.00, so the cut at
    # 0.25 is 10, the cut at 0.50 is 20 and the cut at 0.75 is 30.
    value = jnp.array([10.0, 20.0, 30.0, 40.0])
    weight = jnp.ones(4)
    age = jnp.array([40, 40, 40, 40])

    cuts = within_age_cutoffs(value, weight, age, AGES, N_BINS)

    np.testing.assert_allclose(cuts[0], [10.0, 20.0, 30.0])


def test_a_heavy_record_can_own_two_cut_points():
    # Values 10, 20 and 30 with weights 1, 2 and 1: cumulative shares 0.25, 0.75
    # and 1.00.  The 0.25 cut is 10.  The 0.50 and 0.75 cuts are both 20, since
    # 20 is the first value whose cumulative share reaches either.  A cumulative
    # share sitting exactly on the probability counts as reaching it.
    value = jnp.array([10.0, 20.0, 30.0])
    weight = jnp.array([1.0, 2.0, 1.0])
    age = jnp.array([40, 40, 40])

    cuts = within_age_cutoffs(value, weight, age, AGES, N_BINS)

    np.testing.assert_allclose(cuts[0], [10.0, 20.0, 20.0])


def test_an_age_with_no_eligible_records_gets_infinite_cut_points():
    # Age 41 holds one record with no weight and age 42 holds nobody at all.
    value = jnp.array([10.0, 20.0, 30.0, 40.0, 500.0])
    weight = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0])
    age = jnp.array([40, 40, 40, 40, 41])

    cuts = within_age_cutoffs(value, weight, age, AGES, N_BINS)

    assert np.all(np.isinf(np.asarray(cuts[1])))
    assert np.all(np.isinf(np.asarray(cuts[2])))


def test_records_at_or_below_zero_and_records_with_no_weight_do_not_set_cut_points():
    # The three records with wealth are the same as in the first test's bottom
    # three; the extra records -- no net worth, negative net worth, no weight,
    # and a missing value -- must leave the cut points exactly where the
    # eligible records alone would put them.
    value = jnp.array([10.0, 20.0, 30.0, 40.0, 0.0, -500.0, 999.0, jnp.nan])
    weight = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    age = jnp.full(8, 40)

    cuts = within_age_cutoffs(value, weight, age, AGES, N_BINS)

    np.testing.assert_allclose(cuts[0], [10.0, 20.0, 30.0])


def test_cut_points_are_found_within_each_age_separately():
    value = jnp.array([10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0])
    weight = jnp.ones(8)
    age = jnp.array([40, 40, 40, 40, 41, 41, 41, 41])

    cuts = within_age_cutoffs(value, weight, age, AGES, N_BINS)

    np.testing.assert_allclose(cuts[0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(cuts[1], [100.0, 200.0, 300.0])


# -----------------------------------------------------------------------------
# Percentile assignment
# -----------------------------------------------------------------------------


def test_a_record_exactly_on_a_boundary_falls_in_the_lower_percentile():
    # Cut points 10, 20 and 30 split age 40 into four percentiles.  A record
    # worth exactly 20 has two cut points at or below it, so it is in the third
    # percentile (index 2), not the fourth.
    cutoffs = jnp.array([[10.0, 20.0, 30.0]] * 3)
    value = jnp.array([5.0, 10.0, 15.0, 20.0, 30.0, 45.0])
    age = jnp.full(6, 40)

    bin_index, has_cell = assign_within_age_bin(value, age, cutoffs, AGES, N_BINS)

    np.testing.assert_array_equal(bin_index, [0, 1, 1, 2, 3, 3])
    assert bool(jnp.all(has_cell))


def test_records_with_zero_negative_or_missing_net_worth_get_no_cell():
    cutoffs = jnp.array([[10.0, 20.0, 30.0]] * 3)
    value = jnp.array([25.0, 0.0, -100.0, jnp.nan, jnp.inf])
    age = jnp.full(5, 40)

    bin_index, has_cell = assign_within_age_bin(value, age, cutoffs, AGES, N_BINS)

    np.testing.assert_array_equal(has_cell, [True, False, False, False, False])
    np.testing.assert_array_equal(bin_index, [2, -1, -1, -1, -1])


def test_a_record_off_the_age_grid_gets_no_cell():
    cutoffs = jnp.array([[10.0, 20.0, 30.0]] * 3)
    value = jnp.array([25.0, 25.0, 25.0])
    age = jnp.array([39, 40, 43])

    bin_index, has_cell = assign_within_age_bin(value, age, cutoffs, AGES, N_BINS)

    np.testing.assert_array_equal(has_cell, [False, True, False])
    np.testing.assert_array_equal(bin_index, [-1, 2, -1])


def test_every_record_in_an_empty_age_lands_in_the_bottom_percentile():
    # Infinite cut points mean no record is above any boundary.
    cutoffs = jnp.array([[10.0, 20.0, 30.0], [np.inf] * 3, [np.inf] * 3])
    value = jnp.array([1e12, 5.0])
    age = jnp.array([41, 41])

    bin_index, _ = assign_within_age_bin(value, age, cutoffs, AGES, N_BINS)

    np.testing.assert_array_equal(bin_index, [0, 0])


def test_percentile_assignment_matches_counting_the_cut_points_one_by_one():
    # The assignment walks each age's row of cut points by halving rather than
    # reading the whole row, because the whole row per record would not fit in
    # memory at the model's size.  This checks it against the plain count.
    rng = np.random.default_rng(11)
    n_bins = 100
    cutoffs = np.sort(rng.normal(size=(len(AGES), n_bins - 1)) * 1e5, axis=1)
    value = rng.normal(size=400) * 1e5
    age = rng.choice(AGES, size=400)

    bin_index, has_cell = assign_within_age_bin(
        jnp.asarray(value), jnp.asarray(age), jnp.asarray(cutoffs), AGES, n_bins
    )

    rows = np.asarray(age) - AGES[0]
    expected = (cutoffs[rows] <= value[:, None]).sum(axis=1)
    expected = np.where(value > 0, np.minimum(expected, n_bins - 1), -1)
    np.testing.assert_array_equal(np.asarray(bin_index), expected)
    np.testing.assert_array_equal(np.asarray(has_cell), value > 0)


# -----------------------------------------------------------------------------
# Raking the transition matrix
# -----------------------------------------------------------------------------


def test_raked_rows_sum_to_one_and_columns_come_within_tolerance():
    rng = np.random.default_rng(3)
    matrix = jnp.asarray(rng.random((20, 20)) + 0.01)

    raked = rake_to_doubly_stochastic(matrix)

    # The last thing raking does is normalize rows, so the rows are exact to
    # floating point and the columns only to the raking tolerance.  That
    # asymmetry is deliberate: rows summing to one is what conserves dollars.
    np.testing.assert_allclose(np.asarray(raked.sum(axis=1)), 1.0, atol=4e-16)
    assert float(jnp.max(jnp.abs(raked.sum(axis=0) - 1.0))) < SINKHORN_TOL


def test_raking_leaves_the_identity_and_a_uniform_matrix_alone():
    identity = jnp.eye(5)
    uniform = jnp.full((5, 5), 7.0)

    np.testing.assert_allclose(
        np.asarray(rake_to_doubly_stochastic(identity)), np.eye(5)
    )
    np.testing.assert_allclose(
        np.asarray(rake_to_doubly_stochastic(uniform)), np.full((5, 5), 0.2)
    )


def test_a_matrix_with_an_empty_row_is_seeded_before_raking():
    # A percentile nobody moves into is an empty row, which cannot be scaled to
    # sum to one.  Seeding the whole matrix with a tiny amount first keeps every
    # number finite instead of dividing by zero.
    #
    # This particular matrix -- the identity with one row emptied -- is degenerate
    # enough that the column sums are still 7e-4 away from one when the pass cap
    # runs out.  The rows are
    # normalized last regardless, so the dollars are still conserved.
    matrix = jnp.asarray(np.eye(4))
    matrix = matrix.at[2].set(0.0)

    raked = rake_to_doubly_stochastic(matrix)

    assert np.all(np.isfinite(np.asarray(raked)))
    np.testing.assert_allclose(np.asarray(raked.sum(axis=1)), 1.0, atol=4e-16)


# -----------------------------------------------------------------------------
# Stepping the cells forward
# -----------------------------------------------------------------------------


def _cells(rows):
    return jnp.asarray(np.array(rows, dtype=float))


def test_the_identity_transition_leaves_every_percentile_where_it_was():
    cells = _cells(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )

    np.testing.assert_allclose(
        np.asarray(apply_percentile_transition(cells, jnp.eye(N_BINS))),
        np.asarray(cells),
    )


def test_people_and_dollars_survive_the_transition_and_the_aging():
    # With no growth and no inflow, the total across all cells has to be exactly
    # what it was: the shortfall moves between cells and none of it is created
    # or destroyed.
    rng = np.random.default_rng(7)
    cells = jnp.asarray(rng.random((len(AGES), N_BINS)) * 1000.0)
    transition = rake_to_doubly_stochastic(
        jnp.asarray(rng.random((N_BINS, N_BINS)) + 0.1)
    )
    aging = build_aging_matrix(len(AGES))

    stepped = cohort_recurrence_step(
        cells_last_year=cells,
        growth=jnp.ones_like(cells),
        inflow=jnp.zeros_like(cells),
        aging=aging,
        transition=transition,
    )

    np.testing.assert_allclose(float(stepped.sum()), float(cells.sum()), rtol=1e-12)


def test_the_top_age_keeps_what_it_had_and_takes_in_the_age_below():
    # Age 42 is the top of this grid, so it is an absorbing pool: it gathers age
    # 41 and holds on to itself.  Age 40, the youngest, starts each year empty.
    cells = _cells(
        [[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0], [30.0, 0.0, 0.0, 0.0]]
    )
    aging = build_aging_matrix(len(AGES))

    stepped = cohort_recurrence_step(
        cells_last_year=cells,
        growth=jnp.ones_like(cells),
        inflow=jnp.zeros_like(cells),
        aging=aging,
        transition=jnp.eye(N_BINS),
    )

    np.testing.assert_allclose(np.asarray(stepped[:, 0]), [0.0, 10.0, 50.0])


def test_this_years_inflow_enters_without_growing():
    # Nothing carried in, growth of 2, and an inflow of 5.  The answer is 5: if
    # the inflow grew in its first year it would be 10.
    zero = jnp.zeros((len(AGES), N_BINS))
    inflow = jnp.full((len(AGES), N_BINS), 5.0)

    stepped = cohort_recurrence_step(
        cells_last_year=zero,
        growth=jnp.full((len(AGES), N_BINS), 2.0),
        inflow=inflow,
        aging=build_aging_matrix(len(AGES)),
        transition=jnp.eye(N_BINS),
    )

    np.testing.assert_allclose(np.asarray(stepped), np.full((len(AGES), N_BINS), 5.0))
    # The same inflow does grow in its second year, so the test above is not
    # accidentally checking a growth factor of one.
    second = cohort_recurrence_step(
        cells_last_year=stepped,
        growth=jnp.full((len(AGES), N_BINS), 2.0),
        inflow=jnp.zeros_like(inflow),
        aging=build_aging_matrix(len(AGES)),
        transition=jnp.eye(N_BINS),
    )
    np.testing.assert_allclose(np.asarray(second[1]), np.full(N_BINS, 10.0))


def test_no_saving_means_no_shortfall_however_large_the_tax_change():
    # The inflow is the saving share times the change in tax.  A profile of all
    # zeros keeps the state at zero forever, whatever the tax change is.
    saving_share = jnp.zeros((len(AGES), N_BINS))
    tax_change = jnp.full((len(AGES), N_BINS), 1e9)
    cells = jnp.zeros((len(AGES), N_BINS))
    aging = build_aging_matrix(len(AGES))

    for _ in range(5):
        cells = cohort_recurrence_step(
            cells_last_year=cells,
            growth=jnp.full((len(AGES), N_BINS), 1.05),
            inflow=saving_share * tax_change,
            aging=aging,
            transition=jnp.eye(N_BINS),
        )

    np.testing.assert_array_equal(np.asarray(cells), np.zeros((len(AGES), N_BINS)))


# -----------------------------------------------------------------------------
# The growth factor
# -----------------------------------------------------------------------------


def test_the_shortfall_grows_at_the_plain_return():
    # A 5% return gives 1.05. The tax the eroded wealth no longer pays is
    # accounted for in the calculators, which every year see a balance sheet
    # the carried shortfall has already been taken out of.
    factor = growth_factor(r_total=0.05, saving_share=jnp.array([0.5]))

    np.testing.assert_allclose(np.asarray(factor), [1.05])


def test_one_growth_factor_comes_back_for_every_cell():
    # The saving-share surface fixes the age-by-percentile shape that the
    # wealth diagnostics record one growth factor per cell in.
    factor = growth_factor(
        r_total=0.04, saving_share=jnp.zeros((len(AGES), N_BINS))
    )

    assert np.asarray(factor).shape == (len(AGES), N_BINS)
    np.testing.assert_allclose(np.asarray(factor), 1.04)


def test_a_return_below_minus_one_is_held_at_the_floor():
    # A growth factor cannot go negative. The floor binds only on an economy whose total return is below -100 percent, which no
    # configuration ships, so this is a guard rather than a case.
    factor = growth_factor(r_total=-2.0, saving_share=jnp.array([1.0]))

    np.testing.assert_allclose(np.asarray(factor), [WEALTH_DYN_G_FLOOR])


# -----------------------------------------------------------------------------
# Totalling records into cells
# -----------------------------------------------------------------------------


def test_records_with_no_cell_contribute_nothing_to_any_cell():
    values = jnp.array([100.0, 200.0, 999.0])
    row = jnp.array([0, 0, 1])
    bin_index = jnp.array([2, 2, -1])
    has_cell = jnp.array([True, True, False])

    totals = sum_by_cell(values, row, bin_index, has_cell, len(AGES), N_BINS)

    assert float(totals[0, 2]) == 300.0
    assert float(totals.sum()) == 300.0


# -----------------------------------------------------------------------------
# The haircut
# -----------------------------------------------------------------------------

# One cell, age 40 percentile 2 (index 1), holding two records:
#
#   record A: net worth 100, weight 2, gross assets 200, debts 100
#   record B: net worth 300, weight 1, gross assets 300, no debts
#
# The cell's total weighted positive net worth is 2 * 100 + 1 * 300 = 500.
HAIRCUT_NET_WORTH = jnp.array([100.0, 300.0])
HAIRCUT_WEIGHT = jnp.array([2.0, 1.0])
HAIRCUT_GROSS = jnp.array([200.0, 300.0])
HAIRCUT_ROW = jnp.array([0, 0])
HAIRCUT_BIN = jnp.array([1, 1])
HAIRCUT_HAS_CELL = jnp.array([True, True])


def _shortfall_of(amount):
    return jnp.zeros((len(AGES), N_BINS)).at[0, 1].set(amount)


def test_the_shortfall_is_split_in_proportion_to_positive_net_worth():
    # A shortfall of $50 over a cell holding $500 of weighted net worth:
    #   A gets 50 * 100 / 500 = 10, and 10 / 200 of assets is a 5% haircut
    #   B gets 50 * 300 / 500 = 30, and 30 / 300 of assets is a 10% haircut
    allocated, fraction = haircut_fractions(
        cell_shortfall=_shortfall_of(50.0),
        row=HAIRCUT_ROW,
        bin_index=HAIRCUT_BIN,
        has_cell=HAIRCUT_HAS_CELL,
        net_worth=HAIRCUT_NET_WORTH,
        weight=HAIRCUT_WEIGHT,
        gross_assets=HAIRCUT_GROSS,
        fmax=0.9,
    )

    np.testing.assert_allclose(np.asarray(allocated), [10.0, 30.0])
    np.testing.assert_allclose(np.asarray(fraction), [0.05, 0.10])


def test_the_whole_cell_shortfall_is_allocated_and_no_more():
    allocated, _ = haircut_fractions(
        cell_shortfall=_shortfall_of(50.0),
        row=HAIRCUT_ROW,
        bin_index=HAIRCUT_BIN,
        has_cell=HAIRCUT_HAS_CELL,
        net_worth=HAIRCUT_NET_WORTH,
        weight=HAIRCUT_WEIGHT,
        gross_assets=HAIRCUT_GROSS,
        fmax=0.9,
    )

    np.testing.assert_allclose(float((HAIRCUT_WEIGHT * allocated).sum()), 50.0)


def test_a_tax_cut_scales_the_balance_sheet_up():
    # The shortfall is negative when the reform cuts taxes: the household saves
    # the money instead of drawing wealth down, so the fractions are negative
    # and assets are scaled up.
    allocated, fraction = haircut_fractions(
        cell_shortfall=_shortfall_of(-50.0),
        row=HAIRCUT_ROW,
        bin_index=HAIRCUT_BIN,
        has_cell=HAIRCUT_HAS_CELL,
        net_worth=HAIRCUT_NET_WORTH,
        weight=HAIRCUT_WEIGHT,
        gross_assets=HAIRCUT_GROSS,
        fmax=0.9,
    )

    np.testing.assert_allclose(np.asarray(allocated), [-10.0, -30.0])
    np.testing.assert_allclose(np.asarray(fraction), [-0.05, -0.10])


def test_the_haircut_fraction_is_clamped_the_same_amount_in_both_directions():
    # With fmax at 5%, record B's 10% haircut is clamped to 5%, and its 10%
    # scale-up under a tax cut is clamped to 5% as well.
    drawdown = haircut_fractions(
        cell_shortfall=_shortfall_of(50.0),
        row=HAIRCUT_ROW,
        bin_index=HAIRCUT_BIN,
        has_cell=HAIRCUT_HAS_CELL,
        net_worth=HAIRCUT_NET_WORTH,
        weight=HAIRCUT_WEIGHT,
        gross_assets=HAIRCUT_GROSS,
        fmax=0.05,
    )[1]
    buildup = haircut_fractions(
        cell_shortfall=_shortfall_of(-50.0),
        row=HAIRCUT_ROW,
        bin_index=HAIRCUT_BIN,
        has_cell=HAIRCUT_HAS_CELL,
        net_worth=HAIRCUT_NET_WORTH,
        weight=HAIRCUT_WEIGHT,
        gross_assets=HAIRCUT_GROSS,
        fmax=0.05,
    )[1]

    np.testing.assert_allclose(np.asarray(drawdown), [0.05, 0.05])
    np.testing.assert_allclose(np.asarray(buildup), [-0.05, -0.05])


def test_a_record_with_no_cell_and_a_cell_with_no_net_worth_are_left_alone():
    # Three records: one in a cell that holds a shortfall, one with no cell at
    # all, and one alone in a cell whose only net worth is negative.
    shortfall = jnp.zeros((len(AGES), N_BINS)).at[0, 1].set(50.0).at[1, 0].set(50.0)
    allocated, fraction = haircut_fractions(
        cell_shortfall=shortfall,
        row=jnp.array([0, 0, 1]),
        bin_index=jnp.array([1, -1, 0]),
        has_cell=jnp.array([True, False, True]),
        net_worth=jnp.array([100.0, 100.0, -100.0]),
        weight=jnp.array([1.0, 1.0, 1.0]),
        gross_assets=jnp.array([200.0, 200.0, 200.0]),
        fmax=0.9,
    )

    np.testing.assert_allclose(np.asarray(allocated), [50.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(fraction), [0.25, 0.0, 0.0])


def test_a_record_with_no_weight_takes_no_dollars_out_of_its_cell():
    # A weightless record adds nothing to its cell's net worth, so a cell
    # holding only weightless records allocates nothing, and a weightless
    # record sharing a cell does not dilute the record that is really there.
    shortfall = jnp.zeros((len(AGES), N_BINS)).at[0, 1].set(50.0).at[1, 1].set(50.0)
    allocated, _ = haircut_fractions(
        cell_shortfall=shortfall,
        row=jnp.array([0, 0, 1]),
        bin_index=jnp.array([1, 1, 1]),
        has_cell=jnp.array([True, True, True]),
        net_worth=jnp.array([100.0, 100.0, 100.0]),
        weight=jnp.array([1.0, 0.0, 0.0]),
        gross_assets=jnp.array([200.0, 200.0, 200.0]),
        fmax=0.9,
    )

    np.testing.assert_allclose(np.asarray(allocated), [50.0, 50.0, 0.0])
    # Only the weighted record's allocation is real dollars, and it is the whole
    # cell shortfall.
    weighted = np.asarray(allocated) * np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(weighted.sum(), 50.0)


def test_a_record_with_no_assets_gets_no_haircut_rather_than_a_division_by_zero():
    _, fraction = haircut_fractions(
        cell_shortfall=_shortfall_of(50.0),
        row=jnp.array([0]),
        bin_index=jnp.array([1]),
        has_cell=jnp.array([True]),
        net_worth=jnp.array([100.0]),
        weight=jnp.array([1.0]),
        gross_assets=jnp.array([0.0]),
        fmax=0.9,
    )

    np.testing.assert_allclose(np.asarray(fraction), [0.0])


def test_assets_scale_by_one_minus_f_and_pass_through_flows_by_the_capital_share():
    # A 5% haircut takes 5% off assets, off pure capital income, and off
    # retirement distributions, but only 0.2 * 5% = 1% off pass-through income,
    # because only the capital share of business income is a return to the
    # assets that shrank.
    fraction = jnp.array([0.05])
    scaled = apply_haircut(
        assets=jnp.array([[200.0]]),
        debts=jnp.array([[100.0]]),
        pure_capital_flows=jnp.array([[1000.0]]),
        pass_through_flows=jnp.array([[1000.0]]),
        retirement_distributions=jnp.array([[1000.0]]),
        fraction=fraction,
        pass_through_weight=PASS_THROUGH_WEIGHT,
    )

    np.testing.assert_allclose(np.asarray(scaled["assets"]), [[190.0]])
    np.testing.assert_allclose(np.asarray(scaled["pure_capital_flows"]), [[950.0]])
    np.testing.assert_allclose(
        np.asarray(scaled["retirement_distributions"]), [[950.0]]
    )
    np.testing.assert_allclose(np.asarray(scaled["pass_through_flows"]), [[990.0]])


def test_debts_are_left_alone_and_net_worth_is_recomputed_from_the_eroded_assets():
    # Assets of 200 against debts of 100 is net worth of 100.  A 5% haircut
    # leaves 190 of assets and the same 100 of debt, so net worth is 90 -- the
    # household loses 10, which is 10% of its net worth, not 5%.
    scaled = apply_haircut(
        assets=jnp.array([[120.0, 80.0]]),
        debts=jnp.array([[60.0, 40.0]]),
        pure_capital_flows=jnp.zeros((1, 1)),
        pass_through_flows=jnp.zeros((1, 1)),
        retirement_distributions=jnp.zeros((1, 1)),
        fraction=jnp.array([0.05]),
        pass_through_weight=PASS_THROUGH_WEIGHT,
    )

    np.testing.assert_allclose(np.asarray(scaled["assets"]), [[114.0, 76.0]])
    np.testing.assert_allclose(np.asarray(scaled["net_worth"]), [90.0])


def test_a_zero_haircut_leaves_every_column_exactly_as_it_was():
    assets = jnp.array([[100.0, 50.0], [0.0, 0.0]])
    scaled = apply_haircut(
        assets=assets,
        debts=jnp.array([[10.0], [0.0]]),
        pure_capital_flows=jnp.array([[7.0], [0.0]]),
        pass_through_flows=jnp.array([[3.0], [0.0]]),
        retirement_distributions=jnp.array([[1.0], [0.0]]),
        fraction=jnp.zeros(2),
        pass_through_weight=PASS_THROUGH_WEIGHT,
    )

    np.testing.assert_array_equal(np.asarray(scaled["assets"]), np.asarray(assets))
    np.testing.assert_allclose(np.asarray(scaled["net_worth"]), [140.0, 0.0])


# -----------------------------------------------------------------------------
# The whole pass, end to end on a dozen records
# -----------------------------------------------------------------------------


def test_the_dollars_drained_out_of_records_equal_the_dollars_in_the_cells():
    # Twelve records across three ages, ranked and binned by the module's own
    # cut points, then given a shortfall of $100 in every cell that holds
    # anyone.  Summed over records with their weights, the dollars taken out
    # have to equal the dollars the cells held.
    value = jnp.asarray(
        np.array(
            [
                100.0,
                200.0,
                300.0,
                400.0,
                150.0,
                250.0,
                350.0,
                450.0,
                500.0,
                600.0,
                0.0,
                -50.0,
            ]
        )
    )
    weight = jnp.asarray(
        np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 5.0, 5.0])
    )
    age = jnp.asarray(np.array([40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42]))

    cuts = within_age_cutoffs(value, weight, age, AGES, N_BINS)
    bin_index, has_cell = assign_within_age_bin(value, age, cuts, AGES, N_BINS)
    row = cell_rows(age, AGES)

    occupied = sum_by_cell(weight, row, bin_index, has_cell, len(AGES), N_BINS) > 0
    shortfall = jnp.where(occupied, 100.0, 0.0)

    allocated, _ = haircut_fractions(
        cell_shortfall=shortfall,
        row=row,
        bin_index=bin_index,
        has_cell=has_cell,
        net_worth=value,
        weight=weight,
        gross_assets=value * 2.0,
        fmax=0.9,
    )

    np.testing.assert_allclose(
        float((weight * allocated).sum()), float(shortfall.sum()), rtol=1e-12
    )
    # The two records with no net worth are untouched.
    np.testing.assert_allclose(np.asarray(allocated)[-2:], [0.0, 0.0])


# -----------------------------------------------------------------------------
# Financing profiles
# -----------------------------------------------------------------------------


def test_the_profile_spec_reads_all_three_forms():
    assert parse_profile_spec("none") == ProfileSpec("off", None)
    assert parse_profile_spec("OFF") == ProfileSpec("off", None)
    assert parse_profile_spec("flat:0.5") == ProfileSpec("scalar", 0.5)
    assert parse_profile_spec(" some_profile ") == ProfileSpec("folder", "some_profile")
    assert parse_profile_spec(None) == ProfileSpec("folder", "default")
    assert parse_profile_spec("") == ProfileSpec("folder", "default")


def test_a_flat_shorthand_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError, match="flat:<number>"):
        parse_profile_spec("flat:half")


def test_the_off_profile_saves_nothing_and_the_flat_profile_saves_everywhere():
    off = resolve_profile("none", ".", AGES, N_BINS)
    flat = resolve_profile("flat:0.25", ".", AGES, N_BINS)

    np.testing.assert_array_equal(off.saving_shares, np.zeros((len(AGES), N_BINS)))
    np.testing.assert_array_equal(
        flat.saving_shares, np.full((len(AGES), N_BINS), 0.25)
    )
    np.testing.assert_array_equal(flat.transition, np.eye(N_BINS))


def test_a_profile_folder_is_read_into_a_saving_surface(tmp_path):
    lines = ["age,nw_pctile,s"]
    for age in AGES:
        for pctile in range(1, N_BINS + 1):
            lines.append(f"{age},{pctile},{0.1 * pctile:.4f}")
    (tmp_path / "s.csv").write_text("\n".join(lines) + "\n")

    shares = load_saving_shares(tmp_path / "s.csv", AGES, N_BINS)

    np.testing.assert_allclose(shares[0], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(shares[-1], [0.1, 0.2, 0.3, 0.4])


def test_a_profile_missing_a_cell_is_refused_rather_than_filled(tmp_path):
    lines = ["age,nw_pctile,s"]
    for age in AGES:
        for pctile in range(1, N_BINS + 1):
            if age == AGES[1] and pctile == 3:
                continue
            lines.append(f"{age},{pctile},0.5")
    (tmp_path / "s.csv").write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="cells exactly once"):
        load_saving_shares(tmp_path / "s.csv", AGES, N_BINS)


def test_a_saving_share_outside_zero_to_one_is_refused(tmp_path):
    lines = ["age,nw_pctile,s"]
    for age in AGES:
        for pctile in range(1, N_BINS + 1):
            lines.append(f"{age},{pctile},{1.5 if age == AGES[0] else 0.5}")
    (tmp_path / "s.csv").write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match=r"s must be in \[0, 1\]"):
        load_saving_shares(tmp_path / "s.csv", AGES, N_BINS)


def test_no_transition_file_means_full_persistence(tmp_path):
    np.testing.assert_array_equal(
        load_transition_matrix(tmp_path, N_BINS), np.eye(N_BINS)
    )


def test_a_transition_file_is_raked_on_the_way_in(tmp_path):
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    np.savetxt(tmp_path / "M.csv", matrix, delimiter=",")

    loaded = load_transition_matrix(tmp_path, 2)

    np.testing.assert_allclose(loaded.sum(axis=1), 1.0, atol=4e-16)
    assert abs(loaded.sum(axis=0) - 1.0).max() < SINKHORN_TOL


def test_a_transition_file_of_the_wrong_size_is_refused(tmp_path):
    np.savetxt(tmp_path / "M.csv", np.eye(3), delimiter=",")

    with pytest.raises(ValueError, match="must be 4 x 4"):
        load_transition_matrix(tmp_path, N_BINS)
