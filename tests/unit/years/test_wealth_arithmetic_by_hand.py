"""Three small cases of the wealth-financing arithmetic, worked by hand.

The comments in each test carry the arithmetic step by step with the test's
numbers, so a reviewer can check the expected values on paper.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from taxmodel.years.wealth_arithmetic import (
    apply_haircut,
    assign_within_age_bin,
    build_aging_matrix,
    cohort_recurrence_step,
    haircut_fractions,
    within_age_cutoffs,
)


def test_weighted_quantile_cuts_on_three_records_of_unequal_weight():
    # Three records in age 40, ranked on net worth, split into four percentiles.
    #
    #   net worth  50, 10, 30
    #   weight      1,  3,  2
    #   cut points at the 25th, 50th and 75th percentiles
    #
    # Sorted by net worth the records are 10, 30, 50 with weights 3, 2, 1, so
    # the cumulative weight shares are 3/6, 5/6 and 6/6 = 0.5000, 0.8333, 1.0.
    # Each cut point is the value of the first record whose cumulative share
    # reaches the percentile:
    #
    #   25th : 0.5000 is the first share at or above 0.25   -> 10
    #   50th : 0.5000 is the first share at or above 0.50   -> 10
    #   75th : 0.8333 is the first share at or above 0.75   -> 30
    ages = np.array([40, 41])
    value = jnp.array([50.0, 10.0, 30.0])
    weight = jnp.array([1.0, 3.0, 2.0])
    age = jnp.array([40, 40, 40])

    cuts = within_age_cutoffs(value, weight, age, ages, n_bins=4)

    np.testing.assert_allclose(cuts[0], [10.0, 10.0, 30.0])

    # A record's percentile, counted from 0, is the number of cut points at or
    # below its net worth, capped at the top percentile:
    #
    #   value 50 : cuts 10, 10 and 30 are at or below -> 3
    #   value 10 : cuts 10 and 10 are at or below     -> 2
    #   value 30 : cuts 10, 10 and 30 are at or below -> 3
    #
    # The repeated cut of 10 leaves the second percentile empty, which is what
    # happens whenever one record carries more than a percentile's worth of
    # weight.
    bin_index, _ = assign_within_age_bin(value, age, cuts, ages, n_bins=4)

    np.testing.assert_array_equal(bin_index, [3, 2, 3])


def test_one_year_of_the_recurrence_on_a_two_age_two_percentile_grid():
    # Two ages, 40 and 41, and two percentiles.  Age 41 is the top of the grid,
    # so it is the absorbing pool.
    #
    #   last year's cells   age 40: (100, 200)   age 41: (300, 400)
    #   transition matrix   (0.75, 0.25)
    #                       (0.25, 0.75)         # already doubly stochastic
    #   growth              age 40: 1.05         age 41: 1.02
    #   inflow              age 40: (10, 20)     age 41: (30, 40)
    #
    # First each age's cells move between percentiles:
    #     age 40: (100*0.75 + 200*0.25,  100*0.25 + 200*0.75) = (125, 175)
    #     age 41: (300*0.75 + 400*0.25,  300*0.25 + 400*0.75) = (325, 375)
    #
    # Then everyone ages one year:
    #     age 40: nobody is younger than the bottom of the grid    -> (0, 0)
    #     age 41: age 40 arrives and age 41 stays
    #             (125 + 325, 175 + 375)                           -> (450, 550)
    #
    # Then growth times the carried stock, plus the inflow:
    #     age 40: (1.05*0   + 10, 1.05*0   + 20) = ( 10,  20)
    #     age 41: (1.02*450 + 30, 1.02*550 + 40) = (489, 601)
    #
    # The transition and the aging together move 1000 dollars and lose none:
    # 125 + 175 + 325 + 375 = 1000, the same as 100 + 200 + 300 + 400.
    cells = jnp.array([[100.0, 200.0], [300.0, 400.0]])
    transition = jnp.array([[0.75, 0.25], [0.25, 0.75]])
    growth = jnp.array([[1.05, 1.05], [1.02, 1.02]])
    inflow = jnp.array([[10.0, 20.0], [30.0, 40.0]])

    stepped = cohort_recurrence_step(
        cells_last_year=cells,
        growth=growth,
        inflow=inflow,
        aging=build_aging_matrix(2),
        transition=transition,
    )

    np.testing.assert_allclose(np.asarray(stepped), [[10.0, 20.0], [489.0, 601.0]])


def test_the_haircut_on_two_records_sharing_one_cell():
    # One cell holding a $90 shortfall, and two records in it:
    #
    #   record A: net worth 400, weight 3, gross assets 500, debts 100
    #   record B: net worth 600, weight 1, gross assets 800, debts 200
    #
    # The shortfall is shared out in proportion to positive net worth:
    #
    #   positive net worth                        A: 400, B: 600
    #   weighted total in the cell                3*400 + 1*600 = 1800
    #   each record's share of the shortfall
    #                  A: 90 * 400 / 1800 = 20
    #                  B: 90 * 600 / 1800 = 30
    #
    # Weighting those back up returns the cell's shortfall exactly:
    # 3*20 + 1*30 = 90.
    #
    # The haircut fraction is the share over gross assets, capped at 0.9 in
    # either direction:
    #                  A: 20 / 500 = 0.0400
    #                  B: 30 / 800 = 0.0375       # neither reaches the cap
    #
    #   pure capital income scaled by 1 - f       A: 0.9600, B: 0.96250
    #   pass-through income scaled by 1 - 0.2 f   A: 0.9920, B: 0.99250
    #
    #   assets scaled by (1 - f)
    #                  A: 500 * 0.9600 = 480
    #                  B: 800 * 0.9625 = 770
    #   net_worth = scaled assets - debts        # debts are untouched
    #                  A: 480 - 100 = 380
    #                  B: 770 - 200 = 570
    #
    # A $1,000 flow in each class, for the two scalings:
    #   pure capital income A: 1000 * 0.9600 = 960
    #   pass-through income A: 1000 * 0.9920 = 992
    shortfall = jnp.array([[90.0]])
    allocated, fraction = haircut_fractions(
        cell_shortfall=shortfall,
        row=jnp.array([0, 0]),
        bin_index=jnp.array([0, 0]),
        has_cell=jnp.array([True, True]),
        net_worth=jnp.array([400.0, 600.0]),
        weight=jnp.array([3.0, 1.0]),
        gross_assets=jnp.array([500.0, 800.0]),
        fmax=0.9,
    )

    np.testing.assert_allclose(np.asarray(allocated), [20.0, 30.0])
    np.testing.assert_allclose(np.asarray(fraction), [0.04, 0.0375])

    scaled = apply_haircut(
        assets=jnp.array([[500.0], [800.0]]),
        debts=jnp.array([[100.0], [200.0]]),
        pure_capital_flows=jnp.array([[1000.0], [1000.0]]),
        pass_through_flows=jnp.array([[1000.0], [1000.0]]),
        retirement_distributions=jnp.zeros((2, 1)),
        fraction=fraction,
        pass_through_weight=0.2,
    )

    np.testing.assert_allclose(np.asarray(scaled["assets"]), [[480.0], [770.0]])
    np.testing.assert_allclose(np.asarray(scaled["net_worth"]), [380.0, 570.0])
    np.testing.assert_allclose(
        np.asarray(scaled["pure_capital_flows"]), [[960.0], [962.5]]
    )
    np.testing.assert_allclose(
        np.asarray(scaled["pass_through_flows"]), [[992.0], [992.5]]
    )
