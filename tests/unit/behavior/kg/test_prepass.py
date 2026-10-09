"""Turning households into the age cells the capital-gains model runs on.

Each test here is a small hand-built population whose cell answers can be
worked out on paper.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from taxmodel.behavior.kg import prepass

# The five asset classes the gains model tracks, in the order the frame's
# `value.` and `basis.` columns are read in.
CLASSES = ("equities", "pass_throughs", "primary_home", "other_home", "re_fund")

CHARITY = prepass.CharitableBequestSettings(
    extensive_intercept=-2.415,
    extensive_log_estate_slope=0.458,
    intensive_intercept=-1.872,
    intensive_log_estate_slope=0.468,
    base_year=2026,
)


def households(**columns):
    """A frame of households, with every column the pre-pass reads present.

    Ages are given as `age1` on single returns, and that puts each household
    in its age cell. Everything not given is zero.
    """
    n = len(columns["age1"])
    frame = {
        "weight": jnp.ones(n),
        "kg_lt": jnp.zeros(n),
        "filing_status": jnp.ones(n),
        "age2": jnp.full(n, jnp.nan),
        "m_household": jnp.zeros(n),
        "estate_economic_gross": jnp.zeros(n),
    }
    for name in CLASSES:
        frame[f"value.{name}"] = jnp.zeros(n)
        frame[f"basis.{name}"] = jnp.zeros(n)
    frame.update({k: jnp.asarray(v, dtype=jnp.float64) for k, v in columns.items()})
    frame["age1"] = jnp.asarray(columns["age1"], dtype=jnp.float64)
    return frame


def cell_average_rate(frame, marginal_rate, *, n_ages):
    """The rate on realized gains averaged within each age cell."""
    weights = prepass.cell_weights_from_records(frame, n_ages=n_ages)
    return prepass.marginal_rate_cell_sums(
        weights, marginal_rate, n_ages=n_ages
    ).average()


def test_a_household_lands_in_the_cell_of_its_age():
    """Ages 18 and 19 are the first two cells; 90 pools into the top one."""

    frame = households(
        age1=[18.0, 19.0, 90.0],
        weight=[1.0, 1.0, 1.0],
        **{"value.equities": [100.0, 200.0, 400.0]},
    )
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=63
    )

    assert float(cells.stock[0]) == pytest.approx(100.0)
    assert float(cells.stock[1]) == pytest.approx(200.0)
    # Age 80 is the top-coded cell, the last of the sixty-three.
    assert float(cells.stock[-1]) == pytest.approx(400.0)


def test_the_stock_is_weighted_market_value_less_basis_with_losses_left_out():
    """A class under water contributes nothing rather than a negative amount."""

    frame = households(
        age1=[18.0],
        weight=[10.0],
        **{
            "value.equities": [100.0],
            "basis.equities": [40.0],
            "value.other_home": [50.0],
            "basis.other_home": [90.0],
        },
    )
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=2
    )

    # Sixty dollars of gain on the equities, nothing from the house that is
    # worth less than it cost, ten households behind the record.
    assert float(cells.stock[0]) == pytest.approx(600.0)
    assert float(cells.gain_by_asset_class[0, 0]) == pytest.approx(600.0)
    assert float(cells.gain_by_asset_class[0, 3]) == pytest.approx(0.0)


def test_the_stock_counts_pass_through_gain_at_full_market_value():
    """The tax at death values pass-through interests below market value, but
    the stock the selling choice starts from does not: $100 of value against a
    $40 basis is $60 of gain here.
    """

    frame = households(
        age1=[18.0],
        **{"value.pass_throughs": [100.0], "basis.pass_throughs": [40.0]},
    )
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=2
    )

    assert float(cells.stock[0]) == pytest.approx(60.0)
    assert float(cells.gain_by_asset_class[0, 1]) == pytest.approx(60.0)


def test_only_realized_gains_count_toward_turnover():
    """A realized loss is not a sale of an appreciated asset."""

    frame = households(
        age1=[18.0, 18.0],
        kg_lt=[100.0, -400.0],
        **{"value.equities": [1000.0, 1000.0]},
    )
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=2
    )

    assert float(cells.realizations[0]) == pytest.approx(100.0)
    assert float(cells.turnover[0]) == pytest.approx(100.0 / 2000.0)


def test_a_cell_that_sells_nothing_takes_the_population_turnover():
    """Otherwise inherited gains would sit in a young cell forever.

    The population rate is taken over the cells that do sell, so the cell being
    filled in does not drag it toward zero.
    """

    frame = households(
        age1=[18.0, 19.0],
        kg_lt=[0.0, 500.0],
        **{"value.equities": [1000.0, 2000.0]},
    )
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=2
    )

    assert float(cells.turnover[1]) == pytest.approx(500.0 / 2000.0)
    assert float(cells.turnover[0]) == pytest.approx(500.0 / 2000.0)


def test_mortality_comes_back_three_ways():
    """The plain rate per household, and the two weighted numerators."""

    frame = households(
        age1=[18.0, 18.0],
        weight=[1.0, 3.0],
        kg_lt=[100.0, 0.0],
        m_household=[0.5, 0.1],
        **{"value.equities": [1000.0, 2000.0]},
    )
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=2
    )

    assert float(cells.mortality_headcount[0]) == pytest.approx(
        (1.0 * 0.5 + 3.0 * 0.1) / 4.0
    )
    assert float(cells.mortality_gain_weighted[0]) == pytest.approx(
        1.0 * 0.5 * 1000.0 + 3.0 * 0.1 * 2000.0
    )


def test_the_charitable_share_follows_the_gain_dollars_that_die():
    """A household that cannot die contributes nothing to the cell's share.

    The share decides how much of the gain at death escapes tax, so it is
    averaged over mortality times holdings, not over households.
    """

    frame = households(
        age1=[18.0, 18.0],
        m_household=[0.5, 0.0],
        estate_economic_gross=[50e6, 50e6],
        **{"value.equities": [1000.0, 1000.0]},
    )
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=2
    )

    one_household = prepass.charitable_bequest_probability(
        jnp.asarray([50e6]), 1.0, CHARITY
    )
    assert float(cells.charitable_bequest_share[0]) == pytest.approx(
        float(one_household[0])
    )


def test_a_household_with_no_estate_leaves_nothing_to_charity():
    """Zero, not the logit's value at an estate of zero dollars."""

    share = prepass.charitable_bequest_probability(
        jnp.asarray([0.0, -10.0, 1e6]), 1.0, CHARITY
    )
    assert float(share[0]) == 0.0
    assert float(share[1]) == 0.0
    assert 0.0 < float(share[2]) < 1.0


def test_the_rate_on_gains_is_averaged_over_the_dollars_that_sell():
    """Weighted by realizations, which is the base the elasticity is fit on."""

    frame = households(
        age1=[18.0, 18.0],
        kg_lt=[300.0, 100.0],
        **{"value.equities": [1000.0, 1000.0]},
    )
    rate = cell_average_rate(
        frame, jnp.asarray([0.20, 0.40]), n_ages=2
    )

    assert float(rate[0]) == pytest.approx((300.0 * 0.20 + 100.0 * 0.40) / 400.0)


def test_a_cell_that_sells_nothing_prices_its_holdings_instead():
    """And a cell with neither gains nor sales gets zero rather than a NaN."""

    frame = households(
        age1=[18.0, 19.0],
        kg_lt=[0.0, 0.0],
        **{"value.equities": [1000.0, 0.0]},
    )
    rate = cell_average_rate(
        frame, jnp.asarray([0.25, 0.90]), n_ages=2
    )

    assert float(rate[0]) == pytest.approx(0.25)
    assert float(rate[1]) == 0.0


def test_a_measured_rate_outside_the_unit_interval_leaves_the_average():
    """The household leaves the numerator and the denominator together.

    A one-dollar derivative of a statute with discrete steps in it can measure
    at something that is not a rate; averaging it in would let one household
    set a thin cell's price.
    """

    frame = households(
        age1=[18.0, 18.0],
        kg_lt=[100.0, 100.0],
        **{"value.equities": [1000.0, 1000.0]},
    )
    rate = cell_average_rate(
        frame, jnp.asarray([0.20, 50.0]), n_ages=2
    )

    assert float(rate[0]) == pytest.approx(0.20)


def test_rounding_noise_at_zero_does_not_drop_a_household_from_its_cell():
    """Batch width must not move a cell's average rate.

    The measured rate is a next-dollar re-run, so a household whose true rate
    is exactly zero can come out of the subtraction at minus a few
    ten-billionths under one compiled batch shape and at exactly zero under
    another. Both readings must keep the household's dollars in the cell:
    dropping it on the sign of rounding noise moved $0.8 trillion of holdings
    out of the denominators when a one-scenario batch was padded to two lanes.
    """

    frame = households(
        age1=[18.0, 18.0],
        kg_lt=[300.0, 100.0],
        **{"value.equities": [1000.0, 1000.0]},
    )
    exactly_zero = cell_average_rate(
        frame, jnp.asarray([0.20, 0.0]), n_ages=2
    )
    noisy_zero = cell_average_rate(
        frame, jnp.asarray([0.20, -1e-10]), n_ages=2
    )

    assert float(exactly_zero[0]) == pytest.approx((300.0 * 0.20) / 400.0)
    assert float(noisy_zero[0]) == pytest.approx(float(exactly_zero[0]))

    # The band reads noise as the boundary, nothing more: a rate that is
    # genuinely negative still leaves the average, numerator and denominator
    # together.
    genuinely_negative = cell_average_rate(
        frame, jnp.asarray([0.20, -0.25]), n_ages=2
    )
    assert float(genuinely_negative[0]) == pytest.approx(0.20)


def test_the_heir_distribution_covers_the_whole_age_grid():
    """The shares are read positionally, so the file's ages must be the grid."""

    shares = prepass.read_heir_age_distribution()

    assert shares.shape == (63,)
    assert float(jnp.sum(shares)) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.min(shares)) >= 0.0


def test_the_charity_coefficients_come_from_the_settings_file():
    """The four coefficients are not restated in code."""

    settings = prepass.read_charitable_bequest_settings()

    assert settings.base_year == 2026
    assert settings.extensive_log_estate_slope > 0.0


def test_the_joint_share_of_the_top_cell_is_weighted_by_dollars_of_gain():
    """Joint returns hold three of the top cell's four dollars of gain.

    Two households sit in the top cell: a joint return holding $300 of gain and
    a single return holding $100, each behind one weighted household. The share
    counts dollars, not households, so it is 0.75 and not 0.5. A household
    below the top cell does not count at all, however it files.
    """

    frame = households(
        age1=[85.0, 85.0, 40.0],
        weight=[1.0, 1.0, 1.0],
        filing_status=[2.0, 1.0, 2.0],
        **{"value.equities": [300.0, 100.0, 900.0]},
    )

    share = prepass.joint_share_of_top_cell_gain(frame, n_ages=63)

    assert float(share) == pytest.approx(0.75)


def test_the_joint_share_is_zero_when_the_top_cell_holds_nothing():
    """No dollars to weight by, so the mortality tail gets the life table."""

    frame = households(age1=[40.0], weight=[1.0], **{"value.equities": [900.0]})

    assert float(prepass.joint_share_of_top_cell_gain(frame, n_ages=63)) == 0.0
