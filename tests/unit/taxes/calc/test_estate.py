import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.estate import (
    ESTATE_ASSET_COLS,
    ESTATE_DEBT_COLS,
    EstateLaw,
    EstateParameters,
    calc_estate,
)

LAW = EstateLaw(
    exemption=5_000.0,
    brackets=jnp.arange(12, dtype=float) * 1_000.0,
    rates=jnp.arange(1, 13, dtype=float) / 100.0,
    portability=1.0,
    income_tax_ded=1.0,
)
PARAMS = EstateParameters(
    reporting_rate=1.0,
    pass_through_valuation_rate=1.0,
    gift_ratio=0.0,
    bin_lower_bounds=jnp.array([0.0]),
    deduction_shares=jnp.array([0.10]),
    dsue_probabilities=jnp.array([0.40]),
    dsue_shares=jnp.array([0.25]),
)


def records(**values):
    names = ESTATE_ASSET_COLS + ESTATE_DEBT_COLS
    result = {name: jnp.asarray(values.get(name, 0.0)) for name in names}
    result["filing_status"] = jnp.asarray(values.get("filing_status", 1))
    result["q_death2"] = jnp.asarray(values.get("q_death2", 0.0))
    for optional in (
        "estate_income_tax_ded",
        "estate_concealed_frac",
        "estate_base_bump",
    ):
        if optional in values:
            result[optional] = jnp.asarray(values[optional])
    shapes = [value.shape for value in result.values() if value.ndim > 0]
    if shapes:
        result = {
            name: jnp.broadcast_to(value, shapes[0]) for name, value in result.items()
        }
    return result


def test_single_estate_uses_full_twelve_bracket_schedule_and_dsue_state():
    result = calc_estate(records(**{"value.cash": 20_000.0}), LAW, PARAMS)

    # The 10% non-debt deduction leaves $18,000. Tax above the $5,000 exclusion
    # is $1,000 × (6%+...+11%) + $7,000 × 12% = $1,350. DSUE raises the
    # exclusion to $10,000, leaving $1,000 × 11% + $7,000 × 12% = $950.
    np.testing.assert_allclose(result["estate_distributable"], 18_000.0)
    np.testing.assert_allclose(result["liab_estate_nodsue"], 1_350.0)
    np.testing.assert_allclose(result["liab_estate_dsue"], 950.0)
    np.testing.assert_allclose(result["estate_p_dsue"], 0.40)


def test_joint_both_death_event_uses_two_exclusions_and_no_dsue_probability():
    result = calc_estate(
        records(**{"value.cash": 20_000.0}, filing_status=2, q_death2=0.01),
        LAW,
        PARAMS,
    )

    np.testing.assert_allclose(result["liab_estate_nodsue"], 950.0)
    np.testing.assert_allclose(result["liab_estate_dsue"], 950.0)
    np.testing.assert_allclose(result["estate_p_dsue"], 0.0)


def test_estate_optional_base_deductions_do_not_change_distributable_amount():
    result = calc_estate(
        records(
            **{"value.cash": 20_000.0},
            estate_income_tax_ded=1_000.0,
            estate_concealed_frac=0.10,
        ),
        LAW,
        PARAMS,
    )

    np.testing.assert_allclose(result["estate_distributable"], 18_000.0)
    assert result["liab_estate_nodsue"] < 1_350.0


# A one-dollar bump to a $38 million estate must survive float precision.
# The model runs in 64-bit floats (set when taxmodel is imported); in 32-bit
# floats the bump would be lost.
def test_estate_next_dollar_bump_needs_float64():
    big = 38_000_000.0
    base = calc_estate(records(**{"value.cash": big}), LAW, PARAMS)
    bumped = calc_estate(
        records(**{"value.cash": big}, estate_base_bump=1.0), LAW, PARAMS
    )
    # The base is far above the top bracket, so the next dollar is taxed at 12%.
    np.testing.assert_allclose(
        bumped["liab_estate_nodsue"] - base["liab_estate_nodsue"], 0.12, rtol=1e-6
    )
