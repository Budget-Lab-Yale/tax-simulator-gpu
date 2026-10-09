import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.wealth import WealthLaw, calc_wealth

LAW = WealthLaw(
    brackets=jnp.array([0.0, 1_000.0, 2_000.0, jnp.nan]),
    rates=jnp.array([0.0, 0.01, 0.02, jnp.nan]),
)


def test_wealth_tax_is_zero_at_and_below_the_zero_rate_threshold():
    result = calc_wealth(jnp.array([-50.0, 0.0, 1_000.0]), LAW)

    np.testing.assert_allclose(result["liab_wealth"], [0.0, 0.0, 0.0])


def test_wealth_tax_integrates_each_bracket():
    result = calc_wealth(jnp.array([1_500.0, 2_500.0]), LAW)

    # $500 is taxed at 1% in the first case.  The second adds $500 at 2%.
    np.testing.assert_allclose(result["liab_wealth"], [5.0, 20.0])
