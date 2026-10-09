import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.surtax import SurtaxLaw, calc_agi_surtax

LAW = SurtaxLaw(10_000.0, 0.01, 20_000.0, 0.02, 30_000.0, 0.03)


def test_surtax_bracket_boundaries_and_arithmetic():
    result = calc_agi_surtax({"agi": jnp.array([10_000.0, 20_000.0, 40_000.0])}, LAW)[
        "liab_surtax"
    ]

    # At $40,000: $10,000 × 1% + $10,000 × 2% + $10,000 × 3% = $600.
    np.testing.assert_allclose(result, [0.0, 100.0, 600.0])
