import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.remit import remit_taxes


def test_remit_taxes_allocates_a_small_constructed_household_set():
    result = remit_taxes(
        {
            "agi": jnp.array([100.0, 0.0, 100.0]),
            "txbl_int": jnp.array([10.0, 3.0, -50.0]),
            "div_ord": jnp.array([20.0, 0.0, 0.0]),
            "div_pref": jnp.array([0.0, 0.0, 0.0]),
            "txbl_kg": jnp.array([20.0, 0.0, 0.0]),
            "liab_iit": jnp.array([100.0, 20.0, 100.0]),
            "ref_other": jnp.array([10.0, 5.0, 0.0]),
            "refund": jnp.array([8.0, 4.0, 0.0]),
            "liab_pr": jnp.array([30.0, 5.0, 0.0]),
        }
    )

    np.testing.assert_allclose(result["pmt_iit_nonwithheld"], [45.0, 0.0, 0.0])
    np.testing.assert_allclose(result["pmt_iit_withheld"], [45.0, 15.0, 100.0])
    np.testing.assert_allclose(result["pmt_refund_nonwithheld"], [8.0, 4.0, 0.0])
    np.testing.assert_allclose(result["pmt_refund_withheld"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result["pmt_pr_nonwithheld"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result["pmt_pr_withheld"], [30.0, 5.0, 0.0])
