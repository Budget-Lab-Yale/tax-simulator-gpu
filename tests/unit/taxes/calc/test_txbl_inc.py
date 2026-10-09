import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.txbl_inc import TaxableIncomeLaw, calc_txbl_inc

LAW = TaxableIncomeLaw(100000.0, 0.0, 0.0)


def test_exact_itemization_tie_takes_standard_deduction():
    # At the $10,000 tie the standard deduction is taken; the $1,000 senior
    # deduction raises it to $11,000, leaving $89,000 taxable income.
    data = {"agi": jnp.array([100000.0]), "std_ded": jnp.array([10000.0]), "item_ded": jnp.array([10000.0]), "salt_item_ded": jnp.array([0.0]), "pe_ded": jnp.array([0.0]), "qbi_ded": jnp.array([0.0]), "tip_ded": jnp.array([0.0]), "ot_ded": jnp.array([0.0]), "senior_ded": jnp.array([1000.0])}
    out = calc_txbl_inc(data, LAW)
    assert not out["itemizing"][0]
    np.testing.assert_allclose([out["ded"][0], out["txbl_inc"][0]], [11000.0, 89000.0])
