import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.niit import NiitLaw, calc_niit

NAMES = (
    "txbl_int",
    "div_ord",
    "div_pref",
    "txbl_kg",
    "sch_e",
    "scorp_active",
    "scorp_active_loss",
    "scorp_179",
    "part_active",
    "part_active_loss",
    "part_179",
    "part_se1",
    "part_se2",
    "inv_int_item_ded",
    "agi",
)
LAW = NiitLaw(0.0, jnp.array([0.0, 0.04]), jnp.array([0.0, 20_000.0]))


def records(**values):
    return {name: jnp.asarray(values.get(name, 0.0)) for name in NAMES}


def test_niit_taxes_lesser_top_band_of_agi_and_investment_income():
    result = calc_niit(records(agi=30_000.0, txbl_int=20_000.0), LAW)

    # The calibrated base is $20,000 × 70% = $14,000. As the top $14,000 of
    # $30,000 AGI, $10,000 lies above the invented $20,000 threshold: $400 tax.
    np.testing.assert_allclose(result["liab_niit"], 400.0)


def test_niit_active_income_switch_and_batch_vectorization():
    batch = records(
        agi=jnp.array([30_000.0, 30_000.0]),
        sch_e=jnp.array([20_000.0, 20_000.0]),
        part_active=jnp.array([10_000.0, 10_000.0]),
    )
    excluded = calc_niit(batch, LAW)["liab_niit"]
    included_law = NiitLaw(1.0, LAW.rates, LAW.brackets)
    included = calc_niit(batch, included_law)["liab_niit"]

    assert np.all(included > excluded)
    shape = batch["agi"].shape
    expected = jnp.array(
        [
            calc_niit(
                {
                    key: jnp.broadcast_to(value, shape)[i]
                    for key, value in batch.items()
                },
                LAW,
            )["liab_niit"]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(excluded, expected)
