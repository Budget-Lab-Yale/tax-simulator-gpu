import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.payroll import PayrollLaw, calc_pr

LAW = PayrollLaw(
    tips_exempt=0.0,
    ot_exempt=0.0,
    seca_taxable_rate=0.9235,
    se_thresh=400.0,
    oasdi_ee_rates=jnp.array([0.10, 0.0]),
    oasdi_ee_brackets=jnp.array([0.0, 10_000.0]),
    oasdi_er_rates=jnp.array([0.10, 0.0]),
    oasdi_er_brackets=jnp.array([0.0, 10_000.0]),
    hi_ee_rates=jnp.array([0.02]),
    hi_ee_brackets=jnp.array([0.0]),
    hi_er_rates=jnp.array([0.02]),
    hi_er_brackets=jnp.array([0.0]),
    add_med_rates=jnp.array([0.0, 0.01]),
    add_med_brackets=jnp.array([0.0, 20_000.0]),
)

NAMES = (
    "wages_base1",
    "wages_base2",
    "tips1",
    "tips2",
    "ot1",
    "ot2",
    "trad_contr_er1",
    "trad_contr_er2",
    "sole_prop1",
    "sole_prop2",
    "farm1",
    "farm2",
    "part_se1",
    "part_se2",
)


def records(**values):
    return {name: jnp.asarray(values.get(name, 0.0)) for name in NAMES}


def test_mixed_wage_and_self_employment_coordinates_oasdi_base():
    result = calc_pr(records(wages_base1=8_000.0, sole_prop1=5_000.0), LAW)

    # SECA taxes 92.35% × $5,000 = $4,617.50, but wages leave only $2,000 of
    # the $10,000 OASDI base. OASDI is 2 sides × 10% × ($8,000 + $2,000) =
    # $2,000. HI is 2 sides × 2% × ($8,000 + $4,617.50) = $504.70.
    np.testing.assert_allclose(result["liab_oasdi"], 2_000.0)
    np.testing.assert_allclose(result["liab_hi"], 504.70)
    np.testing.assert_allclose(result["liab_pr"], 2_504.70)
    np.testing.assert_allclose(result["liab_seca_er"], 292.35)


def test_self_employment_reporting_threshold_is_strict():
    at_threshold = calc_pr(records(sole_prop1=400.0), LAW)
    above_threshold = calc_pr(records(sole_prop1=401.0), LAW)

    np.testing.assert_allclose(at_threshold["liab_seca"], 0.0)
    # $401 × 92.35% × two sides × (10% OASDI + 2% HI).
    np.testing.assert_allclose(above_threshold["liab_seca"], 88.87764)


def test_additional_medicare_uses_combined_earnings_without_employer_match():
    result = calc_pr(records(wages_base1=12_000.0, wages_base2=10_000.0), LAW)

    # Combined wages exceed the invented $20,000 threshold by $2,000.
    np.testing.assert_allclose(result["liab_add_med"], 20.0)
    np.testing.assert_allclose(result["liab_pr_ee"] - result["liab_pr_er"], 20.0)
