import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.agi import AgiLaw, SocialSecurityLaw, calc_agi, calc_ss

SS = SocialSecurityLaw(0.5, jnp.array([0.5, 0.85]), jnp.array([25000.0, 34000.0]))
LAW = AgiLaw(2100.0, 1000.0, 4000.0, 5000.0, 300.0, 2500.0, 100000.0, 0.1, 1.0, SS)
NAMES = ["wages_base1", "wages_base2", "tips1", "tips2", "ot1", "ot2", "txbl_int", "exempt_int", "div_ord", "div_pref", "txbl_ira_dist", "txbl_pens_dist", "gross_ss", "txbl_kg", "other_gains", "state_ref", "alimony", "divorce_year", "sole_prop1", "sole_prop2", "sch_e", "pt", "farm1", "farm2", "ui", "other_inc", "ed_exp", "hsa_contr", "liab_seca_er", "trad_contr_ira", "sl_int_ded", "keogh_contr", "se_health", "early_penalty", "alimony_exp", "tuition_ded", "dpad", "char_cash", "char_noncash", "other_above_ded", "auto_int_exp", "r.bus_loss", "r.new_car"]


def records(**values):
    return {key: jnp.asarray(values.get(key, 0.0), dtype=jnp.float64) for key in NAMES}


def test_ss_thresholds_and_agi_arithmetic():
    # At $25,000 combined income §86 includes $0; at $34,000 it includes
    # ($34,000 - $25,000) × 50% = $4,500 of $20,000 benefits.
    out = calc_ss({"gross_ss": jnp.array([20000.0, 20000.0]), "magi_ss": jnp.array([15000.0, 24000.0])}, SS)
    np.testing.assert_allclose(out["txbl_ss"], [0.0, 4500.0])
    np.testing.assert_allclose(calc_agi(records(wages_base1=40100.0, sl_int_ded=100.0), LAW)["agi"], 40000.0)


# The Social Security inclusion thresholds differ by filing status:
# $25,000/$34,000 for single filers, $32,000/$44,000 for joint returns.
def test_ss_brackets_may_vary_by_filing_status():
    law = SocialSecurityLaw(
        0.5,
        jnp.array([[0.0, 0.5, 0.85], [0.0, 0.5, 0.85]]),
        jnp.array([[0.0, 25_000.0, 34_000.0], [0.0, 32_000.0, 44_000.0]]),
    )
    out = calc_ss(
        {"gross_ss": jnp.array([20_000.0, 20_000.0]), "magi_ss": jnp.array([40_000.0, 40_000.0])},
        law,
    )
    # Single: capped at 85% of $20,000. Joint: 50% of ($44,000 - $32,000)
    # plus 85% of ($50,000 - $44,000) = $6,000 + $5,100.
    np.testing.assert_allclose(out["txbl_ss"], [17_000.0, 11_100.0])
