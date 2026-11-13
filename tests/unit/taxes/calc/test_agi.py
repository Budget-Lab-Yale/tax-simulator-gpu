import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.agi import AgiLaw, SocialSecurityLaw, calc_agi, calc_ss

SS = SocialSecurityLaw(0.5, jnp.array([0.5, 0.85]), jnp.array([25000.0, 34000.0]))
LAW = AgiLaw(2100.0, 1000.0, 4000.0, 5000.0, 300.0, 0.0, 150000.0, SS)
NAMES = ["wages_base1", "wages_base2", "tips1", "tips2", "ot1", "ot2", "txbl_int", "exempt_int", "div_ord", "div_pref", "txbl_ira_dist", "txbl_pens_dist", "gross_ss", "txbl_kg", "other_gains", "state_ref", "alimony", "divorce_year", "sole_prop1", "sole_prop2", "sch_e", "pt", "farm1", "farm2", "ui", "other_inc", "ed_exp", "hsa_contr", "liab_seca_er", "trad_contr_ira", "sl_int_ded", "keogh_contr", "se_health", "early_penalty", "alimony_exp", "tuition_ded", "dpad", "char_cash", "char_noncash", "other_above_ded", "r.bus_loss"]


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


def test_car_loan_interest_does_not_reduce_agi():
    # Car loan interest is deducted below AGI (IRC 163(h)(4)), so $3,000 of it
    # leaves $40,000 of wages as $40,000 of AGI. calc_agi does not read it.
    out = calc_agi({**records(wages_base1=40_000.0), "auto_int_exp": jnp.asarray(3_000.0)}, LAW)
    np.testing.assert_allclose(out["agi"], 40_000.0)
    assert "auto_int_ded" not in out


# 2020: up to $10,200 of unemployment compensation excluded per return when AGI
# figured without the exclusion is under $150,000 (ARPA section 9042).
def test_ui_exclusion_2020():
    law = AgiLaw(2100.0, 1000.0, 4000.0, 5000.0, 300.0, 10_200.0, 150_000.0, SS)
    wages = jnp.array([20_000.0, 20_000.0, 145_000.0, 139_000.0, 20_000.0])
    ui = jnp.array([15_000.0, 5_000.0, 10_000.0, 10_000.0, 0.0])
    out = calc_agi(records(wages_base1=wages, ui=ui), law)
    # capped at $10,200; all of $5,000; none at $155,000 before the exclusion;
    # all of $10,000 at $149,000; nothing without benefits
    np.testing.assert_allclose(out["ui_excl"], [10_200.0, 5_000.0, 0.0, 10_000.0, 0.0])
    np.testing.assert_allclose(out["agi"], wages + ui - out["ui_excl"])
    # with no exclusion in the law the same returns exclude nothing
    np.testing.assert_allclose(calc_agi(records(wages_base1=wages, ui=ui), LAW)["ui_excl"], 0.0)


def test_ui_exclusion_lowers_taxable_social_security():
    law = AgiLaw(2100.0, 1000.0, 4000.0, 5000.0, 300.0, 10_200.0, 150_000.0, SS)
    rec = records(wages_base1=20_000.0, ui=10_000.0, gross_ss=20_000.0)
    with_excl = calc_agi(rec, law)
    without = calc_agi(rec, LAW)
    assert float(with_excl["txbl_ss"]) < float(without["txbl_ss"])
    np.testing.assert_allclose(with_excl["ui_excl"], 10_000.0)

