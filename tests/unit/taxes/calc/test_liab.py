import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.liab import LiabilityLaw, calc_liab

NAMES = (
    "liab_bc",
    "ftc",
    "cdctc_nonref",
    "ed_nonref",
    "savers_nonref",
    "caregiver_cred_nonref",
    "res_energy_cred",
    "old_cred",
    "ctc_nonref",
    "gbc",
    "prior_yr_cred",
    "other_nonref",
    "ctc_ref",
    "ed_ref",
    "caregiver_cred_ref",
    "net_ptc",
    "eitc",
    "rebate",
    "wage_subsidy1",
    "wage_subsidy2",
    "cdctc_ref",
    "savers_ref",
    "liab_niit",
    "liab_surtax",
    "liab_seca",
    "recapture_tax",
    "ira_penalty",
)
LAW = LiabilityLaw(0.0, 0.0, 0.0, 0.0, 0.0)


def records(**values):
    return {name: jnp.asarray(values.get(name, 0.0)) for name in NAMES}


def test_liability_allocates_refundables_to_iit_then_other_tax_then_refund():
    result = calc_liab(
        records(
            liab_bc=1_000.0,
            ftc=200.0,
            ctc_nonref=900.0,
            ctc_ref=1_500.0,
            liab_niit=100.0,
            liab_surtax=50.0,
            liab_seca=300.0,
        ),
        LAW,
    )

    # Nonrefundable credits are capped at $1,000. Refundable credits then pay
    # $300 of SECA and leave a $1,200 outlay; NIIT and surtax remain $150.
    np.testing.assert_allclose(result["nonref"], 1_000.0)
    np.testing.assert_allclose(result["ref_iit"], 0.0)
    np.testing.assert_allclose(result["ref_other"], 300.0)
    np.testing.assert_allclose(result["refund"], 1_200.0)
    np.testing.assert_allclose(result["liab_iit"], 150.0)
    np.testing.assert_allclose(result["liab_iit_net"], -1_350.0)
    np.testing.assert_allclose(result["number_of_credits"], 2)


def test_caregiver_credit_enters_liability_and_the_credit_count():
    # The caregiver credit counts toward the number of credits claimed. It is
    # booked at zero in every law the model runs.
    result = calc_liab(records(liab_bc=500.0, caregiver_cred_nonref=100.0), LAW)
    np.testing.assert_allclose(result["nonref"], 100.0)
    np.testing.assert_allclose(result["number_of_credits"], 1)
