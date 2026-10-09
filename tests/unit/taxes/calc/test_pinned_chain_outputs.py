"""Every chain output and marginal rate on 300 synthetic households, compared
with stored values.

The stored values are in `tests/golden/fixtures/calc/pinned_chain_outputs.npz`.
Dollar amounts must match to a cent and rates to a millionth.

The households span the phase-out ranges of many provisions (the caregiver
credit, the newborn bonus, the tip, overtime, senior and auto-loan-interest
deductions, the AMT exemption, the education credits and the personal
exemption), and a dozen of them sit exactly on a phase-out threshold read off
the bound law, so the rate reported at a tie is pinned along with the amounts.

To record the pins again, run this file as a script from the repository root:
    JAX_PLATFORMS=cpu PYTHONPATH=src:. python tests/unit/taxes/calc/test_pinned_chain_outputs.py
Do that only for a change that is meant to move a number, and say so in the
commit that carries the new pins.
"""

from __future__ import annotations

import numpy as np

from taxmodel.scenario.provenance import read_compiled_npz
from taxmodel.taxes import law_for_calculators, marginal_rates, statutory_order
from tests.conftest import COMPILED_BASELINE, REPO_ROOT
from tests.unit.taxes.conftest import ESTATE_MEASUREMENT, synthetic_records

PINS = REPO_ROOT / "tests" / "golden" / "fixtures" / "calc" / "pinned_chain_outputs.npz"
YEAR = 2027
N_HOUSEHOLDS = 300
SEED = 20260901

# A cent for a dollar amount; a millionth for a marginal rate.
DOLLAR_TOLERANCE = 0.005
RATE_TOLERANCE = 1e-6

# The tips and overtime every exact-threshold household earns on top of its
# base pay, so that the tip and overtime deductions have something to phase out.
TIE_TIPS = 4_000.0
TIE_OVERTIME = 3_000.0


def _spread(rng, n, low, high, share_nonzero):
    """Amounts spread evenly between `low` and `high`, zero for the rest."""
    values = rng.uniform(low, high, n)
    return np.where(rng.uniform(size=n) < share_nonzero, values, 0.0)


def households():
    """Three hundred households and the 2027 law bound to them."""
    rng = np.random.default_rng(SEED)
    n = N_HOUSEHOLDS
    filing_status = np.tile([1, 2, 3, 4], n // 4 + 1)[:n]
    joint = filing_status == 2
    values, years = read_compiled_npz(COMPILED_BASELINE)
    law = law_for_calculators.bind_year(values, years, YEAR).for_records(filing_status)

    columns = {
        "age1": rng.uniform(22, 85, n).round(),
        "age2": np.where(joint, rng.uniform(22, 85, n).round(), np.nan),
        # Wages from nothing to well past every phase-out; a tenth earn none.
        "wages_base1": _spread(rng, n, 0.0, 700_000.0, 0.9),
        "wages_base2": np.where(joint, _spread(rng, n, 0.0, 300_000.0, 0.8), 0.0),
        "tips1": _spread(rng, n, 0.0, 30_000.0, 0.4),
        "tips2": np.where(joint, _spread(rng, n, 0.0, 20_000.0, 0.3), 0.0),
        "tips_lh1": (rng.uniform(size=n) < 0.5).astype(float),
        "tips_lh2": (rng.uniform(size=n) < 0.5).astype(float),
        "ot1": _spread(rng, n, 0.0, 40_000.0, 0.4),
        "ot2": np.where(joint, _spread(rng, n, 0.0, 25_000.0, 0.3), 0.0),
        "trad_contr_er1": _spread(rng, n, 0.0, 15_000.0, 0.5),
        "sole_prop1": _spread(rng, n, -20_000.0, 400_000.0, 0.3),
        "sole_prop2": np.where(joint, _spread(rng, n, 0.0, 100_000.0, 0.2), 0.0),
        "farm1": _spread(rng, n, -10_000.0, 60_000.0, 0.1),
        "part_se1": _spread(rng, n, 0.0, 200_000.0, 0.1),
        "txbl_int": _spread(rng, n, 0.0, 20_000.0, 0.5),
        "div_ord": _spread(rng, n, 0.0, 30_000.0, 0.4),
        "div_pref": _spread(rng, n, 0.0, 40_000.0, 0.4),
        "kg_st": _spread(rng, n, -10_000.0, 50_000.0, 0.3),
        "kg_lt": _spread(rng, n, -20_000.0, 500_000.0, 0.4),
        "rent": _spread(rng, n, 0.0, 60_000.0, 0.2),
        "gross_ss": _spread(rng, n, 10_000.0, 45_000.0, 0.2),
        "char_cash": _spread(rng, n, 0.0, 60_000.0, 0.5),
        "salt_prop": _spread(rng, n, 0.0, 25_000.0, 0.6),
        "first_mort_int": _spread(rng, n, 0.0, 30_000.0, 0.4),
        "care_exp": _spread(rng, n, 0.0, 12_000.0, 0.3),
        "aoc_exp": _spread(rng, n, 0.0, 6_000.0, 0.2),
        "llc_exp": _spread(rng, n, 0.0, 10_000.0, 0.2),
        "auto_int_exp": _spread(rng, n, 0.0, 6_000.0, 0.4),
        "med_exp": _spread(rng, n, 0.0, 40_000.0, 0.2),
        "net_worth": _spread(rng, n, 0.0, 30_000_000.0, 0.5),
        # The synthetic defaults put every random draw above every take-up
        # threshold. Half the households draw low instead, so the child-care
        # credit is taken and the auto-loan deduction's new-car test passes.
        "r.cdctc_takeup": np.where(rng.uniform(size=n) < 0.5, 0.0, 0.999),
        "r.new_car": np.where(rng.uniform(size=n) < 0.5, 0.0, 0.999),
    }
    # Dependents: up to three, with ages and Social Security numbers, for
    # the child credits, the child-care credit and the earned income credit.
    n_dep = rng.integers(0, 4, n)
    columns["n_dep"] = n_dep.astype(float)
    columns["n_dep_eitc"] = n_dep.astype(float)
    for k in (1, 2, 3):
        has = n_dep >= k
        columns[f"dep_age{k}"] = np.where(has, rng.uniform(0, 17, n).round(), np.nan)
        columns[f"dep_ssn{k}"] = np.where(has, 1.0, 0.0)
        columns[f"dep_ctc{k}"] = np.where(has, 1.0, 0.0)

    # A dozen households sit exactly on a phase-out threshold. Their only
    # income is wages, and nothing is deducted above the line, so adjusted
    # gross income is base pay plus tips plus overtime, and base pay is set to
    # land that sum on the threshold. Each threshold is read off the law bound
    # to that household's filing status. The auto-loan household is the one
    # exception: its deduction is above the line, and its own phase-out is
    # tested against income before that deduction, which is again the wage.
    thresholds = (
        law.below_ded.tip_ded_po_thresh,
        law.below_ded.ot_ded_po_thresh,
        law.below_ded.senior_ded_po_thresh,
        law.agi.auto_int_ded_po_thresh,
        law.caregiver_cred.po_thresh,
        law.ctc.baby_bonus_po_thresh,
        law.amt.exempt_po_thresh,
        law.ed_cred.aoc_po_thresh,
        law.ed_cred.llc_po_thresh,
        law.pe_ded.po_thresh,
        law.ctc.po_thresh1,
        law.below_ded.tip_ded_po_thresh,
    )
    exact = np.arange(n - len(thresholds), n)
    for slot, threshold in zip(exact, thresholds, strict=True):
        agi = float(np.asarray(threshold)[slot])
        if not np.isfinite(agi):
            agi = 250_000.0
        kept = ("age1", "age2", "tips_lh1", "tips_lh2", "r.cdctc_takeup", "r.new_car")
        for name, column in columns.items():
            if name not in kept:
                column[slot] = np.nan if name.startswith("dep_age") else 0.0
        columns["wages_base1"][slot] = agi - TIE_TIPS - TIE_OVERTIME
        columns["tips1"][slot] = TIE_TIPS
        columns["ot1"][slot] = TIE_OVERTIME
        if threshold is law.agi.auto_int_ded_po_thresh:
            columns["auto_int_exp"][slot] = 2_000.0
        columns["care_exp"][slot] = 5_000.0
        columns["aoc_exp"][slot] = 3_000.0
        columns["age1"][slot] = 67.0
        columns["n_dep"][slot] = 1.0
        columns["n_dep_eitc"][slot] = 1.0
        columns["dep_age1"][slot] = 0.0
        columns["dep_ssn1"][slot] = 1.0
        columns["dep_ctc1"][slot] = 1.0

    records = synthetic_records(filing_status, **columns)
    return records, law


def compute():
    """Every chain output and every marginal rate, as NumPy arrays."""
    records, law = households()
    taxes = statutory_order.run_taxes(records, law, ESTATE_MEASUREMENT)
    rates = marginal_rates.marginal_rates(records, law, ESTATE_MEASUREMENT)
    out = {f"tax:{name}": np.asarray(value, dtype=np.float64) for name, value in taxes.items()}
    out.update(
        {f"rate:{name}": np.asarray(value, dtype=np.float64) for name, value in rates.items()}
    )
    return out


def test_every_chain_output_and_marginal_rate_matches_its_pin():
    pinned = dict(np.load(PINS))
    actual = compute()
    assert set(actual) == set(pinned), (
        "the chain writes a different set of columns than the pins hold: "
        f"missing {sorted(set(pinned) - set(actual))}, "
        f"new {sorted(set(actual) - set(pinned))}"
    )
    for name in sorted(pinned):
        tolerance = RATE_TOLERANCE if name.startswith("rate:") else DOLLAR_TOLERANCE
        np.testing.assert_allclose(
            actual[name], pinned[name], rtol=0.0, atol=tolerance, err_msg=name
        )


def test_the_exact_threshold_households_are_where_they_are_meant_to_be():
    """A pin at a tie is only worth something if the household is on the tie."""
    records, law = households()
    taxes = statutory_order.run_taxes(records, law, ESTATE_MEASUREMENT)
    slot = N_HOUSEHOLDS - 12
    assert float(taxes["agi"][slot]) == float(
        np.asarray(law.below_ded.tip_ded_po_thresh)[slot]
    )


if __name__ == "__main__":
    np.savez_compressed(PINS, **compute())
    print(f"recorded {PINS}")
