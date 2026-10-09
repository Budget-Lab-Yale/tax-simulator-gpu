from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc._schedules import schedule_tax
from taxmodel.taxes.calc.tax import (
    AlternativeMaximumTaxLaw,
    AmtLaw,
    TaxLaw,
    calc_alt_max,
    calc_amt,
    calc_tax,
)

TAX_LAW = TaxLaw(
    ord_rates=jnp.array([0.10, 0.20]),
    ord_brackets=jnp.array([0.0, 10_000.0]),
    pref_rates=jnp.array([0.0, 0.15]),
    pref_brackets=jnp.array([0.0, 5_000.0]),
    unrecapture_rate=0.25,
    collectibles_rate=0.28,
    tax_at_ord=0.0,
    no_ord_cap=0.0,
    kg_top_rate_addon=0.0,
    div_top_rate_addon=0.0,
)


def test_ordinary_and_preferred_income_stack():
    records = {
        "txbl_inc": jnp.array([20_000.0]),
        "div_pref": jnp.array([10_000.0]),
        "kg_pref": jnp.zeros(1),
        "kg_1250": jnp.zeros(1),
        "kg_collect": jnp.zeros(1),
    }
    result = calc_tax(records, TAX_LAW)

    # Ordinary: $10,000 × 10% = $1,000. Preferred income is the top $10,000
    # of $20,000, all above the invented $5,000 zero-rate threshold: $1,500.
    np.testing.assert_allclose(result["liab_ord"], [1_000.0])
    np.testing.assert_allclose(result["liab_pref"], [1_500.0])
    np.testing.assert_allclose(result["liab"], [2_500.0])


def test_special_gains_stack_1250_before_collectibles():
    records = {
        "txbl_inc": jnp.array([20_000.0]),
        "div_pref": jnp.zeros(1),
        "kg_pref": jnp.array([10_000.0]),
        "kg_1250": jnp.array([4_000.0]),
        "kg_collect": jnp.array([3_000.0]),
    }
    result = calc_tax(records, TAX_LAW)

    # $10,000 ordinary income costs $1,000. Then $4,000 of §1250 gain and
    # $3,000 of collectibles gain each face the second ordinary rate of 20%.
    # The remaining $3,000 gain is the top slice of taxable income and faces 15%.
    np.testing.assert_allclose(result["liab_1250"], [800.0])
    np.testing.assert_allclose(result["liab_collect"], [600.0])
    np.testing.assert_allclose(result["liab"], [2_850.0])


def test_schedule_d_ordinary_tax_cap_and_override():
    records = {
        "txbl_inc": jnp.array([10_000.0]),
        "div_pref": jnp.array([10_000.0]),
        "kg_pref": jnp.zeros(1),
        "kg_1250": jnp.zeros(1),
        "kg_collect": jnp.zeros(1),
    }
    high_preferred = TaxLaw(
        **{**TAX_LAW.__dict__, "pref_rates": jnp.array([0.20, 0.20])}
    )
    uncapped = TaxLaw(**{**high_preferred.__dict__, "no_ord_cap": 1.0})

    np.testing.assert_allclose(calc_tax(records, high_preferred)["liab"], [1_000.0])
    np.testing.assert_allclose(calc_tax(records, uncapped)["liab"], [2_000.0])


def test_alt_max_uses_strict_qualification_and_excludes_mfs_itemizer():
    law = AlternativeMaximumTaxLaw(rate=0.10, qualify_mult=2.0, exempt=10_000.0)
    records = {
        "agi": jnp.array([19_999.0, 20_000.0, 15_000.0]),
        "gross_ss": jnp.zeros(3),
        "txbl_ss": jnp.zeros(3),
        "filing_status": jnp.array([1, 1, 3]),
        "itemizing": jnp.array([False, False, True]),
        "liab": jnp.full(3, 2_000.0),
    }
    result = calc_alt_max(records, law)

    assert result["alt_max_cap_binds"].tolist() == [True, False, False]


def test_amt_swaps_only_ordinary_schedule_and_nets_regular_liability():
    records = {
        "agi": jnp.array([30_000.0]),
        "ded": jnp.array([5_000.0]),
        "pe_ded": jnp.zeros(1),
        "qbi_ded": jnp.zeros(1),
        "tip_ded": jnp.zeros(1),
        "ot_ded": jnp.zeros(1),
        "senior_ded": jnp.zeros(1),
        "itemizing": jnp.array([False]),
        "salt_item_ded": jnp.zeros(1),
        "misc_item_ded": jnp.zeros(1),
        "item_ded_ex_limits": jnp.zeros(1),
        "item_ded": jnp.zeros(1),
        "std_ded": jnp.array([5_000.0]),
        "state_ref": jnp.zeros(1),
        "amt_nols": jnp.zeros(1),
        "amt_ftc": jnp.array([100.0]),
        "liab": jnp.array([2_500.0]),
        "excess_ptc": jnp.array([50.0]),
        "txbl_inc": jnp.array([25_000.0]),
        "div_pref": jnp.array([5_000.0]),
        "kg_pref": jnp.zeros(1),
        "kg_1250": jnp.zeros(1),
        "kg_collect": jnp.zeros(1),
    }
    law = AmtLaw(
        pe_pref=0.0,
        exempt=5_000.0,
        exempt_po_thresh=100_000.0,
        exempt_po_rate=0.25,
        rates=jnp.array([0.20]),
        brackets=jnp.array([0.0]),
    )
    result = calc_amt(records, TAX_LAW, law)

    # AMT gross income restores the $5,000 standard deduction: $30,000.
    # After a $5,000 exemption, $20,000 ordinary income costs $4,000 and the top
    # $5,000 preferred income costs $750. Net AMT is $4,750 - $100 - $2,500.
    np.testing.assert_allclose(result["amt_gross_inc"], [30_000.0])
    np.testing.assert_allclose(result["amt_txbl_inc"], [25_000.0])
    np.testing.assert_allclose(result["liab_amt"], [2_150.0])
    np.testing.assert_allclose(result["liab_bc"], [4_700.0])


# An invented three-bracket preferential schedule whose top bracket starts at
# $50,000, padded with NaN as the compiled law is. The ordinary-tax cap is off so
# that the preferential tax is charged in full.
ADDON = 0.10
THREE_BRACKET_LAW = TaxLaw(
    ord_rates=jnp.array([0.10, 0.20, 0.30, jnp.nan]),
    ord_brackets=jnp.array([0.0, 10_000.0, 40_000.0, jnp.nan]),
    pref_rates=jnp.array([0.0, 0.15, 0.20, jnp.nan]),
    pref_brackets=jnp.array([0.0, 5_000.0, 50_000.0, jnp.nan]),
    unrecapture_rate=0.25,
    collectibles_rate=0.28,
    tax_at_ord=0.0,
    no_ord_cap=1.0,
    kg_top_rate_addon=0.0,
    div_top_rate_addon=0.0,
)
WITH_ADDON = TaxLaw(**{**THREE_BRACKET_LAW.__dict__, "kg_top_rate_addon": ADDON})
WITH_DIVIDEND_ADDON = TaxLaw(
    **{**THREE_BRACKET_LAW.__dict__, "div_top_rate_addon": ADDON}
)


def _extra_preferential_tax(ordinary, div_pref, kg_pref, law=WITH_ADDON):
    records = {
        "txbl_inc": jnp.array([ordinary + div_pref + kg_pref]),
        "div_pref": jnp.array([div_pref]),
        "kg_pref": jnp.array([kg_pref]),
        "kg_1250": jnp.zeros(1),
        "kg_collect": jnp.zeros(1),
    }
    return (
        calc_tax(records, law)["liab_pref"]
        - calc_tax(records, THREE_BRACKET_LAW)["liab_pref"]
    )


def test_gains_rate_addon_leaves_dividends_alone():
    np.testing.assert_allclose(_extra_preferential_tax(30_000.0, 70_000.0, 0.0), [0.0])


def test_gains_rate_addon_on_gains_entirely_in_the_top_bracket():
    # Ordinary income of $60,000 already passes the $50,000 top threshold.
    np.testing.assert_allclose(
        _extra_preferential_tax(60_000.0, 0.0, 40_000.0), [ADDON * 40_000.0]
    )


def test_gains_rate_addon_on_gains_straddling_the_top_threshold():
    # Gains run from $30,000 to $100,000; the $50,000 above the threshold pays.
    np.testing.assert_allclose(
        _extra_preferential_tax(30_000.0, 0.0, 70_000.0), [ADDON * 50_000.0]
    )
    # Dividends fill $30,000 to $90,000 first, so only the top $10,000 of the
    # top bracket is gains.
    np.testing.assert_allclose(
        _extra_preferential_tax(30_000.0, 60_000.0, 10_000.0), [ADDON * 10_000.0]
    )



def test_dividend_rate_addon_leaves_gains_alone():
    np.testing.assert_allclose(
        _extra_preferential_tax(30_000.0, 0.0, 70_000.0, WITH_DIVIDEND_ADDON), [0.0]
    )


def test_dividend_rate_addon_on_dividends_entirely_in_the_top_bracket():
    # Ordinary income of $60,000 already passes the $50,000 top threshold.
    np.testing.assert_allclose(
        _extra_preferential_tax(60_000.0, 40_000.0, 0.0, WITH_DIVIDEND_ADDON),
        [ADDON * 40_000.0],
    )


def test_dividend_rate_addon_on_dividends_straddling_the_top_threshold():
    # Dividends run from $30,000 to $100,000; the $50,000 above the threshold
    # pays.
    np.testing.assert_allclose(
        _extra_preferential_tax(30_000.0, 70_000.0, 0.0, WITH_DIVIDEND_ADDON),
        [ADDON * 50_000.0],
    )


def test_dividend_rate_addon_with_gains_stacked_above_the_dividends():
    # Dividends fill $30,000 to $60,000 and gains sit above them, so only the
    # $10,000 of dividends above $50,000 pays, however large the gains are.
    np.testing.assert_allclose(
        _extra_preferential_tax(30_000.0, 30_000.0, 40_000.0, WITH_DIVIDEND_ADDON),
        [ADDON * 10_000.0],
    )
    # Dividends that end below the threshold pay nothing, even when gains carry
    # the household far above it.
    np.testing.assert_allclose(
        _extra_preferential_tax(10_000.0, 20_000.0, 70_000.0, WITH_DIVIDEND_ADDON),
        [0.0],
    )

# A schedule padded with NaN slots is integrated over its live brackets only.
def test_schedule_tolerates_nan_padded_slots():
    padded = TaxLaw(
        ord_rates=jnp.array([0.10, 0.20, jnp.nan, jnp.nan]),
        ord_brackets=jnp.array([0.0, 10_000.0, jnp.nan, jnp.nan]),
        pref_rates=jnp.array([0.0, 0.15, jnp.nan, jnp.nan]),
        pref_brackets=jnp.array([0.0, 5_000.0, jnp.nan, jnp.nan]),
        unrecapture_rate=0.25,
        collectibles_rate=0.28,
        tax_at_ord=0.0,
        no_ord_cap=0.0,
        kg_top_rate_addon=0.0,
        div_top_rate_addon=0.0,
    )
    records = {
        "txbl_inc": jnp.array([20_000.0]),
        "div_pref": jnp.zeros(1),
        "kg_pref": jnp.zeros(1),
        "kg_1250": jnp.zeros(1),
        "kg_collect": jnp.zeros(1),
    }
    # Same two live brackets as TAX_LAW: $1,000 + $2,000 = $3,000.
    np.testing.assert_allclose(calc_tax(records, padded)["liab"], [3_000.0])


# The frozen compiled baseline is the shape the calculators will actually be
# handed: 16 slots per schedule, 7 of them live in the ordinary rate schedule
# and the other 9 NaN. This runs the ordinary schedule straight off that file
# for all four filing statuses.
def test_schedule_runs_on_the_frozen_compiled_law():
    fixture = (
        Path(__file__).resolve().parents[3]
        / "golden"
        / "fixtures"
        / "law"
        / "compiled_baseline.npz"
    )
    arrays = np.load(fixture)
    year = list(arrays["years"]).index(2026)
    brackets = arrays["value/ord.brackets"][year]
    rates = arrays["value/ord.rates"][year]
    active = arrays["active/ord.brackets"][year]
    # The point of the test is the padding, so make sure there is some.
    assert active.sum() < active.size

    # Four incomes per filing status, from below the first bracket to above the
    # top one.
    income = np.tile(np.array([0.0, 45_000.0, 400_000.0, 3_000_000.0]), (4, 1))
    actual = schedule_tax(
        jnp.asarray(income),
        jnp.asarray(brackets[:, None, :]),
        jnp.asarray(rates[:, None, :]),
    )
    # Compare against the same integration over the live slots only.
    expected = np.empty_like(income)
    for status in range(brackets.shape[0]):
        live_brackets = brackets[status][active[status]]
        live_rates = rates[status][active[status]]
        edges = np.append(live_brackets[1:], np.inf)
        for row, amount in enumerate(income[status]):
            slices = np.maximum(0.0, np.minimum(amount, edges) - live_brackets)
            expected[status, row] = (slices * live_rates).sum()
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(np.asarray(actual)).all()


def test_no_preferred_income_prices_the_next_dollar_of_gain_at_the_preferred_rate():
    """At the Schedule D limit the next dollar of gain is priced at the
    preferred rate.

    A household with nothing taxed at preferred rates sits exactly on the limit
    that stops Schedule D tax exceeding ordinary tax on all taxable income,
    because with no preferred income the two sides are the same number term
    for term. A dollar of gain raises Schedule D tax by less than ordinary
    tax, so it lands the household below the limit.

    The household below has $20,000 of taxable income and no preferred income at
    all, so its next dollar of gain sits in the second preferred bracket at 15
    percent while its ordinary rate is 20 percent. Those two differ under this
    test's invented law, which is the point: the wrong tie shows up as the wrong
    number rather than hiding behind a coincidence.
    """

    def liability(preferred_income):
        records = {
            "txbl_inc": jnp.array([20_000.0]) + preferred_income,
            "div_pref": jnp.zeros(1),
            "kg_pref": preferred_income,
            "kg_1250": jnp.zeros(1),
            "kg_collect": jnp.zeros(1),
        }
        return jnp.sum(calc_tax(records, TAX_LAW)["liab"])

    np.testing.assert_allclose(jax.grad(liability)(jnp.zeros(1)), [0.15])

    # The value itself does not depend on which way the tie goes: at exactly
    # zero preferred income the two sides are equal, so both branches return the
    # same liability. $10,000 at 10 percent and $10,000 at 20 percent.
    np.testing.assert_allclose(liability(jnp.zeros(1)), 3_000.0)
