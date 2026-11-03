import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.txbl_inc import TaxableIncomeLaw, calc_txbl_inc

LAW = TaxableIncomeLaw(100000.0, 0.0, 0.0, 0.0)

# The non-itemizer charitable deduction from 2026 for a single filer, $1,000
# (IRC 170(p) as amended by OBBBA sec. 70424).
NONITEM_LIMIT_SINGLE = 1000.0
LAW_2026_SINGLE = TaxableIncomeLaw(100000.0, 0.0, 0.0, NONITEM_LIMIT_SINGLE)


def one_return(**columns):
    """One return with $100,000 of AGI and a $10,000 standard deduction."""
    data = {
        "agi": 100000.0,
        "std_ded": 10000.0,
        "item_ded": 0.0,
        "salt_item_ded": 0.0,
        "pe_ded": 0.0,
        "qbi_ded": 0.0,
        "tip_ded": 0.0,
        "ot_ded": 0.0,
        "senior_ded": 0.0,
        "auto_int_ded": 0.0,
        "char_cash": 0.0,
        "char_above_ded": 0.0,
    }
    data.update(columns)
    return {name: jnp.array([value]) for name, value in data.items()}


def test_exact_itemization_tie_takes_standard_deduction():
    # At the $10,000 tie the standard deduction is taken; the $1,000 senior
    # deduction raises it to $11,000, leaving $89,000 taxable income.
    out = calc_txbl_inc(one_return(item_ded=10000.0, senior_ded=1000.0), LAW)
    assert not out["itemizing"][0]
    np.testing.assert_allclose([out["ded"][0], out["txbl_inc"][0]], [11000.0, 89000.0])


def test_a_non_itemizer_deducts_cash_gifts_up_to_the_limit():
    # $600 of cash gifts is deducted in full; of $2,500, $1,000 is deducted.
    for gifts, deducted in ((600.0, 600.0), (2500.0, 1000.0)):
        out = calc_txbl_inc(one_return(char_cash=gifts), LAW_2026_SINGLE)
        assert not out["itemizing"][0]
        np.testing.assert_allclose(out["char_nonitem_ded"], [deducted])
        np.testing.assert_allclose(out["ded"], [10000.0 + deducted])
        np.testing.assert_allclose(out["txbl_inc"], [90000.0 - deducted])


def test_the_itemize_choice_counts_the_charitable_deduction_given_up():
    # Itemized deductions of $10,500 beat the $10,000 standard deduction alone
    # but not the standard deduction plus $1,000 of charitable deduction.
    out = calc_txbl_inc(one_return(item_ded=10500.0, char_cash=3000.0), LAW_2026_SINGLE)
    assert not out["itemizing"][0]
    np.testing.assert_allclose([out["ded"][0], out["txbl_inc"][0]], [11000.0, 89000.0])
    # At $11,500 itemizing wins, and an itemizer takes no non-itemizer deduction.
    out = calc_txbl_inc(one_return(item_ded=11500.0, char_cash=3000.0), LAW_2026_SINGLE)
    assert out["itemizing"][0]
    np.testing.assert_allclose(out["char_nonitem_ded"], [0.0])
    np.testing.assert_allclose([out["ded"][0], out["txbl_inc"][0]], [11500.0, 88500.0])


def test_a_gift_deducted_above_the_line_is_not_deducted_again():
    # $1,200 of cash gifts, $500 of it deducted above the line: $700 is left for
    # the non-itemizer deduction.
    out = calc_txbl_inc(
        one_return(char_cash=1200.0, char_above_ded=500.0), LAW_2026_SINGLE
    )
    np.testing.assert_allclose(out["char_nonitem_ded"], [700.0])


def test_no_non_itemizer_deduction_where_the_law_has_none():
    out = calc_txbl_inc(one_return(char_cash=3000.0), LAW)
    np.testing.assert_allclose(out["char_nonitem_ded"], [0.0])
    np.testing.assert_allclose(out["txbl_inc"], [90000.0])


def test_car_loan_interest_is_deducted_whether_or_not_the_filer_itemizes():
    # $2,000 of car loan interest comes off taxable income for a non-itemizer
    # ($100,000 - $10,000 - $2,000) and for an itemizer ($100,000 - $15,000 -
    # $2,000). It is not part of the deduction `ded` either way.
    out = calc_txbl_inc(one_return(auto_int_ded=2000.0), LAW)
    assert not out["itemizing"][0]
    np.testing.assert_allclose([out["ded"][0], out["txbl_inc"][0]], [10000.0, 88000.0])
    out = calc_txbl_inc(one_return(item_ded=15000.0, auto_int_ded=2000.0), LAW)
    assert out["itemizing"][0]
    np.testing.assert_allclose([out["ded"][0], out["txbl_inc"][0]], [15000.0, 83000.0])
