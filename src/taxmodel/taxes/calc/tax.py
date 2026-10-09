"""Section 1 income tax, alternative maximum tax, and AMT."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import capped_at, floored_at, phased_out
from taxmodel.taxes.calc._schedules import (
    schedule_tax,
    stacked_tax_exclusive,
    stacked_tax_inclusive,
    top_bracket_only,
)

Array = jnp.ndarray


@dataclass(frozen=True)
class TaxLaw:
    """Ordinary and preferred Section 1 rate schedules and special-gain caps."""

    ord_rates: Array
    ord_brackets: Array
    pref_rates: Array
    pref_brackets: Array
    unrecapture_rate: Array
    collectibles_rate: Array
    tax_at_ord: Array
    no_ord_cap: Array
    kg_top_rate_addon: Array
    div_top_rate_addon: Array


@dataclass(frozen=True)
class AlternativeMaximumTaxLaw:
    """IRC §1A qualification and maximum-tax parameters."""

    rate: Array
    qualify_mult: Array
    exempt: Array


@dataclass(frozen=True)
class AmtLaw:
    """Alternative minimum tax preferences, exemption, and ordinary schedule."""

    pe_pref: Array
    exempt: Array
    exempt_po_thresh: Array
    exempt_po_rate: Array
    rates: Array
    brackets: Array


def calc_tax(records: Mapping[str, Array], law: TaxLaw) -> dict[str, Array]:
    """Return Section 1 liability allocated among four income buckets."""
    # The model partitions taxable income into ordinary income, non-special
    # preferred income, unrecaptured §1250 gain, and collectibles gain.
    adjusted_kg_pref = floored_at(
        records["kg_pref"] - records["kg_1250"] - records["kg_collect"], 0.0
    )
    preferred_income = records["div_pref"] + records["kg_pref"]
    adjusted_preferred_income = records["div_pref"] + adjusted_kg_pref
    taxable_ordinary_income = floored_at(records["txbl_inc"] - preferred_income, 0.0)
    taxable_adjusted_preferred_income = floored_at(
        capped_at(adjusted_preferred_income, records["txbl_inc"]), 0.0
    )

    # Unrecaptured §1250 gain fills the taxable-income remainder before
    # collectibles gain, matching the Schedule D worksheet's stacking order.
    taxable_1250 = floored_at(
        capped_at(
            capped_at(records["kg_1250"], records["kg_pref"]),
            records["txbl_inc"]
            - taxable_adjusted_preferred_income
            - taxable_ordinary_income,
        ),
        0.0,
    )
    taxable_ordinary_and_1250 = floored_at(
        taxable_ordinary_income + taxable_1250, 0.0
    )
    taxable_collectibles = floored_at(
        capped_at(
            capped_at(records["kg_collect"], records["kg_pref"]),
            records["txbl_inc"]
            - taxable_adjusted_preferred_income
            - taxable_ordinary_income
            - taxable_1250,
        ),
        0.0,
    )

    liab_pref = stacked_tax_inclusive(
        taxable_adjusted_preferred_income,
        records["txbl_inc"],
        law.pref_brackets,
        law.pref_rates,
    )
    # The additional rates on long-term gains and on qualified dividends apply
    # to the part of each taxed in the top preferential bracket. Qualified
    # dividends fill the preferential brackets before gains do, so gains are the
    # top slice of preferred income and dividends the slice just below it. The
    # additional tax is part of the preferential tax, and so counts toward the
    # amount the ordinary-tax cap below limits.
    taxable_adjusted_kg_pref = capped_at(
        adjusted_kg_pref, taxable_adjusted_preferred_income
    )
    liab_pref = liab_pref + stacked_tax_inclusive(
        taxable_adjusted_kg_pref,
        records["txbl_inc"],
        law.pref_brackets,
        top_bracket_only(law.pref_rates, law.kg_top_rate_addon),
    )
    liab_pref = liab_pref + stacked_tax_inclusive(
        taxable_adjusted_preferred_income - taxable_adjusted_kg_pref,
        records["txbl_inc"] - taxable_adjusted_kg_pref,
        law.pref_brackets,
        top_bracket_only(law.pref_rates, law.div_top_rate_addon),
    )
    # Preferred income occupies the top of taxable income for the Section 1
    # worksheet; special gains then stack above ordinary income in their order.
    # Both of these compare one law rate with another and never touch a record
    # amount, so no household derivative passes through them. They also have to
    # keep propagating the NaN that marks an unused bracket slot, which is how
    # live_slots knows to switch that slot off.
    unrecapture_rate = jnp.asarray(law.unrecapture_rate)[..., None]
    collectibles_rate = jnp.asarray(law.collectibles_rate)[..., None]
    unrecapture_rates = jnp.minimum(law.ord_rates, unrecapture_rate)  # tie: law rates only, and the unused-slot NaN must survive
    collectibles_rates = jnp.minimum(law.ord_rates, collectibles_rate)  # tie: law rates only, and the unused-slot NaN must survive
    liab_1250 = stacked_tax_exclusive(
        taxable_1250,
        taxable_ordinary_income,
        law.ord_brackets,
        unrecapture_rates,
    )
    liab_collect = stacked_tax_exclusive(
        taxable_collectibles,
        taxable_ordinary_and_1250,
        law.ord_brackets,
        collectibles_rates,
    )
    liab_1250 = capped_at(liab_1250, records["kg_1250"] * law.unrecapture_rate)
    liab_collect = capped_at(
        liab_collect, records["kg_collect"] * law.collectibles_rate
    )

    liab_ord = schedule_tax(taxable_ordinary_income, law.ord_brackets, law.ord_rates)
    liab_max = schedule_tax(records["txbl_inc"], law.ord_brackets, law.ord_rates)
    component_sum = liab_ord + liab_pref + liab_1250 + liab_collect
    # The Schedule D tax cannot exceed ordinary tax on all taxable income unless
    # the reform explicitly removes that cap; tax-at-ordinary overrides all
    # preferred and special-gain allocations.
    #
    # A household with no income at preferred rates sits exactly on this cap,
    # not near it: with nothing taken out of taxable income for preferred
    # treatment, `liab_ord` is `liab_max` and the other three pieces are zero,
    # so the two sides are the same number term for term. Which way that tie
    # goes decides the rate we report on such a household's next dollar of
    # long-term gain. One more dollar of gain raises the preferred computation
    # by the preferred rate and raises the cap by the higher ordinary rate, so
    # the household lands below the cap and the tie belongs to the sloped
    # branch. That is what preferred rates are: tax that grows more slowly than
    # ordinary tax on the same income, which is the sinking-against-the-limit
    # case (CONTRIBUTING.md, "Writing a threshold in a calculator"). The value
    # is the same either way; only the slope changes.
    liab = jnp.where(
        law.tax_at_ord != 0,
        liab_max,
        jnp.where(
            law.no_ord_cap != 0,
            component_sum,
            capped_at(component_sum, liab_max, sinks_against_the_limit=True),
        ),
    )
    at_preferred_rates = law.tax_at_ord == 0
    return {
        "liab_ord": jnp.where(at_preferred_rates, liab_ord, liab),
        "liab_pref": jnp.where(at_preferred_rates, liab_pref, 0.0),
        "liab_1250": jnp.where(at_preferred_rates, liab_1250, 0.0),
        "liab_collect": jnp.where(at_preferred_rates, liab_collect, 0.0),
        "liab": liab,
    }


def calc_alt_max(
    records: Mapping[str, Array], law: AlternativeMaximumTaxLaw
) -> dict[str, Array]:
    """Return whether the IRC §1A maximum tax binds and its capped liability."""
    # Model MAGI adds back nonincluded Social Security benefits; foreign earned
    # income exclusions named in IRC §1A are not observed in the microdata.
    magi = records["agi"] + floored_at(records["gross_ss"] - records["txbl_ss"], 0.0)
    alt_max_liab = law.rate * floored_at(magi - law.exempt, 0.0)
    # Section 63(c)(6) is approximated as an MFS filer who itemizes;
    # nonresident-alien and short-year-return status are not observed.
    # The itemizing flag is compared to zero so it may arrive as a bool or as a
    # 0/1 number.
    is_63c6 = (records["filing_status"] == 3) & (records["itemizing"] != 0)
    cap_binds = (
        (magi < law.qualify_mult * law.exempt)
        & ~is_63c6
        & (alt_max_liab < records["liab"])
    )
    return {"alt_max_cap_binds": cap_binds, "alt_max_liab": alt_max_liab}


def calc_amt(
    records: Mapping[str, Array], tax_law: TaxLaw, law: AmtLaw
) -> dict[str, Array]:
    """Return AMT and pre-credit liability after a second Section 1 schedule run."""
    # AMT adds back the standard deduction for non-itemizers. For itemizers it
    # adds SALT and miscellaneous deductions while restoring the Pease reduction.
    addback = jnp.where(
        records["itemizing"],
        records["salt_item_ded"]
        + records["misc_item_ded"]
        - (records["item_ded_ex_limits"] - records["item_ded"]),
        records["std_ded"],
    )
    amt_gross_inc = (
        records["agi"]
        - records["ded"]
        - records["pe_ded"] * (law.pe_pref == 0)
        - records["qbi_ded"]
        - records["tip_ded"]
        - records["ot_ded"]
        - records["senior_ded"]
        + addback
        - records["state_ref"]
        + records["amt_nols"]
    )
    # The AMT exemption phases out from AMT gross income before the alternative
    # ordinary schedule is applied.
    exemption = phased_out(
        law.exempt, amt_gross_inc, law.exempt_po_thresh, law.exempt_po_rate
    )
    amt_taxable_income = floored_at(amt_gross_inc - exemption, 0.0)

    # AMT substitutes only its ordinary schedule. Preferred income and special
    # gains retain the same Section 1 stacking and caps as the regular tax run.
    amt_tax_law = replace(tax_law, ord_rates=law.rates, ord_brackets=law.brackets)
    amt_records = dict(records)
    amt_records["txbl_inc"] = amt_taxable_income
    liab_amt_gross = calc_tax(amt_records, amt_tax_law)["liab"]
    # The alternative minimum tax is only what it adds to the regular tax, so
    # it is floored at zero. A household that owes no tax at all sits exactly
    # on that floor, both taxes being zero. Its next dollar of income creates
    # regular tax while the alternative minimum tax exemption still covers it,
    # so this amount sinks against the floor and the tie takes the flat branch.
    liab_amt = floored_at(
        liab_amt_gross - records["amt_ftc"] - records["liab"],
        0.0,
        sinks_against_the_limit=True,
    )
    return {
        "amt_gross_inc": amt_gross_inc,
        "amt_txbl_inc": amt_taxable_income,
        "liab_amt": liab_amt,
        "liab_bc": records["liab"] + liab_amt + records["excess_ptc"],
    }
