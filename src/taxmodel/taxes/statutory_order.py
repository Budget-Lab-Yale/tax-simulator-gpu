"""The tax calculation in statutory order: one year's records and law in, tax results out.

Every number here comes out of a function in `taxmodel.taxes.calc`. This module
supplies the order they run in, the columns each one hands the next, and a few
steps that belong to no single calculator.
The order changes the answer: credits are limited by the liability left after
the credits above them, and that liability is not known until the alternative
maximum tax cap and the AMT have both been applied.

Everything here is a pure function of arrays. The one decision taken in Python
is whether the charitable double run is needed, which is a fact about the law
(see `charity_double_run_applies`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp

from taxmodel.law.batch import charity_double_run_offered
from taxmodel.taxes.calc._corners import capped_at, floored_at
from taxmodel.taxes.calc.agi import calc_agi
from taxmodel.taxes.calc.credits import (
    calc_cdctc,
    calc_ctc,
    calc_ed_cred,
    calc_eitc,
    calc_rebate,
    calc_savers_cred,
    calc_wage_subsidy,
)
from taxmodel.taxes.calc.deductions import (
    calc_below_ded,
    calc_item_ded,
    calc_pe_ded,
    calc_qbi_ded,
    calc_std_ded,
)
from taxmodel.taxes.calc.estate import EstateParameters, calc_estate
from taxmodel.taxes.calc.kg import calc_kg
from taxmodel.taxes.calc.liab import calc_liab
from taxmodel.taxes.calc.niit import calc_niit
from taxmodel.taxes.calc.payroll import calc_pr
from taxmodel.taxes.calc.remit import remit_taxes
from taxmodel.taxes.calc.surtax import calc_agi_surtax
from taxmodel.taxes.calc.tax import calc_alt_max, calc_amt, calc_tax
from taxmodel.taxes.calc.txbl_inc import calc_txbl_inc
from taxmodel.taxes.calc.wealth import calc_wealth

# The column manifests live in their own module and are read off this one by
# everything that imports them, so the import below is part of this module's
# interface even where this file does not use a name itself.
from taxmodel.taxes.chain_columns import (  # noqa: F401
    CALCULATED_COLUMNS,
    DERIVED_COLUMNS,
    ESTATE_COLUMNS,
    ESTATE_RECORD_COLUMNS,
    FILER_AND_EXPANDED_INCOME_COLUMNS,
    ITEMIZED_COMPONENTS,
    ONE_1040_COLUMNS,
    PAYROLL_COLUMNS,
    RECORD_COLUMNS,
    REMITTANCE_COLUMNS,
    WEALTH_TAX_COLUMNS,
)
from taxmodel.taxes.law_for_calculators import YearLaw

Array = jnp.ndarray
Records = Mapping[str, Array]

def missing_record_columns(
    records: Records, with_estate: bool = False
) -> list[str]:
    """Columns the chain will read that the frame does not have."""
    wanted = RECORD_COLUMNS + (ESTATE_RECORD_COLUMNS if with_estate else ())
    return [name for name in wanted if name not in records]


# --------------------------------------------------------------------------
# Schedule E aggregates
# --------------------------------------------------------------------------


def derive_vars(records: Records) -> dict[str, Array]:
    """Schedule E aggregates, the same for every scenario and every year.

    Partnership and S-corporation results arrive split into active, passive
    and Section 179 pieces; the 1040 wants them netted. `part_scorp` and
    `part_scorp_loss` are the netted amount split into its positive and
    negative halves, which the EITC's investment-income test and the reporting
    tables use.
    """
    part = (
        records["part_active"]
        + records["part_passive"]
        - records["part_active_loss"]
        - records["part_passive_loss"]
        - records["part_179"]
    )
    scorp = (
        records["scorp_active"]
        + records["scorp_passive"]
        - records["scorp_active_loss"]
        - records["scorp_passive_loss"]
        - records["scorp_179"]
    )
    pass_through = part + scorp
    net_rent = records["rent"] - records["rent_loss"]
    net_estate = records["estate"] - records["estate_loss"]
    part_scorp = floored_at(pass_through, 0.0)
    # The loss half shrinks as pass-through income rises, so a unit with exactly
    # zero net pass-through income is treated as having the gain, not the loss.
    part_scorp_loss = floored_at(-pass_through, 0.0, sinks_against_the_limit=True)
    return {
        "part": part,
        "scorp": scorp,
        "part_scorp": part_scorp,
        "part_scorp_loss": part_scorp_loss,
        "pt": pass_through + records["sole_prop1"] + records["sole_prop2"],
        "net_rent": net_rent,
        "net_estate": net_estate,
        "sch_e": part_scorp + net_rent + net_estate,
    }


# --------------------------------------------------------------------------
# The 1040
# --------------------------------------------------------------------------


def _mask_charity(frame: dict[str, Array], mask: bool) -> dict[str, Array]:
    """Zero the charitable gifts, for one leg of the double run.

    `mask` is a plain Python bool: within a leg every filer is forced onto the
    same side of the 1040.
    """
    if not mask:
        return frame
    masked = dict(frame)
    masked["char_cash"] = jnp.zeros_like(frame["char_cash"])
    masked["char_noncash"] = jnp.zeros_like(frame["char_noncash"])
    return masked


def run_1040(
    records: Records,
    law: YearLaw,
    char_form: str | None = None,
) -> dict[str, Array]:
    """Run the individual income tax for one year, in statutory order.

    `records` holds the frame's columns plus whatever `calc_pr` produced;
    `law` is one year of compiled law bound to these records. `char_form` is
    the charitable double run's leg: `'above'` forces every filer to take the
    above-the-line deduction, `'item'` forces the Schedule A deduction, and
    `None`, the ordinary case, lets both apply as the data say.

    Returns the columns in `ONE_1040_COLUMNS`. Several columns are written more
    than once on the way down the form (the itemized components are zeroed for
    non-itemizers, and the four liability components are reduced by the
    alternative maximum tax cap), so the result is the finished return.
    """
    if char_form not in (None, "above", "item"):
        raise ValueError(f"char_form must be 'above', 'item' or None, got {char_form!r}")
    frame = dict(records)

    # -- adjusted gross income ---------------------------------------------

    frame.update(calc_kg(frame, law.kg, law.year))
    # Forcing the itemized form means the gift cannot also be deducted above
    # the line, so AGI is computed with the gift masked out.
    frame.update(calc_agi(_mask_charity(frame, char_form == "item"), law.agi))

    # -- taxable income -----------------------------------------------------

    frame.update(calc_std_ded(frame, law.std_ded))

    # Both mortgage-incurred years are set to infinity before the balance limit
    # is looked up, so every mortgage takes the newest, tightest vintage limit.
    # The vintage limits stay available to a reform.
    item_ded_inputs = _mask_charity(frame, char_form == "above")
    item_ded_inputs["first_mort_year"] = jnp.full_like(
        jnp.asarray(frame["first_mort_bal"], dtype=jnp.float64), jnp.inf
    )
    item_ded_inputs["second_mort_year"] = item_ded_inputs["first_mort_year"]
    frame.update(calc_item_ded(item_ded_inputs, law.item_ded))

    frame.update(calc_pe_ded(frame, law.pe_ded))
    # The QBI deduction forms its own provisional taxable income from the
    # larger of the standard and itemized deductions, before the itemize
    # choice is made and before the tip, overtime and senior deductions exist,
    # in the order the statute sets.
    frame.update(calc_qbi_ded(frame, law.qbi_ded))
    frame.update(calc_below_ded(frame, law.below_ded))
    frame.update(calc_txbl_inc(frame, law.txbl_inc))

    # The itemized total carried forward is the one after the OBBBA tax-value
    # clawback, and the components are zeroed for whoever ends up taking the
    # standard deduction (and the standard deduction for whoever itemizes),
    # so that no return reports both.
    itemizing = frame["itemizing"]
    zeroed = {
        column: jnp.where(itemizing, frame[column], 0.0)
        for column in ITEMIZED_COMPONENTS
    }
    zeroed["item_ded"] = jnp.where(itemizing, frame["item_ded_limited"], 0.0)
    zeroed["std_ded"] = jnp.where(itemizing, 0.0, frame["std_ded"])
    zeroed["itemizing"] = itemizing
    frame.update(zeroed)

    # -- liability before credits -------------------------------------------

    frame.update(calc_tax(frame, law.tax))
    frame.update(calc_alt_max(frame, law.alt_max))

    # Where the Section 1A maximum-tax cap binds, the saving is taken off the
    # four liability components in the order they were stacked: ordinary
    # income first, then preferred income, then unrecaptured Section 1250
    # gain, then collectibles, because reporting needs the four pieces as well
    # as the total.
    binds = frame["alt_max_cap_binds"]
    remaining = jnp.where(binds, frame["liab"] - frame["alt_max_liab"], 0.0)
    alt_max_ord = capped_at(remaining, frame["liab_ord"])
    remaining = remaining - alt_max_ord
    alt_max_pref = capped_at(remaining, frame["liab_pref"])
    remaining = remaining - alt_max_pref
    alt_max_1250 = capped_at(remaining, frame["liab_1250"])
    remaining = remaining - alt_max_1250
    alt_max_coll = capped_at(remaining, frame["liab_collect"])
    frame.update(
        {
            "alt_max_rem": remaining,
            "alt_max_ord": alt_max_ord,
            "alt_max_pref": alt_max_pref,
            "alt_max_1250": alt_max_1250,
            "alt_max_coll": alt_max_coll,
            "liab_ord": frame["liab_ord"] - alt_max_ord,
            "liab_pref": frame["liab_pref"] - alt_max_pref,
            "liab_1250": frame["liab_1250"] - alt_max_1250,
            "liab_collect": frame["liab_collect"] - alt_max_coll,
            "liab": jnp.where(binds, frame["alt_max_liab"], frame["liab"]),
        }
    )

    # The AMT runs the whole Section 1 calculation a second time with the AMT
    # rate schedule in place of the ordinary one, and adds the excess premium
    # tax credit. Liability before credits (`liab_bc`) comes out of here, and
    # every credit below is limited against it.
    frame.update(calc_amt(frame, law.tax, law.amt))

    # -- credits, in the order they claim liability -------------------------

    frame.update(calc_cdctc(frame, law.cdctc))
    frame.update(calc_ed_cred(frame, law.ed_cred))
    frame.update(calc_savers_cred(frame, law.savers_cred))
    # The caregiver credit is booked at zero. Calling
    # calc.credits.calc_caregiver_cred in place of these zeros turns it on.
    zeros = jnp.zeros_like(frame["liab_bc"])
    frame.update({"caregiver_cred_nonref": zeros, "caregiver_cred_ref": zeros})
    frame.update(calc_ctc(frame, law.ctc))
    frame.update(calc_eitc(frame, law.eitc))
    frame.update(calc_rebate(frame, law.rebate))
    frame.update(calc_wage_subsidy(frame, law.wage_subsidy))

    # -- allocating the finished liability ----------------------------------

    frame.update(calc_niit(frame, law.niit))
    frame.update(calc_agi_surtax(frame, law.surtax))
    frame.update(calc_liab(frame, law.liab))

    return {name: frame[name] for name in ONE_1040_COLUMNS}


# --------------------------------------------------------------------------
# The charitable double run
# --------------------------------------------------------------------------


def charity_double_run_applies(law: YearLaw) -> bool:
    """Whether a filer can choose where to report a charitable gift.

    When the law offers both an above-the-line charitable deduction and a
    Schedule A one, the two are mutually exclusive on the form, so the filer
    computes the return both ways and reports whichever costs less. That means
    running the whole 1040 twice.

    The answer comes from `law/batch.charity_double_run_offered`, applied to
    the bound law. It is computed in Python from the law arrays before the
    chain is traced, so each law compiles to one program.

    A law that has not yet been read for records holds its four filing-status
    values, so the question is asked of the four filing statuses. That gives
    the same answer for any population containing all four.
    """
    return charity_double_run_offered(law.agi.char_above_limit, law.item_ded.char_limit)


def charity_double_run_for_projection(
    year_laws: Sequence[YearLaw],
    years: Sequence[int],
    description: str = "this run",
) -> bool:
    """Whether this projection computes the 1040 twice, settled for all its years.

    This question changes which functions the model traces, so a run settles
    it once and holds every year to the same answer. A law that offers the
    choice in some projection years and not others is refused with an
    explanation. ``description`` is what to call the run in that refusal.
    """
    answer: bool | None = None
    settled_by: int | None = None
    for law, year in zip(year_laws, years, strict=True):
        this_year = charity_double_run_applies(law)
        if answer is None:
            answer, settled_by = this_year, int(year)
            continue
        if this_year != answer:
            offers = int(year) if this_year else settled_by
            withholds = settled_by if this_year else int(year)
            raise ValueError(
                f"{description} offers the choice between the "
                "above-the-line charitable deduction and the Schedule A one in "
                f"{offers} but not in {withholds}. The two are mutually "
                "exclusive on the form, so offering both means computing the "
                "whole return twice and taking whichever costs the filer less "
                "— a different program, not a different number, and a run "
                "traces one program. Split the projection into windows that "
                "answer the question the same way and run them separately."
            )
    return bool(answer)


def _choose_cheaper_leg(
    above: Mapping[str, Array], item: Mapping[str, Array]
) -> tuple[Array, dict[str, Array]]:
    """Pick each filer's better charitable form, and take that leg whole.

    The comparison is on income tax net of refundable credits, and ties go to
    the above-the-line form. The choice is per record and applies to every
    column at once, since a return is filed one way or the other.
    """
    prefers_above = above["liab_iit_net"] <= item["liab_iit_net"]
    chosen = {
        name: jnp.where(prefers_above, above[name], item[name])
        for name in ONE_1040_COLUMNS
    }
    return prefers_above, chosen


# --------------------------------------------------------------------------
# Payroll, income tax, estate, and the reporting columns on top
# --------------------------------------------------------------------------


def run_taxes(
    records: Records,
    law: YearLaw,
    estate_params: EstateParameters | None = None,
    double_run: bool | None = None,
) -> dict[str, Array]:
    """Every tax the model computes for one record-year, in statutory order.

    Payroll tax first, because the employer's half of self-employment tax is
    an above-the-line deduction and so has to exist before AGI. Then the 1040,
    once or twice depending on the charitable rule. Then the estate tax, which
    is computed conditional on the record dying this year and never feeds back
    into income tax, and the wealth tax on net worth (zero under baseline
    law). Then the reporting columns from `filer_and_expanded_income`, and last
    the remittance split: which part of each liability is withheld, paid
    quarterly, or due at filing.

    `estate_params` is the frozen SOI measurement bridge from the frame layer.
    Left out, the estate tax is skipped.

    `double_run` answers the charitable question ahead of time. Left out, it
    is answered from the law here. A caller compiling the chain has to answer
    it before tracing starts, because by then the law is a placeholder and not
    a number.

    Returns the computed columns only; the caller keeps its own record columns.
    """
    absent = missing_record_columns(records, with_estate=estate_params is not None)
    if absent:
        raise KeyError(
            "the frame is missing "
            f"{len(absent)} column(s) the calculators read: {', '.join(absent)}"
        )
    out: dict[str, Array] = {}
    frame = dict(records)

    def keep(produced: dict[str, Array]) -> None:
        """What a step produced goes out to the caller and on to the next step."""
        out.update(produced)
        frame.update(produced)

    keep(derive_vars(frame))
    keep(calc_pr(frame, law.payroll))

    if double_run is None:
        double_run = charity_double_run_applies(law)
    if double_run:
        above = run_1040(frame, law, "above")
        item = run_1040(frame, law, "item")
        _, income_tax = _choose_cheaper_leg(above, item)
    else:
        income_tax = run_1040(frame, law, None)
    keep(income_tax)

    if estate_params is not None:
        keep(calc_estate(frame, law.estate, estate_params))

    keep(calc_wealth(frame["net_worth"], law.wealth))
    keep(filer_and_expanded_income(frame))
    keep(remit_taxes(frame))
    return out



def filer_and_expanded_income(records: Records) -> dict[str, Array]:
    """Four reporting columns added after the taxes are computed.

    A household with no filing requirement still files when a refundable
    credit is worth money to it, so the filer flag is recomputed once the child
    credit and the rebate are known. It goes out as `filer_recomputed`; the
    input column `filer` is Tax-Data's own and is left as it arrived.

    Expanded income is the distribution tables' income measure: gross realized
    income before any deduction, plus the employer's share of payroll tax,
    plus the pass-through income that the SALT workaround moved off the return.

    A simple filer is one whose return the IRS could plausibly pre-fill:
    no itemizing and nothing but wages and Social Security.
    """
    became_filer = (records["become_filer_ctc"] == 1) | (
        records["become_filer_rebate"] == 1
    )
    filer_recomputed = records["filer"] + became_filer.astype(records["filer"].dtype)
    expanded_inc = (
        records["wages_base1"]
        + records["tips1"]
        + records["ot1"]
        + records["wages_base2"]
        + records["tips2"]
        + records["ot2"]
        + records["trad_contr_er1"]
        + records["trad_contr_er2"]
        + records["txbl_int"]
        + records["exempt_int"]
        + records["div_ord"]
        + records["div_pref"]
        + records["state_ref"]
        + records["txbl_ira_dist"]
        + records["gross_pens_dist"]
        + records["kg_st"]
        + records["kg_lt"]
        + records["other_gains"]
        + records["alimony"]
        + records["sole_prop1"]
        + records["sole_prop2"]
        + records["sch_e"]
        + records["farm1"]
        + records["farm2"]
        + records["gross_ss"]
        + records["ui"]
        + records["other_inc"]
        + records["salt_workaround_part"]
        + records["salt_workaround_scorp"]
        + records["liab_pr_er"]
    )
    nothing_but_wages_and_social_security = (
        (records["txbl_int"] == 0)
        & (records["div_ord"] == 0)
        & (records["div_pref"] == 0)
        & (records["state_ref"] == 0)
        & (records["txbl_ira_dist"] == 0)
        & (records["gross_pens_dist"] == 0)
        & (records["kg_st"] == 0)
        & (records["kg_lt"] == 0)
        & (records["other_gains"] == 0)
        & (records["alimony"] == 0)
        & (records["sole_prop1"] + records["sole_prop2"] == 0)
        & (records["sch_e"] == 0)
        & (records["farm1"] + records["farm2"] == 0)
        & (records["ui"] == 0)
        & (records["tip_ded"] == 0)
        & (records["ot_ded"] == 0)
        & (records["auto_int_ded"] == 0)
    )
    simple_filer = (
        (filer_recomputed != 0)
        & (records["itemizing"] == 0)
        & nothing_but_wages_and_social_security
    )
    produced = {
        "filer_recomputed": filer_recomputed,
        "expanded_inc": expanded_inc,
        "simple_filer": simple_filer.astype(jnp.int64),
    }
    # The entity shifting module's `corp_tax_change` is passed through where
    # that module ran and set to zero where it did not, so the revenue tables
    # can always find it.
    if "corp_tax_change" in records:
        produced["corp_tax_change"] = jnp.asarray(records["corp_tax_change"])
    else:
        produced["corp_tax_change"] = jnp.zeros_like(filer_recomputed)
    return produced
