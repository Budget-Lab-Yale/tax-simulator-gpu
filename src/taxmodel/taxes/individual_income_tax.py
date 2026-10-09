"""Marginal tax rates from the tax calculators.

The calculator takes a household frame and one year of bound law, then works
down the return: payroll tax, AGI, deductions, taxable income, credits, and
final liability. Differentiating that whole calculation carries an input dollar
through every one of those steps, so the marginal rate comes out of the same
code that produces the liability.

There are four outcomes, and every function here requires the caller to name
one, because "the marginal tax rate" is not defined until the taxes counted in
it are named:

- `net_income_tax` is individual income tax after refundable credits. A
  household whose refundable credits exceed its tax has a negative value.
- `income_tax_plus_employee_payroll` adds the employee's side of FICA and SECA
  plus the additional Medicare tax. That is the price a worker actually faces
  on a dollar of earnings.
- `wealth_tax` is the annual tax on net worth, zero under a law that has none.
- `estate_tax` is the estate tax the household owes if it dies this year. It is
  a price conditional on death, not an annual liability, and it needs the
  estate measurement bridge, so it is built by `estate_tax_outcome`.

The income-plus-employee-payroll outcome is the complete price of a dollar of
earnings, and the one a labor-supply response should use.

The deemed-gain tax has no outcome function, because nothing asks for its
marginal rate.

Callers read gradients by frame column name: a caller that wants a dollar of
the primary earner's ordinary pay reads `gradient["wages_base1"]`. No dollar is
stored in two columns, so almost every economic change is a change in a single
column. The exception the law itself creates is documented below as
`GENERAL_PARTNER_DOLLAR`.

At an exact corner (a bracket edge, a phase-out start) the calculators treat a
household sitting exactly on the threshold as having crossed it in the
direction of more income. `dollar_increment` reruns the calculator a dollar
higher and subtracts, so it needs no corner convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.taxes import statutory_order
from taxmodel.taxes.calc.estate import (
    ESTATE_ASSET_COLS,
    ESTATE_DEBT_COLS,
    EstateParameters,
    calc_estate,
)
from taxmodel.taxes.calc.wealth import calc_wealth
from taxmodel.taxes.law_for_calculators import YearLaw

Array = jnp.ndarray
Records = Mapping[str, Array]
Direction = Mapping[str, Real]
# An outcome takes the household frame and a bound annual law and returns one
# liability per household. Every outcome accepts `double_run`.
Outcome = Callable[..., Array]
# An outcome read off the columns `statutory_order.run_taxes` produced.
# `taxes_and_derivatives` runs the calculators once and reads several outcomes
# off that one run, so it takes outcomes in this form.
OutcomeFromResults = Callable[[Mapping[str, Array]], Array]


# The frame column that adds a dollar to the taxable estate itself. See the
# comment at the end of `CONTINUOUS_TAX_INPUTS`.
ESTATE_BASE_DOLLAR = "estate_base_bump"


# The household's continuous calculator inputs: an infinitesimal change in each
# of these means something a tax economist would recognize. All but three are
# dollar amounts; the three exceptions are named under DOLLAR_TAX_INPUTS below.
#
# Nothing here overlaps anything else. An earner's W-2 wages are
# `wages_base1 + tips1 + ot1`, and the household's wage line is that same sum
# over both earners; each calculator adds up the pieces it wants. So a dollar
# of the primary earner's ordinary pay is a change in `wages_base1` alone, and
# a dollar of tips is a change in `tips1` alone (a dollar of tips is also a
# dollar of wages, but the tip deduction and the payroll-tax exemption treat it
# differently).
CONTINUOUS_TAX_INPUTS = (
    "wages_base1",
    "wages_base2",
    "tips1",
    "tips2",
    "tips_lh1",
    "tips_lh2",
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
    "ei_prior_yr",
    "part_active",
    "part_active_loss",
    "part_passive",
    "part_passive_loss",
    "part_179",
    "scorp_active",
    "scorp_active_loss",
    "scorp_passive",
    "scorp_passive_loss",
    "scorp_179",
    "rent",
    "rent_loss",
    "estate",
    "estate_loss",
    "wagebill_sole_prop",
    "wagebill_part",
    "wagebill_scorp",
    "wagebill_farm",
    "txbl_int",
    "exempt_int",
    "div_ord",
    "div_pref",
    "state_ref",
    "txbl_ira_dist",
    "txbl_pens_dist",
    "gross_pens_dist",
    "gross_ss",
    "ui",
    "alimony",
    "other_inc",
    "other_gains",
    "kg_st",
    "kg_lt",
    "kg_lt_basis",
    "kg_lt_cpi_ratio",
    "kg_1250",
    "kg_collect",
    "ed_exp",
    "hsa_contr",
    "trad_contr_ira",
    "keogh_contr",
    "se_health",
    "early_penalty",
    "alimony_exp",
    "tuition_ded",
    "dpad",
    "other_above_ded",
    "sl_int_ded",
    "auto_int_exp",
    "med_exp",
    "salt_inc_sales",
    "salt_prop",
    "salt_pers",
    "first_mort_bal",
    "second_mort_bal",
    "first_mort_int",
    "second_mort_int",
    "inv_int_exp",
    "casualty_exp",
    "job_exp",
    "tax_prep_exp",
    "other_misc_exp",
    "other_item_exp",
    "char_cash",
    "char_noncash",
    "ftc",
    "amt_ftc",
    "amt_nols",
    "excess_ptc",
    "net_ptc",
    "care_exp",
    "aoc_exp",
    "llc_exp",
    "savers_cred",
    "old_cred",
    "res_energy_cred",
    "gbc",
    "prior_yr_cred",
    "other_nonref",
    "recapture_tax",
    "ira_penalty",
    "salt_workaround_part",
    "salt_workaround_scorp",
    # Net worth is here because the frame carries it and the classification
    # below has to account for every column the chain reads. Nothing on the
    # 1040 reads it, and the wealth tax below rebuilds it from the balance
    # sheet, so its derivative is zero for every outcome this module offers.
    "net_worth",
    # The household's balance sheet: fourteen kinds of asset and six kinds of
    # debt. These are the inputs the wealth tax and the estate tax are charged
    # on. A rate on net worth itself is not offered, because net worth is not an
    # input: it is assets less debts, and each of those is a column a household
    # can hold more or less of.
    *ESTATE_ASSET_COLS,
    *ESTATE_DEBT_COLS,
    # A dollar added straight to the taxable estate, downstream of the bridge
    # that turns economic wealth into the reported gross estate. Every real
    # household holds zero of it; it exists so that the estate tax's rate on a
    # dollar of its own taxable base can be measured, which is a different
    # number from the rate on a dollar of any particular asset.
    # `taxes/calc/estate.py` adds it inside the zero floor and defaults it to
    # zero for a frame that does not carry it.
    ESTATE_BASE_DOLLAR,
)

_CONTINUOUS_TAX_INPUT_SET = frozenset(CONTINUOUS_TAX_INPUTS)

# Every balance-sheet column, for a caller that wants the wealth tax's or the
# estate tax's gradient over all of them.
BALANCE_SHEET_INPUTS: tuple[str, ...] = (*ESTATE_ASSET_COLS, *ESTATE_DEBT_COLS)

# Three continuous inputs are not measured in dollars, so "one more dollar of
# it" means nothing and `dollar_increment` refuses them:
#
# - `kg_lt_cpi_ratio` is the price-index ratio a reform uses when it indexes
#   the basis of a long-held asset.
# - `tips_lh1` and `tips_lh2` are shares, not amounts. Each is the fraction of
#   that earner's tips received in a traditionally tipped occupation, and the
#   tip deduction multiplies the share against the earner's tips
#   (`taxes/calc/deductions.py`, `calc_below_ded`). The derivative with respect to a
#   share is meaningful, since it says what widening the qualifying occupation
#   list is worth to the household, but it is not a rate per dollar.
_NONDOLLAR_CONTINUOUS_INPUTS = ("kg_lt_cpi_ratio", "tips_lh1", "tips_lh2")
DOLLAR_TAX_INPUTS = tuple(
    name for name in CONTINUOUS_TAX_INPUTS if name not in _NONDOLLAR_CONTINUOUS_INPUTS
)
_DOLLAR_TAX_INPUT_SET = frozenset(DOLLAR_TAX_INPUTS)

# Filing status, ages, counts, eligibility flags, holding periods and the
# random draws the calculators consume are real calculator inputs, but an
# infinitesimal change in them is not an interpretable tax price. Writing them
# out means a new frame column forces a classification decision instead of
# quietly ending up with no derivative: a test checks that these two lists
# together are exactly the columns the chain reads, which is
# `statutory_order.RECORD_COLUMNS` plus the estate tax's own inputs.
NONDIFFERENTIABLE_TAX_INPUTS = (
    # The probability the second spouse is already dead. It scales the estate
    # tax, but a derivative with respect to a mortality probability is a
    # demographic quantity rather than a tax price.
    "q_death2",
    "filing_status",
    "filer",
    "dep_status",
    "age1",
    "age2",
    "blind1",
    "blind2",
    "n_dep",
    "n_dep_eitc",
    "dep_age1",
    "dep_age2",
    "dep_age3",
    "dep_ssn1",
    "dep_ssn2",
    "dep_ssn3",
    "dep_ctc1",
    "dep_ctc2",
    "dep_ctc3",
    "sstb_sole_prop",
    "sstb_part",
    "sstb_scorp",
    "sstb_farm",
    "divorce_year",
    "kg_lt_years_held",
    "r.bus_loss",
    "r.cdctc_takeup",
    "r.new_car",
    "r.eitc_precert",
)

# An economic change that moves two frame columns at once.
#
# A general partner who works in the partnership earns a dollar that is both
# ordinary partnership income on Schedule E (`part_active`) and net earnings
# from self-employment subject to SECA (`part_se1`). The law puts the same
# dollar in both bases, so the derivative for that partner is the sum of two
# partial derivatives:
#
#     gradient["part_active"] + gradient["part_se1"]
#
# `part_active` on its own is a limited partner's dollar: partnership income
# that is not self-employment earnings. `part_se1` on its own has no meaning,
# because self-employment earnings that are not partnership income belong to a
# sole proprietor and the frame stores those as `sole_prop1`. So
# `dollar_increment` refuses to move `part_se1` or `part_se2` without
# `part_active`.
#
# The self-employment half is always `part_se1`, the first earner's column,
# whichever spouse is the partner. On a joint return whose general partner is
# the second earner, that attributes the self-employment earnings to the wrong
# spouse and can put them under the wrong OASDI taxable maximum. A caller that
# wants the other spouse adds `gradient["part_active"] + gradient["part_se2"]`
# itself.
GENERAL_PARTNER_DOLLAR: Direction = {"part_active": 1.0, "part_se1": 1.0}


def net_income_tax(
    records: Records,
    law: YearLaw,
    *,
    double_run: bool | None = None,
) -> Array:
    """Individual income tax after refundable credits, one value per household.

    Payroll, estate, wealth and deemed-gain liabilities are all excluded.

    `law` is any bound annual law. This function has no notion of baseline or
    reform; a caller that wants both calls it twice.

    `double_run` says whether the law lets a filer choose between an
    above-the-line and an itemized charitable deduction. An ordinary call may
    leave it out and have the answer read off the law. Code putting this
    function inside `jax.jit` has to pass the plain Python boolean, because by
    tracing time the law is a placeholder rather than a number.
    """
    return net_income_tax_from_results(
        statutory_order.run_taxes(records, law, double_run=double_run)
    )


def net_income_tax_from_results(results: Mapping[str, Array]) -> Array:
    """`net_income_tax` read off a calculation that has already been done."""
    return results["liab_iit_net"]


def income_tax_plus_employee_payroll(
    records: Records,
    law: YearLaw,
    *,
    double_run: bool | None = None,
) -> Array:
    """Income tax after refundable credits, plus the employee's payroll tax.

    The payroll piece is `liab_pr_ee`: the employee's half of FICA, the
    employee's half of SECA, and the additional Medicare tax, which has no
    employer match. The employer's half is left out because this is the price
    facing the worker, not the whole tax wedge on the job.

    The arguments mean what they mean in `net_income_tax`.
    """
    return income_tax_plus_employee_payroll_from_results(
        statutory_order.run_taxes(records, law, double_run=double_run)
    )


def income_tax_plus_employee_payroll_from_results(
    results: Mapping[str, Array],
) -> Array:
    """`income_tax_plus_employee_payroll` read off a finished calculation."""
    return results["liab_iit_net"] + results["liab_pr_ee"]


def net_worth_from_balance_sheet(records: Records) -> Array:
    """Assets less debts, added up from the balance-sheet columns.

    The frame already carries a `net_worth` column, and for a real household it
    holds exactly this sum: `frame/steps_jax.add_net_worth` forms it that way, the
    corporate-capitalization adjuster re-forms it that way after it marks asset
    values down, and the wealth bathtub re-forms it that way after it takes the
    carried shortfall out of the balance sheet.

    The sum is taken here so that the derivative with respect to a single
    balance-sheet column, such as `value.equities`, is the tax on one more
    dollar of that asset. A missing balance-sheet entry counts as zero.

    The sum differs from the frame's column only in slots that hold no
    household. `frame/steps_jax.add_net_worth` writes zero into a slot whose
    `valid` flag is false and this sum does not, so an empty slot left holding a
    previous year's balance sheet gets a rate here. An empty slot has zero
    population weight, and the per-household detail file carries the validity
    flag so that a reader can drop those rows.
    """
    missing = [name for name in BALANCE_SHEET_INPUTS if name not in records]
    if missing:
        raise KeyError(
            "the frame is missing balance-sheet column(s) the wealth tax is "
            "charged on: " + ", ".join(missing)
        )
    assets = sum(
        (jnp.nan_to_num(jnp.asarray(records[name]), nan=0.0) for name in ESTATE_ASSET_COLS),
        jnp.asarray(0.0),
    )
    debts = sum(
        (jnp.nan_to_num(jnp.asarray(records[name]), nan=0.0) for name in ESTATE_DEBT_COLS),
        jnp.asarray(0.0),
    )
    return assets - debts


def wealth_tax(
    records: Records,
    law: YearLaw,
    *,
    double_run: bool | None = None,
) -> Array:
    """The annual wealth tax, one value per household.

    Zero for every household under a law with no wealth tax.

    The tax is charged on net worth rebuilt from the balance sheet, for the
    reason `net_worth_from_balance_sheet` gives. It equals the chain's
    `liab_wealth`, because the frame's `net_worth` column is that same sum.

    `double_run` is accepted and ignored, since the charitable choice is an
    income-tax question. It is in the signature so this function can be used
    anywhere the other outcomes can.
    """
    del double_run
    return calc_wealth(net_worth_from_balance_sheet(records), law.wealth)["liab_wealth"]


def estate_tax(
    records: Records,
    law: YearLaw,
    estate_params: EstateParameters,
    *,
    double_run: bool | None = None,
) -> Array:
    """The estate tax the household owes if it dies this year, per household.

    This is a price conditional on death. It is not multiplied by the
    household's mortality probability, which is how the population totals turn
    it into expected revenue; a behavioral response to the estate tax is a
    response to the tax a decedent's estate would actually pay.

    Whether the surviving spouse's unused exclusion is available is not known
    at the time the tax is computed, so the estate calculator returns the
    liability both ways along with the probability of the portability election.
    The price here is the average of the two weighted by that probability,
    which is the same combination the published estate totals report
    (`years/one_year.py` forms it per household, `reporting/tax_totals.py` writes
    it out).

    `estate_params` is the frozen bridge from economic net worth to the SOI
    reported gross estate. There is no estate tax without it, so it has no
    default; `estate_tax_outcome` binds it for a caller that needs the plain
    outcome signature.

    `double_run` is accepted and ignored, the same as in `wealth_tax`.
    """
    del double_run
    return estate_tax_from_results(calc_estate(records, law.estate, estate_params))


def estate_tax_from_results(results: Mapping[str, Array]) -> Array:
    """`estate_tax` read off a calculation that has already been done.

    The columns it reads are the estate calculator's own three, which the chain
    also puts in its results, so this works on either.
    """
    probability = results["estate_p_dsue"]
    return (
        probability * results["liab_estate_dsue"]
        + (1.0 - probability) * results["liab_estate_nodsue"]
    )


def estate_tax_outcome(estate_params: EstateParameters) -> Outcome:
    """`estate_tax` with the estate measurement bridge bound to it."""

    def outcome(
        records: Records, law: YearLaw, *, double_run: bool | None = None
    ) -> Array:
        return estate_tax(records, law, estate_params, double_run=double_run)

    return outcome


def derivative(
    records: Records,
    law: YearLaw,
    *,
    outcome: Outcome,
    input_names: Sequence[str],
    double_run: bool | None = None,
) -> dict[str, Array]:
    """Differentiate a tax outcome with respect to continuous household inputs.

    `outcome` is one of the four this module offers: `net_income_tax`,
    `income_tax_plus_employee_payroll`, `wealth_tax`, or what
    `estate_tax_outcome` returns. The caller chooses which taxes belong in the
    price.

    `input_names` lists the frame columns to differentiate with respect to. The
    result has one array per requested input and one value per household,
    keyed by frame column name. All requested inputs come out of a single
    reverse-mode pass.

    The calculation sums liability across households before differentiating.
    Each household still gets its own derivative, because the calculator is
    rowwise: one household's inputs never enter another household's return.
    """
    names, changing, fixed, double_run = _split_for_differentiation(
        records, law, input_names, double_run
    )

    def total_liability(inputs: Mapping[str, Array], annual_law: YearLaw) -> Array:
        complete = dict(fixed)
        complete.update(inputs)
        return jnp.sum(outcome(complete, annual_law, double_run=double_run))

    calculated = jax.grad(total_liability, argnums=0)(changing, law)
    return {name: calculated[name] for name in names}


def taxes_and_derivatives(
    records: Records,
    law: YearLaw,
    estate_params: EstateParameters | None,
    outcomes: Mapping[str, OutcomeFromResults],
    *,
    input_names: Sequence[str],
    double_run: bool | None = None,
) -> tuple[dict[str, Array], dict[str, dict[str, Array]]]:
    """Run the calculators once and take several derivatives off that one run.

    `jax.vjp` runs the calculation forward and returns a function that carries
    a derivative backwards through it. Calling that function once per outcome
    gives each outcome its own gradient from a single forward pass. The
    liabilities from that pass are returned alongside the gradients.

    `outcomes` names each price and says how to read it off the finished
    results: the individual income tax is the `liab_iit_net` column, and the
    price a worker faces is that column plus `liab_pr_ee`. Only an outcome that
    can be read off one run of the chain can share that run. The wealth tax is
    charged on net worth added up from the balance sheet (see
    `net_worth_from_balance_sheet`), so it keeps its own `derivative` call.

    Returns the chain's output columns and one gradient per named outcome,
    keyed as in `derivative`: frame column name to one value per household.
    """
    if not outcomes:
        raise ValueError("at least one outcome is needed to differentiate")
    names, changing, fixed, double_run = _split_for_differentiation(
        records, law, input_names, double_run
    )

    outcome_names = tuple(outcomes)

    def forward(
        inputs: Mapping[str, Array],
    ) -> tuple[tuple[Array, ...], dict[str, Array]]:
        complete = dict(fixed)
        complete.update(inputs)
        results = statutory_order.run_taxes(
            complete, law, estate_params, double_run=double_run
        )
        # Summing each outcome across households before differentiating gives
        # every household its own derivative anyway, because the calculators are
        # rowwise: one household's inputs never enter another's return.
        totals = tuple(jnp.sum(outcomes[name](results)) for name in outcome_names)
        return totals, results

    totals, backwards, results = jax.vjp(forward, changing, has_aux=True)

    gradients: dict[str, dict[str, Array]] = {}
    for position, name in enumerate(outcome_names):
        # One dollar of this outcome and nothing of the others: the backward
        # pass then reports how much each input changes this outcome alone.
        seed = tuple(
            jnp.ones_like(total) if slot == position else jnp.zeros_like(total)
            for slot, total in enumerate(totals)
        )
        (whole_gradient,) = backwards(seed)
        gradients[name] = {column: whole_gradient[column] for column in names}
    return dict(results), gradients


def dollar_increment(
    records: Records,
    law: YearLaw,
    direction: Direction,
    *,
    outcome: Outcome,
    amount: Real = 1.0,
    double_run: bool | None = None,
) -> Array:
    """Change in the outcome after moving dollar inputs by `amount` dollars.

    This is the next-dollar marginal rate: run the calculator, add a dollar,
    run it again, subtract. A dollar that crosses a bracket edge lands in the
    higher bracket, so no convention is needed at a kink.

    `direction` maps frame column names to how many dollars each one moves per
    dollar of `amount`. Nearly every economic change moves a single column; the
    exception is `GENERAL_PARTNER_DOLLAR`, described above.

    `outcome` is any of the four, the same choice `derivative` asks for.
    """
    coefficients = _validate_direction(direction)
    missing = sorted(set(coefficients) - set(records))
    if missing:
        raise KeyError("the frame is missing direction input(s): " + ", ".join(missing))
    if not isinstance(amount, Real) or not np.isfinite(float(amount)):
        raise ValueError("amount must be a finite number")

    changed = dict(records)
    for name, coefficient in coefficients.items():
        changed[name] = records[name] + float(amount) * coefficient
    return outcome(changed, law, double_run=double_run) - outcome(
        records, law, double_run=double_run
    )


def _split_for_differentiation(
    records: Records,
    law: YearLaw,
    input_names: Sequence[str],
    double_run: bool | None,
) -> tuple[tuple[str, ...], dict[str, Array], dict[str, Array], bool]:
    """Set up a derivative: which frame columns vary, and which are held fixed.

    JAX differentiates with respect to the first argument of the function it is
    handed, so the frame is split in two: the requested columns, which become
    that argument, and everything else, which the function closes over. When
    the caller did not say, whether the charity double run applies is read off
    the law.
    """
    names = _validate_input_names(records, input_names)
    selected = set(names)
    changing = {name: records[name] for name in names}
    fixed = {name: value for name, value in records.items() if name not in selected}
    if double_run is None:
        double_run = statutory_order.charity_double_run_applies(law)
    return names, changing, fixed, double_run


def _validate_input_names(
    records: Records, input_names: Sequence[str]
) -> tuple[str, ...]:
    names = tuple(input_names)
    if not names:
        raise ValueError("at least one continuous tax input is required")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError("input names are repeated: " + ", ".join(repeated))
    unsupported = sorted(set(names) - _CONTINUOUS_TAX_INPUT_SET)
    if unsupported:
        raise ValueError(
            "these are not continuous calculator inputs: " + ", ".join(unsupported)
        )
    missing = sorted(set(names) - set(records))
    if missing:
        raise KeyError("the frame is missing tax input(s): " + ", ".join(missing))
    return names


def check_dollar_columns(names: Sequence[str]) -> None:
    """Refuse a set of columns that cannot be moved in dollars together.

    Two rules apply to any way of moving a column by dollars, whether that
    means adding an amount to it or setting it to a new amount:

    - every column has to be measured in dollars, so the three inputs listed
      under `DOLLAR_TAX_INPUTS` that are shares or ratios are refused;
    - the partnership self-employment columns cannot move on their own, for
      the reason `GENERAL_PARTNER_DOLLAR` gives.
    """
    named = set(names)
    if not named:
        raise ValueError("at least one dollar column is required")
    unsupported = sorted(named - _DOLLAR_TAX_INPUT_SET)
    if unsupported:
        raise ValueError(
            "only dollar inputs can be moved by dollars: " + ", ".join(unsupported)
        )
    self_employment = sorted({"part_se1", "part_se2"} & named)
    if self_employment and "part_active" not in named:
        raise ValueError(
            "a dollar of partnership self-employment earnings is also a dollar of "
            "active partnership income, so " + ", ".join(self_employment) + " cannot "
            "move without part_active; a dollar of self-employment earnings that is "
            "not partnership income is sole_prop1 or sole_prop2 instead"
        )


def _validate_direction(direction: Direction) -> dict[str, float]:
    if not direction:
        raise ValueError("a direction must move at least one tax input")
    check_dollar_columns(tuple(direction))
    coefficients: dict[str, float] = {}
    for name, coefficient in direction.items():
        if not isinstance(coefficient, Real) or not np.isfinite(float(coefficient)):
            raise ValueError(f"the coefficient for {name} must be a finite number")
        coefficients[name] = float(coefficient)
    return coefficients
