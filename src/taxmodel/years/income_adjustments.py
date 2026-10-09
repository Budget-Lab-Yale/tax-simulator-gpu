"""The mechanical income adjustments applied while a reform-side frame is built."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.kg.applier import CORPORATE_MARKDOWN_GAINS_COLUMN
from taxmodel.frame.steps_jax import ESTATE_ASSET_COLS, ESTATE_DEBT_COLS
from taxmodel.taxes.calc.payroll import PayrollLaw, calc_pr
from taxmodel.years.corporate.paths import CORPORATE_ZERO_TOLERANCE
from taxmodel.years.record_columns import (
    PASS_THROUGH_FLOW_COLUMNS,
    first_array,
    net_pass_through_income,
    sum_columns,
)

Array = jnp.ndarray
Records = Mapping[str, Array]

# ASSUMPTION: 85 percent of a change in employer payroll tax changes cash
# wages and the other 15 percent changes nontaxable fringe compensation.
EMPLOYER_PAYROLL_CASH_WAGE_SHARE = 0.85

# Number of passes in the employer-payroll wage solve. Each pass sets the wage
# to the anchor minus the employer tax owed on the previous pass's wage, so
# the remaining error shrinks by the marginal employer payroll rate each time.
# That rate is at most 7.65 percent of wages under current law (6.2 percent
# for Social Security plus 1.45 percent for Medicare), so five passes hold the
# identity (new wage plus employer tax equals baseline wage plus baseline tax)
# to under a cent for any reform of realistic size. The count is fixed because
# the frame builder compiles under jax.jit, which cannot take a data-dependent
# loop bound.
EMPLOYER_PAYROLL_SOLVE_PASSES = 5

_DIVIDEND_COLUMNS = ("div_ord", "div_pref")
_INTEREST_COLUMNS = ("txbl_int", "exempt_int")
_RENT_COLUMNS = ("rent", "rent_loss")
_REALIZED_GAIN_COLUMNS = ("kg_lt", "kg_st", "kg_lt_basis")
_RETIREMENT_DISTRIBUTION_COLUMNS = ("txbl_ira_dist", "gross_pens_dist")

# Frame columns recording, for the distribution table, what the corporate step
# changed in income. They are signed as the step changed income, so they are
# negative for a corporate tax increase. Nothing but the report reads them;
# `reporting/corporate_spread.py` says what each one holds.
RETIREMENT_CHANGE_COLUMN = "corp_dY_retirement"
RETIREMENT_CHANGE_ACCRUAL_COLUMN = "corp_dY_retirement_accrual"
GAINS_CHANGE_COLUMN = "corp_dY_gains"

# The short- and long-term realized gains whose change `corp_dY_gains` records.
# The long-term basis is scaled too but is not income.
_GAIN_INCOME_COLUMNS = ("kg_lt", "kg_st")

# Frame columns recording, for the distribution table, the corporate step's
# change in one group of income columns, signed as income (a larger loss is a
# fall in income). The groups are the income columns that later steps of the
# conventional world scale the same way; `reporting/corporate_spread.py`
# scales each piece by those steps. The flow pieces add up to `corp_dY_exog`,
# and the two gain pieces to the part of `corp_dY_gains` this step records.
CORPORATE_CHANGE_PIECES: dict[str, tuple[tuple[str, float], ...]] = {
    "corp_dY.div_ord": (("div_ord", 1.0),),
    "corp_dY.div_pref": (("div_pref", 1.0),),
    "corp_dY.txbl_int": (("txbl_int", 1.0),),
    "corp_dY.exempt_int": (("exempt_int", 1.0),),
    "corp_dY.rent": (("rent", 1.0),),
    "corp_dY.rent_loss": (("rent_loss", -1.0),),
    "corp_dY.part_active": (("part_active", 1.0),),
    "corp_dY.part_passive": (("part_passive", 1.0),),
    "corp_dY.scorp_active": (("scorp_active", 1.0),),
    "corp_dY.scorp_passive": (("scorp_passive", 1.0),),
    "corp_dY.sole_prop": (("sole_prop1", 1.0), ("sole_prop2", 1.0)),
    "corp_dY.farm": (("farm1", 1.0), ("farm2", 1.0)),
    "corp_dY.pass_through_losses": (
        ("part_active_loss", -1.0),
        ("part_passive_loss", -1.0),
        ("part_179", -1.0),
        ("scorp_active_loss", -1.0),
        ("scorp_passive_loss", -1.0),
        ("scorp_179", -1.0),
    ),
    "corp_dY.kg_st": (("kg_st", 1.0),),
    "corp_dY.kg_lt": (("kg_lt", 1.0),),
}
GAIN_CHANGE_PIECES: tuple[str, ...] = ("corp_dY.kg_st", "corp_dY.kg_lt")
FLOW_CHANGE_PIECES: tuple[str, ...] = tuple(
    name for name in CORPORATE_CHANGE_PIECES if name not in GAIN_CHANGE_PIECES
)

# The balances behind a household's pensions. Haig-Simons income leaves out
# the defined-contribution share of gross pension distributions
# (`reporting/etr_measures.haig_simons_income`).
_DEFINED_CONTRIBUTION_BALANCE = "value.dc"
_DEFINED_BENEFIT_BALANCE = "value.db"


def _scale_present_columns(
    out: dict[str, Array], names: tuple[str, ...], factor: Array
) -> None:
    for name in names:
        if name in out:
            out[name] = jnp.asarray(out[name]) * factor


def _change_in_columns(
    before: Records, after: Records, names: tuple[str, ...], template: Array
) -> Array:
    """The dollar change in the sum of the named columns, a gap read as zero."""
    return sum_columns(after, names, template) - sum_columns(before, names, template)


def defined_contribution_share(records: Records, template: Array) -> Array:
    """The defined-contribution share of a household's retirement balances.

    The defined-contribution balance over defined-contribution plus
    defined-benefit balances, zero for a household with no balances, a gap read
    as zero. This is the share `reporting/etr_measures.haig_simons_income`
    takes out of gross pension distributions.
    """
    defined_contribution = sum_columns(
        records, (_DEFINED_CONTRIBUTION_BALANCE,), template
    )
    balances = defined_contribution + sum_columns(
        records, (_DEFINED_BENEFIT_BALANCE,), template
    )
    has_balances = balances > 0.0
    return jnp.where(
        has_balances,
        defined_contribution / jnp.where(has_balances, balances, 1.0),
        0.0,
    )


def apply_corporate_gain_quantity_margin(
    records: Records, gain_quantity_factor: Array
) -> dict[str, Array]:
    """Scale gains from sales after the capital-gains applier has run.

    A corporate payout change changes the number of shares repurchased by the
    company. It therefore changes realized gains and the basis of the shares
    sold, but not gains deemed realized at death. The step is skipped when the
    factor is within ``CORPORATE_ZERO_TOLERANCE`` of one.

    The change in short- plus long-term realized gains is added to
    ``corp_dY_gains``, which starts at zero where the records do not carry it.
    """
    present = tuple(name for name in _REALIZED_GAIN_COLUMNS if name in records)
    if not present:
        return dict(records)

    factor = jnp.asarray(gain_quantity_factor, dtype=jnp.float64)
    template = jnp.asarray(records[present[0]], dtype=jnp.float64)
    gains_change_before = bs.column(records, GAINS_CHANGE_COLUMN, template)

    def leave_records_unchanged(_: None) -> dict[str, Array]:
        unchanged = dict(records)
        unchanged[GAINS_CHANGE_COLUMN] = gains_change_before
        return unchanged

    def scale_sale_columns(_: None) -> dict[str, Array]:
        scaled = dict(records)
        _scale_present_columns(scaled, present, factor)
        scaled[GAINS_CHANGE_COLUMN] = gains_change_before + _change_in_columns(
            records, scaled, _GAIN_INCOME_COLUMNS, template
        )
        return scaled

    return jax.lax.cond(
        jnp.abs(factor - 1.0) < CORPORATE_ZERO_TOLERANCE,
        leave_records_unchanged,
        scale_sale_columns,
        operand=None,
    )


def record_corporate_markdown_gains(records: Records) -> dict[str, Array]:
    """Add the markdown's change in realized gains to ``corp_dY_gains``.

    With the gains model running, the corporate equity markdown lowers each
    age cell's extra realizations, and the gains applier records each
    household's share of that change (``kg_corporate_markdown``). Records
    without that column are returned unchanged.
    """
    if CORPORATE_MARKDOWN_GAINS_COLUMN not in records:
        return dict(records)
    markdown_change = bs.column(records, CORPORATE_MARKDOWN_GAINS_COLUMN)
    out = dict(records)
    out[GAINS_CHANGE_COLUMN] = (
        bs.column(records, GAINS_CHANGE_COLUMN, markdown_change) + markdown_change
    )
    return out


def apply_corporate_adjustment(
    records: Records,
    *,
    dividend_factor: Array,
    interest_factor: Array,
    rent_factor: Array,
    pass_through_factor: Array,
    markdown_rate: Array,
    flow_factor: Array,
    asset_exposure: Mapping[str, Array],
    corporate_gain_share: Array,
    gain_dynamics_active: bool,
) -> dict[str, Array]:
    """Apply one year's corporate incidence path to household records.

    The four flow factors are the proportional changes in dividends, interest,
    rent, and pass-through income from the corporate path. ``markdown_rate`` is
    the price decline for fully exposed corporate equity. ``flow_factor`` is
    the path's change in after-tax payouts. ``asset_exposure`` gives the share
    of each balance-sheet column invested in corporate equity.

    ``corporate_gain_share`` is the corporate-equity share of realized gains.
    When the gains model is inactive, it controls both the change in sales and
    the price loss on long-term gains. When that model is active, this function
    leaves gains alone because the gains stock and its later record allocation
    apply those changes.

    The dollar changes are recorded for the distribution table:
    ``corp_dY_exog`` for the four flows, ``corp_dY_retirement`` and
    ``corp_dY_retirement_accrual`` for retirement distributions,
    ``corp_dY_gains`` for realized gains (zero when the gains model is active),
    and the same changes by group of income columns
    (``CORPORATE_CHANGE_PIECES``).

    These values come from ``years/corporate/paths.py``. The function refuses
    a second application, which would apply the same incidence change twice.
    """
    if "corp_dY_exog" in records:
        raise ValueError("the corporate adjustment was already applied")

    template = first_array(records)
    dividend_factor = jnp.asarray(dividend_factor)
    interest_factor = jnp.asarray(interest_factor)
    rent_factor = jnp.asarray(rent_factor)
    pass_through_factor = jnp.asarray(pass_through_factor)
    markdown_rate = jnp.asarray(markdown_rate)
    flow_factor = jnp.asarray(flow_factor)
    corporate_gain_share = jnp.asarray(corporate_gain_share)

    pass_through_income = net_pass_through_income(records, template)
    outside_income_change = (
        (dividend_factor - 1.0) * sum_columns(records, _DIVIDEND_COLUMNS, template)
        + (interest_factor - 1.0) * sum_columns(records, _INTEREST_COLUMNS, template)
        + (rent_factor - 1.0)
        * (
            bs.column(records, "rent", template)
            - bs.column(records, "rent_loss", template)
        )
        + (pass_through_factor - 1.0) * pass_through_income
    )

    gross_assets_before = sum_columns(records, ESTATE_ASSET_COLS, template)
    markdown_amount = jnp.zeros_like(template, dtype=jnp.float64)
    for name, exposure in asset_exposure.items():
        if name in records:
            exposure = jnp.asarray(exposure)
            markdown_amount = markdown_amount + (
                exposure * markdown_rate * bs.column(records, name, template)
            )

    defined_contribution = bs.column(records, "value.dc", template)
    defined_benefit = bs.column(records, "value.db", template)
    dc_exposure = jnp.asarray(asset_exposure.get("value.dc", 0.0))
    retirement_total = defined_contribution + defined_benefit
    dc_share = jnp.where(
        retirement_total > CORPORATE_ZERO_TOLERANCE,
        defined_contribution
        / jnp.where(retirement_total > CORPORATE_ZERO_TOLERANCE, retirement_total, 1.0),
        0.0,
    )
    ira_factor = 1.0 - dc_exposure * markdown_rate * (
        defined_contribution > CORPORATE_ZERO_TOLERANCE
    )
    pension_factor = 1.0 - dc_exposure * markdown_rate * dc_share

    out = dict(records)
    _scale_present_columns(out, _DIVIDEND_COLUMNS, dividend_factor)
    _scale_present_columns(out, _INTEREST_COLUMNS, interest_factor)
    _scale_present_columns(out, _RENT_COLUMNS, rent_factor)
    _scale_present_columns(out, PASS_THROUGH_FLOW_COLUMNS, pass_through_factor)
    _scale_present_columns(out, ("txbl_ira_dist",), ira_factor)
    _scale_present_columns(out, ("txbl_pens_dist", "gross_pens_dist"), pension_factor)

    for name, exposure in asset_exposure.items():
        if name in out:
            out[name] = jnp.asarray(out[name]) * (
                1.0 - jnp.asarray(exposure) * markdown_rate
            )

    if not gain_dynamics_active:
        gain_quantity_factor = 1.0 + corporate_gain_share * flow_factor
        if "kg_lt" in out:
            long_term_gain = bs.column(records, "kg_lt", template)
            long_term_basis = bs.column(records, "kg_lt_basis", template)
            # ASSUMPTION: gain plus basis is read as sale proceeds and floored
            # at zero. A negative sum has no stated economic interpretation.
            price_loss_base = jnp.maximum(long_term_gain + long_term_basis, 0.0)
            out["kg_lt"] = jnp.asarray(out["kg_lt"]) + corporate_gain_share * (
                flow_factor * long_term_gain - markdown_rate * price_loss_base
            )
        # ASSUMPTION: the corp.yaml corporate share of long-term gains is
        # used for short-term gains too. Short-term gains
        # get the sales-quantity change but no price change, because the
        # records carry no short-term basis.
        _scale_present_columns(out, ("kg_lt_basis", "kg_st"), gain_quantity_factor)

    assets_after = sum_columns(out, ESTATE_ASSET_COLS, template)
    debts = sum_columns(records, ESTATE_DEBT_COLS, template)
    out["net_worth_before_corporate_markdown"] = jnp.asarray(records["net_worth"])
    out["net_worth"] = assets_after - debts
    out["corp_dY_exog"] = outside_income_change
    out[RETIREMENT_CHANGE_COLUMN] = _change_in_columns(
        records, out, _RETIREMENT_DISTRIBUTION_COLUMNS, template
    )
    # Haig-Simons income counts the defined-benefit share of the gross pension
    # change. The share is read off the marked-down balances. The wealth
    # haircut scales every balance by the same factor, so the report's
    # Haig-Simons income reads the same share. The markdown also moves the
    # share itself, and that part of the change in Haig-Simons income (the
    # gross pension times the fall in the share) is not recorded here.
    out[RETIREMENT_CHANGE_ACCRUAL_COLUMN] = _change_in_columns(
        records, out, ("gross_pens_dist",), template
    ) * (1.0 - defined_contribution_share(out, template))
    # With the gains model running, the gains changes are recorded where they
    # happen: the sale-volume factor in `apply_corporate_gain_quantity_margin`
    # and the markdown in the gains applier.
    out[GAINS_CHANGE_COLUMN] = _change_in_columns(
        records, out, _GAIN_INCOME_COLUMNS, template
    )
    for piece, columns in CORPORATE_CHANGE_PIECES.items():
        out[piece] = sum(
            sign * _change_in_columns(records, out, (name,), template)
            for name, sign in columns
        )
    out["corp_markdown"] = jnp.where(
        gross_assets_before > CORPORATE_ZERO_TOLERANCE,
        markdown_amount
        / jnp.where(
            gross_assets_before > CORPORATE_ZERO_TOLERANCE,
            gross_assets_before,
            1.0,
        ),
        0.0,
    )
    out["corp_flow_factor"] = jnp.broadcast_to(flow_factor, template.shape)
    return out


def apply_employer_payroll_adjustment(
    records: Records,
    law: PayrollLaw,
    baseline_liab_fica_er1: Array,
    baseline_liab_fica_er2: Array,
    baseline_wages1: Array,
    baseline_wages2: Array,
    *,
    cash_wage_share: Array = EMPLOYER_PAYROLL_CASH_WAGE_SHARE,
) -> dict[str, Array]:
    """Pass a change in employer payroll tax through to wages.

    A cut in the employer half of payroll tax does not stay with the employer.
    The worker's total compensation (what the employer pays out in wages plus
    the tax it owes on them) is held at its baseline level, so the tax cut
    shows up as higher wages instead. The reverse holds for an increase.

    For each earner this solves

        wage + employer_payroll_tax(wage, reform law)
            = baseline wage + baseline employer payroll tax,

    by starting at the earner's current wage and repeating
    ``EMPLOYER_PAYROLL_SOLVE_PASSES`` times: compute the employer tax the
    current guess owes under reform law, then set the wage to the anchor minus
    that tax. The payroll calculator supplies the tax, so the solve follows
    the Social Security taxable maximum and any other kink in the schedule.

    The anchor, baseline wage plus baseline employer tax, is supplied per
    earner from the baseline scenario's evaluation, so wages already moved
    upstream do not move the target.

    Only 85 percent of the solved wage change reaches cash wages
    (``cash_wage_share``); the remaining 15 percent is assumed to be absorbed
    by untaxed fringe benefits. The quantity held constant is therefore cash
    wages plus fringe benefits plus employer tax. Earners with no wages are
    left alone. Self-employment earnings are untouched.

    Tips and overtime stay fixed, so the whole change is written to base wages.
    ``pr_dY_exog`` carries only the cash-wage change, leaving out the fringe
    share, which is compensation in kind.

    The adjustment is applied at the frame bridge, before household behavior.

    The function refuses records that already contain ``pr_dY_exog`` because a
    second application would change wages twice.
    """
    if "pr_dY_exog" in records:
        raise ValueError("the employer payroll adjustment was already applied")

    cash_wage_share = jnp.asarray(cash_wage_share)
    baseline_liabilities = (
        jnp.asarray(baseline_liab_fica_er1),
        jnp.asarray(baseline_liab_fica_er2),
    )
    baseline_wages = (jnp.asarray(baseline_wages1), jnp.asarray(baseline_wages2))
    earners = (1, 2)

    tips = {e: jnp.asarray(records[f"tips{e}"]) for e in earners}
    overtime = {e: jnp.asarray(records[f"ot{e}"]) for e in earners}
    # Each earner's W-2 wages on the frame as it stands, before this change.
    current_wages = {
        e: jnp.asarray(records[f"wages_base{e}"]) + tips[e] + overtime[e]
        for e in earners
    }
    has_wages = {e: current_wages[e] != 0.0 for e in earners}
    # Total compensation held constant: baseline wages plus the baseline
    # employer payroll tax on them.
    anchor = {
        e: wage + liability
        for e, wage, liability in zip(
            earners, baseline_wages, baseline_liabilities, strict=True
        )
    }

    solved_wages = dict(current_wages)
    trial = dict(records)
    for _ in range(EMPLOYER_PAYROLL_SOLVE_PASSES):
        for e in earners:
            trial[f"wages_base{e}"] = solved_wages[e] - tips[e] - overtime[e]
        employer_tax = calc_pr(trial, law)
        for e in earners:
            solved_wages[e] = jnp.where(
                has_wages[e],
                anchor[e] - employer_tax[f"liab_fica_er{e}"],
                current_wages[e],
            )

    out = dict(records)
    income_change = jnp.zeros_like(jnp.asarray(records["wages_base1"]))
    for e in earners:
        new_wages = current_wages[e] + cash_wage_share * (
            solved_wages[e] - current_wages[e]
        )
        out[f"wages_base{e}"] = new_wages - tips[e] - overtime[e]
        income_change = income_change + new_wages - current_wages[e]

    out["pr_dY_exog"] = income_change
    return out
