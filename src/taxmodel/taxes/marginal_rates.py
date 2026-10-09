"""The marginal tax rates a production run stores for every household.

`taxmodel.taxes.individual_income_tax` differentiates the tax calculators with
respect to any input they read. This module chooses which dollars every run
prices, which taxes go into the price, and what the stored columns are called.

A behavioral response needs the price change a reform causes for one household:
the rate under the reform's law minus the rate under baseline law, with the
household held fixed. A run stores both rates for every record, read off the
same frame, so only the law changes between them.

A run stores four gradients under each law, one for each tax concept, because
a reverse-mode derivative gives the rate of change of one number with respect
to every input at once:

- individual income tax after refundable credits;
- the same plus the employee's payroll tax (the employee's half of FICA, the
  employee's half of SECA, and the additional Medicare tax), which is the
  price a worker faces on a dollar of earnings;
- the annual wealth tax;
- the estate tax the household owes if it dies this year.

The first two are reported for the twenty-seven priced dollars in
`PRICED_DOLLARS`. The last two are reported for each of the fourteen asset
columns and six debt columns of the household's balance sheet, because those
taxes are charged on the balance sheet. There is no rate on net worth; each
component gets its own rate.

The estate tax gets one rate more than the balance sheet has columns: the rate
on a dollar of the taxable estate itself, which is the statutory next-dollar
rate with none of the SOI bridge's shaving in it. `ESTATE_BASE_RATE` below
describes it.

Nothing here collapses a balance-sheet gradient into one rate on a chosen base.
A consumer that wants the rate on, say, publicly marketable wealth names its own
columns and does the collapse itself: the rate on a base made of asset columns
`a` is the sum over those columns of the derivative times the amount held,
divided by the amount held.

Every rate here is a derivative of the calculator.
`individual_income_tax.dollar_increment` measures the rate by adding a dollar
and rerunning the calculator; at a threshold the two can differ.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from taxmodel.taxes import individual_income_tax, statutory_order
from taxmodel.taxes.calc.estate import EstateParameters
from taxmodel.taxes.law_for_calculators import YearLaw

Array = jnp.ndarray
Records = Mapping[str, Array]

# The twenty-seven dollars every run prices, and the frame columns each one
# moves. The reverse pass computes the derivative with respect to every
# calculator input; these are the ones kept as stored columns, handed to the
# behavior modules, and reported in the average-marginal-rate table. Adding
# another is one line here.
#
# The name on the left is the name in the stored column. Every entry but one
# moves a single frame column and is named for it. The exception is the general
# partner's dollar, which the law puts in two bases at once (ordinary
# partnership income on Schedule E, and net earnings from self-employment
# subject to SECA), so it moves two columns. It is named `part_active` for the
# Schedule E column; `GENERAL_PARTNER_DOLLAR` in the incentives module says why
# the self-employment half always lands on the first earner.
#
# A dollar of tips is `tips1` alone, and a dollar of overtime is `ot1` alone,
# because the frame keeps base pay, tips and overtime as separate
# non-overlapping columns and each earner's W-2 wages are their sum.
PRICED_DOLLARS: dict[str, individual_income_tax.Direction] = {
    # Pay, earner by earner: base pay, tips, and overtime premium pay. The tip
    # deduction and the overtime deduction apply to the last two and not to the
    # first, and tips carry no payroll tax exemption of their own, so the three
    # prices differ under current law and differ much more under a reform that
    # exempts one of them.
    "wages_base1": {"wages_base1": 1.0},
    "wages_base2": {"wages_base2": 1.0},
    "tips1": {"tips1": 1.0},
    "tips2": {"tips2": 1.0},
    "ot1": {"ot1": 1.0},
    "ot2": {"ot2": 1.0},
    # Business income, by the form the owner takes it in. The active/passive
    # split matters because only the active owner's dollar is self-employment
    # earnings or wage-limited under the pass-through deduction.
    "part_active": individual_income_tax.GENERAL_PARTNER_DOLLAR,
    "part_passive": {"part_passive": 1.0},
    "sole_prop1": {"sole_prop1": 1.0},
    "sole_prop2": {"sole_prop2": 1.0},
    "scorp_active": {"scorp_active": 1.0},
    "scorp_passive": {"scorp_passive": 1.0},
    "rent": {"rent": 1.0},
    # Investment and retirement income. Long-term gain and qualified dividends
    # are taxed on the preferential schedule; short-term gain and taxable
    # interest are ordinary. `gross_ss` is a dollar of Social Security benefit
    # before the inclusion rules, so its price carries the phase-in of the
    # taxable share as well as the rate on the included part.
    "txbl_int": {"txbl_int": 1.0},
    "div_pref": {"div_pref": 1.0},
    "kg_lt": {"kg_lt": 1.0},
    "kg_st": {"kg_st": 1.0},
    "gross_ss": {"gross_ss": 1.0},
    "ui": {"ui": 1.0},
    # Itemized deductions and the charitable gift, where the price of a dollar
    # is what the deduction is worth: zero for a household that takes the
    # standard deduction or has run into a cap, and the negative of a rate for
    # one that itemizes. The three state-and-local columns are priced
    # separately because the SALT cap binds on their sum, so a household at the
    # cap faces the same zero on all three while one under it does not.
    "med_exp": {"med_exp": 1.0},
    "salt_inc_sales": {"salt_inc_sales": 1.0},
    "salt_prop": {"salt_prop": 1.0},
    "salt_pers": {"salt_pers": 1.0},
    "first_mort_int": {"first_mort_int": 1.0},
    "auto_int_exp": {"auto_int_exp": 1.0},
    "char_cash": {"char_cash": 1.0},
    "char_noncash": {"char_noncash": 1.0},
}

@dataclass(frozen=True)
class Price:
    """One of the two taxes a priced dollar's marginal rate is measured in.

    `ending` is what the price adds to the end of a rate's column name. The
    income-tax-only price adds nothing, so its column for long-term gains is
    `mtr_kg_lt`.

    `outcome` is the price as a calculation of its own, and `from_results` is
    the same price read off a finished run of the calculators.
    `individual_income_tax.taxes_and_derivatives` takes the second form: it
    runs the calculators once and carries a derivative backwards through that
    one run for each price.
    """

    ending: str
    outcome: individual_income_tax.Outcome
    from_results: individual_income_tax.OutcomeFromResults


# The two prices. Every column name and every derivative pass below is built
# from this tuple.
#
# The two taxes on wealth get their own passes. The wealth tax is charged on
# net worth added up from the balance sheet, so no run of the chain produces
# it. The estate tax is kept separate because carrying its transposed program
# through every shared backward pass makes a bigger compiled program than one
# forward run of the estate calculator.
PRICES: tuple[Price, ...] = (
    Price("", individual_income_tax.net_income_tax, individual_income_tax.net_income_tax_from_results),
    Price(
        "_plus_employee_payroll",
        individual_income_tax.income_tax_plus_employee_payroll,
        individual_income_tax.income_tax_plus_employee_payroll_from_results,
    ),
)

# The two taxes charged on what a household owns rather than on what it earns.
# Each gets a rate for every asset column and every debt column, and the name
# in the stored column says which tax and which column: the derivative of the
# wealth tax with respect to a dollar of cash is `mtr_wealth_tax_on_value.cash`.
#
# The wealth tax is charged on net worth, so every asset carries the same rate
# and every debt the negative of it, until a reform charges different kinds of
# wealth differently. The estate tax already charges them differently today,
# because closely held pass-through wealth takes a valuation discount on its
# way into the reported gross estate.
# `wealth_tax` is the annual wealth tax; `estate_tax` is the estate tax owed if
# the household dies this year.
BALANCE_SHEET_TAXES: tuple[str, ...] = ("wealth_tax", "estate_tax")

# The estate tax's marginal rate on a dollar of the taxable estate itself. The
# per-asset rates above are this rate times the SOI bridge's shaving for that
# asset class (95.1 cents on a marketable dollar, 61.2 on a closely held
# pass-through dollar, then the deduction share of the household's measurement
# bin). This one is the statutory next-dollar rate with no shaving, zero for a
# household below the exemption. The capital-gains model's estate offset and the estate
# avoidance module read it: a gains tax paid at death enters the estate return
# dollar for dollar, and the Kopczuk-Slemrod elasticity the avoidance response
# uses was estimated against rates of this kind.
ESTATE_BASE_RATE = "mtr_estate_tax_on_taxable_base"

# Which law the rate was computed under. A run stores both.
BASELINE_LAW = "baseline_law"
REFORM_LAW = "reform_law"

# Every frame column any priced dollar moves. One reverse-mode pass over the
# calculator produces the derivative with respect to all of them at once, so
# pricing twenty-seven dollars costs about what pricing one costs.
_MOVED_COLUMNS: tuple[str, ...] = tuple(
    sorted({column for direction in PRICED_DOLLARS.values() for column in direction})
)


class PricedDollarRate(NamedTuple):
    """One priced dollar's rate under one price: the stored name, without a law side."""

    name: str
    dollar: str
    ending: str


# Every priced-dollar marginal rate the model computes, named without a law
# side on the front, with the dollar it prices and the price it is measured in.
# The stored column names put a law side in front of these, and the behavior
# modules read them as they are. The average-rate table in
# `reporting/mtr_table.py` reads the dollar off each entry to weight the rate
# by.
PRICED_DOLLAR_RATES: tuple[PricedDollarRate, ...] = tuple(
    PricedDollarRate(f"mtr_{dollar}{price.ending}", dollar, price.ending)
    for price in PRICES
    for dollar in PRICED_DOLLARS
)
PRICED_DOLLAR_RATE_NAMES: tuple[str, ...] = tuple(
    rate.name for rate in PRICED_DOLLAR_RATES
)

# Every asset and debt column priced against each of the two taxes on wealth,
# and then the estate tax's rate on a dollar of the taxable base, which has no
# balance-sheet column of its own to be named after.
BALANCE_SHEET_RATE_NAMES: tuple[str, ...] = (
    *(
        f"mtr_{tax}_on_{column}"
        for tax in BALANCE_SHEET_TAXES
        for column in individual_income_tax.BALANCE_SHEET_INPUTS
    ),
    ESTATE_BASE_RATE,
)


def rate_columns(law_side: str) -> tuple[str, ...]:
    """The names of the stored priced-dollar rate columns for one law.

    `law_side` is `BASELINE_LAW` or `REFORM_LAW`.
    """
    return tuple(f"{law_side}_{name}" for name in PRICED_DOLLAR_RATE_NAMES)


def balance_sheet_rate_columns(law_side: str) -> tuple[str, ...]:
    """The names of the stored wealth-and-estate rate columns for one law.

    `law_side` is `BASELINE_LAW` or `REFORM_LAW`.
    """
    return tuple(f"{law_side}_{name}" for name in BALANCE_SHEET_RATE_NAMES)


# The priced-dollar rate columns a run writes: twenty-seven dollars times the
# two prices, under each law. This is what the average-marginal-rate table in
# `reporting/mtr_table.py` reports, and each of these rates has a dollar base
# that table can weight it by.
MARGINAL_RATE_COLUMNS: tuple[str, ...] = (
    *rate_columns(BASELINE_LAW),
    *rate_columns(REFORM_LAW),
)

# The balance-sheet rate columns a run writes: two taxes times twenty columns,
# under each law. They are kept apart from the priced dollars above because no
# single dollar base weights them (picking the base is the consumer's job).
BALANCE_SHEET_RATE_COLUMNS: tuple[str, ...] = (
    *balance_sheet_rate_columns(BASELINE_LAW),
    *balance_sheet_rate_columns(REFORM_LAW),
)

# Every rate column a run writes, of either kind.
ALL_RATE_COLUMNS: tuple[str, ...] = (
    *MARGINAL_RATE_COLUMNS,
    *BALANCE_SHEET_RATE_COLUMNS,
)


def marginal_rates(
    records: Records,
    law: YearLaw,
    estate_params: EstateParameters,
    *,
    double_run: bool | None = None,
) -> dict[str, Array]:
    """Every marginal rate a run stores under one law, for every record.

    This is `taxes_and_rates` below with the liabilities thrown away, for a
    caller that wants only the rates.
    """
    return taxes_and_rates(records, law, estate_params, double_run=double_run)[1]


def taxes_and_rates(
    records: Records,
    law: YearLaw,
    estate_params: EstateParameters,
    *,
    double_run: bool | None = None,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """One law's liabilities and every marginal rate a run stores under it.

    The liabilities come first and the rates second. The run of the
    calculators that produces the liabilities is the run the rates are
    derivatives of, and `individual_income_tax.taxes_and_derivatives` returns
    both from it.

    Four reverse-mode passes: income tax, income tax plus the employee's
    payroll tax, the wealth tax, and the estate tax. The first two are carried
    backwards through one shared run of the calculators, which is also the run
    the liabilities come from. The two taxes on wealth get a run each, for the
    reasons given where `PRICES` is defined. The first two passes are reported
    for every priced dollar in `PRICED_DOLLARS`, the last two for each
    balance-sheet column, and the estate tax also for a dollar of the taxable
    estate itself (`ESTATE_BASE_RATE`).

    The keys are the rate names without a law side on the front, such as
    `mtr_kg_lt` and `mtr_wealth_tax_on_value.equities`. `under_law` puts the
    law side on; this function does not know whether the law it is given is
    the baseline's or a reform's.

    `estate_params` is the frozen bridge from economic net worth to the SOI
    reported gross estate. A run that has none computes no estate tax, so this
    function refuses rather than report a zero estate rate.

    `double_run` says whether the law lets a filer choose between an
    above-the-line and an itemized charitable deduction. Left out, it is read
    off the law. Code inside `jax.jit` has to pass the plain Python boolean,
    because by tracing time the law is a placeholder rather than a number.
    """
    if estate_params is None:
        raise ValueError(
            "pricing the estate tax needs the estate measurement bridge; a run "
            "without it computes no estate tax and so has no estate rate to store"
        )
    if double_run is None:
        double_run = statutory_order.charity_double_run_applies(law)

    # The shared run differentiates with respect to one input the frame does
    # not carry: the dollar added straight to the taxable estate. Every
    # household holds zero of it, and the estate calculator already treats a
    # frame without it as holding zero, so putting the column of zeros here
    # changes no liability and gives the base-dollar rate its own derivative.
    with_estate_base_dollar = dict(records)
    with_estate_base_dollar.setdefault(
        individual_income_tax.ESTATE_BASE_DOLLAR,
        jnp.zeros_like(jnp.asarray(records["filing_status"], dtype=float)),
    )

    taxes, gradients = individual_income_tax.taxes_and_derivatives(
        with_estate_base_dollar,
        law,
        estate_params,
        {price.ending: price.from_results for price in PRICES},
        input_names=_MOVED_COLUMNS,
        double_run=double_run,
    )

    priced: dict[str, Array] = {}
    for rate in PRICED_DOLLAR_RATES:
        gradient = gradients[rate.ending]
        # A dollar that lands in two bases at once costs the sum of what it
        # costs in each; every dollar but the general partner's moves one
        # column, and that sum has one term.
        priced[rate.name] = sum(
            float(dollars_moved) * gradient[column]
            for column, dollars_moved in PRICED_DOLLARS[rate.dollar].items()
        )

    # The two taxes on wealth each get a pass of their own, for the reasons
    # given where `PRICES` is defined.
    balance_sheet_gradients = {
        "wealth_tax": individual_income_tax.derivative(
            records,
            law,
            outcome=individual_income_tax.wealth_tax,
            input_names=individual_income_tax.BALANCE_SHEET_INPUTS,
            double_run=double_run,
        ),
        "estate_tax": individual_income_tax.derivative(
            with_estate_base_dollar,
            law,
            outcome=individual_income_tax.estate_tax_outcome(estate_params),
            input_names=(
                *individual_income_tax.BALANCE_SHEET_INPUTS,
                individual_income_tax.ESTATE_BASE_DOLLAR,
            ),
            double_run=double_run,
        ),
    }
    for tax in BALANCE_SHEET_TAXES:
        gradient = balance_sheet_gradients[tax]
        for column in individual_income_tax.BALANCE_SHEET_INPUTS:
            priced[f"mtr_{tax}_on_{column}"] = gradient[column]
    priced[ESTATE_BASE_RATE] = balance_sheet_gradients["estate_tax"][
        individual_income_tax.ESTATE_BASE_DOLLAR
    ]
    return taxes, priced


def priced_rate_names() -> tuple[str, ...]:
    """Every marginal rate a run computes, named as the behavior modules see it.

    These are the keys `taxes_and_rates` returns: no law side on the front,
    because a behavior module is handed the reform's rates and the baseline's
    rates as two separate dictionaries with the same keys.

    A behavior module's manifest declares which of these it reads.
    `run/plan.check_behavior_legs` checks the declaration against this list
    before a run starts.
    """
    return PRICED_DOLLAR_RATE_NAMES + BALANCE_SHEET_RATE_NAMES


def under_law(rates: Mapping[str, Array], law_side: str) -> dict[str, Array]:
    """Label a set of rates with the law they were computed under.

    `law_side` is `BASELINE_LAW` or `REFORM_LAW`.
    """
    if law_side not in (BASELINE_LAW, REFORM_LAW):
        raise ValueError(
            f"a rate is computed under {BASELINE_LAW} or {REFORM_LAW}, not {law_side!r}"
        )
    return {f"{law_side}_{name}": value for name, value in rates.items()}

