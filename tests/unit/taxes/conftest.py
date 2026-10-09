"""Shared pieces for the engine tests: the frozen law and synthetic records."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.scenario import prepare
from taxmodel.taxes import individual_income_tax, law_for_calculators, statutory_order
from taxmodel.taxes.calc.estate import EstateParameters

# The year the first real-data smoke runs, so the tests and the job agree.
YEAR = 2027


# The estate measurement bridge, with the smallest numbers that let the estate
# calculator run: every dollar of economic net worth is reported, closely held
# wealth takes no valuation discount, no lifetime gifts, one measurement bin
# whose deduction share is a tenth, and a two-in-five chance the surviving
# spouse's unused exclusion is elected at a quarter of the reported estate.
ESTATE_MEASUREMENT = EstateParameters(
    reporting_rate=1.0,
    pass_through_valuation_rate=1.0,
    gift_ratio=0.0,
    bin_lower_bounds=jnp.array([0.0]),
    deduction_shares=jnp.array([0.10]),
    dsue_probabilities=jnp.array([0.40]),
    dsue_shares=jnp.array([0.25]),
)


def synthetic_records(filing_status, **overrides) -> dict[str, jnp.ndarray]:
    """Households with every column the chain reads, zero unless named.

    Zero is a legitimate value for every one of them: a household with no
    income, no dependents, no deductions, an empty balance sheet, and no draws
    below any take-up threshold. Each test then sets only the handful of
    columns it is about.

    The estate tax's own inputs -- the balance sheet and the second spouse's
    chance of already being dead -- are here too, so a test can price the
    wealth tax or the estate tax without assembling a second frame.
    """
    codes = np.asarray(filing_status, dtype=np.float64)
    n = codes.size
    records = {
        name: jnp.zeros(n)
        for name in (
            *statutory_order.RECORD_COLUMNS,
            *statutory_order.ESTATE_RECORD_COLUMNS,
            # A dollar added straight to the taxable estate. No household holds
            # any, and the estate calculator defaults it to zero, but it has to
            # be here for a test to differentiate with respect to it.
            individual_income_tax.ESTATE_BASE_DOLLAR,
        )
    }
    records["filing_status"] = jnp.asarray(codes)
    # No second spouse, no dependents, no mortgage vintage: the calculators
    # read a missing value here, not a zero.
    for name in ("age2", "blind2", "dep_age1", "dep_age2", "dep_age3", "divorce_year"):
        records[name] = jnp.full(n, jnp.nan)
    # The draws are uniform on (0, 1); putting them above every take-up
    # threshold in the calculators keeps the baseline case deterministic.
    for name in ("r.bus_loss", "r.cdctc_takeup", "r.new_car", "r.eitc_precert"):
        records[name] = jnp.full(n, 0.999)
    records["filer"] = jnp.ones(n)
    for name, value in overrides.items():
        if name not in records:
            raise KeyError(f"{name} is not a column the chain reads")
        records[name] = jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), (n,))
    return records


class LawStandIn(NamedTuple):
    """The one attribute of a year's `AnnualLaw` the two calculator closures read."""

    calculator: law_for_calculators.YearLaw


def compiled_calculators(law: law_for_calculators.YearLaw, estate_params: EstateParameters | None):
    """The tax chain and the price gradient, compiled the way a run compiles them.

    A run builds these two closures in `scenario/prepare.make_calculators`, which
    settles the one question that changes which functions run (whether the law
    lets a filer choose where to report a charitable gift) and then reads the
    bound calculator law off each year's `AnnualLaw`; the year program compiles
    them from there. Here the same two closures are compiled on their own, with
    `LawStandIn` in place of the full `AnnualLaw`.

    Returns the compiled chain, the compiled gradient, and the stand-in
    constructor to wrap a bound law in before calling either. The gradient is
    only callable when `estate_params` is given, since it prices the estate tax.
    """
    taxes, prices, _taxes_and_prices, _double_run = prepare.make_calculators(
        [LawStandIn(law)], [YEAR], estate_params
    )
    return jax.jit(taxes), jax.jit(prices), LawStandIn


def law_for(compiled_baseline, filing_status, year=YEAR):
    """Baseline law read for households with the filing statuses given."""
    values, years = compiled_baseline
    return law_for_calculators.bind_year(values, years, year).for_records(np.asarray(filing_status))


@pytest.fixture
def baseline_law_by_filing_status(compiled_baseline):
    """Baseline law for 2027 as it is stored: four filing statuses by slot."""
    values, years = compiled_baseline
    return law_for_calculators.bind_year(values, years, YEAR)


@pytest.fixture
def baseline_law(baseline_law_by_filing_status):
    """Baseline law for 2027, read for four households, one per filing status."""
    return baseline_law_by_filing_status.for_records(np.array([1, 2, 3, 4]))
