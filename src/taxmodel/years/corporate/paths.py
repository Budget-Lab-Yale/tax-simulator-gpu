"""The corporate incidence paths: one year-by-year table per scenario.

A corporate tax increase reaches households five ways, and this module works
out the size of each in every year.  Dividends, interest, rent and
pass-through income are cut.  The value of corporate equity is marked down.
Realized capital gains change, because fewer shares are sold and because each
sells for less.  Households save less, which the wealth model carries.  The
individual income tax consequences follow from the tax calculation on the
adjusted records, so nothing here books an offset.

Everything is a closed-form function of three inputs: the change in corporate
receipts (the "wedge", scenario less baseline, in billions of dollars per
year), the national accounts projections, and the parameters in
``config/scenarios/economy/default/corp.yaml``.  No household data is read, so
the whole table is built once per scenario on the host in NumPy.

``taxmodel.years.corporate.channel`` turns the table into per-year arrays on
the law (``factor_arrays_for_years``), and the frame builder applies them
through ``taxmodel.years.income_adjustments.apply_corporate_adjustment``.

Two terms used throughout.  The **payout cut** is the proportional fall in
corporate distributions; it is a flow, negative for a tax increase.  The
**equity markdown** is the proportional fall in the price of corporate equity;
it is a stock, so difference it across years before comparing it with a flow.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Years past the end of the simulation window over which the paths are built,
# so that the present values below close.
PRESENT_VALUE_TAIL_YEARS = 80

# Anything smaller than this counts as zero throughout the corporate channel.
# `years/income_adjustments.py` reads it from here.
CORPORATE_ZERO_TOLERANCE = 1e-9

# A revenue path that runs materially in both directions is a timing provision
# (bonus depreciation costs money now and raises it later), not a rate change,
# and the arithmetic here would get its sign backwards.  The check rejects a
# path whose smaller side is worth more than half a billion dollars and more
# than a tenth of its larger side.
WEDGE_TWO_WAY_DOLLARS_FLOOR_BILLIONS = 0.5
WEDGE_TWO_WAY_SHARE_LIMIT = 0.10

# Sanity bound on the equity markdown.  Going from a 21 percent to a 28 percent
# corporate rate implies a markdown near 9 percent, so a markdown approaching
# one half means the inputs are mis-scaled.
EQUITY_MARKDOWN_LIMIT = 0.5

# Tolerance for the check that nothing moves before enactment. The markdowns
# and factors compared should be exactly zero or exactly one, so only rounding
# is allowed.
INERT_BEFORE_ENACTMENT_TOLERANCE = 1e-12

# corp.asset_exposure_*. The share of each balance-sheet column invested in
# corporate equity, and so the share of the equity markdown it takes. A
# scenario that says nothing about exposure gets these. Columns not listed take
# none, including every pass-through asset and every defined benefit balance.
DEFAULT_ASSET_EXPOSURE: Mapping[str, float] = {
    "value.equities": 1.00,
    "value.dc": 0.55,
    "value.trusts": 0.50,
    "value.re_fund": 0.30,
}

# The discount rate has to clear the tail growth rate by this much or the
# growing-perpetuity terminal value diverges.
MINIMUM_DISCOUNT_MARGIN_OVER_GROWTH = 0.005

# Profit is extended past the projections at the growth rate implied by its
# last six known years, which is five years of growth.
PROFIT_GROWTH_LOOKBACK_YEARS = 6

# The maturity schedule that sets the debt rollover ramp.
DEBT_MATURITY_SCHEDULE = (
    Path(__file__).resolve().parents[4] / "resources" / "debt_maturities.csv"
)


@dataclass(frozen=True)
class CorporateParameters:
    """The economy leg's corporate assumptions, one scenario's worth.

    Every field is one entry of ``config/scenarios/economy/default/corp.yaml``,
    where the value, its kind and its source are recorded.  The YAML name is
    given in each field's comment.
    """

    # corp.sigma_n.  The share of the wedge falling on the normal (risk-adjusted
    # marginal) return rather than on economic rent.  Tax on the normal return
    # is shifted, because capital migrates until returns re-equalize; tax on
    # rent is capitalized into share prices.
    normal_return_share: float = 0.375

    # corp.kappa.  The C corporation share of the economy's normal capital
    # stock.  Of the normal-return burden that has migrated, this share stays
    # on corporate flows and the rest lands on noncorporate ones.
    corporate_share_of_normal_capital: float = 0.40

    # corp.delta_nipa.  The NIPA economic depreciation rate.  Capital can only
    # move between corporate and noncorporate use as it is replaced, so this
    # rate is the speed of migration.
    depreciation_rate: float = 0.057

    # corp.theta.  Scales the cut to distributions, absorbing the gap between
    # profit as the national accounts measure it and profit taxable in the US.
    us_taxable_exposure: float = 1.0

    # corp.omega_div.  The C corporation share of dividends, as against REIT
    # and bond fund distributions.
    corporate_share_of_dividends: float = 0.85

    # corp.omega_kg.  The C corporation equity share of realized long-term
    # capital gains, as against sales of pass-through interests and real
    # estate.
    corporate_share_of_gains: float = 0.50

    # corp.equity_premium.  Added to the nominal ten-year Treasury rate in the
    # enactment year to form the rate at which equity is discounted.
    equity_premium: float = 0.05

    # wealth.cap_flows_pt_weight.  The capital content of pass-through income;
    # the rest is labor.  Read from the wealth channel's entries.
    pass_through_capital_weight: float = 0.20

    # corp.asset_exposure_*.  See DEFAULT_ASSET_EXPOSURE above.
    asset_exposure: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ASSET_EXPOSURE)
    )

    # corp.priced_as_permanent.  When true the equity market is assumed not to
    # believe a sunset, so pricing continues past expiry even though the cut to
    # flows follows the statute.
    priced_as_permanent: bool = False

    def __post_init__(self) -> None:
        for name, low, high in (
            ("normal_return_share", 0.0, 1.0),
            ("corporate_share_of_normal_capital", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < low or value > high:
                raise ValueError(
                    f"the corporate assumption {name} is {value!r}, which is not "
                    f"a number between {low} and {high}"
                )


@dataclass(frozen=True)
class MacroAggregates:
    """The national accounts series the channel needs, one value per year.

    All dollar figures are in billions, matching the Macro-Projections vintage
    and the wedge.  ``after_tax_profit`` is NIPA pre-tax corporate profits less
    corporate receipts (``gdp_corp - rev_corp``).

    A year with no projected value carries ``nan``.  Pre-tax profit exists only
    in the projections, so early years can be missing; the builder stops if a
    year with a nonzero wedge is one of them.
    """

    years: np.ndarray
    after_tax_profit: np.ndarray
    treasury_10_year_percent: np.ndarray
    interest_income: np.ndarray
    rental_income: np.ndarray
    proprietors_income: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "years", np.asarray(self.years, dtype=np.int64))
        for name in (
            "after_tax_profit",
            "treasury_10_year_percent",
            "interest_income",
            "rental_income",
            "proprietors_income",
        ):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=np.float64)
            )
        lengths = {
            len(self.years),
            len(self.after_tax_profit),
            len(self.treasury_10_year_percent),
            len(self.interest_income),
            len(self.rental_income),
            len(self.proprietors_income),
        }
        if len(lengths) != 1:
            raise ValueError(
                "the macro series handed to the corporate channel are not all "
                "the same length"
            )


@dataclass(frozen=True)
class CorporatePaths:
    """One row per year of the grid, plus the scalars the rows were built from.

    The grid runs from the first simulation year to eighty years past the last
    one.  A run applies the simulation years; the tail exists so the present
    values close.  Use ``row_index_for_year`` to find a year.

    Dollar fields are in billions per year, matching the wedge.  Factors are
    multiplicative and equal one in a year the channel does nothing.
    """

    years: np.ndarray
    after_tax_profit: np.ndarray

    # The share of the capital stock that has been replaced since enactment,
    # and so has had the chance to migrate between corporate and noncorporate
    # use.
    capital_turned_over_share: np.ndarray

    # The burden still borne by corporate flows: all of the rent piece, plus
    # the normal piece that either has not migrated yet or migrated within the
    # corporate sector.
    burden_staying_on_corporate_flows: np.ndarray

    # The proportional cut to corporate distributions.  Negative for a tax
    # increase.
    payout_cut: np.ndarray

    # The part of the burden capitalized into share prices in each year, and
    # its present value.  The markdown below is that present value over the
    # present value of profit.
    priced_burden: np.ndarray
    priced_burden_present_value: np.ndarray

    # The proportional fall in the price of fully exposed corporate equity.
    # Zero before enactment, because the change is a surprise.
    equity_markdown: np.ndarray

    # The burden that migrated out of the corporate sector, and its split
    # across the three noncorporate income lines.  Only debt that has rolled
    # over can reprice, so the rest of the interest piece is reported as a
    # revaluation of outstanding bonds instead of a cut to interest income.
    migrated_burden: np.ndarray
    interest_flow_cut: np.ndarray
    bondholder_revaluation: np.ndarray
    rent_flow_cut: np.ndarray
    pass_through_flow_cut: np.ndarray

    # The four factors the record step multiplies income columns by.
    dividend_factor: np.ndarray
    interest_factor: np.ndarray
    rent_factor: np.ndarray
    pass_through_factor: np.ndarray

    # The two capital-gains margins.  Realized corporate gains scale by the
    # quantity factor because the company does fewer buybacks; each sale then
    # fetches less because the shares are marked down.  No tax rate enters
    # either formula.  The price margin is computed from ``equity_markdown``
    # where it is applied: on realized gains in
    # ``income_adjustments.apply_corporate_adjustment`` when the capital-gains
    # model is off, and on the unrealized-gain stock when it is on.
    gain_quantity_factor: np.ndarray

    # Scalars fixed at the time the paths were built.
    discount_rate: float
    enactment_year: int
    parameters: CorporateParameters

    def row_index_for_year(self, year: int) -> int:
        """Position of a calendar year in the grid.

        A year outside the grid raises, since a silent zero would look like an
        expired policy.
        """
        matches = np.flatnonzero(self.years == year)
        if matches.size == 0:
            raise KeyError(
                f"the corporate paths cover {int(self.years[0])} through "
                f"{int(self.years[-1])} and were asked for {year}"
            )
        return int(matches[0])


def read_debt_maturity_schedule() -> tuple[np.ndarray, np.ndarray]:
    """Read the corporate debt maturity schedule.

    Returns the tenors in years and the cumulative share of debt that has
    matured, and so repriced, by each tenor.  The default schedule reaches
    about 0.82 at ten years and 1.0 at thirty.
    """
    tenors: list[int] = []
    shares: list[float] = []
    with open(DEBT_MATURITY_SCHEDULE, newline="") as handle:
        for row in csv.DictReader(handle):
            tenors.append(int(row["tenor"]))
            shares.append(float(row["share"]))
    order = np.argsort(np.asarray(tenors))
    tenor = np.asarray(tenors)[order]
    cumulative = np.minimum(np.cumsum(np.asarray(shares)[order]), 1.0)
    return tenor, cumulative


def debt_rollover_share(
    years_since_enactment: np.ndarray,
    schedule: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Share of corporate debt refinanced a given number of years after enactment.

    Nothing has repriced in the enactment year itself.  Past the end of the
    schedule debt counts as fully rolled, at exactly one.
    """
    if schedule is None:
        schedule = read_debt_maturity_schedule()
    tenor, cumulative = schedule
    elapsed = np.asarray(years_since_enactment)
    out = np.zeros(elapsed.shape, dtype=np.float64)
    longest = int(tenor[-1])
    within = (elapsed >= 1) & (elapsed <= longest)
    if np.any(within):
        positions = np.searchsorted(tenor, elapsed[within])
        found = tenor[positions] == elapsed[within]
        matched = np.where(found, cumulative[positions], 1.0)
        out[within] = matched
    out[elapsed > longest] = 1.0
    return out


def assemble_corporate_wedge(
    off_model_years: Sequence[int],
    off_model_change: Sequence[float],
    on_model_rate_years: Sequence[int] | None = None,
    on_model_rate_change: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the on-model rate change to the off-model residual.

    The wedge has two pieces.  The off-model residual is the scenario's
    Off-Model-Estimates corporate revenue line less the baseline's, which must
    have been produced without the statutory rate change or the rate is counted
    twice.  The on-model piece is the revenue effect of a statutory rate change
    scored inside the model.  A year present in one and absent from the other
    contributes zero from the missing side.

    Returns years and the change in corporate receipts in billions.
    """
    totals: dict[int, float] = {}
    for year, value in zip(off_model_years, off_model_change, strict=True):
        totals[int(year)] = totals.get(int(year), 0.0) + float(value)
    if on_model_rate_years is not None and on_model_rate_change is not None:
        for year, value in zip(
            on_model_rate_years, on_model_rate_change, strict=True
        ):
            totals[int(year)] = totals.get(int(year), 0.0) + float(value)
    years = np.array(sorted(totals), dtype=np.int64)
    return years, np.array([totals[int(y)] for y in years], dtype=np.float64)


def check_wedge_runs_one_way(wedge_change: np.ndarray) -> None:
    """Refuse a revenue path that runs materially in both directions.

    Bonus depreciation costs about $61 billion in 2026 and then raises about
    $35 billion by 2030.  That is a timing provision, and it revalues existing
    capital in the opposite direction to a rate change, so the arithmetic here
    would get the sign wrong.  A rate change on a fixed base moves receipts one
    way apart from rounding, which the dollar floor allows for.

    The eligibility check runs this before the paths are built.
    """
    change = np.asarray(wedge_change, dtype=np.float64)
    raises = float(np.nansum(np.maximum(change, 0.0)))
    costs = float(np.nansum(np.maximum(-change, 0.0)))
    smaller, larger = min(raises, costs), max(raises, costs)
    share = smaller / larger if larger > 0 else 0.0
    if (
        smaller > WEDGE_TWO_WAY_DOLLARS_FLOOR_BILLIONS
        and share > WEDGE_TWO_WAY_SHARE_LIMIT
    ):
        raise ValueError(
            f"the corporate revenue path runs both ways: ${larger:.1f}B in one "
            f"direction against ${smaller:.1f}B in the other, which is "
            f"{100 * share:.0f} percent of the larger side and above the "
            f"{100 * WEDGE_TWO_WAY_SHARE_LIMIT:.0f} percent the guard allows. "
            "That is the receipts signature of a depreciation or other timing "
            "provision, not of a rate change, and the corporate incidence "
            "channel would get its sign backwards. Such a provision is "
            "permanently ineligible for this channel."
        )


def enactment_year_of(
    wedge_years: np.ndarray, wedge_change: np.ndarray
) -> int:
    """The first year the corporate revenue path is nonzero.

    This is the enactment year, when the equity market learns of the policy.
    A path that is zero in every year stops the run.
    """
    years = np.asarray(wedge_years)
    live = np.abs(np.asarray(wedge_change, dtype=np.float64)) > CORPORATE_ZERO_TOLERANCE
    if not np.any(live):
        raise ValueError("the corporate revenue path is zero in every year")
    return int(years[live].min())


def check_enactment_year_is_in_window(
    wedge_years: np.ndarray,
    wedge_change: np.ndarray,
    simulation_years: Sequence[int],
) -> None:
    """Refuse a wedge that starts before the simulation does.

    The equity markdown happens once, in the enactment year.  A simulation
    that starts after enactment would miss it and report the flow cuts alone.
    House convention is to start the simulation one year before the policy.
    """
    enactment = enactment_year_of(wedge_years, wedge_change)
    first_simulated = int(min(simulation_years))
    if enactment < first_simulated:
        raise ValueError(
            f"the corporate revenue change begins in {enactment} but the "
            f"simulation starts in {first_simulated}, so the one-time equity "
            f"markdown at enactment would be missed. Start the simulation in "
            f"{enactment} or earlier; house convention is one year before the "
            "policy."
        )


def build_corporate_paths(
    wedge_years: Sequence[int],
    wedge_change: Sequence[float],
    macro: MacroAggregates,
    simulation_years: Sequence[int],
    *,
    parameters: CorporateParameters | None = None,
) -> CorporatePaths:
    """Turn a corporate revenue change into the year-by-year incidence paths.

    The steps, for each year from enactment on:

    1. Split the revenue change into the part falling on economic rent and the
       part falling on the normal return.
    2. Work out how much of the capital stock has been replaced since
       enactment, since only replaced capital can move between corporate and
       noncorporate use.  That sets how much of the normal-return burden has
       migrated.
    3. Add up what still lands on corporate flows and divide by after-tax
       profit: that is the proportional cut to distributions.
    4. Take the part of the burden that is capitalized into share prices and
       discount it forward.  Capital that has already migrated is left out — it
       is no longer priced in corporate equity.
    5. Spread what migrated across interest, rent and pass-through income in
       proportion to their national totals.  Only refinanced debt reprices, so
       interest delivers the rolled-over share as a flow cut and the rest is
       reported as a revaluation of outstanding bonds.

    Past the end of the revenue file the change persists at its last level,
    growing with profit, since a statutory rate change stays in force unless
    repealed.  A change that sunsets inside the file's years is carried past
    the file at its last level, which is zero.  Profit grows past the
    projections at the rate implied by its last six known years, and a growing
    perpetuity closes the present values past the grid.

    The two eligibility checks above (that the revenue path runs one way and
    that the simulation covers the enactment year) are not called from here.
    """
    if parameters is None:
        parameters = CorporateParameters()

    wedge_year = np.asarray(wedge_years, dtype=np.int64)
    wedge_value = np.asarray(wedge_change, dtype=np.float64)
    enactment_year = enactment_year_of(wedge_year, wedge_value)
    last_file_year = int(wedge_year.max())

    simulation_years = tuple(int(y) for y in simulation_years)
    grid = np.arange(
        min(simulation_years),
        max(simulation_years) + PRESENT_VALUE_TAIL_YEARS + 1,
        dtype=np.int64,
    )

    tail_growth_rate = _tail_growth_rate(macro)
    profit_grid, last_profit_year, last_profit = _profit_over_grid(macro, grid)
    profit_grid = _extend_profit(
        profit_grid, grid, last_profit_year, last_profit, tail_growth_rate
    )

    wedge_grid = _wedge_over_grid(
        wedge_year,
        wedge_value,
        grid,
        last_file_year,
        profit_grid,
    )

    needs_profit = np.abs(wedge_grid) > CORPORATE_ZERO_TOLERANCE
    if np.any(needs_profit & ~np.isfinite(profit_grid)):
        first_bad = int(grid[needs_profit & ~np.isfinite(profit_grid)][0])
        raise ValueError(
            "after-tax corporate profit (pre-tax profits less corporate "
            f"receipts) is missing for {first_bad}, but the corporate revenue "
            "change is nonzero there. Pre-tax profits exist only in the "
            "projections file; check the Macro-Projections vintage."
        )

    years_since_enactment = np.maximum(grid - enactment_year, 0)
    capital_turned_over_share = np.where(
        grid < enactment_year,
        0.0,
        1.0 - (1.0 - parameters.depreciation_rate) ** years_since_enactment,
    )
    debt_rolled_over = debt_rollover_share(years_since_enactment)

    wedge_on_rents = (1.0 - parameters.normal_return_share) * wedge_grid
    wedge_on_normal_return = parameters.normal_return_share * wedge_grid

    burden_staying = wedge_on_rents + wedge_on_normal_return * (
        (1.0 - capital_turned_over_share)
        + capital_turned_over_share * parameters.corporate_share_of_normal_capital
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        payout_cut = np.where(
            (np.abs(wedge_grid) > CORPORATE_ZERO_TOLERANCE)
            & np.isfinite(profit_grid)
            & (profit_grid > 0),
            -parameters.us_taxable_exposure
            * burden_staying
            / np.where(profit_grid > 0, profit_grid, 1.0),
            0.0,
        )

    discount_rate = _discount_rate(
        macro, enactment_year, simulation_years, parameters, tail_growth_rate
    )

    priced_wedge = _priced_wedge(wedge_grid, profit_grid, parameters)
    priced_burden = (
        1.0 - parameters.normal_return_share
    ) * priced_wedge + (
        1.0 - parameters.corporate_share_of_normal_capital
    ) * (1.0 - capital_turned_over_share) * (
        parameters.normal_return_share * priced_wedge
    )

    priced_burden_pv, profit_pv = _present_values(
        priced_burden, profit_grid, discount_rate, tail_growth_rate
    )
    # The change is a surprise at enactment, so nothing is priced in before it.
    priced_burden_pv = np.where(grid < enactment_year, 0.0, priced_burden_pv)

    with np.errstate(divide="ignore", invalid="ignore"):
        equity_markdown = np.where(
            profit_pv > CORPORATE_ZERO_TOLERANCE,
            priced_burden_pv
            / np.where(profit_pv > CORPORATE_ZERO_TOLERANCE, profit_pv, 1.0),
            0.0,
        )
    if np.any(np.abs(equity_markdown) > EQUITY_MARKDOWN_LIMIT):
        worst = int(grid[np.nanargmax(np.abs(equity_markdown))])
        raise ValueError(
            f"the equity markdown reaches "
            f"{np.nanmax(np.abs(equity_markdown)):.3f} in {worst}, past the "
            f"sanity bound of {EQUITY_MARKDOWN_LIMIT:.2f}. Going from a 21 to a "
            "28 percent corporate rate implies about 0.09, so a markdown this "
            "large means the inputs are mis-scaled — wrong units, or the wrong "
            "baseline off-model leg — rather than an enormous policy."
        )

    (
        migrated_burden,
        interest_flow_cut,
        bondholder_revaluation,
        rent_flow_cut,
        pass_through_flow_cut,
        interest_factor,
        rent_factor,
        pass_through_factor,
    ) = _noncorporate_lines(
        macro,
        grid,
        simulation_years,
        wedge_on_normal_return,
        capital_turned_over_share,
        debt_rolled_over,
        parameters,
    )
    dividend_factor = 1.0 + parameters.corporate_share_of_dividends * payout_cut

    factors = np.concatenate(
        [dividend_factor, interest_factor, rent_factor, pass_through_factor]
    )
    if not np.all(np.isfinite(factors)) or np.any(factors < 0):
        raise ValueError(
            "a corporate income scaling factor is not a finite non-negative "
            f"number (smallest is {np.nanmin(factors)}). The revenue change is "
            "implausibly large next to the national income totals it is being "
            "spread over. Refusing to clamp it silently."
        )

    paths = CorporatePaths(
        years=grid,
        after_tax_profit=profit_grid,
        capital_turned_over_share=capital_turned_over_share,
        burden_staying_on_corporate_flows=burden_staying,
        payout_cut=payout_cut,
        priced_burden=priced_burden,
        priced_burden_present_value=priced_burden_pv,
        equity_markdown=equity_markdown,
        migrated_burden=migrated_burden,
        interest_flow_cut=interest_flow_cut,
        bondholder_revaluation=bondholder_revaluation,
        rent_flow_cut=rent_flow_cut,
        pass_through_flow_cut=pass_through_flow_cut,
        dividend_factor=dividend_factor,
        interest_factor=interest_factor,
        rent_factor=rent_factor,
        pass_through_factor=pass_through_factor,
        gain_quantity_factor=1.0
        + parameters.corporate_share_of_gains * payout_cut,
        discount_rate=discount_rate,
        enactment_year=enactment_year,
        parameters=parameters,
    )
    _check_nothing_moves_before_enactment(paths)
    return paths


def _check_nothing_moves_before_enactment(paths: CorporatePaths) -> None:
    """Refuse a set of paths that prices anything before the enactment year.

    The change is a surprise until it is enacted, so every markdown must be
    zero and every income factor exactly one before then.
    """
    before = paths.years < paths.enactment_year
    if np.any(before):
        inert = (
            np.all(
                np.abs(paths.equity_markdown[before])
                < INERT_BEFORE_ENACTMENT_TOLERANCE
            )
            and np.all(
                np.abs(paths.dividend_factor[before] - 1.0)
                < INERT_BEFORE_ENACTMENT_TOLERANCE
            )
            and np.all(
                np.abs(paths.interest_factor[before] - 1.0)
                < INERT_BEFORE_ENACTMENT_TOLERANCE
            )
            and np.all(
                np.abs(paths.rent_factor[before] - 1.0)
                < INERT_BEFORE_ENACTMENT_TOLERANCE
            )
            and np.all(
                np.abs(paths.pass_through_factor[before] - 1.0)
                < INERT_BEFORE_ENACTMENT_TOLERANCE
            )
        )
        if not inert:
            raise AssertionError(
                "the corporate paths change something before the enactment year "
                f"{paths.enactment_year}: either the equity markdown or one of "
                "the income factors differs from its no-change value."
            )


# ---------------------------------------------------------------------------
# Helpers for build_corporate_paths
# ---------------------------------------------------------------------------


def _tail_growth_rate(macro: MacroAggregates) -> float:
    """Growth rate of after-tax profit implied by its last six known years."""
    known = np.isfinite(macro.after_tax_profit)
    if int(np.count_nonzero(known)) < PROFIT_GROWTH_LOOKBACK_YEARS:
        raise ValueError(
            "the Macro-Projections vintage has fewer than "
            f"{PROFIT_GROWTH_LOOKBACK_YEARS} years of after-tax corporate "
            "profit (pre-tax profits less corporate receipts), so the profit "
            "path cannot be extended past the projections."
        )
    profit = macro.after_tax_profit[known][-PROFIT_GROWTH_LOOKBACK_YEARS:]
    return float((profit[-1] / profit[0]) ** (1.0 / (len(profit) - 1)) - 1.0)


def _profit_over_grid(
    macro: MacroAggregates, grid: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """After-tax profit lined up with the grid, missing where unprojected.

    Also returns the last year the projections cover and its profit, the
    starting point for the extension past the projections.
    """
    known = np.isfinite(macro.after_tax_profit)
    known_years = np.asarray(macro.years)[known]
    known_profit = np.asarray(macro.after_tax_profit)[known]
    profit = _series_over_grid(known_years, known_profit, grid)
    last_year = int(known_years.max())
    return profit, last_year, _value_at_year(known_years, known_profit, last_year)


def _extend_profit(
    profit: np.ndarray,
    grid: np.ndarray,
    last_known_year: int,
    last_known_profit: float,
    growth: float,
) -> np.ndarray:
    """Grow profit past the last projected year at its trailing growth rate."""
    beyond = grid > last_known_year
    if np.any(beyond):
        profit = profit.copy()
        profit[beyond] = last_known_profit * (1.0 + growth) ** (
            grid[beyond] - last_known_year
        )
    return profit


def _wedge_over_grid(
    wedge_year: np.ndarray,
    wedge_value: np.ndarray,
    grid: np.ndarray,
    last_file_year: int,
    profit: np.ndarray,
) -> np.ndarray:
    """The revenue change lined up with the grid and extended past the file.

    A year inside the file's range but absent from it carries no change.  Past
    the file's last year the change continues at its last level, growing with
    profit.
    """
    lookup = dict(zip(wedge_year.tolist(), wedge_value.tolist(), strict=True))
    over_grid = np.array(
        [lookup.get(int(y), np.nan) for y in grid], dtype=np.float64
    )
    over_grid[np.isnan(over_grid) & (grid <= last_file_year)] = 0.0
    missing = np.isnan(over_grid)
    if not np.any(missing):
        return over_grid
    profit_at_file_end = profit[grid == last_file_year]
    if profit_at_file_end.size == 0 or not np.isfinite(profit_at_file_end[0]):
        raise ValueError(
            "after-tax corporate profit is missing in "
            f"{last_file_year}, the last year of the off-model revenue file, "
            "so the revenue change cannot be extended past it."
        )
    last_value = lookup[last_file_year]
    over_grid[missing] = (
        last_value * profit[missing] / float(profit_at_file_end[0])
    )
    return over_grid


def _discount_rate(
    macro: MacroAggregates,
    enactment_year: int,
    simulation_years: Sequence[int],
    parameters: CorporateParameters,
    tail_growth_rate: float,
) -> float:
    """Nominal ten-year Treasury at enactment plus the equity risk premium.

    Fixed at enactment and used for the whole path.  Nominal because the
    distributions being discounted are nominal.
    """
    treasury = _value_at_year(
        macro.years, macro.treasury_10_year_percent, enactment_year
    )
    if not np.isfinite(treasury):
        # Enactment precedes the projections only in unusual vintages, and
        # those years are inert, so fall back to the first simulated year.
        treasury = _value_at_year(
            macro.years, macro.treasury_10_year_percent, int(min(simulation_years))
        )
    if not np.isfinite(treasury):
        raise ValueError(
            "the ten-year Treasury rate is missing in the enactment year "
            f"{enactment_year}, so the rate at which equity is discounted "
            "cannot be formed."
        )
    rate = treasury / 100.0 + parameters.equity_premium
    if rate <= tail_growth_rate + MINIMUM_DISCOUNT_MARGIN_OVER_GROWTH:
        raise ValueError(
            f"the equity discount rate {rate:.3f} is not safely above the "
            f"profit growth rate {tail_growth_rate:.3f}, so the growing "
            "perpetuity that closes the present values diverges. Check the "
            "ten-year Treasury projection, the equity premium, or the profit "
            "series."
        )
    return float(rate)


def _priced_wedge(
    wedge: np.ndarray, profit: np.ndarray, parameters: CorporateParameters
) -> np.ndarray:
    """The revenue change as the equity market prices it.

    Normally that is the change itself.  Under ``priced_as_permanent`` the
    market is assumed not to believe the sunset, so for pricing the change
    continues past expiry from its last nonzero level, growing with profit.
    The cut to flows still follows the statute.
    """
    if not parameters.priced_as_permanent:
        return wedge
    live = np.flatnonzero(np.abs(wedge) > CORPORATE_ZERO_TOLERANCE)
    if live.size == 0:
        return wedge
    last_live = int(live[-1])
    if last_live >= len(wedge) - 1:
        return wedge
    priced = wedge.copy()
    ahead = np.arange(last_live + 1, len(wedge))
    priced[ahead] = wedge[last_live] * profit[ahead] / profit[last_live]
    return priced


def _present_values(
    priced_burden: np.ndarray,
    profit: np.ndarray,
    discount_rate: float,
    tail_growth_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discount the priced burden and profit back over the grid.

    Both are the value of everything from next year on, so the last row holds
    only the growing perpetuity past the end of the grid.  The markdown is the
    ratio of the two.
    """
    n = len(priced_burden)
    burden_pv = np.zeros(n, dtype=np.float64)
    profit_pv = np.zeros(n, dtype=np.float64)
    perpetuity = (1.0 + tail_growth_rate) / (discount_rate - tail_growth_rate)
    burden_pv[-1] = priced_burden[-1] * perpetuity
    profit_pv[-1] = profit[-1] * perpetuity
    for i in range(n - 2, -1, -1):
        burden_pv[i] = (priced_burden[i + 1] + burden_pv[i + 1]) / (
            1.0 + discount_rate
        )
        profit_pv[i] = (profit[i + 1] + profit_pv[i + 1]) / (1.0 + discount_rate)
    return burden_pv, profit_pv


def _noncorporate_lines(
    macro: MacroAggregates,
    grid: np.ndarray,
    simulation_years: Sequence[int],
    wedge_on_normal_return: np.ndarray,
    capital_turned_over_share: np.ndarray,
    debt_rolled_over: np.ndarray,
    parameters: CorporateParameters,
) -> tuple[np.ndarray, ...]:
    """Spread the migrated burden over interest, rent and pass-through income.

    The burden that has migrated out of the corporate sector lands on
    noncorporate capital in proportion to the national income totals of the
    three lines.  Pass-through income enters at its capital weight, because
    those columns carry labor content too.

    An existing bond's coupon is fixed, so only debt that has been refinanced
    since enactment can reprice.  The rest of the interest piece is reported
    as a revaluation of outstanding bonds, not as a cut to interest income.

    The pass-through factor applies the capital weight a second time: the
    burden is spread over the capital fifth of pass-through income, and then
    only that fifth of each household's pass-through column is cut.
    """
    interest_base = _series_over_grid(macro.years, macro.interest_income, grid)
    rent_base = _series_over_grid(macro.years, macro.rental_income, grid)
    pass_through_base = (
        _series_over_grid(macro.years, macro.proprietors_income, grid)
        * parameters.pass_through_capital_weight
    )

    migrated = (
        (1.0 - parameters.corporate_share_of_normal_capital)
        * capital_turned_over_share
        * wedge_on_normal_return
    )
    needed = np.abs(migrated) > CORPORATE_ZERO_TOLERANCE
    unavailable = (
        ~np.isfinite(interest_base)
        | ~np.isfinite(rent_base)
        | ~np.isfinite(pass_through_base)
    )
    simulated = np.isin(grid, np.asarray(simulation_years, dtype=np.int64))
    if np.any(needed & unavailable & simulated):
        first_bad = int(grid[needed & unavailable & simulated][0])
        raise ValueError(
            "the national totals for interest, rental or proprietors' income "
            f"are missing for simulated year {first_bad}, but part of the "
            "corporate burden has migrated to noncorporate capital there."
        )

    total_base = interest_base + rent_base + pass_through_base
    usable = np.isfinite(total_base) & (total_base > 0)
    safe_total = np.where(usable, total_base, 1.0)
    interest_share = np.where(usable, interest_base / safe_total, 0.0)
    rent_share = np.where(usable, rent_base / safe_total, 0.0)
    pass_through_share = np.where(usable, pass_through_base / safe_total, 0.0)

    interest_if_all_repriced = migrated * interest_share
    interest_flow_cut = interest_if_all_repriced * debt_rolled_over
    bondholder_revaluation = interest_if_all_repriced * (1.0 - debt_rolled_over)
    rent_flow_cut = migrated * rent_share
    pass_through_flow_cut = migrated * pass_through_share

    interest_cut_share = _share_of_base(interest_flow_cut, interest_base)
    rent_cut_share = _share_of_base(rent_flow_cut, rent_base)
    pass_through_capital_cut_share = _share_of_base(
        pass_through_flow_cut, pass_through_base
    )

    return (
        migrated,
        interest_flow_cut,
        bondholder_revaluation,
        rent_flow_cut,
        pass_through_flow_cut,
        1.0 - interest_cut_share,
        1.0 - rent_cut_share,
        1.0
        - parameters.pass_through_capital_weight * pass_through_capital_cut_share,
    )


def _share_of_base(amount: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Amount over base where the base is positive and finite, else zero.

    The numpy version of ``balance_sheet.divide_where_positive``, for the
    host-side corporate paths.
    """
    usable = np.isfinite(base) & (base > 0)
    return np.where(usable, amount / np.where(usable, base, 1.0), 0.0)


def _series_over_grid(
    years: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    lookup = dict(zip(years.tolist(), values.tolist(), strict=True))
    return np.array([lookup.get(int(y), np.nan) for y in grid], dtype=np.float64)


def _value_at_year(years: np.ndarray, values: np.ndarray, year: int) -> float:
    match = np.flatnonzero(np.asarray(years) == year)
    if match.size == 0:
        return float("nan")
    return float(values[match[0]])


# The corp.yaml entry for each of the channel's parameters, and the
# balance-sheet column for each asset-exposure entry.
CORP_ENTRY_FOR_PARAMETER: dict[str, str] = {
    "normal_return_share": "sigma_n",
    "corporate_share_of_normal_capital": "kappa",
    "depreciation_rate": "delta_nipa",
    "us_taxable_exposure": "theta",
    "corporate_share_of_dividends": "omega_div",
    "corporate_share_of_gains": "omega_kg",
    "equity_premium": "equity_premium",
    "priced_as_permanent": "priced_as_permanent",
}

CORP_ENTRY_FOR_ASSET_EXPOSURE: dict[str, str] = {
    "value.equities": "asset_exposure_equities",
    "value.dc": "asset_exposure_dc",
    "value.trusts": "asset_exposure_trusts",
    "value.re_fund": "asset_exposure_re_fund",
}

# The capital content of pass-through income is a wealth channel entry.
PASS_THROUGH_CAPITAL_WEIGHT_ENTRY = ("wealth", "cap_flows_pt_weight")


def corporate_parameters_from_economy_leg(config) -> CorporateParameters:
    """Read the corporate channel's assumptions out of a run's economy leg.

    ``config`` is the run's live ``ScenarioConfig``, so every number comes from
    the scenario tree, including any alternative that overrides a default.

    These are transmission entries, which a static result excludes by
    definition. Reading them on the static pass raises, so this is called only
    on the mechanical and conventional passes.
    """

    values = {
        name: config.economy_param("corp", entry)
        for name, entry in CORP_ENTRY_FOR_PARAMETER.items()
    }
    values["priced_as_permanent"] = bool(values["priced_as_permanent"])
    values["asset_exposure"] = {
        column: float(config.economy_param("corp", entry))
        for column, entry in CORP_ENTRY_FOR_ASSET_EXPOSURE.items()
    }
    values["pass_through_capital_weight"] = float(
        config.economy_param(*PASS_THROUGH_CAPITAL_WEIGHT_ENTRY)
    )
    for name in CORP_ENTRY_FOR_PARAMETER:
        if name != "priced_as_permanent":
            values[name] = float(values[name])
    return CorporateParameters(**values)
