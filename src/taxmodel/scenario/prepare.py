"""Host-side preparation of one scenario for the year loop.

`prepare_scenario` resolves the scenario's provenance, compiles the reform and
baseline laws, binds each year, reads the wealth financing profile and the
return path, builds the calculators, scores a change in the corporate
statutory rate, and builds the national corporate figures the distribution
table spreads. It returns a `PreparedScenario`.

`years/loop.py` runs a group of prepared scenarios through one compiled
projection year. `reporting/assemble.py` (`projection_from_simulation`) turns
the results into the published tables. `run/scenarios.py` calls all three.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pandas as pd

from taxmodel.config import ScenarioConfig
from taxmodel.config.errors import ConfigError
from taxmodel.config.interfaces import interface_root
from taxmodel.frame import build_frame
from taxmodel.frame.population import PopulationSource
from taxmodel.law import batch
from taxmodel.law.indexes import read_macro_spliced
from taxmodel.reporting import corporate_spread, receipts_inputs
from taxmodel.reporting import detail as detail_output
from taxmodel.scenario import law_by_year
from taxmodel.scenario.provenance import (
    DEFAULT_CONFIG_ROOT,
    Provenance,
    Scenario,
    load_scenario,
    resolve_scenario_provenance,
)
from taxmodel.taxes import marginal_rates, statutory_order
from taxmodel.years import (
    one_year,
    wealth_arithmetic,
    world_names,
    worlds,
)
from taxmodel.years.corporate import channel as corporate_channel
from taxmodel.years.corporate import paths as corporate_paths_module
from taxmodel.years.corporate import scorer as corporate_scorer

Array = jnp.ndarray

# Root folder for the saving surfaces and percentile transitions. The economy
# entry `wealth.financing_profile` names a folder under this root, or says
# 'none', or gives the 'flat:<s>' shorthand.
DEFAULT_WEALTH_PROFILES_ROOT = Path("config/calibrations/wealth_profiles")

# ---------------------------------------------------------------------------
# The wealth channel's return path
# ---------------------------------------------------------------------------


def total_return_by_year(
    macro_root: str | Path,
    years: Sequence[int],
    additive_delta: float = 0.0,
) -> np.ndarray:
    """Nominal return path for the wealth channel: growth in nominal GDP per capita.

        r_total(t) = (gdp_t / gdp_t-1) / (pop_t / pop_t-1) - 1

    The growth rate is nominal, to match the nominal wealth stock. Population
    is the sum of the married and unmarried tax-unit counts by age that the
    Macro-Projections publication carries as one column per age. History and
    projection are spliced so that the first projection year has a real
    predecessor to difference against, and where the two files give the same
    year the historical row wins.

    The column read is total nominal `gdp`.
    """
    macro = read_macro_spliced(macro_root)
    population_columns = [
        name
        for name in macro.columns
        if name.startswith(("married_", "unmarried_")) and name.split("_")[-1].isdigit()
    ]
    if not population_columns:
        raise ValueError(
            f"{macro_root} has no married_*/unmarried_* tax-unit counts, so "
            "nominal GDP per capita cannot be formed"
        )
    if "gdp" not in macro.columns:
        raise ValueError(f"{macro_root} does not publish a gdp column")

    table = pd.DataFrame(
        {
            "year": macro["year"].astype(int),
            "gdp": macro["gdp"].astype(float),
            "population": macro[population_columns].fillna(0.0).sum(axis=1),
        }
    ).drop_duplicates(subset="year", keep="first").sort_values("year")

    growth = (table["gdp"] / table["gdp"].shift(1)) / (
        table["population"] / table["population"].shift(1)
    ) - 1.0
    by_year = pd.Series(growth.to_numpy(), index=table["year"].to_numpy())
    requested = [int(year) for year in years]
    missing = [year for year in requested if year not in by_year.index]
    unusable = [
        year for year in requested if year in by_year.index and pd.isna(by_year[year])
    ]
    if missing or unusable:
        raise ValueError(
            "nominal GDP per capita growth is undefined for "
            f"{sorted(set(missing) | set(unusable))} in the spliced "
            f"Macro-Projections tables at {macro_root}"
        )
    return by_year.reindex(requested).to_numpy(dtype=np.float64) + float(additive_delta)


@dataclass(frozen=True)
class WealthSettings:
    """The wealth channel's resolved inputs for a run."""

    profile: wealth_arithmetic.FinancingProfile
    total_return_by_year: np.ndarray
    haircut_cap: float
    pass_through_weight: float


def resolve_wealth_settings(
    config: ScenarioConfig,
    years: Sequence[int],
    macro_root: str | Path,
    profiles_root: str | Path = DEFAULT_WEALTH_PROFILES_ROOT,
) -> WealthSettings:
    """Read the economy's wealth entries and build the profile and return path."""
    n_bins = int(config.economy_param("wealth", "n_pctiles"))
    ages = np.asarray(wealth_arithmetic.age_grid())
    profile = wealth_arithmetic.resolve_profile(
        config.economy_param("wealth", "financing_profile"),
        profiles_root,
        ages,
        n_bins,
    )
    return WealthSettings(
        profile=profile,
        total_return_by_year=total_return_by_year(
            macro_root,
            years,
            float(config.economy_param("wealth", "r_total_additive_delta")),
        ),
        haircut_cap=float(config.economy_param("wealth", "fmax")),
        pass_through_weight=float(
            config.economy_param("wealth", "cap_flows_pt_weight")
        ),
    )


# ---------------------------------------------------------------------------
# Law
# ---------------------------------------------------------------------------


def bind_baseline_laws(
    baseline_scenario: Scenario, years: Sequence[int]
) -> law_by_year.BoundLawSeries:
    """Bind baseline law over the whole projection.

    A batch of scenarios measured against one baseline law calls this once
    and shares the result. Baseline law is its own baseline, so the
    employer-payroll wage change is measured against this law's own payroll
    parameters.
    """
    return law_by_year.bind_annual_laws(
        baseline_scenario.values,
        baseline_scenario.years,
        baseline_scenario.law_table,
        years,
    )


def stacked_annual_laws(
    scenario: Scenario,
    baseline_scenario: Scenario,
    years: Sequence[int],
    baseline_laws: law_by_year.BoundLawSeries | None = None,
) -> tuple[law_by_year.BoundLawSeries, law_by_year.BoundLawSeries]:
    """Bind the reform and baseline law over the projection, stacked by year.

    Returns the reform's and the baseline's bound series. Each holds the
    year-stacked form the projection reads and every year's law on its own.
    The per-year form is used to settle whether the law lets a filer choose
    where to report a charitable gift, and the corporate factors are attached
    to it.

    `baseline_laws` is baseline law already bound for these same years, passed
    in by a caller running several scenarios against the same baseline. Left
    out, it is bound here.

    Binding does not read the population. Every law parameter is bound as its
    four filing-status values, and each household reads the value for its own
    filing status inside the compiled program.
    """
    if baseline_laws is None:
        baseline_laws = bind_baseline_laws(baseline_scenario, years)
    if scenario is baseline_scenario:
        # A baseline run scores baseline law against itself, so there is one
        # law here, not two.
        reform_laws = baseline_laws
    else:
        reform_laws = law_by_year.bind_annual_laws(
            scenario.values,
            scenario.years,
            scenario.law_table,
            years,
            baseline_scenario.values,
            baseline_scenario.years,
        )
    return reform_laws, baseline_laws


def make_calculators(
    year_laws: Sequence[law_by_year.AnnualLaw],
    years: Sequence[int],
    estate_params,
) -> tuple[
    worlds.TaxCalculator,
    worlds.PriceCalculator,
    worlds.TaxesAndPrices,
    bool,
]:
    """Build the tax calculator, the marginal-rate calculator, their combined
    form, and the charitable double-run switch.

    Each calculator takes the year's
    :class:`~taxmodel.scenario.law_by_year.AnnualLaw` and reads the bound
    calculator law off it.

    The combined calculator is for the step in a year where the same households
    are taxed and priced under the same law. It runs the calculators once and
    takes the marginal rates as derivatives of that run.

    Whether the law lets a filer choose where to report a charitable gift is
    settled once for the whole projection by
    ``statutory_order.charity_double_run_for_projection``, which rejects a law
    that answers differently in different years. The answer is returned as the
    fourth item because two scenarios can share a compiled program only when
    they answer it the same way.
    """
    double_run = statutory_order.charity_double_run_for_projection(
        [law.calculator for law in year_laws], years, "the law this run scores"
    )

    def taxes(records, law):
        return statutory_order.run_taxes(
            records, law.calculator, estate_params, double_run=double_run
        )

    def standard_prices(records, law):
        return marginal_rates.marginal_rates(
            records, law.calculator, estate_params, double_run=double_run
        )

    def standard_taxes_and_prices(records, law):
        return marginal_rates.taxes_and_rates(
            records, law.calculator, estate_params, double_run=double_run
        )

    return taxes, standard_prices, standard_taxes_and_prices, double_run



# ---------------------------------------------------------------------------
# The corporate statutory rate
# ---------------------------------------------------------------------------

# Economy-leg entry for the elasticity of the corporate tax base to the
# net-of-tax statutory rate, read by the scorer. Its value and citation are in
# `config/scenarios/economy/default/`. The compiled law's corporate rate itself
# is read by `years/corporate/channel.corporate_rate_by_year`.
BASE_ELASTICITY_ENTRY = ("corp", "rate_eti")


class CorporateChannel(NamedTuple):
    """What scenario preparation works out about the corporate income tax.

    `rate_score_by_world` is the revenue effect of the scenario's change in the
    corporate statutory rate, in dollars, one series per world and one entry
    per run year, for the published corporate revenue line. `dollars` is the
    national corporate figures the distribution table spreads over households.
    """

    rate_score_by_world: dict[str, np.ndarray]
    dollars: corporate_spread.CorporateDollars


def prepare_corporate_channel(
    config: ScenarioConfig,
    years: Sequence[int],
    baseline_rate: np.ndarray,
    scenario_rate: np.ndarray,
    economy: str | None,
    config_root: str | Path,
) -> CorporateChannel:
    """Score the scenario's change in the corporate statutory rate, and build
    the national corporate figures the distribution table spreads.

    ``baseline_rate`` and ``scenario_rate`` are the two laws' statutory rates
    by year, read by `corporate_channel.corporate_rate_by_year`.

    The corporate tax base is CBO's receipts divided by the baseline statutory
    rate, and moving the rate moves receipts by the amount
    `years/corporate/scorer.py` derives. Only the conventional world lets the
    base erode as the rate rises. The score is computed on the host because it
    needs CBO's corporate receipts level.

    The distribution table's figures are CBO's corporate receipts plus each
    world's rate score and off-model corporate score, and the cost-recovery
    change (`corporate_spread.from_receipts_inputs`). They are read from the
    same receipts inputs as the score.

    A scenario that does not change the corporate rate scores zero in every
    world and every year. A failure to read the receipts inputs stops every
    scenario, since the distribution table's corporate tax starts from them.
    The corporate tax is zero, with a printed line, when the economy leg
    states no corporate incidence assumptions.
    """
    years = tuple(int(year) for year in years)
    n_years = len(years)
    rate_moves = corporate_channel.corporate_rate_moves(baseline_rate, scenario_rate)
    no_score = {
        world: np.zeros(n_years, dtype=np.float64)
        for world in world_names.WORLD_NAMES
    }

    try:
        inputs = receipts_inputs.load_receipts_inputs(years, economy, config_root)
    except (receipts_inputs.ReceiptsInputError, ConfigError, OSError) as problem:
        raise receipts_inputs.ReceiptsInputError(
            "the distribution table's corporate tax is spread from CBO's "
            "projection of corporate receipts and the other receipts inputs, "
            f"and they could not be read: {problem}"
            + (
                " This scenario also changes the corporate statutory rate, and "
                "scoring that change needs the same receipts."
                if rate_moves
                else ""
            )
        ) from problem

    score_by_world = no_score
    if rate_moves:
        score_by_world = corporate_scorer.corporate_rate_score_by_world(
            world_names.WORLD_NAMES,
            baseline_rate,
            scenario_rate,
            inputs.cbo_levels["rev_corp"],
            base_elasticity=_corporate_base_elasticity(config),
        )

    incidence = corporate_incidence_from_economy_leg(config)
    if incidence is None:
        print(
            "The distribution table's corporate tax is zero for this scenario, "
            "because its economy leg does not state the corporate incidence "
            "assumptions "
            + ", ".join(".".join(entry) for entry in INCIDENCE_ENTRIES)
            + "."
        )
        dollars = corporate_spread.no_corporate_tax(n_years)
    else:
        dollars = corporate_spread.from_receipts_inputs(
            inputs, score_by_world, incidence
        )
    return CorporateChannel(score_by_world, dollars)


def _corporate_base_elasticity(config: ScenarioConfig) -> float:
    """The elasticity of the corporate tax base to the net-of-tax statutory rate."""
    channel, name = BASE_ELASTICITY_ENTRY
    try:
        return float(config.economy_param(channel, name))
    except ConfigError as missing:
        raise ConfigError(
            f"this scenario changes the corporate statutory rate, and scoring "
            f"that change needs the elasticity of the corporate tax base to the "
            f"net-of-tax rate, which the economy leg states as {channel}.{name}. "
            f"This economy leg does not state it."
        ) from missing


@dataclass(frozen=True)
class PreparedScenario:
    """One scenario, prepared on the host and ready for `years/loop.py`.

    `name` is the scenario's ID. `haircut_cap` and `provenance` are carried
    because assembling the output tables needs them.
    """

    name: str
    years: tuple[int, ...]
    reform_laws: object
    baseline_laws: object
    wealth_profiles: object
    frame_builder: object
    tax_calculator: object
    price_calculator: object
    # Calculator returning the mechanical frame's liabilities and marginal
    # rates under the reform's law from one run.
    taxes_and_prices: object
    switch_fingerprint: str
    haircut_cap: float
    provenance: Provenance
    detail_request: detail_output.DetailRequest | None = None
    # The two compiled laws the scenario ran, which `tax_law.csv` publishes.
    compiled_law: object = None
    compiled_baseline_law: object = None
    # The charitable double-run switch, which the driver needs to build the
    # gains model's rate pricers.
    charity_double_run: bool = False
    # Corporate-channel inputs to the gains solve: the economy leg's
    # asset-exposure shares and the equity markdown by year (None without a
    # corporate change), from the eroded-base paths for the conventional world
    # and from the fixed-base paths for the two mechanical worlds.
    corporate_asset_exposure: dict | None = None
    equity_markdown_by_year: object = None
    mechanical_equity_markdown_by_year: object = None
    # Revenue effect of the scenario's change in the corporate statutory rate,
    # in dollars, one series per world and one entry per run year. It goes onto
    # the published corporate revenue line.
    corporate_rate_score_by_world: dict[str, np.ndarray] | None = None
    # The national corporate figures the distribution table spreads, one
    # entry per run year (`reporting/corporate_spread.CorporateDollars`).
    # None is read as zeros.
    corporate_dollars: corporate_spread.CorporateDollars | None = None


# The economy-leg entries holding the share of the corporate tax falling on
# the normal return, the share of the corporate capital burden borne by
# foreign owners, and the share of net home equity that is structures rather
# than land. The asset-exposure entries are
# `years/corporate/paths.CORP_ENTRY_FOR_ASSET_EXPOSURE`.
NORMAL_RETURN_SHARE_ENTRY = ("corp", "sigma_n")
CORPORATE_FOREIGN_SHARE_ENTRY = ("distribution", "corp_foreign_share")
HOUSING_STRUCTURE_SHARE_ENTRY = ("distribution", "housing_structure_share")
INCIDENCE_ENTRIES: tuple[tuple[str, str], ...] = (
    NORMAL_RETURN_SHARE_ENTRY,
    CORPORATE_FOREIGN_SHARE_ENTRY,
    HOUSING_STRUCTURE_SHARE_ENTRY,
    *(
        ("corp", entry)
        for entry in corporate_paths_module.CORP_ENTRY_FOR_ASSET_EXPOSURE.values()
    ),
)


def corporate_incidence_from_economy_leg(
    config: ScenarioConfig,
) -> corporate_spread.Incidence | None:
    """The incidence assumptions the corporate spread uses.

    None when the economy leg states any of `INCIDENCE_ENTRIES` nowhere.
    """
    try:
        normal_return_share = config.economy_param(*NORMAL_RETURN_SHARE_ENTRY)
        foreign_share = config.economy_param(*CORPORATE_FOREIGN_SHARE_ENTRY)
        structure_share = config.economy_param(*HOUSING_STRUCTURE_SHARE_ENTRY)
        asset_exposure = {
            column: float(config.economy_param("corp", entry))
            for column, entry in corporate_paths_module.CORP_ENTRY_FOR_ASSET_EXPOSURE.items()
        }
    except ConfigError:
        return None
    structure_share = float(structure_share)
    if not 0.0 <= structure_share <= 1.0:
        raise ConfigError(
            f"{'.'.join(HOUSING_STRUCTURE_SHARE_ENTRY)} is {structure_share}; the "
            "share of net home equity that is structures must be between 0 and 1"
        )
    return corporate_spread.Incidence(
        normal_return_share=float(normal_return_share),
        foreign_share=float(foreign_share),
        asset_exposure=asset_exposure,
        housing_structure_share=structure_share,
    )


@dataclass(frozen=True)
class SharedBaseline:
    """Baseline law loaded and bound once for a batch of scenarios.

    `compiled_law` is the compiled baseline law this was built from. A scenario
    reuses the shared baseline only when it was handed that same object.
    """

    compiled_law: object
    years: tuple[int, ...]
    scenario: Scenario
    laws: law_by_year.BoundLawSeries


def load_shared_baseline(
    years: Sequence[int],
    config_root: str | Path,
    economy: str | None,
    compiled_baseline_law: object,
) -> SharedBaseline:
    """Read and bind one compiled baseline law for a whole batch of scenarios."""
    requested = tuple(int(year) for year in years)
    scenario = load_scenario(
        requested,
        config_root,
        alternative=None,
        economy=economy,
        compiled_law=compiled_baseline_law,
    )
    return SharedBaseline(
        compiled_law=compiled_baseline_law,
        years=requested,
        scenario=scenario,
        laws=bind_baseline_laws(scenario, requested),
    )


def prepare_scenario(
    population: PopulationSource,
    years: Sequence[int],
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    alternative: str | None = None,
    economy: str | None = None,
    behavior: str | None = None,
    detail_request: detail_output.DetailRequest | None = None,
    name: str | None = None,
    compiled_law: object = None,
    compiled_baseline_law: object = None,
    provenance: Provenance | None = None,
    shared_baseline: SharedBaseline | None = None,
) -> PreparedScenario:
    """Prepare one scenario for the device.

    The steps, in order: resolve the scenario and refuse to run on stale
    provenance; compile the reform's law and baseline law over the same years
    and the same growth rates; resolve the wealth profile and return path; bind
    each year's law and stack it; score the scenario's change in the
    corporate statutory rate and build the distribution table's corporate
    figures; and build the calculators.

    `compiled_law` and `compiled_baseline_law` are laws already compiled inside
    a batch, so that every scenario in the batch has bound laws of identical
    shape and can run in one compiled program. Left out, each law is compiled
    on its own.

    `provenance` is this scenario's three legs already resolved. For one rung
    of a stacked reform, resolving it here would read the law from the rung's
    last folder alone rather than from the whole fold, so the caller passes it
    in.

    `shared_baseline` is baseline law already read and bound for these years.
    It is used only when it was built from the same compiled baseline law
    object this scenario was handed.
    """
    requested = tuple(int(year) for year in years)
    if provenance is None:
        provenance = resolve_scenario_provenance(
            config_root, alternative, economy, behavior
        )
        for line in provenance.staleness.warnings:
            print(f"stale (warning): {line}")
    reuse_baseline = (
        shared_baseline is not None
        and compiled_baseline_law is not None
        and compiled_baseline_law is shared_baseline.compiled_law
        and shared_baseline.years == requested
    )
    # The index table and the estate bridge are the same for every scenario of
    # a run, so a scenario sharing a baseline takes them from it rather than
    # reading the macro publication and the bridge file again.
    if reuse_baseline:
        baseline_scenario = shared_baseline.scenario
        shared_reads = {
            "growth": baseline_scenario.indexes,
            "estate_bridge": (
                baseline_scenario.estate_params,
                baseline_scenario.cluster_death_weight_cap,
            ),
        }
    else:
        shared_reads = {}
    scenario = load_scenario(
        requested,
        config_root,
        alternative=alternative,
        economy=economy,
        compiled_law=compiled_law,
        **shared_reads,
    )
    baseline_laws = None
    if reuse_baseline:
        baseline_laws = shared_baseline.laws
    elif alternative is None and compiled_baseline_law is None:
        baseline_scenario = scenario
    else:
        baseline_scenario = load_scenario(
            requested,
            config_root,
            alternative=None,
            economy=economy,
            compiled_law=compiled_baseline_law,
        )
    build_frame.check_population_covers_the_frame_steps(population)

    macro_root = interface_root(
        "Macro-Projections", economy=economy, config_root=config_root
    )
    wealth = resolve_wealth_settings(provenance.config, requested, macro_root)
    profiles = one_year.stack_wealth_profiles(
        jnp.asarray(wealth.profile.saving_shares),
        jnp.asarray(wealth.profile.transition),
        jnp.asarray(wealth.total_return_by_year),
        fmax=wealth.haircut_cap,
        pass_through_weight=wealth.pass_through_weight,
    )

    reform_series, baseline_series = stacked_annual_laws(
        scenario, baseline_scenario, requested, baseline_laws
    )

    # The two laws' corporate statutory rates for each run year. Both the
    # incidence channel below and the rate score at the end of this function
    # work from them, so they are read once here.
    scenario_corporate_rate = corporate_channel.corporate_rate_by_year(
        scenario.compiled_law, len(requested)
    )
    baseline_corporate_rate = corporate_channel.corporate_rate_by_year(
        baseline_scenario.compiled_law, len(requested)
    )

    # The corporate incidence channel, for a scenario whose law changes the
    # corporate statutory rate. The paths are built twice: from the revenue
    # change with the tax base eroding, for the conventional world, and with
    # the base fixed, for the two mechanical worlds. Their per-year factors are
    # attached to the reform series as law data, so a corporate scenario runs
    # the same compiled program as any other.
    def paths(base_erodes: bool):
        return corporate_channel.corporate_paths_for_scenario(
            scenario_corporate_rate,
            baseline_corporate_rate,
            requested,
            macro_root,
            provenance.config,
            base_erodes=base_erodes,
        )

    conventional_paths = paths(base_erodes=True)
    equity_markdown_by_year = None
    mechanical_equity_markdown_by_year = None
    if conventional_paths is not None:
        conventional_factors = corporate_channel.factor_arrays_for_years(
            conventional_paths, requested
        )
        mechanical_factors = corporate_channel.factor_arrays_for_years(
            paths(base_erodes=False), requested
        )
        reform_series = law_by_year.with_corporate_factors(
            reform_series, conventional_factors, mechanical_factors
        )
        # The gains solve debits the same equity markdowns the frame's
        # corporate step applies in each world.
        equity_markdown_by_year = np.asarray(
            conventional_factors["markdown_rate"], dtype=np.float64
        )
        mechanical_equity_markdown_by_year = np.asarray(
            mechanical_factors["markdown_rate"], dtype=np.float64
        )
    try:
        corp_parameters = corporate_paths_module.corporate_parameters_from_economy_leg(
            provenance.config
        )
        corporate_asset_exposure = dict(corp_parameters.asset_exposure)
        corporate_gain_share = corp_parameters.corporate_share_of_gains
    except ConfigError:
        # An economy leg without corporate entries runs no corporate channel.
        # A leg that declares a corporate reform but is missing an entry fails
        # earlier, when the paths are built.
        corporate_asset_exposure = {}
        corporate_gain_share = 0.0

    gains_model_active = (
        provenance.behavior_leg is not None
        and provenance.behavior_leg.runs_the_gains_model
    )
    corporate = prepare_corporate_channel(
        provenance.config,
        requested,
        baseline_corporate_rate,
        scenario_corporate_rate,
        economy,
        config_root,
    )
    reform_year_laws = reform_series.by_year
    taxes, prices, taxes_and_prices, double_run = make_calculators(
        reform_year_laws, requested, scenario.estate_params
    )
    return PreparedScenario(
        name=name if name is not None else (alternative or "baseline"),
        years=requested,
        reform_laws=reform_series.stacked,
        baseline_laws=baseline_series.stacked,
        wealth_profiles=profiles,
        frame_builder=build_frame.make_frame_builder(
            scenario.indexes,
            scenario.cluster_death_weight_cap,
            corporate_asset_exposure=corporate_asset_exposure,
            corporate_gain_share=corporate_gain_share,
            gain_dynamics_active=gains_model_active,
        ),
        tax_calculator=taxes,
        price_calculator=prices,
        taxes_and_prices=taxes_and_prices,
        # The two switches that decide which functions run: the charitable
        # double-run switch, and whether the corporate step leaves realized
        # gains to the gains model. Scenarios share calculators only when both
        # match.
        switch_fingerprint=batch.program_fingerprint(double_run, gains_model_active),
        haircut_cap=wealth.haircut_cap,
        provenance=provenance,
        detail_request=detail_request,
        compiled_law=scenario.compiled_law,
        compiled_baseline_law=baseline_scenario.compiled_law,
        charity_double_run=double_run,
        corporate_asset_exposure=corporate_asset_exposure,
        equity_markdown_by_year=equity_markdown_by_year,
        mechanical_equity_markdown_by_year=mechanical_equity_markdown_by_year,
        corporate_rate_score_by_world=corporate.rate_score_by_world,
        corporate_dollars=corporate.dollars,
    )
