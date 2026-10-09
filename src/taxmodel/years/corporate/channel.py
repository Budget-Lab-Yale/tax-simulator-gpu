"""Connection between the corporate incidence channel and a scenario.

``years/corporate/paths.py`` turns a corporate revenue change into
year-by-year incidence paths, and ``years/corporate/scorer.py`` turns a
statutory rate change into that revenue change. This module connects the two
to a scenario.

Steps per scenario, on the host, before compilation:

1. The corporate statutory rate is read off the compiled reform and baseline
   laws. Where the two agree in every year, the channel is off.
2. The rate change is scored against CBO's corporate receipts from the
   Macro-Projections publication, either with the tax base eroding (for the
   conventional world) or with the base held at its baseline level (for the
   two mechanical worlds). The rate is held at its last simulated value past
   the window, and years before the window carry no change.
3. The wedge passes two eligibility checks (it must run one way, and the
   simulation must cover the enactment year), and the paths are built.

Use of the paths in the run:

- The four flow factors, the markdown, the payout cut, and the gains quantity
  factor become per-year arrays on the scenario's law object
  (`scenario/law_by_year.CorporateYearFactors`), one set from each of the two
  paths. The frame builder applies them on the flows branch.
- For a scenario that also runs the capital-gains model, both paths' equity
  markdowns and the economy leg's asset-exposure shares go to the gains
  solve, which debits the marked-down corporate equity from the stock of
  unrealized gain: the eroded path's markdown in the behavioral solve, and the
  fixed-base path's markdown in the mechanical worlds' frozen pass
  (``years/gains_solve.py``).

The off-model corporate residual (international provisions, credits, and base
broadeners scored outside the model) is not included, because this model has
no Off-Model-Estimates interface yet. The wedge is the statutory rate score
alone. The off-model piece would enter at ``paths.assemble_corporate_wedge``,
which this module calls with an empty off-model side.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from taxmodel.law.indexes import read_macro_spliced
from taxmodel.years.corporate import paths as corporate_paths
from taxmodel.years.corporate import scorer

# The top corporate statutory rate. It is a compiled-law field holding one
# number, the same for every filing status (`docs/compiled_law_schema.md`).
# The frame join puts every compiled law parameter on under its own name, so
# this is also the frame column the entity shifting response reads the
# reform's rate from.
CORPORATE_RATE_COLUMN = "corp.rate"

# The frame is built under the law being scored, so baseline law's corporate
# rate is not on it. The annual step adds it under this name for the entity
# shifting response.
BASELINE_CORPORATE_RATE_COLUMN = "corp.rate_baseline"

# The Macro-Projections columns the channel reads: corporate receipts (the
# base the rate score works from), pre-tax corporate profits, the ten-year
# Treasury rate, and the three noncorporate income lines the migrated burden
# is spread across.
_RECEIPTS_COLUMN = "rev_corp"
_MACRO_COLUMNS = (
    _RECEIPTS_COLUMN,
    "gdp_corp",
    "tsy_10y",
    "gdp_interest",
    "gdp_rent",
    "gdp_proprietors",
)


def corporate_rate_by_year(law: object, n_years: int) -> np.ndarray:
    """The top corporate statutory rate a compiled law states in each year.

    The rate does not vary by filing status, so the first status's value is
    read. A year the law says nothing about, either because the field is absent
    or because the law does not switch that entry on, comes back as
    not-a-number, which `years/corporate/scorer.py` scores at zero. A stated
    rate of zero is different: the scorer raises on it.

    The law is compiled over the run's years. The corporate incidence channel
    runs over the longer macro horizon and extends the series with
    `scorer.rate_held_flat_past_the_window`.
    """
    values = getattr(law, "values", {})
    if CORPORATE_RATE_COLUMN not in values:
        return np.full(n_years, np.nan)
    stated = np.asarray(values[CORPORATE_RATE_COLUMN], dtype=np.float64).reshape(
        n_years, -1
    )
    active = getattr(law, "active", {}).get(CORPORATE_RATE_COLUMN)
    if active is None:
        return stated[:, 0]
    stated_here = np.asarray(active).reshape(n_years, -1)[:, 0]
    return np.where(stated_here, stated[:, 0], np.nan)


def corporate_rate_moves(baseline_rate: np.ndarray, scenario_rate: np.ndarray) -> bool:
    """Whether the scenario's corporate rate differs from baseline law's anywhere.

    A year either law says nothing about does not count. The tolerance is the
    scorer's, so this answer and the scorer's year-by-year answer agree.
    """
    both_stated = np.isfinite(baseline_rate) & np.isfinite(scenario_rate)
    moved = np.abs(scenario_rate - baseline_rate) >= scorer.NO_CHANGE_TOLERANCE
    return bool(np.any(both_stated & moved))


def corporate_paths_for_scenario(
    reform_rate: np.ndarray,
    baseline_rate: np.ndarray,
    years: Sequence[int],
    macro_root: str | Path,
    config,
    *,
    base_erodes: bool,
) -> corporate_paths.CorporatePaths | None:
    """One scenario's corporate incidence paths, or None where nothing changes.

    ``reform_rate`` and ``baseline_rate`` are the two laws' statutory corporate
    rates year by year, read by `corporate_rate_by_year`. ``config`` is the
    run's live ScenarioConfig, whose economy leg carries the channel's
    assumptions and the base-erosion elasticity. ``base_erodes`` says whether
    the revenue change the paths spread lets the corporate tax base erode
    (the conventional world) or holds it at its baseline level (the two
    mechanical worlds).
    """
    if not corporate_rate_moves(baseline_rate, reform_rate):
        return None

    macro = read_macro_spliced(macro_root).drop_duplicates("year").sort_values("year")
    missing = [name for name in _MACRO_COLUMNS if name not in macro.columns]
    if missing:
        raise ValueError(
            f"{macro_root} does not publish {', '.join(missing)}, which the "
            "corporate incidence channel needs"
        )
    grid_years = macro["year"].to_numpy(dtype=np.int64)

    # The simulated rate series over the whole macro horizon, held at its last
    # simulated value past the window and carrying no change before it.
    full_reform = scorer.rate_held_flat_past_the_window(
        grid_years, np.asarray(years, dtype=np.int64), reform_rate
    )
    full_baseline = scorer.rate_held_flat_past_the_window(
        grid_years, np.asarray(years, dtype=np.int64), baseline_rate
    )
    receipts = macro[_RECEIPTS_COLUMN].to_numpy(dtype=np.float64)
    rate_score = scorer.corporate_rate_revenue_change(
        full_baseline,
        full_reform,
        receipts,
        base_erodes=base_erodes,
        base_elasticity=float(config.economy_param("corp", "rate_eti")),
    )

    wedge_years, wedge_change = corporate_paths.assemble_corporate_wedge(
        (), (), grid_years, rate_score
    )
    corporate_paths.check_wedge_runs_one_way(wedge_change)
    corporate_paths.check_enactment_year_is_in_window(
        wedge_years, wedge_change, years
    )

    aggregates = corporate_paths.MacroAggregates(
        years=grid_years,
        after_tax_profit=(
            macro["gdp_corp"].to_numpy(dtype=np.float64) - receipts
        ),
        treasury_10_year_percent=macro["tsy_10y"].to_numpy(dtype=np.float64),
        interest_income=macro["gdp_interest"].to_numpy(dtype=np.float64),
        rental_income=macro["gdp_rent"].to_numpy(dtype=np.float64),
        proprietors_income=macro["gdp_proprietors"].to_numpy(dtype=np.float64),
    )
    built = corporate_paths.build_corporate_paths(
        wedge_years,
        wedge_change,
        aggregates,
        years,
        parameters=corporate_paths.corporate_parameters_from_economy_leg(config),
    )
    return built


def factor_arrays_for_years(
    built: corporate_paths.CorporatePaths | None,
    years: Sequence[int],
) -> dict[str, np.ndarray]:
    """The seven per-year numbers the corporate channel uses in a run.

    One array per factor, one value per simulation year. When there is no
    corporate reform the factors are at their identity values, so the frame's
    corporate step changes nothing and the compiled program keeps the same
    shape.
    """
    n_years = len(years)
    if built is None:
        return {
            "dividend_factor": np.ones(n_years),
            "interest_factor": np.ones(n_years),
            "rent_factor": np.ones(n_years),
            "pass_through_factor": np.ones(n_years),
            "markdown_rate": np.zeros(n_years),
            "flow_factor": np.zeros(n_years),
            "gain_quantity_factor": np.ones(n_years),
        }
    rows = [built.row_index_for_year(int(year)) for year in years]
    return {
        "dividend_factor": built.dividend_factor[rows],
        "interest_factor": built.interest_factor[rows],
        "rent_factor": built.rent_factor[rows],
        "pass_through_factor": built.pass_through_factor[rows],
        "markdown_rate": built.equity_markdown[rows],
        "flow_factor": built.payout_cut[rows],
        "gain_quantity_factor": built.gain_quantity_factor[rows],
    }


__all__ = [
    "BASELINE_CORPORATE_RATE_COLUMN",
    "CORPORATE_RATE_COLUMN",
    "corporate_paths_for_scenario",
    "corporate_rate_by_year",
    "corporate_rate_moves",
    "factor_arrays_for_years",
]
