"""A scenario's three legs, their vintages, and the run-level inputs read from disk.

This module compiles a scenario's law, resolves its three legs, refuses to run
on stale provenance, derives the price and wage index series, and reads the
frozen estate measurement bridge. `run/plan.py` reads a runscript into a plan,
`run/scenarios.py` executes the plan, `years/loop.py` runs the scenarios
through the compiled year loop, and `run/output_directory.py` writes the
results.

The population comes from the bridge output for a Tax-Data vintage
(`frame/bridge.py`): one Parquet file per year, with the wage split done and
the random draws already in columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from taxmodel.behavior.config import BehaviorLeg, read_behavior_leg
from taxmodel.calibration.staleness import check_for_run as check_gains_calibration
from taxmodel.config import (
    ResolvedLaw,
    ResolvedLeg,
    ScenarioConfig,
    StalenessReport,
    check_leg,
    check_toolchain,
    interface_vintages,
    load_defaults,
    resolve,
    resolve_law,
    resolve_law_stack,
)
from taxmodel.config.interfaces import interface_root
from taxmodel.law import compile_law
from taxmodel.law.compile import CompiledLaw, compile_resolved
from taxmodel.law.indexes import derive_index_table, index_table_from_frame
from taxmodel.taxes.calc.estate import EstateParameters

# The tax-law and scenario configuration tree.
DEFAULT_CONFIG_ROOT = Path("config/scenarios")
DEFAULT_ESTATE_BRIDGE = Path("config/calibrations/estate/bridge.yaml")
# Root that the provenance paths on calibrated config entries are relative to.
# Those entries name the files that produced them or that they depend on, and
# the paths resolve against this repo, run from the repo root.
DEFAULT_PROVENANCE_ROOT = Path(".")


# --------------------------------------------------------------------------
# Reading law
# --------------------------------------------------------------------------


def read_compiled_npz(path: str | Path) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    """Read a frozen compiled law: its value arrays and its year axis.

    The compiler writes each field twice, as `value/<name>` and
    `active/<name>` (law/compile.py). Only the values are read here. Where a
    field is inactive its value is NaN, and the calculators' schedule helpers
    treat NaN as inactive.
    """
    with np.load(path) as archive:
        years = tuple(int(y) for y in archive["years"])
        values = {
            name[len("value/") :]: archive[name]
            for name in archive.files
            if name.startswith("value/")
        }
    return values, years


def wide_law_table(law: CompiledLaw) -> pd.DataFrame:
    """The compiled law as a table with a row per year and filing status.

    The frame layer joins law onto records by year and filing status, with one
    column per parameter and a trailing number on the columns of a schedule.
    The table comes from the compiler's own table writer.
    """
    header, rows = law.to_frame()
    return pd.DataFrame(rows, columns=header)


def index_series(
    index_table: str | Path | None = None,
    economy: str | None = None,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
) -> tuple[pd.DataFrame, str]:
    """The growth rates for indexed law parameters, and the path they came from.

    By default they are derived from the Macro-Projections publication the run
    is pinned to (law/indexes.py). Passing a path to a three-column CSV reads
    that file instead, so that a test or a rerun can pin an old table.

    The table is returned as a data frame. The frame layer splices its own CPI
    level series out of it for the capital-gain basis ratio.

    The second return value is the path the table came from, for the run
    manifest.
    """
    if index_table is not None:
        return pd.read_csv(index_table), str(index_table)
    macro_root = interface_root(
        "Macro-Projections", economy=economy, config_root=config_root
    )
    return derive_index_table(macro_root), str(macro_root)


def read_estate_bridge(
    path: str | Path = DEFAULT_ESTATE_BRIDGE,
) -> tuple[EstateParameters, float]:
    """The frozen estate measurement parameters, and the donor-clone cap.

    The parameters are measurement, not law: the share of economic wealth that
    shows up on a reported estate (`r`), the extra discount closely held
    business interests get (`rho_pt`), lifetime gifts as a share of the
    reported estate (`gamma`), and four size bins carrying the deduction share,
    the probability the estate uses a deceased spouse's unused exclusion, and
    how much of it. A reform cannot change any of them.

    `config/calibrations/estate/bridge.yaml` holds the estate calculator's
    inputs (the valuation factors and the per-bin SOI shares) and the frame's
    mortality-step input (the cap on expected deaths in one cluster of
    replicated donor records). The two were calibrated together.
    """
    bridge = yaml.safe_load(Path(path).read_text())
    ordered = pd.DataFrame(bridge["bins"]).sort_values("lo")
    parameters = EstateParameters(
        reporting_rate=jnp.asarray(float(bridge["r"])),
        pass_through_valuation_rate=jnp.asarray(float(bridge["rho_pt"])),
        gift_ratio=jnp.asarray(float(bridge["gamma"])),
        bin_lower_bounds=jnp.asarray(ordered["lo"].to_numpy(dtype=np.float64)),
        deduction_shares=jnp.asarray(ordered["f_ded"].to_numpy(dtype=np.float64)),
        dsue_probabilities=jnp.asarray(
            ordered["p_dsue"].to_numpy(dtype=np.float64)
        ),
        dsue_shares=jnp.asarray(ordered["f_dsue"].to_numpy(dtype=np.float64)),
    )
    return parameters, float(bridge["cluster_death_weight_cap"])


# --------------------------------------------------------------------------
# Resolving the scenario and checking its provenance
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """A scenario's three resolved legs and the findings of the provenance checks.

    Records which law, economy and behavior folders were in force, which
    upstream data vintages they name, and every stale finding and warning.
    `config` is the configuration the model reads economy values from.
    """

    law: ResolvedLaw
    economy: ResolvedLeg
    behavior: ResolvedLeg
    vintages: dict[str, str]
    staleness: StalenessReport
    config: ScenarioConfig
    # The behavior leg as the simulation reads it: which pieces of the gains
    # machinery it binds, and which modules it runs. `behavior` above is the
    # leg's calibrated parameter entries.
    behavior_leg: BehaviorLeg | None = None
    # Findings of the capital-gains calibration check, split into those that
    # stop the run and those that only warn. Empty for a run whose behavior
    # leg does not bind the gains machinery.
    calibration_staleness: StalenessReport = field(
        default_factory=StalenessReport
    )


def resolve_scenario_provenance(
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    alternative: str | None = None,
    economy: str | None = None,
    behavior: str | None = None,
    provenance_root: str | Path = DEFAULT_PROVENANCE_ROOT,
    law_layers: Sequence[str] | None = None,
    accept_stale_calibration: bool = False,
) -> Provenance:
    """Resolve the scenario's three legs and refuse to run on stale provenance.

    All of this happens before any record is read. The steps, in order:

    1. the toolchain check, which rejects a known-bad jax and CUDA pairing;
    2. the three legs resolved, each as its complete default layer with the
       named alternative laid over it;
    3. the staleness check on both configured legs and on the calibrated
       constants declared in model code, with all findings collected before
       any of them stops the run;
    4. the behavior leg read as the simulation reads it (the gains machinery
       it binds and the modules it runs) and, when it binds the gains
       machinery, the capital-gains calibration checked against what this run
       declares. Drift in the declared targets or the data vintages stops the
       run unless `accept_stale_calibration` is set; code drift only warns.

    `alternative` names one reform folder. `law_layers` names several folders
    folded in order, as one rung of a stacked reform runs, so that the rung's
    provenance is resolved against the whole fold. Pass one or the other, not
    both.
    """
    check_toolchain()

    config_root = Path(config_root)
    base_dir = Path(provenance_root)
    if law_layers is not None and alternative is not None:
        raise ValueError(
            "a scenario's law is either one reform folder (`alternative`) or a "
            "list of folders folded in order (`law_layers`), not both"
        )
    law = (
        resolve_law(config_root, alternative)
        if law_layers is None
        else resolve_law_stack(config_root, [str(layer) for layer in law_layers])[-1]
    )
    economy_leg = resolve(
        config_root, load_defaults(config_root, "economy", base_dir=base_dir), economy
    )
    behavior_leg = resolve(
        config_root, load_defaults(config_root, "behavior", base_dir=base_dir), behavior
    )

    vintages = interface_vintages(economy_leg)
    findings = check_leg(economy_leg, vintages)
    findings.extend(check_leg(behavior_leg, vintages))
    findings.raise_if_stopping()

    # The behavior leg as the simulation reads it, and the gains model's own
    # staleness check against it. A leg that binds no gains machinery is not
    # checked, and `read_behavior_leg` answers the reserved word `none`
    # without reading any file.
    behavior_modules = read_behavior_leg(config_root, behavior)
    gains_staleness = check_gains_calibration(
        {"kg_dynamics": dict(behavior_modules.kg_dynamics)},
        vintages,
        repo_root=Path.cwd(),
    )
    gains_staleness.raise_if_stopping(accept_stale_calibration)

    config = ScenarioConfig(economy=economy_leg)
    return Provenance(
        law=law,
        economy=economy_leg,
        behavior=behavior_leg,
        vintages=vintages,
        staleness=findings,
        config=config,
        behavior_leg=behavior_modules,
        calibration_staleness=gains_staleness,
    )


# --------------------------------------------------------------------------
# Run-level inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """The run-level inputs that do not come from Tax-Data.

    Read once and used for every year: the compiled law, the growth series,
    and the frozen estate measurement parameters. `values` is the law the run
    computes tax under. Baseline law is its own Scenario built by the same
    function.
    """

    years: tuple[int, ...]
    # The law alternative folder, or None on a baseline run.
    alternative: str | None
    values: dict[str, np.ndarray]
    law_table: pd.DataFrame
    indexes: pd.DataFrame
    estate_params: EstateParameters
    cluster_death_weight_cap: float
    # The law as the compiler produced it. `tax_law.csv` publishes its
    # year-by-parameter table and the time-burden table reads two switches off
    # it. `values` above is this object's numbers.
    compiled_law: CompiledLaw | None = None


def load_scenario(
    years: Sequence[int],
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    index_table: str | Path | None = None,
    alternative: str | None = None,
    economy: str | None = None,
    resolved_law: ResolvedLaw | None = None,
    compiled_law: CompiledLaw | None = None,
    growth: pd.DataFrame | None = None,
    estate_bridge: tuple[EstateParameters, float] | None = None,
) -> Scenario:
    """Compile the law and read the other run-level inputs.

    `resolved_law` is a parameter tree that several provision folders were
    folded into (`config/law_stacks.expand_stack`), as one rung of a stacked
    reform runs. Supplying it compiles that tree instead of a folder;
    `alternative` then only names the scenario.

    `growth` and `estate_bridge` are the index table and the estate bridge
    already read for another scenario of the same run. Left out, they are read
    from disk here.
    """
    if growth is None:
        growth, _source = index_series(index_table, economy, config_root)
    compile_years = [int(y) for y in years]
    index = index_table_from_frame(growth)
    if compiled_law is not None:
        # A law compiled inside a batch carries the batch's layout, which the
        # binding step reads.
        law = compiled_law
    elif resolved_law is not None:
        law = compile_resolved(resolved_law, compile_years, index)
        alternative = resolved_law.alternative
    else:
        law = compile_law(Path(config_root), compile_years, index, alternative=alternative)
    if estate_bridge is None:
        estate_bridge = read_estate_bridge()
    estate_params, cluster_cap = estate_bridge
    return Scenario(
        years=tuple(law.years),
        alternative=alternative,
        values=dict(law.values),
        law_table=wide_law_table(law),
        indexes=growth,
        estate_params=estate_params,
        cluster_death_weight_cap=cluster_cap,
        compiled_law=law,
    )
