"""Capital-gains model configuration.

Three kinds of settings are read here:

- the scenario's law on gains at death (the death regime for each asset
  class, the section 121 and death-gain exclusions, and whether income tax
  paid at death is deductible from the estate), from the compiled law (:func:`read_gains_law`);
- the calibrated numbers and judgment settings of the realization choice (the
  elasticity and the timeable share, the timing window, the appreciation
  rate, and the applier's two settings), from the behavior leg,
  the kg settings files and the estate valuation fit (:func:`read_calibration`,
  :func:`read_response_settings`, :func:`read_applier_settings`);
- the conversion channel's settings and its pool of convertible income
  (:func:`read_conversion_settings`, :func:`build_conversion_pool`).

Everything here runs on the host. The pieces traced inside the year are in
`behavior/gains/builders.py`; the host-side solves between the two executions
of the year loop are in `years/gains_solve.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml

from taxmodel.behavior.kg import applier as applier_module
from taxmodel.behavior.kg import stock_inputs
from taxmodel.behavior.kg.applier import DEEMED_ASSET_CLASSES
from taxmodel.behavior.kg.inputs import Calibration
from taxmodel.behavior.kg.prepass import DEFAULT_SETTINGS_FILE
from taxmodel.behavior.kg.stock_inputs import KG_N_AGES
from taxmodel.law.compile import FILING_STATUSES

Array = jnp.ndarray

# ---------------------------------------------------------------------------
# The scenario's law, read on the host
# ---------------------------------------------------------------------------

# The compiled law's names for the gains model's parameters. A compiled field
# is named for the file it was written in and then the parameter; these are
# in the preferred-rates file, `pref.yaml`.
_DEATH_REGIME_FIELDS: tuple[str, ...] = tuple(
    f"pref.kg_death_regime_{name}" for name in DEEMED_ASSET_CLASSES
)
_SECTION_121_FIELD = "pref.kg_sec121_excl"
_DEATH_GAIN_EXCLUSION_FIELD = "pref.kg_death_gain_excl"
# The row of the compiled law's filing-status axis holding the
# married-filing-jointly amount. A widowed filer's death-gain exclusion is the
# joint amount.
JOINT_FILING_STATUS_ROW = FILING_STATUSES.index(2)


@dataclass(frozen=True)
class GainsLaw:
    """The law values the capital-gains model reads, one row per run year.

    ``death_regime_by_class`` is [year, asset class] and holds 0 for a
    stepped-up basis, 1 for carryover basis and 2 for deemed realization at
    death, in :data:`~behavior.kg.applier.DEEMED_ASSET_CLASSES` order.

    The two exclusions keep their filing-status axis: [year, filing status], in
    the order of :data:`~taxmodel.law.compile.FILING_STATUSES`.
    """

    death_regime_by_class: np.ndarray
    section_121_exclusion: np.ndarray
    death_gain_exclusion: np.ndarray


def _law_value_by_year(
    law: object, name: str, n_years: int, absent: float
) -> np.ndarray:
    """One law parameter that does not vary by filing status, by year.

    The compiled law stores every parameter as [year, filing status, slot]. A
    parameter stated once for everyone has the same number in all four
    filing-status rows, so the first is read. A law that does not carry the
    parameter takes ``absent``.
    """
    values = getattr(law, "values", {})
    if name not in values:
        return np.full(n_years, absent, dtype=np.float64)
    stated = np.asarray(values[name], dtype=np.float64).reshape(n_years, -1)
    return np.nan_to_num(stated[:, 0], nan=absent)


def _law_value_by_year_and_filing_status(
    law: object, name: str, n_years: int
) -> np.ndarray:
    """One law parameter that does vary by filing status, [year, filing status].

    A parameter the law does not carry is zero, which for both exclusions
    means no exclusion.
    """
    values = getattr(law, "values", {})
    if name not in values:
        return np.zeros((n_years, len(FILING_STATUSES)), dtype=np.float64)
    stated = np.asarray(values[name], dtype=np.float64)
    stated = stated.reshape(n_years, len(FILING_STATUSES), -1)
    return np.nan_to_num(stated[:, :, 0], nan=0.0)


def read_gains_law(law: object, n_years: int) -> GainsLaw:
    """Read the gains model's law parameters off one compiled law."""
    return GainsLaw(
        death_regime_by_class=np.stack(
            [
                _law_value_by_year(law, name, n_years, absent=0.0)
                for name in _DEATH_REGIME_FIELDS
            ],
            axis=-1,
        ),
        section_121_exclusion=_law_value_by_year_and_filing_status(
            law, _SECTION_121_FIELD, n_years
        ),
        death_gain_exclusion=_law_value_by_year_and_filing_status(
            law, _DEATH_GAIN_EXCLUSION_FIELD, n_years
        ),
    )


def _filing_status_row(filing_status: Array) -> Array:
    """Each household's row in a law parameter's filing-status axis.

    Filing statuses are numbered from one and the axis is indexed from zero.
    """
    codes = jnp.asarray(filing_status).astype(jnp.int32)
    return jnp.clip(codes - min(FILING_STATUSES), 0, len(FILING_STATUSES) - 1)


# The name of the applier's allocation setting in the gains settings file
# (`prepass.DEFAULT_SETTINGS_FILE`): how the cell's extra realizations are
# spread within the cell.
_ALLOCATION_SETTING = "applier_allocation"

# The estate tax's fitted valuation factors. The tax at death values a
# decedent's pass-through interests at the same fraction of market value the
# estate tax does, `rho_pt` in this file.
ESTATE_VALUATION_FILE = Path("config/calibrations/estate/valuation_fit.yaml")
_PASS_THROUGH_VALUATION_SETTING = "rho_pt"

_TIMING_WINDOW_SETTING = "timing_window"
_TIMING_REFERENCE_WEDGE_SETTING = "timing_ref_wedge"
_APPRECIATION_SETTING = "appreciation"

# The two fitted numbers in a calibration file
# (config/calibrations/kg/realization.yaml).
_ELASTICITY_SETTING = "eta"
_TIMEABLE_SHARE_SETTING = "timeable_share"

# The piece of the kg_dynamics section that binds the fitted values, and the
# file bound when a leg does not name one.
REALIZATION_PIECE = "realization"
DEFAULT_REALIZATION_FILE = Path("config/calibrations/kg/realization.yaml")


def _setting(body: Mapping[str, Any], name: str, where: Path) -> Any:
    """One entry's value out of a calibration file."""
    entry = body.get(name)
    if entry is None:
        raise ValueError(
            f"{where} has no entry for {name}, which the capital-gains model needs"
        )
    return entry["value"] if isinstance(entry, Mapping) else entry


@dataclass(frozen=True)
class ResponseSettings:
    """The judgment settings of the selling response.

    ``timing_reference_wedge`` is the rate change at which a whole year's
    timeable dollars move, ``timing_window`` is how many years either side
    they may move to, and ``appreciation`` is the yearly nominal rate at which
    assets appreciate. All three are in ``config/calibrations/kg/settings.yaml``. The two fitted
    numbers are conditioned on them, so changing one requires a refit.

    The calibrate command reads these while solving for the two fitted
    numbers, so they are read separately from the calibration.
    """

    timing_reference_wedge: float
    timing_window: int
    appreciation: float


def read_response_settings(
    settings_file: Path | str = DEFAULT_SETTINGS_FILE,
) -> ResponseSettings:
    """Read the judgment settings from the settings file."""
    settings_file = Path(settings_file)
    settings = yaml.safe_load(settings_file.read_text()) or {}
    return ResponseSettings(
        timing_reference_wedge=float(
            _setting(settings, _TIMING_REFERENCE_WEDGE_SETTING, settings_file)
        ),
        timing_window=int(_setting(settings, _TIMING_WINDOW_SETTING, settings_file)),
        appreciation=float(_setting(settings, _APPRECIATION_SETTING, settings_file)),
    )


def placeholder_calibration(
    settings_file: Path | str = DEFAULT_SETTINGS_FILE,
) -> Calibration:
    """A calibration with the two fitted numbers set to zero.

    The frozen pass pins turnover to baseline law's, so it does not read the
    elasticity or the timeable share. The calibrate command uses this while
    solving for the real values. The judgment settings are read from the
    settings file.
    """
    return _calibration(read_response_settings(settings_file), 0.0, 0.0)


def _calibration(
    judgments: ResponseSettings, elasticity: float, timeable_share: float
) -> Calibration:
    """The two fitted numbers plus the judgment settings."""
    return Calibration(
        elasticity=elasticity,
        timeable_share=timeable_share,
        appreciation=judgments.appreciation,
        timing_reference_wedge=judgments.timing_reference_wedge,
        timing_window=judgments.timing_window,
    )


def read_calibration(
    behavior_leg: Any,
    settings_file: Path | str = DEFAULT_SETTINGS_FILE,
) -> Calibration:
    """The calibrated numbers the realization schedule is solved with.

    The two retiming settings and the appreciation rate are in the settings
    file.
    The elasticity and the timeable share come from the file the scenario's
    behavior leg binds under ``kg_dynamics: realization:``, so a sweep over the
    elasticity binds a different file of the same shape.
    """
    return read_fitted_calibration(
        behavior_leg.kg_dynamics.get(REALIZATION_PIECE) or DEFAULT_REALIZATION_FILE,
        settings_file,
    )


def read_fitted_calibration(
    realization_file: Path | str = DEFAULT_REALIZATION_FILE,
    settings_file: Path | str = DEFAULT_SETTINGS_FILE,
) -> Calibration:
    """The two fitted numbers from one calibration file.

    A caller without a behavior leg (the calibrate command) uses this to read
    the default file.
    """
    judgments = read_response_settings(settings_file)
    realization_file = Path(realization_file)
    fitted = yaml.safe_load(realization_file.read_text()) or {}
    return _calibration(
        judgments,
        float(_setting(fitted, _ELASTICITY_SETTING, realization_file)),
        float(_setting(fitted, _TIMEABLE_SHARE_SETTING, realization_file)),
    )


@dataclass(frozen=True)
class ApplierSettings:
    """The applier's two settings.

    ``allocation_weight_on_holdings`` is a judgment value.
    ``pass_through_valuation_factor`` is the fraction of market value at which
    a decedent's pass-through interests are valued for the tax at death.
    """

    allocation_weight_on_holdings: float
    pass_through_valuation_factor: float


def read_applier_settings(
    settings_file: Path | str = DEFAULT_SETTINGS_FILE,
    valuation_file: Path | str = ESTATE_VALUATION_FILE,
) -> ApplierSettings:
    """Read the applier's allocation weight from the gains settings file and
    the pass-through valuation factor from the estate tax's valuation fit.
    """
    settings_file = Path(settings_file)
    body = yaml.safe_load(settings_file.read_text()) or {}
    valuation_file = Path(valuation_file)
    valuation = yaml.safe_load(valuation_file.read_text()) or {}
    return ApplierSettings(
        allocation_weight_on_holdings=float(
            _setting(body, _ALLOCATION_SETTING, settings_file)
        ),
        pass_through_valuation_factor=float(
            _setting(valuation, _PASS_THROUGH_VALUATION_SETTING, valuation_file)
        ),
    )


# ---------------------------------------------------------------------------
# The conversion channel's inputs
# ---------------------------------------------------------------------------

# The income-conversion channel's two numbers: sigma, the dollars relabeled
# per dollar of convertible income per unit of rate wedge, and the labor
# content of active pass-through income. Both are in the conversion file the
# behavior leg binds under `kg_dynamics: conversion:`.
CONVERSION_PIECE = "conversion"
_SIGMA_SETTING = "conv"
_PASS_THROUGH_LABOR_SHARE_SETTING = "pt_labor_share"


def read_conversion_settings(conversion_file: Path | str) -> tuple[float, float]:
    """Sigma and the pass-through labor share, from the conversion file."""
    body = yaml.safe_load(Path(conversion_file).read_text()) or {}
    return (
        float(_setting(body, _SIGMA_SETTING, Path(conversion_file))),
        float(_setting(body, _PASS_THROUGH_LABOR_SHARE_SETTING, Path(conversion_file))),
    )


# The five ordinary-income legs a household can relabel as capital gain, and
# the stored rate column that prices each one. The sole-proprietorship leg is
# the two earners' income together, priced at the first earner's rate.
CONVERSION_LEG_RATE_NAMES: tuple[str, ...] = (
    "wages_base1",
    "wages_base2",
    "part_active",
    "scorp_active",
    "sole_prop1",
)

# The compiled law's ordinary-rate brackets. The last finite value per year
# and filing status is the top-bracket threshold the conversion pool is
# limited to.
_ORDINARY_BRACKETS_FIELD = "ord.brackets"


def _top_bracket_threshold(law: object, n_years: int) -> np.ndarray:
    """The taxable income where the top ordinary bracket starts, [year, status]."""
    values = getattr(law, "values", {})
    if _ORDINARY_BRACKETS_FIELD not in values:
        raise ValueError(
            "the compiled law carries no ordinary-rate brackets "
            f"({_ORDINARY_BRACKETS_FIELD}), so the conversion pool's "
            "top-bracket test cannot be resolved"
        )
    brackets = np.asarray(values[_ORDINARY_BRACKETS_FIELD], dtype=np.float64)
    brackets = brackets.reshape(n_years, len(FILING_STATUSES), -1)
    thresholds = np.full((n_years, len(FILING_STATUSES)), np.nan)
    for year in range(n_years):
        for status in range(len(FILING_STATUSES)):
            finite = brackets[year, status][np.isfinite(brackets[year, status])]
            if finite.size:
                thresholds[year, status] = finite[-1]
    return thresholds


def build_conversion_pool(
    host_records: Mapping[str, Array],
    rate_exports: Mapping[str, Array],
    reform_law: object,
    n_years: int,
) -> stock_inputs.ConversionPool:
    """The conversion channel's per-household inputs, from the first execution.

    ``host_records`` holds the run's households stacked with a row per year.
    A converted dollar lands in the household's age cell.

    The leg amounts, taxable income and filing status come off the mechanical
    frame, and the leg rates and the rate on realized long-term gain are the
    same frame's prices under both laws. The weight and the age come from the
    host records, which describe the same households in the same order. The
    top-bracket threshold is the reform law's, resolved by filing status.
    """

    def stacked(name: str) -> Array:
        return exported_column(rate_exports, name)

    weight = jnp.asarray(host_records["weight"])
    age_index = applier_module.cell_row_for_household(host_records, KG_N_AGES)

    filing_status = np.asarray(stacked("mechanical_filing_status"))
    thresholds = _top_bracket_threshold(reform_law, n_years)
    row = np.asarray(_filing_status_row(filing_status))
    threshold_by_household = jnp.asarray(
        thresholds[np.arange(n_years)[:, None], row]
    )

    leg_amount = jnp.stack(
        [
            stacked("mechanical_wages_base1"),
            stacked("mechanical_wages_base2"),
            stacked("mechanical_part_active"),
            stacked("mechanical_scorp_active"),
            stacked("mechanical_sole_prop1") + stacked("mechanical_sole_prop2"),
        ],
        axis=-1,
    )
    reform_rates = jnp.stack(
        [
            stacked(f"reform_on_mechanical_mtr_{name}")
            for name in CONVERSION_LEG_RATE_NAMES
        ],
        axis=-1,
    )
    baseline_rates = jnp.stack(
        [
            stacked(f"baseline_on_mechanical_mtr_{name}")
            for name in CONVERSION_LEG_RATE_NAMES
        ],
        axis=-1,
    )
    return stock_inputs.ConversionPool(
        weight=weight,
        age_index=age_index,
        taxable_income=stacked("mechanical_txbl_inc"),
        top_bracket_threshold=threshold_by_household,
        leg_amount=leg_amount,
        marginal_rate_scenario=reform_rates,
        marginal_rate_baseline=baseline_rates,
        gains_rate_scenario=stacked("reform_on_mechanical_mtr_kg_lt"),
        gains_rate_baseline=stacked("baseline_on_mechanical_mtr_kg_lt"),
    )


# ---------------------------------------------------------------------------
# Reading one exported column back off the device
# ---------------------------------------------------------------------------


def exported_column(rate_exports: Mapping[str, Array], name: str) -> Array:
    """One column from the compiled program's exports."""
    if name not in rate_exports:
        raise KeyError(
            f"the first execution's exports are missing {name}; the run that "
            "produced them was built without the gains model's rate columns"
        )
    return jnp.asarray(rate_exports[name])
