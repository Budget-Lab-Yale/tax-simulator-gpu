"""Revenue change from a corporate statutory rate change, on top of CBO's receipts.

The corporate tax is not simulated firm by firm. CBO's projection of corporate
receipts is the level, and this module supplies the change from a reform in
the statutory rate. Depreciation reform is scored elsewhere, by the
Cost-Recovery-Simulator; other corporate changes stay off-model.

**The arithmetic.** The corporate tax base is assumed to have a constant
elasticity with respect to the net-of-tax rate. Write the baseline statutory
rate as t0, the scenario's rate as t, CBO's baseline receipts as R0, and the
elasticity as e. Then

    base    B(t) = (R0 / t0) * ((1 - t) / (1 - t0)) ** e
    revenue R(t) = t * B(t)

The base at the baseline rate is R0 / t0. Two ways to score a rate change
follow:

- Holding the base at its baseline level, the change is R0 * (t / t0 - 1).
  This is the mechanical change.
- Letting the base erode, the change is R0 * ((t / t0) * ((1 - t) / (1 - t0))
  ** e - 1).

Revenue is maximized at a rate of 1 / (1 + e), where e is the elasticity
``corp.rate_eti``.

**Scores by world.** The conventional world gets the eroded change and the
other reform worlds get the mechanical change; `corporate_rate_score_by_world`
applies this rule. The baseline world is scored at zero because it runs
baseline law.

**Two biases in the estimate**, neither corrected here:

- It is an upper bound on the lasting effect. The elasticity is measured from
  bunching at the zero-income kink, so it captures income moved out of a
  single year but not income moving back. A flat change in the statutory rate
  leaves no rate difference between years in the long run, only a one-time
  shift at enactment, so treating the whole elasticity as permanent overstates
  the persistent response.
- The base is too large. Dividing all of CBO's corporate receipts by the
  statutory rate treats every dollar as taxed at that rate, which dollars
  under the corporate alternative minimum tax and the international
  provisions are not.

**Timing.** The rate for calendar year t is applied to CBO's receipts labelled
year t, with no lag and no splitting of a calendar year across two fiscal
years. CBO's receipts series is on a fiscal year basis, and a calendar year of
corporate liability is collected partly in the following fiscal year, so this
is a simplification.

The rate series is passed in as an argument, already resolved from the law.
"""

from __future__ import annotations

import numpy as np

from taxmodel.years.world_names import BASELINE_WORLD

Array = np.ndarray

# A rate change smaller than this counts as no change.
NO_CHANGE_TOLERANCE = 1e-12

# The world whose corporate score lets the base erode. Every other world gets
# the mechanical change.
WORLD_WITH_BASE_EROSION = "conventional"


def corporate_rate_revenue_change(
    baseline_rate: Array,
    scenario_rate: Array,
    cbo_corporate_receipts: Array,
    *,
    base_erodes: bool,
    base_elasticity: float,
) -> Array:
    """The change in corporate receipts from a statutory rate change, by year.

    All three series are indexed by the same years, in the same order. The
    result is in the same money unit as `cbo_corporate_receipts`.

    Arguments:
      baseline_rate: the top corporate statutory rate under baseline law, by
        year, resolved from the baseline law's `corp.rate`.
      scenario_rate: the same rate under the scenario's law.
      cbo_corporate_receipts: CBO's projection of corporate receipts, by year.
      base_erodes: True lets the base shrink as the rate rises (the
        conventional estimate); False holds it at its baseline level (the
        mechanical estimate).
      base_elasticity: the elasticity of the corporate tax base to the
        net-of-tax statutory rate, from the economy leg,
        `economy_param("corp", "rate_eti")`. The value and its citation are in
        config/scenarios/economy/default/corp.yaml.

    A year is scored at zero when either rate is missing, when receipts are
    missing, or when the two rates differ by less than `NO_CHANGE_TOLERANCE`.
    Missing values are not-a-number entries.
    """
    baseline_rate = np.asarray(baseline_rate, dtype=np.float64)
    scenario_rate = np.asarray(scenario_rate, dtype=np.float64)
    receipts = np.asarray(cbo_corporate_receipts, dtype=np.float64)
    if not (baseline_rate.shape == scenario_rate.shape == receipts.shape):
        raise ValueError(
            "the two rate series and the receipts series must cover the same "
            f"years: got {baseline_rate.shape}, {scenario_rate.shape} and "
            f"{receipts.shape}"
        )

    scored = (
        np.isfinite(baseline_rate)
        & np.isfinite(scenario_rate)
        & np.isfinite(receipts)
        & (np.abs(scenario_rate - baseline_rate) >= NO_CHANGE_TOLERANCE)
    )
    if np.any(scored & (baseline_rate == 0.0)):
        # The baseline rate is the denominator of the base, so there is no
        # finite answer. A baseline with no corporate tax is a configuration
        # mistake.
        raise ValueError(
            "the baseline corporate statutory rate is zero in a year whose "
            "rate changes, so the corporate tax base is undefined there"
        )

    # Unscored years get a stand-in rate of one half in both series so that a
    # missing or zero rate does not divide by zero. Their result is discarded
    # below.
    stand_in_rate = 0.5
    baseline_where_scored = np.where(scored, baseline_rate, stand_in_rate)
    scenario_where_scored = np.where(scored, scenario_rate, stand_in_rate)
    receipts_where_scored = np.where(scored, receipts, 0.0)

    rate_ratio = scenario_where_scored / baseline_where_scored
    if base_erodes:
        net_of_tax_ratio = (1.0 - scenario_where_scored) / (1.0 - baseline_where_scored)
        change = receipts_where_scored * (
            rate_ratio * net_of_tax_ratio**base_elasticity - 1.0
        )
    else:
        change = receipts_where_scored * (rate_ratio - 1.0)
    return np.where(scored, change, 0.0)


def corporate_rate_score_by_world(
    worlds: tuple[str, ...],
    baseline_rate: Array,
    scenario_rate: Array,
    cbo_corporate_receipts: Array,
    *,
    base_elasticity: float,
) -> dict[str, Array]:
    """One corporate rate score per world, ready for the receipts corporate line.

    The baseline world is scored at zero, since it runs baseline law. The
    conventional world's score lets the base erode and the others' do not.
    """
    n_years = len(np.asarray(cbo_corporate_receipts))
    scores: dict[str, Array] = {}
    for world in worlds:
        if world == BASELINE_WORLD:
            scores[world] = np.zeros(n_years, dtype=np.float64)
            continue
        scores[world] = corporate_rate_revenue_change(
            baseline_rate,
            scenario_rate,
            cbo_corporate_receipts,
            base_erodes=(world == WORLD_WITH_BASE_EROSION),
            base_elasticity=base_elasticity,
        )
    return scores


def rate_held_flat_past_the_window(
    years: Array,
    rate_years: Array,
    rate: Array,
) -> Array:
    """A rate series spread over a longer run of years, held flat at the end.

    The receipts line needs only the simulation years, but the corporate
    incidence channel needs the same rate change over the whole macro horizon.
    Years before the first year of the rate series carry no rate, and years
    after the last one hold the last value.

    Years with no rate come back as not-a-number, which
    `corporate_rate_revenue_change` scores at zero.
    """
    years = np.asarray(years)
    rate_years = np.asarray(rate_years)
    rate = np.asarray(rate, dtype=np.float64)
    if rate_years.shape != rate.shape:
        raise ValueError(
            "the rate series needs one rate per year: got "
            f"{rate.shape} rates for {rate_years.shape} years"
        )
    order = np.argsort(rate_years)
    sorted_years = rate_years[order]
    sorted_rate = rate[order]
    # For each year, the last rate-series year at or before it. Years before
    # the series starts get index -1 and are blanked below.
    position = np.searchsorted(sorted_years, years, side="right") - 1
    before_the_series = position < 0
    held = sorted_rate[np.clip(position, 0, None)]
    return np.where(before_the_series, np.nan, held)


__all__ = [
    "NO_CHANGE_TOLERANCE",
    "WORLD_WITH_BASE_EROSION",
    "corporate_rate_revenue_change",
    "corporate_rate_score_by_world",
    "rate_held_flat_past_the_window",
]
