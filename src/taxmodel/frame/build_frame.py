"""Building the household frame for one world of one year.

This sits between the population block and the tax calculators. Each year,
each world starts from the raw household arrays for that year and needs the
frame steps run over them: the head-of-household recode, the law join, the
state-tax-workaround step, the Social Security step, the capital-gains basis
ratio, estate mortality, and net worth (`frame/steps_jax.build_world_frame`).
The reform-side worlds then take the two mechanical income adjustments, the
corporate incidence path and the employer-payroll wage change, which are
incidence assumptions rather than household behavior.

:func:`make_frame_builder` closes over what does not change from year to
year or from scenario to scenario, and returns the callable
`years/worlds.py` calls with one year's raw population arrays, that year's
bound law (`scenario/law_by_year.AnnualLaw`), the calendar year, and whether
to apply the income adjustments. The callable returns a dictionary of
equal-length arrays whose key set does not change from year to year.

The employer-payroll adjustment is defined against baseline employer FICA
liability. The builder is handed one law at a time, so baseline payroll law
is carried on the year's law and the adjustment computes the baseline leg
from it.

Four inputs are not Population columns. The comment above
``FRAME_STEP_RECORD_COLUMNS`` below says what they are and where each one
comes from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import pandas as pd

from taxmodel.frame import steps_jax, widowhood
from taxmodel.frame.population import PopulationSource
from taxmodel.scenario.law_by_year import AnnualLaw, cpi_level_series
from taxmodel.taxes.calc.payroll import calc_pr
from taxmodel.years import income_adjustments

Array = jnp.ndarray
Records = Mapping[str, Array]

# The floor on the gross estate a record must clear before it can join a
# donor-clone cluster.
DONOR_CLUSTER_GROSS_ESTATE_FLOOR = 5e6

# ---------------------------------------------------------------------------
# Frame step inputs
# ---------------------------------------------------------------------------
#
# Every name the JAX frame steps read off a record is a Population field under
# that name: the block stores Tax-Data's columns without renaming them. The
# frame steps produce kg_lt_cpi_ratio, salt_workaround_part,
# salt_workaround_scorp and net_worth, plus three from the capital-gains step:
# unrealized_gain_stock, m_household and p_widow.
#
# Four inputs are not Population fields:
#
# - "valid", the Population validity mask for the year, which the runner puts
#   in the raw arrays under this name;
# - "year", the calendar year, passed to the frame steps as a scalar;
# - the annual law columns, the wide law table's row for this year and the four
#   filing statuses, built by annual_frame_law_columns_for_years;
# - the CPI level series, spliced CPI-U and chained CPI-U levels from the run's
#   growth-rate table, built once on the host by cpi_level_series.

# The record columns the frame steps read. A Population missing one is refused
# before the year program is compiled, with the missing names in the message.
FRAME_STEP_RECORD_COLUMNS: tuple[str, ...] = (
    "id",
    "filing_status",
    "weight",
    "age1",
    "age2",
    # Sex and any divorce on record, which together with age decide how likely
    # an unmarried filer is to be a surviving spouse.
    "male1",
    "divorce_year",
    "gross_ss",
    "kg_lt",
    "kg_lt_years_held",
    "q_death1",
    "q_death2",
    "salt_inc_sales",
    "sstb_part",
    "sstb_scorp",
    "part_active",
    "part_passive",
    "part_active_loss",
    "part_passive_loss",
    "part_179",
    "scorp_active",
    "scorp_passive",
    "scorp_active_loss",
    "scorp_passive_loss",
    "scorp_179",
    "r.salt_workaround",
    *steps_jax.SALT_OTHER_INCOME_COLUMNS,
    *steps_jax.ESTATE_ASSET_COLS,
    *steps_jax.ESTATE_DEBT_COLS,
    # The market values are already in the balance sheet above; these are the
    # matching cost bases, which the capital-gains model differences against
    # them to get the stock of unrealized gain.
    *steps_jax.CAPITAL_GAINS_BASIS_COLS,
)


def missing_frame_step_inputs(available: Sequence[str]) -> list[str]:
    """Frame-step inputs this population does not carry."""
    have = set(available)
    return sorted(set(FRAME_STEP_RECORD_COLUMNS) - have)


def check_population_covers_the_frame_steps(population: PopulationSource) -> None:
    """Stop before tracing if the population cannot feed the frame steps."""
    missing = missing_frame_step_inputs(population.field_names)
    if missing:
        raise ValueError(
            "the population block does not carry every column the frame steps "
            "read: " + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# The frame builder itself
# ---------------------------------------------------------------------------


def apply_reform_income_adjustments(
    frame: Records,
    law: AnnualLaw,
    corporate_asset_exposure: Mapping[str, float] | None = None,
    corporate_gain_share: float = 0.0,
    gain_dynamics_active: bool = False,
) -> dict[str, Array]:
    """Apply the corporate and employer-payroll income changes, once each.

    The corporate incidence path runs first, then the employer-payroll wage
    change, which is measured against employer FICA under baseline law on the
    same households.

    The corporate step's per-year numbers come off the law
    (:class:`CorporateYearFactors`) and are the identity for a scenario with no
    corporate reform. The three arguments do not vary with the scenario: which
    balance-sheet columns are exposed to corporate equity, the corporate share
    of realized gains, and whether the capital-gains model is running, in
    which case the gains stock takes the markdown and this step leaves
    realized gains alone.

    The baseline wage and the baseline employer FICA on it are computed from
    this frame under baseline payroll law. No earlier frame step changes a
    wage, so this equals the baseline run's own wage and employer FICA.
    """
    factors = law.corporate
    adjusted = income_adjustments.apply_corporate_adjustment(
        frame,
        dividend_factor=factors.dividend_factor,
        interest_factor=factors.interest_factor,
        rent_factor=factors.rent_factor,
        pass_through_factor=factors.pass_through_factor,
        markdown_rate=factors.markdown_rate,
        flow_factor=factors.flow_factor,
        asset_exposure=dict(corporate_asset_exposure or {}),
        corporate_gain_share=corporate_gain_share,
        gain_dynamics_active=gain_dynamics_active,
    )
    baseline_payroll = calc_pr(adjusted, law.baseline_payroll)
    baseline_wages1 = adjusted["wages_base1"] + adjusted["tips1"] + adjusted["ot1"]
    baseline_wages2 = adjusted["wages_base2"] + adjusted["tips2"] + adjusted["ot2"]
    return income_adjustments.apply_employer_payroll_adjustment(
        adjusted,
        law.calculator.payroll,
        baseline_payroll["liab_fica_er1"],
        baseline_payroll["liab_fica_er2"],
        baseline_wages1,
        baseline_wages2,
    )


def make_frame_builder(
    indexes: pd.DataFrame,
    cluster_death_weight_cap: float,
    corporate_asset_exposure: Mapping[str, float] | None = None,
    corporate_gain_share: float = 0.0,
    gain_dynamics_active: bool = False,
):
    """Return the production frame builder the annual step's contract describes.

    The returned callable runs the annual frame steps under the law it is
    handed and, on the flows branch only, the two reform-side income
    adjustments. The wealth haircut and household behavior are applied by the
    annual step. The builder only adds columns, so every wealth-channel column
    the later steps read (the balance sheet, the capital flows, ages, filing
    status and weight) passes through unchanged.

    The three corporate arguments are the economy-leg inputs the corporate
    step closes over; the per-year factors are carried on the law
    (:class:`CorporateYearFactors`). ``gain_dynamics_active`` changes which
    code runs, so two scenarios that differ on it cannot share a builder. The
    switch fingerprint carries it.
    """
    index_years, cpi_levels = cpi_level_series(indexes)
    index_years = jnp.asarray(index_years)
    cpi_levels = jnp.asarray(cpi_levels)
    cap = jnp.asarray(float(cluster_death_weight_cap))
    floor = jnp.asarray(DONOR_CLUSTER_GROSS_ESTATE_FLOOR)
    # The widowhood shares are a fixed table, read once here.
    widowhood_shares = widowhood.read_widowhood_shares()

    def build(
        raw_arrays: Records,
        law: AnnualLaw,
        year: Array,
        *,
        apply_adjusters: bool,
    ) -> dict[str, Array]:
        valid = jnp.asarray(raw_arrays["valid"], dtype=bool)
        frame = steps_jax.build_world_frame(
            raw_arrays,
            year,
            law.frame_columns,
            index_years,
            cpi_levels,
            cap,
            floor,
            valid,
            widowhood_shares=widowhood_shares,
        )
        if not apply_adjusters:
            return frame
        return apply_reform_income_adjustments(
            frame,
            law,
            corporate_asset_exposure=corporate_asset_exposure,
            corporate_gain_share=corporate_gain_share,
            gain_dynamics_active=gain_dynamics_active,
        )

    return build
