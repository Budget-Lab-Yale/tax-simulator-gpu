"""The two halves of the capital-gains model, joined at the schedule bundle.

The schedule solver's output fed into the stock recurrence, on the solver's
five-age, four-year toy economy (``test_schedule.toy_inputs``). The baseline
age cells the recurrence needs are built from the same inputs in
:func:`baseline_cells_from`. The solver returns age-by-year matrices and the
recurrence works year by age; ``stepper.schedule_by_year()`` turns one into the
other.
"""

from __future__ import annotations

import numpy as np
import pytest

from taxmodel.behavior.kg import (
    BaselineCells,
    bellman_age,
    cell_table_column,
    run_stock_recurrence,
    solve_schedule,
)
from taxmodel.behavior.kg.inputs import ScheduleInputs

from .conftest import (
    assert_stock_flow_identity,
    build_pool,
    priced_conversions,
)
from .test_conversion_inflow import PASS_THROUGH_LABOR_SHARE, SIGMA
from .test_schedule import toy_calibration, toy_inputs

# Claims that a number is exactly one are checked to floating-point noise,
# since the solver reaches them through a fixed-point sweep rather than by
# construction.
FLOATING_POINT_NOISE = 1e-12


def baseline_cells_from(inputs: ScheduleInputs) -> BaselineCells:
    """The stepper's baseline cells, taken from the solver's own inputs.

    The two halves describe the same cells from different sides.  The solver
    wants age by year and takes the mortality rate a dollar of gain faces
    ready-made; the stepper wants year by age and rebuilds that rate from the
    numerator, so the numerator here is the rate times what the cell holds.

    The stepper's ``charitable_bequest_share`` — the share of a decedent's
    gains that goes to charity, and so is never taxed — is the solver's
    ``charity_probability``, which is the same quantity read as the chance the
    whole estate is given away.
    """

    stock = inputs.baseline_gain_stock.T
    # The long grid's first rows are the stock's own ages.
    mortality = inputs.gain_weighted_mortality[: inputs.n_stock_ages].T
    return BaselineCells(
        stock=stock,
        realizations=inputs.baseline_realizations.T,
        turnover=inputs.baseline_turnover.T,
        mortality_headcount=mortality,
        mortality_gain_weighted=mortality * stock,
        charitable_bequest_share=inputs.charity_probability.T,
    )


def test_the_solved_schedule_reaches_the_cell_table_unchanged():
    """Every schedule number the cell table reports is the solver's own.

    Four of the cell table's columns are copied straight from the bundle, and
    this is the check that the copy reads the right field: baseline law's tax
    priced on a deferred dollar, and the three shares saying what happens to a
    decedent's gains.  A transposed or mis-mapped field would show up here rather than in
    a revenue number months later.

    The pace of selling and reform law's tax on a deferred dollar are not
    copied: the ledger reads them off the solved grids at the gain share each
    cell carries.  In the first year every cell holds exactly its baseline
    stock, so they agree with the bundle's baseline-share readings there, and
    that is checked.
    """

    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())
    result = run_stock_recurrence(
        baseline_cells_from(inputs),
        bundle,
        inputs.aging_operator,
        inputs.heir_operator,
    )

    for column, solved in (
        ("equivalent_tax_rate_baseline", bundle.tax_on_deferred_gain_baseline),
        ("forgiven_share", bundle.death_share_forgiven),
        ("routed_to_heirs_share", bundle.death_share_routed_to_heirs),
        ("realized_at_death_share", bundle.death_share_taxed_at_death),
    ):
        np.testing.assert_array_equal(
            np.asarray(cell_table_column(result.cell_tables, column)),
            np.asarray(solved).T,
            err_msg=f"the cell table's {column} is not the solver's own",
        )

    first_year = np.asarray(
        cell_table_column(result.cell_tables, "scenario_realization_rate")
    )[0]
    np.testing.assert_allclose(
        first_year, np.asarray(bundle.reform_turnover)[:, 0], rtol=0.0, atol=1e-15
    )
    scenario_tax = np.asarray(
        cell_table_column(result.cell_tables, "equivalent_tax_rate_scenario")
    )
    np.testing.assert_allclose(
        scenario_tax[0],
        np.asarray(bundle.tax_on_deferred_gain_reform)[:, 0],
        rtol=0.0,
        atol=1e-15,
    )
    gain_share = cell_table_column(result.cell_tables, "scenario_gain_share")
    for year in range(scenario_tax.shape[0]):
        np.testing.assert_allclose(
            scenario_tax[year],
            np.asarray(
                bellman_age.at_gain_share(
                    bundle.reform_tax_on_deferred_gain_by_gain_share[year],
                    gain_share[year],
                )
            ),
            rtol=0.0,
            atol=1e-15,
        )
    np.testing.assert_allclose(
        np.asarray(cell_table_column(result.cell_tables, "scenario_gain_share"))[0],
        np.clip(
            np.asarray(bundle.baseline_gain_stock)[:, 0]
            / np.asarray(bundle.baseline_asset_value)[:, 0],
            0.0,
            1.0,
        ),
        rtol=0.0,
        atol=1e-15,
    )

    # The fixture has to be a real reform for that to mean anything: the tax on
    # a deferred dollar must actually differ between the two laws.
    assert np.abs(np.asarray(bundle.tax_on_deferred_gain_reform - bundle.tax_on_deferred_gain_baseline)).max() > 1e-4


def test_water_conservation_through_the_solved_schedule():
    """Nothing appears or disappears that the accounts do not name.

    The realization rates and the death shares come from the solve: what the cells hold above baseline at the end of a year equals
    what they held above baseline at the start, less the extra sales, less what
    the decedents took with them, plus what the heirs received, plus the
    compensation converted at year end.
    """

    inputs = toy_inputs()
    baseline = baseline_cells_from(inputs)
    bundle = solve_schedule(inputs, toy_calibration())
    result = run_stock_recurrence(
        baseline,
        bundle,
        inputs.aging_operator,
        inputs.heir_operator,
        priced_conversions=priced_conversions(
            build_pool(marginal_rate_change=0.10),
            SIGMA,
            PASS_THROUGH_LABOR_SHARE,
        ),
    )

    assert_stock_flow_identity(baseline, result)

    # The reform has to be doing something for the identity to be worth
    # checking: the stock must have moved, and some pay must have been
    # relabeled as equity.
    leaving = np.asarray(result.stock_change_end_of_year)
    inflow = np.asarray(cell_table_column(result.cell_tables, "conversion_inflow"))
    assert np.abs(leaving).max() > 1.0
    assert inflow.sum() != 0.0


def test_the_stepper_refuses_a_bundle_of_the_wrong_size():
    """A bundle solved over a different number of years stops the recurrence."""

    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())
    baseline = baseline_cells_from(inputs)
    one_year_short = BaselineCells(
        **{
            name: getattr(baseline, name)[:-1]
            for name in baseline.__dataclass_fields__
        }
    )

    with pytest.raises(ValueError, match="years"):
        run_stock_recurrence(
            one_year_short, bundle, inputs.aging_operator, inputs.heir_operator
        )
