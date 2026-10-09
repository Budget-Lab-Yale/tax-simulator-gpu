"""The wiring between a priced run and the calibration's baseline economy.

What is checked here is the seam, not the economics: that the cell rates an
execution exports come back out as the rates that went in, on both legs of
the selling choice, and that the economy built from them is a baseline one,
reform law equal to current law to the last bit. The moment builder refuses
to proceed otherwise.
The measurement itself is exercised on a toy economy in test_kg_moments.py.

The whole thing on the real population is a Slurm job: jobs/kg_calibrate.sbatch.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# tests/unit is what pytest puts on the path, so the gains model's own test
# population is a plain import from a sibling package.
from behavior.kg.test_assemble import households
from years.test_gains_solve import _CompiledLawStub

from taxmodel.behavior.gains import builders, settings
from taxmodel.behavior.kg import prepass
from taxmodel.calibration import kg_moments, targets
from taxmodel.law.compile import FILING_STATUSES
from taxmodel.years import gains_solve

# The window the calibration runs on: the year it opens plus the thirty
# after it.
YEARS = tuple(range(2025, 2025 + targets.CALIBRATION_WINDOW_YEARS + 1))

# A plausible cell rate on a realized dollar of long-term gain under current
# law: the 20 percent top rate plus the 3.8 percent investment income tax.
CURRENT_LAW_RATE_ON_GAINS = 0.238


def current_law_stub() -> _CompiledLawStub:
    """Current law as the gains model reads it: everything steps up at death."""
    n_years = len(YEARS)
    everyone = np.zeros((n_years, len(FILING_STATUSES), 1))
    return _CompiledLawStub(
        {
            "pref.kg_death_regime_equities": everyone,
        }
    )


def rate_exports_at(rate: float) -> dict[str, jnp.ndarray]:
    """What an execution would export if every cell measured the same rate.

    The compiled program hands back weighted sums rather than averages, so
    these are built the way it builds them: the rate times the dollars, and
    the dollars, for each age cell of each year.
    """
    records = households()
    n_households = records["weight"].shape[0]
    weights = prepass.cell_weights_from_records(records)
    sums = prepass.marginal_rate_cell_sums(weights, jnp.full(n_households, rate))

    def every_year(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([value] * len(YEARS))

    return {
        name: every_year(value)
        for name, value in sums.named(builders.BASELINE_ON_MECHANICAL_LEG).items()
    }


def current_law_prices(exports) -> dict[str, jnp.ndarray]:
    """The three cell rate matrices, from what an execution exported."""
    return gains_solve.current_law_cell_prices(exports)


def test_the_cell_rate_comes_back_as_the_rate_that_went_in():
    """Dividing the exported sums returns the rate the households faced."""

    prices = current_law_prices(rate_exports_at(CURRENT_LAW_RATE_ON_GAINS))

    # [cell, year], on the model's own grid.
    assert prices["baseline"].shape == (prepass.KG_N_AGES, len(YEARS))
    holds_something = np.asarray(
        prepass.marginal_rate_cell_sums(
            prepass.cell_weights_from_records(households()),
            jnp.full(households()["weight"].shape[0], CURRENT_LAW_RATE_ON_GAINS),
        ).holdings
    )
    measured = np.asarray(prices["baseline"])[:, 0]
    np.testing.assert_allclose(
        measured[holds_something > 0.0], CURRENT_LAW_RATE_ON_GAINS
    )
    # Both legs of the selling choice read the same array.
    assert prices["reform"] is prices["baseline"]
    assert prices["law_only"] is prices["baseline"]


def test_the_economy_it_builds_is_one_where_reform_law_is_current_law(macro_root):
    """The two shocks are measured against current law, so nothing else is in it.

    The moment builder refuses an economy whose reform rate differs from its
    baseline rate anywhere, because a reform already sitting in the economy
    would measure the calibration against the wrong counterfactual. Building
    both legs from one set of cell prices is what makes that hold exactly.
    """

    group = gains_solve.prepare_gains_solve_group(
        [households() for _ in YEARS],
        YEARS,
        scenario_ids=["current_law"],
        compiled_laws=[current_law_stub()],
        behavior_legs=[None],
        macro_root=macro_root,
        corporate_asset_exposures=[None],
        equity_markdowns=[None],
        # The fitted numbers are what the calibrate command is solving for,
        # so the group is built on the placeholder, as the command does.
        calibrations=[settings.placeholder_calibration()],
    )
    baseline_inputs = gains_solve.schedule_inputs_for_scenario(
        group, 0, current_law_prices(rate_exports_at(CURRENT_LAW_RATE_ON_GAINS))
    )

    scenarios = kg_moments.moment_scenarios_from_baseline(
        baseline_inputs, gains_solve.population_cells(group)
    )

    # The permanent leg is current law plus five points, from the year the
    # rise takes effect and not before.
    rise = np.asarray(
        scenarios.permanent_shock.marginal_rate_reform
        - scenarios.permanent_shock.marginal_rate_baseline
    )
    assert rise[:, kg_moments.EFFECTIVE_YEAR_OFFSET - 1].max() == 0.0
    np.testing.assert_allclose(
        rise[:, kg_moments.EFFECTIVE_YEAR_OFFSET], targets.PERMANENT_SHOCK_SIZE
    )
