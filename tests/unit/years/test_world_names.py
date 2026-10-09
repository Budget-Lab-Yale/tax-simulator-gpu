"""The explicit frame-by-law result grid, and the test helpers that subtract its cells."""

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.years.world_names import Frame, Law, ResultGrid, ResultNotEvaluatedError
from tests.unit.years.grid_differences import (
    conventional_minus_baseline,
    conventional_minus_mechanical,
    first_order_minus_baseline,
    mechanical_ex_saving_minus_first_order,
    mechanical_minus_mechanical_ex_saving,
)

YEAR = 2026


def result(amounts):
    return {
        "liab_iit_net": jnp.array(amounts, dtype=jnp.float64),
        "liab_pr": jnp.array(amounts, dtype=jnp.float64),
    }


def complete_grid() -> ResultGrid:
    grid = ResultGrid()
    grid.put(Frame.RAW, Law.BASELINE, YEAR, result([100.0, 200.0]))
    grid.put(Frame.RAW, Law.REFORM, YEAR, result([110.0, 190.0]))
    grid.put(Frame.FLOWS, Law.REFORM, YEAR, result([120.0, 195.0]))
    grid.put(Frame.MECHANICAL, Law.REFORM, YEAR, result([118.0, 192.0]))
    grid.put(Frame.BEHAVIOR, Law.REFORM, YEAR, result([125.0, 185.0]))
    return grid


def test_absent_cells_are_an_explicit_error():
    grid = ResultGrid()
    grid.put(Frame.RAW, Law.BASELINE, YEAR, result([100.0]))

    # No world is named for the baseline law on the income-flow frame, so the
    # message falls back to naming the frame and the law.
    with pytest.raises(
        ResultNotEvaluatedError,
        match=r"flows frame under baseline law.*not evaluated",
    ):
        grid.get(Frame.FLOWS, Law.BASELINE, YEAR)


def test_a_year_cannot_be_overwritten_silently():
    grid = ResultGrid()
    grid.put(Frame.RAW, Law.BASELINE, YEAR, result([100.0]))

    with pytest.raises(ValueError, match="already exist"):
        grid.put(Frame.RAW, Law.BASELINE, YEAR, result([200.0]))


def test_the_five_differences_use_only_frame_and_law_cells():
    grid = complete_grid()

    np.testing.assert_allclose(
        first_order_minus_baseline(grid, YEAR)["liab_iit_net"], [10.0, -10.0]
    )
    np.testing.assert_allclose(
        mechanical_ex_saving_minus_first_order(grid, YEAR)["liab_iit_net"], [10.0, 5.0]
    )
    np.testing.assert_allclose(
        mechanical_minus_mechanical_ex_saving(grid, YEAR)["liab_iit_net"],
        [-2.0, -3.0],
    )
    np.testing.assert_allclose(
        conventional_minus_mechanical(grid, YEAR)["liab_iit_net"],
        [7.0, -7.0],
    )
    np.testing.assert_allclose(
        conventional_minus_baseline(grid, YEAR)["liab_iit_net"],
        [25.0, -15.0],
    )


def test_the_channel_differences_telescope_to_the_complete_difference():
    grid = complete_grid()
    pieces = (
        first_order_minus_baseline(grid, YEAR),
        mechanical_ex_saving_minus_first_order(grid, YEAR),
        mechanical_minus_mechanical_ex_saving(grid, YEAR),
        conventional_minus_mechanical(grid, YEAR),
    )
    complete = conventional_minus_baseline(grid, YEAR)

    for name in complete:
        summed = sum((piece[name] for piece in pieces), start=jnp.zeros(2))
        np.testing.assert_allclose(summed, complete[name])
