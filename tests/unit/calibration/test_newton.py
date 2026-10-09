"""Newton's method, on a problem whose answer we already know.

The real moments cost a model run each. Two inexpensive functions stand in
for them. The first moment depends only on the first parameter, so the 2x2
sensitivity matrix is triangular. In the gains model it is nearly
triangular: the long-run moment barely moves with the timeable share.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.calibration import newton

# The answer the synthetic problem is built to have.
TRUE_ELASTICITY = 1.6342
TRUE_TIMEABLE_SHARE = 0.2452


def separable_moments(parameters):
    """Two moments shaped like the gains model's, with a known solution.

    The first moment falls as the elasticity rises and ignores the share
    entirely. The second rises with the share, and the elasticity tilts it
    — because in the real model a bigger elasticity means the retimed
    dollars come out of a stock that is itself responding.

    Curvature is deliberate. On a linear problem Newton lands in one step
    and proves nothing about whether it converges.
    """

    elasticity, share = parameters[0], parameters[1]
    permanent = -0.30 * elasticity - 0.020 * elasticity**2
    transitory = 4.0 * share * (1.0 + 0.30 * elasticity) - 0.5 * share**2
    return jnp.stack([permanent, transitory])


TARGETS = tuple(
    float(value)
    for value in separable_moments(
        jnp.asarray([TRUE_ELASTICITY, TRUE_TIMEABLE_SHARE])
    )
)


def test_newton_finds_the_known_answer_in_a_handful_of_steps():
    result = newton.solve(
        separable_moments,
        targets=TARGETS,
        start=(1.0, 0.5),
        settings=newton.NewtonSettings(tolerance=1e-8, max_iterations=12),
        parameter_names=("realization_elasticity", "timeable_share"),
        moment_names=("permanent moment", "transitory moment"),
    )

    assert result.parameters[0] == pytest.approx(TRUE_ELASTICITY, abs=1e-6)
    assert result.parameters[1] == pytest.approx(TRUE_TIMEABLE_SHARE, abs=1e-6)
    assert result.worst_residual <= 1e-8
    # Newton on a mildly curved two-parameter problem is done in a few
    # steps; anything close to the cap would mean it is limping.
    assert result.iterations <= 5


def test_the_progress_messages_say_where_it_is_in_plain_words():
    said: list[str] = []
    newton.solve(
        separable_moments,
        targets=TARGETS,
        start=(1.0, 0.5),
        settings=newton.NewtonSettings(tolerance=1e-8),
        parameter_names=("realization_elasticity", "timeable_share"),
        moment_names=("permanent moment", "transitory moment"),
        report=said.append,
    )

    assert any("Starting from" in sentence for sentence in said)
    assert any("realization elasticity" in sentence for sentence in said)
    assert any("furthest a moment is from its target" in sentence for sentence in said)
    assert any(sentence.startswith("Converged after") for sentence in said)


def test_the_short_run_target_does_not_move_the_long_run_moment():
    """The toy problem's separability, read straight off the sensitivity matrix."""

    at_the_answer = jnp.asarray([TRUE_ELASTICITY, TRUE_TIMEABLE_SHARE])
    sensitivity = np.asarray(jax.jacfwd(separable_moments)(at_the_answer))

    # Row 0 is the long-run moment, column 1 is the timeable share.
    assert sensitivity[0, 1] == 0.0
    # The other three entries are real responses, so the two targets are
    # separately identified.
    assert abs(sensitivity[0, 0]) > 0.1
    assert abs(sensitivity[1, 1]) > 0.1
    assert abs(sensitivity[1, 0]) > 0.1


def test_a_parameter_nothing_responds_to_stops_the_solve_and_says_so():
    def blind_to_the_second_parameter(parameters):
        first = parameters[0]
        return jnp.stack([first, 2.0 * first])

    with pytest.raises(newton.NewtonFailure) as stopped:
        newton.solve(
            blind_to_the_second_parameter,
            targets=(1.0, 5.0),
            start=(0.0, 0.0),
            parameter_names=("realization_elasticity", "timeable_share"),
            moment_names=("permanent moment", "transitory moment"),
        )

    complaint = str(stopped.value)
    assert "timeable share" in complaint
    assert "does not move either moment" in complaint


def test_two_moments_that_say_the_same_thing_stop_the_solve_and_say_so():
    def the_same_moment_twice(parameters):
        total = parameters[0] + parameters[1]
        return jnp.stack([total, 2.0 * total])

    with pytest.raises(newton.NewtonFailure) as stopped:
        newton.solve(
            the_same_moment_twice,
            targets=(1.0, 3.0),
            start=(0.0, 0.0),
            parameter_names=("realization_elasticity", "timeable_share"),
            moment_names=("permanent moment", "transitory moment"),
        )

    assert "not separately identified" in str(stopped.value)


def test_a_step_outside_the_unit_interval_stops_with_an_economic_reason():
    """A share cannot be negative, and the message says that, not "nan"."""

    def wants_a_negative_share(parameters):
        return jnp.stack([parameters[0], 4.0 * parameters[1]])

    with pytest.raises(newton.NewtonFailure) as stopped:
        newton.solve(
            wants_a_negative_share,
            targets=(1.0, -8.0),
            start=(0.5, 0.5),
            parameter_names=("realization_elasticity", "timeable_share"),
            moment_names=("permanent moment", "transitory moment"),
            sensible_ranges=((0.0, 20.0), (0.0, 1.0)),
        )

    complaint = str(stopped.value)
    assert "timeable share" in complaint
    assert "outside the range 0 to 1" in complaint


def test_running_out_of_steps_says_how_far_it_got():
    with pytest.raises(newton.NewtonFailure) as stopped:
        newton.solve(
            separable_moments,
            targets=TARGETS,
            start=(1.0, 0.5),
            settings=newton.NewtonSettings(tolerance=1e-12, max_iterations=1),
        )

    assert "without getting within" in str(stopped.value)
