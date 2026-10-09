"""Newton's method for a two-target, two-parameter calibration.

The problem is two parameters p and a function m(p) that runs the model and
reports two moments; the solve finds the p at which m(p) equals two declared
targets. Each step evaluates the moments at the point and at a small step
either side in each parameter, forms the 2x2 matrix of sensitivities by
central differences, then solves the 2x2 system for the step. The model's
response is close to linear over the relevant range, so the solve converges
in a few steps.

This module is independent of the gains model. It takes a callable mapping
two parameters to two moments and two targets. The selling choice inside the
gains model is found by bisection, which has no derivative, so the
sensitivities are measured by finite differences rather than automatic
differentiation. The gains model's callable is built in
:mod:`taxmodel.calibration.kg_moments`.

In the gains model the 2x2 matrix is nearly triangular: the long-run moment
barely moves when the timeable share moves, because retiming shifts dollars
between neighbouring years and nets to zero over a permanent rate change. The
solve does not rely on this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# The step for the finite-difference sensitivities: a thousandth of the
# parameter, and never below the smallest step.
FINITE_DIFFERENCE_RELATIVE_STEP = 1e-3
FINITE_DIFFERENCE_SMALLEST_STEP = 1e-4

# A moment evaluation: two parameters in, two moments out.
MomentEvaluation = Callable[[jnp.ndarray], jnp.ndarray]


class NewtonFailure(Exception):
    """The solve stopped without reaching the targets, and says why."""


@dataclass(frozen=True)
class NewtonSettings:
    """Convergence tolerance, iteration cap, and singularity threshold."""

    # How close a moment has to be to its target before the solve stops, in
    # the moments' own units. The gains targets are elasticities around 1.
    tolerance: float = 1e-3
    # How many Newton steps the solve may take. Each step runs the model five
    # times. A two-parameter solve usually converges in three or four steps.
    max_iterations: int = 12
    # The smallest determinant of the 2x2 sensitivity matrix the solve will
    # invert, relative to the size of the matrix so that it does not depend
    # on the units of the moments.
    smallest_usable_determinant: float = 1e-8


@dataclass(frozen=True)
class NewtonResult:
    """Where the solve landed."""

    parameters: tuple[float, ...]
    moments: tuple[float, ...]
    residuals: tuple[float, ...]
    iterations: int

    @property
    def worst_residual(self) -> float:
        return max(abs(value) for value in self.residuals)


def _describe_point(
    names: Sequence[str],
    values: Sequence[float],
) -> str:
    return ", ".join(
        f"{name.replace('_', ' ')} {value:.6g}"
        for name, value in zip(names, values, strict=True)
    )


def _singularity_complaint(
    sensitivity: np.ndarray,
    parameter_names: Sequence[str],
    moment_names: Sequence[str],
) -> str:
    """One sentence saying why the sensitivity matrix cannot be inverted."""

    column_sizes = np.abs(sensitivity).max(axis=0)
    row_sizes = np.abs(sensitivity).max(axis=1)
    largest = float(np.abs(sensitivity).max())

    if largest == 0.0:
        return (
            "neither moment moved when either parameter moved. The moment "
            "evaluation is not seeing the parameters at all."
        )

    dead_parameters = [
        name
        for name, size in zip(parameter_names, column_sizes, strict=True)
        if size <= 1e-12 * largest
    ]
    if dead_parameters:
        return (
            "moving "
            + " or ".join(name.replace("_", " ") for name in dead_parameters)
            + " does not move either moment, so the targets cannot tell us "
            "what its value should be."
        )

    dead_moments = [
        name
        for name, size in zip(moment_names, row_sizes, strict=True)
        if size <= 1e-12 * largest
    ]
    if dead_moments:
        return (
            "the "
            + " and ".join(name.replace("_", " ") for name in dead_moments)
            + " moment does not respond to either parameter, so no choice of "
            "parameters can hit its target."
        )

    return (
        "the two moments respond to the parameters in almost exactly the same "
        "proportions, so the pair cannot be told apart: many different "
        "parameter combinations give the same two moments. The targets are "
        "not separately identified here."
    )


def solve(
    evaluate: MomentEvaluation,
    targets: Sequence[float],
    start: Sequence[float],
    *,
    settings: NewtonSettings | None = None,
    parameter_names: Sequence[str] = ("first parameter", "second parameter"),
    moment_names: Sequence[str] = ("first moment", "second moment"),
    sensible_ranges: Sequence[tuple[float, float]] | None = None,
    report: Callable[[str], None] | None = None,
) -> NewtonResult:
    """Find the parameters that make the model's moments equal the targets.

    ``evaluate`` takes an array of parameters and returns an array of
    moments of the same length. ``sensible_ranges``, if given, is the economically
    allowed range of each parameter (a share is in [0, 1]); a step outside
    it stops the solve.

    ``report`` receives one sentence per step. The default discards them;
    the calibrate command passes ``print``.
    """

    settings = settings or NewtonSettings()
    announce = report if report is not None else (lambda _sentence: None)

    target_vector = jnp.asarray(targets, dtype=float)
    point = jnp.asarray(start, dtype=float)
    if point.shape != target_vector.shape:
        raise NewtonFailure(
            f"there are {target_vector.size} targets but "
            f"{point.size} starting parameter values; they have to match."
        )

    def measure(parameters: jnp.ndarray) -> tuple[jnp.ndarray, np.ndarray]:
        """Return the moments at this point and the 2x2 sensitivity matrix.

        Each column of the matrix is a central difference: the moments at the
        parameter moved up and down by a step proportional to its size.
        """
        moments = jnp.asarray(evaluate(parameters))
        values = np.asarray(parameters, dtype=float)
        sensitivity = np.empty((moments.size, values.size), dtype=float)
        for column, value in enumerate(values):
            step = max(abs(value) * FINITE_DIFFERENCE_RELATIVE_STEP, FINITE_DIFFERENCE_SMALLEST_STEP)
            up = values.copy()
            down = values.copy()
            up[column] += step
            down[column] -= step
            higher = np.asarray(evaluate(jnp.asarray(up)), dtype=float)
            lower = np.asarray(evaluate(jnp.asarray(down)), dtype=float)
            sensitivity[:, column] = (higher - lower) / (2.0 * step)
        return moments, sensitivity

    announce(
        "Starting from "
        + _describe_point(parameter_names, [float(v) for v in point])
        + "; aiming for "
        + _describe_point(moment_names, [float(v) for v in target_vector])
        + f", within {settings.tolerance:g}."
    )

    moments, sensitivity = measure(point)

    for step in range(1, settings.max_iterations + 1):
        residuals = np.asarray(moments - target_vector, dtype=float)
        worst = float(np.max(np.abs(residuals)))

        announce(
            f"Step {step}: at "
            + _describe_point(parameter_names, [float(v) for v in point])
            + " the model gives "
            + _describe_point(moment_names, [float(v) for v in np.asarray(moments)])
            + f"; the furthest a moment is from its target is {worst:.6g}."
        )

        if worst <= settings.tolerance:
            announce(
                f"Converged after {step - 1} step"
                + ("" if step - 1 == 1 else "s")
                + "."
            )
            return NewtonResult(
                parameters=tuple(float(v) for v in point),
                moments=tuple(float(v) for v in np.asarray(moments)),
                residuals=tuple(float(v) for v in residuals),
                iterations=step - 1,
            )

        # The determinant is compared against the size of the matrix, so the
        # singularity check does not depend on the units of the moments.
        determinant = float(np.linalg.det(sensitivity))
        scale = float(
            np.linalg.norm(sensitivity[0]) * np.linalg.norm(sensitivity[1])
        )
        too_small = settings.smallest_usable_determinant * scale
        if scale == 0.0 or abs(determinant) <= too_small:
            raise NewtonFailure(
                f"The solve stopped at step {step} because the model's "
                "sensitivities cannot be inverted: "
                + _singularity_complaint(sensitivity, parameter_names, moment_names)
                + " The sensitivity matrix was "
                + repr(sensitivity.tolist())
                + "."
            )

        step_vector = np.linalg.solve(sensitivity, -residuals)
        proposed = np.asarray(point, dtype=float) + step_vector

        if sensible_ranges is not None:
            for name, value, (low, high) in zip(
                parameter_names, proposed, sensible_ranges, strict=True
            ):
                if not (low <= value <= high):
                    raise NewtonFailure(
                        f"The solve stopped at step {step}: it wanted to set "
                        f"{name.replace('_', ' ')} to {value:.6g}, which is "
                        f"outside the range {low:g} to {high:g} that value can "
                        "economically take. Either the targets are not "
                        "reachable by this model, or the starting point was "
                        "too far away for Newton to home in."
                    )

        point = jnp.asarray(proposed)
        moments, sensitivity = measure(point)

    final_residuals = np.asarray(moments - target_vector, dtype=float)
    raise NewtonFailure(
        f"The solve ran its full {settings.max_iterations} steps without "
        f"getting within {settings.tolerance:g} of the targets. It finished at "
        + _describe_point(parameter_names, [float(v) for v in point])
        + ", where the furthest a moment is from its target is "
        f"{float(np.max(np.abs(final_residuals))):.6g}. Either the model's "
        "response is not smooth enough for Newton over this range, or the "
        "targets are not reachable."
    )
