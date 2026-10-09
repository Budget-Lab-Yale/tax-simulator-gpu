"""One scenario's behavior parameters.

Every behavioral margin runs in every compiled program. Behavioral feedback is
controlled by parameter values: an elasticity can be set to a positive value
to include a response, or to zero to turn the channel off. This module turns a
resolved behavior leg into those values, stored in a nested dictionary keyed
by behavioral margin and parameter name, each value a single-element array.

The dictionary is put on the device once per scenario, stacked lane by lane
when a batch is assembled (``years/loop.py``), and read inside the traced year
loop by the composed chain (``behavior/loader.py``). Two scenarios that differ
only in a parameter value have the same array shapes and share one compiled
program.

A margin a scenario leaves out of its behavior.yaml gets its neutral values:
the module's defaults with the ``zero_response`` values from its
manifest laid over them. At those values the module returns its columns
unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from . import config as behavior_config
from . import loader

Array = jnp.ndarray

# The parameter block, margin by parameter.
BehaviorParameters = dict[str, dict[str, Any]]


def neutral_values(family: str) -> dict[str, float]:
    """The parameter values at which one margin's module computes no response.

    These are the module's defaults with its manifest's
    ``zero_response`` values laid over them. A module that declares no
    zero-response values cannot be run at neutral, so the missing declaration
    is an error.
    """

    _function, manifest, defaults = loader.import_family(family)
    supplied_by_the_run = behavior_config.RUN_SUPPLIED_PARAMS.get(family, ())
    values = {
        name: value
        for name, value in defaults.items()
        if name not in supplied_by_the_run
    }
    zeroing = manifest.get("zero_response")
    if zeroing is None:
        raise loader.ManifestError(
            f'behavior module "{manifest.get("name", family)}" does not '
            "declare the parameter values that make it compute no response, "
            f"so the `{family}` margin cannot run at neutral for a scenario "
            "that leaves it out."
        )
    values.update(
        {
            name: value
            for name, value in zeroing.items()
            if name not in supplied_by_the_run
        }
    )
    return values


def values_for_leg(leg: behavior_config.BehaviorLeg) -> dict[str, dict[str, float]]:
    """Every margin's parameter values for one scenario.

    Margins the behavior file names carry the values it states; the rest carry
    their neutral values.
    """

    values: dict[str, dict[str, float]] = {}
    for family in loader.RESPONSE_FAMILIES:
        stated = leg.margins.get(family)
        if stated is None:
            values[family] = neutral_values(family)
        else:
            values[family] = dict(stated)
    return values


def as_arrays(values: Mapping[str, Mapping[str, float]]) -> BehaviorParameters:
    """The parameter values as float64 single-element arrays.

    Whole numbers such as the first year a provision is in law and the length
    of a phase-in are float64 too, because they are compared against the
    traced calendar year, which is float64.
    """

    return {
        family: {
            name: jnp.asarray(value, dtype=jnp.float64)
            for name, value in parameters.items()
        }
        for family, parameters in values.items()
    }


def for_leg(leg: behavior_config.BehaviorLeg) -> BehaviorParameters:
    """One scenario's behavior parameters, ready for the device."""

    return as_arrays(values_for_leg(leg))


def all_neutral() -> BehaviorParameters:
    """Every margin at its neutral values, for a scenario with no behavior leg."""

    return for_leg(behavior_config.BehaviorLeg(name=behavior_config.NO_BEHAVIOR))
