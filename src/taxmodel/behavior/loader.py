"""Composition of the behavior modules into the function the year loop calls.

A behavior module is a function from household columns and prices to the
columns it changed. This module
orders them, checks each one's declaration against the frame, and composes
them into a single function.

Every family runs in every scenario, in the fixed order below. A scenario
without a response gives that family its neutral parameter values, at which
the module returns its columns unchanged. Two scenarios that differ only in
a parameter value share one compiled program.

Each module's manifest is checked before anything compiles. A module that
names a column the frame does not carry, or a marginal rate the run does not
compute, stops the run.

A module's parameter values are passed when the chain is called, as
per-scenario arrays alongside the law (``years/loop.py``).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp

# The two equivalent tax rates the gains applier exports and entity shifting
# uses to price retained corporate earnings. The neutral placeholder below
# writes them when no gains model runs.
from taxmodel.behavior.kg.applier import (
    EQUIVALENT_TAX_RATE_BASELINE_COLUMN,
    EQUIVALENT_TAX_RATE_SCENARIO_COLUMN,
)

Array = jnp.ndarray
Records = Mapping[str, Array]
Prices = Mapping[str, Array]

# The composed chain: the same household columns and prices, plus this
# scenario's parameter values, returning the whole frame with every module's
# changes written into it.
BehaviorChain = Callable[..., dict[str, Array]]

# The order the families run in. Every family on this list runs in every
# scenario. A family reading a column runs after the family that writes it:
#
#   kg_dynamics      writes the realized gains and the deferred-dollar tax
#                    rates the families below read
#   tips             relabels part of an earner's base wages as tips
#   overtime         relabels part of an earner's base wages as overtime
#   auto             changes deductible auto-loan interest
#   conversion       moves salary out of base wages into the stock of
#                    unrealized gain
#   entity_shifting  moves business income between corporate and pass-through
#                    form, priced off that stock
#   evasion          hides a share of reported income
#   wealth           hides a share of reported net worth, reading what evasion
#                    hid
#   charity          responds to the income tax rates after the moves above
#   estate           reads everything hidden above
#
# Tips and overtime come before conversion because they move the same
# dollars: tips and overtime take pay out of an earner's base wages, and
# conversion takes salary out of what is left.
FAMILY_ORDER: tuple[str, ...] = (
    "kg_dynamics",
    "tips",
    "overtime",
    "auto",
    "conversion",
    "entity_shifting",
    "evasion",
    "wealth",
    "charity",
    "estate",
)

# The module each family runs. A family is a behavioral margin, named in a
# scenario's behavior.yaml; the module is the function that implements it.
# The kg_dynamics slot is filled by the runner, not from this table (see
# ``chain_entries``).
FAMILY_MODULES: dict[str, str] = {
    "tips": "taxmodel.behavior.tips.fringe_low",
    "overtime": "taxmodel.behavior.overtime.cahuc_carcillo",
    "auto": "taxmodel.behavior.auto.hanson",
    "conversion": "taxmodel.behavior.conversion.sigma",
    "entity_shifting": "taxmodel.behavior.entity_shifting.pearce_prisinzano",
    "evasion": "taxmodel.behavior.evasion.debacker",
    "wealth": "taxmodel.behavior.wealth.avoidance",
    "charity": "taxmodel.behavior.charity.tax_price",
    "estate": "taxmodel.behavior.estate.avoidance",
}

# The gains applier, added by the loader when kg_dynamics binds. A scenario
# does not name it.
GAINS_APPLIER_MODULE = "taxmodel.behavior.kg.applier"

# The families a scenario's behavior.yaml can name as margins: everything
# above except the capital-gains machinery, which is bound in its own section.
RESPONSE_FAMILIES: tuple[str, ...] = tuple(
    family for family in FAMILY_ORDER if family in FAMILY_MODULES
)

# The manifest keys the loader checks. A manifest may carry more.
#
# ``reads`` and ``draws`` are checked against the columns available. A
# module's ``writes`` may name a column that does not exist yet: the gains
# applier writes the deemed death-gain columns and the two equivalent tax
# rates, which the frame does not carry, and entity shifting then reads those
# rates. ``compose`` adds each module's writes to the available columns as it
# goes.
#
# ``draws`` are frame columns: the random numbers are drawn once per record
# and carried on the frame (`frame/prepare.DRAW_COLUMNS`).
#
# ``optional_reads`` is not checked. It names columns one module leaves for
# another, with a default the reader uses when the writer did not run: the
# evasion module's three retained-reporting factors default to one, and the
# wealth module's two concealed shares default to zero. A module declaring an
# optional read supplies its own default.
_COLUMN_KEYS = ("reads", "draws")
_PRICE_KEYS = ("prices", "discrete_prices")

# The per-household random numbers a behavior module may read. The frame draws
# these once per record and carries them all year; `frame/prepare.py` names
# them and says what each one decides. These two are reserved for behavioral
# responses; the tips module uses them to decide which untipped workers start
# receiving tips.
MODULE_DRAW_COLUMNS: tuple[str, ...] = ("r.behavior1", "r.behavior2")


def run_fact(
    records: Records, params: Mapping[str, Any], name: str, module: str
) -> Array:
    """A value the run supplies, read from the frame.

    The current year is a record column, and a deduction limit set in law is a
    bound-law column the frame join puts there under the law's own name. The
    fallback to `params` is for a caller testing a module on a hand-built
    frame that carries neither. `module` names the module in the error
    message.
    """
    if name in records:
        return jnp.asarray(records[name])
    if name in params:
        return jnp.asarray(params[name])
    raise KeyError(
        f"the {module} module needs {name!r}, either as a record column or in params"
    )


def provision_applies(
    records: Records, params: Mapping[str, Any], limit_name: str, module: str
) -> Array:
    """Whether the deduction this module responds to is in force this year.

    Two conditions: the year has reached the response's first year, and the
    deduction limit set in law is nonzero for anyone in the population. If
    anyone's limit is nonzero, every affected household responds, whatever its
    own limit.
    """
    year = run_fact(records, params, "year", module)
    deduction_limit = run_fact(records, params, limit_name, module)
    return (year >= jnp.asarray(params["first_year"])) & jnp.any(
        deduction_limit != 0
    )


class ManifestError(ValueError):
    """A module's declaration does not match the run it was asked to join."""


def _names(manifest: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return tuple(manifest.get(key, ()) or ())


def manifest_problems(
    manifest: Mapping[str, Any],
    columns: Iterable[str] | None,
    prices: Iterable[str],
) -> list[str]:
    """Problems with one module's declaration, as sentences; empty if none.

    ``columns`` is every column available to this module (the frame's own,
    plus what the modules ahead of it in the chain wrote) and ``prices`` every
    marginal rate the run computes. The driver's dry run has no frame yet and
    passes ``columns=None`` to check the prices and the family alone.
    """

    known_prices = set(prices)
    problems: list[str] = []

    if columns is not None:
        known_columns = set(columns)
        for key in _COLUMN_KEYS:
            missing = [n for n in _names(manifest, key) if n not in known_columns]
            if missing:
                problems.append(
                    f"declares {key} of columns the frame does not carry: "
                    + ", ".join(sorted(missing))
                )
    for key in _PRICE_KEYS:
        missing = [n for n in _names(manifest, key) if n not in known_prices]
        if missing:
            problems.append(
                f"declares {key} the run does not compute: "
                + ", ".join(sorted(missing))
            )
    if not manifest.get("family"):
        problems.append("has no family, so the loader cannot place it in the order")
    return problems


def validate_manifest(
    manifest: Mapping[str, Any],
    columns: Iterable[str],
    prices: Iterable[str],
) -> None:
    """Stop the run if a module's declaration does not match it."""

    problems = manifest_problems(manifest, columns, prices)
    if problems:
        name = manifest.get("name", "<unnamed module>")
        raise ManifestError(
            f'behavior module "{name}" ' + "; and ".join(problems) + "."
        )


def order_modules(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Put the module entries into the fixed family order.

    Stable within a family, so two modules of the same family keep the order
    the caller gave them. A family not in the fixed order stops the run.
    """

    def rank(entry: Mapping[str, Any]) -> int:
        family = entry["manifest"].get("family")
        if family not in FAMILY_ORDER:
            raise ManifestError(
                f'behavior module "{entry["manifest"].get("name", "?")}" belongs '
                f'to the family "{family}", which is not one of the families '
                "the chain runs: " + ", ".join(FAMILY_ORDER)
            )
        return FAMILY_ORDER.index(family)

    return sorted(entries, key=rank)


def compose(
    entries: Sequence[Mapping[str, Any]],
    columns: Iterable[str],
    prices: Iterable[str],
) -> BehaviorChain:
    """Validate, order, and compose the behavior modules into one function.

    Each entry is ``{"function": ..., "manifest": ..., "params": ...}``: the
    module, its declaration, and any parameter values fixed when the chain was
    built. Only the gains applier's cell tables are fixed at build time; a
    response module's parameter values arrive when the chain is called, as
    ``parameters``.

    The returned function takes the household frame, both laws' prices and
    this scenario's parameter values, and returns the whole frame with every
    module's changes written in, each module seeing what the ones before it
    wrote.
    """

    ordered = order_modules(entries)
    # Each module is checked against the frame's columns plus everything the
    # modules ahead of it have written by the time it runs.
    available = set(columns)
    for entry in ordered:
        validate_manifest(entry["manifest"], available, prices)
        available.update(_names(entry["manifest"], "writes"))

    def chain(
        records: Records,
        prices_reform: Prices,
        prices_baseline: Prices,
        parameters: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Array]:
        frame = dict(records)
        # The random draws are frame columns. A hand-built test frame may
        # carry none, and a module that needs one then raises its own error.
        supplied_draws = {
            name: records[name] for name in MODULE_DRAW_COLUMNS if name in records
        }
        by_family = {} if parameters is None else parameters
        for entry in ordered:
            family = entry["manifest"].get("family")
            params = {
                **dict(entry.get("params", {})),
                **dict(by_family.get(family, {})),
            }
            changed = entry["function"](
                frame,
                prices_reform,
                prices_baseline,
                params,
                supplied_draws,
            )
            frame.update(changed)
        return frame

    return chain


def import_module_function(
    module_name: str,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """Import one behavior module: its function, its manifest, its defaults.

    A behavior module file holds one response function, named in its
    ``MANIFEST`` under ``entry_point``. The manifest declares what the module
    reads, writes, prices and draws. The defaults are the parameter values the
    module ships with.
    """

    try:
        imported = importlib.import_module(module_name)
    except ImportError as problem:
        raise ManifestError(
            f"the behavior module `{module_name}` could not be imported: "
            f"{problem}"
        ) from problem

    manifest = getattr(imported, "MANIFEST", None)
    if not isinstance(manifest, Mapping):
        raise ManifestError(
            f"`{module_name}` has no MANIFEST, so there is no way to know "
            "which columns and prices it uses or which family it belongs to"
        )

    entry_point = manifest.get("entry_point") or manifest.get("name")
    function = getattr(imported, str(entry_point), None)
    if function is None or not callable(function):
        raise ManifestError(
            f"`{module_name}` declares its function as `{entry_point}`, "
            "but no function by that name is defined in it"
        )

    # A module with no parameters of its own has no `DEFAULTS`.
    found = getattr(imported, "DEFAULTS", None)
    defaults: Mapping[str, Any] = found if isinstance(found, Mapping) else {}

    return function, manifest, defaults


def import_family(family: str) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """One family's module function, manifest and default parameters."""

    if family not in FAMILY_MODULES:
        raise ManifestError(
            f'"{family}" is not a behavioral margin this model has a module '
            "for. The margins are: " + ", ".join(RESPONSE_FAMILIES)
        )
    return import_module_function(FAMILY_MODULES[family])


def chain_entries() -> list[dict[str, Any]]:
    """The fixed chain, ready to compose: every family, in order.

    The kg_dynamics slot is filled by ``kg_applier_at_neutral``, which writes
    the gains applier's columns at their no-response values. A run that
    solves the gains model uses ``gains_applier_entry`` instead
    (``behavior/gains/builders.py``).

    Every other family's entry carries its module's defaults, which
    the scenario's parameter values are laid over when the chain is called.
    """

    entries: list[dict[str, Any]] = [
        {"function": kg_applier_at_neutral, "manifest": _PLACEHOLDER_APPLIER_MANIFEST}
    ]
    for family in RESPONSE_FAMILIES:
        function, manifest, defaults = import_family(family)
        entries.append(
            {"function": function, "manifest": manifest, "params": dict(defaults)}
        )
    return entries


_PLACEHOLDER_APPLIER_MANIFEST: dict[str, Any] = {
    "name": "kg_applier_at_neutral",
    "entry_point": "kg_applier_at_neutral",
    "family": "kg_dynamics",
    "reads": ("kg_lt",),
    "writes": (
        EQUIVALENT_TAX_RATE_BASELINE_COLUMN,
        EQUIVALENT_TAX_RATE_SCENARIO_COLUMN,
    ),
    "prices": (),
    "discrete_prices": (),
    "draws": (),
}


def plain_behavior_builder() -> Callable[..., dict[str, Array]]:
    """The behavior step for a scenario that does not solve the gains model.

    The step runs the fixed chain with the gains applier's slot filled by the
    neutral placeholder, and has the signature the annual step calls it with.
    The annual step passes a plan value and the year's capital-gains tables
    alongside the frame, the prices and this scenario's behavior parameters.
    No module here reads either, so both are dropped. A scenario that solves
    the gains model gets its step from ``behavior/gains/builders.py`` instead.

    The chain is composed inside the traced function, where the frame's
    column names are available to validate the manifests against; the imports
    and file reads have already happened on the host. Build the step once and
    share it between scenarios: the batch runner compares behavior steps by
    identity when deciding which scenarios can share a compiled program.
    """

    entries = chain_entries()

    def build(
        records: Records,
        gains_year: Any,
        prices_reform: Prices,
        prices_baseline: Prices,
        parameters: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Array]:
        del gains_year
        run = compose(entries, tuple(records), tuple(prices_reform))
        return run(records, prices_reform, prices_baseline, parameters)

    return build


def kg_applier_at_neutral(
    records: Records,
    prices_reform: Prices,
    prices_baseline: Prices,
    params: Mapping[str, Any],
    draws: Mapping[str, Array],
) -> dict[str, Array]:
    """The gains applier's slot for a scenario that runs no gains model.

    Writes the two equivalent tax rate columns as zeros. Realized gains are
    unchanged and no gain is deemed realized at death. Entity shifting reads a
    missing equivalent tax rate as zero, so the numbers are the same as if the
    columns were absent; the chain has the same shape whether or not the gains
    model runs.
    """

    del prices_reform, prices_baseline, params, draws
    template = jnp.zeros_like(jnp.asarray(records["kg_lt"], dtype=jnp.float64))
    return {
        EQUIVALENT_TAX_RATE_BASELINE_COLUMN: template,
        EQUIVALENT_TAX_RATE_SCENARIO_COLUMN: template,
    }


def gains_applier_entry() -> tuple[Any, Mapping[str, Any], dict[str, Any]]:
    """The gains applier's function, manifest and default parameters.

    The applier is composed at the head of the chain with the cell table the
    stock recurrence solved. It is imported once, on the host.
    """

    return import_module_function(GAINS_APPLIER_MODULE)
