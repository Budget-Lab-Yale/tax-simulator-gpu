"""The detail spec: which per-household columns a scenario's run writes.

A run asks for per-household output by naming a detail-spec file in the
runscript's `detail` column. The spec is a YAML file with three keys:

    # config/detail/ctc_audit.yaml -- which per-household columns to write.
    variables: [liab_iit, credit_ctc, mtr_wages]   # calculated names only
    worlds: [baseline, first_order]
    years: 2030:2035                               # or the word: all

`variables` are names the model computes: the calculator's results and the
priced marginal rates. Input columns are not written, since a downstream
consumer has them from Tax-Data and links by household ID. `worlds` are the
five names for the five evaluated (frame, law) cells. `years` uses the
runscript's year syntax, or the word `all`. The year list filters what is
written and does not change what the compiled program computes.

Household ID and weight are always written and are not requested here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import ConfigError
from .runscript import parse_year_spec

# The folder a runscript's `detail` file name is resolved against.
DEFAULT_DETAIL_ROOT = Path("config/detail")

# The three keys a detail spec may carry, and the two it must.
KEYS = ("variables", "worlds", "years")
REQUIRED_KEYS = ("variables", "worlds")

# The word that asks for every year the scenario runs.
ALL_YEARS = "all"


@dataclass(frozen=True)
class DetailSpec:
    """One scenario's request for per-household columns.

    `years` is None when the spec said `all`, meaning every year in the
    runscript's years column.
    """

    path: Path
    variables: tuple[str, ...]
    worlds: tuple[str, ...]
    years: tuple[int, ...] | None

    def kept_years(self, run_years: list[int]) -> tuple[int, ...]:
        """The years to write, given the years the scenario actually runs."""
        if self.years is None:
            return tuple(int(year) for year in run_years)
        return tuple(year for year in self.years if year in set(run_years))

    def request(self, run_years: Sequence[int]):
        """This spec as the request the compiled run reads, with `all` years
        resolved against the years the scenario runs."""
        from taxmodel.reporting.detail import DetailRequest

        return DetailRequest.same_columns_from_each_world(
            columns=self.variables,
            worlds=self.worlds,
            years=self.kept_years(list(run_years)),
        )


def read_detail_spec(path: str | Path) -> DetailSpec:
    """Read a detail spec, reporting every problem in it at once.

    Whether the requested variables exist is checked later, when
    `DetailSpec.request` builds the request and `reporting/detail.py` compares
    the names against the columns each world offers.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"there is no detail spec at {path}")
    body = yaml.safe_load(path.read_text()) or {}
    if not isinstance(body, dict):
        raise ConfigError(
            f"the detail spec {path} does not read as a mapping of "
            f"{', '.join(KEYS)}"
        )

    problems: list[str] = []
    unknown_keys = sorted(set(body) - set(KEYS))
    if unknown_keys:
        problems.append(
            f"keys a detail spec does not have: {', '.join(unknown_keys)} "
            f"(the spec is exactly {', '.join(KEYS)})"
        )
    for key in REQUIRED_KEYS:
        if not body.get(key):
            problems.append(f"'{key}' is missing or empty, and a spec needs it")

    # Imported here rather than at the top because years/world_names.py loads
    # JAX, and reading a detail spec should not.
    from taxmodel.years.world_names import WORLD_NAMES

    variables = _as_names(body.get("variables"), "variables", problems)
    worlds = _as_names(body.get("worlds"), "worlds", problems)
    unknown_worlds = [name for name in worlds if name not in WORLD_NAMES]
    if unknown_worlds:
        problems.append(
            f"worlds that do not exist: {', '.join(unknown_worlds)} "
            f"(the five are {', '.join(WORLD_NAMES)})"
        )

    years = _read_years(body.get("years"), problems)

    if problems:
        raise ConfigError(
            f"invalid detail spec '{path}':\n  - " + "\n  - ".join(problems)
        )
    return DetailSpec(path=path, variables=variables, worlds=worlds, years=years)


def _as_names(value: object, key: str, problems: list[str]) -> tuple[str, ...]:
    """One name or a list of names, as a tuple, complaining about anything else."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(str(item).strip() for item in value)
    problems.append(f"'{key}' must be a name or a list of names, not {value!r}")
    return ()


def _read_years(value: object, problems: list[str]) -> tuple[int, ...] | None:
    """The requested years, or None for `all`."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() == ALL_YEARS:
        return None
    try:
        return tuple(parse_year_spec(text))
    except ConfigError as error:
        problems.append(f"'years' does not parse: {error}")
        return None
