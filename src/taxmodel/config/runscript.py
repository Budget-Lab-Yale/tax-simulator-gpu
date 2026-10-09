"""The runscript: the list of scenarios a run produces.

A runscript names folders and files, never values. Each leg cell is either
the reserved word `default` or a path under that leg's `alternatives/`. The
schema is six columns, and any other column is an error. A value typed into
a cell would have no provenance, so there is no per-value column of any kind.

The `tax_law` cell may name either a reform folder or a stack file (a YAML
file listing provision folders in order). A directory is a reform and a
`.yaml` file is a stack. The optional `detail` column names a detail-spec
file saying which per-household columns this scenario's run writes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .errors import ConfigError
from .scenario import DEFAULT_LAYER, TAX_LAW_LEG, leg_path

COLUMNS = (
    "ID",
    "tax_law",
    "economy",
    "behavior",
    "years",
    "detail",
)

REQUIRED_COLUMNS = ("ID", "tax_law", "years")

# What a stack file's name ends in. A tax_law cell naming a directory is one
# reform; a cell naming a file with this extension is a stack of provisions.
STACK_FILE_SUFFIX = ".yaml"

# The row post-processing looks up by name.
BASELINE_ID = "baseline"


@dataclass(frozen=True)
class ScenarioRow:
    """One row of a runscript: its three leg pointers and the years it runs.

    `tax_law` is the cell as written, either `default`, a reform folder, or a
    stack file, and `classify_tax_law` says which of the three it is.
    `detail` names the detail-spec file this scenario asks for, and is empty
    in the usual case of a scenario that writes no per-household columns.
    """

    id: str
    tax_law: str
    economy: str
    behavior: str
    years: list[int]
    detail: str


class TaxLawKind(str, Enum):
    """What a runscript's tax_law cell turned out to name."""

    BASELINE = "baseline law"
    REFORM_FOLDER = "reform folder"
    STACK_FILE = "stack file"


@dataclass(frozen=True)
class TaxLawTarget:
    """A resolved tax_law cell: which of the three things it names, and where."""

    kind: TaxLawKind
    # The cell as the runscript wrote it. This is what the law layer resolver
    # takes and what the manifest calls the reform.
    cell: str
    # The folder or file the cell resolved to, or None for baseline law.
    path: Path | None

    @property
    def is_stack(self) -> bool:
        """Whether this row has to be expanded into a stack's scenarios first."""
        return self.kind is TaxLawKind.STACK_FILE


def validate_columns(columns: list[str], name: str) -> None:
    """Check that the columns are exactly the six known ones, reporting every problem at once."""
    bad = [f"{c} : unknown column" for c in columns if c not in COLUMNS]
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        bad.append(f"missing required column(s): {', '.join(missing)}")
    if not bad:
        return
    raise ConfigError(
        f"invalid runscript '{name}':\n  - "
        + "\n  - ".join(bad)
        + "\nA runscript names folders and files, not values. The columns are:\n  "
        + ", ".join(COLUMNS)
    )


def parse_year_spec(spec: str) -> list[int]:
    """Parse a runscript's years: one year, a `{start}:{end}` range, or a list of both."""
    text = str(spec).strip()
    if not text:
        raise ConfigError("empty year specification")

    years: set[int] = set()
    for token in text.split():
        if ":" in token:
            ends = token.split(":")
            if len(ends) != 2:
                raise ConfigError(
                    f"malformed year range '{token}' -- expected {{start}}:{{end}}"
                )
            try:
                start, end = int(ends[0]), int(ends[1])
            except ValueError:
                raise ConfigError(
                    f"malformed year range '{token}' -- expected {{start}}:{{end}}"
                ) from None
            years.update(range(start, end + 1))
        else:
            try:
                years.add(int(token))
            except ValueError:
                raise ConfigError(f"malformed year '{token}'") from None
    return sorted(years)


def read_runscript(path: Path) -> list[ScenarioRow]:
    """Read a runscript, filling in omitted leg columns."""
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        validate_columns(columns, path.stem)
        raw_rows = list(reader)

    rows: list[ScenarioRow] = []
    for raw in raw_rows:
        rows.append(
            ScenarioRow(
                id=(raw["ID"] or "").strip(),
                tax_law=(raw["tax_law"] or DEFAULT_LAYER).strip() or DEFAULT_LAYER,
                economy=(raw.get("economy") or DEFAULT_LAYER).strip() or DEFAULT_LAYER,
                behavior=(raw.get("behavior") or DEFAULT_LAYER).strip() or DEFAULT_LAYER,
                years=parse_year_spec(raw["years"]),
                detail=(raw.get("detail") or "").strip(),
            )
        )
    return rows


def classify_tax_law(cell: str, config_root: str | Path) -> TaxLawTarget:
    """Say whether a tax_law cell names baseline law, one reform, or a stack.

    The cell resolves under `tax_law/alternatives/` in the scenario
    configuration tree. A directory is one reform, and a `.yaml` file is a
    stack of provisions. A missing path or a file with another suffix is an
    error.
    """
    cell = cell.strip()
    if cell in ("", DEFAULT_LAYER):
        return TaxLawTarget(kind=TaxLawKind.BASELINE, cell=DEFAULT_LAYER, path=None)

    path = leg_path(Path(config_root), TAX_LAW_LEG, cell)
    if path.is_dir():
        return TaxLawTarget(kind=TaxLawKind.REFORM_FOLDER, cell=cell, path=path)
    if path.is_file() and path.suffix == STACK_FILE_SUFFIX:
        return TaxLawTarget(kind=TaxLawKind.STACK_FILE, cell=cell, path=path)
    if path.is_file():
        raise ConfigError(
            f"tax_law '{cell}' resolves to {path}, a file that is not a stack "
            f"file: a stack file's name ends in {STACK_FILE_SUFFIX}"
        )
    raise ConfigError(
        f"tax_law '{cell}' names neither a reform folder nor a stack file: "
        f"there is nothing at {path}"
    )


def classify_tax_law_cells(
    rows: list[ScenarioRow], config_root: str | Path
) -> dict[str, TaxLawTarget]:
    """Classify every row's tax_law cell, reporting every bad cell at once."""
    targets: dict[str, TaxLawTarget] = {}
    problems: list[str] = []
    for row in rows:
        try:
            targets[row.id] = classify_tax_law(row.tax_law, config_root)
        except ConfigError as error:
            problems.append(f"row '{row.id}': {error}")
    if problems:
        raise ConfigError("unresolvable tax_law cells:\n  - " + "\n  - ".join(problems))
    return targets
