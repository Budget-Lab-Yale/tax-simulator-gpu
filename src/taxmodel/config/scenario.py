"""The three-leg scenario configuration: economy and behavior legs.

A scenario is three pointers and a year range: the tax law (`tax_law`),
the economic assumptions (`economy`), and the behavioral assumptions
(`behavior`). Tax law has its own layering in `law_layers.py`. This file
handles the other two legs (`config/scenarios/README.md`).

A runscript names folders, never values. A value in a folder entry carries
its kind and provenance.

Every leg is two layers: a `default/` folder holding a complete
specification, and `alternatives/` folders holding only what they change.
The alternative wins. An override replaces a whole named entry, value and
kind and provenance together.

Economy files are grouped into channels, one file per channel. Each channel
declares a role, `state` or `transmission`, which the run manifest reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

# The reserved runscript cell naming a leg's complete layer. A folder of the
# same name under alternatives/ is an error.
DEFAULT_LAYER = "default"


def leg_or_none(cell: str) -> str | None:
    """An economy or behavior cell as the run functions want it: None for the
    default layer, which is asked for by naming no alternative at all."""
    return None if cell in ("", DEFAULT_LAYER) else cell


LEGS = ("economy", "behavior")

# Tax law has its own layering (`law_layers.py`). Its folders sit under the
# scenario root in the same shape, so `leg_path` locates them too.
TAX_LAW_LEG = "tax_law"

# The fields each kind of entry requires. A calibrated value records the
# conditions it was derived under and how to re-derive it; a sourced value
# records its citation; a judgment or structural choice records the reason.
ENTRY_SCHEMA: dict[str, tuple[str, ...]] = {
    "calibrated": (
        "value",
        "kind",
        "set",
        "target",
        "derived_under",
        "invalidated_by",
        "rederive",
    ),
    "sourced": ("value", "kind", "citation"),
    "judgment": ("value", "kind", "note"),
    "structural": ("value", "kind", "note"),
}

ROLES = ("state", "transmission")

# Whether a stale finding stops the run or only warns. `warn` is for locked
# entries, which cannot be overridden, such as the estate valuation bridge
# when it is run against older Tax-Data vintages.
ENFORCEMENTS = ("stop", "warn")


@dataclass(frozen=True)
class Entry:
    """One named value in one channel, with the provenance its kind owes."""

    channel: str
    name: str
    value: Any
    kind: str
    role: str
    locked: bool = False
    enforcement: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)
    # True for a file line like `sigma_n: 0.375`, a number with no kind and
    # no provenance. Bare entries are rejected.
    bare: bool = False

    @property
    def label(self) -> str:
        return f"{self.channel}.{self.name}"

    @property
    def derived_under(self) -> dict[str, Any]:
        """Data vintages the value was calibrated against."""
        return self.raw.get("derived_under") or {}

    @property
    def invalidated_by(self) -> list[str]:
        """Files whose contents the value depends on."""
        return list(self.raw.get("invalidated_by") or [])

    @property
    def invalidated_by_hashes(self) -> dict[str, str] | None:
        """The md5s those files had when the value was pinned."""
        return self.raw.get("invalidated_by_hashes")

    @property
    def rederive(self) -> str | None:
        return self.raw.get("rederive")


@dataclass(frozen=True)
class Override:
    """An entry the scenario's alternative replaced, and with what."""

    channel: str
    name: str
    default: Any
    value: Any
    source: str


@dataclass
class DefaultLayer:
    """A leg's complete specification: every channel, every entry."""

    leg: str
    entries: dict[str, dict[str, Entry]]
    roles: dict[str, str]
    base_dir: Path


@dataclass
class ResolvedLeg:
    """One leg resolved for one scenario: default layer, alternative on top."""

    leg: str
    alternative: str
    defaults: DefaultLayer
    values: dict[str, dict[str, Any]]
    roles: dict[str, dict[str, str]]
    overrides: list[Override]
    # The entry actually in force per channel and name — the default's, or the
    # alternative's where it restated one. What the manifest reports.
    entries: dict[str, dict[str, Entry]] = field(default_factory=dict)

    @property
    def overridden(self) -> set[str]:
        return {f"{o.channel}.{o.name}" for o in self.overrides}


def leg_path(root: Path, leg: str, name: str) -> Path:
    """Locate one layer of one leg.

    The default layer is `<root>/<leg>/default`; every other layer is
    `<root>/<leg>/alternatives/<name>`. The same shape serves the economy and
    behavior legs, the tax-law reform folders, and the stacked-reform files.
    """
    root = Path(root)
    if name == DEFAULT_LAYER:
        return root / leg / DEFAULT_LAYER
    return root / leg / "alternatives" / name


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} is not a mapping of named entries")
    return loaded


def _channel_files(folder: Path) -> list[Path]:
    """The channel files in a layer.

    `behavior.yaml` is the behavior leg's module stack, not a channel of
    values, and is read by the behavior loader instead.
    """
    return sorted(p for p in folder.glob("*.yaml") if p.name != "behavior.yaml")


def load_defaults(root: Path, leg: str, base_dir: Path | None = None) -> DefaultLayer:
    """Read and check a leg's default layer.

    `base_dir` is what the provenance paths (`invalidated_by`,
    `pointer_root`) are relative to — the repository root in a real run.
    """
    if leg not in LEGS:
        raise ConfigError(f"unknown leg '{leg}'; expected one of {', '.join(LEGS)}")
    root = Path(root)
    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    folder = leg_path(root, leg, DEFAULT_LAYER)
    if not folder.is_dir():
        raise ConfigError(
            f"no default layer found for the {leg} leg (expected {folder})"
        )

    shadow = root / leg / "alternatives" / DEFAULT_LAYER
    if shadow.is_dir():
        raise ConfigError(
            f"`{DEFAULT_LAYER}` is the reserved runscript cell naming the {leg} "
            f"leg's default layer, so {shadow} can never be selected. Rename it."
        )

    files = _channel_files(folder)
    if not files and leg != "behavior":
        raise ConfigError(f"no channel files found under {folder}")

    entries: dict[str, dict[str, Entry]] = {}
    roles: dict[str, str] = {}

    for path in files:
        channel = path.stem
        block = _read_yaml(path)
        meta = block.pop("_channel", None)

        if leg == "economy":
            channel_role = (meta or {}).get("role")
            if channel_role not in ROLES:
                raise ConfigError(
                    f"economy channel `{channel}` must declare "
                    f"_channel: {{role: state|transmission}}"
                )
            roles[channel] = channel_role
        elif meta is not None:
            raise ConfigError(
                f"behavior channel `{channel}` carries a _channel block; roles "
                f"apply to the economy leg only"
            )
        else:
            channel_role = "state"

        entries[channel] = {
            name: _build_entry(channel, name, body, channel_role)
            for name, body in block.items()
        }

    layer = DefaultLayer(leg=leg, entries=entries, roles=roles, base_dir=base_dir)
    validate(layer)
    return layer


def _build_entry(channel: str, name: str, body: Any, channel_role: str) -> Entry:
    """Turn one YAML block into an entry, taking the channel's role if it declares none."""
    if not isinstance(body, dict):
        return Entry(
            channel=channel,
            name=name,
            value=body,
            kind=None,
            role=channel_role,
            bare=True,
        )
    return Entry(
        channel=channel,
        name=name,
        value=body.get("value"),
        kind=body.get("kind"),
        role=body.get("role", channel_role),
        locked=bool(body.get("locked", False)),
        enforcement=body.get("enforcement", "stop"),
        raw=body,
    )


def entry_problems(entry: Entry, base_dir: Path) -> list[str]:
    """Everything wrong with one entry: unknown kind, provenance it owes, bad fields."""
    label = entry.label
    body = entry.raw
    problems: list[str] = []

    if entry.bare:
        return [
            (
                f"{label}: must be a mapping with at least `value` and `kind` "
                f"(got a bare value)"
            )
        ]

    if entry.kind not in ENTRY_SCHEMA:
        return [
            (
                f"{label}: kind must be one of {'/'.join(ENTRY_SCHEMA)} "
                f"(got {entry.kind or 'nothing'})"
            )
        ]

    missing = [f for f in ENTRY_SCHEMA[entry.kind] if f not in body]
    if missing:
        problems.append(f"{label}: kind `{entry.kind}` requires {', '.join(missing)}")

    if entry.role not in ROLES:
        problems.append(
            f"{label}: role must be state or transmission (got {entry.role})"
        )
    if "locked" in body and not isinstance(body["locked"], bool):
        problems.append(f"{label}: locked must be true/false")
    if entry.enforcement not in ENFORCEMENTS:
        problems.append(
            f"{label}: enforcement must be stop or warn (got {entry.enforcement})"
        )

    if entry.kind == "calibrated":
        # The staleness check hashes these files, so each must exist.
        for dep in entry.invalidated_by:
            if not (base_dir / dep).exists():
                problems.append(
                    f"{label}: invalidated_by names a file that does not exist ({dep})"
                )
        # A pointer entry names a file or folder, such as the wealth saving
        # profile table, so that the staleness check covers table-valued inputs.
        pointer_root = body.get("pointer_root")
        if pointer_root is not None:
            target = base_dir / pointer_root / str(entry.value)
            if not target.exists():
                problems.append(
                    f"{label}: pointer value `{entry.value}` does not exist under "
                    f"{pointer_root}"
                )

    return problems


def validate(layer: DefaultLayer) -> None:
    """Check every entry declares a known kind and carries that kind's fields.

    Every problem in the layer is reported at once.
    """
    problems: list[str] = []
    for channel_entries in layer.entries.values():
        for entry in channel_entries.values():
            problems.extend(entry_problems(entry, layer.base_dir))

    if problems:
        raise ConfigError(
            f"invalid {layer.leg} configuration:\n  - " + "\n  - ".join(problems)
        )


def resolve(
    root: Path,
    layer: DefaultLayer,
    alternative: str | None = None,
) -> ResolvedLeg:
    """Lay the alternative the runscript names over the default layer.

    An alternative may only name entries the default layer already has, so
    that a misspelled entry is an error.
    """
    leg = layer.leg
    alternative = str(alternative) if alternative else DEFAULT_LAYER

    values = {
        ch: {nm: e.value for nm, e in entries.items()}
        for ch, entries in layer.entries.items()
    }
    roles = {
        ch: {nm: e.role for nm, e in entries.items()}
        for ch, entries in layer.entries.items()
    }
    effective = {ch: dict(entries) for ch, entries in layer.entries.items()}
    overrides: list[Override] = []

    if alternative == DEFAULT_LAYER:
        return ResolvedLeg(leg, alternative, layer, values, roles, overrides, effective)

    alt_folder = leg_path(root, leg, alternative)
    if not alt_folder.is_dir():
        raise ConfigError(
            f"unknown {leg} alternative '{alternative}' -- no folder {alt_folder}"
        )
    source = f"{leg} alternative '{alternative}'"

    for path in _channel_files(alt_folder):
        channel = path.stem
        for name, body in _read_yaml(path).items():
            if not isinstance(body, dict):
                raise ConfigError(
                    f"{source}: {channel}.{name} must be a mapping supplying `value`"
                )
            if channel not in layer.entries:
                raise ConfigError(f"{source} names an unknown {leg} channel: {channel}")
            if name not in layer.entries[channel]:
                raise ConfigError(
                    f"{source} names an unknown {leg} entry: {channel}.{name}"
                )

            new_value = body.get("value")

            # A locked entry measures the data, for example how the data maps
            # to reported estates, so no scenario may override it.
            if layer.entries[channel][name].locked and new_value is not None:
                raise ConfigError(
                    f"{channel}.{name} is locked ({source} tried to override it). "
                    f"Locked entries are never scenario-overridable."
                )

            if new_value is None:
                raise ConfigError(
                    f"{source} override {channel}.{name} must supply a `value`"
                )

            overrides.append(
                Override(
                    channel=channel,
                    name=name,
                    default=layer.entries[channel][name].value,
                    value=new_value,
                    source=f"alternative:{alternative}",
                )
            )
            values[channel][name] = new_value

            # An alternative that restates the kind must supply that kind's
            # provenance in full. One that supplies only a value changes the
            # number, and the run records it as overridden.
            if "kind" in body:
                replacement = _build_entry(
                    channel, name, body, layer.roles.get(channel, "state")
                )
                problems = entry_problems(replacement, layer.base_dir)
                if problems:
                    raise ConfigError(
                        f"invalid {source}:\n  - " + "\n  - ".join(problems)
                    )
                effective[channel][name] = replacement
                roles[channel][name] = replacement.role

    return ResolvedLeg(leg, alternative, layer, values, roles, overrides, effective)


# ---------------------------------------------------------------------------
# Reading values during a run
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """The resolved legs a run reads from. Values are read at the point of use."""

    economy: ResolvedLeg | None = None

    def economy_param(self, channel: str, name: str) -> Any:
        resolved = self.economy
        if resolved is None:
            raise ConfigError(
                f"economy_param('{channel}', '{name}') was read, but no economy "
                "leg has been resolved"
            )
        if channel not in resolved.values:
            raise ConfigError(f"unknown economy channel: {channel}")
        if name not in resolved.values[channel]:
            raise ConfigError(f"unknown economy entry: {channel}.{name}")
        return resolved.values[channel][name]


def interface_vintages(economy: ResolvedLeg) -> dict[str, str]:
    """The upstream data vintages this run reads, from the interfaces channel.

    The staleness check compares these against the vintages each calibration
    was derived under.
    """
    entries = economy.values.get("interfaces")
    if entries is None:
        raise ConfigError("resolved economy leg has no interfaces channel")
    return {
        key[: -len("_vintage")]: str(value)
        for key, value in entries.items()
        if key.endswith("_vintage")
    }
