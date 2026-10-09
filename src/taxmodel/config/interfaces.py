"""Disk locations of data published by the other models in the family.

Macro-Projections publishes economic projections and Tax-Data publishes the
microdata. A run reads one published copy of each, at

    <root>/<type>/<Interface>/v<version>/<vintage>/<scenario id>/

where

- the root is read from `config/output_roots.yaml`, where `production` is the
  shared project space holding released data and `local` is scratch;
- the type is the pin file's `type` field, `model_data` for every interface
  the model reads today;
- the version is pinned in `config/interfaces/interface_versions.yaml`. It
  says which output schema the published data follows, so it is fixed by the
  code rather than by the scenario;
- the vintage (a timestamp naming one publication) and the scenario id (which
  of that publication's scenarios) are scenario configuration. They come from
  the economy leg's `interfaces.yaml`, and an economy alternative can override
  them in its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

DEFAULT_CONFIG_ROOT = Path("config/scenarios")
DEFAULT_OUTPUT_ROOTS = Path("config/output_roots.yaml")
DEFAULT_INTERFACE_VERSIONS = Path("config/interfaces/interface_versions.yaml")


def interface_key(interface: str) -> str:
    """The economy-configuration name for an interface: `Macro-Projections` reads
    `macro_projections_vintage` and `macro_projections_id`."""
    return interface.lower().replace(" ", "_").replace("-", "_")


def _read_interfaces_channel(path: Path) -> dict[str, Any]:
    """The values in one `interfaces.yaml`: each entry is a mapping with a `value`."""
    if not path.is_file():
        return {}
    body = yaml.safe_load(path.read_text()) or {}
    return {
        name: entry["value"]
        for name, entry in body.items()
        # `_channel` declares the whole channel's role, not an interface pin.
        if isinstance(entry, dict) and "value" in entry
    }


def _economy_interface_entries(
    config_root: Path, economy: str | None
) -> dict[str, Any]:
    """The vintage and scenario id pinned for each interface: the economy default
    layer, with the named alternative's `interfaces.yaml` laid over it.

    Only the interfaces channel is read here. Resolving the whole economy leg
    through `scenario.resolve` would validate calibration provenance that is
    unrelated to which vintage to read.
    """
    root = Path(config_root) / "economy"
    entries = _read_interfaces_channel(root / "default" / "interfaces.yaml")
    if economy is None:
        return entries

    alternative = root / "alternatives" / economy / "interfaces.yaml"
    if not (root / "alternatives" / economy).is_dir():
        raise ConfigError(
            f"unknown economy alternative '{economy}' -- no folder "
            f"{root / 'alternatives' / economy}"
        )
    overrides = _read_interfaces_channel(alternative)
    unknown = set(overrides) - set(entries)
    if unknown:
        raise ConfigError(
            f"economy alternative '{economy}' pins interfaces the default layer "
            f"does not name: {', '.join(sorted(unknown))}"
        )
    return entries | overrides


def interface_root(
    interface: str,
    economy: str | None = None,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    output_roots: str | Path = DEFAULT_OUTPUT_ROOTS,
    interface_versions: str | Path = DEFAULT_INTERFACE_VERSIONS,
) -> Path:
    """The folder holding one upstream model's published data for this run.

    `interface` is the model's name as the pin file spells it, e.g.
    `Macro-Projections`. `economy` names an economy alternative folder whose
    `interfaces.yaml` may pin a different vintage or scenario id than the
    default layer does.

    Upstream data is always read from the shared production root. The local
    root holds only this model's own output.
    """
    vintage, scenario_id = interface_pin(
        interface, economy, config_root, interface_versions
    )
    pin = yaml.safe_load(Path(interface_versions).read_text())[interface]

    roots = yaml.safe_load(Path(output_roots).read_text())
    root = Path(roots["production"])
    return (
        root
        / str(pin["type"])
        / interface
        / f"v{pin['version']}"
        / str(vintage)
        / str(scenario_id)
    )


def interface_pin(
    interface: str,
    economy: str | None = None,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    interface_versions: str | Path = DEFAULT_INTERFACE_VERSIONS,
) -> tuple[str, str]:
    """The vintage and scenario id of the upstream publication this run reads.

    `interface_root` turns these into a path under the standard convention.
    The pair is also needed on its own for Tax-Data, whose converted copy is
    kept in a separate tree that the run names from the same pin.
    """
    versions = yaml.safe_load(Path(interface_versions).read_text())
    if interface not in versions:
        raise ConfigError(
            f"'{interface}' is not pinned in {interface_versions}; the run cannot "
            f"tell which version of it to read"
        )
    key = interface_key(interface)
    entries = _economy_interface_entries(Path(config_root), economy)
    vintage = entries.get(f"{key}_vintage")
    scenario_id = entries.get(f"{key}_id")
    if vintage is None or scenario_id is None:
        raise ConfigError(
            f"no vintage and scenario id for '{interface}': the economy leg's "
            f"interfaces.yaml does not supply {key}_vintage / {key}_id"
        )
    return str(vintage), str(scenario_id)


__all__ = ["interface_key", "interface_pin", "interface_root"]
