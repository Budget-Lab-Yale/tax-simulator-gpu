"""The run manifest: everything needed to read a past run back.

The manifest is one JSON file per run (ARCHITECTURE.md D20). It records the
law the run used, the upstream data vintages, the calibrated values in
force, the git commit with a flag for uncommitted changes, and the runtime
stack (Python, jax, jaxlib, numpy, platform, device).
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import toolchain
from .law_layers import ResolvedLaw
from .scenario import ResolvedLeg
from .staleness import StalenessReport


def git_state(repo_root: Path | None = None) -> dict[str, Any]:
    """The commit this ran on, and whether the tree had uncommitted changes."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        # None rather than False when we could not tell: "not a repository" and
        # "clean tree" are different claims about reproducibility.
        "dirty": None if status is None else bool(status),
        # `git status --porcelain` gives a two-letter state then the path.
        "dirty_files": [] if not status else sorted(
            line.split(maxsplit=1)[-1] for line in status.splitlines()
        ),
    }


def runtime_stack() -> dict[str, Any]:
    """Versions and hardware the numbers were produced on."""
    stack: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in ("jax", "jaxlib", "numpy", "pandas", "yaml"):
        try:
            module = __import__(package)
        except ImportError:
            stack[package] = None
            continue
        stack[package] = getattr(module, "__version__", None)

    try:
        import jax

        stack["jax_backend"] = jax.default_backend()
        stack["jax_devices"] = [str(d) for d in jax.devices()]
        stack["x64_enabled"] = bool(jax.config.jax_enable_x64)
    except Exception as exc:  # noqa: BLE001 - no accelerator on a login node is normal
        stack["jax_backend"] = f"unavailable ({exc.__class__.__name__})"
        stack["jax_devices"] = []
    return stack


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def leg_rows(resolved: ResolvedLeg | None) -> list[dict[str, Any]]:
    """One row per resolved value: what it is, and whether the scenario changed it."""
    if resolved is None:
        return []
    sources = {(o.channel, o.name): o.source for o in resolved.overrides}
    rows = []
    for channel, entries in resolved.values.items():
        for name, value in entries.items():
            entry = resolved.entries[channel][name]
            source = sources.get((channel, name))
            rows.append(
                {
                    "leg": resolved.leg,
                    "alternative": resolved.alternative,
                    "channel": channel,
                    "name": name,
                    "value": _jsonable(value),
                    "kind": entry.kind,
                    "role": resolved.roles[channel][name],
                    "locked": entry.locked,
                    "overridden": source is not None,
                    "source": source or "default",
                }
            )
    return rows


def build_manifest(
    scenario_id: str,
    economy: ResolvedLeg | None = None,
    behavior: ResolvedLeg | None = None,
    law: ResolvedLaw | None = None,
    years: list[int] | None = None,
    vintages: dict[str, str] | None = None,
    staleness: StalenessReport | None = None,
    repo_root: Path | None = None,
    code_version: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the record of one scenario run.

    `code_version` is the commit the run was planned under. Reading it means
    shelling out to git, so a run that writes many scenario manifests reads
    it once and passes it in; left out, it is read here.
    """
    if code_version is None:
        code_version = git_state(repo_root)
    return {
        "scenario_id": scenario_id,
        "written_at": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        "legs": {
            "tax_law": law.alternative if law else None,
            "economy": economy.alternative if economy else None,
            "behavior": behavior.alternative if behavior else None,
        },
        # Every law layer that went into this run's law, baseline first. One
        # reform folder in the ordinary case; for a stacked reform, every
        # provision folded in.
        "law_layers": list(law.layers) if law else [],
        "years": years or [],
        "interface_vintages": _jsonable(vintages or {}),
        "config": leg_rows(economy) + leg_rows(behavior),
        "law_replacements": [
            {
                "parameter": r.parameter,
                "subparameter": r.subparameter,
                # Which folder's file holds this block. In a stack that is the
                # individual provision, not the whole stack.
                "written_by": r.written_by,
                # Whether this block is the one in force. A later provision
                # replacing the same subparameter wins, and both replacements
                # are recorded, so a reader can see what was overridden and by
                # which folder.
                "in_force": law.written_by(r.parameter, r.subparameter)
                == r.written_by,
                "new_subparameter": r.new_subparameter,
                "dropped_indexation": r.dropped_indexation,
            }
            for r in (law.replacements if law else [])
        ],
        # Subparameters more than one provision wrote, and which folders wrote
        # them, in order. Empty for a single reform.
        "law_conflicts": (
            {label: list(folders) for label, folders in law.conflicts().items()}
            if law
            else {}
        ),
        "staleness": {
            "stops": list(staleness.stops) if staleness else [],
            "warnings": list(staleness.warnings) if staleness else [],
        },
        "code_version": code_version,
        "runtime": runtime_stack(),
        "toolchain": toolchain.toolchain_record(),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    """Write the manifest beside the run's output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    return path
