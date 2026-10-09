"""Scenario configuration and provenance (ARCHITECTURE.md D20).

A scenario is three pointers to folders: an economy, a behavior setting,
and a tax law. Every configured number declares what kind of number it is
and carries the matching provenance. A calibrated value is rejected once
the inputs it was derived under have changed. Every run writes a manifest
describing itself.

Modules:

- `scenario.py`   — the economy and behavior legs: two-layer resolution
                    and channel roles
- `law_layers.py` — the tax law leg: whole-subparameter replacement
- `law_stacks.py` — a stacked reform's provisions, and the scenarios it
                    expands into
- `runscript.py`  — the six-column runscript whitelist
- `detail.py`     — the detail spec: which per-household columns to write
- `staleness.py`  — derived_under (data vintages) / invalidated_by (files)
- `manifest.py`   — the per-run record
- `toolchain.py`  — the startup check of the verified jax versions (D1)
"""

from .detail import read_detail_spec
from .errors import ConfigError, StaleCalibrationError
from .law_layers import ResolvedLaw, resolve_law, resolve_law_stack
from .manifest import build_manifest, write_manifest
from .runscript import (
    TaxLawKind,
    classify_tax_law,
    classify_tax_law_cells,
    parse_year_spec,
    read_runscript,
)
from .scenario import (
    DEFAULT_LAYER,
    ResolvedLeg,
    ScenarioConfig,
    interface_vintages,
    leg_or_none,
    load_defaults,
    resolve,
)
from .staleness import StalenessReport, check_leg
from .toolchain import ToolchainError, check_toolchain

__all__ = [
    "DEFAULT_LAYER",
    "ConfigError",
    "ResolvedLaw",
    "ResolvedLeg",
    "ScenarioConfig",
    "StaleCalibrationError",
    "StalenessReport",
    "TaxLawKind",
    "ToolchainError",
    "build_manifest",
    "check_leg",
    "check_toolchain",
    "classify_tax_law",
    "classify_tax_law_cells",
    "interface_vintages",
    "leg_or_none",
    "load_defaults",
    "parse_year_spec",
    "read_detail_spec",
    "read_runscript",
    "resolve",
    "resolve_law",
    "resolve_law_stack",
    "write_manifest",
]
