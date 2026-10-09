"""File paths and key names for the capital-gains calibration files.

Every path and key used in this package is defined once here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

# Written by a person: the two declared targets, in economic units, each
# with one line naming the source of the number.
DEFAULT_TARGETS_FILE = Path("config/calibrations/kg/targets.yaml")

# Written by the calibrate command: the record of the last solve (targets,
# achieved moments, commit, date).
DEFAULT_DERIVED_INTERNALS_FILE = Path("config/calibrations/kg/derived_internals.yaml")

# Also written by the calibrate command: the two fitted numbers, in the file a
# run reads, each carrying the vintages, targets and commit it was fitted under.
DEFAULT_REALIZATION_FILE = Path("config/calibrations/kg/realization.yaml")

# The gains model's source. The staleness check asks git whether anything
# in this folder has changed since the internals were derived.
GAINS_MODEL_SOURCE_FOLDER = "src/taxmodel/behavior/kg/"

# Keys inside a behavior-leg configuration. The gains dynamics are a named
# section of their own, holding the declared targets and the derived file.
KG_DYNAMICS_SECTION = "kg_dynamics"
TARGETS_KEY = "targets"
DERIVED_INTERNALS_KEY = "derived_internals"
# The piece of `kg_dynamics` naming the file a run reads its two numbers from.
REALIZATION_KEY = "realization"

# The two declared targets, by name.
PERMANENT_TARGET = "permanent_realization_elasticity"
TRANSITORY_TARGET = "transitory_realization_elasticity"

# The two fitted parameters, by the names the solve reports them under.
REALIZATION_ELASTICITY = "realization_elasticity"
TIMEABLE_SHARE = "timeable_share"


def file_named_by_behavior(
    behavior: Mapping[str, object], key: str, default: Path
) -> Path:
    """The file a resolved behavior leg names under `kg_dynamics`, or the default.

    A leg may switch the gains machinery off (`kg_dynamics: none`), in which
    case there is no mapping to read a filename from.
    """

    section = behavior.get(KG_DYNAMICS_SECTION)
    if isinstance(section, dict):
        named = section.get(key)
        if named:
            return Path(str(named))
    return default
