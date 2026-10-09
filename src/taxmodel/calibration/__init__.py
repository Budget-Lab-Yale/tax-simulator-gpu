"""Calibration of the capital-gains model.

A person declares a target in :mod:`.targets`: how much realizations respond
to the tax rate on gains, in economic units, with one line naming the source
of the number. The `calibrate` command (:mod:`.cli`) solves for the two
internal parameters that reproduce the targets, using Newton's method
(:mod:`.newton`) with sensitivities of the moments in :mod:`.kg_moments`
measured by finite differences. The solved numbers are written, with the data
vintages they were fitted under, to the calibration file a run reads
(:mod:`.realization_file`), and the record of the solve to a second machine-written
file (:mod:`.derived`). Each run then reports whether the targets, the data
vintages, or the gains model's source code have changed since the fit
(:mod:`.staleness`).
"""

from .derived import (
    DerivationRecord,
    DerivedInternals,
    read_derived_internals,
    write_derived_internals,
)
from .newton import NewtonFailure, NewtonResult, NewtonSettings
from .staleness import check_for_run
from .targets import DeclaredTarget, KgDynamicsTargets, read_targets

__all__ = [
    "DeclaredTarget",
    "DerivationRecord",
    "DerivedInternals",
    "KgDynamicsTargets",
    "NewtonFailure",
    "NewtonResult",
    "NewtonSettings",
    "check_for_run",
    "read_derived_internals",
    "read_targets",
    "write_derived_internals",
]
