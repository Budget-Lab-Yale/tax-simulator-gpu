"""Overtime responses to a tax exemption for overtime pay."""

from taxmodel.behavior.overtime.cahuc_carcillo import (
    DEFAULTS,
    INTENSIVE_ELASTICITY,
    MANIFEST,
    THREE_YEAR_PHASE_IN,
    overtime_cahuc_carcillo,
    unfunded_overtime_dollars,
)

__all__ = [
    "DEFAULTS",
    "INTENSIVE_ELASTICITY",
    "MANIFEST",
    "THREE_YEAR_PHASE_IN",
    "overtime_cahuc_carcillo",
    "unfunded_overtime_dollars",
]
