"""Errors raised by the config layer.

They are separate types so that a caller can tell a misspecified scenario
from a well-specified one whose calibrations are out of date.
"""


class ConfigError(Exception):
    """A scenario is misspecified: unknown entry, missing provenance, bad layer."""


class StaleCalibrationError(ConfigError):
    """A calibrated value's inputs no longer match the run's.

    For example, a share measured off one Tax-Data vintage is wrong against
    another.
    """
