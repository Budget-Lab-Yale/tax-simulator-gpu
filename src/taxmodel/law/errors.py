"""Errors raised by the law compiler.

These are separate from the config layer's errors so a caller can tell a
misspecified scenario ("this reform names a law folder that does not exist")
from a misspecified law ("this reform widened the rate schedule past the
compiled maximum"). Both descend from ConfigError.
"""

from ..config.errors import ConfigError


class LawError(ConfigError):
    """A tax law file cannot be parsed into a year-by-filing-status schedule."""


class LawLintError(LawError):
    """A law tree that parses but says something its author did not write.

    The checks are listed in `lint.py`.
    """
