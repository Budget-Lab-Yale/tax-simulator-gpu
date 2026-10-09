"""Startup check of the jax and jaxlib versions (ARCHITECTURE.md D1).

The model runs only on verified versions of jax and jaxlib. One earlier
version miscompiled the tax calculation, so an unverified version stops
the run instead of producing wrong liability.
"""

import jax
import jaxlib

# The verified versions. Any other version stops the run.
PINNED = {
    "jax": "0.7.0",
    "jaxlib": "0.7.0",
}


class ToolchainError(RuntimeError):
    """Raised at startup when the loaded jax or jaxlib version is not the verified one."""


def found_versions() -> dict[str, str]:
    """The jax and jaxlib versions actually loaded."""
    return {"jax": jax.__version__, "jaxlib": jaxlib.__version__}


def toolchain_record() -> dict[str, object]:
    """The toolchain entry for the run manifest.

    Records the pinned versions, the loaded versions, and whether they agree.
    """
    found = found_versions()
    return {
        "pinned": dict(PINNED),
        "found": found,
        "matches_pins": all(found[pkg] == want for pkg, want in PINNED.items()),
    }


def check_toolchain() -> None:
    found = found_versions()
    for pkg, want in PINNED.items():
        got = found[pkg]
        if got != want:
            raise ToolchainError(
                f"{pkg} is {got}, verified stack pins {want}; "
                f"results on unverified toolchains are not trusted"
            )
