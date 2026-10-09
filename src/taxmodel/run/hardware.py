"""What kind of machine a run is on, answered without starting JAX.

Starting the JAX backend takes 45 to 110 seconds on a cluster node. Two
decisions are made before that, and both need to know only whether there is a
graphics card: whether to set the C library's allocation thresholds
(`run/host_memory.py`), and whether the population is read whole or a year at
a time (`years/device.population_stays_resident`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The device files the NVIDIA driver creates. With no driver loaded there is
# no card for JAX to use.
NVIDIA_DEVICE_FILES = ("/dev/nvidiactl", "/dev/nvidia0")


def graphics_card_is_present() -> bool | None:
    """Whether JAX will find a graphics card on this machine.

    Answers from the JAX_PLATFORMS setting when the run names its platform,
    and otherwise from the driver's device files on Linux. Returns None where
    neither applies, which leaves the caller to ask JAX itself.
    """
    named = [
        name.strip().lower()
        for name in os.environ.get("JAX_PLATFORMS", "").split(",")
        if name.strip()
    ]
    if named:
        return any(name != "cpu" for name in named)
    if sys.platform.startswith("linux"):
        return any(Path(name).exists() for name in NVIDIA_DEVICE_FILES)
    return None
