"""Tax model package.

Two settings are made at package import, before anything else is imported.

Double precision is switched on. Single precision cannot carry a $1 change in
a $38m estate, which the estate marginal rate needs, and the ceiling() in the
pre-TCJA exemption phaseout turns a rounding error into a two-percentage-point
step of exemption. See ARCHITECTURE D13.

The C library's allocation thresholds are set on a Linux machine with no
graphics card, so that large freed blocks go back to the operating system.
This has to happen before JAX and its libraries take any memory, because the
C library keeps whatever it has already taken.
"""

from taxmodel.run.host_memory import set_the_c_library_allocation_thresholds

set_the_c_library_allocation_thresholds()

import jax

jax.config.update("jax_enable_x64", True)
