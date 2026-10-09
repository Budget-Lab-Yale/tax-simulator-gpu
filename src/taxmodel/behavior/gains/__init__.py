"""Capital gains behavior: the settings and the pieces that run inside the
compiled year.

The solver that prices the selling choice is `behavior/kg/`. This package
reads the settings (`settings.py`) and builds what the year loop traces
(`builders.py`). The host-side solve between the two executions of the year
loop is `years/gains_solve.py`.
"""
