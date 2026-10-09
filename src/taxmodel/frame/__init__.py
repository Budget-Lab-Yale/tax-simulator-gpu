"""Turning Tax-Data microdata into the household arrays the calculators read.

Before a run starts, `bridge.py` converts a Tax-Data vintage once,
`prepare.py` decides who is in the run and in which years, and `population.py`
lays those records out as fixed year-by-slot arrays with their random draws
(`draws.py`). Inside the compiled year, `build_frame.py` builds one world's
household frame from a year of that population and its bound law, running the
per-year steps in `steps_jax.py` (the head-of-household recode, the law join,
the SALT-cap workaround, the Social Security step, the capital-gains basis
price ratio, estate mortality, net worth).

- `bridge.py`         — one Tax-Data vintage, converted once into the form the
                        model reads
- `prepare.py`        — the record universe: who is in the run, and in which years
- `population.py`     — the run's population as fixed year-by-slot NumPy arrays
- `draws.py`          — each record's random numbers, derived from the run seed
- `narrow_columns.py` — whole-number columns stored as 16-bit integers
- `widowhood.py`      — how likely an unmarried filer is to be a widow or widower
- `build_frame.py`    — the household frame for one world of one year
- `steps_jax.py`      — the annual frame-preparation steps, as JAX array operations
"""

from taxmodel.frame import population, prepare

__all__ = ["population", "prepare"]
