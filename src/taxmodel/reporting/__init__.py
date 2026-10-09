"""Reporting: the published tables and the sums they are formed from.

This is the last step of a run. Two modules run on the device, inside the
compiled year, because the sums they take need every household's columns and
those never leave the device: `device_sums.py` (the population totals, the
distribution tables, the marginal-rate sums, the tax owed at death) and the
JAX code it calls (`distribution_jax.py`, `estate_allocator_jax.py`,
`mtr_table.py`'s reduction). Everything else is host work on results already
in memory: `assemble.py` arranges one scenario's answers into its tables, and
the rest build the published files: revenue by year, the four tax-specific
totals files, receipts, the stacked table, the distribution table, the
marginal-rate table, the law table, the wealth diagnostics, and per-household
detail. `stacked.StackedTable.write_csv` writes the stacked table; every
other file is written by `run/output_directory.py`.

A result is identified by the household frame it was computed on and the law
applied to that frame, and every comparison is written as one (frame, law)
pair minus another. One-word names for the five worlds are used for file
names and column headers. They are defined in `years/world_names.py`: `WORLD_CELLS` maps
each name to its (frame, law) cell and `world_name` maps a cell back to its
name.
"""
