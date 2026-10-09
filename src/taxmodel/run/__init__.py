"""Running a runscript and writing the run directory.

`cli.py` is the terminal command. `plan.py` reads the runscript into a plan.
`scenarios.py` runs the plan's scenarios in fixed-size groups and hands each
finished group to `output_directory.py`, which writes the tables and the run
manifest.

- `cli.py`              — the terminal command and its operational flags
- `plan.py`             — reading a runscript, expanding stacks, checking the
                          scenario ids and behavior legs, naming the population
- `scenarios.py`        — the scenario loop and `execute_run`, the run's entry point
- `grouping.py`         — which scenarios share one compiled program, the group
                          size, the program cache, and the run's counters
- `group_size.py`       — the number of scenarios one compiled projection year carries
- `calibrate.py`        — the calibrate command's pricing pass under current law
- `output_directory.py` — the run directory: its name, layout, Parquet tables,
                          and run-level manifest
- `file_writer.py`      — the helper thread that writes finished tables to disk
- `compile_cache.py`    — the on-disk cache of compiled programs

The run directory layout is described in `output_directory.py`.
"""
