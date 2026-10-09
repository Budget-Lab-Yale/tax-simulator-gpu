# Performance

This document holds the model's speed and memory goals, the current
measurements, the settings that affect run time, and the rules for
performance work. The earlier work log is in git history.

## 1. Goals

1. **A ten-scenario, thirty-year, full-behavior run in 3 minutes on a
   GPU, compile included.** Four minutes is acceptable. The time is the
   wall clock from launch to the last file written, with the compile
   cache empty.
2. **One more scenario costs seconds**, whether added to a run or run
   alone.
3. **The model runs on an ordinary 16 GB laptop**, not necessarily in
   3 minutes, but fast enough to work with.
4. **Published numbers never change for speed.** A change must show that
   its outputs are within the measured run-to-run variation of the
   unchanged code. Approximations such as subsampling or lower precision
   are adopted only as a labeled opt-in.
5. **Simple speed is preferred to parallel coordination.** Where there is
   a choice, take the design that is fast on one core or one device.

## 2. What the model does, computationally

One simulated year is four kinds of work:

1. **Taxing each household** under a handful of laws. No household affects
   another. The arithmetic takes milliseconds on a GPU and seconds on a
   CPU.
2. **The derivative of that calculation**, for marginal tax rates. It
   costs two to three times a plain calculation, and is the most expensive
   part to compile.
3. **Sums** to totals and tables.
4. **A small amount of state carried to the next year**: the wealth
   shortfall and the capital-gains stock. This is why years run in order.

On a GPU the arithmetic is fast and the time goes to compiling, host-side
Python, and moving data. On a laptop CPU the arithmetic itself takes the
time, and memory is the binding limit.

## 3. Where a run's time goes

The compiled programs are the projection year and the small programs
that form the population totals, the tax at death spread to heirs, and the
distribution and marginal-rate sums (ARCHITECTURE.md D18). Everything else
is plain Python.

```
plain Python    read the runscript, compile the laws, read the population,
                stack all scenarios' laws once
compile         the projection year (the largest single cost of a cold run),
                then the totals and report programs
execution 1     plain Python loop over the years calling the projection year.
                A scenario with the capital-gains model keeps only the
                exported gains rates from this pass; one without it forms
                all its tables here
host solve      the capital-gains selling decision, between the executions
execution 2     the same loop again with the solved gains tables, forming
                every reported number (capital-gains scenarios only)
plain Python    arrange the answers into tables; files are written on a
                helper thread
```

The latest whole-run measurements, on one H200 with the ten-scenario,
thirty-year runscript. All of them were taken before the capital-gains
model was rebuilt in September 2026 and before the distribution table was
redesigned, so they should be measured again before they are relied on.

| Measurement | Result | Slurm job |
| --- | --- | --- |
| Whole run, compile cache empty | 209 to 221 s | 25476909 |
| Whole run, compile cache warm | about 100 s | 25479852, 25480171 |
| Projection year before it runs, cache empty | 17.5 s tracing plus 82 s compiling | 25479852, 25480171 |
| Projection year before it runs, cache warm | 24 s: 17.5 tracing, 3.6 loading from the cache, 3.5 building sample inputs | 25479852, 25480171 |
| One pass of thirty years, beyond compile | 18.8 s | 25168544 |

Tracing is the Python step that records the year's array operations before
XLA compiles them or loads them from the cache, so every run pays it. A
profile found about 157,000 array operations in one simulated year, at
about 90 microseconds each, with no single hotspot; the model's own Python
is under one percent (jobs 25551884, 25552321). The tax calculation is
traced 26 times per year: once per world, eight times to price the tax on
gains deemed realized at death, and the rest for marginal rates. Only
reducing the number of operations reduces tracing time.

## 4. Settled findings

Each row settles a question by measurement. The job logs hold the detail.

| Finding | Number | Slurm job or commit |
| --- | --- | --- |
| Capping the XLA autotuner does not shorten the compile | about 124 s cold at the default level, level 1 and level 0 | 21336460 |
| Stacking all scenarios' laws once per run, not slicing them year by year in Python | a pass from 88.5 s to 36.1 s | commit 4c1e338 |
| Moving the report sums out of the year program | year compile from 131 s to 100 s; the marginal-rate table alone compiles in about 1 s | commit e020ab9 |
| A shared grouped-sum program for all report tables does not help; the apparent saving came from deleting the gradients | 136.6 s with the gradients kept, against a 129.7 s control | 21399979 |
| Removing the per-year synchronization waits does not change the time | 94.6 s against 94.9 s | not recorded |
| The number of calls to the device does not matter at this scale | 420 calls cost the same as 30 | not recorded |
| One compiled program solving every scenario's gains decision at once | 46.6 s to 8.7 s; whole run 312 s to 270 s | 22018125 |
| Small programs compile fast at production size | the four batched gains-solve programs compile in 3.5 s together | 22019578 |
| Reading the population's year files on eight threads | 13.8 s to 9.9 s; four threads get nearly all of it | 22018968, 22019209 |
| Writing files on a helper thread, and Parquet for the large tables | about 33 s off writing | 21993118, 21993168 |
| Walking the registered discrete changes in a fixed order makes every compiled program identical between runs, so the cache can hit | all 495 programs byte-identical across two runs | 22018583 |
| The compile cache | whole run 229 s cold against 139 s warm; year compile 93.9 s to 22.0 s | 22021628 |
| Cutting each year out of the stacked laws in one compiled operation, and returning the totals as one array per world | a pass from 28.6 s to 18.8 s; whole run from about 215 s to 191 s | 25168544 |
| The first execution of a gains scenario forms no reported numbers | first execution from about 143 s to 127 s | 25476909 |
| Tracing on a background thread does not help, because tracing and the work it overlapped both hold Python's interpreter lock | 99.8 s against 100.5 s warm | 25482161, 25482160 |
| The compile cache on the test suite, sixteen workers | 24 minutes to under 3 | not recorded |
| A faster no-heir distribution sum on the GPU (kept) | 1.120 ms to 0.59 ms; two other small-table changes were reverted because their outputs moved beyond the run-to-run variation | 25024550, 25026393 |
| Two duplicated computations in the year program (an extra payroll calculation and the wealth channel run twice) | about 15 s of compile, not merged | not recorded |

GPU runs of identical code are not bit-for-bit reproducible. Receipts
differ by about one part in 10^8; counts of households whose liability
sits exactly on zero differ by up to about 5 percent. Every output comparison is judged against that measured variation.

## 5. Settings that affect run time

**GPU.** `jobs/runscript_run.sbatch`, which `taxmodel-run --slurm`
submits, asks for one H200 on the `gpu_h200` partition with 8 CPUs and
240 GB of host memory. The `gpu` partition holds L40S cards. On B200 nodes
the pinned JAX build can crash at compile time; disabling XLA
command-buffer capture (`--xla_gpu_enable_command_buffer=`) avoids the
crash at some cost in speed.

**Scenarios per compiled program.** Scenarios whose laws switch on the
same code paths share one compiled year and run side by side in it. The
compile time does not grow with the number of scenarios, and sixteen run
in under twice the time of one (`years/loop.py`). The number that fit on
the device is computed from measured memory per scenario
(`run/group_size.py`); a capital-gains scenario costs about 3.3 times a
plain one. Grouping many scenarios into one run is much faster than
running them one at a time.

**Compile cache.** On by default, in `~/.cache/tax-simulator-gpu/xla` (or
under `XDG_CACHE_HOME`). Set `TAXMODEL_COMPILE_CACHE` to a path to use
another directory, or to `off` to run without it. The cache is keyed on
the program text, the compiler version and the device, so changed code
always recompiles. Programs that compiled in under 2 seconds are not
stored. The 2 GB size limit is enforced only when the `filelock` package
is installed. Deleting the directory is always safe
(`run/compile_cache.py`).

**Laptop.** With no graphics card the population is read one year at a
time instead of all at once, and the scenario count per program is set
for a 12 GB host memory budget (`years/device.py`). Section 6 covers host
memory.

**Memory diagnostics.** `TAXMODEL_MEMORY_LEDGER=on` prints memory at each
phase of a run; `TAXMODEL_MEMORY_TRIM_EVERY_YEAR=on` asks the C library to
return freed memory after every year; `TAXMODEL_MEMORY_PYTHON_HEAP=on`
traces Python's own allocations (`run/host_memory.py`).

## 6. Host memory on a machine with no graphics card

The peak host memory of a projection run has been measured on Bouchet
with no graphics card.

- The largest holder was the C library, keeping the freed temporaries of
  every projection year. The run now asks it at startup to serve blocks of
  128 KB and over with `mmap`, which are returned when freed
  (`run/host_memory.set_the_c_library_allocation_thresholds`, called from
  `taxmodel/__init__.py`). On Linux with no graphics card that took the
  thirty-year two-scenario run from 26.0 GB to 13.5 GB of peak resident
  memory, with the same numbers and about a tenth more time (jobs 24852776
  and 24852778, the second under a hard 16 GB cap).
- The call has to come before JAX is imported. Made after, the same run
  peaks at 15.1 GB (job 24848951), because the C library keeps the
  thresholds it was using for memory it has already taken.
- Windows and macOS have different allocators, and neither has been
  measured.

### Measuring it on a laptop

`config/runscripts/tests/laptop_gains_30yr.csv` is the runscript to use:
the baseline and one capital-gains reform, 2026 through 2055, which is
the largest thing a laptop is asked to do. Run it with the ledger on:

```
set TAXMODEL_MEMORY_LEDGER=on
taxmodel-run config/runscripts/tests/laptop_gains_30yr.csv --accept-stale-calibration
```

The ledger prints one line at each named phase of the run: resident
memory, the peak so far, the total size of the live JAX arrays, and
Arrow's pool. Send back the whole log. The three numbers that matter are
the highest peak any line reports, which phase it belongs to, and the
wall clock the run prints at the end.

On Windows the ledger reads the operating system's memory figures through
`ctypes`; on macOS it reports only the peak. That code has never run on
either. If a line shows `n/a` where a gigabyte figure belongs, or the run
stops in `run/host_memory.py`, report it.

The two one-scenario runscripts beside it (`laptop_plain_30yr.csv`,
`laptop_gains_only_30yr.csv`) split the peak between the baseline and the
gains scenario, and `laptop_gains_3yr.csv` is the same two scenarios over
three years, for a quick check that a machine runs the model at all. On
Bouchet the same measurement runs as a Slurm job:
`jobs/run_with_memory_sampling.sbatch` for one run's peak, and
`jobs/memory_decomposition.sbatch` to break a peak into holders.

## 7. Rules for performance work

1. One change per worktree.
2. Every change is measured on the ten-scenario, thirty-year H200 run
   against a same-node control, and every published file is compared
   with `jobs/compare_run_outputs.py` against the measured run-to-run
   variation (section 4).
3. The full test suite runs green in a Slurm job before any merge.
4. A speed claim cites a Slurm job ID, or says it is an estimate.
5. A settled measurement gets a row in section 4.
