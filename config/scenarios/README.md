# Scenario configuration: the three legs

A scenario is three pointers and a year range. Each pointer names a FOLDER:

| Leg | Answers | Lives in |
|---|---|---|
| `tax_law` | what is the policy | `tax_law/` |
| `economy` | how does the world work | `economy/` |
| `behavior` | how do agents respond | `behavior/` (module code in `src/behavior/`) |

Every leg has the same shape: a complete `default/` layer, plus sparse deltas
under `alternatives/`. A runscript cell is either the reserved word `default` or
a path under that leg's `alternatives/`. Nesting under `alternatives/` is
arbitrary and the folder names are human-chosen.

A runscript names files, never values. Its schema is exactly

```
ID, tax_law, economy, behavior, years, detail
```

and any other column is a parse error whose message names the replacement. The
reason is provenance: a number in a CSV cell has no citation, no record of what
it was derived under, and nothing to check it against. An entry in a folder has
all three.

### Retired columns

Every one of these hard-errors. There are no fallbacks, silent or otherwise.
`<alt>` below means `config/scenarios/economy/alternatives/<name>/`.

| Retired column | Replacement |
|---|---|
| `dep.{Interface}.vintage` | `<alt>interfaces.yaml` → `{interface}_vintage` (name lowercased, hyphens to underscores) |
| `dep.{Interface}.ID` | `<alt>interfaces.yaml` → `{interface}_id` |
| `s` | `<alt>wealth.yaml` → `financing_profile: flat:<s>` |
| `wealth_financing` | `<alt>wealth.yaml` → `financing_profile` (a profile folder name, or `none`) |
| `assumptions` | the `economy` and `behavior` columns |
| `assumption.{channel}.{name}` | an entry in an economy alternative, or — if only a behavior module reads it — the module's own file |
| `behavior` holding module paths | a behavior alternative folder whose `behavior.yaml` lists them |
| `excess_growth`, `excess_growth_start_year`, `excess_growth_all_rev` | nothing: the machinery was removed from the model |
| `corp_incidence_phasein` | nothing: it was never read |
| `first_year`, `last_year` | the `years` column, as `{start}:{end}` |
| any dotted `{leg}.{channel}.{name}` | the same entry in an alternative folder |

## Where a number goes

Three questions, in order.

1. **Does one behavior module read it, and nothing else?** Then it lives in that
   module's file, hardcoded, with its citation beside it. No exceptions — this is
   the rule that keeps modules drop-in. A variant is a separate file
   (`charity/50.R` beside `charity/100.R`), never a config override.
2. **Is it a number the capital-gains machinery calibrates, or a switch such a
   calibration was conditioned on?** Then it lives in `config/calibrations/`,
   written by the calibrator itself.
3. **Otherwise** it is economy-leg: a description of how the world works, or a
   mechanical rule for how a shock reaches records.

Numerical plumbing (tolerances, epsilons), structural bounds (age topcodes) and
operational toggles are none of the above and stay in code.

## The economy leg

`economy/default/` holds one file per channel: `interfaces.yaml` (the data
vintages), `corp.yaml`, `distribution.yaml`, `estate.yaml`, `wealth.yaml`. An
alternative supplies the same file names with only the entries it changes.

Override granularity is the whole named entry. An alternative's entry replaces
value, kind and provenance together; nothing merges within an entry, so a
scenario can never inherit half of a justification. `locked: true` entries — the
estate valuation bridge — refuse override outright.

Values are read at the point of use, through `economy_param('kg', 'eta')`, never
captured at source time, because they are scenario-scoped.

Each file declares a `_channel` role: `transmission` for how a tax change
reaches households (corporate incidence, the saving response), `state` for
how the world is. The role is recorded in the run manifest so a reader can
see which kind of assumption each value is; the static pass reads no economy
value at all, so nothing polices the roles at run time.

**Every entry declares a `kind`, and each kind owes different provenance:**

| kind | meaning | required fields | can go stale |
|---|---|---|---|
| `calibrated` | output of a procedure | `set`, `target`, `derived_under`, `invalidated_by`, `rederive` | yes |
| `sourced` | from a paper or convention | `citation` | no |
| `judgment` | someone chose it | `note` | no |
| `structural` | a model-form switch | `note` | no |

Only `calibrated` values can go stale, because only they have inputs.

## Staleness

A stale value hard-stops the run. The check runs once, at parse time, which is
also what covers the SLURM path. Two checks:

- the data vintages in `derived_under` no longer match the run's,
- a file in `invalidated_by` has changed content since the value was pinned
  (checked by hash, and only for files that live in this repository —
  generated tables and calibration files the model reads directly).

Two legitimate ways past a stop, both visible in the run manifest: re-derive
and re-pin, or override the value in an alternative.

## The behavior leg

One `behavior.yaml` per folder, two sections. The default is no response at
all:

```yaml
kg_dynamics: none
margins: {}
```

An alternative replaces either section wholesale:

```yaml
kg_dynamics:
  realization: config/calibrations/kg/realization.yaml
  conversion:  config/calibrations/kg/conversion.yaml

margins:
  entity_shifting:
    semi_elasticity: 0.6313333333333334
    distributed_share: 0.45
    retained_deferral_proxy: 0.25
  charity:
    elasticity: -0.5
    max_percent_change: 1.0
```

`margins` names the behavioral margins the scenario turns on, each with every
parameter it takes written out. There is no on-at-defaults shorthand: a
scenario that lets giving respond says what response it is running, in the
file, in numbers.

**Every margin runs in every scenario.** A margin left out of the block above
is not left out of the compiled program; it runs at its neutral values, which
return the columns it was given untouched. Two scenarios that differ only in
an elasticity therefore share one compiled program.

The margins are `tips`, `overtime`, `auto`, `entity_shifting`, `evasion`,
`wealth`, `charity` and `estate`. `src/taxmodel/behavior/loader.py` holds the
list, the module each one runs, and the order they run in, which is not the
order the file lists them: later margins read what earlier ones wrote.

`kg_dynamics` is `none` or the pieces of the capital-gains machinery the
scenario binds, each pointed at the calibration file carrying its value. The
gains applier comes in automatically and naming it is an error. The in-year
conversion of salary into equity is bound here rather than named as a margin,
because its response parameter lives in the calibration file this section
binds.

Reading the leg stops the run, before anything compiles, on an unknown margin
name, an unknown parameter name, or a margin that is on without stating one of
its parameters.

## Sweeps

A sweep is a family of generated folders, so it is recorded in the vintage rather
than vanishing with the shell that set it. Whatever writes the runscript rows
writes the folders too;
`projects/top_tax/build_dial_runs.py` is an example.
There are no environment-variable back doors — the nineteen that used to exist
are gone.

One trap: **a generated file must keep the BASE NAME of the file it stands in
for** (`.../eta_15/realization.yaml`, never `.../eta_15.yaml`).
Entries are labelled `{file stem}.{entry}`, so renaming the file relabels
every entry and breaks the link to any `derived_under` or `invalidated_by`
pin recorded against it.

## What each run records

Every run directory carries a `manifest.json` at its root; its contents are
described in `src/taxmodel/run/output_directory.py`.

## Related

- `config/calibrations/` — machine-written values and their provenance. Never
  hand-edited, and never round-tripped through a YAML library: in those files the
  comments ARE the record of where each number came from. The two exceptions are
  `kg/settings.yaml` and `estate/settings.yaml`, which are person-owned.
- `src/taxmodel/config/scenario.py` — the resolution engine for the economy leg.
- `src/taxmodel/config/law_layers.py` — tax law's subparameter-replacement
  parser.
- `src/taxmodel/behavior/config.py` and `src/taxmodel/behavior/loader.py` — the
  behavior leg reader, the pinned margin order, and the parse-time checks.
- `config/runscripts/archive/` — runscripts frozen on an earlier schema.
  The parser rejects them, which is correct; that folder's README gives the
  revival recipe.
