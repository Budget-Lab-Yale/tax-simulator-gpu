# Contributing

## Comments (control requirement, not a style preference)

Every comment in this codebase is written for a tax economist reading the
code as a statement of tax law and modeling choices. Reviewers enforce
this rule.

- Say what the code means in the economics: "the additional standard
  deduction for age 65+ stacks per qualifying spouse", not "iterate over
  eligibility flags".
- Cite the law or the modeling decision when the code embodies one
  (IRC section, ARCHITECTURE.md decision number).
- No invented shorthand. A term that is not standard in public finance or
  plainly self-explanatory is defined where it is used, or not used.
- No LLM-ese and no software-engineering filler: never "robust",
  "handle edge cases", "for clarity", "note that", or comments that
  restate the line below them.
- A magic number is a bug. Every constant comes from compiled law, from
  the scenario configuration, or is a named constant in the code carrying
  the citation for the assumption it stands for. A number fitted to data
  belongs in a calibration file under `config/calibrations/`, bound by the
  scenario configuration, not written into the code.

## Writing a threshold in a calculator

Marginal tax rates in this model are gradients: one reverse-mode pass over
the calculator chain. Every threshold in the tax code is a corner, and
exactly on the corner there is no single slope — the one autodiff reports
is decided by nothing but how the expression was written. This model's
convention is that **at an exact threshold the household is treated as
having crossed it in the direction of more income.**

The question to ask at each site is therefore: one more dollar of the
household's income — which branch does that land it in? That branch takes
the tie. Use the two functions in `src/taxmodel/taxes/calc/_corners.py`, whose
first argument is always the amount and whose second is always the edge:

- `floored_at(amount, floor)` for "cannot go below".
- `capped_at(amount, cap)` for "cannot go above".

Both take a keyword saying which way the amount moves **relative to the
edge it is tested against**, because the convention is stated in the
direction of more *income*, not more of the amount. For an amount that
rises against its edge — taxable income against a bracket edge, the excess
over a phase-out threshold, a bracket's filled width — the default is
right and the comparison is `>=`. For an amount that *sinks* against its
edge — a credit being phased out toward zero, a cap phasing down, an
expense measured against a floor that is a percentage of AGI — pass
`sinks_against_the_limit=True` and the comparison becomes `>`, so a tie
takes the other branch. These are the same one rule seen from both sides:
past the corner is where more income puts you.

"Relative to the edge" is load-bearing wherever the edge is itself a tax
computation rather than a fixed dollar amount in the law. Where the edge
cannot move, the only question is whether the amount falls. Where the edge
grows with income too, the question is which of the two grows faster: an
amount that rises with income but rises more slowly than its edge is
sinking against it, and the tie goes the same way as if it had fallen
outright. The Schedule D limit in `taxes/calc/tax.py` is the case that taught us
this — tax at preferred rates rises with income but more slowly than
ordinary tax on the same income, which is the point of preferred rates —
and getting it backwards cost a behavioral response a third of its size
before anyone noticed.

An amount that does not move with income at all, or that moves either way
depending on what kind of income the extra dollar is, has no more-income
side. Leave those on the default and say so in a comment.

Writing the comparison out as `jnp.where(amount >= edge, ...)` is equally
fine where that reads better.

Do not use `jnp.maximum`, `jnp.minimum`, `jnp.clip`, `jnp.abs` or
`jax.nn.relu` in `src/taxmodel/taxes/calc/` or `src/taxmodel/taxes/statutory_order.py`.
The first three report the average of the two slopes at a tie, a marginal
rate no household faces; the last two pick a side of their own.
`tests/unit/taxes/calc/test_corner_spellings.py` fails on any of them unless the
line carries a trailing `# tie: <one-line reason>` comment marking a site
that was looked at and deliberately kept — an integer count, a comparison
between two law parameters, a choice between two amounts where neither is
the household's income. The marker records a decision; it is not a way to
skip making one.

`tests/unit/taxes/calc/test_autodiff_tie_conventions.py` pins what JAX actually
does at a tie for each of these spellings. It is the test that fails if a
toolchain upgrade moves the ground the convention stands on.

Two cautions when changing an existing expression. First, the rewrite has
to return the same number at every input, ties included — only the slope
exactly at a tie may change; if a value-level test moves, the rewrite was
wrong. Second, `floored_at` and `capped_at` do not propagate NaN the way
`jnp.maximum` does, and the compiled law pads unused schedule slots with
NaN, so a law array goes through `live_slots` before it meets a floor or a
cap.

## Environment

`uv sync` builds the venv from the committed lockfile (Python 3.12,
jax 0.7.0). On the cluster add the CUDA wheels: `uv sync --extra cuda`.
Tests: `uv run pytest`. Lint: `uv run ruff check`.

Cluster rules: all compute runs in Slurm jobs (login nodes have no GPU
driver); model data never leaves Bouchet; nothing PUF-derived is ever
committed — public CI sees synthetic fixtures only.
