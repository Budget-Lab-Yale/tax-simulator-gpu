"""The law layer: YAML tax law in, fixed-shape arrays out.

`compile_law(config_root, years, index_table, alternative)` is the public
entry point. Carry-forward and the `Inf`/`NA` conventions are in parse.py,
the filing-status mappers in mappers.py, indexation in indexation.py, and the
checks on reforms in lint.py.

`batch.compile_batch` compiles a set of laws that share one compiled program.
It is imported by name from `taxmodel.law.batch`, because it reads the
calculators' law-to-parameter tables and re-exporting it here would make the
law layer import the engine.
"""

from .compile import (
    FILING_STATUSES,
    MAX_BRACKETS,
    CompiledLaw,
    FieldSpec,
    compile_law,
    compile_resolved,
)
from .errors import LawError, LawLintError
from .indexation import IndexTable

__all__ = [
    "FILING_STATUSES",
    "MAX_BRACKETS",
    "CompiledLaw",
    "FieldSpec",
    "IndexTable",
    "LawError",
    "LawLintError",
    "compile_law",
    "compile_resolved",
]
