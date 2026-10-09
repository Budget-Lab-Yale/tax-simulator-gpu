"""Helpers the law-stack tests share: where a stack file sits, and the law
with every provision of a stack in force."""

from __future__ import annotations

from pathlib import Path

from taxmodel.config.law_layers import ResolvedLaw, resolve_law_stack
from taxmodel.config.law_stacks import read_law_stack
from taxmodel.config.scenario import TAX_LAW_LEG, leg_path


def stack_file_path(root: Path, name: str) -> Path:
    """The path of a stack file of this name under a scenario tree, beside the
    reform folders."""
    return leg_path(Path(root), TAX_LAW_LEG, f"{name}.yaml")


def combined_law(stack_path: str | Path, config_root: str | Path) -> ResolvedLaw:
    """The law with every provision of the stack in force: the last cumulative law.

    Only the tax-law legs fold here. Economy and behavior changes made by the
    provisions are not part of the result.
    """
    stack = read_law_stack(stack_path)
    return resolve_law_stack(Path(config_root), stack.tax_law_folders)[-1]
