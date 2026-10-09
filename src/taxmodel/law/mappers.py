"""Filing-status mappers.

The law tree states the ordinary brackets for single, joint, and head of
household filers, and then says what a married-filing-separate return gets:

    filing_status_mapper:
      brackets:
        '1': brackets_single
        '2': brackets_married
        '3': brackets_married / 2
        '4': brackets_head

Filing statuses are 1 single, 2 joint, 3 married filing separately, 4 head of
household. The right-hand side is arithmetic on other subparameters of the same
file, applied slot by slot. This is how the tree encodes the marriage
adjustments the Code writes as halves and doublings: the separate-return
bracket is half the joint one, the additional Medicare tax threshold likewise,
and the student loan interest deduction is not doubled for joint returns
(`sl_limit_single * 0` for the separate return that cannot claim it).

The evaluator accepts identifiers, numbers, `+ - * /`, and parentheses, and
refuses everything else. Nothing here calls `eval()`, so a law file cannot
execute code.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from .errors import LawError

_TOKENS = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<op>[-+*/()])
    """,
    re.VERBOSE,
)

Operand = float | np.ndarray


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    position: int


def _tokenize(expression: str) -> list[_Token]:
    tokens: list[_Token] = []
    at = 0
    while at < len(expression):
        match = _TOKENS.match(expression, at)
        if match is None:
            raise LawError(
                f"mapper expression {expression!r} has an unreadable character "
                f"at position {at}: {expression[at]!r}. Mappers may use "
                "subparameter names, numbers, + - * / and parentheses."
            )
        kind = match.lastgroup
        assert kind is not None
        if kind != "space":
            tokens.append(_Token(kind, match.group(), at))
        at = match.end()
    if not tokens:
        raise LawError("mapper expression is empty")
    return tokens


class _Parser:
    """Recursive descent over the mapper grammar, with the usual precedence.

        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('+' | '-') factor | number | name | '(' expression ')'
    """

    def __init__(self, expression: str, resolve: Callable[[str], Operand]):
        self.expression = expression
        self.tokens = _tokenize(expression)
        self.at = 0
        self.resolve = resolve

    def run(self) -> Operand:
        value = self.expression_rule()
        if self.at != len(self.tokens):
            leftover = self.tokens[self.at]
            raise LawError(
                f"mapper expression {self.expression!r} has trailing "
                f"{leftover.text!r} at position {leftover.position}"
            )
        return value

    def peek(self) -> _Token | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> _Token:
        token = self.tokens[self.at]
        self.at += 1
        return token

    def expression_rule(self) -> Operand:
        value = self.term()
        while (token := self.peek()) is not None and token.text in ("+", "-"):
            self.take()
            right = self.term()
            value = value + right if token.text == "+" else value - right
        return value

    def term(self) -> Operand:
        value = self.factor()
        while (token := self.peek()) is not None and token.text in ("*", "/"):
            self.take()
            right = self.factor()
            if token.text == "*":
                value = value * right
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    value = value / right
        return value

    def factor(self) -> Operand:
        token = self.peek()
        if token is None:
            raise LawError(
                f"mapper expression {self.expression!r} ends where a value was "
                "expected"
            )
        if token.text in ("+", "-"):
            self.take()
            value = self.factor()
            return value if token.text == "+" else -value
        if token.kind == "number":
            self.take()
            return float(token.text)
        if token.kind == "name":
            self.take()
            return self.resolve(token.text)
        if token.text == "(":
            self.take()
            value = self.expression_rule()
            closing = self.peek()
            if closing is None or closing.text != ")":
                raise LawError(
                    f"mapper expression {self.expression!r} has an unclosed "
                    "parenthesis"
                )
            self.take()
            return value
        raise LawError(
            f"mapper expression {self.expression!r} has {token.text!r} at "
            f"position {token.position} where a value was expected"
        )


def evaluate(expression: str, operands: Mapping[str, Operand]) -> Operand:
    """Evaluate one mapper expression against the file's other subparameters."""

    def resolve(name: str) -> Operand:
        if name not in operands:
            known = ", ".join(sorted(operands)) or "(none)"
            raise LawError(
                f"mapper expression {expression!r} names {name!r}, which is not "
                f"a subparameter of this file. Available: {known}"
            )
        return operands[name]

    return _Parser(str(expression), resolve).run()


def identifiers(expression: str) -> set[str]:
    """The subparameter names an expression actually reads."""
    return {t.text for t in _tokenize(str(expression)) if t.kind == "name"}

