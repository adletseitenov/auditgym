"""Safe calculator for the `calculator` action (docs/CASE.md section 04, docs/SPEC.md).

Grammar (recursive descent, nothing is ever evaluated as Python):

    expression := term (("+" | "-") term)*
    term       := unary (("*" | "/") unary)*
    unary      := "-" unary | primary
    primary    := NUMBER | "(" expression ")"
    NUMBER     := [0-9]+ ("." [0-9]+)?        ASCII digits only

Whitespace (ASCII space and tab) may separate tokens. Anything else -- names, calls, `**`,
exponent notation (`1e5`), unary plus, unicode digits, empty input, input longer than
200 characters -- raises CalcError.

Arithmetic runs on Decimal with one fixed context (CONTEXT below): 34 significant digits,
ROUND_HALF_EVEN, exponent range +-999; division by zero, invalid operations, overflow and
underflow raise CalcError. A literal with more than 34 significant digits raises CalcError
instead of being rounded; results of operations are rounded to 34 significant digits.
Nesting deeper than 100 levels (parentheses or unary minus) raises CalcError.

Result format: fixed-point notation of the exact Decimal result, no exponent, no thousands
separator, `-` for negatives, never `-0`. Trailing zeros follow Decimal arithmetic, so
`125000 * 465.37` gives `58171250.00` and `10 / 4` gives `2.5`.
"""
from __future__ import annotations

import decimal
from decimal import Decimal

MAX_LENGTH = 200
MAX_DEPTH = 100

CONTEXT = decimal.Context(
    prec=34, rounding=decimal.ROUND_HALF_EVEN, Emin=-999, Emax=999, capitals=1, clamp=0,
    flags=[], traps=[decimal.DivisionByZero, decimal.InvalidOperation, decimal.Overflow,
                     decimal.Underflow, decimal.Subnormal, decimal.Clamped],
)

_DIGITS = "0123456789"
_SPACE = " \t"
_OPERATORS = "+-*/()"


class CalcError(ValueError):
    """The expression is not allowed or cannot be computed. The message is fixed."""

    def __init__(self) -> None:
        super().__init__("Bad expression.")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in _SPACE:
            i += 1
        elif char in _OPERATORS:
            tokens.append(char)
            i += 1
        elif char in _DIGITS:
            start = i
            while i < len(text) and text[i] in _DIGITS:
                i += 1
            if i < len(text) and text[i] == ".":
                i += 1
                if i >= len(text) or text[i] not in _DIGITS:
                    raise CalcError()
                while i < len(text) and text[i] in _DIGITS:
                    i += 1
            tokens.append(text[start:i])
        else:
            raise CalcError()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0
        self.depth = 0
        self.context = CONTEXT.copy()

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise CalcError()
        self.position += 1
        return token

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalcError()

    def expression(self) -> Decimal:
        self.enter()
        value = self.term()
        while self.peek() in ("+", "-"):
            if self.take() == "+":
                value = self.context.add(value, self.term())
            else:
                value = self.context.subtract(value, self.term())
        self.depth -= 1
        return value

    def term(self) -> Decimal:
        value = self.unary()
        while self.peek() in ("*", "/"):
            if self.take() == "*":
                value = self.context.multiply(value, self.unary())
            else:
                value = self.context.divide(value, self.unary())
        return value

    def unary(self) -> Decimal:
        if self.peek() == "-":
            self.take()
            self.enter()
            value = self.context.minus(self.unary())
            self.depth -= 1
            return value
        return self.primary()

    def primary(self) -> Decimal:
        token = self.take()
        if token == "(":
            value = self.expression()
            if self.take() != ")":
                raise CalcError()
            return value
        if token[0] in _DIGITS:
            literal = Decimal(token)
            if len(literal.as_tuple().digits) > self.context.prec:
                raise CalcError()  # a literal is never silently rounded
            return self.context.create_decimal(literal)
        raise CalcError()


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression; return the result as a fixed-point decimal string."""
    if not isinstance(expression, str) or not expression or len(expression) > MAX_LENGTH:
        raise CalcError()
    tokens = _tokenize(expression)
    if not tokens:
        raise CalcError()
    parser = _Parser(tokens)
    try:
        value = parser.expression()
    except (decimal.DecimalException, ArithmeticError, RecursionError) as exc:
        raise CalcError() from exc
    if parser.peek() is not None:
        raise CalcError()
    if value.is_zero():
        value = value.copy_abs()
    return format(value, "f")
