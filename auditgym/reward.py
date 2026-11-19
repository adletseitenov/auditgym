"""Metrics and terminal reward, exactly as in CASE §05.

    P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); a zero denominator gives P or R = 0;
    empty truth and empty findings give F1 = 1; empty truth with findings gives F1 = 0.
    reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)

Arithmetic is exact (`fractions.Fraction`; 0.002 is the rational 1/500). A value becomes a
float only at the very end, by `_to_float`: the exact rational is rounded once to the nearest
IEEE-754 double, ties to even (integer true division in CPython is correctly rounded). So the
result is a pure function of the integers involved and is bit-identical on every platform with
IEEE doubles; e.g. 2*1 - 1 - 0.002*100 is exactly 0.8, not 0.7999999999999998.
"""
from __future__ import annotations

from fractions import Fraction

STEP_COST = Fraction(2, 1000)
REWARD_MIN = Fraction(-1)
REWARD_MAX = Fraction(1)


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _to_float(value: Fraction) -> float:
    return value.numerator / value.denominator  # correctly rounded, ties to even


def metrics_exact(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and F1 as exact Fractions."""
    tp, fp, fn = _count(tp, "tp"), _count(fp, "fp"), _count(fn, "fn")
    precision = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
    recall = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
    if tp + fp + fn == 0:
        f1 = Fraction(1)  # empty truth and empty findings
    else:
        f1 = Fraction(2 * tp, 2 * tp + fp + fn)  # empty truth with findings: tp = 0, so 0
    return {"precision": precision, "recall": recall, "f1": f1}


def metrics(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and F1 as floats (rounded once from the exact values)."""
    return {name: _to_float(value) for name, value in metrics_exact(tp, fp, fn).items()}


def _rational(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction, float)):
        raise ValueError(f"{name} must be a number")
    exact = Fraction(value)  # a float converts exactly; pass a Fraction to avoid its error
    if not 0 <= exact <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return exact


def terminal_reward_exact(f1: object, inspection_steps: int) -> Fraction:
    value = 2 * _rational(f1, "f1") - 1 - STEP_COST * _count(inspection_steps, "inspection_steps")
    return min(max(value, REWARD_MIN), REWARD_MAX)


def terminal_reward(f1: object, inspection_steps: int) -> float:
    """clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1) as a float rounded once."""
    return _to_float(terminal_reward_exact(f1, inspection_steps))
