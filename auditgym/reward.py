"""Metrics and terminal reward, exactly as in CASE §05.

    P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); a zero denominator gives P or R = 0;
    empty truth and empty findings give F1 = 1; empty truth with findings gives F1 = 0.
    reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)

The reward is computed literally, in IEEE-754 doubles, as the organizers' formula is written:

    f1 = 2*tp / (2*tp + fp + fn)            (1.0 when tp = fp = fn = 0)
    reward = max(-1.0, min(1.0, 2*f1 - 1 - 0.002*steps))

so a checker that evaluates the formula in a float gets the same bits (audit R4: the exactly
rounded value differed from it by 1 ulp for 16 % of the combinations). `metrics_exact` and
`terminal_reward_exact` (`fractions.Fraction`) stay as the mathematical reference for tests.
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
    """Precision, recall and F1 as floats, by the plain float formulas of CASE §05."""
    tp, fp, fn = _count(tp, "tp"), _count(fp, "fp"), _count(fn, "fn")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1}


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
    """max(-1.0, min(1.0, 2*f1 - 1 - 0.002*steps)) in float arithmetic, written exactly so."""
    f1 = float(_rational(f1, "f1"))
    steps = _count(inspection_steps, "inspection_steps")
    return max(-1.0, min(1.0, 2 * f1 - 1 - 0.002 * steps))
