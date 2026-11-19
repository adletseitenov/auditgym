"""Metrics and terminal reward, exactly as in CASE.md section 05.

P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). A zero denominator gives P or R = 0.
If the truth and the findings are both empty, F1 = 1; if the truth is empty but there are
findings, F1 = 0 (both follow from the F1 formula with 0/0 read as 1).

    reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)

Arithmetic is exact (`Fraction`); the result is converted to float once, by an integer division
that Python rounds correctly, so the same inputs give the same bits on every platform.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Union

STEP_COST = Fraction(2, 1000)
Number = Union[int, Fraction, float]


def _to_float(value: Fraction) -> float:
    return value.numerator / value.denominator


def _count(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def f1_exact(tp: int, fp: int, fn: int) -> Fraction:
    tp, fp, fn = _count("tp", tp), _count("fp", fp), _count("fn", fn)
    denominator = 2 * tp + fp + fn
    return Fraction(1) if denominator == 0 else Fraction(2 * tp, denominator)


def metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall and F1 as floats."""
    tp, fp, fn = _count("tp", tp), _count("fp", fp), _count("fn", fn)
    precision = Fraction(0) if tp + fp == 0 else Fraction(tp, tp + fp)
    recall = Fraction(0) if tp + fn == 0 else Fraction(tp, tp + fn)
    return {"precision": _to_float(precision), "recall": _to_float(recall), "f1": _to_float(f1_exact(tp, fp, fn))}


def terminal_reward(f1: Number, inspection_steps: int) -> float:
    """clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1).

    `f1` is preferably exact (`f1_exact`). A float from `metrics()` is accepted too: F1 is a
    ratio of small integers (denominator at most 2*30 + issues), so `limit_denominator`
    recovers it exactly and the reward is the same bits either way.
    """
    if isinstance(f1, bool) or not isinstance(f1, (int, Fraction, float)):
        raise TypeError("f1 must be a number")
    exact = Fraction(f1).limit_denominator(100_000) if isinstance(f1, float) else Fraction(f1)
    if not 0 <= exact <= 1:
        raise ValueError("f1 must be within [0, 1]")
    steps = _count("inspection_steps", inspection_steps)
    raw = 2 * exact - 1 - STEP_COST * steps
    return _to_float(min(Fraction(1), max(Fraction(-1), raw)))
