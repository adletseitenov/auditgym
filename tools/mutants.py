"""Mutation testing of the judge: break one rule at a time and check that something notices.

Each mutant swaps one function or constant of the judge (matcher, reward formula, structure
check) for a wrong version, the kind of slip a hand edit could make. A mutant is killed when the
attack table (`tools.attack.ATTACKS`, exact TP/FP/FN, codes and literal reward) or one of the
reward properties (`tools.checks_judge.CHECKS`, CASE §05) fails under it. A mutant nobody kills
is a rule the tests do not actually protect.

Run: python -m tools.mutants   (exit 1 if any mutant survives)
"""
from __future__ import annotations

import sys
from contextlib import ExitStack
from typing import Callable, Optional
from unittest import mock

from auditgym import findings, judge, matcher, reward

ALWAYS = lambda *args, **kwargs: True  # noqa: E731
HUGE = 10 ** 12


def _reward(f1_of: Callable[[int, int, int], float] | None = None,
            final: Callable[[float, int], float] | None = None) -> list:
    """Patches for judge.metrics / judge.terminal_reward built from the case formula with one change."""
    def metrics(tp, fp, fn):
        base = reward.metrics(tp, fp, fn)
        return dict(base, f1=f1_of(tp, fp, fn)) if f1_of else base

    def terminal(f1, steps):
        return final(float(f1), steps) if final else reward.terminal_reward(f1, steps)

    return [(judge, "metrics", metrics), (judge, "terminal_reward", terminal)]


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _f1(tp, fp, fn):
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def _amount_allows_negative(value):
    return value if type(value) is int else _ORIGINAL_AMOUNT(value)


def _parse_any_episode(payload, episode_id):
    if isinstance(payload, dict) and isinstance(payload.get("episode_id"), str):
        episode_id = payload["episode_id"]
    return _ORIGINAL_PARSE(payload, episode_id)


_ORIGINAL_AMOUNT = findings._amount
_ORIGINAL_PARSE = findings.parse_findings

# (name, [(module, attribute, replacement)])
MUTANTS: list[tuple[str, list]] = [
    # matcher: each condition of a TP dropped or weakened
    ("matcher: type not checked", [(matcher, "same_type", ALWAYS)]),
    ("matcher: affected_ids not checked", [(matcher, "same_affected_ids", ALWAYS)]),
    ("matcher: currency not checked", [(matcher, "same_currency", ALWAYS)]),
    ("matcher: amount not checked", [(matcher, "amount_within_tolerance", ALWAYS)]),
    ("matcher: tolerance ignored (exact amount only)",
     [(matcher, "amount_within_tolerance", lambda f, i, allowed: f["claimed_amount_minor"] == i["expected_amount_minor"])]),
    ("matcher: tolerance doubled",
     [(matcher, "amount_within_tolerance",
       lambda f, i, allowed: abs(f["claimed_amount_minor"] - i["expected_amount_minor"]) <= 2 * allowed + 1)]),
    ("matcher: evidence existence not checked", [(matcher, "evidence_exists", ALWAYS)]),
    ("matcher: evidence set not checked", [(matcher, "evidence_covers", ALWAYS)]),
    ("matcher: one issue may be matched twice", [(matcher, "issue_available", ALWAYS)]),
    ("matcher: superset of evidence never allowed", [(matcher, "superset_allowed", lambda policy: False)]),
    # reward: the case formula with one change
    ("reward: no step cost", _reward(final=lambda f1, s: _clip(2 * f1 - 1))),
    ("reward: step cost x10", _reward(final=lambda f1, s: _clip(2 * f1 - 1 - 0.02 * s))),
    ("reward: submit counted as an inspection step", _reward(final=lambda f1, s: _clip(2 * f1 - 1 - 0.002 * (s + 1)))),
    ("reward: no lower clip", _reward(final=lambda f1, s: min(1.0, 2 * f1 - 1 - 0.002 * s))),
    ("reward: F1 instead of 2*F1 - 1", _reward(final=lambda f1, s: _clip(f1 - 0.002 * s))),
    ("reward: empty truth and empty answer give F1 = 0",
     _reward(f1_of=lambda tp, fp, fn: 0.0 if tp + fp + fn == 0 else _f1(tp, fp, fn))),
    ("reward: findings on clean books give F1 = 1",
     _reward(f1_of=lambda tp, fp, fn: 1.0 if tp + fn == 0 else _f1(tp, fp, fn))),
    ("reward: false positives ignored (recall only)",
     _reward(f1_of=lambda tp, fp, fn: 1.0 if tp + fn == 0 else tp / (tp + fn))),
    ("reward: misses ignored (precision only)",
     _reward(f1_of=lambda tp, fp, fn: 1.0 if tp + fp == 0 else tp / (tp + fp))),
    # structure check: one limit or rule switched off
    ("structure: no limit of 30 findings", [(findings, "MAX_FINDINGS", HUGE)]),
    ("structure: no limit of 10 evidence ids", [(findings, "MAX_EVIDENCE_IDS", HUGE)]),
    ("structure: no limit of 1000 explanation characters", [(findings, "MAX_EXPLANATION_CODE_POINTS", HUGE)]),
    ("structure: no message size limit", [(findings, "MAX_MESSAGE_BYTES", HUGE)]),
    ("structure: negative amount accepted", [(findings, "_amount", _amount_allows_negative)]),
    ("structure: unknown type accepted", [(findings, "TYPES", findings.TYPES + ("fraud",))]),
    ("structure: foreign episode_id accepted", [(judge, "parse_findings", _parse_any_episode)]),
]


def _killed_by(first_only: bool = False) -> list[str]:
    """Names of the detectors that fail right now (attack rows, then reward properties). With
    `first_only`, stop at the first failure: the cheap attack table usually decides, and the
    properties (which generate episodes) run only for a mutant the table missed."""
    from tools.attack import ATTACKS, run_attack
    from tools.checks_judge import CHECKS

    failed = [f"attack: {attack[0]}" for attack in ATTACKS if not run_attack(attack)[0]]
    if failed and first_only:
        return failed[:1]
    for name, check in CHECKS:
        try:
            problem: Optional[str] = check()
        except Exception as error:  # noqa: BLE001 - a crash under a mutant is a detection
            problem = f"crashed: {type(error).__name__}"
        if problem is not None:
            failed.append(f"property: {name}")
            if first_only:
                break
    return failed


def run(mutants: list[tuple[str, list]] = MUTANTS) -> list[tuple[str, list[str]]]:
    """(mutant name, first detector that failed) for every mutant; an empty list means it survived."""
    results = []
    for name, patches in mutants:
        with ExitStack() as stack:
            for module, attribute, replacement in patches:
                stack.enter_context(mock.patch.object(module, attribute, replacement))
            results.append((name, _killed_by(first_only=True)))
    return results


def main() -> int:
    baseline = _killed_by()
    if baseline:
        print("the unmutated judge already fails:", "; ".join(baseline))
        return 1
    results = run()
    for name, failed in results:
        print(f"{'killed  ' if failed else 'SURVIVED'} {name}" + (f"  (by {failed[0]})" if failed else ""))
    killed = sum(1 for _, failed in results if failed)
    print(f"{killed}/{len(results)} judge mutants killed")
    return 0 if killed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
