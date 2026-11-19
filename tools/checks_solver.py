"""Solver checks for tools/score.py. Each returns None on success or a failure message.

CHECKS is a list of (name, function); the score owner wires them into CRITERIA.
"""
from __future__ import annotations

import pathlib
from typing import Callable, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
OFFSETS = ("+05:00", "-11:00", "+14:00", "+05:30")
CURRENCY_SETS = ((), ("USD",), ("USD", "EUR"), ("CNY", "RUB"))


def _key(finding: dict, mapping: dict | None = None) -> tuple:
    def m(value):
        return mapping.get(value, value) if mapping else value

    return (finding["type"], tuple(sorted(m(i) for i in finding["affected_ids"])), finding["claimed_amount_minor"],
            finding["currency"], tuple(sorted(m(i) for i in finding["evidence_ids"])))


def clean_companies_solver_finds_nothing() -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.solver import solve

    for seed in range(50):
        config = {"errors": {}, "currencies": list(CURRENCY_SETS[seed % 4]), "timezone": OFFSETS[seed % 4]}
        findings = solve(generate(seed, config)["public"])
        if findings:
            return f"seed {seed}: {findings[0]['type']} on {findings[0]['affected_ids']}"
    return None


def tiny_fixture_solver_equals_truth() -> Optional[str]:
    from auditgym.solver import solve
    from auditgym.storage import load_episode

    episode = load_episode(ROOT / "tests" / "fixtures" / "tiny")
    findings = solve(episode["public"])
    issues = episode["private"]["truth"]["issues"]
    if len(findings) != len(issues):
        return f"{len(findings)} findings for {len(issues)} issues"
    for issue in issues:
        hits = [f for f in findings if f["type"] == issue["type"] and set(f["affected_ids"]) == set(issue["affected_ids"])
                and f["currency"] == issue["currency"] and f["claimed_amount_minor"] == issue["expected_amount_minor"]
                and any(set(s) <= set(f["evidence_ids"]) for s in issue["evidence_sets"])]
        if len(hits) != 1:
            return f"issue {issue['issue_id']} is not found exactly once"
    return None


def solver_ignores_order_and_ids() -> Optional[str]:
    from auditgym.solver import solve
    from tests.solver.cases import COMPANIES, DISTRACTOR_CASES, ERROR_CASES, build
    from tests.solver.helpers import ID_STYLES, rename_and_shuffle

    for index, (name, builder) in enumerate(sorted({**ERROR_CASES, **DISTRACTOR_CASES}.items())):
        seed, currencies, timezone = COMPANIES[index % len(COMPANIES)]
        built = build(name, builder, seed, currencies, timezone)
        if built is None:
            continue
        public, _ = built
        expected = sorted(_key(f) for f in solve(public))
        renamed, mapping = rename_and_shuffle(public, index, ID_STYLES[index % len(ID_STYLES)])
        inverse = {new: old for old, new in mapping.items()}
        if sorted(_key(f, inverse) for f in solve(renamed)) != expected:
            return f"{name}: findings change after reorder and rename"
    return None


def solver_handles_every_distractor() -> Optional[str]:
    """Every hand-built distractor (tests/solver/cases.py) on every company that fits it: no finding."""
    from auditgym.solver import solve
    from tests.solver.cases import COMPANIES, DISTRACTOR_CASES, build

    for name, builder in sorted(DISTRACTOR_CASES.items()):
        built = 0
        for seed, currencies, timezone in COMPANIES:
            case = build(name, builder, seed, currencies, timezone)
            if case is None:
                continue
            built += 1
            findings = solve(case[0])
            if findings:
                return f"{name} (seed {seed}): {findings[0]['type']} on {findings[0]['affected_ids']}"
        if built < 2:
            return f"{name}: only {built} companies fit the edit"
    return None


CHECKS: list[tuple[str, Callable[[], Optional[str]]]] = [
    ("clean companies: solver finds nothing (50 seeds)", clean_companies_solver_finds_nothing),
    ("tiny fixture: solver finds exactly its truth", tiny_fixture_solver_equals_truth),
    ("solver findings do not depend on record order or ids", solver_ignores_order_and_ids),
    ("solver handles every listed distractor without a finding", solver_handles_every_distractor),
]


def main() -> int:
    failed = 0
    for name, check in CHECKS:
        message = check()
        failed += message is not None
        print(f"[{'ok' if message is None else 'FAIL'}] {name}" + ("" if message is None else f" -- {message}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
