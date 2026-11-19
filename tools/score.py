"""Our mirror of the organizers' scoring (CASE.md, section 10).

Each criterion has a list of named checks. A check is a function returning None on success
or a failure message. Checks that are planned but not written yet are listed with `todo` and
count as NOT passed, so the printed score is never better than what is proven.

Run: python -m tools.score            (exit 1 only when an implemented check fails)
     python -m tools.score --strict   (exit 1 when anything, including a todo, is not passed)
"""
from __future__ import annotations

import sys
from typing import Callable, Optional

Check = Callable[[], Optional[str]]


def todo() -> Optional[str]:
    return "TODO"


def rng_is_reproducible() -> Optional[str]:
    from auditgym.rng import Rng

    first = [Rng(7, "x").u64() for _ in range(3)]
    second = [Rng(7, "x").u64() for _ in range(3)]
    return None if first == second else "same seed gave different values"


def core_is_deterministic_by_lint() -> Optional[str]:
    from tools import lint_determinism

    return None if lint_determinism.main() == 0 else "determinism lint failed"


# (criterion, points, [(check name, function)]). Names are the plan: one per sentence of the case.
CRITERIA: list[tuple[str, float, list[tuple[str, Check]]]] = [
    ("Контракт и изоляция", 25, [
        ("reset returns (observation, info) with task, manifest, policy, budget", todo),
        ("step returns (observation, reward, terminated, truncated, info), JSON-compatible", todo),
        ("list_records: page of at most 50 rows, stable cursors, every row exactly once", todo),
        ("get_record and get_document return only the requested record", todo),
        ("calculator: arithmetic and decimals only; code, names and files rejected", todo),
        ("unknown id or action: bounded error, step is spent", todo),
        ("100 steps including submit; submit gives terminated=true", todo),
        ("step or time limit gives truncated=true and an empty final answer", todo),
        ("reward is zero before the terminal step", todo),
        ("eval mode: reward and diagnostics stay with the evaluator", todo),
        ("evaluator runs outside the agent process; truth is not on the agent side", todo),
        ("no truth, seed, provenance or paths in observations, info, errors", todo),
        ("ids, ordering and counts do not separate affected records from the rest", todo),
        ("episode files follow the schemas: ISO-8601 offsets, integer money, null links", todo),
        ("step after the end and step before reset give a fixed error, no state change", todo),
    ]),
    ("Данные и эталон", 25, [
        ("books balance before injection (per book, per currency, per event group)", todo),
        ("after injection the deviation equals expected_amount_minor of each issue", todo),
        ("every evidence set exists, has at most 10 ids and is sufficient", todo),
        ("independent solver on open records equals the truth (1000 seeds)", todo),
        ("clean companies: solver finds nothing", todo),
        ("distractors do not satisfy any error definition", todo),
        ("each of the four error types can be configured on its own", todo),
        ("both interaction scenarios give an unambiguous truth", todo),
        ("any seed and any valid config gives a valid episode; invalid config is rejected", todo),
        ("scripted policy solves the largest episode within 100 steps and 60 seconds", todo),
    ]),
    ("Надёжность награды", 25, [
        ("F1 and reward formulas match the case, including the empty cases", todo),
        ("full correct answer beats silence on books with errors, for every step count", todo),
        ("silence beats false accusations on clean books", todo),
        ("duplicate findings give no gain", todo),
        ("flag-everything scores below silence on every valid config", todo),
        ("saving steps separates equal answers only slightly", todo),
        ("one-to-one matching is deterministic", todo),
        ("wrong type, ids, amount beyond tolerance, currency or evidence is not a TP", todo),
        ("invalid structure ends the episode with reward -1 and a judge log entry", todo),
        ("limits: 30 findings, 10 evidence ids, 1000 characters", todo),
        ("explanation text does not change the reward", todo),
        ("guessed and non-existent ids give no gain", todo),
    ]),
    ("Воспроизводимость и обобщение", 15, [
        ("rng: same seed gives the same stream", rng_is_reproducible),
        ("core package passes the determinism lint", core_is_deterministic_by_lint),
        ("same seed, config and actions: byte-identical records, transitions, rewards", todo),
        ("reset isolates episodes (A, B, A again equals A)", todo),
        ("independent of PYTHONHASHSEED, locale, system time zone, working directory", todo),
        ("surface_seed change: oracle answer maps by id table, reward is the same", todo),
        ("external reorder and id rename of episode files: same oracle reward", todo),
        ("episode loads from files (Env.from_files) and equals the generated one", todo),
    ]),
    ("Инженерия и объяснение", 10, [
        ("one command runs everything offline in the container", todo),
        ("config and schema versions are recorded in every episode", todo),
        ("trajectory logs are written and contain no truth", todo),
        ("measured failure analysis is in docs/RESULTS.md", todo),
        ("limitations are stated in the README", todo),
    ]),
]


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    total = 0.0
    failed_real = 0
    not_passed = 0
    print(f"{'Критерий':<34}{'Пройдено':>10}{'Баллы':>12}")
    details = []
    for name, points, checks in CRITERIA:
        passed = 0
        for check_name, function in checks:
            message = function()
            if message is None:
                passed += 1
                continue
            not_passed += 1
            if function is not todo:
                failed_real += 1
            details.append(f"  [{'TODO' if function is todo else 'FAIL'}] {name}: {check_name}"
                           + ("" if function is todo else f" -- {message}"))
        earned = points * passed / len(checks)
        total += earned
        print(f"{name:<34}{passed:>6}/{len(checks):<3}{earned:>8.1f}/{points:g}")
    print(f"{'ИТОГО':<34}{'':>10}{total:>8.1f}/100")
    if details:
        print("\nНе пройдено:")
        print("\n".join(details))
    return 1 if (failed_real or (strict and not_passed)) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
