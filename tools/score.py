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


def clean_books_are_well_formed() -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.verify import verify

    for seed in range(20):
        problems = verify(generate(seed, {"errors": {}, "currencies": ["USD", "EUR"]}))
        if problems:
            return f"seed {seed}: {problems[0]}"
    return None


def same_seed_gives_same_files() -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.storage import episode_files

    config = {"errors": {}, "currencies": ["USD"]}
    for seed in range(5):
        if episode_files(generate(seed, config)) != episode_files(generate(seed, config)):
            return f"seed {seed}: files differ between two runs"
    return None


def written_episode_loads_back() -> Optional[str]:
    import tempfile

    from auditgym.generate import generate
    from auditgym.storage import load_episode, write_episode

    episode = generate(3, {"errors": {}, "currencies": ["USD"]})
    with tempfile.TemporaryDirectory() as directory:
        write_episode(episode, directory)
        return None if load_episode(directory) == episode else "loaded episode differs"


def solver_equals_truth() -> Optional[str]:
    """The independent solver, on open records only, must return exactly the generated truth: the
    default config (four types, distractors), both interactions, two currencies and a far offset;
    every finding also cites one full evidence set of its issue."""
    from auditgym.generate import generate
    from auditgym.solver import solve

    def key(item: dict, amount: str) -> tuple:
        return (item["type"], frozenset(item["affected_ids"]), item[amount], item["currency"])

    wider = {"invoices": [48, 100], "background_pairs": [150, 300]}
    configs = [None,
               {"interaction": "I1", "size": wider, "currencies": ["USD", "EUR"]},
               {"interaction": "I2", "size": wider, "timezone": "+14:00"},
               {"errors": {"duplicate_invoice": 2, "allocation_gap": 1, "fx_error": 2, "period_cutoff": 2},
                "currencies": ["USD", "EUR"], "timezone": "-11:00", "size": {"invoices": [56, 100],
                                                                            "background_pairs": [150, 300]}}]
    for index, config in enumerate(configs):
        for seed in range(50):
            episode = generate(seed, config)
            issues = episode["private"]["truth"]["issues"]
            truth = {key(issue, "expected_amount_minor"): issue for issue in issues}
            findings = solve(episode["public"])
            found = [key(finding, "claimed_amount_minor") for finding in findings]
            if set(found) != set(truth) or len(found) != len(truth):
                return f"config {index} seed {seed}: solver and truth differ"
            for finding, item in zip(findings, found):
                if not any(set(s) <= set(finding["evidence_ids"]) for s in truth[item]["evidence_sets"]):
                    return f"config {index} seed {seed}: {finding['type']} cites no full evidence set"
    return None


# Checks written by each zone. A name equal to a planned check replaces its todo;
# any other name is appended to the zone's criterion.
ZONE_CHECKS = {
    "Данные и эталон": ("tools.checks_gen", "tools.checks_solver", "tools.checks_qa"),
    "Надёжность награды": ("tools.checks_judge",),
    "Контракт и изоляция": ("tools.checks_env",),
}
RENAMES = {
    "any seed and any valid config gives a valid episode":
        "any seed and any valid config gives a valid episode; invalid config is rejected",
    "clean companies: solver finds nothing (50 seeds)": "clean companies: solver finds nothing",
}


def _load_zone_checks() -> None:
    import importlib

    planned = {name: checks for _, _, checks in CRITERIA for name, _ in checks}
    for criterion, modules in ZONE_CHECKS.items():
        own = next(c for name, _, c in CRITERIA if name == criterion)
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                if error.name != module_name:  # a broken import inside the module is not "not written yet"
                    raise
                continue
            for name, function in module.CHECKS:
                name = RENAMES.get(name, name)
                checks = planned.get(name)  # a planned check keeps its criterion
                if checks is None:
                    own.append((name, function))
                    planned[name] = own
                else:
                    index = [n for n, _ in checks].index(name)
                    checks[index] = (name, function)


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
        ("books balance before injection (per book, per currency, per event group)", clean_books_are_well_formed),
        ("after injection the deviation equals expected_amount_minor of each issue", todo),
        ("every evidence set exists, has at most 10 ids and is sufficient", todo),
        ("independent solver on open records equals the truth (200 episodes; make fuzz runs more)", solver_equals_truth),
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
        ("same seed and config: byte-identical episode files", same_seed_gives_same_files),
        ("same seed, config and actions: identical transitions and rewards", todo),
        ("reset isolates episodes (A, B, A again equals A)", todo),
        ("independent of PYTHONHASHSEED, locale, system time zone, working directory", todo),
        ("surface_seed change: oracle answer maps by id table, reward is the same", todo),
        ("external reorder and id rename of episode files: same oracle reward", todo),
        ("episode written to files loads back equal", written_episode_loads_back),
        ("Env.from_files gives the same observations as the generated episode", todo),
    ]),
    ("Инженерия и объяснение", 10, [
        ("one command runs everything offline in the container", todo),
        ("config and schema versions are recorded in every episode", todo),
        ("trajectory logs are written and contain no truth", todo),
        ("measured failure analysis is in docs/RESULTS.md", todo),
        ("limitations are stated in the README", todo),
    ]),
]


def _pending(message: str) -> bool:
    return message == "TODO" or message.startswith(("not implemented", "pending:"))


def run_check(function: Check) -> Optional[str]:
    """One check's result; a crash inside it is that check's FAIL (with the exception class), not
    the end of the table."""
    try:
        return function()
    except (Exception, SystemExit) as error:  # noqa: BLE001 - any crash is reported, never raised
        text = str(error).splitlines()[0][:200] if str(error) else ""
        return f"crashed: {type(error).__name__}" + (f": {text}" if text else "")


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    _load_zone_checks()
    total = 0.0
    failed_real = 0
    not_passed = 0
    print(f"{'Критерий':<34}{'Пройдено':>10}{'Баллы':>12}")
    details = []
    for name, points, checks in CRITERIA:
        passed = 0
        for check_name, function in checks:
            message = run_check(function)
            if message is None:
                passed += 1
                continue
            not_passed += 1
            pending = _pending(message)
            if not pending:
                failed_real += 1
            details.append(f"  [{'TODO' if pending else 'FAIL'}] {name}: {check_name}"
                           + ("" if message == "TODO" else f" -- {message}"))
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
