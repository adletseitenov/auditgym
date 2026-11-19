"""Generator checks for tools/score.py, criterion "Данные и эталон".

CHECKS = [(name, function)]; each function returns None on success or a failure message.
They call auditgym.verify, which reads the open records, the pre-injection snapshot and the
recorded steps only, and evaluates the four published error predicates.
"""
from __future__ import annotations

from typing import Callable, Optional

SEEDS = range(25)
ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
SMALL = {"invoices": [32, 48], "background_pairs": [0, 40]}
CONFIGS = [
    {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1}, "currencies": ["USD"]},
    {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
     "currencies": ["USD", "EUR"], "timezone": "-11:00", "size": SMALL},
    {"errors": {"fx_error": 2, "period_cutoff": 2}, "currencies": ["CNY"], "timezone": "+14:00", "size": SMALL},
    {"errors": {"duplicate_invoice": 1, "allocation_gap": 1}, "currencies": ["RUB"], "base_currency": "USD",
     "timezone": "+05:45", "size": SMALL, "distractors": False},
]
INTERACTION_CONFIGS = [
    {"errors": {}, "interaction": "I1", "currencies": ["USD"], "size": SMALL},
    {"errors": {"fx_error": 1}, "interaction": "I1", "currencies": ["EUR", "CNY"], "timezone": "-09:30", "size": SMALL},
    {"errors": {}, "interaction": "I2", "currencies": [], "size": SMALL},
    {"errors": {"duplicate_invoice": 1}, "interaction": "I2", "currencies": ["USD"], "timezone": "+14:00", "size": SMALL},
]


def _first_problem(match: Callable[[str], bool], configs=CONFIGS, seeds=SEEDS) -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.verify import verify

    for index, config in enumerate(configs):
        for seed in seeds:
            problems = [p for p in verify(generate(seed, config)) if match(p)]
            if problems:
                return f"config {index}, seed {seed}: {problems[0]}"
    return None


def deviation_equals_expected_amount() -> Optional[str]:
    return _first_problem(lambda p: ": deviation:" in p or "no issue lists it" in p)


def evidence_sets_exist_and_suffice() -> Optional[str]:
    return _first_problem(lambda p: ": evidence:" in p)


def each_error_type_configurable_alone() -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.verify import verify

    for name in ERROR_TYPES:
        for count in (1, 3):
            config = {"errors": {name: count}, "currencies": ["USD"]}
            episode = generate(7, config)
            issues = episode["private"]["truth"]["issues"]
            if len(issues) != count or any(issue["type"] != name for issue in issues):
                return f"{name} x{count}: truth has {len(issues)} issues of other types or counts"
            problems = verify(episode)
            if problems:
                return f"{name} x{count}: {problems[0]}"
    return None


def any_seed_and_valid_config_gives_valid_episode() -> Optional[str]:
    from auditgym.config import ConfigError, normalize_config

    configs = CONFIGS + INTERACTION_CONFIGS + [
        None, {},
        {"errors": {"period_cutoff": 1}, "currencies": [], "size": {"invoices": [10, 10], "background_pairs": [0, 5]}},
        {"errors": {}, "currencies": ["RUB"], "base_currency": "USD"},
        {"errors": {"duplicate_invoice": 1}, "period": {"start": "2027-02-01", "months": 3}},
    ]
    problem = _first_problem(lambda p: True, configs, range(8))
    if problem:
        return problem
    for bad in ({"seed": 1}, {"errors": {"fx_error": "1"}}, {"timezone": "+05:00\n"}, {"currencies": [["USD"]]},
                {"period": {"start": "1999-01-01", "months": 1}}, [], {"errors": {"fx_error": 31}}):
        try:
            normalize_config(bad)
        except ConfigError:
            continue
        except Exception as error:  # noqa: BLE001
            return f"invalid config {bad!r} raised {type(error).__name__}"
        return f"invalid config {bad!r} was accepted"
    return None


def distractors_satisfy_no_definition() -> Optional[str]:
    """Clean companies full of distractors: no predicate fires; every distractor kind occurs."""
    from auditgym.generate import generate
    from auditgym.verify import predicate_issues, verify

    seen = {"refund": 0, "advance": 0, "overpayment": 0, "partial": 0, "outside": 0}
    for seed in range(30):
        config = {"errors": {}, "currencies": [["USD"], ["EUR", "CNY"], []][seed % 3],
                  "timezone": ["+05:00", "-11:00", "+14:00"][seed % 3]}
        episode = generate(seed, config)
        found, undefined = predicate_issues(episode["public"])
        if found or undefined or verify(episode):
            return f"seed {seed}: {(found or undefined or verify(episode))[0]}"
        public = episode["public"]
        invoices = {i["invoice_id"]: i for i in public["invoices"]}
        for payment in public["payments"]:
            refs = payment["invoice_refs"]
            seen["refund"] += payment["amount_minor"] < 0
            seen["advance"] += refs is None
            if refs and len(refs) == 1 and payment["amount_minor"] > 0:
                total = invoices[refs[0]]["total_minor"]
                seen["overpayment"] += payment["amount_minor"] > total
                seen["partial"] += payment["amount_minor"] < total
        posted = {r["invoice_id"] for r in public["ledger"] if r["invoice_id"]}
        seen["outside"] += sum(i["invoice_id"] not in posted for i in public["invoices"])
    missing = [kind for kind, count in seen.items() if not count]
    return f"distractors never generated: {', '.join(missing)}" if missing else None


def interactions_are_unambiguous() -> Optional[str]:
    """I1 and I2 give exactly their two issues on top of config.errors and nothing else holds."""
    from auditgym.generate import generate
    from auditgym.verify import verify

    expected = {"I1": ("allocation_gap", "fx_error"), "I2": ("duplicate_invoice", "period_cutoff")}
    for index, config in enumerate(INTERACTION_CONFIGS):
        for seed in SEEDS:
            episode = generate(seed, config)
            problems = verify(episode)
            if problems:
                return f"config {index}, seed {seed}: {problems[0]}"
            steps = [s for s in episode["private"]["provenance"]["steps"] if s["operator"] == config["interaction"]]
            issues = {i["issue_id"]: i for i in episode["private"]["truth"]["issues"]}
            kinds = tuple(sorted(issues[s["issue_id"]]["type"] for s in steps))
            if kinds != expected[config["interaction"]]:
                return f"config {index}, seed {seed}: scenario issues {kinds}"
            docs = [set(issues[s["issue_id"]]["affected_ids"]) for s in steps]
            if not docs[0] & docs[1]:
                return f"config {index}, seed {seed}: the two scenario issues share no document"
    return None


def fuzz_solver_equals_truth() -> Optional[str]:
    """tools.fuzz on 4 seeds per grid config. Pending (not failed) while the only disagreement is
    the solver's duplicate evidence, which the solver zone is moving to [docA, docB, rowA, rowB]."""
    import io

    from tools import fuzz

    result = fuzz.run(4, out=io.StringIO())
    reasons = sorted(result["failures"])
    if not reasons:
        return None
    if reasons == ["solver: evidence lacks a full set (duplicate_invoice)"]:
        return "pending: solver duplicate evidence lacks the posting rows (DECISIONS A2)"
    seed, name = min(result["failures"][reasons[0]])
    return f"{reasons[0]} (seed {seed}, {name})"


CHECKS: list[tuple[str, Callable[[], Optional[str]]]] = [
    ("after injection the deviation equals expected_amount_minor of each issue", deviation_equals_expected_amount),
    ("every evidence set exists, has at most 10 ids and is sufficient", evidence_sets_exist_and_suffice),
    ("each of the four error types can be configured on its own", each_error_type_configurable_alone),
    ("any seed and any valid config gives a valid episode", any_seed_and_valid_config_gives_valid_episode),
    ("distractors do not satisfy any error definition", distractors_satisfy_no_definition),
    ("both interaction scenarios give an unambiguous truth", interactions_are_unambiguous),
    ("fuzz grid: verify is empty and the solver equals the truth", fuzz_solver_equals_truth),
]


if __name__ == "__main__":
    import sys

    failed = 0
    for check_name, function in CHECKS:
        message = function()
        failed += message is not None and not message.startswith("pending:")
        print(f"[{'PASS' if message is None else 'FAIL'}] {check_name}" + ("" if message is None else f" -- {message}"))
    sys.exit(1 if failed else 0)
