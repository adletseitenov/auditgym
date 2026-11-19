"""Generator checks for tools/score.py, criterion "Данные и эталон".

CHECKS = [(name, function)]; each function returns None on success or a failure message.
They call auditgym.verify, which reads the open records, pre_ledger and the recorded steps only.
"""
from __future__ import annotations

from typing import Callable, Optional

SEEDS = range(40)
ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
CONFIGS = [
    {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}, "currencies": ["USD"]},
    {"errors": {"duplicate_invoice": 3, "period_cutoff": 3}, "currencies": ["USD", "EUR"], "timezone": "-11:00"},
    {"errors": {"duplicate_invoice": 2, "period_cutoff": 2}, "currencies": ["CNY"], "timezone": "+14:00",
     "size": {"invoices": [20, 30], "background_pairs": [0, 20]}},
]


def _first_problem(match: Callable[[str], bool], configs=CONFIGS) -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.verify import verify

    for index, config in enumerate(configs):
        for seed in SEEDS:
            problems = [p for p in verify(generate(seed, config)) if match(p)]
            if problems:
                return f"config {index}, seed {seed}: {problems[0]}"
    return None


def deviation_equals_expected_amount() -> Optional[str]:
    return _first_problem(lambda p: ": deviation:" in p)


def evidence_sets_exist_and_suffice() -> Optional[str]:
    return _first_problem(lambda p: ": evidence:" in p)


def each_error_type_configurable_alone() -> Optional[str]:
    from auditgym.config import ConfigError
    from auditgym.generate import generate
    from auditgym.verify import verify

    missing = []
    for name in ERROR_TYPES:
        for count in (1, 3):
            config = {"errors": {name: count}, "currencies": ["USD"]}
            try:
                episode = generate(7, config)
            except ConfigError as error:
                if error.code == "not_implemented":
                    missing.append(name)
                    break
                return f"{name} x{count}: {error.code}"
            issues = episode["private"]["truth"]["issues"]
            if len(issues) != count or any(issue["type"] != name for issue in issues):
                return f"{name} x{count}: truth has {len(issues)} issues of other types or counts"
            problems = verify(episode)
            if problems:
                return f"{name} x{count}: {problems[0]}"
    return f"not implemented yet: {', '.join(missing)}" if missing else None


def any_seed_and_valid_config_gives_valid_episode() -> Optional[str]:
    configs = CONFIGS + [
        {"errors": {"period_cutoff": 2}, "currencies": [], "size": {"invoices": [10, 10], "background_pairs": [0, 5]}},
        {"errors": {}, "currencies": ["RUB"], "base_currency": "USD"},
        {"errors": {"duplicate_invoice": 1}, "period": {"start": "2027-02-01", "months": 3}},
    ]
    return _first_problem(lambda p: True, configs)


CHECKS: list[tuple[str, Callable[[], Optional[str]]]] = [
    ("after injection the deviation equals expected_amount_minor of each issue", deviation_equals_expected_amount),
    ("every evidence set exists, has at most 10 ids and is sufficient", evidence_sets_exist_and_suffice),
    ("each of the four error types can be configured on its own", each_error_type_configurable_alone),
    ("any seed and any valid config gives a valid episode", any_seed_and_valid_config_gives_valid_episode),
]


if __name__ == "__main__":
    import sys

    failed = 0
    for check_name, function in CHECKS:
        message = function()
        failed += message is not None
        print(f"[{'PASS' if message is None else 'FAIL'}] {check_name}" + ("" if message is None else f" -- {message}"))
    sys.exit(1 if failed else 0)
