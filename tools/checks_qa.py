"""qa checks for tools/score.py. Each returns None on success or a failure message.

CHECKS is a list of (name, function); the score owner wires the module into ZONE_CHECKS.
"""
from __future__ import annotations

import tempfile
import time
from typing import Callable, Optional

from tools import episodes as episodes_tool

TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
LARGEST_BASE = {"size": {"invoices": [200, 200], "background_pairs": [0, 0]},
                "currencies": ["USD", "EUR", "CNY"], "period": {"start": "2026-10-01", "months": 3}}


def largest_config() -> dict | None:
    """The biggest config the generator builds: most invoices, issues and background postings."""
    from auditgym.config import ConfigError, normalize_config

    supported = [t for t in TYPES if episodes_tool.unsupported({"errors": {t: 1}, "currencies": ["USD"]}) is None]
    if not supported:
        return None

    def config(total: int, pairs: int) -> dict:
        errors = {t: total // len(supported) + (1 if k < total % len(supported) else 0) for k, t in enumerate(supported)}
        return dict(LARGEST_BASE, errors=errors, size={"invoices": [200, 200], "background_pairs": [pairs, pairs]})

    def valid(candidate: dict) -> bool:
        try:
            normalize_config(candidate)
            return True
        except ConfigError:
            return False

    for total in range(30, -1, -1):
        if not valid(config(total, 0)):
            continue
        low, high = 0, 900  # largest valid background_pairs by bisection
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if valid(config(total, middle)) else (low, middle - 1)
        candidate = config(total, low)
        if episodes_tool.unsupported(candidate) is None:
            return candidate
    return None


def scripted_policy_solves_largest_episode() -> Optional[str]:
    from auditgym import Env
    from policies.runner import make_policy, run_episode

    config = largest_config()
    if config is None:
        return "no supported config"
    with Env(mode="train") as env:
        started = time.perf_counter()
        result = run_episode(env, make_policy("solver_policy"), 0, config)
        seconds = time.perf_counter() - started
    if result["truncated"] or result["steps"] > 100:
        return f"truncated after {result['steps']} steps"
    if seconds > 60:
        return f"took {seconds:.1f} s"
    if result["info"].get("f1") != 1.0:
        return f"F1 {result['info'].get('f1')} on the largest episode"
    return None


def policy_reward_ordering() -> Optional[str]:
    from tools.report import generated_cases, ordering_problems, run_generated

    names = ("oracle", "solver_policy", "reconcile", "silence", "flag_all", "flag_all_cutoff", "flag_all_duplicates")
    problems = ordering_problems(run_generated(names, generated_cases(16)))
    return problems[0] if problems else None


def episodes_rebuild_byte_identically() -> Optional[str]:
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        episodes_tool.build(first, quiet=True)
        episodes_tool.build(second, quiet=True)
        a, b = episodes_tool.tree_bytes(first), episodes_tool.tree_bytes(second)
    if not a:
        return "no files written"
    differ = sorted(set(a) ^ set(b)) or [name for name in sorted(a) if a[name] != b[name]]
    return f"rebuild differs: {differ[0]}" if differ else None


CHECKS: list[tuple[str, Callable[[], Optional[str]]]] = [
    ("scripted policy solves the largest episode within 100 steps and 60 seconds",
     scripted_policy_solves_largest_episode),
    ("baseline policies keep the reward order: oracle >= solver > reconcile > flag-all, flag-all near silence",
     policy_reward_ordering),
    ("open train and dev episodes rebuild byte-identically", episodes_rebuild_byte_identically),
]


if __name__ == "__main__":
    import sys

    failed = 0
    for name, function in CHECKS:
        message = function()
        failed += message is not None
        print(f"[{'OK' if message is None else 'FAIL'}] {name}" + ("" if message is None else f" -- {message}"))
    sys.exit(1 if failed else 0)
