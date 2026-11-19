"""Shared helpers for tests/env."""
from __future__ import annotations

import json
import pathlib
import re

from auditgym.evaluator import Evaluator

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TINY = ROOT / "tests" / "fixtures" / "tiny"
CLEAN = {"errors": {}}
TABLES = ("ledger", "invoices", "payments", "fx")

# Words that only the evaluator side may know. Never in an observation, info or error.
PRIVATE_WORDS = ("truth", "provenance", "pre_ledger", "issue_id", "evidence_sets",
                 "expected_amount_minor", "id_map", "generator_version", "private", "seed",
                 "fixtures", "attempt", "operator")


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000

    def now(self):
        return self.t


def tiny(mode: str = "train", results_path=None, clock=None) -> Evaluator:
    evaluator = Evaluator(mode=mode, results_path=results_path, clock=clock or FakeClock())
    evaluator.load(TINY)
    return evaluator


def generated(seed: int = 3, mode: str = "train", results_path=None, clock=None) -> Evaluator:
    evaluator = Evaluator(mode=mode, results_path=results_path, clock=clock or FakeClock())
    evaluator.reset(seed, CLEAN)
    return evaluator


def _outer_lists(value, depth: int = 2):
    """The (observation, reward, ...) tuples travel as JSON arrays; only those outer levels
    may be tuples. Anything deeper must already be plain JSON."""
    if depth and isinstance(value, (tuple, list)):
        return [_outer_lists(item, depth - 1) for item in value]
    return value


def strict_json(value) -> str:
    """JSON text; fails on NaN, inner tuples, non-JSON objects and anything that does not round-trip."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    assert json.loads(text) == _outer_lists(value), "value does not round-trip through JSON"
    return text


def leaves(value, out=None) -> list:
    """Every dict key and every scalar leaf of a JSON value."""
    out = [] if out is None else out
    if isinstance(value, dict):
        for key, item in value.items():
            out.append(key)
            leaves(item, out)
    elif isinstance(value, list):
        for item in value:
            leaves(item, out)
    else:
        out.append(value)
    return out


def full_walk(evaluator: Evaluator, finish: bool = True) -> list:
    """Read every record of every table, touch every error path, then submit. Returns all
    returned values (observations, rewards, flags, infos) in order."""
    transcript = []

    def step(action):
        result = evaluator.step(action)
        transcript.append(result)
        return result

    first_ids = {}
    for table in TABLES:
        cursor = None
        while True:
            observation = step({"action": "list_records", "table": table, "cursor": cursor})[0]
            if observation["type"] != "records":
                return transcript
            if observation["records"] and table not in first_ids:
                first_ids[table] = observation["records"][0]
            cursor = observation["next_cursor"]
            if cursor is None:
                break
    if "invoices" in first_ids:
        step({"action": "get_document", "doc_id": first_ids["invoices"]["doc_id"]})
        step({"action": "get_record", "table": "invoices", "id": first_ids["invoices"]["invoice_id"]})
    step({"action": "get_record", "table": "ledger", "id": first_ids["ledger"]["row_id"]})
    for action in ({"action": "nope"}, {"action": "list_records", "table": "nope"},
                   {"action": "get_record", "table": "ledger", "id": "nope"},
                   {"action": "list_records", "table": "ledger", "cursor": "nope"},
                   {"action": "list_records", "table": "ledger", "limit": -1},
                   {"action": "calculator", "expression": "__import__('os')"},
                   {"action": "calculator", "expression": "125000 * 465.37"}):
        step(action)
    if finish:
        episode_id = evaluator._episode["public"]["manifest"]["episode_id"]
        step({"action": "submit", "findings": {"episode_id": episode_id, "findings": []}})
        step({"action": "list_records", "table": "ledger"})
    return transcript


def forbidden_values(episode: dict, extra: tuple = ()) -> set:
    """Exact values that must never appear as a key or leaf of anything a policy receives."""
    private = episode["private"]
    values = set(extra)
    values |= {issue["issue_id"] for issue in private["truth"].get("issues", [])}
    values |= set(private["provenance"].get("id_map", {}).keys())
    values |= set(private["provenance"].keys())
    public_rows = {row["row_id"] for row in episode["public"]["ledger"]}
    values |= {row["row_id"] for row in private["pre_ledger"] if row["row_id"] not in public_rows}
    values.discard(None)
    return values


def assert_no_leak(test, values: list, episode: dict, extra_values: tuple = (), extra_words: tuple = ()):
    forbidden = forbidden_values(episode, extra_values)
    for value in values:
        text = strict_json(value)
        for leaf in leaves(json.loads(text)):
            test.assertNotIn(leaf, forbidden, f"private value {leaf!r} reached the policy")
        lowered = text.lower()
        for word in PRIVATE_WORDS + tuple(extra_words):
            test.assertNotIn(word.lower(), lowered, f"private word {word!r} reached the policy")
        seed = episode["private"]["provenance"].get("seed")
        if isinstance(seed, int):
            test.assertIsNone(re.search(rf"(?<![0-9]){seed}(?![0-9])", text) if seed > 999 else None,
                              f"seed {seed} reached the policy")
