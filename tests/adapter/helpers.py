"""Fixtures C, E, L, U of the published checks (docs/CHECKS definitions, "Общие фикстуры") and small helpers.

C  a clean episode with an existing record R1 (a ledger row) and document D1 (an invoice);
E  one declared error with the full evidence set available;
L  an episode whose ledger has at least 101 rows (three pages of 50);
U  an id that is not in the open data of the fixture, chosen after reading the data.
All three are made by `generate_episode`, loaded by `reset` through episode_dir.
"""
from __future__ import annotations

import json
import pathlib
import tempfile
import warnings

import judge_adapter
from auditgym.storage import load_episode

SEED_C, SEED_E, SEED_L = 90017, 90029, 90031
SEED_VALUE = 7391056482  # a distinctive seed: its digits must never be seen by a policy
CONFIG_C = {"errors": {}}
CONFIG_E = {"errors": {"duplicate_invoice": 1}}
CONFIG_L = {"size": {"invoices": [60, 80], "background_pairs": [150, 200]}}

_scratch = None
_cache: dict = {}


def generate_quietly(seed, config, directory, **kwargs) -> None:
    """judge_adapter.generate_episode for fixtures: the no-secret RuntimeWarning is meant for the jury's
    hidden runs, not for test output (the warning itself is tested in test_jury9)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="generate_episode: no secret", category=RuntimeWarning)
        judge_adapter.generate_episode(seed, config, directory, **kwargs)


def scratch() -> pathlib.Path:
    global _scratch
    if _scratch is None:
        _scratch = tempfile.TemporaryDirectory()
    return pathlib.Path(_scratch.name)


def cleanup() -> None:
    global _scratch
    _cache.clear()
    if _scratch is not None:
        _scratch.cleanup()
        _scratch = None


def fixture(name: str) -> dict:
    """{"dir", "episode" (the loaded files: evaluator side), "R1", "D1", "U"}; built once per name."""
    if name not in _cache:
        seed, config = {"C": (SEED_C, CONFIG_C), "E": (SEED_E, CONFIG_E), "L": (SEED_L, CONFIG_L),
                        "S": (SEED_VALUE, CONFIG_E)}[name]
        directory = scratch() / name
        generate_quietly(seed, config, str(directory))
        episode = load_episode(directory)
        public = episode["public"]
        text = json.dumps([public[t] for t in ("ledger", "invoices", "payments", "fx")]) + json.dumps(public["manifest"])
        unknown = next(f"U{n}x" for n in range(10 ** 6) if f"U{n}x" not in text)
        _cache[name] = {"dir": str(directory), "episode": episode, "R1": public["ledger"][0]["row_id"],
                        "D1": public["invoices"][0]["doc_id"], "U": unknown}
    return _cache[name]


def closed(directory: str) -> dict:
    return {"mode": "closed", "episode_dir": directory}


def new_env(name: str = "journal"):
    """(env, journal path): a fresh adapter environment with its own evaluator journal."""
    path = scratch() / f"{name}-{len(list(scratch().glob('*.jsonl')))}.jsonl"
    return judge_adapter.make_env(journal_path=path), path


def journal(path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except OSError:
        return []


def finished(path) -> list[dict]:
    return [row for row in journal(path) if row.get("event") == "finished"]


def list_action(table: str = "ledger", cursor=None, limit: int = 50) -> dict:
    return {"name": "list_records", "args": {"table": table, "cursor": cursor, "limit": limit}}


def submit_action(findings: dict) -> dict:
    return {"name": "submit", "args": {"findings": findings}}


def empty_findings(directory_fixture: dict) -> dict:
    return {"episode_id": directory_fixture["episode"]["public"]["manifest"]["episode_id"], "findings": []}


def oracle_findings(directory_fixture: dict) -> dict:
    from policies.oracle import Oracle

    episode = directory_fixture["episode"]
    oracle = Oracle(episode["private"]["truth"])
    oracle.reset({"manifest": {"episode_id": episode["public"]["manifest"]["episode_id"]}})
    return oracle.act({})["findings"]


def is_json(value) -> bool:
    try:
        return json.loads(json.dumps(value, allow_nan=False)) == value
    except (TypeError, ValueError):
        return False
