"""Loads tests/fixtures/tiny into the in-memory episode dict of CONTRACTS.md section 1.

A local reader for the judge tests, so they do not wait for auditgym/storage.py (zone gen).
"""
from __future__ import annotations

import copy
import csv
import json
import pathlib
from typing import Any

TINY = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "tiny"
MONEY = ("debit_minor", "credit_minor")


def _csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in MONEY:
                row[key] = int(value)
            elif value == "":
                row[key] = None
    return rows


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_CACHE: dict[str, Any] = {}


def tiny() -> dict[str, Any]:
    """A fresh deep copy of the tiny episode (tests may mutate it)."""
    if not _CACHE:
        public, private = TINY / "public", TINY / "private"
        _CACHE["episode"] = {
            "public": {
                "manifest": _json(public / "manifest.json"),
                "policy": _json(public / "policy.json"),
                "ledger": _csv(public / "ledger.csv"),
                "invoices": _jsonl(public / "invoices.jsonl"),
                "payments": _jsonl(public / "payments.jsonl"),
                "fx": _csv(public / "fx.csv"),
            },
            "private": {"truth": _json(private / "truth.json")},
        }
    return copy.deepcopy(_CACHE["episode"])


def clean_tiny() -> dict[str, Any]:
    """The same books declared clean: an empty truth (for the 'silence on clean books' cases)."""
    episode = tiny()
    episode["private"]["truth"]["issues"] = []
    return episode


EPISODE_ID = "ep_tiny0001"
DUP_A, DUP_B, OTHER_INVOICE = "k7m2qa", "z5c1ty", "b9f6ge"


def correct_finding(**changes: Any) -> dict[str, Any]:
    finding = {"type": "duplicate_invoice", "affected_ids": [DUP_A, DUP_B],
               "claimed_amount_minor": 5000000, "currency": "KZT",
               "evidence_ids": [DUP_A, DUP_B], "explanation": "Same client, items and total."}
    finding.update(changes)
    return finding


def payload(*findings: dict[str, Any], episode_id: str = EPISODE_ID) -> dict[str, Any]:
    return {"episode_id": episode_id, "findings": list(findings)}
