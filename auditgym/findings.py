"""Structure check and normalization of the agent's findings (CASE.md section 05, CONTRACTS.md section 6).

A payload either passes and comes back as a list of normalized findings, or raises
`StructureError`, which the judge turns into reward -1. Wrong values that are still well
formed (unknown id, wrong amount, wrong currency, missing evidence) are NOT structure errors:
the matcher counts them as false positives.

Published rules:
- The payload is an object with `episode_id` (must equal the episode) and `findings` (a list).
- At most 30 findings. Each finding is an object with all six fields:
  `type` (one of the four error types), `affected_ids` (list of strings),
  `claimed_amount_minor` (a non-negative int; bool, float and numeric strings are rejected),
  `currency` (string), `evidence_ids` (list of at most 10 strings, counted as sent),
  `explanation` (string of at most 1 000 code points).
- The whole payload, serialized as compact UTF-8 JSON, is at most 64 KiB. NaN and infinity
  are not JSON and are rejected.
- Unknown keys are ignored. Repeated ids inside `affected_ids` and `evidence_ids` collapse;
  both lists come back sorted, because they are compared as sets.
"""
from __future__ import annotations

import json
from typing import Any

ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CHARS = 1000
MAX_PAYLOAD_BYTES = 64 * 1024
FINDING_FIELDS = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")


class StructureError(ValueError):
    """The findings payload is malformed. `code` is a short fixed identifier for the judge log."""

    def __init__(self, code: str, index: int | None = None) -> None:
        self.code = code
        self.index = index
        super().__init__(code if index is None else f"{code} (finding {index})")


def _payload_size(payload: Any) -> int:
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise StructureError("not_json") from None
    return len(text.encode("utf-8"))


def _id_list(value: Any, code: str, index: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise StructureError(code, index)
    return sorted(set(value))


def _finding(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise StructureError("finding_not_object", index)
    for field in FINDING_FIELDS:
        if field not in raw:
            raise StructureError("missing_field", index)

    if not isinstance(raw["type"], str) or raw["type"] not in ERROR_TYPES:
        raise StructureError("unknown_type", index)

    amount = raw["claimed_amount_minor"]
    # bool is a subclass of int in Python, so it is excluded explicitly.
    if type(amount) is not int:
        raise StructureError("amount_not_int", index)
    if amount < 0:
        raise StructureError("amount_negative", index)

    if not isinstance(raw["currency"], str):
        raise StructureError("currency_not_string", index)

    evidence = raw["evidence_ids"]
    if isinstance(evidence, list) and len(evidence) > MAX_EVIDENCE_IDS:
        raise StructureError("too_many_evidence_ids", index)

    explanation = raw["explanation"]
    if not isinstance(explanation, str):
        raise StructureError("explanation_not_string", index)
    if len(explanation) > MAX_EXPLANATION_CHARS:
        raise StructureError("explanation_too_long", index)

    return {
        "type": raw["type"],
        "affected_ids": _id_list(raw["affected_ids"], "bad_affected_ids", index),
        "claimed_amount_minor": amount,
        "currency": raw["currency"],
        "evidence_ids": _id_list(evidence, "bad_evidence_ids", index),
        "explanation": explanation,
    }


def parse_findings(payload: Any, episode_id: str) -> list[dict[str, Any]]:
    """Validate a findings payload and return normalized findings in submission order."""
    if not isinstance(payload, dict):
        raise StructureError("not_object")
    if _payload_size(payload) > MAX_PAYLOAD_BYTES:
        raise StructureError("payload_too_large")
    if "episode_id" not in payload or "findings" not in payload:
        raise StructureError("missing_field")
    if not isinstance(payload["episode_id"], str) or payload["episode_id"] != episode_id:
        raise StructureError("wrong_episode_id")
    findings = payload["findings"]
    if not isinstance(findings, list):
        raise StructureError("findings_not_list")
    if len(findings) > MAX_FINDINGS:
        raise StructureError("too_many_findings")
    return [_finding(raw, index) for index, raw in enumerate(findings)]
