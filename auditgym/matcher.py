"""One-to-one matching of normalized findings against the truth (CASE.md section 05).

Published rules (the case asks to publish normalization, tolerances and tie-breaking):

1. A truth issue is identified by the key (type, set of affected_ids). Affected ids are
   canonical, so a key points to at most one issue; a truth with two issues on one key is
   rejected as a generator bug.
2. A finding is a TRUE POSITIVE for the issue with the same key when all hold:
   - the currency string is equal;
   - |claimed_amount_minor - expected_amount_minor| <= policy.amount_tolerance_minor[type]
     (missing entry means 0);
   - every evidence id exists in the open records of the episode (row_id, doc_id,
     invoice_id, payment_id, entity_id, or an FX key "YYYY-MM-DD|SRC|TGT");
   - the evidence ids contain at least one complete acceptable evidence set. If
     policy.evidence_superset_allowed is false, they must equal one of the sets.
3. Findings are processed in submission order. The first finding that satisfies rule 2
   takes the issue; later findings with the same key are false positives. Because a key
   names one issue, this order-based choice never lowers the number of true positives.
4. Every finding that is not a true positive is a false positive; every truth issue left
   without a finding is a false negative.

`reasons` is a judge-side diagnostic (one code per finding). It must never reach the agent.
"""
from __future__ import annotations

from typing import Any, Iterable

TP = "tp"
NO_SUCH_ISSUE = "no_such_issue"
ALREADY_MATCHED = "already_matched"
WRONG_CURRENCY = "wrong_currency"
AMOUNT_OUT_OF_TOLERANCE = "amount_out_of_tolerance"
UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _values(records: Iterable[dict[str, Any]], *fields: str) -> set[str]:
    found = set()
    for record in records:
        for field in fields:
            value = record.get(field)
            if isinstance(value, str) and value:
                found.add(value)
    return found


def known_ids(public: dict[str, Any]) -> set[str]:
    """Every id an agent can see in the open records."""
    ids = _values(public.get("ledger", []), "row_id", "entity_id", "invoice_id", "payment_id")
    ids |= _values(public.get("invoices", []), "doc_id", "invoice_id", "entity_id")
    ids |= _values(public.get("payments", []), "doc_id", "payment_id")
    for rate in public.get("fx", []):
        ids.add(f"{rate['date']}|{rate['source_currency']}|{rate['target_currency']}")
    return ids


def _issue_key(issue_type: str, affected_ids: Iterable[str]) -> tuple[str, frozenset[str]]:
    return issue_type, frozenset(affected_ids)


def _evidence_ok(evidence: frozenset[str], sets: list[list[str]], superset_allowed: bool) -> bool:
    for accepted in sets:
        accepted_set = frozenset(accepted)
        if (accepted_set <= evidence) if superset_allowed else (accepted_set == evidence):
            return True
    return False


def match(truth: dict[str, Any], findings: list[dict[str, Any]], public: dict[str, Any]) -> dict[str, Any]:
    """Return {"tp", "fp", "fn", "pairs": [[finding_index, issue_id]], "reasons": [code per finding]}."""
    policy = public.get("policy") or {}
    tolerances = policy.get("amount_tolerance_minor") or {}
    superset_allowed = policy.get("evidence_superset_allowed", True) is not False
    existing = known_ids(public)

    issues: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
    for issue in truth.get("issues", []):
        key = _issue_key(issue["type"], issue["affected_ids"])
        if key in issues:
            raise ValueError("truth has two issues with the same type and affected_ids")
        issues[key] = issue

    taken: set[str] = set()
    pairs: list[list[Any]] = []
    reasons: list[str] = []
    for index, finding in enumerate(findings):
        issue = issues.get(_issue_key(finding["type"], finding["affected_ids"]))
        evidence = frozenset(finding["evidence_ids"])
        if issue is None:
            reason = NO_SUCH_ISSUE
        elif issue["issue_id"] in taken:
            reason = ALREADY_MATCHED
        elif finding["currency"] != issue["currency"]:
            reason = WRONG_CURRENCY
        elif abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > int(tolerances.get(issue["type"], 0)):
            reason = AMOUNT_OUT_OF_TOLERANCE
        elif not evidence <= existing:
            reason = UNKNOWN_EVIDENCE_ID
        elif not _evidence_ok(evidence, issue["evidence_sets"], superset_allowed):
            reason = INSUFFICIENT_EVIDENCE
        else:
            reason = TP
            taken.add(issue["issue_id"])
            pairs.append([index, issue["issue_id"]])
        reasons.append(reason)

    tp = len(pairs)
    return {"tp": tp, "fp": len(findings) - tp, "fn": len(issues) - tp, "pairs": pairs, "reasons": reasons}
