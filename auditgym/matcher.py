"""One-to-one matching of normalized findings to truth issues (CASE §05, SPEC «Сопоставление»).

A finding is a TP for an issue iff every predicate below holds:
  same_type, same_affected_ids (as sets), same_currency, amount_within_tolerance,
  evidence_exists (every evidence id is a ledger row_id, an invoice or payment doc_id, or an
  fx key "date|source|target"), evidence_covers (contains a full allowed evidence set; extra
  ids only if policy["evidence_superset_allowed"]), and issue_available (not matched yet).

Choice under ambiguity: findings are taken in submission order and issues in issue_id order;
a finding takes the first available issue that satisfies all predicates. With canonical
affected_ids two issues never share (type, set of affected ids) (`truth_key`), so a finding has
at most one candidate; the greedy pass is then a maximum matching and TP/FP/FN do not depend
on the order of findings, issues, records or ids. Only `pairs` (which of two identical findings
scored) follows submission order: the first one wins, the repeat is an FP.

The predicates are module-level functions called through the module namespace so that the
mutation tests (tests/judge/test_mutations.py) can switch each one off.
"""
from __future__ import annotations


def truth_key(issue: dict) -> tuple:
    return (issue["type"], frozenset(issue["affected_ids"]))


def known_ids(public: dict) -> frozenset:
    """Every id an evidence list may cite. The table decides, not the shape of the id."""
    ids = set()
    ids.update(row["row_id"] for row in public.get("ledger", ()))
    ids.update(doc["doc_id"] for doc in public.get("invoices", ()))
    ids.update(doc["doc_id"] for doc in public.get("payments", ()))
    ids.update(f"{r['date']}|{r['source_currency']}|{r['target_currency']}"
               for r in public.get("fx", ()))
    return frozenset(ids)


def tolerance(policy: dict, kind: str) -> int:
    value = policy.get("amount_tolerance_minor", {}).get(kind, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"bad amount tolerance for {kind}")
    return value


# --- predicates -------------------------------------------------------------------------

def same_type(finding: dict, issue: dict) -> bool:
    return finding["type"] == issue["type"]


def same_affected_ids(finding: dict, issue: dict) -> bool:
    return set(finding["affected_ids"]) == set(issue["affected_ids"])


def same_currency(finding: dict, issue: dict) -> bool:
    return finding["currency"] == issue["currency"]


def amount_within_tolerance(finding: dict, issue: dict, allowed: int) -> bool:
    return abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) <= allowed


def evidence_exists(finding: dict, known: frozenset) -> bool:
    return all(evidence_id in known for evidence_id in finding["evidence_ids"])


def evidence_covers(finding: dict, issue: dict, superset_allowed: bool) -> bool:
    cited = set(finding["evidence_ids"])
    for allowed in issue["evidence_sets"]:
        allowed = set(allowed)
        if not allowed:
            continue  # an empty set would accept any evidence; never valid
        if allowed == cited or (superset_allowed and allowed <= cited):
            return True
    return False


def issue_available(issue_id: str, used: set) -> bool:
    return issue_id not in used


def is_match(finding: dict, issue: dict, policy: dict, known: frozenset) -> bool:
    return (same_type(finding, issue)
            and same_affected_ids(finding, issue)
            and same_currency(finding, issue)
            and amount_within_tolerance(finding, issue, tolerance(policy, issue["type"]))
            and evidence_exists(finding, known)
            and evidence_covers(finding, issue, policy.get("evidence_superset_allowed") is True))


# --- matching ---------------------------------------------------------------------------

def match(truth: dict, findings: list, public: dict) -> dict:
    """{"tp", "fp", "fn", "pairs": [[finding_index, issue_id], ...]} for normalized findings."""
    policy = public.get("policy", {})
    known = known_ids(public)
    issues = sorted(truth.get("issues", []), key=lambda issue: issue["issue_id"])
    used: set = set()
    pairs = []
    for index, finding in enumerate(findings):
        for issue in issues:
            if issue_available(issue["issue_id"], used) and is_match(finding, issue, policy, known):
                used.add(issue["issue_id"])
                pairs.append([index, issue["issue_id"]])
                break
    tp = len(pairs)
    return {"tp": tp, "fp": len(findings) - tp, "fn": len(issues) - tp, "pairs": pairs}
