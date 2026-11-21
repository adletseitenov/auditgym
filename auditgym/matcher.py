"""One-to-one matching of normalized findings to truth issues (CASE §05, SPEC «Сопоставление»).

A finding is a TP for an issue iff every predicate below holds:
  same_type, same_affected_ids (as sets), same_currency, amount_within_tolerance,
  evidence_exists (every evidence id is a ledger row_id, an invoice or payment doc_id, or an
  fx key "date|source|target"), evidence_covers (contains a full allowed evidence set; extra
  ids only if policy["evidence_superset_allowed"], and then at most MAX_EVIDENCE_EXTRA of them
  and none of them an fx key, so a finding cannot hedge over neighbouring rates or pad with
  documents), and issue_available (not matched yet).

Choice under ambiguity: the result is a maximum one-to-one matching (augmenting paths,
Kuhn's algorithm). Findings are tried in submission order and each finding's candidate issues
in issue_id order, so the result is deterministic. TP/FP/FN never depend on the order of
findings, issues, records or ids, even if a truth ever had two issues with the same
(type, set of affected ids) (`truth_key`; verify should forbid it). Only `pairs` (which of two
identical findings scored) follows submission order: the first one keeps the issue, the
repeat is an FP. A finding whose `affected_ids` repeats an id matches no issue (an FP), even
though its set of ids would equal a truth set.

The predicates are module-level functions called through the module namespace so that the
mutation tests (tests/judge/test_mutations.py) can switch each one off.
"""
from __future__ import annotations

from .limits import MAX_EVIDENCE_EXTRA


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


def is_rate_key(evidence_id: str) -> bool:
    """An fx key is "date|source|target"; no ledger row_id or doc_id contains a bar."""
    return "|" in evidence_id


def tolerance(policy: dict, kind: str) -> int:
    value = policy.get("amount_tolerance_minor", {}).get(kind, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"bad amount tolerance for {kind}")
    return value


# --- predicates -------------------------------------------------------------------------

def same_type(finding: dict, issue: dict) -> bool:
    return finding["type"] == issue["type"]


def same_affected_ids(finding: dict, issue: dict) -> bool:
    ids = finding["affected_ids"]
    if len(set(ids)) != len(ids):  # a repeated id names one object twice: never a match, always an FP
        return False
    return set(ids) == set(issue["affected_ids"])


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
        if allowed == cited:
            return True
        if superset_allowed and allowed <= cited:
            extra = cited - allowed
            if len(extra) <= MAX_EVIDENCE_EXTRA and not any(is_rate_key(i) for i in extra):
                return True
    return False


def issue_available(issue_id: str, used: set) -> bool:
    return issue_id not in used


def superset_allowed(policy: dict) -> bool:
    """Same default as the task text the agent receives (auditgym/actions.py): allowed unless false."""
    return policy.get("evidence_superset_allowed", True) is not False


def is_match(finding: dict, issue: dict, policy: dict, known: frozenset) -> bool:
    return (same_type(finding, issue)
            and same_affected_ids(finding, issue)
            and same_currency(finding, issue)
            and amount_within_tolerance(finding, issue, tolerance(policy, issue["type"]))
            and evidence_exists(finding, known)
            and evidence_covers(finding, issue, superset_allowed(policy)))


# --- matching ---------------------------------------------------------------------------

def match(truth: dict, findings: list, public: dict) -> dict:
    """{"tp", "fp", "fn", "pairs": [[finding_index, issue_id], ...]} for normalized findings."""
    policy = public.get("policy", {})
    known = known_ids(public)
    issues = sorted(truth.get("issues", []), key=lambda issue: issue["issue_id"])
    candidates = [[issue["issue_id"] for issue in issues if is_match(finding, issue, policy, known)]
                  for finding in findings]
    owner: dict = {}  # issue_id -> finding index
    used: set = set()

    def augment(index: int, visited: set) -> bool:
        for issue_id in candidates[index]:
            if issue_id in visited:
                continue
            visited.add(issue_id)
            if issue_available(issue_id, used) or augment(owner[issue_id], visited):
                owner[issue_id] = index
                used.add(issue_id)
                return True
        return False

    tp = sum(1 for index in range(len(findings)) if augment(index, set()))
    pairs = sorted([[index, issue_id] for issue_id, index in owner.items()])
    return {"tp": tp, "fp": len(findings) - tp, "fn": len(issues) - tp, "pairs": pairs}
