"""One-to-one matching of normalized findings to truth issues (CASE §05; official definitions of
20.11, docs/DECISIONS.md R).

A finding is a TP for an issue iff every predicate below holds:
  same_type (exact string), same_affected_ids (as SETS: order and repeats of an id inside the list
  do not matter), same_currency (exact string), amount_within_tolerance
  (|claimed - expected| <= tau, tau = policy["amount_tolerance_minor"][type], default 0),
  evidence_exists (every cited id is a ledger row_id, an invoice or payment doc_id, or an fx key
  "date|source|target"; one non-existent id makes the finding unmatchable), evidence_covers (the
  cited SET contains a full allowed evidence set; any further EXISTING ids are allowed while
  policy["evidence_superset_allowed"] is not false), and issue_available (not matched yet).

Choice under ambiguity: a maximum one-to-one matching (augmenting paths, Kuhn's algorithm), and
among maximum matchings the official deterministic one: truth issue_ids sorted by code points, the
lexicographically smallest sorted sequence of (issue_id, finding index) pairs. It is built greedily:
issues in order, for each the smallest finding index that still lets the matching reach the maximum
size. A repeated finding stays a separate vertex: it is an FP when the issue is already taken.
TP/FP/FN never depend on the order of findings; only `pairs` (which identical finding scored) does.

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

def _max_matching(candidates: list, allowed_issues: frozenset, blocked: frozenset) -> int:
    """Size of a maximum matching between findings (not in `blocked`) and `allowed_issues`.

    `candidates[index]` lists the issue ids a finding matches. Kuhn's algorithm from the findings.
    """
    owner: dict = {}  # issue_id -> finding index
    used: set = set()

    def augment(index: int, visited: set) -> bool:
        for issue_id in candidates[index]:
            if issue_id not in allowed_issues or issue_id in visited:
                continue
            visited.add(issue_id)
            if issue_available(issue_id, used) or augment(owner[issue_id], visited):
                owner[issue_id] = index
                used.add(issue_id)
                return True
        return False

    return sum(1 for index in range(len(candidates)) if index not in blocked and augment(index, set()))


def match(truth: dict, findings: list, public: dict) -> dict:
    """{"tp", "fp", "fn", "pairs": [[finding_index, issue_id], ...]} for normalized findings."""
    policy = public.get("policy", {})
    known = known_ids(public)
    issues = sorted(truth.get("issues", []), key=lambda issue: issue["issue_id"])
    issue_ids = [issue["issue_id"] for issue in issues]
    candidates = [[issue["issue_id"] for issue in issues if is_match(finding, issue, policy, known)]
                  for finding in findings]
    best = _max_matching(candidates, frozenset(issue_ids), frozenset())
    taken: set = set()  # finding indexes already chosen
    pairs: list = []
    for position, issue_id in enumerate(issue_ids):
        if len(pairs) == best:
            break
        later = frozenset(issue_ids[position + 1:])
        for index in range(len(findings)):
            if index in taken or issue_id not in candidates[index]:
                continue
            if len(pairs) + 1 + _max_matching(candidates, later, frozenset(taken | {index})) == best:
                taken.add(index)
                pairs.append([index, issue_id])
                break
    tp = len(pairs)
    return {"tp": tp, "fp": len(findings) - tp, "fn": len(issues) - tp, "pairs": sorted(pairs)}
