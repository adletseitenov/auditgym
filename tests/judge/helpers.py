"""Hand-built episodes for the judge tests (no generator involved)."""
from __future__ import annotations

TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
EP = "ep_hand0001"
POLICY = {"base_currency": "KZT",
          "amount_tolerance_minor": {"duplicate_invoice": 0, "allocation_gap": 0,
                                     "fx_error": 1, "period_cutoff": 0},
          "evidence_superset_allowed": True}


def _row(row_id):
    return {"row_id": row_id, "timestamp": "2026-10-10T10:00:00+05:00", "account": "cash",
            "debit_minor": 1, "credit_minor": 0, "currency": "KZT", "entity_id": "e",
            "invoice_id": None, "payment_id": None}


def _doc(doc_id):
    return {"doc_id": doc_id, "invoice_id": "inv" + doc_id, "entity_id": "e",
            "issued_at": "2026-10-03T10:00:00+05:00", "due_at": "2026-11-02T10:00:00+05:00",
            "currency": "KZT", "total_minor": 1000, "line_items": []}


def episode(n_issues: int, n_filler_docs: int = 10, superset_allowed: bool = True) -> dict:
    """n_issues issues cycling over the four types; each has its own two docs and one row."""
    issues, ledger, invoices = [], [], []
    for k in range(n_issues):
        a, b, r = f"a{k:02d}", f"b{k:02d}", f"r{k:02d}"
        invoices += [_doc(a), _doc(b)]
        ledger.append(_row(r))
        issues.append({"issue_id": f"t{k:02d}", "type": TYPES[k % 4], "affected_ids": [a, b],
                       "expected_amount_minor": 1000 * (k + 1), "currency": "KZT",
                       "evidence_sets": [[a, b, r]]})
    invoices += [_doc(f"f{k:02d}") for k in range(n_filler_docs)]
    ledger += [_row(f"x{k:02d}") for k in range(5)]
    policy = dict(POLICY, evidence_superset_allowed=superset_allowed)
    public = {"manifest": {"episode_id": EP}, "policy": policy, "ledger": ledger,
              "invoices": invoices, "payments": [], "fx": []}
    return {"public": public, "private": {"truth": {"episode_id": EP, "issues": issues}}}


def correct(issue: dict) -> dict:
    return {"type": issue["type"], "affected_ids": list(issue["affected_ids"]),
            "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
            "evidence_ids": list(issue["evidence_sets"][0]), "explanation": "ok"}


def oracle(ep: dict) -> list[dict]:
    return [correct(issue) for issue in ep["private"]["truth"]["issues"]]


def false_accusation(k: int) -> dict:
    return {"type": TYPES[k % 4], "affected_ids": [f"f{k % 10:02d}"], "claimed_amount_minor": 1000,
            "currency": "KZT", "evidence_ids": [f"f{k % 10:02d}"], "explanation": "looks odd"}


def submit(ep: dict, findings: list) -> dict:
    return {"episode_id": ep["public"]["manifest"]["episode_id"], "findings": findings}
