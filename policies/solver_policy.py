"""The honest strong baseline: read all four tables within the step budget, run the independent
solver on what was read, submit. Uses only list_records and submit.
"""
from __future__ import annotations

from auditgym.solver import solve

from .base import MAX_EVIDENCE, ReadThenSubmit, postings


def _complete_duplicate_evidence(found: dict, public: dict, by_invoice: dict) -> None:
    """Contract v2 (DECISIONS A2) wants one own posting row per invoice of a duplicate. If the
    solver cites documents only, add the first debit row of each invoice's own group."""
    ledger_ids = {row["row_id"] for row in public["ledger"]}
    if any(item in ledger_ids for item in found["evidence_ids"]):
        return
    invoice_of = {inv["doc_id"]: inv["invoice_id"] for inv in public["invoices"]}
    for doc_id in found["affected_ids"]:
        rows = sorted(by_invoice.get(invoice_of.get(doc_id), []),
                      key=lambda row: (-row["debit_minor"], row["account"], row["row_id"]))
        if rows and len(found["evidence_ids"]) < MAX_EVIDENCE:
            found["evidence_ids"].append(rows[0]["row_id"])


class SolverPolicy(ReadThenSubmit):
    name = "solver_policy"

    def decide(self, public: dict) -> list[dict]:
        # solve() may return the same list object for affected_ids and evidence_ids: copy both.
        findings = [dict(found, affected_ids=list(found["affected_ids"]), evidence_ids=list(found["evidence_ids"]))
                    for found in solve(public)]
        by_invoice, _ = postings(public["ledger"])
        for found in findings:
            if found["type"] == "duplicate_invoice":
                _complete_duplicate_evidence(found, public, by_invoice)
        return findings[:30]
