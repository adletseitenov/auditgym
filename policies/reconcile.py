"""A deliberately simple reconciliation, the kind of script an auditor writes in ten minutes.

It reads all invoices and payments and only a sample of the ledger (the first `ledger_pages`
pages), never the fx table, and applies two rules:
- two invoices of the same customer with the same total are a duplicate (no check of currency,
  line items or whether both are posted);
- an invoice whose issue date (the date part of `issued_at`) is outside the manifest period is a
  period cutoff (payments are not checked).
Evidence is the documents plus the posting rows it happened to see in its ledger sample. It has
no fx reasoning and never looks for allocation gaps, so it finds some issues and misses others.
"""
from __future__ import annotations

from .base import ReadThenSubmit, finding, postings


class Reconcile(ReadThenSubmit):
    name = "reconcile"
    tables = ("invoices", "payments", "ledger")

    def __init__(self, ledger_pages: int = 6) -> None:
        self.max_pages = {"ledger": ledger_pages}

    def decide(self, public: dict) -> list[dict]:
        start = public["manifest"].get("period_start", "")[:10]
        end = public["manifest"].get("period_end", "")[:10]
        by_invoice, _ = postings(public["ledger"])

        def rows_of(invoice: dict) -> list[str]:
            return [row["row_id"] for row in by_invoice.get(invoice["invoice_id"], [])]

        findings = []
        seen: dict[tuple, dict] = {}
        for invoice in public["invoices"]:
            key = (invoice.get("entity_id"), invoice.get("total_minor"))
            first = seen.get(key)
            if first is None:
                seen[key] = invoice
                continue
            findings.append(finding(
                "duplicate_invoice", [first["doc_id"], invoice["doc_id"]], invoice["total_minor"],
                invoice["currency"], [first["doc_id"], invoice["doc_id"], *rows_of(first)[:1], *rows_of(invoice)[:1]],
                "Same customer and the same total."))
        for invoice in public["invoices"]:
            day = (invoice.get("issued_at") or "")[:10]
            if day and not start <= day < end:
                findings.append(finding(
                    "period_cutoff", [invoice["doc_id"]], invoice["total_minor"], invoice["currency"],
                    [invoice["doc_id"], *rows_of(invoice)], "Issued outside the reporting period."))
        return findings[:30]
