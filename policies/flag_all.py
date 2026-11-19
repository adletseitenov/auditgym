"""The flag-everything attack: read invoices, payments and the ledger, then accuse up to 30
documents with amounts and evidence taken from the records.

Variants:
- "cutoff": every invoice as period_cutoff (document amount and currency, the document and its
  own posting rows as evidence);
- "duplicates": every pair of invoices of the same customer as duplicate_invoice;
- "mixed" (default): round robin over all four types — same-customer pairs, every invoice as a
  cutoff, every foreign-currency invoice as an fx_error, every referenced payment as an
  allocation_gap.
It must score no better than silence: the reward is built to punish it.
"""
from __future__ import annotations

from .base import MAX_FINDINGS, ReadThenSubmit, finding, postings

VARIANTS = ("mixed", "cutoff", "duplicates")


class FlagAll(ReadThenSubmit):
    tables = ("invoices", "payments", "ledger")

    def __init__(self, variant: str = "mixed") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.name = "flag_all" if variant == "mixed" else f"flag_all_{variant}"

    def decide(self, public: dict) -> list[dict]:
        base = public["policy"].get("base_currency") or public["manifest"].get("base_currency", "")
        by_invoice, by_payment = postings(public["ledger"])
        invoices, payments = public["invoices"], public["payments"]
        doc_of = {inv["invoice_id"]: inv["doc_id"] for inv in invoices}

        def rows(invoice: dict) -> list[dict]:
            return by_invoice.get(invoice["invoice_id"], [])

        def ids(rows_: list[dict]) -> list[str]:
            return [row["row_id"] for row in rows_]

        duplicates, cutoffs, fx, gaps = [], [], [], []
        for i, first in enumerate(invoices):
            for second in invoices[i + 1:]:
                if first.get("entity_id") == second.get("entity_id") and len(duplicates) < MAX_FINDINGS:
                    duplicates.append(finding(
                        "duplicate_invoice", [first["doc_id"], second["doc_id"]], first["total_minor"],
                        first["currency"], [first["doc_id"], second["doc_id"], *ids(rows(first))[:1],
                                            *ids(rows(second))[:1]], "Flagged."))
        for invoice in invoices:
            cutoffs.append(finding("period_cutoff", [invoice["doc_id"]], invoice["total_minor"],
                                   invoice["currency"], [invoice["doc_id"], *ids(rows(invoice))], "Flagged."))
            if invoice.get("currency") != base:
                posted = max([row["debit_minor"] for row in rows(invoice)] or [invoice["total_minor"]])
                fx.append(finding("fx_error", [invoice["doc_id"]], posted, base,
                                  [invoice["doc_id"], *ids(rows(invoice))], "Flagged."))
        for payment in payments:
            refs = payment.get("invoice_refs") or []
            if refs and refs[0] in doc_of:
                own = by_payment.get(payment["payment_id"], [])
                amount = max([row["credit_minor"] for row in own] or [abs(payment["amount_minor"])])
                gaps.append(finding("allocation_gap", [payment["doc_id"], doc_of[refs[0]]], amount, base,
                                    [payment["doc_id"], doc_of[refs[0]], *ids(own)], "Flagged."))

        if self.variant == "cutoff":
            return cutoffs[:MAX_FINDINGS]
        if self.variant == "duplicates":
            return duplicates[:MAX_FINDINGS]
        mixed, pools = [], [duplicates, cutoffs, fx, gaps]
        position = 0
        while len(mixed) < MAX_FINDINGS and any(position < len(pool) for pool in pools):
            for pool in pools:
                if position < len(pool) and len(mixed) < MAX_FINDINGS:
                    mixed.append(pool[position])
            position += 1
        return mixed
