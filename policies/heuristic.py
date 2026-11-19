"""A cheap rule-of-thumb auditor, kept as a demonstration and not as a baseline of the check.

It reads every table and applies four one-line rules that look sensible on a first glance:
- duplicate_invoice: two invoices of one customer with the same total and currency (no look at line
  items or the service period);
- allocation_gap: every payment that has a row on `unallocated` is a gap of that amount, whatever
  the reason (an overpayment looks the same to it);
- period_cutoff: every document whose printed date lies outside the period while its postings lie
  inside it (the date as printed, no offset arithmetic);
- fx_error: a posted conversion that differs from amount x the reference rate of the printed date.
It ignores partial payments, positions, the company's offset for the rate date, and overpayments.
That is the point: the legitimate cases the environment plants as distractors look exactly like
errors to it, so it is right about half the time. It never changes the check score; it only
shows (docs/RESULTS.md) that the distractors do their job.
"""
from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

from .base import MAX_FINDINGS, ReadThenSubmit, finding, postings


def _day(stamp: str) -> str:
    return (stamp or "")[:10]


class Heuristic(ReadThenSubmit):
    name = "heuristic"

    def decide(self, public: dict) -> list[dict]:
        found = self.rules(public)
        # round robin over the four rules so that the 30-finding limit cuts every rule evenly
        queues = [found[name] for name in ("duplicate_invoice", "allocation_gap", "period_cutoff", "fx_error")]
        answer: list[dict] = []
        while any(queues) and len(answer) < MAX_FINDINGS:
            for queue in queues:
                if queue and len(answer) < MAX_FINDINGS:
                    answer.append(queue.pop(0))
        return answer

    def rules(self, public: dict) -> dict[str, list[dict]]:
        """Every finding of every rule, before the limit of one answer."""
        policy, manifest = public["policy"], public["manifest"]
        base = policy.get("base_currency") or manifest.get("base_currency", "")
        exponents = policy.get("currency_exponents") or {}
        accounts = policy.get("accounts") or {}
        by_invoice, by_payment = postings(public["ledger"])
        invoices, payments = public["invoices"], public["payments"]
        start, end = _day(manifest.get("period_start", "")), _day(manifest.get("period_end", ""))
        rates = {(row["date"], row["source_currency"]): Decimal(row["rate"]) for row in public["fx"]}
        doc_of = {invoice["invoice_id"]: invoice["doc_id"] for invoice in invoices}
        found: dict[str, list[dict]] = {"duplicate_invoice": [], "allocation_gap": [], "period_cutoff": [],
                                        "fx_error": []}

        seen: dict[tuple, dict] = {}
        for invoice in invoices:
            key = (invoice.get("entity_id"), invoice.get("currency"), invoice.get("total_minor"))
            first = seen.setdefault(key, invoice)
            if first is invoice:
                continue
            rows = [group[0]["row_id"] for group in (by_invoice.get(first["invoice_id"]),
                                                     by_invoice.get(invoice["invoice_id"])) if group]
            found["duplicate_invoice"].append(finding(
                "duplicate_invoice", [first["doc_id"], invoice["doc_id"]], invoice["total_minor"],
                invoice["currency"], [first["doc_id"], invoice["doc_id"], *rows], "Same customer, currency and total."))

        for payment in payments:
            refs = payment.get("invoice_refs") or []
            for row in by_payment.get(payment["payment_id"], []):
                if row["account"] == accounts.get("unallocated", "unallocated") and row["credit_minor"] > 0:
                    if len(refs) != 1 or refs[0] not in doc_of:
                        continue
                    receivable = accounts.get("receivable", "receivable")
                    owed = [r["row_id"] for r in by_invoice.get(refs[0], [])
                            if r["account"] == receivable and r["debit_minor"] > 0][:1]
                    paid = [r["row_id"] for r in by_payment[payment["payment_id"]]
                            if r["account"] == receivable and r["credit_minor"] > 0][:1]
                    found["allocation_gap"].append(finding(
                        "allocation_gap", [payment["doc_id"], doc_of[refs[0]]], row["credit_minor"], base,
                        [payment["doc_id"], doc_of[refs[0]], row["row_id"], *owed, *paid], "Unallocated remainder."))

        for kind, documents, groups, stamp_key, id_key in (
                ("invoice", invoices, by_invoice, "issued_at", "invoice_id"),
                ("payment", payments, by_payment, "timestamp", "payment_id")):
            for document in documents:
                day = _day(document.get(stamp_key))
                rows = groups.get(document[id_key], [])
                if not day or not start or not rows:
                    continue
                # printed date outside the period, but posted inside it (no offset arithmetic)
                outside = not start <= day < end
                posted = any(start <= _day(row["timestamp"]) < end for row in rows)
                if outside and posted:
                    debit = next((row for row in rows if row["debit_minor"] > 0), None)
                    credit = next((row for row in rows if row["credit_minor"] > 0), None)
                    amount = abs(document.get("total_minor", document.get("amount_minor", 0)))
                    found["period_cutoff"].append(finding(
                        "period_cutoff", [document["doc_id"]], amount, document["currency"],
                        [document["doc_id"], *[row["row_id"] for row in (debit, credit) if row]],
                        "Dated outside the period, posted inside it."))
                currency = document.get("currency")
                if currency == base or currency not in exponents or base not in exponents:
                    continue
                rate = rates.get((day, currency))
                if rate is None:
                    continue
                amount = document.get("total_minor", document.get("amount_minor", 0))
                scale = Decimal(10) ** (exponents[base] - exponents[currency])
                expected = int((Decimal(abs(amount)) * scale * rate).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
                money = [row for row in rows if row["account"] == (accounts.get("receivable", "receivable")
                                                                    if kind == "invoice" else accounts.get("cash", "cash"))]
                for row in money[:1]:
                    booked = row["debit_minor"] or row["credit_minor"]
                    if abs(booked - expected) > 1:
                        found["fx_error"].append(finding(
                            "fx_error", [document["doc_id"]], abs(booked - expected), base,
                            [document["doc_id"], row["row_id"], f"{day}|{currency}|{base}"],
                            "Booked at another rate than the reference one."))
        return found
