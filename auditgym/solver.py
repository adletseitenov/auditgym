"""Independent solver: findings from the open records only (docs/SPEC.md, "Определения ошибок").

solve(public) evaluates the four published predicates over the final open records and returns
findings in the shape of docs/CONTRACTS.md section 6. It never reads `private` and does not
import the generator or the error operators.

Readings chosen where the definitions leave a case open (listed for the definition owner):
- duplicate_invoice fires for every pair of posted invoices with the same signature, so three
  identical posted invoices give three findings; an unposted copy gives none.
- allocation_gap needs exactly one unallocated row with a positive credit for the payment;
  a payment with several such rows is left unreported.
- fx_error is evaluated only for foreign-currency documents dated inside the period (an
  out-of-period posting is already a period_cutoff and its "correct" value is undefined) and
  only when the policy has a rate for the document date.
- period_cutoff reports the absolute document amount, so a refund never yields a negative claim.
"""
from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal

from .money import convert, fx_key, in_period, local_date

MAX_EVIDENCE = 10


def _canonical_items(line_items: list) -> tuple:
    """Line items as a multiset: every field of every item, order of items ignored."""
    return tuple(sorted(json.dumps(item, sort_keys=True, ensure_ascii=False) for item in line_items or []))


def _signature(invoice: dict) -> tuple:
    return (invoice.get("entity_id"), invoice.get("currency"), invoice.get("total_minor"),
            _canonical_items(invoice.get("line_items")))


def _row_order(row: dict) -> tuple:
    """Content first, id last: the choice of rows does not depend on how ids are spelled."""
    return (row["account"], -row["debit_minor"], -row["credit_minor"], row["timestamp"], row["row_id"])


class _Books:
    def __init__(self, public: dict) -> None:
        self.policy = public["policy"]
        self.base = self.policy["base_currency"]
        self.offset = self.policy["timezone"]
        self.start = self.policy["period_start"]
        self.end = self.policy["period_end"]
        self.accounts = self.policy["accounts"]
        self.exponents = self.policy["currency_exponents"]
        self.tolerance = self.policy["amount_tolerance_minor"]
        self.invoices = list(public["invoices"])
        self.payments = list(public["payments"])
        self.invoice_by_id = {inv["invoice_id"]: inv for inv in self.invoices}
        self.payment_by_id = {pay["payment_id"]: pay for pay in self.payments}
        self.rates = {fx_key(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"])
                      for r in public["fx"]}

        # Event groups (CONTRACTS 2.3) and receivable balances per invoice, in one pass.
        self.invoice_group: dict[str, list[dict]] = defaultdict(list)
        self.payment_group: dict[str, list[dict]] = defaultdict(list)
        self.receivable: dict[str, int] = defaultdict(int)
        receivable = self.accounts["receivable"]
        for row in public["ledger"]:
            if row["payment_id"] is not None:
                self.payment_group[row["payment_id"]].append(row)
            elif row["invoice_id"] is not None:
                self.invoice_group[row["invoice_id"]].append(row)
            if row["account"] == receivable and row["invoice_id"] is not None:
                self.receivable[row["invoice_id"]] += row["debit_minor"] - row["credit_minor"]

    def inside(self, timestamp: str) -> bool:
        return in_period(timestamp, self.start, self.end)

    def policy_amount(self, amount_minor: int, currency: str, timestamp: str) -> int | None:
        """Document amount at the policy rate, or None when the policy gives no rate."""
        rate = self.rates.get(fx_key(local_date(timestamp, self.offset), currency, self.base))
        if rate is None or currency not in self.exponents or self.base not in self.exponents:
            return None
        return convert(amount_minor, rate, self.exponents[currency], self.exponents[self.base])


def _duplicates(books: _Books) -> list[dict]:
    by_signature: dict[tuple, list[dict]] = defaultdict(list)
    for invoice in books.invoices:
        if books.invoice_group.get(invoice["invoice_id"]):
            by_signature[_signature(invoice)].append(invoice)
    findings = []
    for group in by_signature.values():
        group = sorted(group, key=lambda inv: inv["doc_id"])
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                if first["doc_id"] == second["doc_id"] or first["invoice_id"] == second["invoice_id"]:
                    continue
                ids = sorted([first["doc_id"], second["doc_id"]])
                findings.append({
                    "type": "duplicate_invoice", "affected_ids": ids,
                    "claimed_amount_minor": first["total_minor"], "currency": first["currency"],
                    "evidence_ids": ids,
                    "explanation": "Both invoices are posted and share customer, currency, total and line items.",
                })
    return findings


def _allocation_gaps(books: _Books) -> list[dict]:
    unallocated = books.accounts["unallocated"]
    findings = []
    for payment_id, rows in books.payment_group.items():
        payment = books.payment_by_id.get(payment_id)
        if payment is None or not payment.get("invoice_refs"):
            continue  # unknown payment, or an advance without references
        credits = [row for row in rows if row["account"] == unallocated and row["credit_minor"] > 0]
        if len(credits) != 1:
            continue  # no gap, or several unallocated rows: undefined, left unreported
        row = credits[0]
        for ref in sorted(set(payment["invoice_refs"])):
            invoice = books.invoice_by_id.get(ref)
            if invoice is None or books.receivable.get(ref, 0) <= 0:
                continue
            findings.append({
                "type": "allocation_gap", "affected_ids": sorted([payment["doc_id"], invoice["doc_id"]]),
                "claimed_amount_minor": row["credit_minor"], "currency": books.base,
                "evidence_ids": [payment["doc_id"], invoice["doc_id"], row["row_id"]],
                "explanation": "Part of the payment sits on unallocated while the invoice it refers to stays open.",
            })
    return findings


def _fx_finding(books: _Books, doc_id: str, deviations: list[tuple[int, dict]]) -> dict | None:
    wrong = [(amount, row) for amount, row in deviations if amount > books.tolerance["fx_error"]]
    if not wrong:
        return None
    wrong.sort(key=lambda pair: (-pair[0], _row_order(pair[1])))
    rows = [row["row_id"] for _, row in wrong][:MAX_EVIDENCE - 1]
    return {
        "type": "fx_error", "affected_ids": [doc_id], "claimed_amount_minor": wrong[0][0],
        "currency": books.base, "evidence_ids": [doc_id, *rows],
        "explanation": "The posted base amount differs from document amount times the policy rate.",
    }


def _fx_errors(books: _Books) -> list[dict]:
    receivable, revenue, cash = (books.accounts[name] for name in ("receivable", "revenue", "cash"))
    findings = []
    for invoice in books.invoices:
        rows = books.invoice_group.get(invoice["invoice_id"])
        if not rows or invoice["currency"] == books.base or not books.inside(invoice["issued_at"]):
            continue
        correct = books.policy_amount(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        if correct is None:
            continue
        deviations = [(abs(row["debit_minor"] - correct), row) for row in rows
                      if row["account"] == receivable and row["debit_minor"] > 0]
        deviations += [(abs(row["credit_minor"] - correct), row) for row in rows
                       if row["account"] == revenue and row["credit_minor"] > 0]
        finding = _fx_finding(books, invoice["doc_id"], deviations)
        if finding:
            findings.append(finding)
    for payment in books.payments:
        rows = books.payment_group.get(payment["payment_id"])
        if not rows or payment["currency"] == books.base or not books.inside(payment["timestamp"]):
            continue
        correct = books.policy_amount(payment["amount_minor"], payment["currency"], payment["timestamp"])
        cash_rows = [row for row in rows if row["account"] == cash]
        if correct is None or not cash_rows:
            continue
        booked = sum(row["debit_minor"] - row["credit_minor"] for row in cash_rows)
        difference = abs(booked - correct)
        finding = _fx_finding(books, payment["doc_id"], [(difference, row) for row in cash_rows])
        if finding:
            findings.append(finding)
    return findings


def _cutoffs(books: _Books) -> list[dict]:
    findings = []
    documents = [(inv["doc_id"], inv["issued_at"], inv["total_minor"], inv["currency"],
                  books.invoice_group.get(inv["invoice_id"])) for inv in books.invoices]
    documents += [(pay["doc_id"], pay["timestamp"], pay["amount_minor"], pay["currency"],
                   books.payment_group.get(pay["payment_id"])) for pay in books.payments]
    for doc_id, moment, amount, currency, rows in documents:
        if not rows or books.inside(moment):
            continue
        evidence = [row["row_id"] for row in sorted(rows, key=_row_order)][:MAX_EVIDENCE - 1]
        findings.append({
            "type": "period_cutoff", "affected_ids": [doc_id], "claimed_amount_minor": abs(amount),
            "currency": currency, "evidence_ids": [doc_id, *evidence],
            "explanation": "The document is dated outside the period in the company timezone but posted in it.",
        })
    return findings


def _sort_key(finding: dict) -> tuple:
    return (finding["type"], finding["affected_ids"], finding["claimed_amount_minor"], finding["currency"],
            finding["evidence_ids"])


def solve(public: dict) -> list[dict]:
    books = _Books(public)
    findings = _duplicates(books) + _allocation_gaps(books) + _fx_errors(books) + _cutoffs(books)
    return sorted(findings, key=_sort_key)
