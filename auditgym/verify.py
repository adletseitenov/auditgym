"""verify(episode) -> list of problems. An empty list means the episode is well formed.

Checks the open records and the pre-injection ledger with nothing but the published policy,
the way an outside checker would. It does not import the generator.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from .money import convert, fx_key, in_period, local_date

LEDGER_FIELDS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                 "entity_id", "invoice_id", "payment_id"]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def group_key(row: dict) -> tuple:
    """Event group of a posting (docs/CONTRACTS.md, section 2.3)."""
    if row["payment_id"] is not None:
        return ("payment", row["payment_id"])
    if row["invoice_id"] is not None:
        return ("invoice", row["invoice_id"])
    return ("background", row["timestamp"], row["entity_id"])


class _Books:
    """Indexes over the open records."""

    def __init__(self, public: dict) -> None:
        self.policy = public["policy"]
        self.base = self.policy["base_currency"]
        self.offset = self.policy["timezone"]
        self.invoices = {inv["invoice_id"]: inv for inv in public["invoices"]}
        self.payments = {pay["payment_id"]: pay for pay in public["payments"]}
        self.rates = {fx_key(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"])
                      for r in public["fx"]}

    def to_base(self, amount_minor: int, currency: str, timestamp: str) -> int | None:
        if currency == self.base:
            return amount_minor
        rate = self.rates.get(fx_key(local_date(timestamp, self.offset), currency, self.base))
        if rate is None:
            return None
        exponents = self.policy["currency_exponents"]
        return convert(amount_minor, rate, exponents[currency], exponents[self.base])


def _check_ledger_shape(rows: list[dict], books: _Books, name: str) -> list[str]:
    problems = []
    accounts = set(books.policy["accounts"].values())
    seen = set()
    for row in rows:
        rid = row.get("row_id")
        if list(row) != LEDGER_FIELDS:
            problems.append(f"{name}: row {rid} has wrong fields")
            continue
        if rid in seen:
            problems.append(f"{name}: duplicate row_id {rid}")
        seen.add(rid)
        debit, credit = row["debit_minor"], row["credit_minor"]
        if not (_is_int(debit) and _is_int(credit)) or debit < 0 or credit < 0 or (debit > 0) == (credit > 0):
            problems.append(f"{name}: row {rid} must have exactly one positive integer side")
        if row["currency"] != books.base:
            problems.append(f"{name}: row {rid} is not in the base currency")
        if row["account"] not in accounts:
            problems.append(f"{name}: row {rid} uses an unknown account")
        if row["invoice_id"] is not None and row["invoice_id"] not in books.invoices:
            problems.append(f"{name}: row {rid} links to a missing invoice")
        if row["payment_id"] is not None and row["payment_id"] not in books.payments:
            problems.append(f"{name}: row {rid} links to a missing payment")
        if not in_period(row["timestamp"], books.policy["period_start"], books.policy["period_end"]):
            problems.append(f"{name}: row {rid} is outside the period")
    return problems


def _check_balance(rows: list[dict], name: str) -> list[str]:
    problems = []
    if sum(r["debit_minor"] for r in rows) != sum(r["credit_minor"] for r in rows):
        problems.append(f"{name}: debits and credits differ")
    groups = defaultdict(int)
    for row in rows:
        groups[group_key(row)] += row["debit_minor"] - row["credit_minor"]
    for key, balance in sorted(groups.items(), key=repr):
        if balance != 0:
            problems.append(f"{name}: group {key} is unbalanced by {balance}")
    return problems


def _check_documents(public: dict, books: _Books) -> list[str]:
    problems = []
    doc_ids = [d["doc_id"] for d in public["invoices"]] + [d["doc_id"] for d in public["payments"]]
    if len(set(doc_ids)) != len(doc_ids):
        problems.append("documents: doc_id is not unique")
    if len(books.invoices) != len(public["invoices"]) or len(books.payments) != len(public["payments"]):
        problems.append("documents: invoice_id or payment_id is not unique")
    for invoice in public["invoices"]:
        if invoice["total_minor"] != sum(item["amount_minor"] for item in invoice["line_items"]):
            problems.append(f"invoice {invoice['doc_id']}: total differs from its line items")
        if books.to_base(invoice["total_minor"], invoice["currency"], invoice["issued_at"]) is None:
            problems.append(f"invoice {invoice['doc_id']}: no rate for its date")
    for payment in public["payments"]:
        for ref in payment["invoice_refs"] or []:
            if ref not in books.invoices:
                problems.append(f"payment {payment['doc_id']}: refers to a missing invoice")
        if books.to_base(payment["amount_minor"], payment["currency"], payment["timestamp"]) is None:
            problems.append(f"payment {payment['doc_id']}: no rate for its date")
    counts = public["manifest"]["counts"]
    for table in ("ledger", "invoices", "payments", "fx"):
        if counts[table] != len(public[table]):
            problems.append(f"manifest: count of {table} differs from the file")
    return problems


def _check_clean_postings(pre: list[dict], books: _Books, injected_docs: set[str]) -> list[str]:
    """Before injection every document in the period is posted once, at the policy amount."""
    problems = []
    policy = books.policy
    by_invoice = defaultdict(list)
    by_payment = defaultdict(list)
    for row in pre:
        if row["payment_id"] is not None:
            by_payment[row["payment_id"]].append(row)
        elif row["invoice_id"] is not None:
            by_invoice[row["invoice_id"]].append(row)

    booked = {}
    for invoice_id, invoice in books.invoices.items():
        rows = by_invoice.get(invoice_id, [])
        inside = in_period(invoice["issued_at"], policy["period_start"], policy["period_end"])
        if not rows:
            if inside and invoice["doc_id"] not in injected_docs:
                problems.append(f"pre: invoice {invoice['doc_id']} is in the period but not posted")
            continue
        if not inside:
            problems.append(f"pre: invoice {invoice['doc_id']} is outside the period but posted")
        expected = books.to_base(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        debit = [r for r in rows if r["account"] == "receivable" and r["debit_minor"] == expected]
        credit = [r for r in rows if r["account"] == "revenue" and r["credit_minor"] == expected]
        if len(rows) != 2 or len(debit) != 1 or len(credit) != 1:
            problems.append(f"pre: invoice {invoice['doc_id']} is not posted once at the policy amount")
        if any(r["timestamp"] != invoice["issued_at"] for r in rows):
            problems.append(f"pre: invoice {invoice['doc_id']} is posted at another time")
        booked[invoice_id] = expected

    closed = defaultdict(int)
    for payment_id, payment in books.payments.items():
        rows = by_payment.get(payment_id, [])
        inside = in_period(payment["timestamp"], policy["period_start"], policy["period_end"])
        if not rows:
            if inside and payment["doc_id"] not in injected_docs:
                problems.append(f"pre: payment {payment['doc_id']} is in the period but not posted")
            continue
        if not inside:
            problems.append(f"pre: payment {payment['doc_id']} is outside the period but posted")
        expected = books.to_base(payment["amount_minor"], payment["currency"], payment["timestamp"])
        cash = sum(r["debit_minor"] - r["credit_minor"] for r in rows if r["account"] == "cash")
        if cash != expected:
            problems.append(f"pre: payment {payment['doc_id']} cash differs from the policy amount")
        if any(r["timestamp"] != payment["timestamp"] for r in rows):
            problems.append(f"pre: payment {payment['doc_id']} is posted at another time")
        refs = set(payment["invoice_refs"] or [])
        for row in rows:
            if row["account"] == "receivable":
                if row["invoice_id"] not in refs:
                    problems.append(f"pre: payment {payment['doc_id']} closes an invoice it does not refer to")
                closed[row["invoice_id"]] += row["credit_minor"] - row["debit_minor"]
    for invoice_id, amount in closed.items():
        if invoice_id in booked and amount > booked[invoice_id]:
            problems.append(f"pre: invoice {books.invoices[invoice_id]['doc_id']} is closed for more than it was booked")
    return problems


def verify(episode: dict) -> list[str]:
    public, private = episode["public"], episode["private"]
    books = _Books(public)
    injected_docs = {record["doc_id"] for step in private["provenance"]["steps"]
                     for record in step.get("after", []) if "doc_id" in record}
    problems = _check_documents(public, books)
    for name, rows in (("ledger", public["ledger"]), ("pre", private["pre_ledger"])):
        problems += _check_ledger_shape(rows, books, name)
        problems += _check_balance(rows, name)
    problems += _check_clean_postings(private["pre_ledger"], books, injected_docs)
    if private["truth"]["episode_id"] != public["manifest"]["episode_id"]:
        problems.append("truth: episode_id differs from the manifest")
    return problems
