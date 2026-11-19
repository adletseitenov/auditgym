"""Hand edits of clean companies for the solver tests.

The solver is tested without the error generator: every case starts from a clean generated
company and applies small, explicit edits to the open records. Edits keep the books balanced the
way a bookkeeper would (a payment is rebooked as a whole: cash at the policy rate, receivable
closed at the booked invoice value, any remainder on fx_difference).
"""
from __future__ import annotations

import copy
import datetime as dt
from decimal import Decimal

from auditgym.generate import generate
from auditgym.money import convert, in_period, local_date, parse_offset, parse_time
from auditgym.rng import Rng
from auditgym.verify import group_key

LEDGER_FIELDS = ("row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                 "entity_id", "invoice_id", "payment_id")


def shift(timestamp: str, **delta) -> str:
    """Move an ISO timestamp, keeping its offset."""
    return (parse_time(timestamp) + dt.timedelta(**delta)).isoformat()


def in_offset(timestamp: str, offset: str) -> str:
    """The same instant written in another offset."""
    return parse_time(timestamp).astimezone(parse_offset(offset)).isoformat()


def moved_offset(offset: str, hours: int) -> str:
    """An offset `hours` away from `offset` (may lie beyond +-14:00; still a valid ISO offset)."""
    total = int(parse_offset(offset).utcoffset(None).total_seconds() // 60) + hours * 60
    sign = "-" if total < 0 else "+"
    return f"{sign}{abs(total) // 60:02d}:{abs(total) % 60:02d}"


def shift_date(date: str, days: int) -> str:
    return (dt.date.fromisoformat(date) + dt.timedelta(days=days)).isoformat()


class Company:
    """A clean generated company plus edit helpers. `public` is a private deep copy."""

    def __init__(self, seed: int, currencies=("USD",), timezone: str = "+05:00", **config) -> None:
        episode = generate(seed, {"errors": {}, "currencies": list(currencies), "timezone": timezone, **config})
        self.public = copy.deepcopy(episode["public"])
        self.policy = self.public["policy"]
        self.base = self.policy["base_currency"]
        self.offset = self.policy["timezone"]
        self._counter = 0

    # ----- lookups -----
    def new_id(self) -> str:
        self._counter += 1
        return f"zz{self._counter:05d}"

    @property
    def ledger(self) -> list[dict]:
        return self.public["ledger"]

    def rate(self, currency: str, date: str) -> Decimal:
        for row in self.public["fx"]:
            if row["date"] == date and row["source_currency"] == currency:
                return Decimal(row["rate"])
        raise KeyError(date)

    def set_rate(self, currency: str, date: str, rate: Decimal) -> None:
        for row in self.public["fx"]:
            if row["date"] == date and row["source_currency"] == currency:
                row["rate"] = str(rate)
                return
        raise KeyError(date)

    def to_base(self, amount: int, currency: str, timestamp: str, date: str | None = None) -> int:
        if currency == self.base:
            return amount
        date = date or local_date(timestamp, self.offset)
        exponents = self.policy["currency_exponents"]
        return convert(amount, self.rate(currency, date), exponents[currency], exponents[self.base])

    def inside(self, timestamp: str) -> bool:
        return in_period(timestamp, self.policy["period_start"], self.policy["period_end"])

    def invoice_rows(self, invoice: dict) -> list[dict]:
        return [r for r in self.ledger if r["payment_id"] is None and r["invoice_id"] == invoice["invoice_id"]]

    def payment_rows(self, payment: dict) -> list[dict]:
        return [r for r in self.ledger if r["payment_id"] == payment["payment_id"]]

    def booked(self, invoice: dict) -> int:
        return sum(r["debit_minor"] for r in self.invoice_rows(invoice) if r["account"] == "receivable")

    def payments_of(self, invoice: dict) -> list[dict]:
        return [p for p in self.public["payments"] if invoice["invoice_id"] in (p["invoice_refs"] or [])]

    def row(self, rows: list[dict], account: str) -> dict:
        (found,) = [r for r in rows if r["account"] == account]
        return found

    def posted_inside(self, document: dict) -> bool:
        """Dated inside the period and posted (distractor companies also hold unposted documents of the
        neighbouring periods)."""
        moment = document["issued_at"] if "issued_at" in document else document["timestamp"]
        rows = self.invoice_rows(document) if "invoice_id" in document and "payment_id" not in document \
            else self.payment_rows(document)
        return self.inside(moment) and bool(rows)

    def ordinary_payment(self, invoice: dict, payment: dict) -> bool:
        """A posted, in-period payment of exactly the invoice total, with the invoice as its only reference
        and no unallocated row: not an overpayment, partial or split payment, refund or advance."""
        return (payment["invoice_refs"] == [invoice["invoice_id"]] and payment["amount_minor"] == invoice["total_minor"]
                and payment["amount_minor"] > 0 and self.posted_inside(payment)
                and not any(r["account"] == "unallocated" for r in self.payment_rows(payment)))

    def has_twin(self, invoice: dict) -> bool:
        """Another invoice with the same customer, currency and total (e.g. a recurring invoice for
        another service period): edits that move dates or service periods avoid these."""
        return any(other["doc_id"] != invoice["doc_id"] and other["entity_id"] == invoice["entity_id"]
                   and other["currency"] == invoice["currency"] and other["total_minor"] == invoice["total_minor"]
                   for other in self.public["invoices"])

    def _plain(self, invoice: dict, foreign: bool | None) -> bool:
        return ((foreign is None or (invoice["currency"] != self.base) == foreign)
                and self.posted_inside(invoice) and not self.has_twin(invoice))

    def paid_once(self, foreign: bool | None = None) -> list[tuple[dict, dict]]:
        """(invoice, payment): a posted in-period invoice closed by exactly one ordinary payment (see
        `ordinary_payment`) and referenced by no other payment or refund."""
        pairs = []
        for invoice in self.public["invoices"]:
            if not self._plain(invoice, foreign):
                continue
            payments = self.payments_of(invoice)
            if len(payments) == 1 and self.ordinary_payment(invoice, payments[0]):
                pairs.append((invoice, payments[0]))
        return pairs

    def unpaid(self, foreign: bool | None = None) -> list[dict]:
        """Posted in-period invoices that no payment (posted or not, of any period) references."""
        return [inv for inv in self.public["invoices"] if self._plain(inv, foreign) and not self.payments_of(inv)]

    # ----- edits -----
    def add_row(self, timestamp, account, debit, credit, entity_id, invoice_id=None, payment_id=None) -> dict:
        row = dict(zip(LEDGER_FIELDS, (self.new_id(), timestamp, account, debit, credit, self.base,
                                       entity_id, invoice_id, payment_id)))
        self.ledger.append(row)
        return row

    def remove_rows(self, rows: list[dict]) -> None:
        ids = {r["row_id"] for r in rows}
        self.public["ledger"] = [r for r in self.ledger if r["row_id"] not in ids]

    def book_invoice(self, invoice: dict, amount: int | None = None, at: str | None = None) -> list[dict]:
        if amount is None:
            amount = self.to_base(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        at = at or invoice["issued_at"]
        return [self.add_row(at, "receivable", amount, 0, invoice["entity_id"], invoice["invoice_id"]),
                self.add_row(at, "revenue", 0, amount, invoice["entity_id"], invoice["invoice_id"])]

    def add_invoice(self, like: dict, post: bool = True, **fields) -> dict:
        invoice = copy.deepcopy(like)
        invoice.update(doc_id=self.new_id(), invoice_id=self.new_id(), **fields)
        invoice["total_minor"] = sum(item["amount_minor"] for item in invoice["line_items"])
        self.public["invoices"].append(invoice)
        if post:
            self.book_invoice(invoice)
        return invoice

    def book_payment(self, payment: dict, closes: dict[str, int], unallocated: int = 0,
                     cash: int | None = None, entity_id: str | None = None) -> list[dict]:
        """Cash at the policy rate (unless given), receivable closed by `closes`, rest to fx_difference."""
        at = payment["timestamp"]
        if entity_id is None:
            refs = payment["invoice_refs"] or []
            invoices = {inv["invoice_id"]: inv for inv in self.public["invoices"]}
            entity_id = invoices[refs[0]]["entity_id"]
        if cash is None:
            cash = self.to_base(payment["amount_minor"], payment["currency"], at)
        single = payment["invoice_refs"][0] if payment["invoice_refs"] and len(payment["invoice_refs"]) == 1 else None
        pid = payment["payment_id"]
        rows = [self.add_row(at, "cash", max(cash, 0), max(-cash, 0), entity_id, single, pid)]
        for invoice_id, amount in sorted(closes.items()):
            rows.append(self.add_row(at, "receivable", max(-amount, 0), max(amount, 0), entity_id, invoice_id, pid))
        if unallocated:
            rows.append(self.add_row(at, "unallocated", 0, unallocated, entity_id, None, pid))
        rest = cash - sum(closes.values()) - unallocated
        if rest:
            rows.append(self.add_row(at, "fx_difference", max(-rest, 0), max(rest, 0), entity_id, None, pid))
        return rows

    def rebook_invoice(self, invoice: dict) -> list[dict]:
        """Rebook the invoice rows (kept in place, same ids and times) at the rate of its own date,
        then let every single-reference payment close the new booked value."""
        amount = self.to_base(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        rows = self.invoice_rows(invoice)
        for row in rows:
            row["debit_minor" if row["debit_minor"] else "credit_minor"] = amount
        for payment in self.payments_of(invoice):
            if payment["invoice_refs"] == [invoice["invoice_id"]] and self.payment_rows(payment):
                self.rebook_payment(payment, closes={invoice["invoice_id"]: amount})
        return rows

    def lag_postings(self, rows: list[dict], **delta) -> bool:
        """Move posting times later by `delta`, staying inside the period. False if it would leave it."""
        moved = [shift(r["timestamp"], **delta) for r in rows]
        if not all(self.inside(m) for m in moved):
            return False
        for row, moment in zip(rows, moved):
            row["timestamp"] = moment
        return True

    def rebook_payment(self, payment: dict, **kwargs) -> list[dict]:
        entity_id = self.payment_rows(payment)[0]["entity_id"]
        self.remove_rows(self.payment_rows(payment))
        return self.book_payment(payment, entity_id=entity_id, **kwargs)

    def add_payment(self, **fields) -> dict:
        payment = {"doc_id": self.new_id(), "payment_id": self.new_id(), **fields}
        payment = {key: payment[key] for key in ("doc_id", "payment_id", "timestamp", "currency",
                                                 "amount_minor", "invoice_refs")}
        self.public["payments"].append(payment)
        return payment

    def done(self) -> dict:
        """Refresh manifest counts and check that the books still balance. Returns `public`."""
        counts = self.public["manifest"]["counts"]
        for table in ("ledger", "invoices", "payments", "fx"):
            counts[table] = len(self.public[table])
        groups: dict = {}
        for row in self.ledger:
            key = group_key(row)
            groups[key] = groups.get(key, 0) + row["debit_minor"] - row["credit_minor"]
            if (row["debit_minor"] > 0) == (row["credit_minor"] > 0) or min(row["debit_minor"], row["credit_minor"]) < 0:
                raise AssertionError(f"test edit left row {row['row_id']} with a bad side")
        unbalanced = [key for key, balance in groups.items() if balance]
        if unbalanced:
            raise AssertionError(f"test edit left group {unbalanced[0]} unbalanced")
        return self.public


# ----- metamorphic helpers -----
ID_FIELDS = {"ledger": ("row_id", "entity_id", "invoice_id", "payment_id"),
             "invoices": ("doc_id", "invoice_id", "entity_id"),
             "payments": ("doc_id", "payment_id")}


ID_STYLES = ("token", "numeric", "hex64", "cyrillic", "separators")
_CYRILLIC = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяӘәҒғҚқҢңӨөҰұҮүҺһІі"


def _new_id(rng: Rng, style: str) -> str:
    if style == "numeric":
        return str(rng.randint(0, 10 ** 9))
    if style == "hex64":
        return rng.token(64, "0123456789abcdef")
    if style == "cyrillic":
        return rng.token(8, _CYRILLIC)
    if style == "separators":
        return rng.token(4) + rng.choice((":", "|", "::", "|:")) + rng.token(4) + rng.choice(("", "|", ":"))
    return rng.token(12)


def rename_and_shuffle(public: dict, seed: int, style: str = "token") -> tuple[dict, dict]:
    """Rename every id consistently (in the given id style) and shuffle every table, every invoice's
    line items and every payment's references. Returns (new public, old -> new)."""
    rng = Rng(seed, "metamorph")
    public = copy.deepcopy(public)
    old_ids = set()
    for table, fields in ID_FIELDS.items():
        for record in public[table]:
            old_ids |= {record[f] for f in fields if record[f] is not None}
    for payment in public["payments"]:
        old_ids |= set(payment["invoice_refs"] or [])
    mapping, taken = {}, set()
    for old in sorted(old_ids):
        while True:
            new = _new_id(rng, style)
            if new not in taken and new not in old_ids:
                break
        taken.add(new)
        mapping[old] = new
    for table, fields in ID_FIELDS.items():
        for record in public[table]:
            for f in fields:
                if record[f] is not None:
                    record[f] = mapping[record[f]]
    for payment in public["payments"]:
        if payment["invoice_refs"] is not None:
            payment["invoice_refs"] = [mapping[ref] for ref in payment["invoice_refs"]]
    for table in ("ledger", "invoices", "payments", "fx"):
        rng.shuffle(public[table])
    for invoice in public["invoices"]:
        rng.shuffle(invoice["line_items"])
    for payment in public["payments"]:
        if payment["invoice_refs"]:
            rng.shuffle(payment["invoice_refs"])
    return public, mapping


def canonical(findings: list[dict], mapping: dict | None = None) -> list[tuple]:
    """Findings as comparable tuples, optionally after renaming ids."""
    def m(value):
        return mapping.get(value, value) if mapping else value

    return sorted((f["type"], tuple(sorted(m(i) for i in f["affected_ids"])), f["claimed_amount_minor"],
                   f["currency"], tuple(sorted(m(i) for i in f["evidence_ids"]))) for f in findings)
