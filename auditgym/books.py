"""Double-entry postings derived from the story. The ledger is kept in the base currency.

The books are a pure function of the world: error operators change documents (and their
`rate_override` / `gap_minor` fields) and the books are rebuilt, so every row has a stable key
derived from its document and role ("R:I3:dr", "R:P5:ar:I3", ...). Keys never reach the public
files; surface.py maps them to opaque ids.

Payment rule (published in the design note; verify.py checks it independently). Payments of an
invoice are taken in document-time order; `remaining` is the invoice total minus earlier positive
single-reference payments, in the document currency:
- several references: each invoice closed at its booked (posted) amount; amount = sum of totals;
- refund (amount < 0, one reference): receivable debited at the policy value of the refund;
- amount < remaining: partial, receivable credited at the policy value of the payment;
- amount = remaining: final, receivable credited with what is still open (booked - closed);
- amount > remaining (only as the sole payment): overpayment, the invoice closed in full and the
  excess, at the policy value, credited to unallocated;
- no reference: advance, the whole policy value on unallocated.
Cash is the policy value of the payment (or the wrong rate of an fx_error); an allocation_gap
moves `gap_minor` from the receivable credit to unallocated; the rest is fx_difference. Closings
never depend on the posted cash, so two errors on one payment do not interact.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from decimal import Decimal

from .money import convert


def rate_on(world: dict, currency: str, moment: dt.datetime) -> Decimal:
    """Policy rate: the document date in the company's timezone."""
    date = moment.astimezone(world["zone"]).date().isoformat()
    rate = world["rates"].get((date, currency))
    if rate is None:
        raise KeyError(f"no rate for {currency} on {date}")
    return rate


def to_base(world: dict, amount_minor: int, currency: str, moment: dt.datetime, rate: Decimal | None = None) -> int:
    base = world["base_currency"]
    if currency == base:
        return amount_minor
    exponents = world["currency_exponents"]
    rate = rate_on(world, currency, moment) if rate is None else rate
    return convert(amount_minor, rate, exponents[currency], exponents[base])


def booked(world: dict, invoice: dict) -> int:
    """The base amount an invoice is posted at (its own-date policy rate, or an fx_error's rate)."""
    return to_base(world, invoice["total_minor"], invoice["currency"], invoice["issued_at"], invoice["rate_override"])


def _row(key, moment, account, debit, credit, world, entity=None, invoice=None, payment=None) -> dict:
    return {"key": key, "timestamp": moment.astimezone(world["zone"]), "account": account,
            "debit_minor": debit, "credit_minor": credit, "currency": world["base_currency"],
            "entity": entity, "invoice": invoice, "payment": payment}


def _side(amount: int) -> tuple[int, int]:
    return (amount, 0) if amount > 0 else (0, -amount)


def invoice_rows(world: dict, invoice: dict) -> list[dict]:
    if invoice["posted_at"] is None:
        return []
    amount, at, key = booked(world, invoice), invoice["posted_at"], invoice["key"]
    return [_row(f"R:{key}:dr", at, "receivable", amount, 0, world, invoice["entity"], key),
            _row(f"R:{key}:cr", at, "revenue", 0, amount, world, invoice["entity"], key)]


def payment_order(payment: dict) -> tuple:
    return (payment["timestamp"], int(payment["key"][1:]))


def payment_parts(world: dict) -> dict[str, dict]:
    """Per posted payment: {"cash", "correct_cash", "closes": {invoice: base amount}, "unallocated"}."""
    invoices = {inv["key"]: inv for inv in world["invoices"]}
    remaining = {key: inv["total_minor"] for key, inv in invoices.items()}
    closed: dict[str, int] = defaultdict(int)
    parts = {}
    for payment in sorted((p for p in world["payments"] if p["posted_at"] is not None), key=payment_order):
        at, currency, amount = payment["timestamp"], payment["currency"], payment["amount_minor"]
        correct = to_base(world, amount, currency, at)
        cash = correct if payment["rate_override"] is None else to_base(world, amount, currency, at, payment["rate_override"])
        refs = payment["invoice_refs"]
        closes, unallocated = {}, 0
        if refs is None:
            unallocated = correct
        elif len(refs) > 1:
            for ref in refs:
                closes[ref] = booked(world, invoices[ref]) - closed[ref]
                remaining[ref] -= invoices[ref]["total_minor"]
        else:
            ref = refs[0]
            if amount < 0:
                closes[ref] = correct  # negative: the receivable is debited
            elif amount < remaining[ref]:
                closes[ref] = correct
            else:
                closes[ref] = booked(world, invoices[ref]) - closed[ref]
                if amount > remaining[ref]:
                    unallocated = to_base(world, amount - remaining[ref], currency, at)
            if amount > 0:
                remaining[ref] -= amount
        if payment["gap_minor"]:
            ref = refs[0]
            closes[ref] -= payment["gap_minor"]
            unallocated += payment["gap_minor"]
        for ref, value in closes.items():
            closed[ref] += value
        parts[payment["key"]] = {"cash": cash, "correct_cash": correct, "closes": closes, "unallocated": unallocated}
    return parts


def payment_rows(world: dict, payment: dict, part: dict) -> list[dict]:
    at, entity, key = payment["posted_at"], payment["entity"], payment["key"]
    refs = payment["invoice_refs"] or []
    single = refs[0] if len(refs) == 1 else None
    rows = [_row(f"R:{key}:cash", at, "cash", *_side(part["cash"]), world, entity, single, key)]
    for ref in sorted(part["closes"], key=lambda k: int(k[1:])):
        credit, debit = _side(part["closes"][ref])
        rows.append(_row(f"R:{key}:ar:{ref}", at, "receivable", debit, credit, world, entity, ref, key))
    if part["unallocated"]:
        rows.append(_row(f"R:{key}:un", at, "unallocated", 0, part["unallocated"], world, entity, None, key))
    difference = part["cash"] - sum(part["closes"].values()) - part["unallocated"]
    if difference:
        credit, debit = _side(difference)
        rows.append(_row(f"R:{key}:fx", at, "fx_difference", debit, credit, world, entity, None, key))
    return rows


def build_books(world: dict) -> list[dict]:
    rows: list[dict] = []
    for invoice in world["invoices"]:
        rows += invoice_rows(world, invoice)
    parts = payment_parts(world)
    for payment in world["payments"]:
        if payment["key"] in parts:
            rows += payment_rows(world, payment, parts[payment["key"]])
    for pair in world["background"]:
        at, amount, key = pair["timestamp"], pair["amount_minor"], pair["key"]
        rows.append(_row(f"R:{key}:dr", at, "expense", amount, 0, world, pair["entity"]))
        rows.append(_row(f"R:{key}:cr", at, "cash", 0, amount, world, pair["entity"]))
    return rows
