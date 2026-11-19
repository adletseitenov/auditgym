"""Double-entry postings derived from the story. The ledger is kept in the base currency."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

from .money import convert


def rate_on(world: dict, currency: str, moment: dt.datetime) -> Decimal:
    """Policy rate: the document date in the company's timezone."""
    date = moment.astimezone(world["period_start"].tzinfo).date().isoformat()
    for row in world["fx"]:
        if row["date"] == date and row["source_currency"] == currency:
            return Decimal(row["rate"])
    raise KeyError(f"no rate for {currency} on {date}")


def to_base(world: dict, amount_minor: int, currency: str, moment: dt.datetime) -> int:
    base = world["base_currency"]
    if currency == base:
        return amount_minor
    exponents = world["currency_exponents"]
    return convert(amount_minor, rate_on(world, currency, moment), exponents[currency], exponents[base])


def _row(moment, account, debit, credit, world, entity=None, invoice=None, payment=None) -> dict:
    return {"timestamp": moment, "account": account, "debit_minor": debit, "credit_minor": credit,
            "currency": world["base_currency"], "entity": entity, "invoice": invoice, "payment": payment}


def invoice_rows(world: dict, invoice: dict) -> list[dict]:
    amount = to_base(world, invoice["total_minor"], invoice["currency"], invoice["issued_at"])
    at, entity, key = invoice["issued_at"], invoice["entity"], invoice["key"]
    return [_row(at, "receivable", amount, 0, world, entity, key),
            _row(at, "revenue", 0, amount, world, entity, key)]


def payment_rows(world: dict, payment: dict, invoices: dict[str, dict]) -> list[dict]:
    """Cash at the payment-date rate; receivable closed at each invoice's booked amount."""
    at, entity, key = payment["timestamp"], payment["entity"], payment["key"]
    refs = payment["invoice_refs"]
    cash = to_base(world, payment["amount_minor"], payment["currency"], at)
    rows = [_row(at, "cash", cash, 0, world, entity, refs[0] if len(refs) == 1 else None, key)]
    closed = 0
    for ref in refs:
        invoice = invoices[ref]
        booked = to_base(world, invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        rows.append(_row(at, "receivable", 0, booked, world, entity, ref, key))
        closed += booked
    difference = cash - closed  # derived from booked values; never an error by itself
    if difference > 0:
        rows.append(_row(at, "fx_difference", 0, difference, world, entity, None, key))
    elif difference < 0:
        rows.append(_row(at, "fx_difference", -difference, 0, world, entity, None, key))
    return rows


def build_books(world: dict) -> list[dict]:
    invoices = {invoice["key"]: invoice for invoice in world["invoices"]}
    rows: list[dict] = []
    for invoice in world["invoices"]:
        rows += invoice_rows(world, invoice)
    for payment in world["payments"]:
        rows += payment_rows(world, payment, invoices)
    for pair in world["background"]:
        rows.append(_row(pair["timestamp"], "expense", pair["amount_minor"], 0, world, pair["entity"]))
        rows.append(_row(pair["timestamp"], "cash", 0, pair["amount_minor"], world, pair["entity"]))
    for index, row in enumerate(rows):
        row["key"] = f"R{index}"  # identity of a posting; survives into the pre-injection snapshot
    return rows
