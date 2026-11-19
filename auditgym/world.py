"""The hidden, correct story of one company and one reporting period.

Everything here uses internal keys ("E3", "I12", "P4"); public ids are assigned later by
surface.py, after errors are injected, so ids carry no information about the story.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

from .config import CURRENCY_EXPONENTS
from .money import parse_offset
from .rng import Rng

# Rough relative values, only to make synthetic rates look plausible. Not real quotes.
_VALUE = {"KZT": 21, "USD": 10000, "EUR": 10900, "RUB": 110, "CNY": 1380}
_SERVICES = ("Consulting", "Software licence", "Support plan", "Hosting", "Training",
             "Integration work", "Audit preparation", "Data migration", "Maintenance", "Design work")
_TERMS_DAYS = (14, 30, 45)
FX_MARGIN_DAYS = 5


def period_bounds(config: dict) -> tuple[dt.datetime, dt.datetime]:
    zone = parse_offset(config["timezone"])
    year, month, _ = (int(part) for part in config["period"]["start"].split("-"))
    start = dt.datetime(year, month, 1, tzinfo=zone)
    month_index = (year * 12 + (month - 1)) + config["period"]["months"]
    end = dt.datetime(month_index // 12, month_index % 12 + 1, 1, tzinfo=zone)
    return start, end


def _moment(rng: Rng, start: dt.datetime, end: dt.datetime) -> dt.datetime:
    """A whole-second moment in [start, end)."""
    return start + dt.timedelta(seconds=rng.below(int((end - start).total_seconds())))


def _rates(rng: Rng, config: dict, start: dt.datetime, end: dt.datetime) -> list[dict]:
    base = config["base_currency"]
    first_day = start.date() - dt.timedelta(days=FX_MARGIN_DAYS)
    days = (end.date() - first_day).days + FX_MARGIN_DAYS
    table = []
    for currency in config["currencies"]:
        stream = rng.child(f"fx:{currency}")
        # Episode level: +-10% around the rough value; then a daily walk of at most +-0.6%.
        level = Decimal(_VALUE[currency]) / Decimal(_VALUE[base]) * Decimal(900 + stream.below(201)) / 1000
        for day in range(days):
            level = level * Decimal(10000 + stream.randint(-60, 60)) / 10000
            rate = level.quantize(Decimal("0.0001"))
            table.append({"date": (first_day + dt.timedelta(days=day)).isoformat(),
                          "source_currency": currency, "target_currency": base, "rate": str(rate)})
    return table


def _line_items(rng: Rng, start: dt.datetime, end: dt.datetime) -> list[dict]:
    items = []
    last_day = (end - dt.timedelta(days=1)).date()
    for _ in range(rng.randint(1, 3)):
        quantity = rng.randint(1, 5)
        unit_price = rng.randint(20, 900) * 1000  # whole "tens" in a two-decimal currency
        items.append({"description": rng.choice(_SERVICES), "quantity": quantity,
                      "unit_price_minor": unit_price, "amount_minor": quantity * unit_price,
                      "service_start": start.date().isoformat(), "service_end": last_day.isoformat()})
    return items


def invoice_signature(invoice: dict) -> tuple:
    """Two invoices with the same signature are duplicates by definition (docs/SPEC.md)."""
    items = sorted((item["description"], item["quantity"], item["unit_price_minor"], item["amount_minor"],
                    item["service_start"], item["service_end"]) for item in invoice["line_items"])
    return (invoice["entity"], invoice["currency"], invoice["total_minor"], tuple(items))


def build_world(rng: Rng, config: dict) -> dict:
    start, end = period_bounds(config)
    base = config["base_currency"]
    low, high = config["size"]["invoices"]
    invoice_count = rng.randint(low, high)

    customers = [f"E{i}" for i in range(max(4, invoice_count // 6))]
    vendors = [f"E{len(customers) + i}" for i in range(rng.randint(3, 6))]

    invoices, signatures = [], set()
    stream = rng.child("invoices")
    for index in range(invoice_count):
        while True:  # bounded in practice: the signature space is far larger than the invoice count
            currency = base
            if config["currencies"] and stream.chance(2, 5):
                currency = stream.choice(config["currencies"])
            items = _line_items(stream, start, end)
            issued = _moment(stream, start, end)
            invoice = {"key": f"I{index}", "entity": stream.choice(customers), "currency": currency,
                       "issued_at": issued, "due_at": issued + dt.timedelta(days=stream.choice(_TERMS_DAYS)),
                       "total_minor": sum(item["amount_minor"] for item in items), "line_items": items}
            signature = invoice_signature(invoice)
            if signature not in signatures:
                signatures.add(signature)
                invoices.append(invoice)
                break

    # Payments: most invoices are paid in full inside the period; some stay open.
    payments = []
    stream = rng.child("payments")
    payable = [inv for inv in invoices if inv["issued_at"] + dt.timedelta(hours=1) < end and stream.chance(7, 10)]
    used: set[str] = set()
    for invoice in payable:
        if invoice["key"] in used:
            continue
        group = [invoice]
        # One payment for two invoices of the same customer and currency, both paid in full.
        partner = next((other for other in payable
                        if other["key"] not in used and other is not invoice
                        and other["entity"] == invoice["entity"] and other["currency"] == invoice["currency"]), None)
        if partner is not None and stream.chance(1, 6):
            group.append(partner)
        for member in group:
            used.add(member["key"])
        earliest = max(member["issued_at"] for member in group) + dt.timedelta(hours=1)
        payments.append({"key": f"P{len(payments)}", "entity": invoice["entity"], "currency": invoice["currency"],
                         "timestamp": _moment(stream, earliest, end),
                         "amount_minor": sum(member["total_minor"] for member in group),
                         "invoice_refs": [member["key"] for member in group]})

    # Background activity without documents: balanced expense/cash pairs.
    stream = rng.child("background")
    background, taken = [], set()
    low, high = config["size"]["background_pairs"]
    for _ in range(stream.randint(low, high)):
        while True:
            moment, vendor = _moment(stream, start, end), stream.choice(vendors)
            if (moment, vendor) not in taken:  # the pair is its own event group
                taken.add((moment, vendor))
                break
        background.append({"timestamp": moment, "entity": vendor, "amount_minor": stream.randint(5, 4000) * 1000})

    return {"base_currency": base, "timezone": config["timezone"], "period_start": start, "period_end": end,
            "currency_exponents": {c: CURRENCY_EXPONENTS[c] for c in [base, *config["currencies"]]},
            "entities": customers + vendors, "invoices": invoices, "payments": payments,
            "background": background, "fx": _rates(rng, config, start, end)}
