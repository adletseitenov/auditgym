"""The hidden, correct story of one company and one reporting period.

Everything here uses internal keys ("E3", "I12", "P4"); public ids are assigned later by
surface.py, after errors are injected, so ids carry no information about the story.

Times are aware datetimes. A document keeps its own UTC offset (usually the company's; with
distractors some documents are written in another offset, so their naive date differs from the
company-local date). A posted document has `posted_at`: the moment of its ledger postings, a
random lag (one minute to three weeks) after the document date and always inside the period; the
documents of one invoice are posted in their document order (order_postings).

Invoices of the neighbouring periods (dated just before or after the period, with that period's
service) are part of every story and are never posted by it: they are the pool period_cutoff
posts from (docs/AUDIT.md D4). They are paid with the usual probability, by payments dated on
their own side of the period, which are not posted either.

With `config["distractors"]` the story also contains legitimate cases that look suspicious but
satisfy no error definition (docs/SPEC.md): overpayments, advances, partial and split payments,
refunds, recurring invoices that split the period's service in two halves, extra neighbouring-
period invoices, documents at the period boundary, documents in other offsets, weekend rates
equal to Friday's.

The rates are drawn last, from a stream keyed by a digest of the whole story: the first rate of
an episode cannot be computed from the seed without building the story (docs/AUDIT.md L1).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from decimal import Decimal

from .config import CURRENCY_EXPONENTS, interaction_issue_count
from .limits import MAX_LEDGER_ROWS, ROWS_PER_INVOICE, ROWS_PER_ISSUE
from .money import parse_offset, round_rate
from .rng import Rng

# Rough relative values, only to make synthetic rates look plausible. Not real quotes.
_VALUE = {"KZT": 21, "USD": 10000, "EUR": 10900, "RUB": 110, "CNY": 1380}
# Minor units per price step, so documents in every currency have comparable real value.
PRICE_UNIT = {"USD": 1000, "EUR": 1000, "CNY": 7000, "RUB": 90000, "KZT": 500000}
_SERVICES = ("Consulting", "Software licence", "Support plan", "Hosting", "Training",
             "Integration work", "Audit preparation", "Data migration", "Maintenance", "Design work")
TERMS_DAYS = (14, 30, 45)
FX_MARGIN_DAYS = 5
# Offsets a document may be written in (distractors), from -11:00 to +14:00 with :30 and :45.
OFFSETS = ("-11:00", "-09:30", "-08:00", "-05:00", "-03:00", "-02:30", "+00:00", "+01:00", "+03:00",
           "+04:30", "+05:00", "+05:30", "+05:45", "+08:00", "+09:30", "+12:45", "+14:00")
CUSTOMERS = (4, 40)  # number of customers: its own range, independent of the invoice count
EDGE_GAP = dt.timedelta(seconds=120)  # in-period documents are dated at least this long before the end
PAY_AFTER = dt.timedelta(hours=1)  # a payment comes at least this long after its invoice
PAID_CHANCE = (7, 10)
MIN_LAG, MAX_LAG = 60, 21 * 86400  # legitimate posting lag: a minute to three weeks
BOUNDARY_DAYS = 4  # documents outside the period stay this close to it (inside the fx margin)
# Overpayment excess and allocation gap: the same share of a document amount, converted the same
# way, so their roundness does not tell them apart (docs/AUDIT.md L6).
SHARE_PERCENT = (2, 30)


def period_bounds(config: dict) -> tuple[dt.datetime, dt.datetime]:
    zone = parse_offset(config["timezone"])
    year, month, _ = (int(part) for part in config["period"]["start"].split("-"))
    start = dt.datetime(year, month, 1, tzinfo=zone)
    return start, add_months(start, config["period"]["months"])


def add_months(moment: dt.datetime | dt.date, months: int):
    index = moment.year * 12 + (moment.month - 1) + months
    return moment.replace(year=index // 12, month=index % 12 + 1, day=1)


def service_period(world: dict, shift: int) -> tuple[str, str]:
    """The reporting period moved by `shift` periods of the same length, as (first day, last day)."""
    months = world["months"]
    first = add_months(world["period_start"].date(), shift * months)
    last = add_months(first, months) - dt.timedelta(days=1)
    return first.isoformat(), last.isoformat()


def moment_between(rng: Rng, low: dt.datetime, high: dt.datetime) -> dt.datetime:
    """A whole-second moment in [low, high); `low` when the interval is shorter than a second."""
    span = int((high - low).total_seconds())
    return low + dt.timedelta(seconds=rng.below(span)) if span > 0 else low


def draw_lag(rng: Rng) -> int:
    """Seconds between a document and its posting: minutes, hours, days or a few weeks."""
    band = rng.below(20)
    if band < 7:
        return rng.randint(MIN_LAG, 3599)
    if band < 13:
        return rng.randint(3600, 86399)
    if band < 17:
        return rng.randint(86400, 4 * 86400)
    return rng.randint(4 * 86400, MAX_LAG)


def posting_lag(rng: Rng, moment: dt.datetime, end: dt.datetime) -> dt.datetime:
    """Posting moment of a document dated `moment` (>= EDGE_GAP before `end`): a drawn lag later,
    never on the document's own second and never after the period."""
    lag = draw_lag(rng)
    room = int((end - moment).total_seconds()) - 1
    if lag > room:
        lag = rng.randint(MIN_LAG, room)
    return moment + dt.timedelta(seconds=lag)


def _document_moment(document: dict) -> dt.datetime:
    return document["issued_at"] if "issued_at" in document else document["timestamp"]


def order_postings(world: dict) -> None:
    """Posted documents linked by references (an invoice, its payments and refunds; the invoices of
    one multi-reference payment) are posted in document order: every payment after its invoices and
    after their earlier payments, so the ledger's posting order and the documents' order agree
    (policy.payment_rule). The moments already drawn for such a group are handed out again, the
    earliest to the earliest document: each was at least MIN_LAG after its own document, so the
    k-th smallest is at least MIN_LAG after the k-th document, and no new draw is needed. Equal
    moments are pulled apart by one second, backwards. A group already in order is left unchanged."""
    docs = [d for d in world["invoices"] + world["payments"] if d["posted_at"] is not None]
    parent = {d["key"]: d["key"] for d in docs}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for payment in world["payments"]:
        if payment["posted_at"] is None:
            continue
        for ref in payment["invoice_refs"] or []:
            if ref in parent:
                parent[root(ref)] = root(payment["key"])
    groups: dict[str, list[dict]] = {}
    for document in docs:
        groups.setdefault(root(document["key"]), []).append(document)
    for key in sorted(groups):
        group = groups[key]
        if len(group) < 2:
            continue
        group.sort(key=lambda d: (_document_moment(d), "issued_at" not in d, int(d["key"][1:])))
        moments = sorted(d["posted_at"] for d in group)
        for index in range(len(moments) - 1, 0, -1):
            if moments[index - 1] >= moments[index]:
                moments[index - 1] = moments[index] - dt.timedelta(seconds=1)
        for document, moment in zip(group, moments):
            document["posted_at"] = moment


def outside_posting(rng: Rng, world: dict, issued: dt.datetime) -> dt.datetime:
    """A posting moment inside the period for a document dated just outside it (period_cutoff and
    I2): a lag drawn like a legitimate one, counted from the document or from the nearer edge."""
    start, end = world["period_start"], world["period_end"]
    lag = draw_lag(rng)
    if issued < start:
        posted = issued + dt.timedelta(seconds=lag)
        if posted < start:
            posted = start + dt.timedelta(seconds=rng.randint(MIN_LAG, lag))
    else:
        posted = end - dt.timedelta(seconds=rng.randint(1, lag))
    if not start <= posted < end:
        posted = moment_between(rng, start, end)
    return posted.astimezone(world["zone"])


def share_of(rng: Rng, amount: int) -> int:
    """A share (SHARE_PERCENT) of a positive document amount, at least one minor unit."""
    return max(1, amount * rng.randint(*SHARE_PERCENT) // 100)


def doc_zone(rng: Rng, world: dict, numerator: int = 3) -> dt.timezone:
    """The offset a document is written in: the company's, or with distractors sometimes another."""
    if world["distractors"] and rng.chance(numerator, 20):
        return parse_offset(rng.choice([o for o in OFFSETS if o != world["timezone"]]))
    return world["zone"]


def in_period(world: dict, moment: dt.datetime) -> bool:
    return world["period_start"] <= moment < world["period_end"]


def _rates(rng: Rng, config: dict, start: dt.datetime, end: dt.datetime) -> list[dict]:
    """Daily rates foreign -> base, FX_MARGIN_DAYS beyond both ends. With distractors, Saturday and
    Sunday repeat Friday's rate; without, every day differs from the day before."""
    base = config["base_currency"]
    first_day = start.date() - dt.timedelta(days=FX_MARGIN_DAYS)
    days = (end.date() - first_day).days + FX_MARGIN_DAYS
    table = []
    for currency in config["currencies"]:
        stream = rng.child(f"fx:{currency}")
        # Episode level: +-10% around the rough value; then a daily walk of at most +-0.8%.
        level = Decimal(_VALUE[currency]) / Decimal(_VALUE[base]) * Decimal(900 + stream.below(201)) / 1000
        rate = None
        for day in range(days):
            date = first_day + dt.timedelta(days=day)
            if not (config["distractors"] and date.weekday() >= 5 and rate is not None):
                step = stream.randint(-80, 80)
                while step == 0 and not config["distractors"]:
                    step = stream.randint(-80, 80)
                level = level * Decimal(10000 + step) / 10000
                rate = round_rate(level)
            table.append({"date": date.isoformat(), "source_currency": currency, "target_currency": base,
                          "rate": str(rate)})
    return table


def line_items(rng: Rng, currency: str, service: tuple[str, str]) -> list[dict]:
    items = []
    for _ in range(rng.randint(1, 3)):
        quantity = rng.randint(1, 5)
        unit_price = rng.randint(20, 900) * PRICE_UNIT[currency]
        items.append({"description": rng.choice(_SERVICES), "quantity": quantity,
                      "unit_price_minor": unit_price, "amount_minor": quantity * unit_price,
                      "service_start": service[0], "service_end": service[1]})
    return items


def service_halves(world: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    """The reporting period's service split in two halves, both inside the period (recurring twins)."""
    first, last = (dt.date.fromisoformat(day) for day in service_period(world, 0))
    middle = first + dt.timedelta(days=(last - first).days // 2 + 1)
    return ((first.isoformat(), (middle - dt.timedelta(days=1)).isoformat()),
            (middle.isoformat(), last.isoformat()))


def invoice_signature(invoice: dict) -> tuple:
    """Two invoices with the same signature are duplicates by definition (docs/SPEC.md)."""
    items = sorted((item["description"], item["quantity"], item["unit_price_minor"], item["amount_minor"],
                    item["service_start"], item["service_end"]) for item in invoice["line_items"])
    return (invoice["entity"], invoice["currency"], invoice["total_minor"], tuple(items))


def new_invoice(world: dict, key: str, entity: str, currency: str, issued: dt.datetime, terms: int,
                items: list[dict]) -> dict:
    """An unposted invoice; the caller sets `posted_at` (it needs a stream)."""
    return {"key": key, "entity": entity, "currency": currency, "issued_at": issued,
            "due_at": issued + dt.timedelta(days=terms), "total_minor": sum(i["amount_minor"] for i in items),
            "line_items": items, "posted_at": None, "rate_override": None}


def new_payment(key: str, entity: str, currency: str, moment: dt.datetime, amount: int,
                refs: list[str] | None) -> dict:
    return {"key": key, "entity": entity, "currency": currency, "timestamp": moment, "amount_minor": amount,
            "invoice_refs": refs, "posted_at": None, "rate_override": None, "gap_minor": 0}


def paid_like_clean(rng: Rng, world: dict, issued: dt.datetime) -> dt.datetime | None:
    """The clean story's payment rule for one invoice: paid with PAID_CHANCE when there is room in
    the period, at a moment between an hour after the invoice and the end. None means unpaid."""
    latest = world["period_end"] - EDGE_GAP
    if issued + PAY_AFTER >= latest or not rng.chance(*PAID_CHANCE):
        return None
    return moment_between(rng, issued + PAY_AFTER, latest)


def paid_outside(rng: Rng, world: dict, issued: dt.datetime) -> dt.datetime | None:
    """The payment rule for an invoice dated outside the period: paid with PAID_CHANCE, at a moment
    on the same side of the period (so the payment is not posted either). None means unpaid."""
    if issued < world["period_start"]:
        low, high = issued + PAY_AFTER, world["period_start"]
    else:
        low, high = issued + PAY_AFTER, world["period_end"] + dt.timedelta(days=FX_MARGIN_DAYS)
    if low >= high or not rng.chance(*PAID_CHANCE):
        return None
    return moment_between(rng, low, high)


def _edge_moment(rng: Rng, world: dict) -> dt.datetime:
    """A moment inside the period within six hours of one of its ends."""
    start, end = world["period_start"], world["period_end"] - EDGE_GAP
    if rng.chance(1, 2):
        return start + dt.timedelta(seconds=rng.below(6 * 3600))
    return end - dt.timedelta(seconds=1 + rng.below(6 * 3600))


def outside_moment(rng: Rng, world: dict, before: bool) -> dt.datetime:
    """A moment just outside the period, within BOUNDARY_DAYS: one in ten exactly on the edge
    (start - 1 s, or the end itself, which is already outside), four in ten within six hours."""
    band = rng.below(10)
    if band == 0:
        distance = 1 if before else 0
    elif band <= 4:
        distance = rng.randint(1, 6 * 3600)
    else:
        distance = rng.randint(1, BOUNDARY_DAYS * 86400 - 1)
    if before:
        return world["period_start"] - dt.timedelta(seconds=distance)
    return world["period_end"] + dt.timedelta(seconds=distance)


def _payment_rows(payment: dict, foreign: bool, unallocated: bool) -> int:
    refs = payment["invoice_refs"] or []
    return 1 + len(refs) + int(unallocated) + int(foreign)


def _story_digest(world: dict) -> str:
    """A digest of every document and background pair: keys the rate stream (docs/AUDIT.md L1)."""
    def moment(value):
        return None if value is None else value.isoformat()
    body = [[inv["key"], inv["entity"], inv["currency"], inv["total_minor"], moment(inv["issued_at"]),
             moment(inv["posted_at"])] for inv in world["invoices"]]
    body += [[pay["key"], pay["entity"], pay["currency"], pay["amount_minor"], moment(pay["timestamp"])]
             for pay in world["payments"]]
    body += [[pair["key"], pair["entity"], pair["amount_minor"], moment(pair["timestamp"])] for pair in world["background"]]
    return hashlib.sha256(json.dumps(body).encode("utf-8")).hexdigest()


def build_world(rng: Rng, config: dict) -> dict:
    start, end = period_bounds(config)
    base = config["base_currency"]
    world = {"base_currency": base, "timezone": config["timezone"], "zone": start.tzinfo,
             "period_start": start, "period_end": end, "months": config["period"]["months"],
             "distractors": config["distractors"],
             "currency_exponents": {c: CURRENCY_EXPONENTS[c] for c in [base, *config["currencies"]]}}

    low, high = config["size"]["invoices"]
    shape = rng.child("shape")
    invoice_count = shape.randint(low, high)
    customers = [f"E{i}" for i in range(shape.randint(*CUSTOMERS))]
    vendors = [f"E{len(customers) + i}" for i in range(shape.randint(3, 6))]
    world["entities"], world["customers"] = customers + vendors, customers
    distractors = config["distractors"]
    plan = rng.child("distractor_plan")
    # Neighbouring-period invoices: one per configured cutoff (its pool), plus legitimate extras.
    outside_count = config["errors"]["period_cutoff"] + (plan.randint(1, 3) if distractors else 0)
    recurring_pairs = plan.randint(1, max(1, invoice_count // 20)) if distractors else 0
    regular_count = invoice_count - outside_count - 2 * recurring_pairs

    invoices: list[dict] = []
    signatures: set = set()
    stream = rng.child("invoices")
    own_service = service_period(world, 0)

    def add(invoice: dict) -> bool:
        signature = invoice_signature(invoice)
        if signature in signatures:
            return False
        signatures.add(signature)
        invoices.append(invoice)
        return True

    def currency_of(s: Rng) -> str:
        return s.choice(config["currencies"]) if config["currencies"] and s.chance(2, 5) else base

    def inside_invoice(s: Rng, entity: str, currency: str, items: list[dict], edge: bool = False) -> dict:
        if edge:
            issued = _edge_moment(s, world).astimezone(doc_zone(s, world, numerator=10))
        else:
            issued = moment_between(s, start, end - EDGE_GAP).astimezone(doc_zone(s, world))
        invoice = new_invoice(world, f"I{len(invoices)}", entity, currency, issued, s.choice(TERMS_DAYS), items)
        invoice["posted_at"] = posting_lag(s, issued, end)
        return invoice

    for _ in range(regular_count):
        while True:  # bounded in practice: the signature space is far larger than the invoice count
            currency = currency_of(stream)
            items = line_items(stream, currency, own_service)
            edge = distractors and stream.chance(1, 12)
            if add(inside_invoice(stream, stream.choice(customers), currency, items, edge)):
                break

    # Recurring: the same customer, items and total for the two halves of the period's service.
    # Both are legitimate; their signatures differ by the service period (docs/AUDIT.md D3).
    stream = rng.child("recurring")
    halves = service_halves(world)
    for _ in range(recurring_pairs):
        while True:
            currency, entity = currency_of(stream), stream.choice(customers)
            items = line_items(stream, currency, halves[0])
            first = inside_invoice(stream, entity, currency, items)
            twin_items = [dict(item, service_start=halves[1][0], service_end=halves[1][1]) for item in items]
            second = inside_invoice(stream, entity, currency, twin_items)
            if invoice_signature(first) in signatures or invoice_signature(second) in signatures:
                continue
            add(first)
            second["key"] = f"I{len(invoices)}"
            add(second)
            break

    # Just outside the period on both sides: dated outside, not posted, service of that period.
    stream = rng.child("outside")
    for _ in range(outside_count):
        while True:
            before = stream.chance(1, 2)
            currency = currency_of(stream)
            items = line_items(stream, currency, service_period(world, -1 if before else 1))
            issued = outside_moment(stream, world, before).astimezone(doc_zone(stream, world, numerator=10))
            invoice = new_invoice(world, f"I{len(invoices)}", stream.choice(customers), currency, issued,
                                  stream.choice(TERMS_DAYS), items)
            if add(invoice):
                break

    world["invoices"] = invoices
    world["payments"] = _payments(rng.child("payments"), world, invoice_count)
    order_postings(world)
    issues = sum(config["errors"].values()) + interaction_issue_count(config["interaction"])
    world["background"] = _background(rng.child("background"), config, vendors, start, end,
                                      _background_cap(world, issues))
    fx_stream = Rng(f"{rng.child('fx').token(26)}|{_story_digest(world)}", "fx")
    world["fx"] = _rates(fx_stream, config, start, end)
    world["rates"] = {(row["date"], row["source_currency"]): Decimal(row["rate"]) for row in world["fx"]}
    return world


def _background_cap(world: dict, issues: int) -> int:
    """Background pairs that still fit: the whole ledger, with every issue's extra rows, stays within
    MAX_LEDGER_ROWS (config.check_size guarantees the configured minimum fits)."""
    base = world["base_currency"]
    rows = 2 * sum(1 for inv in world["invoices"] if inv["posted_at"] is not None)
    rows += sum(_payment_rows(pay, pay["currency"] != base, True) for pay in world["payments"]
                if pay["posted_at"] is not None)
    return max(0, (MAX_LEDGER_ROWS - rows - issues * ROWS_PER_ISSUE) // 2)


def _payments(stream: Rng, world: dict, invoice_count: int) -> list[dict]:
    """Payments of the clean story. The ledger budget (ROWS_PER_INVOICE rows per invoice) and the
    document budget (no more payments than invoices) are kept by skipping distractors."""
    end, base, distractors = world["period_end"], world["base_currency"], world["distractors"]
    latest = end - EDGE_GAP
    posted = [inv for inv in world["invoices"] if inv["posted_at"] is not None]
    budget = {"rows": ROWS_PER_INVOICE * invoice_count - 2 * len(posted), "docs": invoice_count}
    payments: list[dict] = []

    def add(payment: dict, unallocated: bool = False) -> bool:
        rows = _payment_rows(payment, payment["currency"] != base, unallocated) if in_period(world, payment["timestamp"]) else 0
        if rows > budget["rows"] or budget["docs"] < 1:
            return False
        budget["rows"] -= rows
        budget["docs"] -= 1
        payment["key"] = f"P{len(payments)}"
        payment["timestamp"] = payment["timestamp"].astimezone(doc_zone(stream, world))
        if in_period(world, payment["timestamp"]):
            payment["posted_at"] = posting_lag(stream, payment["timestamp"], end)
        payments.append(payment)
        return True

    # Invoices of the neighbouring periods first: their payment chance must not depend on the budget.
    for invoice in world["invoices"]:
        if invoice["posted_at"] is None:
            moment = paid_outside(stream, world, invoice["issued_at"])
            if moment is not None:
                add(new_payment("", invoice["entity"], invoice["currency"], moment, invoice["total_minor"],
                                [invoice["key"]]))

    plans = []
    for invoice in posted:
        moment = paid_like_clean(stream, world, invoice["issued_at"])
        if moment is not None:
            plans.append((invoice, moment))
    used: set[str] = set()
    kinds: dict[str, str] = {}
    for index, (invoice, moment) in enumerate(plans):
        if invoice["key"] in used:
            continue
        used.add(invoice["key"])
        # One payment for two invoices of the same customer and currency, both paid in full.
        partner = next((other for other, _ in plans[index + 1:] if other["key"] not in used
                        and other["entity"] == invoice["entity"] and other["currency"] == invoice["currency"]), None)
        if partner is not None and stream.chance(1, 6):
            used.add(partner["key"])
            moment = max(moment, partner["issued_at"] + PAY_AFTER)
            if moment >= latest:
                moment = moment_between(stream, max(invoice["issued_at"], partner["issued_at"]) + PAY_AFTER, latest)
            add(new_payment("", invoice["entity"], invoice["currency"], moment,
                            invoice["total_minor"] + partner["total_minor"], sorted([invoice["key"], partner["key"]])))
            kinds[invoice["key"]] = kinds[partner["key"]] = "multi"
            continue
        roll = stream.below(100) if distractors else 99
        total = invoice["total_minor"]
        if roll < 6:  # overpayment: the excess legitimately stays on unallocated
            amount = total + share_of(stream, total)
            if add(new_payment("", invoice["entity"], invoice["currency"], moment, amount, [invoice["key"]]), True):
                kinds[invoice["key"]] = "over"
                continue
        elif roll < 18:  # partial (and, for half of them, a second payment for the rest: split)
            first = max(1, total * stream.randint(20, 80) // 100)
            if add(new_payment("", invoice["entity"], invoice["currency"], moment, first, [invoice["key"]])):
                kinds[invoice["key"]] = "partial"
                if roll >= 12 and moment + PAY_AFTER < latest:
                    second = moment_between(stream, moment + PAY_AFTER, latest)
                    if add(new_payment("", invoice["entity"], invoice["currency"], second, total - first,
                                       [invoice["key"]])):
                        kinds[invoice["key"]] = "split"
                continue
        if add(new_payment("", invoice["entity"], invoice["currency"], moment, total, [invoice["key"]])):
            kinds[invoice["key"]] = "full"

    if not distractors:
        return payments
    # Refunds of fully paid invoices (never of overpaid ones): negative amount, one reference.
    by_key = {inv["key"]: inv for inv in posted}
    for key in sorted(kinds, key=lambda k: int(k[1:])):
        if kinds[key] != "full" or not stream.chance(1, 12):
            continue
        paid_at = next(p["timestamp"] for p in payments if p["invoice_refs"] == [key])
        if paid_at + PAY_AFTER >= latest:
            continue
        invoice = by_key[key]
        amount = -max(1, invoice["total_minor"] * stream.randint(10, 100) // 100)
        add(new_payment("", invoice["entity"], invoice["currency"], moment_between(stream, paid_at + PAY_AFTER, latest),
                        amount, [key]))
    # Advances: no invoice reference, the whole amount on unallocated.
    for _ in range(stream.randint(0, 2)):
        currency = stream.choice([base, *sorted(set(world["currency_exponents"]) - {base})])
        moment = moment_between(stream, world["period_start"], latest)
        add(new_payment("", stream.choice(world["customers"]), currency, moment,
                        stream.randint(20, 900) * PRICE_UNIT[currency], None), True)
    # Payments just after the period for invoices still open in it: dated outside, not posted.
    unpaid = [inv for inv in posted if inv["key"] not in kinds]
    for _ in range(stream.randint(0, 2)):
        if not unpaid:
            break
        invoice = unpaid.pop(stream.below(len(unpaid)))
        moment = outside_moment(stream, world, before=False)
        add(new_payment("", invoice["entity"], invoice["currency"], moment, invoice["total_minor"], [invoice["key"]]))
    return payments


def _background(stream: Rng, config: dict, vendors: list[str], start: dt.datetime, end: dt.datetime,
                cap: int) -> list[dict]:
    """Background activity without documents: balanced expense/cash pairs, at most `cap` of them."""
    background, taken = [], set()
    low, high = config["size"]["background_pairs"]
    for _ in range(min(stream.randint(low, high), cap)):
        while True:
            moment, vendor = moment_between(stream, start, end), stream.choice(vendors)
            if (moment, vendor) not in taken:  # the pair is its own event group
                taken.add((moment, vendor))
                break
        background.append({"key": f"B{len(background)}", "timestamp": moment, "entity": vendor,
                           "amount_minor": stream.randint(5, 4000) * 1000})
    return background
