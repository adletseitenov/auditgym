"""Error operators. Each one changes documents of the story; the books are rebuilt from them.

Operators work on internal keys ("I3", "P5", "D:I3" for the document of invoice I3, "R:I3:dr" for
a ledger row); surface.py assigns public ids afterwards, so nothing about an injected record's id
or position tells it apart. Each operator has its own random stream (`inject:<type>`) with one
target order drawn over the story's original documents, so enabling one type does not move the
targets of another unless they collide. A document carries at most one error, except inside an
interaction scenario.

An injection is described by a dict in internal keys:
    {"type", "affected": [doc keys], "amount_minor", "currency", "evidence": [[keys]], "params"}
inject_errors adds "before"/"after" (the documents and rows the step changed, internal form);
generate.py turns it into a truth issue and a provenance step in public ids.

Rules decided here (docs/DECISIONS.md B and the orchestrator's rulings):
- duplicate_invoice: the copy is a new invoice with the original's customer, currency, items and
  total, dated at another moment of the period in an offset drawn like any document's, posted at
  its own date's rate after a normal posting lag, and paid with the clean story's probability by a
  separate full payment (a double payment). Refunded invoices are not copied.
- period_cutoff: any posted invoice dated in the period, paid or not (targets are independent of
  payment status). Its date moves at most CUTOFF_MAX_DAYS outside the period, its line items'
  service period moves to the neighbouring period, its due date keeps its terms; the postings keep
  their moment and are re-booked at the new date's rate (so it is not also an fx_error). Payments of
  the invoice keep their documents; the books close the receivable at the new booked amount and the
  fx difference follows (payment rule in books.py). A move whose new signature equals another
  invoice's is skipped (that candidate only).
- fx_error: targets are posted foreign invoices dated in the period and posted ordinary positive
  payments (not advances or refunds). The wrong rate is either another date's rate (up to 5 days
  away, or the document's naive date when its own offset changes the date) or the right rate
  perturbed by 0.8-6 %; the posted value must differ from the policy value by >= 100 minor units
  and >= 0.5 %.
- allocation_gap: a posted payment in the period with exactly one reference and a positive amount
  not above the open total, being the only payment document of an invoice posted in the period; a
  share (5-60 %) of its receivable credit moves to one unallocated row.
"""
from __future__ import annotations

import copy
import datetime as dt
from decimal import Decimal

from .books import booked, build_books, payment_parts, rate_on, to_base
from .money import convert, round_rate
from .rng import Rng
from .world import (EDGE_GAP, OFFSETS, PAY_AFTER, TERMS_DAYS, doc_zone, in_period, invoice_signature,
                    moment_between, new_invoice, new_payment, paid_like_clean, posting_lag, service_period)

OPERATOR_ORDER = ("duplicate_invoice", "period_cutoff", "fx_error", "allocation_gap")
# Moved document dates stay this many days from the period edge, inside the fx margin of the story.
CUTOFF_MAX_DAYS = 4
CUTOFF_NEAR_SECONDS = 6 * 3600
FX_MIN_MINOR = 100
FX_MIN_PER_MILLE = 5  # 0.5 %
WRONG_DATE_DAYS = 5


class NoTarget(Exception):
    """The story has no suitable record for an operator; generate() retries with another attempt."""


def doc_key(key: str) -> str:
    return "D:" + key


def _next_key(prefix: str, keys) -> str:
    used = [int(key[len(prefix):]) for key in keys if key.startswith(prefix) and key[len(prefix):].isdigit()]
    return f"{prefix}{max(used, default=-1) + 1}"


def _copy_world(world: dict) -> dict:
    """Documents are deep-copied, shared read-only parts (fx, background) are not."""
    return dict(world, invoices=copy.deepcopy(world["invoices"]), payments=copy.deepcopy(world["payments"]))


def _invoice(world: dict, key: str) -> dict:
    return next(inv for inv in world["invoices"] if inv["key"] == key)


def _payment(world: dict, key: str) -> dict:
    return next(pay for pay in world["payments"] if pay["key"] == key)


def payments_of(world: dict, invoice_key: str) -> list[dict]:
    return [p for p in world["payments"] if invoice_key in (p["invoice_refs"] or [])]


def _posted_inside(world: dict, invoice: dict) -> bool:
    return invoice["posted_at"] is not None and in_period(world, invoice["issued_at"])


def _order(rng: Rng, keys: list[str]) -> list[str]:
    order = sorted(keys)
    rng.shuffle(order)
    return order


def _signature_taken(world: dict, invoice: dict) -> bool:
    signature = invoice_signature(invoice)
    return any(invoice_signature(other) == signature for other in world["invoices"] if other["key"] != invoice["key"])


def _mark_with_payments(world: dict, used: set, invoice_key: str) -> None:
    used.add(invoice_key)
    used.update(p["key"] for p in payments_of(world, invoice_key))


# ------------------------------------------------------------------ duplicate_invoice
def _add_copy(rng: Rng, world: dict, original: dict) -> dict:
    start, end = world["period_start"], world["period_end"]
    while True:  # one draw in practice: the period has millions of seconds
        issued = moment_between(rng, start, end - EDGE_GAP)
        if issued != original["issued_at"]:
            break
    issued = issued.astimezone(doc_zone(rng, world))
    copy_invoice = new_invoice(world, _next_key("I", [i["key"] for i in world["invoices"]]), original["entity"],
                               original["currency"], issued, rng.choice(TERMS_DAYS),
                               copy.deepcopy(original["line_items"]))
    copy_invoice["posted_at"] = posting_lag(rng, issued, end)
    world["invoices"].append(copy_invoice)
    return copy_invoice


def _duplicate_evidence(a: str, b: str) -> list[list[str]]:
    return [[doc_key(a), doc_key(b), f"R:{a}:{x}", f"R:{b}:{y}"] for x in ("dr", "cr") for y in ("dr", "cr")]


def duplicate_invoice(rng: Rng, world: dict, order: list[str], used: set) -> tuple[dict, dict]:
    candidates = [key for key in order if key.startswith("I") and key not in used
                  and _posted_inside(world, _invoice(world, key))
                  and not any(p["amount_minor"] < 0 for p in payments_of(world, key))]
    if not candidates:
        raise NoTarget("duplicate_invoice")
    world = _copy_world(world)
    original = _invoice(world, candidates[0])
    copy_invoice = _add_copy(rng, world, original)
    paid_at = paid_like_clean(rng, world, copy_invoice["issued_at"])
    params = {"original_doc_id": doc_key(original["key"]), "copy_paid": paid_at is not None}
    if paid_at is not None:  # a double payment: the customer paid the copy as well
        payment = new_payment(_next_key("P", [p["key"] for p in world["payments"]]), original["entity"],
                              original["currency"], paid_at.astimezone(doc_zone(rng, world)),
                              original["total_minor"], [copy_invoice["key"]])
        payment["posted_at"] = posting_lag(rng, payment["timestamp"], world["period_end"])
        world["payments"].append(payment)
        used.add(payment["key"])
    used.update({original["key"], copy_invoice["key"]})
    docs = sorted([doc_key(original["key"]), doc_key(copy_invoice["key"])])
    return world, {"type": "duplicate_invoice", "affected": docs, "amount_minor": original["total_minor"],
                   "currency": original["currency"],
                   "evidence": _duplicate_evidence(original["key"], copy_invoice["key"]), "params": params}


# ------------------------------------------------------------------ period_cutoff
def _cutoff_moment(rng: Rng, start: dt.datetime, end: dt.datetime) -> tuple[dt.datetime, str, int]:
    """A moment just outside [start, end): (moment, direction, seconds from the edge).

    One case in ten sits exactly on the edge (start - 1 s, or end itself, which is already
    outside the half-open period); four in ten within six hours; the rest within four days.
    """
    direction = "before_start" if rng.chance(1, 2) else "after_end"
    band = rng.below(10)
    if band == 0:
        distance = 1 if direction == "before_start" else 0
    elif band <= 4:
        distance = rng.randint(1, CUTOFF_NEAR_SECONDS)
    else:
        distance = rng.randint(1, CUTOFF_MAX_DAYS * 86400 - 1)
    if direction == "before_start":
        return start - dt.timedelta(seconds=distance), direction, distance
    return end + dt.timedelta(seconds=distance), direction, distance


def _move_outside(rng: Rng, world: dict, invoice: dict, shift_service: bool) -> dict:
    """Move an invoice's date just outside the period (in its own offset); postings stay."""
    moved_at, direction, distance = _cutoff_moment(rng, world["period_start"], world["period_end"])
    moved_at = moved_at.astimezone(invoice["issued_at"].tzinfo)
    params = {"direction": direction, "seconds_outside": distance,
              "original_issued_at": invoice["issued_at"].isoformat()}
    invoice["due_at"] = invoice["due_at"] + (moved_at - invoice["issued_at"])
    invoice["issued_at"] = moved_at
    if shift_service:
        first, last = service_period(world, -1 if direction == "before_start" else 1)
        for item in invoice["line_items"]:
            item["service_start"], item["service_end"] = first, last
    return params


def period_cutoff(rng: Rng, world: dict, order: list[str], used: set) -> tuple[dict, dict]:
    for key in order:
        if not key.startswith("I") or key in used or not _posted_inside(world, _invoice(world, key)):
            continue
        attempt = _copy_world(world)
        invoice = _invoice(attempt, key)
        params = _move_outside(rng, attempt, invoice, shift_service=True)
        if _signature_taken(attempt, invoice):
            continue  # ruling 1: the shifted invoice would duplicate another one; try the next
        params["paid"] = bool([p for p in payments_of(attempt, key) if p["posted_at"] is not None])
        _mark_with_payments(attempt, used, key)
        doc = doc_key(key)
        return attempt, {"type": "period_cutoff", "affected": [doc], "amount_minor": invoice["total_minor"],
                         "currency": invoice["currency"],
                         "evidence": [[doc, f"R:{key}:dr"], [doc, f"R:{key}:cr"]], "params": params}
    raise NoTarget("period_cutoff")


# ------------------------------------------------------------------ fx_error
def _far_enough(wrong: int, correct: int) -> bool:
    difference = abs(wrong - correct)
    return difference >= FX_MIN_MINOR and difference * 1000 >= FX_MIN_PER_MILLE * abs(correct)


def wrong_rate(rng: Rng, world: dict, amount: int, currency: str, moment: dt.datetime,
               prefer_naive: bool = False) -> tuple[Decimal, dict] | None:
    """A rate that a careless bookkeeper could use, far enough from the policy value, or None."""
    exponents, base = world["currency_exponents"], world["base_currency"]
    right = rate_on(world, currency, moment)
    correct = convert(amount, right, exponents[currency], exponents[base])

    def ok(rate: Decimal) -> bool:
        return rate != right and _far_enough(convert(amount, rate, exponents[currency], exponents[base]), correct)

    local = moment.astimezone(world["zone"]).date()
    dates = [(local + dt.timedelta(days=d)).isoformat() for d in range(-WRONG_DATE_DAYS, WRONG_DATE_DAYS + 1) if d]
    rng.shuffle(dates)
    naive = moment.date().isoformat()
    if naive != local.isoformat():
        dates = [naive] + [d for d in dates if d != naive] if prefer_naive else dates
    modes = ["wrong_date", "wrong_rate"]
    rng.shuffle(modes)
    if prefer_naive and naive != local.isoformat():
        modes = ["wrong_date", "wrong_rate"]
    for mode in modes:
        if mode == "wrong_date":
            for date in dates:
                rate = world["rates"].get((date, currency))
                if rate is not None and ok(rate):
                    return rate, {"mode": mode, "rate_date": date, "used_rate": str(rate), "policy_rate": str(right)}
        else:
            for _ in range(8):
                per_mille = rng.randint(8, 60) * (1 if rng.chance(1, 2) else -1)
                rate = round_rate(right * Decimal(1000 + per_mille) / 1000)
                if ok(rate):
                    return rate, {"mode": mode, "rate_date": None, "used_rate": str(rate), "policy_rate": str(right)}
    return None


def fx_error(rng: Rng, world: dict, order: list[str], used: set) -> tuple[dict, dict]:
    base = world["base_currency"]
    for key in order:
        if key in used:
            continue
        if key.startswith("I"):
            record = _invoice(world, key)
            if record["currency"] == base or not _posted_inside(world, record):
                continue
            amount, moment = record["total_minor"], record["issued_at"]
        else:
            record = _payment(world, key)
            if (record["currency"] == base or record["posted_at"] is None or record["invoice_refs"] is None
                    or record["amount_minor"] <= 0):
                continue
            amount, moment = record["amount_minor"], record["timestamp"]
        chosen = wrong_rate(rng, world, amount, record["currency"], moment)
        if chosen is None:
            continue
        rate, params = chosen
        world = _copy_world(world)
        doc = doc_key(key)
        correct = to_base(world, amount, record["currency"], moment)
        wrong = to_base(world, amount, record["currency"], moment, rate)
        if key.startswith("I"):
            _invoice(world, key)["rate_override"] = rate
            _mark_with_payments(world, used, key)
            evidence = [[doc, f"R:{key}:dr"], [doc, f"R:{key}:cr"]]
        else:
            payment = _payment(world, key)
            payment["rate_override"] = rate
            used.add(key)
            used.update(payment["invoice_refs"])
            evidence = [[doc, f"R:{key}:cash"]]
        return world, {"type": "fx_error", "affected": [doc], "amount_minor": abs(wrong - correct),
                       "currency": base, "evidence": evidence, "params": params}
    raise NoTarget("fx_error")


# ------------------------------------------------------------------ allocation_gap
def _gap(rng: Rng, closing: int) -> int:
    return min(closing - 1, max(1, closing * rng.randint(5, 60) // 100))


def allocation_gap(rng: Rng, world: dict, order: list[str], used: set) -> tuple[dict, dict]:
    parts = None
    for key in order:
        if key in used or not key.startswith("P"):
            continue
        payment = _payment(world, key)
        refs = payment["invoice_refs"]
        if (payment["posted_at"] is None or refs is None or len(refs) != 1 or payment["amount_minor"] <= 0
                or refs[0] in used or payment["rate_override"] is not None):
            continue
        invoice = _invoice(world, refs[0])
        if (not _posted_inside(world, invoice) or payment["amount_minor"] > invoice["total_minor"]
                or [p["key"] for p in payments_of(world, invoice["key"])] != [key]):
            continue
        parts = parts or payment_parts(world)
        closing = parts[key]["closes"][invoice["key"]]
        if closing < 2:
            continue
        gap = _gap(rng, closing)
        world = _copy_world(world)
        _payment(world, key)["gap_minor"] = gap
        used.update({key, invoice["key"]})
        docs = sorted([doc_key(key), doc_key(invoice["key"])])
        return world, {"type": "allocation_gap", "affected": docs, "amount_minor": gap, "currency": world["base_currency"],
                       "evidence": [docs + [f"R:{key}:un"]], "params": {"closing_before_minor": closing}}
    raise NoTarget("allocation_gap")


OPERATORS = {"duplicate_invoice": duplicate_invoice, "period_cutoff": period_cutoff,
             "fx_error": fx_error, "allocation_gap": allocation_gap}


# ------------------------------------------------------------------ interaction scenarios
def _naive_crossing_zone(rng: Rng, world: dict, moment: dt.datetime) -> dt.timezone:
    """An offset in which `moment` has another calendar date than in the company's, if one exists."""
    from .money import parse_offset

    offsets = list(OFFSETS)
    rng.shuffle(offsets)
    local = moment.astimezone(world["zone"]).date()
    for offset in offsets:
        zone = parse_offset(offset)
        if moment.astimezone(zone).date() != local:
            return zone
    return world["zone"]


def interaction_i1(rng: Rng, world: dict, order: list[str], used: set) -> list[tuple[dict, dict]]:
    """A partial payment in a foreign currency near the period end carrying an allocation_gap and an
    fx_error. Gap: a share of the receivable credit (which is the policy value of the payment) on
    unallocated. fx: cash at a wrong rate, the difference on fx_difference. The two amounts are read
    from different rows and do not depend on each other or on the order of the two steps."""
    base, end = world["base_currency"], world["period_end"]
    latest = end - EDGE_GAP - dt.timedelta(seconds=60)
    for key in order:
        if key in used or not key.startswith("I"):
            continue
        invoice = _invoice(world, key)
        if invoice["currency"] == base or not _posted_inside(world, invoice) or payments_of(world, key):
            continue
        low = max(invoice["issued_at"] + PAY_AFTER, end - dt.timedelta(hours=48))
        if low >= latest:
            continue
        moment = moment_between(rng, low, latest)
        zone = _naive_crossing_zone(rng, world, moment) if rng.chance(1, 2) else doc_zone(rng, world)
        moment = moment.astimezone(zone)
        amount = max(1, invoice["total_minor"] * rng.randint(20, 80) // 100)
        chosen = wrong_rate(rng, world, amount, invoice["currency"], moment, prefer_naive=True)
        correct = to_base(world, amount, invoice["currency"], moment)
        if chosen is None or correct < 2 or correct >= booked(world, invoice):
            continue
        rate, fx_params = chosen
        gap = _gap(rng, correct)
        first = _copy_world(world)
        payment = new_payment(_next_key("P", [p["key"] for p in first["payments"]]), invoice["entity"],
                              invoice["currency"], moment, amount, [key])
        payment["posted_at"] = posting_lag(rng, moment, end)
        payment["gap_minor"] = gap
        first["payments"].append(payment)
        pkey = payment["key"]
        docs = sorted([doc_key(pkey), doc_key(key)])
        gap_issue = {"type": "allocation_gap", "affected": docs, "amount_minor": gap, "currency": base,
                     "evidence": [docs + [f"R:{pkey}:un"]], "params": {"scenario": "I1", "closing_before_minor": correct}}
        second = _copy_world(first)
        _payment(second, pkey)["rate_override"] = rate
        wrong = to_base(second, amount, invoice["currency"], moment, rate)
        fx_issue = {"type": "fx_error", "affected": [doc_key(pkey)], "amount_minor": abs(wrong - correct),
                    "currency": base, "evidence": [[doc_key(pkey), f"R:{pkey}:cash"]],
                    "params": dict(fx_params, scenario="I1")}
        used.update({key, pkey})
        return [(first, gap_issue), (second, fx_issue)]
    raise NoTarget("I1")


def interaction_i2(rng: Rng, world: dict, order: list[str], used: set) -> list[tuple[dict, dict]]:
    """A duplicate whose copy is dated outside the period but posted in it. The copy keeps the
    original's line items and service period exactly (ruling 2) and is not paid."""
    for key in order:
        if key in used or not key.startswith("I"):
            continue
        original = _invoice(world, key)
        if not _posted_inside(world, original) or any(p["amount_minor"] < 0 for p in payments_of(world, key)):
            continue
        first = _copy_world(world)
        copy_invoice = _add_copy(rng, first, _invoice(first, key))
        ckey = copy_invoice["key"]
        docs = sorted([doc_key(key), doc_key(ckey)])
        dup_issue = {"type": "duplicate_invoice", "affected": docs, "amount_minor": original["total_minor"],
                     "currency": original["currency"], "evidence": _duplicate_evidence(key, ckey),
                     "params": {"scenario": "I2", "original_doc_id": doc_key(key)}}
        second = _copy_world(first)
        params = _move_outside(rng, second, _invoice(second, ckey), shift_service=False)
        cut_issue = {"type": "period_cutoff", "affected": [doc_key(ckey)], "amount_minor": original["total_minor"],
                     "currency": original["currency"],
                     "evidence": [[doc_key(ckey), f"R:{ckey}:dr"], [doc_key(ckey), f"R:{ckey}:cr"]],
                     "params": dict(params, scenario="I2")}
        used.update({key, ckey})
        return [(first, dup_issue), (second, cut_issue)]
    raise NoTarget("I2")


INTERACTIONS = {"I1": interaction_i1, "I2": interaction_i2}


# ------------------------------------------------------------------ driver
def _snapshot(world: dict) -> tuple[dict, dict]:
    docs = {inv["key"]: inv for inv in world["invoices"]}
    docs.update({pay["key"]: pay for pay in world["payments"]})
    return docs, {row["key"]: row for row in build_books(world)}


def _diff(before: dict, after: dict) -> tuple[list, list]:
    old = [before[key] for key in sorted(before) if after.get(key) != before[key]]
    new = [after[key] for key in sorted(after) if before.get(key) != after[key]]
    return copy.deepcopy(old), copy.deepcopy(new)


def inject_errors(root: Rng, world: dict, config: dict) -> tuple[dict, list[dict]]:
    """Apply every configured operator in a fixed order, then the interaction scenario.
    Raises NoTarget when the story is unsuitable."""
    used: set[str] = set()
    story_docs = [inv["key"] for inv in world["invoices"]] + [pay["key"] for pay in world["payments"]]
    injections: list[dict] = []
    state = _snapshot(world)

    def record(new_world: dict, injection: dict) -> None:
        nonlocal state
        new_state = _snapshot(new_world)
        docs_before, docs_after = _diff(state[0], new_state[0])
        rows_before, rows_after = _diff(state[1], new_state[1])
        injection["before"], injection["after"] = docs_before + rows_before, docs_after + rows_after
        injection["operator"] = injection["params"].get("scenario", injection["type"])
        injections.append(injection)
        state = new_state

    for name in OPERATOR_ORDER:
        count = config["errors"].get(name, 0)
        if not count:
            continue
        stream = root.child(f"inject:{name}")
        order = _order(stream.child("targets"), story_docs)
        for case in range(count):
            world, injection = OPERATORS[name](stream.child(f"case:{case}"), world, order, used)
            record(world, injection)
    scenario = config["interaction"]
    if scenario != "none":
        stream = root.child(f"interaction:{scenario}")
        order = _order(stream.child("targets"), story_docs)
        for world, injection in INTERACTIONS[scenario](stream.child("case"), world, order, used):
            record(world, injection)
    return world, injections
