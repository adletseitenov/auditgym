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

Decoys of the story (world.py: resent copies, invoices for a neighbouring period's service, and
payments referring to them) are never targets, and a source of a resent copy is never copied again
(duplicate_invoice, I2): the truth never contains a decoy (docs/DESIGN.md).

Rules decided here (docs/DECISIONS.md B and the orchestrator's rulings):
- duplicate_invoice: the copy is a new invoice with the original's customer, currency, items and
  total, dated at another moment of the period in an offset drawn like any document's, posted at
  its own date's rate after a normal posting lag, and paid with the clean story's probability by a
  separate full payment (a double payment). Refunded invoices are not copied, nor an invoice that
  already has a resent copy (world.py decoy A): a signature never has more than two invoices.
- period_cutoff (docs/AUDIT.md D4): one of the story's legitimately unposted invoices of a
  neighbouring period (dated at most BOUNDARY_DAYS outside, service of that period, paid or not
  with the usual probability by a payment on its own side of the period) is posted inside the
  period at its own date's rate, after a lag drawn like a legitimate one. No document changes; the
  ledger gains exactly the invoice's two rows. A resent copy (decoy A) is never posted: posting it
  would also make a duplicate of its source.
- fx_error: targets are posted foreign invoices dated in the period and posted ordinary positive
  payments (not advances or refunds). The wrong rate is either another date's rate (up to 5 days
  away, or the document's naive date when its own offset changes the date) or the right rate
  perturbed by 0.8-6 %; the posted value must differ from the policy value by >= 100 minor units
  and >= 0.5 %. Evidence: the document, a conversion row and the fx key of the document's date.
- allocation_gap: a posted payment in the period with exactly one reference and a positive amount
  not above the open total, being the only payment document of an invoice posted in the period; a
  share of the payment (world.share_of, the same draw as an overpayment's excess) at the payment's
  rate moves from its receivable credit to one unallocated row. Evidence: both documents, the
  unallocated row, the invoice's receivable debit and the payment's receivable credit.
"""
from __future__ import annotations

import copy
import datetime as dt
from decimal import Decimal

from .books import build_books, payment_parts, rate_on, to_base
from .money import convert, fx_key, round_rate
from .rng import Rng
from .world import (EDGE_GAP, PAY_AFTER, TERMS_DAYS, doc_zone, in_period, moment_between, new_invoice,
                    new_payment, order_postings, outside_moment, outside_posting, paid_like_clean, paid_outside,
                    posting_lag, share_of)

OPERATOR_ORDER = ("duplicate_invoice", "period_cutoff", "fx_error", "allocation_gap")
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


def _decoy(invoice: dict) -> bool:
    """A decoy of the story (world.py: a resent copy or an invoice for a neighbouring period's service)
    is never the target of an operator, so no decoy is part of the truth."""
    return bool(invoice.get("decoy"))


def _order(rng: Rng, keys: list[str]) -> list[str]:
    order = sorted(keys)
    rng.shuffle(order)
    return order


def _fx_evidence_key(world: dict, currency: str, moment: dt.datetime) -> str:
    """Internal form of the fx key of a document's company-local date ("FX:" + public key)."""
    date = moment.astimezone(world["zone"]).date().isoformat()
    return "FX:" + fx_key(date, currency, world["base_currency"])


def _gap_evidence(payment_key: str, invoice_key: str) -> list[list[str]]:
    return [sorted([doc_key(payment_key), doc_key(invoice_key)]) + [
        f"R:{payment_key}:un", f"R:{invoice_key}:dr", f"R:{payment_key}:ar:{invoice_key}"]]


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
                  and _posted_inside(world, _invoice(world, key)) and not _decoy(_invoice(world, key))
                  and not _invoice(world, key).get("has_copy")
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
        order_postings(world)
        used.add(payment["key"])
    used.update({original["key"], copy_invoice["key"]})
    docs = sorted([doc_key(original["key"]), doc_key(copy_invoice["key"])])
    return world, {"type": "duplicate_invoice", "affected": docs, "amount_minor": original["total_minor"],
                   "currency": original["currency"],
                   "evidence": _duplicate_evidence(original["key"], copy_invoice["key"]), "params": params}


# ------------------------------------------------------------------ period_cutoff
def _outside_unposted(world: dict, invoice: dict) -> bool:
    return invoice["posted_at"] is None and not in_period(world, invoice["issued_at"])


def period_cutoff(rng: Rng, world: dict, order: list[str], used: set) -> tuple[dict, dict]:
    for key in order:
        if (not key.startswith("I") or key in used or not _outside_unposted(world, _invoice(world, key))
                or _decoy(_invoice(world, key))):
            continue
        world = _copy_world(world)
        invoice = _invoice(world, key)
        invoice["posted_at"] = outside_posting(rng, world, invoice["issued_at"])
        params = {"direction": "before_start" if invoice["issued_at"] < world["period_start"] else "after_end",
                  "paid": bool(payments_of(world, key))}
        _mark_with_payments(world, used, key)
        doc = doc_key(key)
        return world, {"type": "period_cutoff", "affected": [doc], "amount_minor": invoice["total_minor"],
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
            if record["currency"] == base or not _posted_inside(world, record) or _decoy(record):
                continue
            amount, moment = record["total_minor"], record["issued_at"]
        else:
            record = _payment(world, key)
            if (record["currency"] == base or record["posted_at"] is None or record["invoice_refs"] is None
                    or record["amount_minor"] <= 0 or any(_decoy(_invoice(world, ref)) for ref in record["invoice_refs"])):
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
        rate_key = _fx_evidence_key(world, record["currency"], moment)
        if key.startswith("I"):
            _invoice(world, key)["rate_override"] = rate
            _mark_with_payments(world, used, key)
            evidence = [[doc, f"R:{key}:dr", rate_key], [doc, f"R:{key}:cr", rate_key]]
        else:
            payment = _payment(world, key)
            payment["rate_override"] = rate
            used.add(key)
            used.update(payment["invoice_refs"])
            evidence = [[doc, f"R:{key}:cash", rate_key]]
        return world, {"type": "fx_error", "affected": [doc], "amount_minor": abs(wrong - correct),
                       "currency": base, "evidence": evidence, "params": params}
    raise NoTarget("fx_error")


# ------------------------------------------------------------------ allocation_gap
def _gap(rng: Rng, world: dict, payment: dict, closing: int) -> int:
    """A share of the payment converted at its own date's rate (the same draw and conversion as an
    overpayment's excess), leaving at least one minor unit of the receivable credit."""
    share = to_base(world, share_of(rng, payment["amount_minor"]), payment["currency"], payment["timestamp"])
    return max(1, min(closing - 1, share))


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
        if (not _posted_inside(world, invoice) or _decoy(invoice) or payment["amount_minor"] > invoice["total_minor"]
                or [p["key"] for p in payments_of(world, invoice["key"])] != [key]):
            continue
        parts = parts or payment_parts(world)
        closing = parts[key]["closes"][invoice["key"]]
        if closing < 2:
            continue
        gap = _gap(rng, world, payment, closing)
        world = _copy_world(world)
        _payment(world, key)["gap_minor"] = gap
        used.update({key, invoice["key"]})
        docs = sorted([doc_key(key), doc_key(invoice["key"])])
        return world, {"type": "allocation_gap", "affected": docs, "amount_minor": gap, "currency": world["base_currency"],
                       "evidence": _gap_evidence(key, invoice["key"]), "params": {"closing_before_minor": closing}}
    raise NoTarget("allocation_gap")


OPERATORS = {"duplicate_invoice": duplicate_invoice, "period_cutoff": period_cutoff,
             "fx_error": fx_error, "allocation_gap": allocation_gap}


# ------------------------------------------------------------------ interaction scenarios
def interaction_i1(rng: Rng, world: dict, order: list[str], used: set) -> list[tuple[dict, dict]]:
    """A partial payment in a foreign currency near the period end carrying an allocation_gap and an
    fx_error. Its offset is drawn like any payment's. Gap: a share of the payment at its policy rate
    moved from the receivable credit (the proportional closing at the booked rate) to unallocated.
    fx: cash at a wrong rate, the difference on fx_difference. The two amounts are read from
    different rows and do not depend on each other or on the order of the two steps."""
    base, end = world["base_currency"], world["period_end"]
    latest = end - EDGE_GAP - dt.timedelta(seconds=60)
    for key in order:
        if key in used or not key.startswith("I"):
            continue
        invoice = _invoice(world, key)
        if (invoice["currency"] == base or not _posted_inside(world, invoice) or _decoy(invoice)
                or payments_of(world, key)):
            continue
        low = max(invoice["issued_at"] + PAY_AFTER, end - dt.timedelta(hours=48))
        if low >= latest:
            continue
        moment = moment_between(rng, low, latest).astimezone(doc_zone(rng, world))
        amount = max(1, invoice["total_minor"] * rng.randint(20, 80) // 100)
        chosen = wrong_rate(rng, world, amount, invoice["currency"], moment, prefer_naive=True)
        first = _copy_world(world)
        payment = new_payment(_next_key("P", [p["key"] for p in first["payments"]]), invoice["entity"],
                              invoice["currency"], moment, amount, [key])
        payment["posted_at"] = posting_lag(rng, moment, end)
        first["payments"].append(payment)
        order_postings(first)
        pkey = payment["key"]
        closing = payment_parts(first)[pkey]["closes"][key]
        if chosen is None or closing < 2:
            continue
        rate, fx_params = chosen
        gap = _gap(rng, first, payment, closing)
        payment["gap_minor"] = gap
        docs = sorted([doc_key(pkey), doc_key(key)])
        gap_issue = {"type": "allocation_gap", "affected": docs, "amount_minor": gap, "currency": base,
                     "evidence": _gap_evidence(pkey, key), "params": {"scenario": "I1", "closing_before_minor": closing}}
        second = _copy_world(first)
        _payment(second, pkey)["rate_override"] = rate
        correct = to_base(second, amount, invoice["currency"], moment)
        wrong = to_base(second, amount, invoice["currency"], moment, rate)
        fx_issue = {"type": "fx_error", "affected": [doc_key(pkey)], "amount_minor": abs(wrong - correct),
                    "currency": base,
                    "evidence": [[doc_key(pkey), f"R:{pkey}:cash", _fx_evidence_key(second, invoice["currency"], moment)]],
                    "params": dict(fx_params, scenario="I1")}
        used.update({key, pkey})
        return [(first, gap_issue), (second, fx_issue)]
    raise NoTarget("I1")


def interaction_i2(rng: Rng, world: dict, order: list[str], used: set) -> list[tuple[dict, dict]]:
    """A duplicate whose copy is dated just outside the period and posted in it. The copy keeps the
    original's line items and service period exactly (ruling 2) and is paid with the usual
    probability of an invoice dated outside the period. Step one adds the unposted copy (and its
    payment), step two posts it. An original that already has a resent copy (decoy A) is skipped."""
    for key in order:
        if key in used or not key.startswith("I"):
            continue
        original = _invoice(world, key)
        if (not _posted_inside(world, original) or _decoy(original) or original.get("has_copy")
                or any(p["amount_minor"] < 0 for p in payments_of(world, key))):
            continue
        first = _copy_world(world)
        issued = outside_moment(rng, first, before=rng.chance(1, 2)).astimezone(doc_zone(rng, first, numerator=10))
        copy_invoice = new_invoice(first, _next_key("I", [i["key"] for i in first["invoices"]]), original["entity"],
                                   original["currency"], issued, rng.choice(TERMS_DAYS),
                                   copy.deepcopy(original["line_items"]))
        first["invoices"].append(copy_invoice)
        ckey = copy_invoice["key"]
        paid_at = paid_outside(rng, first, issued)
        if paid_at is not None:
            payment = new_payment(_next_key("P", [p["key"] for p in first["payments"]]), original["entity"],
                                  original["currency"], paid_at.astimezone(doc_zone(rng, first)),
                                  original["total_minor"], [ckey])
            first["payments"].append(payment)
            used.add(payment["key"])
        docs = sorted([doc_key(key), doc_key(ckey)])
        dup_issue = {"type": "duplicate_invoice", "affected": docs, "amount_minor": original["total_minor"],
                     "currency": original["currency"], "evidence": _duplicate_evidence(key, ckey),
                     "params": {"scenario": "I2", "original_doc_id": doc_key(key), "copy_paid": paid_at is not None}}
        second = _copy_world(first)
        _invoice(second, ckey)["posted_at"] = outside_posting(rng, second, issued)
        cut_issue = {"type": "period_cutoff", "affected": [doc_key(ckey)], "amount_minor": original["total_minor"],
                     "currency": original["currency"],
                     "evidence": [[doc_key(ckey), f"R:{ckey}:dr"], [doc_key(ckey), f"R:{ckey}:cr"]],
                     "params": {"scenario": "I2", "direction": "before_start" if issued < first["period_start"]
                                else "after_end", "paid": paid_at is not None}}
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
