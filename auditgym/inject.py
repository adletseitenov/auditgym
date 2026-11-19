"""Error operators. Each one changes the story and its books and records what it did.

Operators work on internal keys ("I3", "R17", "D:I3" for the document of invoice I3); surface.py
assigns public ids afterwards, so nothing about an injected record's id or position tells it apart.
Each operator gets its own random stream (`inject:<type>`), so enabling one type does not move
the targets of another. Every operator returns a new world and new rows; inputs are not mutated.

An injection is described by a dict in internal keys:
    {"type", "affected": [doc keys], "amount_minor", "currency", "evidence": [[keys]],
     "params": {...}, "before": [records], "after": [records]}
generate.py turns it into a truth issue and a provenance step in public ids.
"""
from __future__ import annotations

import copy
import datetime as dt

from .books import invoice_rows, to_base
from .rng import Rng
from .world import TERMS_DAYS

IMPLEMENTED = ("duplicate_invoice", "period_cutoff")
# Moved document dates stay this many days from the period edge, inside the fx margin of the story.
CUTOFF_MAX_DAYS = 4
CUTOFF_NEAR_SECONDS = 6 * 3600


class NoTarget(Exception):
    """The story has no suitable record for an operator; generate() retries with another attempt."""


def doc_key(key: str) -> str:
    return "D:" + key


def _next_key(prefix: str, keys) -> str:
    used = {int(key[len(prefix):]) for key in keys if key.startswith(prefix) and key[len(prefix):].isdigit()}
    return f"{prefix}{max(used, default=-1) + 1}"


def _postings(rows: list[dict], invoice_key: str) -> list[dict]:
    """The invoice's own event group: rows with its invoice key and no payment."""
    return [row for row in rows if row["invoice"] == invoice_key and row["payment"] is None]


def target_order(rng: Rng, world: dict) -> list[str]:
    """A per-operator random order over the story's original invoices."""
    order = [invoice["key"] for invoice in world["invoices"]]
    rng.shuffle(order)
    return order


def duplicate_invoice(rng: Rng, world: dict, rows: list[dict], order: list[str],
                      used: set[str]) -> tuple[dict, list[dict], dict]:
    """A second invoice with new ids and an identical signature, issued at another moment in the
    period and posted at its own policy amount. Exactly one copy per original."""
    invoices = {invoice["key"]: invoice for invoice in world["invoices"]}
    target = next((key for key in order if key not in used), None)
    if target is None:
        raise NoTarget("duplicate_invoice")
    original = invoices[target]
    start, end = world["period_start"], world["period_end"]
    while True:  # one draw in practice: the period has millions of seconds
        issued = start + dt.timedelta(seconds=rng.below(int((end - start).total_seconds())))
        if issued != original["issued_at"]:
            break
    copy_invoice = {"key": _next_key("I", invoices), "entity": original["entity"],
                    "currency": original["currency"], "issued_at": issued,
                    "due_at": issued + dt.timedelta(days=rng.choice(TERMS_DAYS)),
                    "total_minor": original["total_minor"], "line_items": copy.deepcopy(original["line_items"])}
    new_rows = invoice_rows(world, copy_invoice)
    row_keys = {row["key"] for row in rows}
    for row in new_rows:
        row["key"] = _next_key("R", row_keys)
        row_keys.add(row["key"])
    used.update({target, copy_invoice["key"]})

    world = dict(world, invoices=[*world["invoices"], copy_invoice])
    docs = sorted([doc_key(target), doc_key(copy_invoice["key"])])
    injection = {"type": "duplicate_invoice", "affected": docs, "amount_minor": copy_invoice["total_minor"],
                 "currency": copy_invoice["currency"], "evidence": [docs],
                 "params": {"original_doc_id": doc_key(target)},
                 "before": [], "after": [copy.deepcopy(copy_invoice), *copy.deepcopy(new_rows)]}
    return world, [*rows, *new_rows], injection


def _cutoff_date(rng: Rng, start: dt.datetime, end: dt.datetime) -> tuple[dt.datetime, str, int]:
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


def period_cutoff(rng: Rng, world: dict, rows: list[dict], order: list[str],
                  used: set[str]) -> tuple[dict, list[dict], dict]:
    """Move an unpaid in-period invoice's document date just outside the period while its two
    postings keep their in-period timestamps.

    The postings are re-booked at the policy rate of the document's new date (the same amount
    for a base-currency invoice), so the record is not also an fx_error. Only unpaid invoices are
    used: a payment would close the receivable at the old booked amount and tie the error to
    another document.
    """
    paid = {ref for payment in world["payments"] for ref in payment["invoice_refs"] or []}
    target = next((key for key in order if key not in used and key not in paid), None)
    if target is None:
        raise NoTarget("period_cutoff")
    invoices = {invoice["key"]: invoice for invoice in world["invoices"]}
    original = invoices[target]
    moved_at, direction, distance = _cutoff_date(rng, world["period_start"], world["period_end"])
    moved = dict(original, issued_at=moved_at, due_at=original["due_at"] + (moved_at - original["issued_at"]),
                 line_items=copy.deepcopy(original["line_items"]))
    amount = to_base(world, moved["total_minor"], moved["currency"], moved_at)

    old_rows = _postings(rows, target)
    debit = [row for row in old_rows if row["debit_minor"] > 0]
    credit = [row for row in old_rows if row["credit_minor"] > 0]
    if len(old_rows) != 2 or len(debit) != 1 or len(credit) != 1:
        raise NoTarget("period_cutoff")  # not a plain posted invoice; never happens in a clean story
    replaced = {debit[0]["key"]: dict(debit[0], debit_minor=amount),
                credit[0]["key"]: dict(credit[0], credit_minor=amount)}
    new_rows = [replaced.get(row["key"], row) for row in rows]
    used.add(target)

    world = dict(world, invoices=[moved if invoice["key"] == target else invoice for invoice in world["invoices"]])
    doc = doc_key(target)
    injection = {"type": "period_cutoff", "affected": [doc], "amount_minor": moved["total_minor"],
                 "currency": moved["currency"],
                 "evidence": [[doc, debit[0]["key"]], [doc, credit[0]["key"]]],
                 "params": {"direction": direction, "seconds_outside": distance,
                            "original_issued_at": original["issued_at"].isoformat()},
                 "before": copy.deepcopy([original, debit[0], credit[0]]),
                 "after": copy.deepcopy([moved, replaced[debit[0]["key"]], replaced[credit[0]["key"]]])}
    return world, new_rows, injection


OPERATORS = {"duplicate_invoice": duplicate_invoice, "period_cutoff": period_cutoff}


def inject_errors(root: Rng, world: dict, rows: list[dict], errors: dict[str, int]) -> tuple[dict, list[dict], list[dict]]:
    """Apply every configured operator in a fixed order. Raises NoTarget when the story is unsuitable.

    Each type has its own stream `inject:<type>`: one draw of a target order over the story's
    original invoices, then one child stream per case. A type's targets therefore change only
    when another type already took the same invoice.
    """
    used: set[str] = set()
    injections = []
    story = world  # target orders are drawn over the original invoices only
    for name in IMPLEMENTED:
        count = errors.get(name, 0)
        if not count:
            continue
        stream = root.child(f"inject:{name}")
        order = target_order(stream.child("targets"), story)
        for case in range(count):
            world, rows, injection = OPERATORS[name](stream.child(f"case:{case}"), world, rows, order, used)
            injections.append(injection)
    return world, rows, injections
