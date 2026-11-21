"""Which clean patterns an episode shows, read from its open records only (docs/DECISIONS.md T).

`distractor_patterns(public)` returns the subset of auditgym.config.DISTRACTOR_KINDS that is
visible in the public records. It is the independent observer used by the tests and by
tools/checks_gen.py to prove that a switch of `config["distractors"]` produces its own pattern and
none of the others. The partial, split, refund, overpayment and advance readings follow the
payment records; boundary_dates counts only documents outside the period that are not resent
copies (those belong to resent_copies).
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

from auditgym.money import parse_offset, parse_time
from auditgym.verify import invoice_signature


def _decoys(public: dict) -> tuple[set, set]:
    """(resent copies, invoices of the period for another period's service), as doc_ids."""
    policy = public["policy"]
    start, end = parse_time(policy["period_start"]), parse_time(policy["period_end"])
    zone = parse_offset(policy["timezone"])
    first = start.astimezone(zone).date().isoformat()
    last = (end - dt.timedelta(seconds=1)).astimezone(zone).date().isoformat()
    posted = {row["invoice_id"] for row in public["ledger"] if row["payment_id"] is None and row["invoice_id"]}
    groups = defaultdict(list)
    for invoice in public["invoices"]:
        groups[invoice_signature(invoice)].append(invoice)
    copies = {unposted[0]["doc_id"] for group in groups.values() if len(group) == 2
              for unposted in [[inv for inv in group if inv["invoice_id"] not in posted]] if len(unposted) == 1}
    shifted = {inv["doc_id"] for inv in public["invoices"]
               if start <= parse_time(inv["issued_at"]) < end
               and all(i["service_end"] < first or i["service_start"] > last for i in inv["line_items"])}
    return copies, shifted


def _recurring(public: dict) -> bool:
    """Two invoices of one customer with the same items and total for different service periods,
    both inside the reporting period (a decoy of shifted_service has one outside)."""
    policy = public["policy"]
    zone = parse_offset(policy["timezone"])
    first = parse_time(policy["period_start"]).astimezone(zone).date().isoformat()
    last = (parse_time(policy["period_end"]) - dt.timedelta(seconds=1)).astimezone(zone).date().isoformat()
    groups = defaultdict(set)
    for invoice in public["invoices"]:
        if any(i["service_start"] < first or i["service_end"] > last for i in invoice["line_items"]):
            continue
        items = tuple(sorted((i["description"], i["quantity"], i["unit_price_minor"]) for i in invoice["line_items"]))
        services = tuple(sorted({(i["service_start"], i["service_end"]) for i in invoice["line_items"]}))
        groups[(invoice["entity_id"], invoice["currency"], invoice["total_minor"], items)].add(services)
    return any(len(services) > 1 for services in groups.values())


def _weekend_rates(public: dict) -> bool:
    """A Saturday or Sunday rate equal to the day before, for some currency."""
    previous: dict = {}
    for row in sorted(public["fx"], key=lambda r: (r["source_currency"], r["date"])):
        date = dt.date.fromisoformat(row["date"])
        before = previous.get(row["source_currency"])
        if date.weekday() >= 5 and before is not None and before == row["rate"]:
            return True
        previous[row["source_currency"]] = row["rate"]
    return False


def distractor_patterns(public: dict) -> set[str]:
    policy = public["policy"]
    start, end = parse_time(policy["period_start"]), parse_time(policy["period_end"])
    zone = parse_offset(policy["timezone"])
    copies, shifted = _decoys(public)
    found: set[str] = set()
    if copies:
        found.add("resent_copies")
    if shifted:
        found.add("shifted_service")
    if _recurring(public):
        found.add("repeated_amounts")
    if _weekend_rates(public):
        found.add("weekend_fx")
    invoices = {inv["invoice_id"]: inv for inv in public["invoices"]}
    moments = [(inv["doc_id"], parse_time(inv["issued_at"]), inv["issued_at"]) for inv in public["invoices"]]
    moments += [(pay["doc_id"], parse_time(pay["timestamp"]), pay["timestamp"]) for pay in public["payments"]]
    for doc_id, moment, text in moments:
        if moment.utcoffset() != zone.utcoffset(None):
            found.add("other_offsets")
        if doc_id not in copies and not start <= moment < end:
            found.add("boundary_dates")
    paid = defaultdict(list)
    for payment in public["payments"]:
        refs = payment["invoice_refs"]
        if refs is None:
            found.add("advances")
        elif payment["amount_minor"] < 0:
            found.add("refunds")
        elif len(refs) == 1:
            paid[refs[0]].append(payment)
            if payment["amount_minor"] > invoices[refs[0]]["total_minor"]:
                found.add("overpayments")
    for ref, payments in paid.items():
        if len(payments) > 1 or payments[0]["amount_minor"] < invoices[ref]["total_minor"]:
            found.add("split_payments")
    return found
