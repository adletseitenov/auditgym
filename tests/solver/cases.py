"""Hand-built error, interaction and distractor cases on clean companies.

Each builder takes a Company, edits it, and returns the list of expected findings
(type, affected set, amount, currency, list of acceptable evidence sets as in CONTRACTS 3.1),
or None when the company has no record fit for the edit.
"""
from __future__ import annotations

from decimal import Decimal

from auditgym.money import convert, local_date, parse_time

from .helpers import Company, in_offset, moved_offset, shift, shift_date

# (seed, currencies, timezone): small companies, offsets from both ends of the clock.
COMPANIES = [
    (0, ("USD",), "+05:00"), (1, ("USD", "EUR"), "-11:00"), (2, ("EUR",), "+14:00"),
    (3, ("CNY", "RUB"), "+05:30"), (4, ("USD",), "-03:00"), (5, ("EUR", "USD"), "+00:00"),
]
SMALL = {"size": {"invoices": [20, 30], "background_pairs": [20, 40]}}
# Per-case additions to SMALL. Clean companies have distractors by default, and with them Saturday and
# Sunday repeat Friday's rate; October 2026 ends on Saturday 31.10 and the next period begins on Sunday,
# so a case that needs different rates on the last day and the next day uses September (Wed 30.09 and
# Thu 01.10) instead of switching distractors off.
CASE_CONFIG = {"cutoff_by_own_offset_written_rate": {"period": {"start": "2026-09-01", "months": 1}}}


def expect(kind, affected, amount, currency, evidence_sets):
    return {"type": kind, "affected": set(affected), "amount": amount, "currency": currency,
            "evidence_sets": [set(s) for s in evidence_sets]}


def _first(items, c: Company | None = None):
    """First candidate whose documents no earlier edit of this company used."""
    for item in items:
        docs = {d["doc_id"] for d in (item if isinstance(item, tuple) else (item,))}
        taken = getattr(c, "taken", set()) if c is not None else set()
        if docs & taken:
            continue
        if c is not None:
            c.taken = taken | docs
        return item
    return None


def _spare_day(c: Company, invoice: dict) -> str | None:
    """issued_at moved by a day, staying inside the period."""
    for days in (1, -1, 2, -2):
        moved = shift(invoice["issued_at"], days=days)
        if c.inside(moved):
            return moved
    return None


# ---------------------------------------------------------------- errors
def duplicate(c: Company, foreign: bool):
    invoice = _first([inv for inv in c.public["invoices"] if c.invoice_rows(inv)
                      and (inv["currency"] != c.base) == foreign and _spare_day(c, inv)], c)
    if invoice is None:
        return None
    items = list(reversed(invoice["line_items"]))  # same multiset, other order
    copy_ = c.add_invoice(invoice, issued_at=_spare_day(c, invoice), line_items=items)
    return [expect_duplicate(c, invoice, copy_)]


def expect_duplicate(c: Company, invoice: dict, copy_: dict):
    """CONTRACTS 3.1: four sets [docA, docB, a row of A's group, a row of B's group]."""
    ids = [invoice["doc_id"], copy_["doc_id"]]
    sets = [ids + [a["row_id"], b["row_id"]] for a in c.invoice_rows(invoice) for b in c.invoice_rows(copy_)]
    return expect("duplicate_invoice", ids, invoice["total_minor"], invoice["currency"], sets)


def duplicate_paid_copy(c: Company, foreign: bool):
    """A double payment: the original is paid once and the copy is paid again. Only the duplicate."""
    pair = _first([(inv, pay) for inv, pay in c.paid_once(foreign) if _spare_day(c, inv)], c)
    if pair is None:
        return None
    invoice, payment = pair
    copy_ = c.add_invoice(invoice, issued_at=_spare_day(c, invoice))
    moment = shift(payment["timestamp"], hours=3) if c.inside(shift(payment["timestamp"], hours=3)) \
        else copy_["issued_at"]
    again = c.add_payment(timestamp=moment, currency=payment["currency"], amount_minor=copy_["total_minor"],
                          invoice_refs=[copy_["invoice_id"]])
    c.book_payment(again, closes={copy_["invoice_id"]: c.booked(copy_)})
    return [expect_duplicate(c, invoice, copy_)]


def allocation_gap(c: Company, foreign: bool):
    pair = _first(c.paid_once(foreign), c)
    if pair is None:
        return None
    invoice, payment = pair
    booked = c.booked(invoice)
    gap = booked // 3 + 7
    rows = c.rebook_payment(payment, closes={invoice["invoice_id"]: booked - gap}, unallocated=gap)
    row = c.row(rows, "unallocated")
    ids = [payment["doc_id"], invoice["doc_id"]]
    return [expect("allocation_gap", ids, gap, c.base, [ids + [row["row_id"]]])]


def _fx_delta(correct: int) -> int:
    return max(100, abs(correct) * 6 // 1000)  # SPEC: at least 100 minor units and 0.5 %


def fx_error_invoice(c: Company, sign: int = 1):
    # Unpaid, or paid by one ordinary payment (the payment is rebooked to close what was booked).
    invoice = _first(c.unpaid(foreign=True) + [inv for inv, _ in c.paid_once(foreign=True)], c)
    if invoice is None:
        return None
    rows = c.invoice_rows(invoice)
    delta = _fx_delta(c.booked(invoice))
    for row in rows:
        side = "debit_minor" if row["debit_minor"] else "credit_minor"
        row[side] += sign * delta
    for payment in c.payments_of(invoice):  # the bookkeeper closes what was booked
        c.rebook_payment(payment, closes={invoice["invoice_id"]: c.booked(invoice)})
    return [expect("fx_error", [invoice["doc_id"]], delta, c.base,
                   [[invoice["doc_id"], row["row_id"]] for row in rows])]


def fx_error_payment(c: Company):
    pair = _first(c.paid_once(foreign=True), c)
    if pair is None:
        return None
    invoice, payment = pair
    correct = c.to_base(payment["amount_minor"], payment["currency"], payment["timestamp"])
    delta = _fx_delta(correct)
    rows = c.rebook_payment(payment, closes={invoice["invoice_id"]: c.booked(invoice)}, cash=correct - delta)
    cash = c.row(rows, "cash")
    return [expect("fx_error", [payment["doc_id"]], delta, c.base, [[payment["doc_id"], cash["row_id"]]])]


def _shift_service(invoice: dict, days: int) -> None:
    for item in invoice["line_items"]:
        item["service_start"] = shift_date(item["service_start"], days)
        item["service_end"] = shift_date(item["service_end"], days)


def cutoff_invoice(c: Company, foreign: bool, where: str = "end", paid: bool = False):
    """Dated outside the period, posted inside it (rows keep their in-period time). DECISIONS A5: the
    service period moves with the date; DECISIONS A4: the posting uses the rate of the new date."""
    candidates = [inv for inv, _ in c.paid_once(foreign)] if paid else c.unpaid(foreign)
    invoice = _first([inv for inv in candidates if c.invoice_rows(inv)], c)
    if invoice is None:
        return None
    if where == "end":  # exactly the end of the half-open period, written in UTC
        invoice["issued_at"] = in_offset(c.policy["period_end"], "+00:00")
        _shift_service(invoice, 31)
    else:  # one second before the start, written in another offset
        invoice["issued_at"] = in_offset(shift(c.policy["period_start"], seconds=-1), "+09:00")
        _shift_service(invoice, -31)
    rows = c.rebook_invoice(invoice)
    return [expect("period_cutoff", [invoice["doc_id"]], invoice["total_minor"], invoice["currency"],
                   [[invoice["doc_id"], row["row_id"]] for row in rows])]


def cutoff_payment(c: Company):
    pair = _first(c.paid_once(), c)
    if pair is None:
        return None
    invoice, payment = pair
    payment["timestamp"] = shift(c.policy["period_end"], hours=2)  # posted rows keep their in-period time
    rows = c.rebook_payment(payment, closes={invoice["invoice_id"]: c.booked(invoice)})  # rate of its own date
    return [expect("period_cutoff", [payment["doc_id"]], payment["amount_minor"], payment["currency"],
                   [[payment["doc_id"], row["row_id"]] for row in rows])]


def interaction_i1(c: Company):
    """Partial foreign payment at the very end of the period: allocation_gap + fx_error."""
    end = c.policy["period_end"]
    pair = _first([(inv, pay) for inv, pay in c.paid_once(foreign=True)
                   if parse_time(inv["issued_at"]) < parse_time(shift(end, hours=-1))], c)
    if pair is None:
        return None
    invoice, payment = pair
    payment["timestamp"] = in_offset(shift(end, minutes=-20), "+00:00")  # last local day, other offset
    booked = c.booked(invoice)
    correct = c.to_base(payment["amount_minor"], payment["currency"], payment["timestamp"])
    delta = _fx_delta(correct)
    gap = booked // 4 + 3
    rows = c.rebook_payment(payment, closes={invoice["invoice_id"]: booked - gap}, unallocated=gap,
                            cash=correct + delta)
    ids = [payment["doc_id"], invoice["doc_id"]]
    return [expect("allocation_gap", ids, gap, c.base, [ids + [c.row(rows, "unallocated")["row_id"]]]),
            expect("fx_error", [payment["doc_id"]], delta, c.base,
                   [[payment["doc_id"], c.row(rows, "cash")["row_id"]]])]


def interaction_i2(c: Company, foreign: bool):
    """A copy dated after the period but posted inside it: duplicate_invoice + period_cutoff."""
    invoice = _first([inv for inv in c.public["invoices"] if c.invoice_rows(inv)
                      and (inv["currency"] != c.base) == foreign], c)
    if invoice is None:
        return None
    copy_ = c.add_invoice(invoice, post=False, issued_at=shift(c.policy["period_end"], days=2, hours=3))
    rows = c.book_invoice(copy_, at=invoice["issued_at"])  # DECISIONS B: each copy at the rate of its own date
    return [expect_duplicate(c, invoice, copy_),
            expect("period_cutoff", [copy_["doc_id"]], copy_["total_minor"], copy_["currency"],
                   [[copy_["doc_id"], row["row_id"]] for row in rows])]


def all_four(c: Company):
    """One of each type on different documents of the same company."""
    expected = []
    for build in (lambda: fx_error_invoice(c), lambda: allocation_gap(c, foreign=False),
                  lambda: duplicate(c, foreign=False), lambda: cutoff_invoice(c, foreign=False)):
        found = build()
        if found is None:
            return None
        expected += found
    return expected


def cutoff_out_of_period_rate(c: Company):
    """A cutoff invoice still booked at its old in-period rate: period_cutoff and fx_error (DECISIONS A4)."""
    for invoice in c.unpaid(foreign=True):
        if not c.invoice_rows(invoice) or invoice["doc_id"] in getattr(c, "taken", set()):
            continue
        old = c.booked(invoice)
        moved = shift(c.policy["period_end"], days=1, hours=5)
        correct = c.to_base(invoice["total_minor"], invoice["currency"], moved)
        if abs(correct - old) <= 1:
            continue
        c.taken = getattr(c, "taken", set()) | {invoice["doc_id"]}
        invoice["issued_at"] = moved
        rows = c.invoice_rows(invoice)
        return [expect("period_cutoff", [invoice["doc_id"]], invoice["total_minor"], invoice["currency"],
                       [[invoice["doc_id"], row["row_id"]] for row in rows]),
                expect("fx_error", [invoice["doc_id"]], abs(correct - old), c.base,
                       [[invoice["doc_id"], row["row_id"]] for row in rows])]
    return None


def cutoff_by_own_offset(c: Company, foreign: bool, booked_at_written_date: bool = False):
    """The wall clock of the document says the last day of the period, but in the company offset the
    instant is already the next period: a cutoff, and the rate date is the company-local date."""
    invoice = _first([inv for inv in c.unpaid(foreign) if c.invoice_rows(inv)], c)
    if invoice is None:
        return None
    instant = shift(c.policy["period_end"], hours=1)
    invoice["issued_at"] = in_offset(instant, moved_offset(c.offset, -6))
    assert invoice["issued_at"][:10] < c.policy["period_end"][:10] and not c.inside(invoice["issued_at"])
    _shift_service(invoice, 31)
    rows = c.rebook_invoice(invoice)
    expected = [expect("period_cutoff", [invoice["doc_id"]], invoice["total_minor"], invoice["currency"],
                       [[invoice["doc_id"], row["row_id"]] for row in rows])]
    if booked_at_written_date:
        if not foreign:
            return None
        correct = c.booked(invoice)
        wrong = c.to_base(invoice["total_minor"], invoice["currency"], instant, date=invoice["issued_at"][:10])
        if abs(wrong - correct) <= 1:
            return None
        for row in rows:
            row["debit_minor" if row["debit_minor"] else "credit_minor"] = wrong
        expected.append(expect("fx_error", [invoice["doc_id"]], abs(wrong - correct), c.base,
                               [[invoice["doc_id"], row["row_id"]] for row in rows]))
    return expected


def fx_error_written_date(c: Company):
    """Inside the period, but the document's own offset puts its wall clock on the next day; booking at
    the rate of the written date instead of the company-local date is an fx_error."""
    for invoice in c.unpaid(foreign=True):
        if not c.invoice_rows(invoice) or invoice["doc_id"] in getattr(c, "taken", set()):
            continue
        local = local_date(invoice["issued_at"], c.offset)
        instant = shift(local + "T23:30:00" + c.offset, days=0)
        if not c.inside(instant):
            continue
        written = in_offset(instant, moved_offset(c.offset, 6))
        right = c.to_base(invoice["total_minor"], invoice["currency"], instant)
        wrong = c.to_base(invoice["total_minor"], invoice["currency"], instant, date=written[:10])
        if abs(right - wrong) <= 1:
            continue
        c.taken = getattr(c, "taken", set()) | {invoice["doc_id"]}
        invoice["issued_at"] = written
        rows = c.invoice_rows(invoice)
        for row in rows:
            row["debit_minor" if row["debit_minor"] else "credit_minor"] = wrong
        return [expect("fx_error", [invoice["doc_id"]], abs(right - wrong), c.base,
                       [[invoice["doc_id"], row["row_id"]] for row in rows])]
    return None


def _quiet_date(c: Company, currency: str):
    """A local date inside the period with no document in `currency`, at noon."""
    busy = {local_date(d["issued_at" if "issued_at" in d else "timestamp"], c.offset)
            for d in c.public["invoices"] + c.public["payments"] if d["currency"] == currency}
    for days in range(2, 26):
        moment = shift(c.policy["period_start"], days=days, hours=12)
        if c.inside(moment) and local_date(moment, c.offset) not in busy:
            return moment
    return None


def long_rate(c: Company, wrong: bool):
    """A rate with many significant digits; booking at it is clean, booking at it rounded to two decimals
    is an fx_error."""
    currency = next((inv["currency"] for inv in c.public["invoices"] if inv["currency"] != c.base), None)
    moment = currency and _quiet_date(c, currency)
    if not moment:
        return None
    date = local_date(moment, c.offset)
    c.set_rate(currency, date, c.rate(currency, date) + Decimal("0.001234567891"))
    like = next(inv for inv in c.public["invoices"] if inv["currency"] == currency)
    items = [{"description": "Long rate", "quantity": 3, "unit_price_minor": 329_218_107,
              "amount_minor": 987_654_321, "service_start": date, "service_end": date}]
    invoice = c.add_invoice(like, post=False, issued_at=moment, due_at=shift(moment, days=30), line_items=items)
    exponents = c.policy["currency_exponents"]
    right = c.to_base(invoice["total_minor"], currency, moment)
    short = convert(invoice["total_minor"], c.rate(currency, date).quantize(Decimal("0.01")),
                    exponents[currency], exponents[c.base])
    rows = c.book_invoice(invoice, amount=short if wrong else right)
    payment = c.add_payment(timestamp=moment, currency=currency, amount_minor=invoice["total_minor"],
                            invoice_refs=[invoice["invoice_id"]])
    c.book_payment(payment, closes={invoice["invoice_id"]: c.booked(invoice)})
    if not wrong:
        return []
    return [expect("fx_error", [invoice["doc_id"]], abs(right - short), c.base,
                   [[invoice["doc_id"], row["row_id"]] for row in rows])]


ERROR_CASES = {
    "duplicate_base": lambda c: duplicate(c, False),
    "duplicate_foreign": lambda c: duplicate(c, True),
    "allocation_gap_base": lambda c: allocation_gap(c, False),
    "allocation_gap_foreign": lambda c: allocation_gap(c, True),
    "fx_error_invoice_up": lambda c: fx_error_invoice(c, 1),
    "fx_error_invoice_down": lambda c: fx_error_invoice(c, -1),
    "fx_error_payment": fx_error_payment,
    "cutoff_invoice_end_base": lambda c: cutoff_invoice(c, False, "end"),
    "cutoff_invoice_start_foreign": lambda c: cutoff_invoice(c, True, "start"),
    "cutoff_payment": cutoff_payment,
    "cutoff_invoice_paid_base": lambda c: cutoff_invoice(c, False, "start", paid=True),
    "cutoff_invoice_paid_foreign": lambda c: cutoff_invoice(c, True, "end", paid=True),
    "cutoff_out_of_period_rate": cutoff_out_of_period_rate,
    "cutoff_by_own_offset_base": lambda c: cutoff_by_own_offset(c, False),
    "cutoff_by_own_offset_foreign": lambda c: cutoff_by_own_offset(c, True),
    "cutoff_by_own_offset_written_rate": lambda c: cutoff_by_own_offset(c, True, booked_at_written_date=True),
    "duplicate_paid_copy_base": lambda c: duplicate_paid_copy(c, False),
    "duplicate_paid_copy_foreign": lambda c: duplicate_paid_copy(c, True),
    "fx_error_written_date": fx_error_written_date,
    "fx_error_long_rate": lambda c: long_rate(c, wrong=True),
    "interaction_I1": interaction_i1,
    "interaction_I2_base": lambda c: interaction_i2(c, False),
    "interaction_I2_foreign": lambda c: interaction_i2(c, True),
    "all_four": all_four,
}


# ---------------------------------------------------------------- distractors (must stay clean)
def overpayment(c: Company, foreign: bool):
    pair = _first(c.paid_once(foreign), c)
    if pair is None:
        return None
    invoice, payment = pair
    extra = invoice["total_minor"] // 10 + 1000
    payment["amount_minor"] += extra
    c.rebook_payment(payment, closes={invoice["invoice_id"]: c.booked(invoice)},
                     unallocated=c.to_base(extra, payment["currency"], payment["timestamp"]))
    return []


def advance(c: Company, foreign: bool):
    currency = next((inv["currency"] for inv in c.public["invoices"] if (inv["currency"] != c.base) == foreign), None)
    if currency is None:
        return None
    entity = c.public["invoices"][0]["entity_id"]
    payment = c.add_payment(timestamp=shift(c.policy["period_start"], days=9, hours=11), currency=currency,
                            amount_minor=4_321_000, invoice_refs=None)
    cash = c.to_base(payment["amount_minor"], currency, payment["timestamp"])
    c.book_payment(payment, closes={}, unallocated=cash, entity_id=entity)
    return []


def partial(c: Company, foreign: bool):
    pair = _first(c.paid_once(foreign), c)
    if pair is None:
        return None
    invoice, payment = pair
    payment["amount_minor"] = payment["amount_minor"] // 2
    cash = c.to_base(payment["amount_minor"], payment["currency"], payment["timestamp"])
    c.rebook_payment(payment, closes={invoice["invoice_id"]: cash})
    return []


def split(c: Company, foreign: bool):
    pair = _first([(inv, pay) for inv, pay in c.paid_once(foreign) if c.inside(shift(pay["timestamp"], hours=1))], c)
    if pair is None:
        return None
    invoice, payment = pair
    booked = c.booked(invoice)
    first = payment["amount_minor"] * 2 // 5
    second = c.add_payment(timestamp=shift(payment["timestamp"], hours=1), currency=payment["currency"],
                           amount_minor=payment["amount_minor"] - first, invoice_refs=[invoice["invoice_id"]])
    payment["amount_minor"] = first
    first_close = c.to_base(first, payment["currency"], payment["timestamp"])
    c.rebook_payment(payment, closes={invoice["invoice_id"]: first_close})
    c.book_payment(second, closes={invoice["invoice_id"]: booked - first_close})
    return []


def refund(c: Company, foreign: bool, full: bool = False):
    """A refund on an ordinary fully paid invoice (part of it, or all of it)."""
    pair = _first([(inv, pay) for inv, pay in c.paid_once(foreign) if c.inside(shift(pay["timestamp"], hours=2))], c)
    if pair is None:
        return None
    invoice, payment = pair
    back = -invoice["total_minor"] if full else -(invoice["total_minor"] // 4)
    refund_ = c.add_payment(timestamp=shift(payment["timestamp"], hours=2), currency=payment["currency"],
                            amount_minor=back, invoice_refs=[invoice["invoice_id"]])
    cash = c.to_base(back, refund_["currency"], refund_["timestamp"])
    c.book_payment(refund_, closes={invoice["invoice_id"]: cash})
    return []


def recurring(c: Company, foreign: bool):
    """Same customer, currency, items and total, but the next service period."""
    invoice = _first([inv for inv in c.public["invoices"] if c.invoice_rows(inv)
                      and (inv["currency"] != c.base) == foreign and _spare_day(c, inv)], c)
    if invoice is None:
        return None
    items = [dict(item, service_start=shift(item["service_start"] + "T00:00:00", days=31)[:10],
                  service_end=shift(item["service_end"] + "T00:00:00", days=31)[:10])
             for item in invoice["line_items"]]
    c.add_invoice(invoice, issued_at=_spare_day(c, invoice), line_items=items)
    return []


def _fresh_invoice(c: Company, foreign: bool, issued_at: str, label: str, post: bool = True):
    like = _first([inv for inv in c.public["invoices"] if (inv["currency"] != c.base) == foreign])
    if like is None:
        return None
    items = [{"description": f"Boundary check {label}", "quantity": 1, "unit_price_minor": 777_000,
              "amount_minor": 777_000, "service_start": c.policy["period_start"][:10],
              "service_end": c.policy["period_start"][:10]}]
    return c.add_invoice(like, post=post, issued_at=issued_at, due_at=shift(issued_at, days=30), line_items=items)


def boundary_offsets(c: Company, foreign: bool):
    """Documents inside the period whose own offset puts them on another calendar day."""
    end, start = c.policy["period_end"], c.policy["period_start"]
    last = in_offset(shift(end, minutes=-1), "+14:00" if c.offset != "+14:00" else "-11:00")
    first = in_offset(shift(start, seconds=30), "-11:00" if c.offset != "-11:00" else "+14:00")
    if _fresh_invoice(c, foreign, last, "last") is None:
        return None
    _fresh_invoice(c, foreign, first, "first")
    # A next-period invoice that is not posted is legitimate.
    _fresh_invoice(c, foreign, shift(end, days=1), "next", post=False)
    return []


def neighbour_day(c: Company):
    """Booked at the local-date rate where the UTC date is the day before (and the rates differ)."""
    for days in range(3, 25):
        moment = shift(c.policy["period_start"], days=days, minutes=30)
        utc = in_offset(moment, "-11:00" if c.offset[0] == "+" else "+14:00")
        currency = next((inv["currency"] for inv in c.public["invoices"] if inv["currency"] != c.base), None)
        if currency is None:
            return None
        utc_date = parse_time(utc).date().isoformat()
        if utc_date != local_date(utc, c.offset) and c.rate(currency, utc_date) != c.rate(currency, local_date(utc, c.offset)):
            like = next(inv for inv in c.public["invoices"] if inv["currency"] == currency)
            items = [{"description": "Neighbour day", "quantity": 1, "unit_price_minor": 5_000_000,
                      "amount_minor": 5_000_000, "service_start": "2026-10-01", "service_end": "2026-10-31"}]
            c.add_invoice(like, issued_at=utc, line_items=items)
            return []
    return None


def weekend(c: Company):
    """Saturday and Sunday carry Friday's rate; a weekend invoice booked at that rate is correct."""
    currency = next((inv["currency"] for inv in c.public["invoices"] if inv["currency"] != c.base), None)
    if currency is None:
        return None
    for days in range(2, 26):
        day = shift(c.policy["period_start"], days=days, hours=12)
        date = local_date(day, c.offset)
        if parse_time(day).weekday() != 5:
            continue
        busy = {local_date(d["issued_at" if "issued_at" in d else "timestamp"], c.offset)
                for d in c.public["invoices"] + c.public["payments"] if d["currency"] == currency}
        sunday = local_date(shift(day, days=1), c.offset)
        if date in busy or sunday in busy:
            continue
        friday = c.rate(currency, local_date(shift(day, days=-1), c.offset))
        c.set_rate(currency, date, friday)
        c.set_rate(currency, sunday, friday)
        like = next(inv for inv in c.public["invoices"] if inv["currency"] == currency)
        items = [{"description": "Weekend work", "quantity": 2, "unit_price_minor": 1_234_000,
                  "amount_minor": 2_468_000, "service_start": "2026-10-01", "service_end": "2026-10-31"}]
        c.add_invoice(like, post=False, issued_at=shift(day, days=1), line_items=items)
        invoice = c.public["invoices"][-1]
        # Booked as of Friday: same rate, so the same value.
        c.book_invoice(invoice, amount=c.to_base(invoice["total_minor"], currency, day,
                                                 date=local_date(shift(day, days=-1), c.offset)))
        return []
    return None


def posting_lag(c: Company):
    """Every document posting is made later than the document date (minutes to days), inside the period,
    some written in another offset. Nothing is an error: predicates read document dates only."""
    groups: dict = {}
    for row in c.ledger:
        key = row["payment_id"] or row["invoice_id"]
        if key is not None:
            groups.setdefault(key, []).append(row)
    moved = 0
    for index, key in enumerate(sorted(groups)):
        delta = ({"minutes": 7}, {"hours": 5}, {"days": 2, "hours": 3}, {"seconds": 1})[index % 4]
        if c.lag_postings(groups[key], **delta):
            moved += 1
            if index % 3 == 0:
                for row in groups[key]:
                    row["timestamp"] = in_offset(row["timestamp"], "-09:30")
    return [] if moved >= 5 else None


def written_date_outside(c: Company, foreign: bool):
    """Inside the period in the company offset, though the wall clock says the next period; booked at
    the rate of the company-local date."""
    instant = shift(c.policy["period_end"], minutes=-40)
    invoice = _fresh_invoice(c, foreign, in_offset(instant, moved_offset(c.offset, 6)), "inside", post=False)
    if invoice is None:
        return None
    assert invoice["issued_at"][:10] >= c.policy["period_end"][:10] and c.inside(invoice["issued_at"])
    c.book_invoice(invoice, at=shift(instant, minutes=30))
    return []


def payment_near_boundary(c: Company, foreign: bool):
    """An in-period invoice paid in the last minutes of the period (posted, in another offset), and another
    paid just after it (the payment belongs to the next period and is not posted)."""
    end = c.policy["period_end"]
    early = [(inv, pay) for inv, pay in c.paid_once(foreign) if parse_time(inv["issued_at"]) < parse_time(shift(end, hours=-2))]
    pair = _first(early, c)
    later = _first(early, c)
    if pair is None or later is None:
        return None
    invoice, payment = pair
    payment["timestamp"] = in_offset(shift(end, minutes=-3), moved_offset(c.offset, -7))
    c.rebook_payment(payment, closes={invoice["invoice_id"]: c.booked(invoice)})
    _, outside = later
    outside["timestamp"] = in_offset(shift(end, minutes=4), moved_offset(c.offset, 3))
    c.remove_rows(c.payment_rows(outside))
    return []


def unposted_neighbours(c: Company, foreign: bool):
    """Documents of the previous and the next period without postings: an invoice, a payment of it,
    an advance, a refund and a recurring copy for the next service period."""
    start, end = c.policy["period_start"], c.policy["period_end"]
    before = _fresh_invoice(c, foreign, shift(start, days=-3), "before", post=False)
    if before is None:
        return None
    c.add_payment(timestamp=shift(start, days=-1), currency=before["currency"], amount_minor=before["total_minor"],
                  invoice_refs=[before["invoice_id"]])
    c.add_payment(timestamp=shift(end, days=2), currency=before["currency"], amount_minor=-1000,
                  invoice_refs=[before["invoice_id"]])
    c.add_payment(timestamp=shift(end, hours=1), currency=before["currency"], amount_minor=123_400, invoice_refs=None)
    invoice = _first([inv for inv in c.public["invoices"] if c.invoice_rows(inv)
                      and (inv["currency"] != c.base) == foreign], c)
    if invoice is None:
        return None
    items = [dict(item, service_start=shift_date(item["service_start"], 31), service_end=shift_date(item["service_end"], 31))
             for item in invoice["line_items"]]
    c.add_invoice(invoice, post=False, issued_at=shift(end, days=1), line_items=items)
    # An exact copy dated in the next period and not posted is not a duplicate either (both must be posted).
    c.add_invoice(invoice, post=False, issued_at=shift(end, days=3))
    return []


DISTRACTOR_CASES = {
    "overpayment_base": lambda c: overpayment(c, False),
    "overpayment_foreign": lambda c: overpayment(c, True),
    "advance_base": lambda c: advance(c, False),
    "advance_foreign": lambda c: advance(c, True),
    "partial_base": lambda c: partial(c, False),
    "partial_foreign": lambda c: partial(c, True),
    "split_base": lambda c: split(c, False),
    "split_foreign": lambda c: split(c, True),
    "refund_base": lambda c: refund(c, False),
    "refund_foreign": lambda c: refund(c, True),
    "refund_full_base": lambda c: refund(c, False, full=True),
    "refund_full_foreign": lambda c: refund(c, True, full=True),
    "posting_lag": posting_lag,
    "written_date_outside_base": lambda c: written_date_outside(c, False),
    "written_date_outside_foreign": lambda c: written_date_outside(c, True),
    "payment_near_boundary_base": lambda c: payment_near_boundary(c, False),
    "payment_near_boundary_foreign": lambda c: payment_near_boundary(c, True),
    "unposted_neighbours_base": lambda c: unposted_neighbours(c, False),
    "unposted_neighbours_foreign": lambda c: unposted_neighbours(c, True),
    "long_rate_digits": lambda c: long_rate(c, wrong=False),
    "recurring_base": lambda c: recurring(c, False),
    "recurring_foreign": lambda c: recurring(c, True),
    "boundary_offsets_base": lambda c: boundary_offsets(c, False),
    "boundary_offsets_foreign": lambda c: boundary_offsets(c, True),
    "neighbour_day_rate": neighbour_day,
    "weekend_rate": weekend,
}


def build(name: str, builder, seed: int, currencies, timezone: str):
    """(public, expected) or None when this company has no record fit for the edit."""
    company = Company(seed, currencies, timezone, **{**SMALL, **CASE_CONFIG.get(name, {})})
    expected = builder(company)
    if expected is None:
        return None
    return company.done(), expected
