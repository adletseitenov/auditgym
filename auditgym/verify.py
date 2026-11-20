"""verify(episode) -> list of problems. An empty list means the episode is well formed.

Checks the open records, the pre-injection snapshot (pre_ledger, pre_invoices, pre_payments) and
the recorded steps with nothing but the published policy, the way an outside checker would. It
does not import the generator. Times are compared as moments, never as strings.

What is checked:
- shape and balance of both books (whole book, every event group, background pairs);
- documents: ids, totals, quantity x unit price, due after issue, currencies, references;
- the fx table: every date of the period plus the margin, at least 6 significant digits;
- clean postings: a document is posted iff it is dated in the period, after its own date, by the
  document's entity, at the policy amount; payments close receivables by the published payment
  rule (auditgym/books.py); fx_difference only on foreign payments. A document a posting issue is
  about (period_cutoff, fx_error, allocation_gap) is checked the same way except for the one
  deviation its issue declares: a cutoff document's period and date, an fx_error's amount (equal
  on both sides of an invoice; the cash row of a payment, whose receivable closings still follow
  the rule), an allocation gap's split between receivable and unallocated (their sum still
  follows the rule). The issue checks verify the deviation itself;
- before injection no error predicate fires; after injection the predicates fire exactly on
  the truth (type, affected ids, amount, currency), and every evidence set is sufficient;
- the pre-injection snapshot plus the recorded steps gives the open records;
- the generator's own rules (docs/DECISIONS.md B).
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from decimal import Decimal, InvalidOperation

from .limits import MAX_EVIDENCE_IDS as MAX_EVIDENCE
from .limits import MAX_LEDGER_ROWS
from .money import closed_share, convert, fx_key, local_date, parse_offset, parse_time, significant_digits

LEDGER_FIELDS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                 "entity_id", "invoice_id", "payment_id"]
FX_MARGIN_DAYS = 5
FX_MIN_MINOR = 100
FX_MIN_PER_MILLE = 5
MIN_RATE_DIGITS = 6
CUTOFF_MAX_DAYS = 4
POSTING_TYPES = ("period_cutoff", "fx_error", "allocation_gap")  # issues about how a document is posted


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def group_key(row: dict) -> tuple:
    """Event group of a posting (docs/CONTRACTS.md, section 2.3)."""
    if row["payment_id"] is not None:
        return ("payment", row["payment_id"])
    if row["invoice_id"] is not None:
        return ("invoice", row["invoice_id"])
    return ("background", row["timestamp"], row["entity_id"])


def invoice_signature(invoice: dict) -> tuple:
    """Customer, currency, total and line items as a multiset (issued_at is not part of it)."""
    items = sorted(tuple(sorted(item.items())) for item in invoice["line_items"])
    return (invoice["entity_id"], invoice["currency"], invoice["total_minor"], tuple(items))


class Books:
    """Indexes over open records (public, or the pre-injection snapshot shaped like public)."""

    def __init__(self, public: dict) -> None:
        self.policy = policy = public["policy"]
        self.base = policy["base_currency"]
        self.offset = policy["timezone"]
        self.start, self.end = parse_time(policy["period_start"]), parse_time(policy["period_end"])
        self.accounts = policy["accounts"]
        self.exponents = policy["currency_exponents"]
        self.tolerance = policy["amount_tolerance_minor"]
        self.ledger = public["ledger"]
        self.invoice_list, self.payment_list = public["invoices"], public["payments"]
        self.invoices = {inv["invoice_id"]: inv for inv in public["invoices"]}
        self.payments = {pay["payment_id"]: pay for pay in public["payments"]}
        self.docs = {d["doc_id"]: d for d in public["invoices"] + public["payments"]}
        self.rates = {fx_key(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"])
                      for r in public["fx"]}
        self.own = defaultdict(list)       # invoice_id -> its own posting pair (no payment id)
        self.by_payment = defaultdict(list)
        self.receivable = defaultdict(int)  # invoice_id -> receivable balance over the whole ledger
        self.paid_by = defaultdict(list)    # invoice_id -> payments referring to it (any)
        for row in public["ledger"]:
            if row["payment_id"] is not None:
                self.by_payment[row["payment_id"]].append(row)
            elif row["invoice_id"] is not None:
                self.own[row["invoice_id"]].append(row)
            if row["account"] == self.accounts["receivable"] and row["invoice_id"] is not None:
                self.receivable[row["invoice_id"]] += row["debit_minor"] - row["credit_minor"]
        for payment in public["payments"]:
            for ref in payment["invoice_refs"] or []:
                self.paid_by[ref].append(payment)

    def inside(self, timestamp: str) -> bool:
        return self.start <= parse_time(timestamp) < self.end

    def policy_amount(self, amount_minor: int, currency: str, timestamp: str) -> int | None:
        if currency == self.base:
            return amount_minor
        rate = self.rates.get(fx_key(local_date(timestamp, self.offset), currency, self.base))
        if rate is None or currency not in self.exponents:
            return None
        return convert(amount_minor, rate, self.exponents[currency], self.exponents[self.base])

    def booked(self, invoice_id: str) -> int:
        return sum(r["debit_minor"] for r in self.own.get(invoice_id, []) if r["account"] == self.accounts["receivable"])


# ============================================================================ predicates
def predicate_issues(public: dict) -> tuple[list[dict], list[str]]:
    """The four published definitions over open records: (issues, undefined cases).

    issue = {"type", "affected_ids" (sorted), "amount", "currency"}."""
    books = Books(public)
    issues, undefined = [], []
    acc = books.accounts

    by_signature = defaultdict(list)
    for invoice in books.invoice_list:
        if books.own.get(invoice["invoice_id"]):
            by_signature[invoice_signature(invoice)].append(invoice)
    for group in by_signature.values():
        if len(group) > 2:
            undefined.append("more than two posted invoices share a signature")
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                issues.append({"type": "duplicate_invoice", "affected_ids": sorted([first["doc_id"], second["doc_id"]]),
                               "amount": first["total_minor"], "currency": first["currency"]})

    for payment in books.payment_list:
        rows = books.by_payment.get(payment["payment_id"], [])
        if not rows or payment["invoice_refs"] is None:
            continue
        unallocated = [r for r in rows if r["account"] == acc["unallocated"]]
        open_refs = [ref for ref in sorted(set(payment["invoice_refs"])) if books.receivable.get(ref, 0) > 0]
        if not unallocated or not open_refs:
            continue
        if len(unallocated) != 1 or unallocated[0]["credit_minor"] <= 0:
            undefined.append(f"payment {payment['doc_id']}: several or debit unallocated rows")
            continue
        for ref in open_refs:
            if ref in books.invoices:
                issues.append({"type": "allocation_gap",
                               "affected_ids": sorted([payment["doc_id"], books.invoices[ref]["doc_id"]]),
                               "amount": unallocated[0]["credit_minor"], "currency": books.base})

    tolerance = books.tolerance["fx_error"]
    for invoice in books.invoice_list:
        rows = books.own.get(invoice["invoice_id"], [])
        if not rows or invoice["currency"] == books.base:
            continue
        correct = books.policy_amount(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        if correct is None:
            undefined.append(f"invoice {invoice['doc_id']}: no rate for its date")
            continue
        # Conversion rows only: the receivable debit and the revenue credit (an extra receivable credit
        # in the invoice's group is not a conversion of the invoice and is not compared).
        values = [r["debit_minor"] for r in rows if r["account"] == acc["receivable"] and r["debit_minor"] > 0]
        values += [r["credit_minor"] for r in rows if r["account"] == acc["revenue"] and r["credit_minor"] > 0]
        deviation = max(abs(value - correct) for value in values) if values else 0
        if deviation > tolerance:
            issues.append({"type": "fx_error", "affected_ids": [invoice["doc_id"]], "amount": deviation,
                           "currency": books.base})
    for payment in books.payment_list:
        rows = books.by_payment.get(payment["payment_id"], [])
        if not rows or payment["currency"] == books.base:
            continue
        correct = books.policy_amount(payment["amount_minor"], payment["currency"], payment["timestamp"])
        if correct is None:
            undefined.append(f"payment {payment['doc_id']}: no rate for its date")
            continue
        cash_rows = [r for r in rows if r["account"] == acc["cash"]]
        if not cash_rows:  # nothing converted the payment: no fx_error (the solver reads it the same way)
            undefined.append(f"payment {payment['doc_id']}: posted without a cash row")
            continue
        cash = sum(r["debit_minor"] - r["credit_minor"] for r in cash_rows)
        if abs(cash - correct) > tolerance:
            issues.append({"type": "fx_error", "affected_ids": [payment["doc_id"]], "amount": abs(cash - correct),
                           "currency": books.base})

    for invoice in books.invoice_list:
        if books.own.get(invoice["invoice_id"]) and not books.inside(invoice["issued_at"]):
            issues.append({"type": "period_cutoff", "affected_ids": [invoice["doc_id"]],
                           "amount": abs(invoice["total_minor"]), "currency": invoice["currency"]})
    for payment in books.payment_list:
        if books.by_payment.get(payment["payment_id"]) and not books.inside(payment["timestamp"]):
            issues.append({"type": "period_cutoff", "affected_ids": [payment["doc_id"]],
                           "amount": abs(payment["amount_minor"]), "currency": payment["currency"]})
    return issues, undefined


# ============================================================================ clean books
def _check_ledger(books: Books, name: str) -> list[str]:
    problems = []
    accounts = set(books.accounts.values())
    seen = set()
    for row in books.ledger:
        rid = row.get("row_id")
        if list(row) != LEDGER_FIELDS:
            problems.append(f"{name}: row {rid} has wrong fields")
            continue
        if rid in seen:
            problems.append(f"{name}: duplicate row_id {rid}")
        seen.add(rid)
        debit, credit = row["debit_minor"], row["credit_minor"]
        if not (_is_int(debit) and _is_int(credit)) or debit < 0 or credit < 0 or (debit > 0) == (credit > 0):
            problems.append(f"{name}: row {rid} must have exactly one positive integer side")
        if row["currency"] != books.base:
            problems.append(f"{name}: row {rid} is not in the base currency")
        if row["account"] not in accounts:
            problems.append(f"{name}: row {rid} uses an unknown account")
        if row["invoice_id"] is not None and row["invoice_id"] not in books.invoices:
            problems.append(f"{name}: row {rid} links to a missing invoice")
        if row["payment_id"] is not None and row["payment_id"] not in books.payments:
            problems.append(f"{name}: row {rid} links to a missing payment")
        if not books.inside(row["timestamp"]):
            problems.append(f"{name}: row {rid} is outside the period")
        if row["account"] == books.accounts["receivable"] and row["invoice_id"] is None:
            problems.append(f"{name}: receivable row {rid} has no invoice")
        if row["entity_id"] is None:
            problems.append(f"{name}: row {rid} has no entity")
    if problems:
        return problems
    if sum(r["debit_minor"] for r in books.ledger) != sum(r["credit_minor"] for r in books.ledger):
        problems.append(f"{name}: debits and credits differ")
    groups = defaultdict(list)
    for row in books.ledger:
        groups[group_key(row)].append(row)
    for key, rows in sorted(groups.items(), key=repr):
        balance = sum(r["debit_minor"] - r["credit_minor"] for r in rows)
        if balance:
            problems.append(f"{name}: group {key} is unbalanced by {balance}")
        if key[0] == "background":
            accounts_sides = sorted((r["account"], r["debit_minor"] > 0) for r in rows)
            if accounts_sides != [(books.accounts["cash"], False), (books.accounts["expense"], True)]:
                problems.append(f"{name}: background group {key} is not an expense/cash pair")
        elif len({parse_time(r["timestamp"]) for r in rows}) != 1:
            problems.append(f"{name}: group {key} is posted at several moments")
    if len(books.ledger) > MAX_LEDGER_ROWS:
        problems.append(f"{name}: more than {MAX_LEDGER_ROWS} ledger rows")
    return problems


def _check_documents(books: Books, name: str) -> list[str]:
    problems = []
    docs = books.invoice_list + books.payment_list
    if len(books.docs) != len(docs):
        problems.append(f"{name}: doc_id is not unique")
    if len(books.invoices) != len(books.invoice_list) or len(books.payments) != len(books.payment_list):
        problems.append(f"{name}: invoice_id or payment_id is not unique")
    for invoice in books.invoice_list:
        doc = invoice["doc_id"]
        if invoice["currency"] not in books.exponents:
            problems.append(f"{name}: invoice {doc} has an unknown currency")
            continue
        items = invoice["line_items"]
        if not items or any(not all(_is_int(item[f]) for f in ("quantity", "unit_price_minor", "amount_minor"))
                            or item["quantity"] <= 0 or item["quantity"] * item["unit_price_minor"] != item["amount_minor"]
                            for item in items):
            problems.append(f"{name}: invoice {doc}: quantity x unit price differs from the amount")
        if not _is_int(invoice["total_minor"]) or invoice["total_minor"] != sum(i["amount_minor"] for i in items):
            problems.append(f"{name}: invoice {doc}: total differs from its line items")
        if any(item["service_start"] > item["service_end"] for item in items):
            problems.append(f"{name}: invoice {doc}: service period ends before it starts")
        if parse_time(invoice["due_at"]) <= parse_time(invoice["issued_at"]):
            problems.append(f"{name}: invoice {doc}: due_at is not after issued_at")
        if books.policy_amount(invoice["total_minor"], invoice["currency"], invoice["issued_at"]) is None:
            problems.append(f"{name}: invoice {doc}: no rate for its date")
    for payment in books.payment_list:
        doc, refs = payment["doc_id"], payment["invoice_refs"]
        if payment["currency"] not in books.exponents:
            problems.append(f"{name}: payment {doc} has an unknown currency")
            continue
        if not _is_int(payment["amount_minor"]) or payment["amount_minor"] == 0:
            problems.append(f"{name}: payment {doc}: amount must be a non-zero integer")
        if refs is not None:
            invoices = [books.invoices.get(ref) for ref in refs]
            if not refs or None in invoices or len(set(refs)) != len(refs):
                problems.append(f"{name}: payment {doc}: references must be existing distinct invoices")
                continue
            if any(inv["currency"] != payment["currency"] for inv in invoices):
                problems.append(f"{name}: payment {doc}: currency differs from its invoice")
            if len({inv["entity_id"] for inv in invoices}) != 1:
                problems.append(f"{name}: payment {doc}: invoices of several customers")
            if len(refs) > 1 and payment["amount_minor"] != sum(inv["total_minor"] for inv in invoices):
                problems.append(f"{name}: payment {doc}: several references but not their full total")
        if refs is not None and any(parse_time(payment["timestamp"]) < parse_time(books.invoices[ref]["issued_at"])
                                    for ref in refs if ref in books.invoices):
            problems.append(f"{name}: payment {doc} is dated before its invoice")
        if payment["amount_minor"] < 0 and (refs is None or len(refs) != 1):
            problems.append(f"{name}: refund {doc} must refer to exactly one invoice")
        if books.policy_amount(payment["amount_minor"], payment["currency"], payment["timestamp"]) is None:
            problems.append(f"{name}: payment {doc}: no rate for its date")
    return problems


def _check_fx(public: dict, books: Books) -> list[str]:
    problems = []
    first = books.start.astimezone(parse_offset(books.offset)).date() - dt.timedelta(days=FX_MARGIN_DAYS)
    last = (books.end - dt.timedelta(seconds=1)).astimezone(parse_offset(books.offset)).date() \
        + dt.timedelta(days=FX_MARGIN_DAYS)
    expected = {(first + dt.timedelta(days=d)).isoformat() for d in range((last - first).days + 1)}
    dates = defaultdict(list)
    for row in public["fx"]:
        if row["target_currency"] != books.base or row["source_currency"] not in books.exponents \
                or row["source_currency"] == books.base:
            problems.append(f"fx: rate {row['date']}|{row['source_currency']} is not foreign -> base")
            continue
        try:
            positive = Decimal(row["rate"]) > 0
        except InvalidOperation:
            positive = False
        if not positive or significant_digits(row["rate"]) < MIN_RATE_DIGITS:
            problems.append(f"fx: rate {row['date']}|{row['source_currency']} needs {MIN_RATE_DIGITS} significant digits")
        dates[row["source_currency"]].append(row["date"])
    for currency in sorted(set(books.exponents) - {books.base}):
        got = dates.get(currency, [])
        if len(got) != len(set(got)) or set(got) != expected:
            problems.append(f"fx: rates of {currency} do not cover the period and its margin exactly once per day")
    return problems


def expected_closings(books: Books) -> dict[str, dict]:
    """Per posted payment: the policy closings {"closes": {invoice_id: base}, "unallocated": base}
    and a rule problem (or None), by policy.payment_rule (money.closed_share)."""
    paid: dict[str, int] = defaultdict(int)
    closed: dict[str, int] = defaultdict(int)
    count: dict[str, int] = defaultdict(int)
    out = {}
    posted = [p for p in books.payment_list if books.by_payment.get(p["payment_id"])]
    posted.sort(key=lambda p: (parse_time(p["timestamp"]), p["doc_id"]))
    for payment in posted:
        refs, amount = payment["invoice_refs"], payment["amount_minor"]
        value = books.policy_amount(amount, payment["currency"], payment["timestamp"])
        closes, unallocated, rule = {}, 0, None
        if value is None:
            out[payment["payment_id"]] = {"closes": {}, "unallocated": 0, "problem": "no rate"}
            continue
        if refs is None:
            unallocated = value
        elif len(refs) > 1:
            for ref in refs:
                closes[ref] = books.booked(ref) - closed[ref]
                paid[ref] = books.invoices[ref]["total_minor"]
        else:
            ref = refs[0]
            total, booked = books.invoices[ref]["total_minor"], books.booked(ref)
            after = paid[ref] + amount
            if amount > 0 and after > total:
                closes[ref] = booked - closed[ref]
                unallocated = books.policy_amount(after - total, payment["currency"], payment["timestamp"])
                if count[ref]:
                    rule = "overpayment after another payment"
                after = total
            elif after < 0:
                rule = "a refund larger than what was paid"
                after = 0
            else:
                closes[ref] = closed_share(booked, total, after) - closed[ref]
            paid[ref] = after
        for ref, part in closes.items():
            closed[ref] += part
            count[ref] += 1
        out[payment["payment_id"]] = {"closes": closes, "unallocated": unallocated, "problem": rule}
    return out


def _check_postings(books: Books, name: str, deviations) -> list[str]:
    """Every document dated in the period is posted once, after its own date, by its entity, at the
    policy amount, and nothing dated outside it is posted. `deviations` maps a doc_id to the posting
    issue types declared on it (an empty set or dict: none); only those deviations are allowed."""
    problems = []
    acc = books.accounts
    allowed = deviations if isinstance(deviations, dict) else {}
    for invoice in books.invoice_list:
        doc = invoice["doc_id"]
        declared = allowed.get(doc, set())
        rows = books.own.get(invoice["invoice_id"], [])
        inside = books.inside(invoice["issued_at"])
        if not rows:
            if inside:
                problems.append(f"{name}: invoice {doc} is in the period but not posted")
            continue
        if not inside and "period_cutoff" not in declared:
            problems.append(f"{name}: invoice {doc} is outside the period but posted")
        expected = books.policy_amount(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        shape = sorted((r["account"], r["debit_minor"], r["credit_minor"]) for r in rows)
        if "fx_error" in declared:
            # The rate is wrong, the posting is not: one pair, the same amount on both sides.
            amount = rows[0]["debit_minor"] or rows[0]["credit_minor"]
            if amount <= 0 or shape != sorted([(acc["receivable"], amount, 0), (acc["revenue"], 0, amount)]):
                problems.append(f"{name}: invoice {doc} is not posted once as one equal pair")
        elif shape != sorted([(acc["receivable"], expected, 0), (acc["revenue"], 0, expected)]):
            problems.append(f"{name}: invoice {doc} is not posted once at the policy amount")
        if ("period_cutoff" not in declared
                and any(parse_time(r["timestamp"]) < parse_time(invoice["issued_at"]) for r in rows)):
            problems.append(f"{name}: invoice {doc} is posted before its date")
        if any(r["entity_id"] != invoice["entity_id"] for r in rows):
            problems.append(f"{name}: invoice {doc} is posted for another entity")

    closings = expected_closings(books)
    for payment in books.payment_list:
        doc, pid, refs = payment["doc_id"], payment["payment_id"], payment["invoice_refs"]
        declared = allowed.get(doc, set())
        rows = books.by_payment.get(pid, [])
        inside = books.inside(payment["timestamp"])
        if not rows:
            if inside:
                problems.append(f"{name}: payment {doc} is in the period but not posted")
            continue
        if not inside and "period_cutoff" not in declared:
            problems.append(f"{name}: payment {doc} is outside the period but posted")
        if ("period_cutoff" not in declared
                and any(parse_time(r["timestamp"]) < parse_time(payment["timestamp"]) for r in rows)):
            problems.append(f"{name}: payment {doc} is posted before its date")
        entities = {books.invoices[ref]["entity_id"] for ref in refs or [] if ref in books.invoices}
        if (entities and {r["entity_id"] for r in rows} != entities) or len({r["entity_id"] for r in rows}) != 1:
            problems.append(f"{name}: payment {doc} is posted for another entity")
        if any(not books.own.get(ref) for ref in refs or []):
            problems.append(f"{name}: payment {doc} refers to an invoice that is not posted")
            continue
        by_account = defaultdict(list)
        for row in rows:
            by_account[row["account"]].append(row)
        if set(by_account) - {acc["cash"], acc["receivable"], acc["unallocated"], acc["fx_difference"]}:
            problems.append(f"{name}: payment {doc} posts to an account a payment does not use")
        cash = by_account[acc["cash"]]
        expected_cash = books.policy_amount(payment["amount_minor"], payment["currency"], payment["timestamp"])
        if len(cash) != 1 or ("fx_error" not in declared
                              and cash[0]["debit_minor"] - cash[0]["credit_minor"] != expected_cash):
            problems.append(f"{name}: payment {doc} cash differs from the policy amount")
        expected = closings[pid]
        if expected["problem"]:
            problems.append(f"{name}: payment {doc}: {expected['problem']}")
        receivable = by_account[acc["receivable"]]
        actual = {r["invoice_id"]: r["credit_minor"] - r["debit_minor"] for r in receivable}
        unallocated = sum(r["credit_minor"] - r["debit_minor"] for r in by_account[acc["unallocated"]])
        if "allocation_gap" in declared:
            # Part of the closing sits on unallocated instead: one row, a positive gap, and the
            # receivable plus unallocated still add up to what the rule gives.
            gap = unallocated - expected["unallocated"]
            if (len(actual) != len(receivable) or set(actual) != set(expected["closes"])
                    or len(by_account[acc["unallocated"]]) != 1 or gap <= 0
                    or sum(actual.values()) + unallocated != sum(expected["closes"].values()) + expected["unallocated"]):
                problems.append(f"{name}: payment {doc} splits its closing other than by the payment rule and one gap")
        else:
            if len(actual) != len(receivable) or actual != expected["closes"]:
                problems.append(f"{name}: payment {doc} does not close its invoices by the payment rule")
            if len(by_account[acc["unallocated"]]) > 1 or unallocated != expected["unallocated"]:
                problems.append(f"{name}: payment {doc} has an unallocated amount the payment rule does not give")
        fx_rows = by_account[acc["fx_difference"]]
        if len(fx_rows) > 1 or (fx_rows and payment["currency"] == books.base):
            problems.append(f"{name}: payment {doc} has an fx_difference row it cannot have")
        if payment["amount_minor"] < 0 and refs:
            others = books.paid_by.get(refs[0], [])
            if any(books.by_payment.get(p["payment_id"]) and any(r["account"] == acc["unallocated"]
                                                                  for r in books.by_payment[p["payment_id"]])
                   for p in others):
                problems.append(f"{name}: refund {doc} on an invoice with an unallocated amount")
    return problems


def check_books(public: dict, name: str, deviations, counts: bool = True) -> list[str]:
    books = Books(public)
    problems = _check_documents(books, name) + _check_ledger(books, name)
    if problems:
        return problems
    problems += _check_postings(books, name, deviations)
    if counts:
        manifest = public["manifest"]["counts"]
        for table in ("ledger", "invoices", "payments", "fx"):
            if manifest[table] != len(public[table]):
                problems.append(f"manifest: count of {table} differs from the file")
    return problems


# ============================================================================ truth
def _record_table(record: dict) -> str:
    if "row_id" in record:
        return "ledger"
    if "line_items" in record:
        return "invoices"
    return "payments"


def _check_replay(public: dict, private: dict) -> list[str]:
    """The pre-injection snapshot with every step's `before` records replaced by its `after` records
    gives the open records, so the two states differ exactly by the recorded changes."""
    problems = []
    ident = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id"}
    state = {"ledger": {r["row_id"]: r for r in private["pre_ledger"]},
             "invoices": {r["doc_id"]: r for r in private["pre_invoices"]},
             "payments": {r["doc_id"]: r for r in private["pre_payments"]}}
    for step in private["provenance"]["steps"]:
        for record in step["before"]:
            table = _record_table(record)
            if state[table].get(record[ident[table]]) != record:
                problems.append(f"provenance: step {step['issue_id']} changes a record that is not in that state")
            state[table].pop(record[ident[table]], None)
        for record in step["after"]:
            table = _record_table(record)
            if record[ident[table]] in state[table]:
                problems.append(f"provenance: step {step['issue_id']} adds an id that already exists")
            state[table][record[ident[table]]] = record
    for table, key in ident.items():
        if state[table] != {r[key]: r for r in public[table]}:
            problems.append(f"provenance: the snapshot plus the recorded steps does not give the {table}")
    return problems


def _key(item: dict, amount: str) -> tuple:
    return (item["type"], tuple(sorted(item["affected_ids"])), item[amount], item["currency"])


def _evidence_problem(books: Books, issue: dict) -> str | None:
    """Each type's evidence sets must be what docs/CONTRACTS.md section 3.1 lists."""
    sets = sorted(sorted(group) for group in issue["evidence_sets"])
    affected = issue["affected_ids"]
    docs = [books.docs[d] for d in affected]

    def own_rows(doc: dict) -> list[dict]:
        if "invoice_id" in doc and "line_items" in doc:
            return books.own.get(doc["invoice_id"], [])
        return books.by_payment.get(doc["payment_id"], [])

    def sides(doc: dict) -> list[str]:
        rows = own_rows(doc)
        debit = [r["row_id"] for r in rows if r["debit_minor"] > 0]
        credit = [r["row_id"] for r in rows if r["credit_minor"] > 0]
        return debit + credit if len(debit) == 1 and len(credit) == 1 and len(rows) == 2 else []

    def rate_key(doc: dict) -> str:
        moment = doc.get("issued_at", doc.get("timestamp"))
        return fx_key(local_date(moment, books.offset), doc["currency"], books.base)

    kind = issue["type"]
    if kind == "duplicate_invoice":
        a, b = (sides(doc) for doc in docs)
        want = sorted(sorted(affected + [ra, rb]) for ra in a for rb in b)
        return None if a and b and sets == want else "duplicate_invoice needs the four [docA, docB, rowA, rowB] sets"
    acc = books.accounts
    if kind == "allocation_gap":
        payment = next((d for d in docs if "payment_id" in d), None)
        invoice = next((d for d in docs if "line_items" in d), None)
        if payment is None or invoice is None:
            return "allocation_gap needs a payment and an invoice"
        rows = books.by_payment.get(payment["payment_id"], [])
        unallocated = [r["row_id"] for r in rows if r["account"] == acc["unallocated"]]
        closing = [r["row_id"] for r in rows if r["account"] == acc["receivable"]
                   and r["invoice_id"] == invoice["invoice_id"] and r["credit_minor"] > 0]
        debit = [r["row_id"] for r in books.own.get(invoice["invoice_id"], [])
                 if r["account"] == acc["receivable"] and r["debit_minor"] > 0]
        ok = len(unallocated) == len(closing) == len(debit) == 1
        return None if ok and sets == [sorted(affected + unallocated + debit + closing)] \
            else "allocation_gap needs [payment, invoice, unallocated row, invoice receivable debit, payment receivable credit]"
    doc = docs[0]
    key = rate_key(doc) if kind == "fx_error" else None
    if kind == "fx_error" and key not in books.rates:
        return "fx_error: no rate for the document's date"
    if kind == "fx_error" and "payment_id" in doc:
        rows = [r["row_id"] for r in own_rows(doc) if r["account"] == acc["cash"]]
        return None if len(rows) == 1 and sets == [sorted([doc["doc_id"], rows[0], key])] \
            else "fx_error on a payment needs [doc, cash row, fx key]"
    pair = sides(doc)
    extra = [key] if kind == "fx_error" else []
    want = sorted(sorted([doc["doc_id"], row, *extra]) for row in pair)
    return None if pair and sets == want else f"{kind} needs [doc, debit row] and [doc, credit row]" + \
        (" with the fx key" if extra else "")


def _check_rules(books: Books, issue: dict, operator: str) -> str | None:
    """Generator rules of docs/DECISIONS.md B for the record an issue is about."""
    kind, docs = issue["type"], [books.docs[d] for d in issue["affected_ids"]]
    if kind == "fx_error":
        doc = docs[0]
        amount = doc.get("total_minor", doc.get("amount_minor"))
        moment = doc.get("issued_at", doc.get("timestamp"))
        correct = books.policy_amount(amount, doc["currency"], moment)
        if correct is None or issue["expected_amount_minor"] < FX_MIN_MINOR \
                or issue["expected_amount_minor"] * 1000 < FX_MIN_PER_MILLE * abs(correct):
            return "the injected fx difference is below 100 minor units or 0.5 %"
        if "invoice_id" in doc and "line_items" in doc:
            rows = books.own.get(doc["invoice_id"], [])
            if len({r["debit_minor"] + r["credit_minor"] for r in rows}) != 1:
                return "fx_error must distort the debit and the credit of an invoice equally"
    elif kind == "allocation_gap":
        payment = next((d for d in docs if "payment_id" in d), None)
        invoice = next((d for d in docs if "line_items" in d), None)
        if payment is None or invoice is None or payment["invoice_refs"] != [invoice["invoice_id"]]:
            return "allocation_gap needs a payment referring to exactly its invoice"
        if not books.inside(invoice["issued_at"]) or not books.own.get(invoice["invoice_id"]):
            return "the invoice of an allocation_gap must be dated and posted in the period"
        if [p["payment_id"] for p in books.paid_by.get(invoice["invoice_id"], [])] != [payment["payment_id"]]:
            return "the invoice of an allocation_gap has other payments or refunds"
    elif kind == "period_cutoff":
        doc = docs[0]
        if "line_items" not in doc:
            return "period_cutoff targets are invoices"
        moment = parse_time(doc["issued_at"])
        edge = books.start if moment < books.start else books.end
        if abs(moment - edge) >= dt.timedelta(days=CUTOFF_MAX_DAYS):
            return "a cutoff date is more than four days outside the period"
        if operator == "period_cutoff":
            first = books.start.astimezone(parse_offset(books.offset)).date().isoformat()
            last = (books.end - dt.timedelta(seconds=1)).astimezone(parse_offset(books.offset)).date().isoformat()
            if any(item["service_end"] >= first and item["service_start"] <= last for item in doc["line_items"]):
                return "a cutoff invoice keeps a service period inside the period"
    return None


def _check_truth(public: dict, private: dict) -> list[str]:
    problems = []
    truth, provenance = private["truth"], private["provenance"]
    issues = truth["issues"]
    books = Books(public)
    open_ids = set(books.docs) | {r["row_id"] for r in public["ledger"]} | set(books.rates)
    currencies = set(books.exponents)

    issue_ids = [issue["issue_id"] for issue in issues]
    step_ids = [step["issue_id"] for step in provenance["steps"]]
    if len(set(issue_ids)) != len(issue_ids) or sorted(issue_ids) != sorted(step_ids):
        problems.append("truth: issues and provenance steps do not match one to one")
    operator = {step["issue_id"]: step["operator"] for step in provenance["steps"]}
    config = provenance.get("config")
    if config:
        scenario = {"I1": ("allocation_gap", "fx_error"), "I2": ("duplicate_invoice", "period_cutoff")}
        for name, count in sorted(config["errors"].items()):
            count += scenario.get(config["interaction"], ()).count(name)
            if sum(1 for issue in issues if issue["type"] == name) != count:
                problems.append(f"truth: number of {name} issues differs from the config")

    for issue in issues:
        tag = f"truth {issue['issue_id']}"
        affected = issue["affected_ids"]
        if not affected or affected != sorted(set(affected)) or not set(affected) <= set(books.docs):
            problems.append(f"{tag}: deviation: affected_ids must be sorted existing doc ids")
            continue
        amount = issue["expected_amount_minor"]
        if not _is_int(amount) or amount <= 0 or issue["currency"] not in currencies:
            problems.append(f"{tag}: deviation: amount must be a positive integer in a known currency")
        if not issue["evidence_sets"]:
            problems.append(f"{tag}: evidence: no evidence set")
        for group in issue["evidence_sets"]:
            if not group or len(group) > MAX_EVIDENCE or len(set(group)) != len(group):
                problems.append(f"{tag}: evidence: a set must have 1..{MAX_EVIDENCE} distinct ids")
            if not set(group) <= open_ids:
                problems.append(f"{tag}: evidence: an id is not in the open records")
        insufficient = _evidence_problem(books, issue)
        if insufficient:
            problems.append(f"{tag}: evidence: {insufficient}")
        broken = _check_rules(books, issue, operator.get(issue["issue_id"], ""))
        if broken:
            problems.append(f"{tag}: rule: {broken}")

    found, undefined = predicate_issues(public)
    problems += [f"truth: undefined case: {case}" for case in undefined]
    by_target = {(i["type"], tuple(i["affected_ids"])): i for i in found}
    declared = {}
    for issue in issues:
        target = (issue["type"], tuple(sorted(issue["affected_ids"])))
        declared[target] = issue
        hit = by_target.get(target)
        if hit is None:
            problems.append(f"truth {issue['issue_id']}: deviation: the {issue['type']} definition does not hold")
        elif (hit["amount"], hit["currency"]) != (issue["expected_amount_minor"], issue["currency"]):
            problems.append(f"truth {issue['issue_id']}: deviation: measured {(hit['amount'], hit['currency'])} "
                            f"differs from the declared amount")
    for target in sorted(set(by_target) - set(declared)):
        problems.append(f"truth: the {target[0]} definition holds on {list(target[1])} but no issue lists it")
    if len(found) != len(by_target):
        problems.append("truth: a definition holds twice on the same documents")
    return problems


def _check_generator_rules(public: dict) -> list[str]:
    counts = defaultdict(int)
    for invoice in public["invoices"]:
        counts[invoice_signature(invoice)] += 1
    return ["rules: more than two invoices share a signature"] if any(n > 2 for n in counts.values()) else []


def verify(episode: dict) -> list[str]:
    try:
        return _verify(episode)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, InvalidOperation) as error:
        return [f"malformed episode: {type(error).__name__}"]


def _verify(episode: dict) -> list[str]:
    public, private = episode["public"], episode["private"]
    issues = private["truth"]["issues"]
    pre_public = dict(public, ledger=private["pre_ledger"], invoices=private["pre_invoices"],
                      payments=private["pre_payments"])
    problems = _check_fx(public, Books(public))
    problems += check_books(pre_public, "pre", set(), counts=False)
    if not problems:
        found, undefined = predicate_issues(pre_public)
        problems += [f"pre: the {i['type']} definition already holds on {i['affected_ids']}" for i in found]
        problems += [f"pre: undefined case: {case}" for case in undefined]
    deviations: dict[str, set[str]] = defaultdict(set)
    for issue in issues:
        if issue["type"] in POSTING_TYPES:
            for doc in issue["affected_ids"]:
                deviations[doc].add(issue["type"])
    books_problems = check_books(public, "ledger", dict(deviations))
    problems += books_problems
    problems += _check_replay(public, private)
    if not books_problems:
        problems += _check_truth(public, private)
    problems += _check_generator_rules(public)
    if private["truth"]["episode_id"] != public["manifest"]["episode_id"]:
        problems.append("truth: episode_id differs from the manifest")
    return problems
