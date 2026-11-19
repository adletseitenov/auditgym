"""Independent solver: findings from the open records only (docs/SPEC.md "Определения ошибок",
overridden by docs/DECISIONS.md where they differ).

solve(public) evaluates the four published predicates over the final open records and returns
findings in the shape of docs/CONTRACTS.md section 6. It never reads `private` and does not
import the generator or the error operators.

Every predicate reads document dates and ids only; posting timestamps are never compared with
document dates (a legitimate posting may lag its document). Dates are compared as instants and
converted to the company offset for the rate date.

Readings chosen where the definitions leave a case open:
- duplicate_invoice fires for every pair of posted invoices with the same signature, so three
  identical posted invoices give three findings; an unposted copy gives none. Evidence: both
  doc_ids plus the rows of each invoice's own posting group (normally Dr and Cr, so six ids).
- allocation_gap needs exactly one unallocated row with a positive credit for the payment;
  a payment with several such rows is left unreported. Evidence: both doc_ids, the unallocated
  row, the invoice's receivable debit and the payment's receivable credit for that invoice.
- fx_error is evaluated for every posted foreign-currency document, in or out of the period, at
  the rate of the document's own company-local date (DECISIONS A4). An error only when
  |booked - correct| is strictly greater than the policy tolerance. A base-currency document is
  never an fx_error, nor is a payment posted without a cash row (nothing converted it). Only
  conversion rows count: the receivable debit and revenue credit of an invoice, the cash of a
  payment. Evidence: the doc_id, the conversion rows and the fx key of the document's date.
- period_cutoff reports the absolute document amount, so a refund never yields a negative claim.

Robustness: a record that lacks a piece a predicate needs (a rate for its date, a parseable
timestamp with an offset, a numeric amount) is skipped by that predicate only; solve() never
raises on it. A timestamp without an offset is malformed (its instant would depend on the
machine's time zone). An id or an FX key that two records share with different content is
ambiguous and treated as missing, so the answer never depends on the order of records. An
unusable policy (missing, or without base currency, offset or period) gives no findings.
"""
from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal

from .limits import MAX_EVIDENCE_IDS as MAX_EVIDENCE
from .money import convert, fx_key, in_period, local_date, parse_time

ROWS_PER_DUPLICATE_INVOICE = 4  # 2 docs + 2 x 4 rows = 10 at most

# What a malformed record can raise inside one predicate. Caught per record, never globally.
_MALFORMED = (KeyError, TypeError, ValueError, ArithmeticError, AttributeError, IndexError)

DEFAULT_ACCOUNTS = {name: name for name in ("cash", "receivable", "revenue", "unallocated", "fx_difference",
                                            "expense")}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_items(line_items) -> tuple:
    """Line items as a multiset: every field of every item, order of items ignored."""
    return tuple(sorted(json.dumps(item, sort_keys=True, ensure_ascii=False) for item in line_items or []))


def _signature(invoice: dict) -> tuple:
    return (invoice["entity_id"], invoice["currency"], invoice["total_minor"],
            _canonical_items(invoice.get("line_items")))


def _row_order(row: dict) -> tuple:
    """Content first, id last: the choice of rows does not depend on how ids are spelled."""
    return (row["account"], -row["debit_minor"], -row["credit_minor"], row["timestamp"], row["row_id"])


def _valid_row(row) -> bool:
    if not isinstance(row, dict):
        return False
    try:
        return (isinstance(row["row_id"], str) and isinstance(row["account"], str)
                and isinstance(row["timestamp"], str)
                and _is_int(row["debit_minor"]) and _is_int(row["credit_minor"])
                and (row["invoice_id"] is None or isinstance(row["invoice_id"], str))
                and (row["payment_id"] is None or isinstance(row["payment_id"], str)))
    except KeyError:
        return False


def _aware(timestamp: object):
    """Parsed ISO timestamp; ValueError unless it carries an offset."""
    if not isinstance(timestamp, str):
        raise ValueError("timestamp is not a string")
    moment = parse_time(timestamp)
    if moment.tzinfo is None:
        raise ValueError("timestamp without an offset")
    return moment


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _unique_by(records: list[dict], field: str) -> dict:
    """Records by a string id; an id carried by two different records is ambiguous and left out
    (an exact repeat of one record is not a conflict)."""
    found: dict = {}
    ambiguous: set = set()
    for record in records:
        key = record.get(field)
        if not isinstance(key, str):
            continue
        if key in found and found[key] != record:
            ambiguous.add(key)
        found.setdefault(key, record)
    return {key: record for key, record in found.items() if key not in ambiguous}


class _Books:
    def __init__(self, public: dict) -> None:
        self.policy = public["policy"]
        self.base = self.policy["base_currency"]
        self.offset = self.policy["timezone"]
        self.start = self.policy["period_start"]
        self.end = self.policy["period_end"]
        if not all(isinstance(value, str) for value in (self.base, self.offset)):
            raise ValueError("policy needs a base currency and an offset")
        _aware(self.start), _aware(self.end)
        self.accounts = {**DEFAULT_ACCOUNTS, **_dict(self.policy.get("accounts"))}
        self.exponents = _dict(self.policy.get("currency_exponents"))
        self.tolerance = _dict(self.policy.get("amount_tolerance_minor"))
        self.invoices = [inv for inv in public.get("invoices") or [] if isinstance(inv, dict)]
        self.payments = [pay for pay in public.get("payments") or [] if isinstance(pay, dict)]
        self.invoice_by_id = _unique_by(self.invoices, "invoice_id")
        self.payment_by_id = _unique_by(self.payments, "payment_id")
        self.rates: dict[str, Decimal] = {}
        conflicting: set = set()
        for rate in public.get("fx") or []:
            try:
                key = fx_key(rate["date"], rate["source_currency"], rate["target_currency"])
                value = Decimal(rate["rate"])
            except _MALFORMED:
                continue  # an unreadable rate is the same as a missing one
            if not value.is_finite() or value <= 0:
                continue
            if key in self.rates and self.rates[key] != value:
                conflicting.add(key)  # two different rates for one day: neither can be trusted
            self.rates[key] = value
        for key in conflicting:
            del self.rates[key]

        # Event groups (CONTRACTS 2.3) and receivable balances per invoice, in one pass.
        self.invoice_group: dict[str, list[dict]] = defaultdict(list)
        self.payment_group: dict[str, list[dict]] = defaultdict(list)
        self.receivable: dict[str, int] = defaultdict(int)
        receivable = self.accounts["receivable"]
        for row in public.get("ledger") or []:
            if not _valid_row(row):
                continue
            if row["payment_id"] is not None:
                self.payment_group[row["payment_id"]].append(row)
            elif row["invoice_id"] is not None:
                self.invoice_group[row["invoice_id"]].append(row)
            if row["account"] == receivable and row["invoice_id"] is not None:
                self.receivable[row["invoice_id"]] += row["debit_minor"] - row["credit_minor"]

    def tol(self, kind: str) -> int:
        value = self.tolerance.get(kind, 0)
        return value if _is_int(value) else 0

    def inside(self, timestamp: str) -> bool:
        _aware(timestamp)
        return in_period(timestamp, self.start, self.end)

    def rate_key(self, currency: str, timestamp: str) -> str:
        return fx_key(local_date(timestamp, self.offset), currency, self.base)

    def policy_amount(self, amount_minor: int, currency: str, timestamp: str) -> int | None:
        """Document amount at the rate of its own company-local date, or None when there is no rate."""
        if not _is_int(amount_minor):
            return None
        _aware(timestamp)
        rate = self.rates.get(fx_key(local_date(timestamp, self.offset), currency, self.base))
        if rate is None or currency not in self.exponents or self.base not in self.exponents:
            return None
        return convert(amount_minor, rate, self.exponents[currency], self.exponents[self.base])


def _each(records, predicate) -> list[dict]:
    """Apply a per-record predicate; a malformed record is skipped by that predicate only."""
    findings = []
    for record in records:
        try:
            finding = predicate(record)
        except _MALFORMED:
            continue
        if finding:
            findings.append(finding)
    return findings


def _duplicates(books: _Books) -> list[dict]:
    by_signature: dict[tuple, list[dict]] = defaultdict(list)

    def collect(invoice: dict) -> None:
        if (books.invoice_group.get(invoice["invoice_id"]) and isinstance(invoice["doc_id"], str)
                and _is_int(invoice["total_minor"])):
            by_signature[_signature(invoice)].append(invoice)

    _each(books.invoices, collect)
    findings = []
    for signature in sorted(by_signature, key=repr):
        group = sorted(by_signature[signature], key=lambda inv: inv["doc_id"])
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                if first["doc_id"] == second["doc_id"] or first["invoice_id"] == second["invoice_id"]:
                    continue
                ids = sorted([first["doc_id"], second["doc_id"]])
                rows = []
                for invoice in (first, second):
                    own = sorted(books.invoice_group[invoice["invoice_id"]], key=_row_order)
                    rows += [row["row_id"] for row in own[:ROWS_PER_DUPLICATE_INVOICE]]
                findings.append({
                    "type": "duplicate_invoice", "affected_ids": ids,
                    "claimed_amount_minor": abs(first["total_minor"]), "currency": first["currency"],
                    "evidence_ids": ids + rows,
                    "explanation": "Both invoices are posted and share customer, currency, total and line items.",
                })
    return findings


def _allocation_gaps(books: _Books) -> list[dict]:
    unallocated, receivable = books.accounts["unallocated"], books.accounts["receivable"]

    def gaps(item) -> list[dict]:
        payment_id, rows = item
        payment = books.payment_by_id.get(payment_id)
        if payment is None or not payment.get("invoice_refs"):
            return []  # unknown payment, or an advance without references
        credits = [row for row in rows if row["account"] == unallocated and row["credit_minor"] > 0]
        if len(credits) != 1:
            return []  # no gap, or several unallocated rows: undefined, left unreported
        row = credits[0]
        found = []
        for ref in sorted(set(payment["invoice_refs"])):
            invoice = books.invoice_by_id.get(ref)
            if invoice is None or books.receivable.get(ref, 0) <= 0:
                continue
            debits = sorted((r for r in books.invoice_group.get(ref, [])
                             if r["account"] == receivable and r["debit_minor"] > 0), key=_row_order)
            closings = sorted((r for r in rows if r["account"] == receivable and r["invoice_id"] == ref
                               and r["credit_minor"] > 0), key=_row_order)
            context = [r["row_id"] for r in debits[:3] + closings[:3]]
            found.append({
                "type": "allocation_gap", "affected_ids": sorted([payment["doc_id"], invoice["doc_id"]]),
                "claimed_amount_minor": row["credit_minor"], "currency": books.base,
                "evidence_ids": [payment["doc_id"], invoice["doc_id"], row["row_id"], *context],
                "explanation": "Part of the payment sits on unallocated while the invoice it refers to stays open.",
            })
        return found

    nested = _each(sorted(books.payment_group.items()), gaps)
    return [finding for group in nested for finding in group]


def _fx_finding(books: _Books, doc_id: str, rate_key: str, deviations: list[tuple[int, dict]]) -> dict | None:
    """deviations: (|booked - correct|, row) for every conversion row of the document."""
    if not deviations:
        return None
    worst = max(amount for amount, _ in deviations)
    if worst <= books.tol("fx_error"):
        return None
    rows = [row["row_id"] for _, row in sorted(deviations, key=lambda pair: (-pair[0], _row_order(pair[1])))]
    return {
        "type": "fx_error", "affected_ids": [doc_id], "claimed_amount_minor": worst,
        "currency": books.base, "evidence_ids": [doc_id, rate_key, *rows[:MAX_EVIDENCE - 2]],
        "explanation": "The posted base amount differs from document amount times the rate of the document date.",
    }


def _fx_errors(books: _Books) -> list[dict]:
    receivable, revenue, cash = (books.accounts[name] for name in ("receivable", "revenue", "cash"))

    def invoice_fx(invoice: dict) -> dict | None:
        rows = books.invoice_group.get(invoice["invoice_id"])
        if not rows or invoice["currency"] == books.base:
            return None
        correct = books.policy_amount(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        if correct is None:
            return None
        deviations = [(abs(row["debit_minor"] - correct), row) for row in rows
                      if row["account"] == receivable and row["debit_minor"] > 0]
        deviations += [(abs(row["credit_minor"] - correct), row) for row in rows
                       if row["account"] == revenue and row["credit_minor"] > 0]
        return _fx_finding(books, invoice["doc_id"], books.rate_key(invoice["currency"], invoice["issued_at"]),
                           deviations)

    def payment_fx(payment: dict) -> dict | None:
        rows = books.payment_group.get(payment["payment_id"])
        if not rows or payment["currency"] == books.base:
            return None
        correct = books.policy_amount(payment["amount_minor"], payment["currency"], payment["timestamp"])
        cash_rows = [row for row in rows if row["account"] == cash]
        if correct is None or not cash_rows:
            return None
        difference = abs(sum(row["debit_minor"] - row["credit_minor"] for row in cash_rows) - correct)
        return _fx_finding(books, payment["doc_id"], books.rate_key(payment["currency"], payment["timestamp"]),
                           [(difference, row) for row in cash_rows])

    return _each(books.invoices, invoice_fx) + _each(books.payments, payment_fx)


def _cutoffs(books: _Books) -> list[dict]:
    def cutoff(document: tuple) -> dict | None:
        doc_id, moment, amount, currency, rows = document
        if not rows or books.inside(moment) or not _is_int(amount):
            return None
        evidence = [row["row_id"] for row in sorted(rows, key=_row_order)][:MAX_EVIDENCE - 1]
        return {
            "type": "period_cutoff", "affected_ids": [doc_id], "claimed_amount_minor": abs(amount),
            "currency": currency, "evidence_ids": [doc_id, *evidence],
            "explanation": "The document is dated outside the period in the company timezone but posted in it.",
        }

    def invoice_doc(inv: dict) -> tuple:
        return (inv["doc_id"], inv["issued_at"], inv["total_minor"], inv["currency"],
                books.invoice_group.get(inv["invoice_id"]))

    def payment_doc(pay: dict) -> tuple:
        return (pay["doc_id"], pay["timestamp"], pay["amount_minor"], pay["currency"],
                books.payment_group.get(pay["payment_id"]))

    return (_each(books.invoices, lambda inv: cutoff(invoice_doc(inv)))
            + _each(books.payments, lambda pay: cutoff(payment_doc(pay))))


def _sort_key(finding: dict) -> tuple:
    return (finding["type"], finding["affected_ids"], finding["claimed_amount_minor"], finding["currency"],
            finding["evidence_ids"])


def solve(public: dict) -> list[dict]:
    try:
        books = _Books(public)
    except _MALFORMED:
        return []  # no usable policy: nothing can be decided from these records
    findings = _duplicates(books) + _allocation_gaps(books) + _fx_errors(books) + _cutoffs(books)
    # A finding must be well-formed: string ids, a non-negative int amount, at most 10 evidence ids.
    findings = [f for f in findings if all(isinstance(i, str) for i in f["affected_ids"] + f["evidence_ids"])
                and isinstance(f["currency"], str) and _is_int(f["claimed_amount_minor"])
                and f["claimed_amount_minor"] >= 0 and len(f["evidence_ids"]) <= MAX_EVIDENCE]
    return sorted(findings, key=_sort_key)
