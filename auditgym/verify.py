"""verify(episode) -> list of problems. An empty list means the episode is well formed.

Checks the open records and the pre-injection ledger with nothing but the published policy,
the way an outside checker would. It does not import the generator.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from .money import convert, fx_key, in_period, local_date

LEDGER_FIELDS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                 "entity_id", "invoice_id", "payment_id"]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def group_key(row: dict) -> tuple:
    """Event group of a posting (docs/CONTRACTS.md, section 2.3)."""
    if row["payment_id"] is not None:
        return ("payment", row["payment_id"])
    if row["invoice_id"] is not None:
        return ("invoice", row["invoice_id"])
    return ("background", row["timestamp"], row["entity_id"])


class _Books:
    """Indexes over the open records."""

    def __init__(self, public: dict) -> None:
        self.policy = public["policy"]
        self.base = self.policy["base_currency"]
        self.offset = self.policy["timezone"]
        self.invoices = {inv["invoice_id"]: inv for inv in public["invoices"]}
        self.payments = {pay["payment_id"]: pay for pay in public["payments"]}
        self.rates = {fx_key(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"])
                      for r in public["fx"]}

    def to_base(self, amount_minor: int, currency: str, timestamp: str) -> int | None:
        if currency == self.base:
            return amount_minor
        rate = self.rates.get(fx_key(local_date(timestamp, self.offset), currency, self.base))
        if rate is None:
            return None
        exponents = self.policy["currency_exponents"]
        return convert(amount_minor, rate, exponents[currency], exponents[self.base])


def _check_ledger_shape(rows: list[dict], books: _Books, name: str) -> list[str]:
    problems = []
    accounts = set(books.policy["accounts"].values())
    seen = set()
    for row in rows:
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
        if not in_period(row["timestamp"], books.policy["period_start"], books.policy["period_end"]):
            problems.append(f"{name}: row {rid} is outside the period")
    return problems


def _check_balance(rows: list[dict], name: str) -> list[str]:
    problems = []
    if sum(r["debit_minor"] for r in rows) != sum(r["credit_minor"] for r in rows):
        problems.append(f"{name}: debits and credits differ")
    groups = defaultdict(int)
    for row in rows:
        groups[group_key(row)] += row["debit_minor"] - row["credit_minor"]
    for key, balance in sorted(groups.items(), key=repr):
        if balance != 0:
            problems.append(f"{name}: group {key} is unbalanced by {balance}")
    return problems


def _check_documents(public: dict, books: _Books) -> list[str]:
    problems = []
    doc_ids = [d["doc_id"] for d in public["invoices"]] + [d["doc_id"] for d in public["payments"]]
    if len(set(doc_ids)) != len(doc_ids):
        problems.append("documents: doc_id is not unique")
    if len(books.invoices) != len(public["invoices"]) or len(books.payments) != len(public["payments"]):
        problems.append("documents: invoice_id or payment_id is not unique")
    for invoice in public["invoices"]:
        if invoice["total_minor"] != sum(item["amount_minor"] for item in invoice["line_items"]):
            problems.append(f"invoice {invoice['doc_id']}: total differs from its line items")
        if books.to_base(invoice["total_minor"], invoice["currency"], invoice["issued_at"]) is None:
            problems.append(f"invoice {invoice['doc_id']}: no rate for its date")
    for payment in public["payments"]:
        for ref in payment["invoice_refs"] or []:
            if ref not in books.invoices:
                problems.append(f"payment {payment['doc_id']}: refers to a missing invoice")
        if books.to_base(payment["amount_minor"], payment["currency"], payment["timestamp"]) is None:
            problems.append(f"payment {payment['doc_id']}: no rate for its date")
    counts = public["manifest"]["counts"]
    for table in ("ledger", "invoices", "payments", "fx"):
        if counts[table] != len(public[table]):
            problems.append(f"manifest: count of {table} differs from the file")
    return problems


def _check_clean_postings(rows_in: list[dict], books: _Books, name: str, exempt: set[str]) -> list[str]:
    """Every document in the period is posted once, at the policy amount, at its own time; nothing
    outside the period is posted. Documents in `exempt` (affected by an issue whose definition is
    about the posting itself) are checked by the issue checks instead."""
    problems = []
    policy = books.policy
    by_invoice = defaultdict(list)
    by_payment = defaultdict(list)
    for row in rows_in:
        if row["payment_id"] is not None:
            by_payment[row["payment_id"]].append(row)
        elif row["invoice_id"] is not None:
            by_invoice[row["invoice_id"]].append(row)

    booked = {}
    for invoice_id, invoice in books.invoices.items():
        doc = invoice["doc_id"]
        expected = books.to_base(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        booked[invoice_id] = expected
        if doc in exempt:
            continue
        rows = by_invoice.get(invoice_id, [])
        inside = in_period(invoice["issued_at"], policy["period_start"], policy["period_end"])
        if not rows:
            if inside:
                problems.append(f"{name}: invoice {doc} is in the period but not posted")
            continue
        if not inside:
            problems.append(f"{name}: invoice {doc} is outside the period but posted")
        debit = [r for r in rows if r["account"] == "receivable" and r["debit_minor"] == expected]
        credit = [r for r in rows if r["account"] == "revenue" and r["credit_minor"] == expected]
        if len(rows) != 2 or len(debit) != 1 or len(credit) != 1:
            problems.append(f"{name}: invoice {doc} is not posted once at the policy amount")
        if any(r["timestamp"] != invoice["issued_at"] for r in rows):
            problems.append(f"{name}: invoice {doc} is posted at another time")

    closed = defaultdict(int)
    for payment_id, payment in books.payments.items():
        doc = payment["doc_id"]
        if doc in exempt:
            continue
        rows = by_payment.get(payment_id, [])
        inside = in_period(payment["timestamp"], policy["period_start"], policy["period_end"])
        if not rows:
            if inside:
                problems.append(f"{name}: payment {doc} is in the period but not posted")
            continue
        if not inside:
            problems.append(f"{name}: payment {doc} is outside the period but posted")
        expected = books.to_base(payment["amount_minor"], payment["currency"], payment["timestamp"])
        cash = sum(r["debit_minor"] - r["credit_minor"] for r in rows if r["account"] == "cash")
        if cash != expected:
            problems.append(f"{name}: payment {doc} cash differs from the policy amount")
        if any(r["timestamp"] != payment["timestamp"] for r in rows):
            problems.append(f"{name}: payment {doc} is posted at another time")
        refs = set(payment["invoice_refs"] or [])
        for row in rows:
            if row["account"] == "receivable":
                if row["invoice_id"] not in refs:
                    problems.append(f"{name}: payment {doc} closes an invoice it does not refer to")
                closed[row["invoice_id"]] += row["credit_minor"] - row["debit_minor"]
    for invoice_id, amount in sorted(closed.items()):
        if invoice_id in booked and amount > booked[invoice_id]:
            problems.append(f"{name}: invoice {books.invoices[invoice_id]['doc_id']} is closed for more than it was booked")
    return problems


# ---------------------------------------------------------------------------------------------
# State before injection, rebuilt from the provenance records (public ids only).

def _record_table(record: dict) -> str:
    if "row_id" in record:
        return "ledger"
    if "line_items" in record:
        return "invoices"
    return "payments"


def _pre_public(public: dict, steps: list[dict]) -> dict:
    """Open documents as they were before injection: undo each step's documents, last step first."""
    tables = {name: {record["doc_id"]: record for record in public[name]} for name in ("invoices", "payments")}
    for step in reversed(steps):
        for record in step["after"]:
            table = _record_table(record)
            if table != "ledger":
                tables[table].pop(record["doc_id"], None)
        for record in step["before"]:
            table = _record_table(record)
            if table != "ledger":
                tables[table][record["doc_id"]] = record
    return dict(public, invoices=list(tables["invoices"].values()), payments=list(tables["payments"].values()))


def _check_replay(public: dict, private: dict) -> list[str]:
    """pre_ledger with every step's `before` rows replaced by its `after` rows equals the ledger,
    so the two books differ exactly by the injected rows, and kept rows keep their row_id."""
    problems = []
    state = {row["row_id"]: row for row in private["pre_ledger"]}
    for step in private["provenance"]["steps"]:
        before = [r for r in step["before"] if _record_table(r) == "ledger"]
        after = [r for r in step["after"] if _record_table(r) == "ledger"]
        for row in before:
            if state.get(row["row_id"]) != row:
                problems.append(f"provenance: step {step['issue_id']} removes a row that is not in the books")
            state.pop(row["row_id"], None)
        for row in after:
            if row["row_id"] in state:
                problems.append(f"provenance: step {step['issue_id']} adds a row id that already exists")
            state[row["row_id"]] = row
    ledger = {row["row_id"]: row for row in public["ledger"]}
    if state != ledger:
        problems.append("provenance: pre_ledger plus the recorded steps does not give the ledger")
    return problems


# ---------------------------------------------------------------------------------------------
# Truth: declared deviation, evidence, affected documents.

MAX_EVIDENCE = 10
POSTING_TYPES = ("period_cutoff", "fx_error", "allocation_gap")  # issues about how a document is posted


def invoice_signature(invoice: dict) -> tuple:
    """Customer, currency, total and line items as a multiset (issued_at is not part of it)."""
    items = sorted(tuple(sorted(item.items())) for item in invoice["line_items"])
    return (invoice["entity_id"], invoice["currency"], invoice["total_minor"], tuple(items))


def _own_rows(rows: list[dict], invoice_id: str) -> list[dict]:
    return [r for r in rows if r["invoice_id"] == invoice_id and r["payment_id"] is None]


def _posted_amount(rows: list[dict]) -> tuple[int, int] | None:
    """(receivable debit, revenue credit) of an invoice posting pair, or None if it is not a pair."""
    debit = [r for r in rows if r["account"] == "receivable" and r["debit_minor"] > 0]
    credit = [r for r in rows if r["account"] == "revenue" and r["credit_minor"] > 0]
    if len(rows) != 2 or len(debit) != 1 or len(credit) != 1:
        return None
    return debit[0]["debit_minor"], credit[0]["credit_minor"]


def measure_duplicate(public: dict, pre_public: dict, pre_ledger: list[dict], issue: dict) -> tuple[int, str] | str:
    """Deviation of a duplicate_invoice issue, in the invoice currency.

    Exactly one of the two affected invoices (the copy) did not exist before injection and had no
    postings in pre_ledger; the ledger now carries its posting pair at the policy amount. The
    deviation is that copy's total_minor: revenue recognised twice for the same goods.
    Returns (amount, currency) or a problem message.
    """
    docs = {d["doc_id"]: d for d in public["invoices"]}
    pre_docs = {d["doc_id"] for d in pre_public["invoices"]}
    affected = [docs.get(doc) for doc in issue["affected_ids"]]
    if len(affected) != 2 or None in affected:
        return "affected_ids must be two invoices"
    if invoice_signature(affected[0]) != invoice_signature(affected[1]):
        return "the two invoices do not share a signature"
    copies = [d for d in affected if d["doc_id"] not in pre_docs]
    if len(copies) != 1:
        return "exactly one affected invoice must be created by injection"
    copy_invoice = copies[0]
    if _own_rows(pre_ledger, copy_invoice["invoice_id"]):
        return "the copy is already posted before injection"
    books = _Books(public)
    for invoice in affected:
        posted = _posted_amount(_own_rows(public["ledger"], invoice["invoice_id"]))
        expected = books.to_base(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        if posted != (expected, expected):
            return f"invoice {invoice['doc_id']} is not posted at its policy amount"
    return copy_invoice["total_minor"], copy_invoice["currency"]


def measure_cutoff(public: dict, pre_public: dict, pre_ledger: list[dict], issue: dict) -> tuple[int, str] | str:
    """Deviation of a period_cutoff issue, in the document currency.

    The document was dated inside the period before injection and is dated outside it now (in the
    company timezone), while its whole posting pair stays in the period's ledger. The deviation is
    the document amount recognised in the wrong period: total_minor, since the pair carries the
    policy amount of the full total at the document's own date (so it is not also an fx_error).
    """
    policy = public["policy"]
    docs = {d["doc_id"]: d for d in public["invoices"]}
    pre_docs = {d["doc_id"]: d for d in pre_public["invoices"]}
    if len(issue["affected_ids"]) != 1 or issue["affected_ids"][0] not in docs:
        return "affected_ids must be one invoice"
    invoice = docs[issue["affected_ids"][0]]
    before = pre_docs.get(invoice["doc_id"])
    if before is None or not in_period(before["issued_at"], policy["period_start"], policy["period_end"]):
        return "the invoice was not dated inside the period before injection"
    if in_period(invoice["issued_at"], policy["period_start"], policy["period_end"]):
        return "the invoice is dated inside the period"
    rows = _own_rows(public["ledger"], invoice["invoice_id"])
    if any(not in_period(r["timestamp"], policy["period_start"], policy["period_end"]) for r in rows):
        return "a posting of the invoice is outside the period"
    expected = _Books(public).to_base(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
    if _posted_amount(rows) != (expected, expected):
        return "the invoice is not posted in the period at its policy amount"
    if {r["row_id"] for r in rows} != {r["row_id"] for r in _own_rows(pre_ledger, invoice["invoice_id"])}:
        return "the posting rows are not the rows the invoice had before injection"
    return invoice["total_minor"], invoice["currency"]


MEASURES = {"duplicate_invoice": measure_duplicate, "period_cutoff": measure_cutoff}


def _evidence_sufficient(public: dict, issue: dict) -> str | None:
    """Each type's evidence sets must contain what docs/CONTRACTS.md section 3.1 lists."""
    sets = [set(group) for group in issue["evidence_sets"]]
    affected = set(issue["affected_ids"])
    if issue["type"] == "duplicate_invoice":
        if sets != [affected]:
            return "duplicate_invoice evidence must be exactly both invoices"
    elif issue["type"] == "period_cutoff":
        doc = issue["affected_ids"][0]
        invoice = next((d for d in public["invoices"] if d["doc_id"] == doc), None)
        if invoice is None:
            return "period_cutoff evidence needs an invoice"
        rows = {r["row_id"]: r for r in _own_rows(public["ledger"], invoice["invoice_id"])}
        sides = []
        for group in sets:
            others = group - {doc}
            if doc not in group or len(others) != 1 or next(iter(others)) not in rows:
                return "period_cutoff evidence must be pairs of the document and one of its posting rows"
            row = rows[next(iter(others))]
            sides.append("debit" if row["debit_minor"] > 0 else "credit")
        if sorted(sides) != ["credit", "debit"]:
            return "period_cutoff evidence needs one debit and one credit row"
    return None


def _check_truth(public: dict, private: dict, pre_public: dict) -> list[str]:
    problems = []
    truth, provenance = private["truth"], private["provenance"]
    issues = truth["issues"]
    doc_ids = {d["doc_id"] for d in public["invoices"]} | {d["doc_id"] for d in public["payments"]}
    open_ids = doc_ids | {r["row_id"] for r in public["ledger"]} | {
        fx_key(r["date"], r["source_currency"], r["target_currency"]) for r in public["fx"]}
    currencies = set(public["policy"]["currency_exponents"])

    issue_ids = [issue["issue_id"] for issue in issues]
    if len(set(issue_ids)) != len(issue_ids):
        problems.append("truth: issue_id is not unique")
    if sorted(issue_ids) != sorted(step["issue_id"] for step in provenance["steps"]):
        problems.append("truth: issues and provenance steps do not match one to one")
    config = provenance.get("config")
    if config:
        for name, count in sorted(config["errors"].items()):
            if sum(1 for issue in issues if issue["type"] == name) != count:
                problems.append(f"truth: number of {name} issues differs from the config")

    for issue in issues:
        tag = f"truth {issue['issue_id']}"
        affected = issue["affected_ids"]
        if not affected or affected != sorted(set(affected)) or not set(affected) <= doc_ids:
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
        measure = MEASURES.get(issue["type"])
        if measure is None:
            problems.append(f"{tag}: deviation: no measure for {issue['type']}")
            continue
        measured = measure(public, pre_public, private["pre_ledger"], issue)
        if isinstance(measured, str):
            problems.append(f"{tag}: deviation: {measured}")
        elif measured != (amount, issue["currency"]):
            problems.append(f"{tag}: deviation: measured {measured} differs from the declared amount")
        insufficient = _evidence_sufficient(public, issue)
        if insufficient:
            problems.append(f"{tag}: evidence: {insufficient}")
    return problems


def _check_no_unlisted(public: dict, issues: list[dict]) -> list[str]:
    """The predicates of the implemented types fire only where the truth says so."""
    problems = []
    policy = public["policy"]
    posted_invoices = {r["invoice_id"] for r in public["ledger"] if r["invoice_id"] and r["payment_id"] is None}
    by_signature = defaultdict(list)
    for invoice in public["invoices"]:
        if invoice["invoice_id"] in posted_invoices:
            by_signature[invoice_signature(invoice)].append(invoice["doc_id"])
    pairs = set()
    for docs in by_signature.values():
        docs = sorted(docs)
        pairs |= {(a, b) for i, a in enumerate(docs) for b in docs[i + 1:]}
    declared = {tuple(issue["affected_ids"]) for issue in issues if issue["type"] == "duplicate_invoice"}
    if pairs != declared:
        problems.append("truth: duplicate invoice pairs in the records differ from the declared ones")

    posted_payments = {r["payment_id"] for r in public["ledger"] if r["payment_id"]}
    outside = {inv["doc_id"] for inv in public["invoices"] if inv["invoice_id"] in posted_invoices
               and not in_period(inv["issued_at"], policy["period_start"], policy["period_end"])}
    outside |= {pay["doc_id"] for pay in public["payments"] if pay["payment_id"] in posted_payments
                and not in_period(pay["timestamp"], policy["period_start"], policy["period_end"])}
    declared_cutoff = {doc for issue in issues if issue["type"] == "period_cutoff" for doc in issue["affected_ids"]}
    if outside != declared_cutoff:
        problems.append("truth: documents dated outside the period but posted differ from the declared ones")
    return problems


def verify(episode: dict) -> list[str]:
    public, private = episode["public"], episode["private"]
    steps = private["provenance"]["steps"]
    issues = private["truth"]["issues"]
    pre_public = _pre_public(public, steps)
    books = _Books(public)
    pre_books = _Books(pre_public)
    problems = _check_documents(public, books)
    problems += _check_ledger_shape(public["ledger"], books, "ledger")
    problems += _check_balance(public["ledger"], "ledger")
    problems += _check_ledger_shape(private["pre_ledger"], pre_books, "pre")
    problems += _check_balance(private["pre_ledger"], "pre")
    problems += _check_clean_postings(private["pre_ledger"], pre_books, "pre", set())
    exempt = {doc for issue in issues if issue["type"] in POSTING_TYPES for doc in issue["affected_ids"]}
    problems += _check_clean_postings(public["ledger"], books, "ledger", exempt)
    problems += _check_replay(public, private)
    problems += _check_truth(public, private, pre_public)
    problems += _check_no_unlisted(public, issues)
    if private["truth"]["episode_id"] != public["manifest"]["episode_id"]:
        problems.append("truth: episode_id differs from the manifest")
    return problems
