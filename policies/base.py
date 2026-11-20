"""Shared pieces of the baseline policies: the agent-facing contract only (docs/CONTRACTS.md §5).

A policy is an object with `reset(observation)` and `act(observation) -> action`. It sees the
observations the environment returns and nothing else: no generator, no truth, no files.
"""
from __future__ import annotations

TABLES = ("ledger", "invoices", "payments", "fx")
PAGE = 50
MAX_FINDINGS = 30
MAX_EVIDENCE = 10


def submit(episode_id: str, findings: list[dict]) -> dict:
    return {"action": "submit", "findings": {"episode_id": episode_id, "findings": findings[:MAX_FINDINGS]}}


def finding(kind: str, affected: list[str], amount: int, currency: str, evidence: list[str],
            explanation: str) -> dict:
    """One finding in the §6 shape; evidence keeps its order, drops repeats, at most 10 ids."""
    cited: list[str] = []
    for item in evidence:
        if item not in cited:
            cited.append(item)
    return {"type": kind, "affected_ids": sorted(set(affected)), "claimed_amount_minor": int(abs(amount)),
            "currency": currency, "evidence_ids": cited[:MAX_EVIDENCE], "explanation": explanation[:1000]}


def _text(value) -> str:
    return "" if value is None else str(value)


def _base_key(table: str, record: dict) -> tuple:
    """The content of a record without any id: what it says, not what it is called."""
    if table == "ledger":
        return (_text(record.get("timestamp")), _text(record.get("account")), record.get("debit_minor") or 0,
                record.get("credit_minor") or 0, _text(record.get("currency")))
    if table == "invoices":
        items = tuple(sorted(tuple(sorted((k, _text(v)) for k, v in item.items()))
                             for item in record.get("line_items") or [] if isinstance(item, dict)))
        return (_text(record.get("issued_at")), _text(record.get("due_at")), _text(record.get("currency")),
                record.get("total_minor") or 0, items)
    if table == "payments":
        return (_text(record.get("timestamp")), _text(record.get("currency")), record.get("amount_minor") or 0)
    return (_text(record.get("date")), _text(record.get("source_currency")), _text(record.get("target_currency")),
            _text(record.get("rate")))


def canonical_public(raw: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """What a policy decides on: every table sorted by the content of its records, never by arrival
    order and never by the text of an id, so that reordering rows and renaming ids changes nothing.

    Ties are broken by the content around a record: an invoice by its posting rows and by what
    its customer was billed in all, a payment by the invoices it pays, a posting row by the
    document it belongs to. Reference lists inside a record are sorted by the content they point to.
    Records that stay equal after that are interchangeable, whatever their ids are."""
    ledger = [dict(row) for row in raw.get("ledger", [])]
    invoices = [dict(row) for row in raw.get("invoices", [])]
    payments = [dict(row) for row in raw.get("payments", [])]
    rows_of: dict[tuple, list[tuple]] = {}
    for row in ledger:
        owner = ("p", row["payment_id"]) if row.get("payment_id") else ("i", row.get("invoice_id"))
        rows_of.setdefault(owner, []).append(_base_key("ledger", row))
    by_entity: dict = {}
    for invoice in invoices:
        by_entity.setdefault(invoice.get("entity_id"), []).append(_base_key("invoices", invoice))
    invoice_key = {}
    for invoice in invoices:
        invoice_key[invoice.get("invoice_id")] = (
            _base_key("invoices", invoice), tuple(sorted(rows_of.get(("i", invoice.get("invoice_id")), []))),
            tuple(sorted(by_entity[invoice.get("entity_id")])))
    # a payment refers to invoices by doc_id or invoice_id: both name the same invoice
    ref_key = {}
    for invoice in invoices:
        ref_key[invoice.get("invoice_id")] = ref_key[invoice.get("doc_id")] = invoice_key[invoice.get("invoice_id")]
    payment_key = {}
    for payment in payments:
        refs = sorted(payment.get("invoice_refs") or [], key=lambda ref: (ref not in ref_key, ref_key.get(ref), ref))
        if isinstance(payment.get("invoice_refs"), list):
            payment["invoice_refs"] = refs
        payment_key[payment.get("payment_id")] = (
            _base_key("payments", payment), tuple(ref_key.get(ref, ()) for ref in refs),
            tuple(sorted(rows_of.get(("p", payment.get("payment_id")), []))))
    owner_key = {("i", k): v for k, v in invoice_key.items()} | {("p", k): v for k, v in payment_key.items()}

    def row_key(row: dict) -> tuple:
        owner = ("p", row["payment_id"]) if row.get("payment_id") else ("i", row.get("invoice_id"))
        return _base_key("ledger", row), owner_key.get(owner, ())

    return {"ledger": sorted(ledger, key=row_key),
            "invoices": sorted(invoices, key=lambda r: invoice_key[r.get("invoice_id")]),
            "payments": sorted(payments, key=lambda r: payment_key[r.get("payment_id")]),
            "fx": sorted((dict(row) for row in raw.get("fx", [])), key=lambda r: _base_key("fx", r))}


class Reader:
    """Pages through tables with list_records. `max_pages` limits pages per table (None: all)."""

    def __init__(self, tables, max_pages: dict | None = None) -> None:
        self.queue = list(tables)
        self.records: dict[str, list[dict]] = {table: [] for table in self.queue}
        self.max_pages = dict(max_pages or {})
        self.pages: dict[str, int] = {table: 0 for table in self.queue}
        self.cursor = None
        self.pending: str | None = None

    def next_action(self, observation: dict) -> dict | None:
        """Consume the reply to the previous page request and return the next one, or None."""
        if self.pending is not None:
            table, self.pending = self.pending, None
            if observation.get("type") == "records" and observation.get("table") == table:
                self.records[table] += observation.get("records") or []
                self.pages[table] += 1
                self.cursor = observation.get("next_cursor")
                limit = self.max_pages.get(table)
                if self.cursor is None or (limit is not None and self.pages[table] >= limit):
                    self._next_table()
            else:  # an error page: give up on this table rather than loop
                self._next_table()
        if not self.queue:
            return None
        self.pending = self.queue[0]
        return {"action": "list_records", "table": self.queue[0], "cursor": self.cursor, "limit": PAGE}

    def _next_table(self) -> None:
        self.queue.pop(0)
        self.cursor = None


class ReadThenSubmit:
    """Read the listed tables (within the step budget), then submit `decide(public)`."""

    name = "read_then_submit"
    tables: tuple = TABLES
    max_pages: dict | None = None

    def reset(self, observation: dict) -> None:
        self.manifest = observation.get("manifest") or {}
        self.policy = observation.get("policy") or {}
        self.episode_id = self.manifest.get("episode_id", "")
        self.reader = Reader(self.tables, self.max_pages)
        self.done = False

    def act(self, observation: dict) -> dict:
        action = None
        if not self.done and observation.get("steps_remaining", 0) > 1:
            action = self.reader.next_action(observation)
        if action is not None:
            return action
        self.done = True
        public = {"manifest": self.manifest, "policy": self.policy}
        public.update(canonical_public(self.reader.records))
        return submit(self.episode_id, self.decide(public))

    def decide(self, public: dict) -> list[dict]:
        return []


def postings(ledger: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Ledger rows grouped as in CONTRACTS §2.3: by payment_id, else by invoice_id."""
    by_invoice: dict[str, list[dict]] = {}
    by_payment: dict[str, list[dict]] = {}
    for row in ledger:
        if row.get("payment_id"):
            by_payment.setdefault(row["payment_id"], []).append(row)
        elif row.get("invoice_id"):
            by_invoice.setdefault(row["invoice_id"], []).append(row)
    return by_invoice, by_payment
