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


SORT_KEYS = {"ledger": ("row_id",), "invoices": ("invoice_id", "doc_id"), "payments": ("payment_id", "doc_id"),
             "fx": ("date", "source_currency", "target_currency")}


def canonical(table: str, records: list[dict]) -> list[dict]:
    """What a policy decides on: records sorted by a stable key (never by arrival order), so that
    a reordered table gives the same answer. Reference lists inside a record are sorted too."""
    keys = SORT_KEYS[table]
    out = []
    for record in records:
        record = dict(record)
        if isinstance(record.get("invoice_refs"), list):
            record["invoice_refs"] = sorted(record["invoice_refs"])
        out.append(record)
    return sorted(out, key=lambda record: tuple(str(record.get(key) or "") for key in keys))


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
        for table in TABLES:
            public[table] = canonical(table, self.reader.records.get(table, []))
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
