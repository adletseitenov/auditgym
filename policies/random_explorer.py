"""Random explorer with a fixed seed: random valid actions for a budget of steps, then either an
empty answer or a few random guesses built from ids and amounts it has seen.

The stream for an episode is derived from the policy seed and the episode_id, so the same
seed gives the same actions on the same episode whatever the order of episodes.
"""
from __future__ import annotations

from auditgym.rng import Rng

from .base import TABLES, finding, submit

TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")


class RandomExplorer:
    name = "random_explorer"

    def __init__(self, seed: int = 0, budget: int = 10, max_guesses: int = 3, guess: bool | None = None) -> None:
        """`guess`: True always guesses, False always submits nothing, None flips a coin."""
        self.seed, self.budget, self.max_guesses, self.guess = seed, budget, max_guesses, guess

    def reset(self, observation: dict) -> None:
        self.episode_id = (observation.get("manifest") or {}).get("episode_id", "")
        policy = observation.get("policy") or {}
        self.currencies = sorted(policy.get("currency_exponents") or {}) or ["KZT"]
        self.rng = Rng(self.seed, f"random_explorer|{self.episode_id}")
        self.steps = 0
        self.cursors: dict[str, str | None] = {}
        self.ids: dict[str, list[str]] = {table: [] for table in TABLES}
        self.docs: list[str] = []
        self.amounts: list[int] = []

    def _see(self, observation: dict) -> None:
        records = observation.get("records") or ([observation["record"]] if observation.get("record") else [])
        table = observation.get("table")
        if observation.get("type") == "records" and table:
            self.cursors[table] = observation.get("next_cursor")
        for record in records:
            if not isinstance(record, dict):
                continue
            key = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id"}.get(table)
            if key and record.get(key) and record[key] not in self.ids[table]:
                self.ids[table].append(record[key])
            if table == "fx" and record.get("date"):
                self.ids["fx"].append(f"{record['date']}|{record['source_currency']}|{record['target_currency']}")
            if record.get("doc_id") and record["doc_id"] not in self.docs:
                self.docs.append(record["doc_id"])
            for field in ("total_minor", "amount_minor", "debit_minor", "credit_minor"):
                if isinstance(record.get(field), int) and record[field] > 0:
                    self.amounts.append(record[field])

    def _token(self) -> str:
        return self.rng.token(10)

    def act(self, observation: dict) -> dict:
        self._see(observation)
        if self.steps >= self.budget or observation.get("steps_remaining", 0) <= 1:
            return self._answer()
        self.steps += 1
        kind = self.rng.choice(("list_records", "list_records", "get_record", "get_document", "calculator"))
        table = self.rng.choice(TABLES)
        if kind == "list_records":
            return {"action": "list_records", "table": table, "cursor": self.cursors.get(table),
                    "limit": self.rng.randint(1, 50)}
        if kind == "get_record":
            known = self.ids[table]
            record_id = self.rng.choice(known) if known and self.rng.chance(3, 4) else self._token()
            return {"action": "get_record", "table": table, "id": record_id}
        if kind == "get_document":
            doc_id = self.rng.choice(self.docs) if self.docs and self.rng.chance(3, 4) else self._token()
            return {"action": "get_document", "doc_id": doc_id}
        left, right = self.rng.randint(1, 10**6), self.rng.randint(1, 10**4)
        return {"action": "calculator", "expression": f"{left} {self.rng.choice('+-*/')} {right}"}

    def _answer(self) -> dict:
        guess = self.rng.chance(1, 2) if self.guess is None else self.guess
        if not guess:
            return submit(self.episode_id, [])
        guesses = []
        for _ in range(self.rng.randint(1, self.max_guesses)):
            pool = self.docs or [self._token()]
            affected = [self.rng.choice(pool) for _ in range(self.rng.randint(1, 2))]
            amount = self.rng.choice(self.amounts) if self.amounts else self.rng.randint(0, 10**7)
            guesses.append(finding(self.rng.choice(TYPES), affected, amount, self.rng.choice(self.currencies),
                                   affected, "Random guess."))
        return submit(self.episode_id, guesses)
