"""Public ids, and record order. Runs after error injection.

Ids are opaque tokens drawn from a stream that depends only on the surface seed and the set of
internal keys, so an id says nothing about when or why a record was created. A different
surface seed gives the same episode under a renaming.
"""
from __future__ import annotations

from .rng import Rng

TOKEN_LENGTH = 10


def _iso(moment) -> str:
    return moment.isoformat()


def apply_surface(rng: Rng, world: dict, ledger: list[dict], pre_ledger: list[dict]) -> dict:
    taken: set[str] = set()
    id_map: dict[str, str] = {}

    def token(key: str) -> str:
        if key not in id_map:
            while True:
                candidate = rng.token(TOKEN_LENGTH)
                if candidate not in taken:
                    taken.add(candidate)
                    id_map[key] = candidate
                    break
        return id_map[key]

    # Sorted keys: the mapping does not depend on the order in which the story was built.
    keys = set(world["entities"])
    for invoice in world["invoices"]:
        keys |= {invoice["key"], "D:" + invoice["key"]}
    for payment in world["payments"]:
        keys |= {payment["key"], "D:" + payment["key"]}
    keys |= {row["key"] for row in ledger} | {row["key"] for row in pre_ledger}
    for key in sorted(keys):
        token(key)

    def link(key):
        return None if key is None else id_map[key]

    def public_row(row: dict) -> dict:
        return {"row_id": id_map[row["key"]], "timestamp": _iso(row["timestamp"]), "account": row["account"],
                "debit_minor": row["debit_minor"], "credit_minor": row["credit_minor"],
                "currency": row["currency"], "entity_id": link(row["entity"]),
                "invoice_id": link(row["invoice"]), "payment_id": link(row["payment"])}

    def ordered(rows: list[dict]) -> list[dict]:
        return sorted((public_row(row) for row in rows), key=lambda r: (r["timestamp"], r["row_id"]))

    invoices = sorted(({"doc_id": id_map["D:" + inv["key"]], "invoice_id": id_map[inv["key"]],
                        "entity_id": id_map[inv["entity"]], "issued_at": _iso(inv["issued_at"]),
                        "due_at": _iso(inv["due_at"]), "currency": inv["currency"],
                        "total_minor": inv["total_minor"],
                        "line_items": [dict(item) for item in inv["line_items"]]}
                       for inv in world["invoices"]), key=lambda r: (r["issued_at"], r["doc_id"]))
    payments = sorted(({"doc_id": id_map["D:" + pay["key"]], "payment_id": id_map[pay["key"]],
                        "timestamp": _iso(pay["timestamp"]), "currency": pay["currency"],
                        "amount_minor": pay["amount_minor"],
                        "invoice_refs": None if pay["invoice_refs"] is None
                        else sorted(id_map[ref] for ref in pay["invoice_refs"])}
                       for pay in world["payments"]), key=lambda r: (r["timestamp"], r["doc_id"]))
    fx = sorted((dict(row) for row in world["fx"]), key=lambda r: (r["date"], r["source_currency"]))
    return {"ledger": ordered(ledger), "pre_ledger": ordered(pre_ledger), "invoices": invoices,
            "payments": payments, "fx": fx, "id_map": dict(sorted(id_map.items()))}
