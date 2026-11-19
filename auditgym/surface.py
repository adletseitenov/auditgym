"""Public ids, and record order. Runs after error injection.

Ids are opaque tokens drawn from a stream that depends only on the surface seed, the config and
the set of internal keys, so an id says nothing about when or why a record was created. A
different surface seed gives the same episode under a renaming.
"""
from __future__ import annotations

from .rng import Rng

TOKEN_LENGTH = 10


def _link(key, id_map: dict[str, str]):
    return None if key is None else id_map[key]


def public_row(row: dict, id_map: dict[str, str]) -> dict:
    return {"row_id": id_map[row["key"]], "timestamp": row["timestamp"].isoformat(), "account": row["account"],
            "debit_minor": row["debit_minor"], "credit_minor": row["credit_minor"],
            "currency": row["currency"], "entity_id": _link(row["entity"], id_map),
            "invoice_id": _link(row["invoice"], id_map), "payment_id": _link(row["payment"], id_map)}


def public_invoice(inv: dict, id_map: dict[str, str]) -> dict:
    return {"doc_id": id_map["D:" + inv["key"]], "invoice_id": id_map[inv["key"]],
            "entity_id": id_map[inv["entity"]], "issued_at": inv["issued_at"].isoformat(),
            "due_at": inv["due_at"].isoformat(), "currency": inv["currency"], "total_minor": inv["total_minor"],
            "line_items": [dict(item) for item in inv["line_items"]]}


def public_payment(pay: dict, id_map: dict[str, str]) -> dict:
    return {"doc_id": id_map["D:" + pay["key"]], "payment_id": id_map[pay["key"]],
            "timestamp": pay["timestamp"].isoformat(), "currency": pay["currency"], "amount_minor": pay["amount_minor"],
            "invoice_refs": None if pay["invoice_refs"] is None
            else sorted(id_map[ref] for ref in pay["invoice_refs"])}


def public_record(record: dict, id_map: dict[str, str]) -> dict:
    """An internal ledger row, invoice or payment in public ids (used for provenance records)."""
    if "account" in record:
        return public_row(record, id_map)
    if "line_items" in record:
        return public_invoice(record, id_map)
    return public_payment(record, id_map)


def assign_ids(rng: Rng, keys: set[str]) -> dict[str, str]:
    """Sorted, then shuffled by the surface stream: the mapping does not depend on the order in which
    the story was built, and the draw position of a token is unrelated to the key's name."""
    order = sorted(keys)
    rng.shuffle(order)
    id_map: dict[str, str] = {}
    taken: set[str] = set()
    for key in order:
        while True:
            candidate = rng.token(TOKEN_LENGTH)
            if candidate not in taken:
                taken.add(candidate)
                id_map[key] = candidate
                break
    return dict(sorted(id_map.items()))


def public_tables(world: dict, ledger: list[dict], id_map: dict[str, str]) -> dict:
    """Open tables, each ordered by its public id (docs/AUDIT.md R6): rows by row_id, documents by
    doc_id, rates by (date, currency). Ids are random, so a position says nothing about a record's
    date, its story or an injection."""
    invoices = [public_invoice(r, id_map) for r in world["invoices"]]
    payments = [public_payment(r, id_map) for r in world["payments"]]
    rows = [public_row(r, id_map) for r in ledger]
    fx = sorted((dict(row) for row in world["fx"]), key=lambda r: (r["date"], r["source_currency"]))
    return {"ledger": sorted(rows, key=lambda r: r["row_id"]),
            "invoices": sorted(invoices, key=lambda r: r["doc_id"]),
            "payments": sorted(payments, key=lambda r: r["doc_id"]), "fx": fx}
