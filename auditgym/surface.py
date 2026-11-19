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


def _link(key, id_map: dict[str, str]):
    return None if key is None else id_map[key]


def public_row(row: dict, id_map: dict[str, str]) -> dict:
    return {"row_id": id_map[row["key"]], "timestamp": _iso(row["timestamp"]), "account": row["account"],
            "debit_minor": row["debit_minor"], "credit_minor": row["credit_minor"],
            "currency": row["currency"], "entity_id": _link(row["entity"], id_map),
            "invoice_id": _link(row["invoice"], id_map), "payment_id": _link(row["payment"], id_map)}


def public_invoice(inv: dict, id_map: dict[str, str]) -> dict:
    return {"doc_id": id_map["D:" + inv["key"]], "invoice_id": id_map[inv["key"]],
            "entity_id": id_map[inv["entity"]], "issued_at": _iso(inv["issued_at"]),
            "due_at": _iso(inv["due_at"]), "currency": inv["currency"], "total_minor": inv["total_minor"],
            "line_items": [dict(item) for item in inv["line_items"]]}


def public_payment(pay: dict, id_map: dict[str, str]) -> dict:
    return {"doc_id": id_map["D:" + pay["key"]], "payment_id": id_map[pay["key"]],
            "timestamp": _iso(pay["timestamp"]), "currency": pay["currency"], "amount_minor": pay["amount_minor"],
            "invoice_refs": None if pay["invoice_refs"] is None
            else sorted(id_map[ref] for ref in pay["invoice_refs"])}


def public_record(record: dict, id_map: dict[str, str]) -> dict:
    """An internal ledger row, invoice or payment in public ids (used for provenance records)."""
    if "account" in record:
        return public_row(record, id_map)
    if "line_items" in record:
        return public_invoice(record, id_map)
    return public_payment(record, id_map)


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

    # Sorted, then shuffled by the surface stream: the mapping does not depend on the order in which
    # the story was built, and the draw position of a token is unrelated to the key's name (injected
    # records get the highest key numbers, which must not correlate with anything public).
    keys = set(world["entities"])
    for invoice in world["invoices"]:
        keys |= {invoice["key"], "D:" + invoice["key"]}
    for payment in world["payments"]:
        keys |= {payment["key"], "D:" + payment["key"]}
    keys |= {row["key"] for row in ledger} | {row["key"] for row in pre_ledger}
    order = sorted(keys)
    rng.shuffle(order)
    for key in order:
        token(key)

    def ordered(rows: list[dict]) -> list[dict]:
        return sorted((public_row(row, id_map) for row in rows), key=lambda r: (r["timestamp"], r["row_id"]))

    # Order is by document date, then by the random id: position says nothing about injection.
    invoices = sorted((public_invoice(inv, id_map) for inv in world["invoices"]),
                      key=lambda r: (r["issued_at"], r["doc_id"]))
    payments = sorted((public_payment(pay, id_map) for pay in world["payments"]),
                      key=lambda r: (r["timestamp"], r["doc_id"]))
    fx = sorted((dict(row) for row in world["fx"]), key=lambda r: (r["date"], r["source_currency"]))
    return {"ledger": ordered(ledger), "pre_ledger": ordered(pre_ledger), "invoices": invoices,
            "payments": payments, "fx": fx, "id_map": dict(sorted(id_map.items()))}
