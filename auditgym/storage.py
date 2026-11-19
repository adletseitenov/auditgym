"""Reading and writing an episode directory (docs/CONTRACTS.md, section 1)."""
from __future__ import annotations

import csv
import io
import json
import pathlib

LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                  "entity_id", "invoice_id", "payment_id"]
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]
_NULLABLE = ("entity_id", "invoice_id", "payment_id")
_INTEGER = ("debit_minor", "credit_minor")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1) + "\n"


def _jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _csv(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: "" if row[column] is None else row[column] for column in columns})
    return buffer.getvalue()


def _read_ledger(text: str) -> list[dict]:
    rows = []
    for raw in csv.DictReader(io.StringIO(text)):
        row = dict(raw)
        for column in _INTEGER:
            row[column] = int(row[column])
        for column in _NULLABLE:
            row[column] = row[column] or None
        rows.append(row)
    return rows


def episode_files(episode: dict) -> dict[str, str]:
    """Relative path -> file content. The same episode always gives the same bytes."""
    public, private = episode["public"], episode["private"]
    return {
        "public/manifest.json": _json(public["manifest"]),
        "public/policy.json": _json(public["policy"]),
        "public/ledger.csv": _csv(public["ledger"], LEDGER_COLUMNS),
        "public/invoices.jsonl": _jsonl(public["invoices"]),
        "public/payments.jsonl": _jsonl(public["payments"]),
        "public/fx.csv": _csv(public["fx"], FX_COLUMNS),
        "private/truth.json": _json(private["truth"]),
        "private/provenance.json": _json(private["provenance"]),
        "private/pre_ledger.csv": _csv(private["pre_ledger"], LEDGER_COLUMNS),
        "private/pre_invoices.jsonl": _jsonl(private["pre_invoices"]),
        "private/pre_payments.jsonl": _jsonl(private["pre_payments"]),
    }


def write_episode(episode: dict, directory: str | pathlib.Path) -> None:
    root = pathlib.Path(directory)
    for relative, content in episode_files(episode).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")


def load_episode(directory: str | pathlib.Path) -> dict:
    """Read an episode directory. A UTF-8 byte order mark is tolerated (files re-saved by a
    spreadsheet or an editor). The evaluator never reads `provenance.json` and `pre_*`, so a
    directory without them loads: `provenance` is then {} and the `pre_*` lists are empty."""
    root = pathlib.Path(directory)

    def text(relative: str) -> str:
        return (root / relative).read_text(encoding="utf-8-sig")

    def optional(relative: str) -> str | None:
        return text(relative) if (root / relative).is_file() else None

    def lines(content: str | None) -> list[dict]:
        return [json.loads(line) for line in (content or "").splitlines() if line.strip()]

    pre_ledger = optional("private/pre_ledger.csv")
    provenance = optional("private/provenance.json")
    return {
        "public": {
            "manifest": json.loads(text("public/manifest.json")),
            "policy": json.loads(text("public/policy.json")),
            "ledger": _read_ledger(text("public/ledger.csv")),
            "invoices": lines(text("public/invoices.jsonl")),
            "payments": lines(text("public/payments.jsonl")),
            "fx": [dict(row) for row in csv.DictReader(io.StringIO(text("public/fx.csv")))],
        },
        "private": {
            "truth": json.loads(text("private/truth.json")),
            "provenance": json.loads(provenance) if provenance is not None else {},
            "pre_ledger": _read_ledger(pre_ledger) if pre_ledger is not None else [],
            "pre_invoices": lines(optional("private/pre_invoices.jsonl")),
            "pre_payments": lines(optional("private/pre_payments.jsonl")),
        },
    }
