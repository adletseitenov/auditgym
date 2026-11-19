"""Action parsing, fixed error codes and the task text (docs/CONTRACTS.md, sections 5.2-5.3).

Nothing here touches episode data; the evaluator owns the episode.

Lenient about the shape of a request, strict about its meaning (docs/AUDIT.md C1-C3, C9, R7):
- an action is a dict or a JSON object string; the name sits under action/tool/name/type, in
  any letter case and with surrounding spaces; arguments are flat or nested under
  args/arguments/params (a dict or a JSON object string; nested ones win);
- table names ignore letter case, a file extension and the singular (`Ledger`, `ledger.csv`,
  `invoice`);
- argument names have synonyms: get_record takes id/record_id/key or a field name
  (row_id, doc_id, document_id, invoice_id, payment_id; for fx also date + source_currency +
  target_currency), get_document takes doc_id/id/document_id, calculator expression/expr;
- a cursor of null, "", 0 or "0" is the first page; a limit may be "50" or 50.0;
- submit takes findings.json as an object, a JSON string, a bare list of findings (the running
  episode's id is filled in), flat in the action (episode_id and findings side by side) or under
  the key "findings.json".
An unknown action, table or id stays an error, and so does every other value.
"""
from __future__ import annotations

import json
import re

from . import findings as _findings

MAX_STEPS = 100
PAGE_SIZE = 50
# Limits of findings.json come from the judge's structure check, so the task text cannot drift.
MAX_FINDINGS = _findings.MAX_FINDINGS
MAX_EVIDENCE_IDS = _findings.MAX_EVIDENCE_IDS
MAX_EXPLANATION_CHARS = _findings.MAX_EXPLANATION_CODE_POINTS
MAX_SUBMIT_BYTES = _findings.MAX_MESSAGE_BYTES
MAX_EXPRESSION_CHARS = 200

ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")
TABLES = ("ledger", "invoices", "payments", "fx")
ERROR_TYPES = _findings.TYPES

# Fixed texts: an error never carries exception text, ids, paths or values.
ERRORS = {
    "unknown_action": "Unknown action.",
    "unknown_table": "Unknown table.",
    "unknown_id": "Unknown id.",
    "bad_cursor": "Bad cursor.",
    "bad_arguments": "Bad arguments.",
    "bad_expression": "Bad expression.",
    "episode_over": "Episode is over.",
    "not_reset": "Call reset first.",
}
# Errors that do not spend a step (there is no running episode to spend it in).
FREE_ERRORS = ("episode_over", "not_reset")

NAME_KEYS = ("action", "tool", "name", "type")
NESTED_KEYS = ("args", "arguments", "params")
_TABLE_NAMES = {"ledger": "ledger", "invoices": "invoices", "invoice": "invoices",
                "payments": "payments", "payment": "payments", "fx": "fx"}
_TABLE_EXTENSIONS = (".csv", ".jsonl", ".json")
# get_record: generic names first, then names that say which id field of the table is meant.
RECORD_ID_KEYS = ("id", "record_id", "key", "row_id", "doc_id", "document_id", "invoice_id", "payment_id")
_FIELD_OF_KEY = {"row_id": "row_id", "doc_id": "doc_id", "document_id": "doc_id",
                 "invoice_id": "invoice_id", "payment_id": "payment_id"}
# Id fields get_record searches, in this order: a document id first (CONTRACTS §3.1 names
# documents by doc_id), so a renamed episode where one record's doc_id equals another record's
# invoice_id still returns the record the doc_id names (AUDIT R10). fx has one composite key.
ID_FIELDS = {"ledger": ("row_id",), "invoices": ("doc_id", "invoice_id"),
             "payments": ("doc_id", "payment_id"), "fx": ("key",)}
FX_KEY_FIELDS = ("date", "source_currency", "target_currency")
DOCUMENT_ID_KEYS = ("doc_id", "id", "document_id")
EXPRESSION_KEYS = ("expression", "expr")
FINDINGS_KEYS = ("findings", "findings.json")
_LIMIT_TEXT = re.compile(r"[0-9]{1,9}")
# ASCII-only case folding, as for currencies in findings.py: Unicode folding could make a
# different string equal to a name.
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _fold(text: str) -> str:
    return text.strip().translate(_ASCII_LOWER)


class ActionError(Exception):
    """`code` is one of ERRORS, or a structure-error code for submit. `name` is the action name
    when it was recognized, so a broken submit can end the episode as an invalid structure."""

    def __init__(self, code: str, name: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.name = name


def error(code: str) -> dict:
    return {"code": code, "message": ERRORS[code]}


def _json_object(value: object) -> dict | None:
    """`value` itself if it is a dict, the decoded object if it is a JSON object string, else None."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def action_name(action: object) -> str | None:
    """The recognized action name of `action` (a dict or a JSON object string), else None.
    Never raises, whatever the object is: the client calls it on values it could not serialize."""
    try:
        action = _json_object(action)
        if action is None:
            return None
        for key in NAME_KEYS:
            if key in action:
                value = action[key]
                if not isinstance(value, str):
                    return None
                name = _fold(value)
                return name if name in ACTIONS else None
    except Exception:  # a hostile mapping (raising __contains__ or __getitem__) has no name
        return None
    return None


def parse_action(action: object) -> tuple[str, dict]:
    """Return (name, arguments). See the module docstring for the accepted shapes."""
    action = _json_object(action)
    name = action_name(action)
    if name is None:
        raise ActionError("unknown_action")
    arguments = {key: value for key, value in action.items()
                 if key not in NAME_KEYS and key not in NESTED_KEYS}
    for key in NESTED_KEYS:
        if key not in action or action[key] is None:
            continue
        nested = _json_object(action[key])
        if nested is None:
            raise ActionError("bad_arguments", name)
        arguments.update(nested)
    return name, arguments


def _present(arguments: dict, keys: tuple[str, ...]) -> list[tuple[str, object]]:
    return [(key, arguments[key]) for key in keys if arguments.get(key) is not None]


def _one_string(found: list[tuple[str, object]]) -> str:
    """The value named by synonyms: all present values must be the same string."""
    if not found or any(not isinstance(value, str) for _, value in found):
        raise ActionError("bad_arguments")
    values = {value for _, value in found}
    if len(values) != 1:
        raise ActionError("bad_arguments")
    return values.pop()


def table_name(arguments: dict) -> str:
    value = arguments.get("table")
    if isinstance(value, str):
        text = _fold(value)
        for extension in _TABLE_EXTENSIONS:
            if text.endswith(extension):
                text = text[:-len(extension)]
                break
        table = _TABLE_NAMES.get(text)
        if table is not None:
            return table
    raise ActionError("unknown_table")


def record_reference(arguments: dict, table: str) -> tuple[tuple[str, ...], str]:
    """(id fields to search in order, id) for get_record on `table`."""
    found = _present(arguments, RECORD_ID_KEYS)
    if not found and table == "fx":
        parts = [arguments.get(field) for field in FX_KEY_FIELDS]
        if all(isinstance(part, str) for part in parts):
            return ID_FIELDS["fx"], "|".join(parts)
    record_id = _one_string(found)
    fields = ID_FIELDS[table]
    named = {_FIELD_OF_KEY[key] for key, _ in found if key in _FIELD_OF_KEY} & set(fields)
    if named:  # e.g. invoice_id on invoices: search that field only
        fields = tuple(field for field in fields if field in named)
    return fields, record_id


def document_id(arguments: dict) -> str:
    return _one_string(_present(arguments, DOCUMENT_ID_KEYS))


def expression(arguments: dict) -> str:
    return _one_string(_present(arguments, EXPRESSION_KEYS))


def cursor(arguments: dict) -> str | None:
    """None for the first page (null, "", 0 or "0"), else the cursor token to look up."""
    value = arguments.get("cursor")
    if value is None or value == "" or value == "0" or (type(value) is int and value == 0):
        return None
    if isinstance(value, str):
        return value
    raise ActionError("bad_cursor")


def limit(arguments: dict) -> int:
    """Page size: a positive integer, also as "50" or 50.0; above PAGE_SIZE it is clipped."""
    value = arguments.get("limit")
    if value is None:
        return PAGE_SIZE
    if type(value) is int:
        number = value
    elif type(value) is float and value.is_integer():  # 50.0; NaN and infinity are not integers
        number = int(value)
    elif isinstance(value, str) and _LIMIT_TEXT.fullmatch(value.strip()):
        number = int(value.strip())
    else:
        raise ActionError("bad_arguments")
    if number < 1:
        raise ActionError("bad_arguments")
    return min(number, PAGE_SIZE)


def submit_payload(arguments: dict, episode_id: str) -> object:
    """The findings.json payload of a submit, from any accepted form. Only the wrapping is
    normalized: the judge still applies every structure rule of CONTRACTS §6 to the result, so a
    wrong episode_id, a missing field or a bad value still gives reward -1. A JSON string that
    does not decode raises ActionError("not_json")."""
    key = next((key for key in FINDINGS_KEYS if arguments.get(key) is not None), None)
    if key is None:
        # No findings (absent or null): an invalid structure the judge reports as a missing field.
        # Never None, which the judge reads as the empty answer of a truncated episode.
        return {}
    value = arguments[key]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            raise ActionError("not_json", "submit") from None
    if isinstance(value, list):
        named = arguments.get("episode_id")
        return {"episode_id": named if named is not None else episode_id, "findings": value}
    return {} if value is None else value  # "null" decoded: still not the empty answer


FINDINGS_SCHEMA = {
    "type": "object",
    "required": ["episode_id", "findings"],
    "properties": {
        "episode_id": {"type": "string", "description": "manifest.episode_id"},
        "findings": {
            "type": "array", "maxItems": MAX_FINDINGS,
            "items": {
                "type": "object",
                "required": ["type", "affected_ids", "claimed_amount_minor", "currency",
                             "evidence_ids", "explanation"],
                "properties": {
                    "type": {"enum": list(ERROR_TYPES)},
                    "affected_ids": {"type": "array", "items": {"type": "string"},
                                     "description": "doc_id of the affected documents"},
                    "claimed_amount_minor": {"type": "integer", "minimum": 0},
                    "currency": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"},
                                     "maxItems": MAX_EVIDENCE_IDS},
                    "explanation": {"type": "string", "maxLength": MAX_EXPLANATION_CHARS},
                },
            },
        },
    },
}

TASK_TEXT = (
    "Проверьте книгу компании за отчётный период и найдите ошибки четырёх типов: "
    "duplicate_invoice, allocation_gap, fx_error, period_cutoff. Ошибок может не быть. "
    "Записи только для чтения. Действия: list_records(table, cursor, limit), "
    "get_record(table, id), get_document(doc_id), calculator(expression), submit(findings). "
    "Таблицы: ledger, invoices, payments, fx; ID курса — \"дата|из|в\". "
    "Страница — до 50 записей, next_cursor = null на последней. Каждое действие, в том числе "
    "с ошибкой, расходует шаг; submit завершает эпизод. "
    f"Бюджет — {MAX_STEPS} шагов, и submit входит в него: последним шагом должен быть submit, "
    f"не позже {MAX_STEPS}-го. Если {MAX_STEPS} шагов прошли без submit, эпизод обрезается "
    "и ответ считается пустым. Ответ — findings по схеме, "
    "episode_id из manifest; ложные выводы снижают награду."
)


def build_task(policy: dict) -> dict:
    """The task block of the first observation. Built only from constants and the open policy."""
    tolerances = policy.get("amount_tolerance_minor")
    return {
        "text": TASK_TEXT,
        "error_types": list(ERROR_TYPES),
        "findings_schema": FINDINGS_SCHEMA,
        "limits": {
            "max_findings": MAX_FINDINGS,
            "max_evidence_ids": MAX_EVIDENCE_IDS,
            "max_explanation_chars": MAX_EXPLANATION_CHARS,
            "max_steps": MAX_STEPS,
            "page_size": PAGE_SIZE,
            "max_expression_chars": MAX_EXPRESSION_CHARS,
            "max_submit_bytes": MAX_SUBMIT_BYTES,
        },
        "tolerances": {"amount_tolerance_minor": dict(tolerances) if isinstance(tolerances, dict) else {},
                       "evidence_superset_allowed": policy.get("evidence_superset_allowed", True)},
        "accounts": dict(policy.get("accounts") or {}),
    }
