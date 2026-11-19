"""Action parsing, fixed error codes and the task text (docs/CONTRACTS.md, sections 5.2-5.3).

Nothing here touches episode data; the evaluator owns the episode.
"""
from __future__ import annotations

import json

MAX_STEPS = 100
PAGE_SIZE = 50
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CHARS = 1000
MAX_SUBMIT_BYTES = 65536
MAX_EXPRESSION_CHARS = 200

ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")
TABLES = ("ledger", "invoices", "payments", "fx")
ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")

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

_NAME_KEYS = ("action", "tool", "name", "type")
_NESTED_KEYS = ("args", "arguments", "params")


class ActionError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def error(code: str) -> dict:
    return {"code": code, "message": ERRORS[code]}


def parse_action(action: object) -> tuple[str, dict]:
    """Return (name, arguments). Accepts the name under action/tool/name/type and the
    arguments flat or nested under args/arguments/params (a dict or a JSON object string).
    Nested arguments win over flat ones with the same key."""
    if not isinstance(action, dict):
        raise ActionError("unknown_action")
    name = None
    for key in _NAME_KEYS:
        if key in action:
            name = action[key]
            break
    if not isinstance(name, str) or name not in ACTIONS:
        raise ActionError("unknown_action")
    arguments = {key: value for key, value in action.items()
                 if key not in _NAME_KEYS and key not in _NESTED_KEYS}
    for key in _NESTED_KEYS:
        if key not in action or action[key] is None:
            continue
        nested = action[key]
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except ValueError:
                raise ActionError("bad_arguments") from None
        if not isinstance(nested, dict):
            raise ActionError("bad_arguments")
        arguments.update(nested)
    return name, arguments


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
