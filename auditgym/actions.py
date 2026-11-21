"""Action parsing, fixed error codes and the task text (docs/CONTRACTS.md, sections 5.2-5.3).

Nothing here touches episode data; the evaluator owns the episode.
"""
from __future__ import annotations

import json
import os
import re
import stat

from .answer_rules import answer_rules
from .limits import (MAX_EVIDENCE_EXTRA, MAX_EVIDENCE_IDS, MAX_EXPLANATION_CHARS, MAX_FINDINGS, MAX_MESSAGE_BYTES,  # noqa: F401
                     MAX_STEPS, PAGE_SIZE)

MAX_SUBMIT_BYTES = MAX_MESSAGE_BYTES  # 256 KiB: one constant for the limit (audit R2)
MAX_EXPRESSION_CHARS = 200
# Where the evaluator writes its journal in eval mode when no path is given (audit C5): the path in
# $AUDITGYM_RESULTS, else DEFAULT_RESULTS_PATH relative to the working directory of whoever starts it.
RESULTS_ENV = "AUDITGYM_RESULTS"
DEFAULT_RESULTS_PATH = "runs/results.jsonl"


def default_results_path() -> str:
    """The journal path of an eval run without an explicit one (one rule for client and server)."""
    return os.environ.get(RESULTS_ENV) or DEFAULT_RESULTS_PATH


_RESULTS_FLAGS = (os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                  | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))


def open_results(path) -> int:
    """An append descriptor of the journal at `path` (client and evaluator use it), also readable:
    a signed journal continues the chain from its last line (auditgym/journal.py).

    The journal holds rewards, so the file is owner-only (0600; an existing one is narrowed,
    audit L3). Against a same-user policy planting things at a guessable path: a symlink is
    refused (no appending to someone else's file), and so is anything but a regular file (opening
    a FIFO would hang). The descriptor is never 0-2: with a closed stdout it would otherwise be
    passed to the server as --results-fd 1, its protocol channel. None of this hides the journal
    from a policy running as the same user: in a closed run the evaluator and its journal belong
    to another user or container."""
    descriptor = os.open(path, _RESULTS_FLAGS, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("the journal must be a regular file")
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)  # the results are not for the policy's user group
            os.set_blocking(descriptor, True)
            if descriptor < 3:
                import fcntl

                moved = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
                os.close(descriptor)
                descriptor = moved
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


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
# Argument containers, in increasing precedence: flat < args < arguments < params < input < parameters.
# (`input` is where a tool_use block of an LLM API puts its arguments.)
_NESTED_KEYS = ("args", "arguments", "params", "input", "parameters")

# Argument synonyms (audit C3). The first key present with a value that is not null wins.
GET_RECORD_ID_KEYS = ("id", "record_id", "row_id", "doc_id", "invoice_id", "payment_id")
GET_DOCUMENT_ID_KEYS = ("doc_id", "id", "document_id")
SUBMIT_FINDINGS_KEYS = ("findings", "findings.json", "findings_json")

# An action named submit in JSON text, also inside a JSON string (escaped quotes).
_SUBMIT_TEXT = re.compile(r'\\?"(?:action|tool|name|type)\\?"\s*:\s*\\?"\s*submit\s*\\?"', re.IGNORECASE)

_TABLE_ALIASES = {"ledger": "ledger", "invoice": "invoices", "invoices": "invoices",
                  "payment": "payments", "payments": "payments", "fx": "fx"}

# Keys of the open policy a policy (the agent) may see: CONTRACTS section 2.2, plus the documented
# additions below. Anything else the generator puts into policy.json stays on the evaluator side,
# so that a key such as an error count cannot reach the agent by accident (audit Q2).
POLICY_KEYS = ("ledger_currency", "base_currency", "timezone", "period_start", "period_end",
               "fx_rate_date", "fx_formula", "rounding", "currency_exponents", "accounts",
               "document_date", "refund", "amount_tolerance_minor", "evidence_superset_allowed",
               "period_basis")
# Documented additions: rules the open policy states for the agent (docs/AUDIT.md D6). A new rule
# key of the generator must be added here, or the agent never sees it: the test
# tests/env/test_evaluator.py::PolicyWhitelistTest fails until it is.
POLICY_KEYS_ADDED = ("period_interval", "payment_rule", "duplicate_signature")


class ActionError(Exception):
    def __init__(self, code: str, action: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.action = action  # the action name when it was recognised before the error


def error(code: str) -> dict:
    return {"code": code, "message": ERRORS[code]}


def _json_value(value: object, rounds: int = 1) -> object:
    """A JSON text is parsed (up to `rounds` times); anything else, or text that is not JSON, is
    returned unchanged. Deep nesting never raises."""
    for _ in range(rounds):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            break
    return value


def parse_action(action: object) -> tuple[str, dict]:
    """Return (name, arguments).

    Accepts the action as a dict or as a JSON object string; the name under action/tool/name/type
    in any letter case; the arguments flat or nested under args/arguments/params/input/parameters
    (a dict or a JSON object string). Nested arguments win over flat ones with the same key, the
    containers in the order of _NESTED_KEYS. Synonyms of an id argument are filled in (C3).
    An unknown name stays `unknown_action`; a malformed container is `bad_arguments`, and the
    error carries the action name so that a malformed submit can end the episode (C9)."""
    text = action if isinstance(action, str) else None
    action = _json_value(action)
    if not isinstance(action, dict):
        if text is not None and _SUBMIT_TEXT.search(text):
            # JSON text that names a submit but does not decode (cut short, nested too deep):
            # a submit whose arguments cannot be read, so the episode ends (audit C9, R1)
            raise ActionError("bad_arguments", "submit")
        raise ActionError("unknown_action")
    name = None
    for key in _NAME_KEYS:
        value = action.get(key)
        if isinstance(value, str) and value.strip().lower() in ACTIONS:
            name = value.strip().lower()
            break
    if name is None:
        raise ActionError("unknown_action")
    arguments = {key: value for key, value in action.items()
                 if key not in _NAME_KEYS and key not in _NESTED_KEYS}
    for key in _NESTED_KEYS:
        if key not in action or action[key] is None:
            continue
        nested = _json_value(action[key])
        if not isinstance(nested, dict):
            raise ActionError("bad_arguments", name)
        arguments.update(nested)
    if name == "get_record":
        _fill(arguments, "id", GET_RECORD_ID_KEYS)
    elif name == "get_document":
        _fill(arguments, "doc_id", GET_DOCUMENT_ID_KEYS)
    return name, arguments


def _fill(arguments: dict, canonical: str, synonyms: tuple[str, ...]) -> None:
    if arguments.get(canonical) is not None:
        return
    for key in synonyms:
        if arguments.get(key) is not None:
            arguments[canonical] = arguments[key]
            return


def table_name(value: object) -> str | None:
    """Canonical table name, or None: ignores case and surrounding spaces, drops `.csv`/`.jsonl`,
    accepts the singular (`invoice`)."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    for suffix in (".jsonl", ".csv"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return _TABLE_ALIASES.get(text)


def parse_limit(value: object) -> int:
    """`limit` as an int in [1, PAGE_SIZE]: null is the default, `"50"` and `50.0` are 50, larger
    values are clipped; anything else (0, negative, true, 1.5, "ten") is `bad_arguments`."""
    return parse_limit_clamped(value)[0]


def parse_limit_clamped(value: object) -> tuple[int, bool]:
    """(limit, clamped): like `parse_limit`, and whether a valid limit above PAGE_SIZE was cut to it.
    The list_records answer says so with `"limit_clamped": true`."""
    if value is None:
        return PAGE_SIZE, False
    if isinstance(value, bool):
        raise ActionError("bad_arguments")
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[0-9]+(\.0+)?", text, re.ASCII):
            raise ActionError("bad_arguments")
        value = int(text.split(".")[0]) if len(text) < 4000 else 10 ** 9
    elif isinstance(value, float):
        if not value.is_integer():
            raise ActionError("bad_arguments")
        value = int(value)
    elif not isinstance(value, int):
        raise ActionError("bad_arguments")
    if value < 1:
        raise ActionError("bad_arguments")
    return min(value, PAGE_SIZE), value > PAGE_SIZE


def is_first_page(cursor: object) -> bool:
    """null, "", 0 and "0" all mean: from the start."""
    if cursor is None or cursor == "" or cursor == "0":
        return True
    return type(cursor) is int and cursor == 0


def submit_payload(arguments: dict, episode_id: str) -> object:
    """The findings message the judge checks, from any of the accepted submit forms (C1):

    - `findings` is the message {"episode_id", "findings": [...]} (the documented form);
    - `findings` is a bare list: the episode is the current one, or `episode_id` given flat;
    - flat `episode_id` and `findings` next to the action name;
    - the message or the list as a JSON string (also the inner `findings` of a message);
    - the message under `findings.json`.
    Nothing else is repaired: a message without `episode_id` stays a structure error."""
    given = [arguments[key] for key in SUBMIT_FINDINGS_KEYS if arguments.get(key) is not None]
    if not given:
        return {}  # nothing to judge: an invalid structure, not silence
    if any(value != given[0] for value in given[1:]):
        raise ActionError("bad_arguments", "submit")  # two different answers: an invalid structure
    raw = _json_value(given[0], rounds=2)
    flat_id = arguments.get("episode_id")
    if isinstance(raw, list):
        # A flat episode_id is used as given, even null (which the judge refuses, as it refuses the
        # same null inside a message); only an absent one means the running episode.
        return {"episode_id": arguments["episode_id"] if "episode_id" in arguments else episode_id,
                "findings": raw}
    if isinstance(raw, dict):
        message = dict(raw)
        if isinstance(message.get("findings"), str):
            message["findings"] = _json_value(message["findings"])
        if "episode_id" not in message and flat_id is not None:
            message["episode_id"] = flat_id
        return message
    return raw


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
                             "evidence_ids"],  # explanation is optional (AUDIT R5)
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
    "Страница — до 50 записей, next_cursor = null на последней; limit от 1 до 50, больше — обрезается "
    "до 50 (ответ тогда содержит limit_clamped: true). Каждое действие, в том числе "
    "с ошибкой, расходует шаг; submit завершает эпизод. "
    f"Бюджет — {MAX_STEPS} шагов, и submit входит в него: последним шагом должен быть submit, "
    f"не позже {MAX_STEPS}-го. Если {MAX_STEPS} шагов прошли без submit, эпизод обрезается "
    "и ответ считается пустым. Лимит времени (60 с) тоже обрезает эпизод с пустым ответом, но тогда "
    "в inspection_steps идут только действия, сделанные на самом деле. В награду входит inspection_steps — число действий до submit; "
    "сам submit в него не считается (в журнале есть и steps_total — все шаги с submit). Ответ — findings по схеме, "
    "episode_id из manifest; submit принимает findings как объект {episode_id, findings}, "
    "а голый список findings принимается как findings текущего эпизода; ложные выводы снижают награду. "
    "Правила ответа по типам (affected_ids, сумма, валюта, допустимые наборы доказательств) — в task.answer_rules."
)


def open_policy(policy: dict) -> dict:
    """The policy as the agent sees it: only the whitelisted keys (POLICY_KEYS, POLICY_KEYS_ADDED)."""
    import copy

    return {key: copy.deepcopy(policy[key]) for key in POLICY_KEYS + POLICY_KEYS_ADDED if key in policy}


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
                       "evidence_superset_allowed": policy.get("evidence_superset_allowed", True),
                       "evidence_extra_max": MAX_EVIDENCE_EXTRA},
        "accounts": dict(policy.get("accounts") or {}),
        "answer_rules": answer_rules(),
    }
