"""Structure check and normalization of submitted findings (docs/CONTRACTS.md, section 6).

`parse_findings(payload, episode_id)` either returns the normalized findings or raises
`StructureError(code)`; the judge turns that into reward -1. It never raises anything else
for a JSON-compatible payload, and it never calls `json.dumps` on agent data, so deep nesting,
huge integers and reference cycles cannot crash it.

Decisions on points the case leaves open (audit R2, R5 relaxed the strictest reading where
a reasonable oracle built slightly differently would otherwise get -1):
- Five finding fields are required; `explanation` is optional and defaults to the empty
  string (when present it must be a string of at most 1 000 code points).
- `evidence_ids` is limited to 10 UNIQUE ids: a list that cites every evidence set in a row
  with repeats is fine as long as it names at most 10 different ids.
- `claimed_amount_minor` is an `int`; a float with an integral value (`125000.0`) is accepted
  and read as that int. `true`, `100.5`, `"100"`, NaN and Infinity stay invalid.
- `type` and `currency` ignore case and surrounding whitespace (ASCII case only).
- Size is the compact UTF-8 JSON serialization (`separators=(",", ":")`, non-ASCII kept
  as UTF-8, lone surrogates counted as 3 bytes) and must be at most 262 144 bytes (256 KiB;
  30 findings x 10 evidence ids x 1 000 Cyrillic characters is 68 KB, so the case's own limits
  always fit).
- A non-finite float anywhere in the payload, even under an ignored extra key, is an
  error: NaN and Infinity are not JSON.
- A value that is not JSON-compatible (a dict key that is not a string, a set, bytes, an
  object) is an error. Tuples are accepted where a JSON array is (they serialize as one) but
  the `findings`, `affected_ids` and `evidence_ids` fields must be real lists.
- `affected_ids` and `evidence_ids` may be empty: such a finding is well-formed but never
  matches, because every truth issue has at least one affected id and a non-empty evidence set.
- Absurdly large non-negative integers are well-formed amounts (they simply never match);
  the 256 KiB limit bounds them. Their size is computed without converting them to a string.
- `currency` is upper-cased and `type` lower-cased in ASCII only (`"usd"` -> `"USD"`), so that
  Unicode case folding (`"ſ".upper() == "S"`) cannot make a different string equal to a code.
- Every id must be a string; the empty string is allowed and matches nothing.
"""
from __future__ import annotations

import math

MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CODE_POINTS = 1000
MAX_MESSAGE_BYTES = 256 * 1024

TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
FIELDS = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")

_ASCII_UPPER = str.maketrans("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
# Above this many bits an integer has more than 264 000 decimal digits: too large outright.
_HUGE_INT_BITS = 880_000
_LOG10_2 = 0.30102999566398120
_SMALL_INT_BITS = 10_000  # below this str() is safe under the 4 300-digit conversion limit


class StructureError(Exception):
    """The findings message is malformed; `code` is a short fixed string for the judge log."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _string_bytes(text: str) -> int:
    """Length in bytes of `text` as a compact UTF-8 JSON string literal, quotes included."""
    size = 2
    for char in text:
        point = ord(char)
        if char in '"\\' or char in "\b\f\n\r\t":
            size += 2
        elif point < 0x20:
            size += 6  # \u00XX
        elif point < 0x80:
            size += 1
        elif point < 0x800:
            size += 2
        elif point < 0x10000:
            size += 3  # lone surrogates included (surrogatepass)
        else:
            size += 4
    return size


def _digits(value: int, bits: int) -> int:
    """Exact number of decimal digits of a positive int without str() (Decimal is too slow here)."""
    estimate = int(bits * _LOG10_2)
    if value >= 10 ** (estimate + 1):
        return estimate + 2
    return estimate + 1 if value >= 10 ** estimate else estimate


def _int_bytes(value: int) -> int:
    bits = value.bit_length()
    if bits > _HUGE_INT_BITS:
        return MAX_MESSAGE_BYTES + 1
    sign = 1 if value < 0 else 0
    if bits < _SMALL_INT_BITS:
        return len(str(abs(value))) + sign
    return _digits(abs(value), bits) + sign  # exact digit count, no str() limit


def _float_bytes(value: float) -> int:
    if not math.isfinite(value):
        raise StructureError("non_finite")
    return len(float.__repr__(value))


def message_size(payload: object) -> int:
    """Compact UTF-8 JSON size of `payload`, or any number above the limit once it is exceeded.

    Iterative (no recursion limit), stops as soon as the budget is spent (so cycles and huge
    inputs end quickly), and raises StructureError for values JSON cannot represent.
    """
    total = 0
    stack: list[object] = [payload]
    while stack:
        item = stack.pop()
        if item is None:
            total += 4
        elif item is True:
            total += 4
        elif item is False:
            total += 5
        elif isinstance(item, int):
            total += _int_bytes(int(item))
        elif isinstance(item, float):
            total += _float_bytes(item)
        elif isinstance(item, str):
            if len(item) > MAX_MESSAGE_BYTES:
                return MAX_MESSAGE_BYTES + 1
            total += _string_bytes(item)
        elif isinstance(item, (list, tuple)):
            total += 2 + max(len(item) - 1, 0)
            if total > MAX_MESSAGE_BYTES:
                return total
            stack.extend(item)
        elif isinstance(item, dict):
            total += 2 + max(len(item) - 1, 0) + len(item)  # braces, commas, colons
            if total > MAX_MESSAGE_BYTES:
                return total
            for key, value in item.items():
                if not isinstance(key, str):
                    raise StructureError("not_json")
                stack.append(key)
                stack.append(value)
        else:
            raise StructureError("not_json")
        if total > MAX_MESSAGE_BYTES:
            return total
    return total


def _id_list(value: object, limit: int | None) -> list[str]:
    if not isinstance(value, list):
        raise StructureError("ids_not_list")
    for item in value:
        if not isinstance(item, str):
            raise StructureError("id_not_string")
    unique = sorted(set(value))
    if limit is not None and len(unique) > limit:  # the limit counts different ids (audit R5)
        raise StructureError("too_many_evidence_ids")
    return unique


def _amount(value: object) -> int:
    if type(value) is float and value.is_integer():  # 125000.0 is the amount 125000 (audit R5)
        value = int(value)
    if type(value) is not int:  # rejects bool, 100.5, str, int subclasses
        raise StructureError("amount_not_int")
    if value < 0:
        raise StructureError("negative_amount")
    return value


def _finding(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise StructureError("finding_not_object")
    for field in FIELDS:
        if field != "explanation" and field not in raw:
            raise StructureError("missing_field")
    kind = raw["type"]
    if isinstance(kind, str):
        kind = kind.strip().translate(_ASCII_LOWER)
    if not isinstance(kind, str) or kind not in TYPES:
        raise StructureError("unknown_type")
    affected = _id_list(raw["affected_ids"], None)
    amount = _amount(raw["claimed_amount_minor"])
    currency = raw["currency"]
    if not isinstance(currency, str):
        raise StructureError("currency_not_string")
    evidence = _id_list(raw["evidence_ids"], MAX_EVIDENCE_IDS)
    explanation = raw.get("explanation", "")
    if not isinstance(explanation, str):
        raise StructureError("explanation_not_string")
    if len(explanation) > MAX_EXPLANATION_CODE_POINTS:
        raise StructureError("explanation_too_long")
    return {"type": kind, "affected_ids": affected, "claimed_amount_minor": amount,
            "currency": currency.strip().translate(_ASCII_UPPER), "evidence_ids": evidence,
            "explanation": explanation}


def parse_findings(payload: object, episode_id: str) -> list[dict]:
    """Normalized findings in submission order, or StructureError(code)."""
    if message_size(payload) > MAX_MESSAGE_BYTES:
        raise StructureError("message_too_large")
    if not isinstance(payload, dict):
        raise StructureError("not_object")
    if "episode_id" not in payload or "findings" not in payload:
        raise StructureError("missing_field")
    if not isinstance(payload["episode_id"], str) or payload["episode_id"] != episode_id:
        raise StructureError("wrong_episode_id")
    findings = payload["findings"]
    if not isinstance(findings, list):
        raise StructureError("findings_not_list")
    if len(findings) > MAX_FINDINGS:
        raise StructureError("too_many_findings")
    return [_finding(raw) for raw in findings]
