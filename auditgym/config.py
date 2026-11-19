"""Episode configuration: defaults and validation (docs/CONTRACTS.md, section 4).

normalize_config accepts any JSON-compatible value and either returns a complete, valid config
or raises ConfigError; no other exception escapes. Values are type-checked before they are
hashed, compared or copied, and the output is rebuilt from validated pieces only.
"""
from __future__ import annotations

import hashlib
import json
import re

CONFIG_VERSION = "1"
ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
INTERACTIONS = ("none", "I1", "I2")
CURRENCIES = ("KZT", "USD", "EUR", "RUB", "CNY")
CURRENCY_EXPONENTS = {"KZT": 2, "USD": 2, "EUR": 2, "RUB": 2, "CNY": 2}

MAX_FINDINGS = 30
MAX_LEDGER_ROWS = 2000
MAX_PAGES = 60
PAGE_SIZE = 50
# Flag-everything must not pay off: at most one error per this many invoices (docs/DECISIONS.md A3).
INVOICES_PER_ERROR = 8
# Ledger budget used by the size check and enforced by the generator: the clean story posts at
# most ROWS_PER_INVOICE rows per invoice on average, a background pair two rows, and an issue at
# most ROWS_PER_ISSUE extra rows (a duplicate copy with its payment is the largest: 2 + 3).
ROWS_PER_INVOICE = 5
ROWS_PER_ISSUE = 5
MIN_YEAR, MAX_YEAR = 2000, 2100
SIZE_LIMITS = {"invoices": (10, 200), "background_pairs": (0, 900)}

DEFAULTS = {
    "config_version": CONFIG_VERSION,
    # 48 invoices: the default four errors plus an interaction scenario (6 issues) fit the density limit.
    "size": {"invoices": [48, 100], "background_pairs": [150, 300]},
    "currencies": ["USD"],
    "base_currency": "KZT",
    "timezone": "+05:00",
    "period": {"start": "2026-10-01", "months": 1},
    "errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
    "interaction": "none",
    "distractors": True,
    "surface_seed": None,
}

# re.fullmatch with [0-9]: no Unicode digits, no trailing newline.
_OFFSET = re.compile(r"[+-](0[0-9]|1[0-4]):(00|30|45)")
_MONTH_START = re.compile(r"([0-9]{4})-(0[1-9]|1[0-2])-01")


class ConfigError(ValueError):
    """Invalid configuration. `code` is a short stable identifier."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _str_keys(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _range(value: object, name: str) -> list[int]:
    low, high = SIZE_LIMITS[name]
    if (not isinstance(value, list) or len(value) != 2 or not all(_is_int(v) for v in value)
            or not low <= value[0] <= value[1] <= high):
        raise ConfigError("bad_size", f"size.{name} must be [min, max] with {low} <= min <= max <= {high}")
    return [value[0], value[1]]


def valid_offset(text: object) -> bool:
    """A fixed offset from -12:00 to +14:00 with minutes 00, 30 or 45."""
    if not isinstance(text, str) or not _OFFSET.fullmatch(text):
        return False
    minutes = int(text[1:3]) * 60 + int(text[4:6])
    return minutes <= (14 * 60 if text[0] == "+" else 12 * 60)


def interaction_issue_count(interaction: str) -> int:
    return 0 if interaction == "none" else 2


def config_digest(config: dict) -> str:
    """Digest of a normalized config without surface_seed: seeds the story and the ids."""
    body = {key: value for key, value in config.items() if key != "surface_seed"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def normalize_config(config: object) -> dict:
    """Fill defaults and validate. Raises ConfigError; never returns a partly valid config."""
    if config is None:
        config = {}
    if not _str_keys(config):
        raise ConfigError("bad_config", "config must be an object with string keys, or null")
    unknown = sorted(key for key in config if key not in DEFAULTS)
    if unknown:
        raise ConfigError("unknown_key", f"unknown config key: {unknown[0][:40]!r}")

    def given(key: str) -> object:
        return config[key] if key in config else DEFAULTS[key]

    out: dict = {}
    if not isinstance(given("config_version"), str) or given("config_version") != CONFIG_VERSION:
        raise ConfigError("bad_version", f"config_version must be {CONFIG_VERSION!r}")
    out["config_version"] = CONFIG_VERSION

    size = given("size")
    if not _str_keys(size) or any(key not in DEFAULTS["size"] for key in size):
        raise ConfigError("bad_size", "size must be an object with only invoices and background_pairs")
    out["size"] = {name: _range(size[name] if name in size else DEFAULTS["size"][name], name)
                   for name in ("invoices", "background_pairs")}

    base = given("base_currency")
    if not isinstance(base, str) or base not in CURRENCIES:
        raise ConfigError("bad_currency", "base_currency is not supported")
    out["base_currency"] = base
    currencies = given("currencies")
    if (not isinstance(currencies, list) or len(currencies) > 3
            or not all(isinstance(c, str) and c in CURRENCIES and c != base for c in currencies)
            or len(set(currencies)) != len(currencies)):
        raise ConfigError("bad_currency", "currencies must be up to 3 distinct supported non-base currencies")
    out["currencies"] = list(currencies)

    if not valid_offset(given("timezone")):
        raise ConfigError("bad_timezone", "timezone must be a fixed offset from -12:00 to +14:00, such as +05:00")
    out["timezone"] = given("timezone")

    period = given("period")
    if not _str_keys(period) or sorted(period) != ["months", "start"]:
        raise ConfigError("bad_period", "period needs exactly start and months")
    start, months = period["start"], period["months"]
    match = _MONTH_START.fullmatch(start) if isinstance(start, str) else None
    if match is None or not MIN_YEAR <= int(match.group(1)) <= MAX_YEAR or not _is_int(months) or not 1 <= months <= 3:
        raise ConfigError("bad_period", f"period needs start YYYY-MM-01 (years {MIN_YEAR}-{MAX_YEAR}) and months 1..3")
    out["period"] = {"start": start, "months": months}

    errors = given("errors")
    if not _str_keys(errors) or any(key not in ERROR_TYPES for key in errors):
        raise ConfigError("bad_errors", "errors keys must be error types")
    if any(not _is_int(value) or value < 0 for value in errors.values()):
        raise ConfigError("bad_errors", "error counts must be non-negative integers")
    out["errors"] = {name: errors.get(name, 0) for name in ERROR_TYPES}

    interaction = given("interaction")
    if not isinstance(interaction, str) or interaction not in INTERACTIONS:
        raise ConfigError("bad_interaction", "interaction must be none, I1 or I2")
    out["interaction"] = interaction
    if not isinstance(given("distractors"), bool):
        raise ConfigError("bad_distractors", "distractors must be true or false")
    out["distractors"] = given("distractors")
    surface_seed = given("surface_seed")
    if surface_seed is not None and not _is_int(surface_seed):
        raise ConfigError("bad_surface_seed", "surface_seed must be an integer or null")
    out["surface_seed"] = surface_seed

    foreign_needed = out["errors"]["fx_error"] > 0 or interaction == "I1"
    if foreign_needed and not currencies:
        raise ConfigError("needs_foreign_currency", "fx_error and I1 need at least one foreign currency")

    total_issues = sum(out["errors"].values()) + interaction_issue_count(interaction)
    if total_issues > MAX_FINDINGS:
        raise ConfigError("too_many_errors", f"at most {MAX_FINDINGS} issues fit into one answer")
    if total_issues * INVOICES_PER_ERROR > out["size"]["invoices"][0]:
        raise ConfigError("too_dense", f"need at least {INVOICES_PER_ERROR} invoices per error")

    invoices_high = out["size"]["invoices"][1]
    rows_high = (invoices_high * ROWS_PER_INVOICE + out["size"]["background_pairs"][1] * 2
                 + total_issues * ROWS_PER_ISSUE)
    if rows_high > MAX_LEDGER_ROWS:
        raise ConfigError("too_large", f"ledger could exceed {MAX_LEDGER_ROWS} rows")
    fx_rows = len(currencies) * (31 * months + 10)
    documents_high = invoices_high + total_issues  # the generator keeps payments <= invoices
    pages = sum(-(-n // PAGE_SIZE) for n in (rows_high, documents_high, documents_high, fx_rows))
    if pages > MAX_PAGES:
        raise ConfigError("too_large", f"a full read could exceed {MAX_PAGES} pages")
    return out
