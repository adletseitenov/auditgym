"""Episode configuration: defaults and validation (docs/CONTRACTS.md, section 4).

normalize_config accepts any JSON-compatible value and either returns a complete, valid config
or raises ConfigError; no other exception escapes. Values are type-checked before they are
hashed, compared or copied, and the output is rebuilt from validated pieces only.
"""
from __future__ import annotations

import hashlib
import json
import re

from .limits import (INVOICES_PER_ERROR, MAX_BACKGROUND_PAIRS, MAX_FINDINGS, MAX_FOREIGN_CURRENCIES,
                     MAX_INVOICES, MAX_LEDGER_ROWS, MAX_PAGES, MAX_PERIOD_MONTHS, MAX_SEED_BITS, MAX_YEAR, MIN_BACKGROUND_PAIRS,
                     MIN_INVOICES, MIN_YEAR, ROWS_PER_INVOICE, ROWS_PER_ISSUE, pages)

CONFIG_VERSION = "1"
ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
# Added later: kept out of ERROR_TYPES so defaults and the config digest of every old config stay
# the same; the key is in a normalized config only when its count is positive.
EXTRA_ERROR_TYPES = ("wrong_customer",)
ALL_ERROR_TYPES = ERROR_TYPES + EXTRA_ERROR_TYPES
INTERACTIONS = ("none", "I1", "I2")
CURRENCIES = ("KZT", "USD", "EUR", "RUB", "CNY")
CURRENCY_EXPONENTS = {"KZT": 2, "USD": 2, "EUR": 2, "RUB": 2, "CNY": 2}

SIZE_LIMITS = {"invoices": (MIN_INVOICES, MAX_INVOICES), "background_pairs": (MIN_BACKGROUND_PAIRS, MAX_BACKGROUND_PAIRS)}
DEFAULT_INVOICES = [48, 100]
FX_DAYS_PER_MONTH = 31
FX_MARGIN_ROWS = 10  # five days of rates on each side of the period

DEFAULTS = {
    "config_version": CONFIG_VERSION,
    "size": {"invoices": DEFAULT_INVOICES, "background_pairs": [150, 300]},
    # Adaptive defaults (docs/AUDIT.md D5): an omitted `currencies` is the first supported currency
    # other than the base; omitted `errors` shrink to the density limit of the size (and drop
    # fx_error without a foreign currency); an omitted size.invoices grows to fit the errors.
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
    size_given = {name: "size" in config and name in size for name in ("invoices", "background_pairs")}
    out["size"] = {name: _range(size[name] if name in size else DEFAULTS["size"][name], name)
                   for name in ("invoices", "background_pairs")}

    base = given("base_currency")
    if not isinstance(base, str) or base not in CURRENCIES:
        raise ConfigError("bad_currency", "base_currency is not supported")
    out["base_currency"] = base
    currencies = config["currencies"] if "currencies" in config else [next(c for c in CURRENCIES if c != base)]
    if (not isinstance(currencies, list) or len(currencies) > MAX_FOREIGN_CURRENCIES
            or not all(isinstance(c, str) and c in CURRENCIES and c != base for c in currencies)
            or len(set(currencies)) != len(currencies)):
        raise ConfigError("bad_currency", f"currencies must be up to {MAX_FOREIGN_CURRENCIES} distinct supported "
                                          "non-base currencies")
    out["currencies"] = list(currencies)

    if not valid_offset(given("timezone")):
        raise ConfigError("bad_timezone", "timezone must be a fixed offset from -12:00 to +14:00, such as +05:00")
    out["timezone"] = given("timezone")

    period = given("period")
    if not _str_keys(period) or sorted(period) != ["months", "start"]:
        raise ConfigError("bad_period", "period needs exactly start and months")
    start, months = period["start"], period["months"]
    match = _MONTH_START.fullmatch(start) if isinstance(start, str) else None
    if (match is None or not MIN_YEAR <= int(match.group(1)) <= MAX_YEAR or not _is_int(months)
            or not 1 <= months <= MAX_PERIOD_MONTHS):
        raise ConfigError("bad_period", f"period needs start YYYY-MM-01 (years {MIN_YEAR}-{MAX_YEAR}) "
                                        f"and months 1..{MAX_PERIOD_MONTHS}")
    out["period"] = {"start": start, "months": months}

    interaction = given("interaction")
    if not isinstance(interaction, str) or interaction not in INTERACTIONS:
        raise ConfigError("bad_interaction", "interaction must be none, I1 or I2")
    out["interaction"] = interaction

    if "errors" in config:
        errors = config["errors"]
        if not _str_keys(errors) or any(key not in ALL_ERROR_TYPES for key in errors):
            raise ConfigError("bad_errors", "errors keys must be error types")
        if any(not _is_int(value) or value < 0 for value in errors.values()):
            raise ConfigError("bad_errors", "error counts must be non-negative integers")
        out["errors"] = {name: errors.get(name, 0) for name in ERROR_TYPES}
        out["errors"].update({name: errors[name] for name in EXTRA_ERROR_TYPES if errors.get(name, 0)})
        if not size_given["invoices"]:  # the default company grows to hold the requested errors
            low = max(DEFAULT_INVOICES[0], INVOICES_PER_ERROR * _issue_count(out["errors"], interaction))
            out["size"]["invoices"] = [min(low, MAX_INVOICES), min(max(DEFAULT_INVOICES[1], low), MAX_INVOICES)]
    else:
        out["errors"] = _default_errors(out["size"]["invoices"][0], interaction, bool(currencies))

    if not isinstance(given("distractors"), bool):
        raise ConfigError("bad_distractors", "distractors must be true or false")
    out["distractors"] = given("distractors")
    surface_seed = given("surface_seed")
    if surface_seed is not None and not _is_int(surface_seed):
        raise ConfigError("bad_surface_seed", "surface_seed must be an integer or null")
    if surface_seed is not None and surface_seed.bit_length() > MAX_SEED_BITS:
        raise ConfigError("bad_surface_seed", f"surface_seed must fit into {MAX_SEED_BITS} bits")
    out["surface_seed"] = surface_seed

    foreign_needed = out["errors"]["fx_error"] > 0 or interaction == "I1"
    if foreign_needed and not currencies:
        raise ConfigError("needs_foreign_currency", "fx_error and I1 need at least one foreign currency")

    total_issues = _issue_count(out["errors"], interaction)
    if total_issues > MAX_FINDINGS:
        raise ConfigError("too_many_errors", f"at most {MAX_FINDINGS} issues fit into one answer")
    if total_issues * INVOICES_PER_ERROR > out["size"]["invoices"][0]:
        raise ConfigError("too_dense", f"need at least {INVOICES_PER_ERROR} invoices per error")
    check_size(out["size"]["invoices"][1], out["size"]["background_pairs"][0], total_issues,
               len(currencies), months)
    return out


def _issue_count(errors: dict, interaction: str) -> int:
    return sum(errors.values()) + interaction_issue_count(interaction)


def _default_errors(invoices_low: int, interaction: str, foreign: bool) -> dict:
    """One error of each type, as many as the density limit allows, in ERROR_TYPES order;
    fx_error only with a foreign currency."""
    room = invoices_low // INVOICES_PER_ERROR - interaction_issue_count(interaction)
    errors = {name: 0 for name in ERROR_TYPES}
    for name in ERROR_TYPES:
        if room > 0 and (foreign or name != "fx_error"):
            errors[name] = 1
            room -= 1
    return errors


def ledger_budget(invoices_high: int, background_low: int, issues: int) -> int:
    """Rows the ledger needs at most for the guaranteed part of the size. The generator caps the
    background pairs so the whole ledger stays within MAX_LEDGER_ROWS (world.py)."""
    return invoices_high * ROWS_PER_INVOICE + background_low * 2 + issues * ROWS_PER_ISSUE


def full_read_pages(rows: int, documents: int, fx_rows: int) -> int:
    """Pages for a full read: ledger, invoices and payments (each at most `documents`), fx."""
    return pages(rows) + 2 * pages(documents) + pages(fx_rows)


def check_size(invoices_high: int, background_low: int, issues: int, currencies: int, months: int) -> None:
    rows = ledger_budget(invoices_high, background_low, issues)
    if rows > MAX_LEDGER_ROWS:
        raise ConfigError("too_large", f"ledger could exceed {MAX_LEDGER_ROWS} rows")
    fx_rows = currencies * (FX_DAYS_PER_MONTH * months + FX_MARGIN_ROWS)
    documents = invoices_high + issues  # the generator keeps payments <= invoices
    if full_read_pages(MAX_LEDGER_ROWS, documents, fx_rows) > MAX_PAGES:
        raise ConfigError("too_large", f"a full read could exceed {MAX_PAGES} pages")
