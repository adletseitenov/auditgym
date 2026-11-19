"""generate(seed, config) -> episode (docs/CONTRACTS.md, section 1)."""
from __future__ import annotations

import hashlib
import json

from . import SCHEMA_VERSION, __version__
from .books import build_books
from .config import CONFIG_VERSION, ConfigError, normalize_config
from .inject import IMPLEMENTED, NoTarget, inject_errors
from .rng import Rng
from .surface import apply_surface, public_record
from .world import build_world

# Attempt k > 0 regenerates the whole story from a derived stream. A miss needs a story with fewer
# unpaid invoices than period_cutoff cases; 64 misses in a row are far below any practical chance.
MAX_ATTEMPTS = 64

ACCOUNTS = ("cash", "receivable", "revenue", "unallocated", "fx_difference", "expense")


def build_policy(world: dict) -> dict:
    return {
        "ledger_currency": "base", "base_currency": world["base_currency"], "timezone": world["timezone"],
        "period_start": world["period_start"].isoformat(), "period_end": world["period_end"].isoformat(),
        "fx_rate_date": "document_date_in_company_timezone",
        "fx_formula": "round(amount_minor * 10^(exp_base - exp_src) * rate)", "rounding": "half_even",
        "currency_exponents": dict(world["currency_exponents"]),
        "accounts": {name: name for name in ACCOUNTS},
        "document_date": {"invoice": "issued_at", "payment": "timestamp"},
        "refund": "payment with negative amount_minor and one invoice ref",
        "amount_tolerance_minor": {"duplicate_invoice": 0, "allocation_gap": 0, "fx_error": 1, "period_cutoff": 0},
        "evidence_superset_allowed": True,
    }


def _episode_id(tables: dict) -> str:
    """A hash of the open records: stable, and it does not contain the seed."""
    body = json.dumps({name: tables[name] for name in ("ledger", "invoices", "payments", "fx")},
                      ensure_ascii=False, sort_keys=True)
    return "ep_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def _attempt_root(seed: int, attempt: int) -> Rng:
    return Rng(seed) if attempt == 0 else Rng(seed, f"attempt:{attempt}")


def _build(seed: int, config: dict) -> tuple[int, dict, list[dict], list[dict], list[dict]]:
    """Deterministic bounded retry over attempts 0..MAX_ATTEMPTS-1."""
    for attempt in range(MAX_ATTEMPTS):
        root = _attempt_root(seed, attempt)
        world = build_world(root.child("world"), config)
        ledger = build_books(world)
        pre_ledger = [dict(row) for row in ledger]
        try:
            world, ledger, injections = inject_errors(root, world, ledger, config["errors"])
        except NoTarget:
            continue
        return attempt, world, ledger, pre_ledger, injections
    raise RuntimeError("no valid episode within the attempt limit")  # unreachable for valid configs


def _public_evidence(group: list[str], id_map: dict[str, str]) -> list[str]:
    """Documents first, sorted by public id (internal key order must not show), then rows as given."""
    docs = sorted(id_map[key] for key in group if key.startswith("D:"))
    return docs + [id_map[key] for key in group if not key.startswith("D:")]


def _public_params(params: dict, id_map: dict[str, str]) -> dict:
    return {name: id_map.get(value, value) if isinstance(value, str) else value for name, value in params.items()}


def generate(seed: int, config: dict | None = None) -> dict:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("bad_seed", "seed must be an integer")
    config = normalize_config(config)
    missing = [name for name, count in config["errors"].items() if count and name not in IMPLEMENTED]
    if missing or config["interaction"] != "none":
        # allocation_gap, fx_error and the interaction scenarios land in tasks G5 and G8.
        raise ConfigError("not_implemented", "this error type or interaction is not implemented yet")

    attempt, world, ledger, pre_ledger, injections = _build(seed, config)

    surface_seed = seed if config["surface_seed"] is None else config["surface_seed"]
    tables = apply_surface(Rng(surface_seed, "surface"), world, ledger, pre_ledger)
    id_map = tables["id_map"]
    issues: list[dict] = []
    steps: list[dict] = []
    for number, injection in enumerate(injections, 1):
        issue_id = f"t{number}"
        issues.append({"issue_id": issue_id, "type": injection["type"],
                       "affected_ids": sorted(id_map[key] for key in injection["affected"]),
                       "expected_amount_minor": injection["amount_minor"], "currency": injection["currency"],
                       "evidence_sets": [_public_evidence(group, id_map) for group in injection["evidence"]]})
        steps.append({"operator": injection["type"], "params": _public_params(injection["params"], id_map),
                      "issue_id": issue_id,
                      "before": [public_record(record, id_map) for record in injection["before"]],
                      "after": [public_record(record, id_map) for record in injection["after"]]})

    episode_id = _episode_id(tables)
    policy = build_policy(world)
    manifest = {
        "schema_version": SCHEMA_VERSION, "episode_id": episode_id, "base_currency": world["base_currency"],
        "period_start": policy["period_start"], "period_end": policy["period_end"], "timezone": world["timezone"],
        "counts": {name: len(tables[name]) for name in ("ledger", "invoices", "payments", "fx")},
        "rounding": policy["rounding"], "config_version": CONFIG_VERSION,
    }
    return {
        "public": {"manifest": manifest, "policy": policy, "ledger": tables["ledger"],
                   "invoices": tables["invoices"], "payments": tables["payments"], "fx": tables["fx"]},
        "private": {
            "truth": {"schema_version": SCHEMA_VERSION, "episode_id": episode_id, "issues": issues},
            "provenance": {"seed": seed, "config": config, "generator_version": __version__, "attempt": attempt,
                           "steps": steps, "id_map": id_map},
            "pre_ledger": tables["pre_ledger"],
        },
    }
