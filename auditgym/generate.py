"""generate(seed, config) -> episode (docs/CONTRACTS.md, section 1)."""
from __future__ import annotations

import hashlib
import json

from . import SCHEMA_VERSION, __version__
from .books import build_books
from .config import CONFIG_VERSION, ConfigError, interaction_issue_count, normalize_config
from .rng import Rng
from .surface import apply_surface
from .world import build_world

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


def generate(seed: int, config: dict | None = None) -> dict:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("bad_seed", "seed must be an integer")
    config = normalize_config(config)
    if sum(config["errors"].values()) or interaction_issue_count(config["interaction"]):
        # Error operators land in task G4; until then only clean companies can be generated.
        raise ConfigError("not_implemented", "error injection is not implemented yet")

    root = Rng(seed)
    world = build_world(root.child("world"), config)
    ledger = build_books(world)
    pre_ledger = [dict(row) for row in ledger]
    issues: list[dict] = []
    steps: list[dict] = []

    surface_seed = seed if config["surface_seed"] is None else config["surface_seed"]
    tables = apply_surface(Rng(surface_seed, "surface"), world, ledger, pre_ledger)
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
            "provenance": {"seed": seed, "config": config, "generator_version": __version__, "attempt": 0,
                           "steps": steps, "id_map": tables["id_map"]},
            "pre_ledger": tables["pre_ledger"],
        },
    }
