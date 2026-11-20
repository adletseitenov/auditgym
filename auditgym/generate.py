"""generate(seed, config, secret="") -> episode (docs/CONTRACTS.md, section 1).

Random streams are derived from the seed and the normalized config without surface_seed, so the
same seed with another config gives another world, and ids depend on the error config too. A
non-empty `secret` (the evaluator's, never stored in the episode) is mixed into every root stream,
so the seed alone no longer reproduces the episode (docs/AUDIT.md L1); the default "" keeps
episodes reproducible from seed and config.
Each attempt builds a story, injects the errors and checks the whole episode with verify() (which
evaluates the four published predicates over the open records). An attempt that has no target for
an operator, or whose episode verify() rejects, is replaced by the next attempt: a bounded
deterministic retry. Exhausting the bound raises ConfigError("no_episode").
"""
from __future__ import annotations

import hashlib
import json
from decimal import localcontext

from . import SCHEMA_VERSION, __version__
from .books import build_books
from .config import CONFIG_VERSION, ConfigError, config_digest, normalize_config
from .limits import MAX_LEDGER_ROWS, MAX_SEED_BITS
from .inject import NoTarget, inject_errors
from .money import DECIMAL_CONTEXT
from .rng import Rng
from .surface import assign_ids, public_record, public_tables
from .verify import POLICY_KEYS, verify
from .world import build_world

MAX_ATTEMPTS = 32
ACCOUNTS = ("cash", "receivable", "revenue", "unallocated", "fx_difference", "expense")


PAYMENT_RULE = ("a payment with one reference moves the invoice's receivable by "
                "round_half_even(booked * paid_after / total) - round_half_even(booked * paid_before / total), "
                "where booked is the invoice's posted base amount and paid counts its posted payments that come "
                "earlier in document-date order (the payment's own timestamp as an instant, not its posting time; "
                "ties by doc_id; every payment of an invoice is also posted after the invoice and after its "
                "earlier payments, so posting order gives the same sequence) "
                "(refunds reopen it the same way); an overpayment closes what is left and puts the excess at the "
                "payment date's rate on unallocated; a payment of several invoices closes each in full; an advance "
                "goes to unallocated; cash is the payment at its date's rate and the rest of every foreign payment "
                "is its own fx_difference row")
DUPLICATE_SIGNATURE = ("entity, currency, total, line items incl. service period; service_start/service_end "
                       "only identify duplicates and never affect period_cutoff, which uses the document date "
                       "alone (period_basis)")
# policy.json is built from the whitelist verify.POLICY_KEYS only (verify checks it on every episode): no
# key or value may come from the truth or the config of errors (docs/AUDIT.md Q2/S4).


def build_policy(world: dict) -> dict:
    values = {
        "ledger_currency": "base", "base_currency": world["base_currency"], "timezone": world["timezone"],
        "period_start": world["period_start"].isoformat(), "period_end": world["period_end"].isoformat(),
        "period_interval": "[start, end)", "period_basis": "document_date",
        "fx_rate_date": "document_date_in_company_timezone",
        "fx_formula": "round(amount_minor * 10^(exp_base - exp_src) * rate)", "rounding": "half_even",
        "currency_exponents": dict(world["currency_exponents"]),
        "accounts": {name: name for name in ACCOUNTS},
        "document_date": {"invoice": "issued_at", "payment": "timestamp"},
        "refund": "payment with negative amount_minor and one invoice ref",
        "payment_rule": PAYMENT_RULE,
        "duplicate_signature": DUPLICATE_SIGNATURE,
        "amount_tolerance_minor": {"duplicate_invoice": 0, "allocation_gap": 0, "fx_error": 1, "period_cutoff": 0},
        "evidence_superset_allowed": True,
    }
    return {key: values[key] for key in POLICY_KEYS}


def _episode_id(tables: dict) -> str:
    """A hash of the open records: stable, and it does not contain the seed."""
    body = json.dumps({name: tables[name] for name in ("ledger", "invoices", "payments", "fx")},
                      ensure_ascii=False, sort_keys=True)
    return "ep_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def _public_evidence(group: list[str], id_map: dict[str, str]) -> list[str]:
    """Documents first, then rows, then fx keys, each sorted by public id (internal key order must
    not show). An fx key ("FX:date|SRC|TGT") is already public."""
    docs = sorted(id_map[key] for key in group if key.startswith("D:"))
    rows = sorted(id_map[key] for key in group if not key.startswith(("D:", "FX:")))
    return docs + rows + sorted(key[3:] for key in group if key.startswith("FX:"))


def _public_params(params: dict, id_map: dict[str, str]) -> dict:
    """Only id-valued parameters (names ending in _id) are translated; `scenario`, modes and dates stay."""
    return {name: id_map[value] if name.endswith("_id") and isinstance(value, str) else value
            for name, value in params.items()}


class _Rejected(Exception):
    pass


def _root_key(seed: int, digest: str, secret: str) -> str:
    if not secret:
        return f"{seed}|{digest}"
    return f"{seed}|{digest}|" + hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _assemble(seed: int, config: dict, digest: str, attempt: int, retries: list[dict], secret: str) -> dict:
    root = Rng(_root_key(seed, digest, secret), f"attempt:{attempt}")
    pre_world = build_world(root.child("world"), config)
    try:
        world, injections = inject_errors(root, pre_world, config)
    except NoTarget as missing:
        raise _Rejected(f"no target: {missing}") from None
    pre_ledger, ledger = build_books(pre_world), build_books(world)
    if len(ledger) > MAX_LEDGER_ROWS:
        raise _Rejected("ledger too large")

    keys = set(world["entities"])
    for document in world["invoices"] + world["payments"]:
        keys |= {document["key"], "D:" + document["key"]}
    keys |= {row["key"] for row in ledger} | {row["key"] for row in pre_ledger}
    for injection in injections:
        keys |= {record["key"] for record in injection["before"] + injection["after"]}
    surface_seed = seed if config["surface_seed"] is None else config["surface_seed"]
    id_map = assign_ids(Rng(_root_key(surface_seed, digest, secret), "surface"), keys)
    tables = public_tables(world, ledger, id_map)
    pre_tables = public_tables(pre_world, pre_ledger, id_map)

    issues, steps = [], []
    for number, injection in enumerate(injections, 1):
        issue_id = f"t{number}"
        issues.append({"issue_id": issue_id, "type": injection["type"],
                       "affected_ids": sorted(id_map[key] for key in injection["affected"]),
                       "expected_amount_minor": injection["amount_minor"], "currency": injection["currency"],
                       "evidence_sets": sorted(_public_evidence(group, id_map) for group in injection["evidence"])})

        def records(items: list[dict]) -> list[dict]:
            return sorted((public_record(record, id_map) for record in items),
                          key=lambda r: (r.get("row_id") is not None, r.get("doc_id") or r.get("row_id")))

        steps.append({"operator": injection["operator"], "params": _public_params(injection["params"], id_map),
                      "issue_id": issue_id, "before": records(injection["before"]), "after": records(injection["after"])})

    episode_id = _episode_id(tables)
    policy = build_policy(world)
    manifest = {
        "schema_version": SCHEMA_VERSION, "episode_id": episode_id, "base_currency": world["base_currency"],
        "period_start": policy["period_start"], "period_end": policy["period_end"], "timezone": world["timezone"],
        "counts": {name: len(tables[name]) for name in ("ledger", "invoices", "payments", "fx")},
        "rounding": policy["rounding"], "config_version": CONFIG_VERSION,
    }
    episode = {
        "public": {"manifest": manifest, "policy": policy, **tables},
        "private": {
            "truth": {"schema_version": SCHEMA_VERSION, "episode_id": episode_id, "issues": issues},
            "provenance": {"seed": seed, "config": config, "generator_version": __version__, "attempt": attempt,
                           "retries": list(retries), "steps": steps, "id_map": id_map},
            "pre_ledger": pre_tables["ledger"],
            "pre_invoices": pre_tables["invoices"],
            "pre_payments": pre_tables["payments"],
        },
    }
    problems = verify(episode)
    if problems:
        raise _Rejected(f"verify: {problems[0]}")
    return episode


def generate(seed: int, config: dict | None = None, secret: str = "") -> dict:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("bad_seed", "seed must be an integer")
    if seed.bit_length() > MAX_SEED_BITS:  # the evaluator's limit, so a direct call cannot take any length
        raise ConfigError("bad_seed", f"seed must fit into {MAX_SEED_BITS} bits")
    if not isinstance(secret, str):
        raise ConfigError("bad_secret", "secret must be a string")
    config = normalize_config(config)
    digest = config_digest(config)
    retries: list[dict] = []
    with localcontext(DECIMAL_CONTEXT):  # the caller's Decimal context must not move a rate
        for attempt in range(MAX_ATTEMPTS):
            try:
                return _assemble(seed, config, digest, attempt, retries, secret)
            except _Rejected as rejected:
                retries.append({"attempt": attempt, "reason": str(rejected)[:200]})
            except Exception as error:  # never escapes as anything but ConfigError; fuzz reports "internal"
                retries.append({"attempt": attempt, "reason": f"internal: {type(error).__name__}"})
    raise ConfigError("no_episode", f"no valid episode within {MAX_ATTEMPTS} attempts")
