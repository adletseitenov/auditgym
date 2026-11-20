"""Measure two shortcuts against the full definitions (docs/CASE.md 02: the difficulty must defeat
simple tricks, not make the task unsolvable). The numbers are in docs/DESIGN.md, section 3.1.

Run: python -m tools.tricks [--seeds 300] [--check]

Shortcuts, each a reading of the open records that skips one part of a definition:
- duplicates by key: every pair of invoices with the same full signature (entity, currency, total,
  line items with the service period), without looking at the ledger. Defeated by decoy A, a resent
  copy that is never posted.
- cutoff by service: every posted invoice whose service lies wholly outside the period. Defeated by
  decoy B, an invoice dated and posted in the period for a neighbouring period's service (and it
  misses the I2 copy, whose service is inside the period).
Full definitions (policy.duplicate_signature, policy.period_basis):
- duplicates: the same signature and both invoices posted in the ledger;
- cutoff: a posted document dated outside [start, end), the date taken as an instant (so in the
  company's zone), invoices and payments.
Each prediction is a (type, set of doc_ids); precision = hits / predictions, recall = hits / truth
issues of that type, both pooled over all episodes of a config. Truth is read only to score.

Configs: the default, every level of configs/levels.json, the default with I1 and with I2, seeds
SEED_BASE..SEED_BASE+seeds-1. The same run also checks that the solver's findings equal the truth on
every episode. `--check` exits 1 unless the full definitions and the solver are exact (precision and
recall 1.0, solver == truth everywhere) and each shortcut's precision on the default config is at
most SHORTCUT_MAX_PRECISION.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict

from auditgym import levels as levels_module
from auditgym.generate import generate
from auditgym.money import parse_offset, parse_time
from auditgym.solver import solve

SEEDS = 300
SEED_BASE = 81_000
SHORTCUT_MAX_PRECISION = 0.6
METHODS = ("duplicates by key", "duplicates (definition)", "cutoff by service", "cutoff (definition)")
TYPE_OF = {"duplicates by key": "duplicate_invoice", "duplicates (definition)": "duplicate_invoice",
           "cutoff by service": "period_cutoff", "cutoff (definition)": "period_cutoff"}


def configs() -> list[tuple[str, dict]]:
    out = [("default", {})]
    out += [(f"level {name}", levels_module.load(name)) for name in levels_module.names()]
    out += [("default + I1", {"interaction": "I1"}), ("default + I2", {"interaction": "I2"})]
    return out


def _signature(invoice: dict) -> tuple:
    items = sorted(json.dumps(item, sort_keys=True) for item in invoice["line_items"])
    return (invoice["entity_id"], invoice["currency"], invoice["total_minor"], tuple(items))


def predictions(public: dict) -> dict[str, set]:
    """Each method's predictions on the open records: {method: {(type, frozenset(doc_ids))}}."""
    policy = public["policy"]
    start, end = parse_time(policy["period_start"]), parse_time(policy["period_end"])
    zone = parse_offset(policy["timezone"])
    first = start.astimezone(zone).date().isoformat()
    last = (end - dt.timedelta(seconds=1)).astimezone(zone).date().isoformat()
    posted_invoices = {row["invoice_id"] for row in public["ledger"] if row["payment_id"] is None and row["invoice_id"]}
    posted_payments = {row["payment_id"] for row in public["ledger"] if row["payment_id"]}

    groups = defaultdict(list)
    for invoice in public["invoices"]:
        groups[_signature(invoice)].append(invoice)
    out = {method: set() for method in METHODS}
    for group in groups.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                pair = ("duplicate_invoice", frozenset([a["doc_id"], b["doc_id"]]))
                out["duplicates by key"].add(pair)
                if a["invoice_id"] in posted_invoices and b["invoice_id"] in posted_invoices:
                    out["duplicates (definition)"].add(pair)
    for invoice in public["invoices"]:
        if invoice["invoice_id"] not in posted_invoices:
            continue
        key = ("period_cutoff", frozenset([invoice["doc_id"]]))
        items = invoice["line_items"]
        if items and all(i["service_end"] < first or i["service_start"] > last for i in items):
            out["cutoff by service"].add(key)
        if not start <= parse_time(invoice["issued_at"]) < end:
            out["cutoff (definition)"].add(key)
    for payment in public["payments"]:
        if payment["payment_id"] in posted_payments and not start <= parse_time(payment["timestamp"]) < end:
            out["cutoff (definition)"].add(("period_cutoff", frozenset([payment["doc_id"]])))
    return out


def _solver_equals_truth(episode: dict) -> bool:
    found = sorted((f["type"], tuple(sorted(f["affected_ids"])), f["claimed_amount_minor"], f["currency"])
                   for f in solve(episode["public"]))
    truth = sorted((i["type"], tuple(sorted(i["affected_ids"])), i["expected_amount_minor"], i["currency"])
                   for i in episode["private"]["truth"]["issues"])
    return found == truth


def measure(config: dict, seeds: int, seed_base: int = SEED_BASE) -> dict:
    """{"episodes", "solver_mismatches", method: {"predicted", "hits", "truth"}} over the seeds."""
    stats = {"episodes": 0, "solver_mismatches": 0}
    stats.update({method: {"predicted": 0, "hits": 0, "truth": 0} for method in METHODS})
    for seed in range(seed_base, seed_base + seeds):
        episode = generate(seed, config)
        truth = {(issue["type"], frozenset(issue["affected_ids"])) for issue in episode["private"]["truth"]["issues"]}
        for method, predicted in predictions(episode["public"]).items():
            of_type = {t for t in truth if t[0] == TYPE_OF[method]}
            stats[method]["predicted"] += len(predicted)
            stats[method]["hits"] += len(predicted & of_type)
            stats[method]["truth"] += len(of_type)
        stats["episodes"] += 1
        stats["solver_mismatches"] += not _solver_equals_truth(episode)
    return stats


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def run(seeds: int, out=sys.stdout) -> dict[str, dict]:
    results = {}
    print("| Конфигурация | Эпизодов | Решатель ≠ эталон | " + " | ".join(f"{m}: P / R" for m in METHODS) + " |",
          file=out)
    print("|---" * (3 + len(METHODS)) + "|", file=out)
    for label, config in configs():
        stats = measure(config, seeds)
        results[label] = stats
        cells = [f"{ratio(stats[m]['hits'], stats[m]['predicted']):.2f} / {ratio(stats[m]['hits'], stats[m]['truth']):.2f}"
                 for m in METHODS]
        print(f"| {label} | {stats['episodes']} | {stats['solver_mismatches']} | " + " | ".join(cells) + " |",
              file=out, flush=True)
    return results


def failures(results: dict[str, dict]) -> list[str]:
    problems = []
    for label, stats in results.items():
        if stats["solver_mismatches"]:
            problems.append(f"{label}: solver differs from the truth on {stats['solver_mismatches']} episodes")
        for method in ("duplicates (definition)", "cutoff (definition)"):
            s = stats[method]
            if s["hits"] != s["predicted"] or s["hits"] != s["truth"]:
                problems.append(f"{label}: {method} is not exact ({s['hits']}/{s['predicted']}/{s['truth']})")
    default = results.get("default")
    if default is not None:
        for method in ("duplicates by key", "cutoff by service"):
            precision = ratio(default[method]["hits"], default[method]["predicted"])
            if precision > SHORTCUT_MAX_PRECISION:
                problems.append(f"default: {method} precision {precision:.2f} > {SHORTCUT_MAX_PRECISION}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    results = run(args.seeds)
    problems = failures(results)
    for problem in problems:
        print("FAIL:", problem)
    episodes = sum(stats["episodes"] for stats in results.values())
    print(f"tricks: {'FAIL' if problems else 'ok'} -- {episodes} episodes, solver == truth on "
          f"{episodes - sum(s['solver_mismatches'] for s in results.values())}")
    return 1 if args.check and problems else 0


if __name__ == "__main__":
    sys.exit(main())
