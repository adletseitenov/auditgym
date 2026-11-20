"""Measure how the 100-step budget meets the config limits, and the ready-made levels.

Run: python -m tools.levels grid            pages for a full read and steps of solver_policy over a config grid
     python -m tools.levels levels [--n 20]  the level table (also built into docs/RESULTS.md by tools.report)
     python -m tools.levels prefix           ledger pages the solver needs before its answer is final

Deterministic: fixed seeds, fixed grid, no wall-clock numbers. Pages are counted on the open
tables of the generated episode (evaluator side, nothing here is shown to a policy).
"""
from __future__ import annotations

import argparse
import statistics
import sys

from auditgym import levels as levels_module
from auditgym.config import interaction_issue_count
from auditgym.limits import INVOICES_PER_ERROR, MAX_PAGES, MAX_STEPS, pages
from policies.base import TABLES

GRID_SEEDS = 5
LEVEL_SEEDS = 20
SEED_BASE = 60_000
TABLE_POLICIES = ("solver_policy", "reconcile", "heuristic", "flag_all")
ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
MAX_GRID_ISSUES = 12
# (label, size.invoices, size.background_pairs, currencies, months, interaction). Errors: as many as
# the density limit allows (up to MAX_GRID_ISSUES), spread over the four types in turn.
GRID = [
    ("small", [24, 32], [30, 60], 1, 1, "none"),
    ("default", [64, 100], [150, 300], 1, 1, "none"),
    ("many-currencies", [64, 100], [150, 300], 3, 1, "none"),
    ("quarter", [64, 100], [150, 300], 3, 3, "none"),
    ("big-ledger", [64, 100], [500, 800], 1, 1, "none"),
    ("big-invoices", [160, 200], [150, 300], 1, 1, "I1"),
    ("max-ish", [200, 200], [440, 440], 3, 3, "I2"),
    ("max-documents", [200, 200], [0, 0], 3, 3, "none"),
]
CURRENCIES = ("USD", "EUR", "RUB")


def spread_errors(invoices_low: int, interaction: str) -> dict:
    room = min(invoices_low // INVOICES_PER_ERROR - interaction_issue_count(interaction), MAX_GRID_ISSUES)
    errors = dict.fromkeys(ERROR_TYPES, 0)
    for k in range(max(room, 0)):
        errors[ERROR_TYPES[k % len(ERROR_TYPES)]] += 1
    return errors


def grid_config(row: tuple) -> dict:
    _, invoices, background, currencies, months, interaction = row
    return {"errors": spread_errors(invoices[0], interaction), "currencies": list(CURRENCIES[:currencies]), "interaction": interaction,
            "period": {"start": "2026-07-01", "months": months},
            "size": {"invoices": list(invoices), "background_pairs": list(background)}}


def prefix_text(n: int = 8) -> str:
    """Evaluator-side analysis: how many ledger pages (in cursor order) the independent solver needs
    before its findings equal those of the full read. The ledger is ordered by row id, so the rows of
    one document are spread over all pages and no page can be skipped without knowing the answer."""
    from auditgym.generate import generate
    from auditgym.solver import solve

    def key(findings: list[dict]) -> list:
        return sorted((f["type"], tuple(f["affected_ids"])) for f in findings)

    lines = ["| Уровень | страниц книги | страниц до полного ответа (среднее, максимум) |", "|---|---:|---:|"]
    for level in levels_module.names():
        total, needed = [], []
        for k in range(n):
            public = generate(SEED_BASE + 2000 + k, levels_module.load(level))["public"]
            ledger, full = public["ledger"], key(solve(public))
            count = pages(len(ledger))
            total.append(count)
            needed.append(next(c for c in range(1, count + 1) if key(solve({**public, "ledger": ledger[:c * 50]})) == full))
        lines.append(f"| {level} | {statistics.fmean(total):.1f} | {statistics.fmean(needed):.1f}, {max(needed)} |")
    return "\n".join(lines)


def full_read_pages(public: dict) -> dict:
    out = {table: pages(len(public[table])) for table in TABLES}
    out["total"] = sum(out.values())
    return out


def run_config(config: dict, seeds: list[int], names=TABLE_POLICIES) -> dict:
    """Pages of a full read and, per policy, mean reward, F1, steps and truncated count."""
    from auditgym import Env
    from auditgym.findings import StructureError, parse_findings
    from auditgym.generate import generate
    from auditgym.matcher import match
    from policies.runner import make_policy, run_episode

    episodes = [generate(seed, config) for seed in seeds]
    out: dict = {"pages": [full_read_pages(e["public"])["total"] for e in episodes],
                 "ledger_rows": [len(e["public"]["ledger"]) for e in episodes], "policies": {}}
    for name in names:
        rows = []
        with Env(mode="train") as env:
            for seed, episode in zip(seeds, episodes):
                result = run_episode(env, make_policy(name), seed, config)
                rows.append({"reward": result["reward"], "f1": result["info"].get("f1", 0.0),
                             "steps": result["steps"], "truncated": result["truncated"]})
        out["policies"][name] = rows
    return out


def mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(row[key] for row in rows)


def grid_text() -> str:
    seeds = [SEED_BASE + k for k in range(GRID_SEEDS)]
    lines = [f"Budget {MAX_STEPS} steps, MAX_PAGES {MAX_PAGES}; {GRID_SEEDS} seeds per row.", "",
             "| Grid row | ledger rows | pages (min-max) | solver steps | truncated | solver reward |",
             "|---|---:|---:|---:|---:|---:|"]
    worst = 0
    for row in GRID:
        data = run_config(grid_config(row), seeds, names=("solver_policy",))
        rows = data["policies"]["solver_policy"]
        worst = max(worst, max(data["pages"]))
        lines.append(f"| {row[0]} | {statistics.fmean(data['ledger_rows']):.0f} | {min(data['pages'])}-{max(data['pages'])} | "
                     f"{mean(rows, 'steps'):.1f} | {sum(r['truncated'] for r in rows)}/{len(rows)} | {mean(rows, 'reward'):.3f} |")
    lines += ["", f"Most pages seen: {worst}; the full-read ceiling of the limits is {MAX_PAGES} pages, plus one submit "
                  f"= {MAX_PAGES + 1} steps, below {MAX_STEPS}."]
    return "\n".join(lines)


def level_stats(n: int = LEVEL_SEEDS) -> dict[str, dict]:
    seeds = [SEED_BASE + 1000 + k for k in range(n)]
    return {name: run_config(levels_module.load(name), seeds, names=TABLE_POLICIES)
            for name in levels_module.names()}


def levels_markdown(n: int = LEVEL_SEEDS, stats: dict | None = None) -> list[str]:
    stats = stats or level_stats(n)
    lines = ["| Уровень | Книга (проводок) | Страниц на полное чтение | Политика | F1 | Возврат | Шаги | Обрезано |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for level, data in stats.items():
        pages_text = f"{statistics.fmean(data['pages']):.1f} ({min(data['pages'])}-{max(data['pages'])})"
        for name, rows in data["policies"].items():
            lines.append(f"| {level} | {statistics.fmean(data['ledger_rows']):.0f} | {pages_text} | {name} | "
                         f"{mean(rows, 'f1'):.3f} | {mean(rows, 'reward'):.3f} | {mean(rows, 'steps'):.1f} | "
                         f"{sum(r['truncated'] for r in rows)}/{len(rows)} |")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("grid", "levels", "prefix"))
    parser.add_argument("--n", type=int, default=LEVEL_SEEDS)
    args = parser.parse_args(argv)
    text = {"grid": grid_text, "levels": lambda: "\n".join(levels_markdown(args.n)), "prefix": prefix_text}
    print(text[args.what]())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
