"""Fuzz the generator over a grid of configs: python -m tools.fuzz --seeds N [--types T ...].

For every episode: generate() returns (or raises ConfigError only), verify() is empty, and the
independent solver (auditgym.solver.solve on the open records) equals the truth: same type and
affected-id set, amount within the policy tolerance, same currency, and the finding's evidence
contains a full truth evidence set; no attempt before the accepted one crashed (`internal:`) or was
rejected by verify (`verify:`), only "no target" retries are normal. Prints failures by reason with the smallest failing seed, the
distribution of retry attempts with their reasons, and time per episode. Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict

from auditgym.config import DISTRACTOR_KINDS

TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
BAD_RETRIES = ("internal:", "verify:")  # retry reasons that count as failures
OFFSETS = ("-11:00", "-09:30", "+00:00", "+05:00", "+05:45", "+14:00", "-03:00", "+05:30", "-12:00", "+12:45")
CURRENCY_SETS = ((), ("USD",), ("USD", "EUR"), ("CNY", "RUB", "EUR"))
PERIODS = ({"start": "2026-10-01", "months": 1}, {"start": "2026-12-01", "months": 2},
           {"start": "2027-02-01", "months": 1}, {"start": "2028-02-01", "months": 1},
           {"start": "2026-11-01", "months": 3})
SIZES = ({"invoices": [10, 12], "background_pairs": [0, 0]}, {"invoices": [16, 24], "background_pairs": [0, 20]},
         {"invoices": [40, 100], "background_pairs": [150, 300]},
         {"invoices": [190, 200], "background_pairs": [380, 400]})


def _errors(*names: str, count: int = 1) -> dict:
    return {name: count for name in names}


def grid() -> list[tuple[str, dict]]:
    """(name, config): each type alone, all pairs, both interactions, clean with and without
    distractors (each pattern alone, as objects too), maximum density; sizes, currencies, offsets and periods cycle over their lists."""
    scenarios = [("clean", {}, "none", True), ("clean_plain", {}, "none", False)]
    scenarios += [(name, _errors(name), "none", True) for name in TYPES]
    scenarios += [(f"{a}+{b}", _errors(a, b), "none", True) for i, a in enumerate(TYPES) for b in TYPES[i + 1:]]
    scenarios += [("I1", {}, "I1", True), ("I2", {}, "I2", True), ("I2+cutoff", _errors("period_cutoff"), "I2", True)]
    scenarios += [("all_plain", _errors(*TYPES), "none", False)]
    # Each clean pattern switched on alone, all of them as an object, none as an object, and one
    # pattern next to every error type (docs/DECISIONS.md T).
    scenarios += [(f"toggle_{kind}", {}, "none", {kind: True}) for kind in DISTRACTOR_KINDS]
    scenarios += [("toggles_all", {}, "none", {kind: True for kind in DISTRACTOR_KINDS}), ("toggles_none", {}, "none", {})]
    scenarios += [("toggle_refunds+errors", _errors(*TYPES), "none", {"refunds": True, "split_payments": True})]
    out = []
    for index, (name, errors, interaction, distractors) in enumerate(scenarios):
        issues = sum(errors.values()) + (2 if interaction != "none" else 0)
        sizes = [s for s in SIZES if s["invoices"][0] >= 8 * issues]
        size = sizes[index % len(sizes)]
        currencies = CURRENCY_SETS[index % len(CURRENCY_SETS)]
        if not currencies and ("fx_error" in errors or interaction == "I1"):
            currencies = ("USD",)
        out.append((name, {"errors": errors, "interaction": interaction, "distractors": distractors,
                           "size": size, "currencies": list(currencies), "timezone": OFFSETS[index % len(OFFSETS)],
                           "period": PERIODS[index % len(PERIODS)],
                           "base_currency": "USD" if index % 7 == 3 and "USD" not in currencies else "KZT"}))
    # Maximum density: one issue per 8 invoices, all four types, foreign currencies.
    out.append(("max_density_small", {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1,
                                                 "period_cutoff": 1}, "size": {"invoices": [32, 40], "background_pairs": [0, 10]},
                                      "currencies": ["USD"], "timezone": "+14:00"}))
    # 12 issues (10 errors + I1) is the most one episode may hold (limits.MAX_ISSUES): 96 invoices.
    out.append(("max_density_large", {"errors": {"duplicate_invoice": 3, "allocation_gap": 3, "fx_error": 2,
                                                 "period_cutoff": 2}, "interaction": "I1",
                                      "size": {"invoices": [96, 100], "background_pairs": [400, 420]},
                                      "currencies": ["USD", "EUR", "CNY"], "timezone": "-11:00",
                                      "period": {"start": "2026-12-01", "months": 2}}))
    # The other variants of an allocation_gap (docs/DECISIONS.md S): fully missing, and a per-issue mix.
    for mode, count, currencies in (("missing", 1, ["USD"]), ("missing", 2, []), ("mixed", 3, ["USD", "EUR"])):
        out.append((f"gap_{mode}_x{count}", {"errors": {"allocation_gap": count}, "allocation_gap_mode": mode,
                                              "size": {"invoices": [40, 60], "background_pairs": [0, 40]},
                                              "currencies": currencies, "timezone": "+05:30"}))
    # The variants of an fx_error (docs/DECISIONS.md V): another date's rate, a perturbed rate, a per-issue mix.
    for mode, count, currencies in (("date", 2, ["USD"]), ("rate", 2, ["EUR"]), ("mixed", 3, ["USD", "CNY"])):
        out.append((f"fx_{mode}_x{count}", {"errors": {"fx_error": count}, "fx_error_mode": mode,
                                             "size": {"invoices": [40, 60], "background_pairs": [0, 40]},
                                             "currencies": currencies, "timezone": "-03:00"}))
    out.append(("fx_date_all", {"errors": _errors(*TYPES), "fx_error_mode": "date",
                                "size": {"invoices": [64, 80], "background_pairs": [50, 100]},
                                "currencies": ["USD"]}))
    out.append(("gap_missing_I1+all", {"errors": _errors(*TYPES), "allocation_gap_mode": "missing", "interaction": "I1",
                                       "size": {"invoices": [64, 80], "background_pairs": [50, 100]},
                                       "currencies": ["USD"]}))
    return out


def _covers(evidence: list, sets: list) -> bool:
    cited = set(evidence)
    return any(set(group) <= cited for group in sets)


def check_episode(seed: int, config: dict) -> tuple[list[str], dict | None]:
    """Failure reasons for one episode (empty list: fine) and the episode."""
    from auditgym.config import ConfigError
    from auditgym.generate import generate
    from auditgym.solver import solve
    from auditgym.verify import verify

    try:
        episode = generate(seed, config)
    except ConfigError as error:
        return [f"generate: ConfigError {error.code}"], None
    except Exception as error:  # noqa: BLE001 - reported, never hidden
        return [f"generate: {type(error).__name__}"], None
    reasons = [f"verify: {problem.split(':')[0]}: {problem.split(':', 2)[-1].strip()[:70]}" for problem in verify(episode)]
    public = episode["public"]
    tolerance = public["policy"]["amount_tolerance_minor"]
    try:
        findings = solve(public)
    except Exception as error:  # noqa: BLE001
        return reasons + [f"solver: crash {type(error).__name__}"], episode
    remaining = list(findings)
    for issue in episode["private"]["truth"]["issues"]:
        same = [f for f in remaining if f["type"] == issue["type"] and set(f["affected_ids"]) == set(issue["affected_ids"])]
        good = [f for f in same if abs(f["claimed_amount_minor"] - issue["expected_amount_minor"]) <= tolerance[issue["type"]]
                and f["currency"] == issue["currency"]]
        full = [f for f in good if _covers(f["evidence_ids"], issue["evidence_sets"])]
        if full:
            remaining.remove(full[0])
        elif good:
            remaining.remove(good[0])
            reasons.append(f"solver: evidence lacks a full set ({issue['type']})")
        elif same:
            remaining.remove(same[0])
            reasons.append(f"solver: amount or currency differs ({issue['type']})")
        else:
            reasons.append(f"solver: missed {issue['type']}")
    reasons += [f"solver: extra {finding['type']}" for finding in remaining]
    return reasons, episode


def run(seeds: int, types: tuple[str, ...] = (), out=sys.stdout) -> dict:
    configs = [(name, cfg) for name, cfg in grid()
               if not types or any(t in name or t in (cfg.get("errors") or {}) or t == cfg.get("interaction") for t in types)]
    failures: dict[str, list] = defaultdict(list)
    attempts, retry_reasons = Counter(), Counter()
    times = []
    for name, config in configs:
        for seed in range(seeds):
            started = time.perf_counter()
            reasons, episode = check_episode(seed, config)
            times.append(time.perf_counter() - started)
            for reason in sorted(set(reasons)):
                failures[reason].append((seed, name))
            if episode is not None:
                provenance = episode["private"]["provenance"]
                attempts[provenance["attempt"]] += 1
                # A retry is only for a story without a target. An attempt that crashed or that verify
                # rejected is a generator bug even when a later attempt succeeds.
                for retry in provenance["retries"]:
                    if retry["reason"].startswith(BAD_RETRIES):
                        failures[f"retry: {retry['reason'][:62]}"].append((seed, name))
                retry_reasons.update(r["reason"].split(":")[0] + (":" + r["reason"].split(":")[1][:50]
                                     if ":" in r["reason"] else "") for r in provenance["retries"])
    episodes = len(times)
    print(f"configs: {len(configs)}, seeds per config: {seeds}, episodes: {episodes}", file=out)
    if failures:
        print(f"\n{'failures by reason':<70}{'count':>7}  smallest seed (config)", file=out)
        for reason, hits in sorted(failures.items(), key=lambda item: (-len(item[1]), item[0])):
            seed, name = min(hits)
            print(f"{reason:<70}{len(hits):>7}  {seed} ({name})", file=out)
    else:
        print("\nno failures", file=out)
    print("\nattempts per episode: " + ", ".join(f"{a}: {n}" for a, n in sorted(attempts.items())), file=out)
    retried = sum(n for a, n in attempts.items() if a)
    print(f"episodes needing a retry: {retried} of {episodes} ({100 * retried / max(episodes, 1):.2f} %)", file=out)
    for reason, count in retry_reasons.most_common(8):
        print(f"  retry reason: {reason} x{count}", file=out)
    if times:
        ordered = sorted(times)
        print(f"time per episode (generate + verify + solve): mean {sum(times) / episodes * 1000:.1f} ms, "
              f"p95 {ordered[int(0.95 * (episodes - 1))] * 1000:.1f} ms, max {ordered[-1] * 1000:.1f} ms", file=out)
    return {"failures": dict(failures), "attempts": dict(attempts), "episodes": episodes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.fuzz", description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--types", nargs="*", default=[], help="error types or I1/I2/clean to keep in the grid")
    args = parser.parse_args(argv)
    result = run(args.seeds, tuple(args.types))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
