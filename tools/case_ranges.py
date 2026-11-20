"""Check that what we ship and what we generate by default stays inside the sizes the case proposes.

Run: python -m tools.case_ranges [--seeds 20]

docs/CASE.md section 03: "Proposed size: 200-2 000 postings, 20-200 documents, 0-12 errors".
Definitions (counted on the open files of a generated episode, the way a reader of the episode
directory counts them):
  postings  = data rows of ledger.csv (len(public["ledger"]), manifest counts.ledger)
  documents = records of invoices.jsonl plus records of payments.jsonl (every record has a doc_id and
              can be opened with get_document; duplicates and background invoices count, fx rates do not)
  errors    = records of the answer key (len(truth["issues"])), the 0-12 the agent has to find
Checked: the default config (reset without config, reset with {}), the 20 shipped episodes
(configs/episodes.json) and every level (configs/levels.json), each over SEEDS seeds. Prints a table
and PASS or FAIL per row; exit code 1 on any FAIL.

An arbitrary config is NOT limited to these ranges: config.py allows up to 2 000 postings, 200 invoices
(plus payments) and 30 issues, and the jury may pass its own configs. Leaving the case ranges is
deliberate for a caller, and never happens by default, in an episode or in a level we ship.
"""
from __future__ import annotations

import argparse
import json
import sys

POSTINGS = (200, 2000)
DOCUMENTS = (20, 200)
ERRORS = (0, 12)
SEEDS = 20
SEED_BASE = 71_000


def measure(episode: dict) -> dict:
    """Sizes of one generated episode: {"postings", "documents", "errors"} (see the module docstring)."""
    public = episode["public"]
    return {"postings": len(public["ledger"]),
            "documents": len(public["invoices"]) + len(public["payments"]),
            "errors": len(episode["private"]["truth"]["issues"])}


def in_ranges(sizes: dict) -> bool:
    return (POSTINGS[0] <= sizes["postings"] <= POSTINGS[1] and DOCUMENTS[0] <= sizes["documents"] <= DOCUMENTS[1]
            and ERRORS[0] <= sizes["errors"] <= ERRORS[1])


def _subjects(seeds_n: int = SEEDS) -> list[tuple[str, list[tuple[int, object]]]]:
    """(label, [(seed, config)]) for the default, each shipped episode and each level."""
    from auditgym import levels
    from tools import episodes

    seeds = [SEED_BASE + k for k in range(seeds_n)]
    out: list[tuple[str, list[tuple[int, object]]]] = [
        ("default (config omitted)", [(s, None) for s in seeds]),
        ("default ({})", [(s, {}) for s in seeds])]
    for entry in episodes.plan(quiet=True):
        out.append((f"{entry['split']}/{entry['name']}", [(entry["seed"], entry["config"])]))
    for name in levels.names():
        out.append((f"level {name}", [(s, levels.load(name)) for s in seeds]))
    return out


def run(seeds: int = SEEDS) -> list[dict]:
    from auditgym.generate import generate

    rows = []
    for label, cases in _subjects(seeds):
        sizes = [measure(generate(seed, config)) for seed, config in cases]
        row = {"label": label, "n": len(sizes), "ok": all(in_ranges(s) for s in sizes)}
        for key in ("postings", "documents", "errors"):
            values = [s[key] for s in sizes]
            row[key] = (min(values), max(values))
        rows.append(row)
    return rows


def _span(pair: tuple[int, int]) -> str:
    return str(pair[0]) if pair[0] == pair[1] else f"{pair[0]}-{pair[1]}"


def table(rows: list[dict]) -> str:
    lines = [f"Case ranges: postings {POSTINGS[0]}-{POSTINGS[1]}, documents {DOCUMENTS[0]}-{DOCUMENTS[1]}, "
             f"errors {ERRORS[0]}-{ERRORS[1]}.",
             f"{'subject':26} {'n':>3} {'postings':>11} {'documents':>11} {'errors':>7}  result"]
    for row in rows:
        lines.append(f"{row['label']:26} {row['n']:>3} {_span(row['postings']):>11} {_span(row['documents']):>11} "
                     f"{_span(row['errors']):>7}  {'PASS' if row['ok'] else 'FAIL'}")
    return "\n".join(lines)


def summary(rows: list[dict]) -> str:
    bad = [row["label"] for row in rows if not row["ok"]]
    return (f"case ranges: PASS ({len(rows)} subjects)" if not bad
            else f"case ranges: FAIL ({len(bad)} of {len(rows)}: {', '.join(bad)})")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=SEEDS, help="seeds per default/level subject")
    args = parser.parse_args(argv)
    rows = run(args.seeds)
    print(table(rows))
    print(summary(rows))
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
