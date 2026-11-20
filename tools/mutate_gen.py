"""Hand-written mutation run of the generator and data checks (books, inject, generate, verify): python3.11 -m tools.mutate_gen [--jobs N] [--only ID ...] [--json PATH].

Every mutation is one small edit of the core (a boundary, a limit, a rounding mode, a dropped
condition, a flipped and/or, ...). The edit is applied to a COPY of the repository in a temporary
directory (the working tree is never touched) and the test suites relevant to that file run in the
copy with a timeout. The mutation is KILLED if any suite fails, errors, crashes or times out, and
SURVIVED if every suite is green.

Survivors must be explained: a mutation that cannot change behaviour carries `equivalent=<reason>`.
A survivor without a reason is a hole in the tests: the command exits with 1 and names it. A
mutation marked equivalent that is killed is reported too (the reason is wrong or stale).

Not part of the fast `python3 -m auditgym check`: a full run takes about 10 to 20 minutes.
Mutants of the judge (matcher, reward, findings) live in tools/mutants.py. Standard library only. The unmutated suites are run first; if they are not green nothing is reported.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "runs", "*.pyc")

# Test commands (module arguments of `python -m unittest -f`), cheapest and most likely to fail first.
BOOKS = (("tests.gen.test_mutation_holes", "tests.gen.test_verify_deviations", "tests.gen.test_golden", "tests.gen.test_inject",
          "tests.gen.test_generate", "tests.gen.test_checks_gen", "tests.gen.test_config_storage"),
         ("discover", "-s", "tests/judge", "-t", "."), ("discover", "-s", "tests/solver", "-t", "."),
         ("discover", "-s", "tests/env", "-t", "."), ("tests.test_fixture_tiny", "tests.test_limits"))


@dataclass(frozen=True)
class Mutation:
    id: str
    file: str
    old: str
    new: str
    what: str
    suites: tuple
    equivalent: str = ""  # non-empty: the edit cannot change behaviour, and why


def _m(id, file, old, new, what, equivalent=""):
    return Mutation(id, f"auditgym/{file}", old, new, what, BOOKS, equivalent)


MUTATIONS: list[Mutation] = [
    # --- money.py and books.py: rounding and postings
    _m("B01", "money.py", "return int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))\n\n\ndef closed_share",
       "return int(scaled.quantize(Decimal(1), rounding=__import__('decimal').ROUND_HALF_UP))\n\n\ndef closed_share",
       "fx conversion rounds half up instead of half even"),
    _m("B02", "money.py", "    return int(share.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))",
       "    return int(share.quantize(Decimal(1), rounding=__import__('decimal').ROUND_HALF_UP))",
       "proportional closing rounds half up instead of half even"),
    _m("B03", "books.py", "if amount > 0 and after > total:", "if amount > 0 and after >= total:",
       "overpayment boundary > -> >=",
       equivalent="at after == total the excess is 0: closes = booked - closed in both branches "
                  "(closed_share(booked, total, total) == booked) and unallocated = to_base(0) = 0"),
    _m("B04", "books.py", 'closes[ref] = booked(world, invoices[ref]) - closed[ref]',
       'closes[ref] = booked(world, invoices[ref])', "a multi-invoice payment ignores what is already closed"),
    _m("B05", "books.py", 'closes[ref] -= payment["gap_minor"]', 'closes[ref] += payment["gap_minor"]',
       "allocation gap moves the wrong way"),
    _m("B06", "books.py", 'difference = part["cash"] - sum(part["closes"].values()) - part["unallocated"]',
       'difference = part["cash"] - sum(part["closes"].values())', "fx_difference ignores unallocated"),
    _m("B07", "books.py", "def _side(amount: int) -> tuple[int, int]:\n    return (amount, 0) if amount > 0 else (0, -amount)",
       "def _side(amount: int) -> tuple[int, int]:\n    return (amount, 0) if amount >= 0 else (0, -amount)",
       "_side(0) boundary > -> >=", equivalent="_side(0) is (0, 0) with either branch"),
    _m("B08", "books.py", 'date = moment.astimezone(world["zone"]).date().isoformat()',
       "date = moment.astimezone(dt.timezone.utc).date().isoformat()",
       "policy rate date taken in UTC instead of the company timezone"),
    _m("B09", "books.py", "closes[ref] = value - closed[ref]\n                unallocated = to_base(world, after - total",
       "closes[ref] = value\n                unallocated = to_base(world, after - total",
       "an overpayment closes the whole booked amount even after earlier closings"),
    # --- inject.py: operators and their rules
    _m("I01", "inject.py", "FX_MIN_MINOR = 100", "FX_MIN_MINOR = 1", "minimum fx difference 100 -> 1 minor unit"),
    _m("I02", "inject.py", "return difference >= FX_MIN_MINOR and", "return difference > FX_MIN_MINOR and",
       "fx difference boundary >= -> >"),
    _m("I03", "inject.py", "return invoice[\"posted_at\"] is None and not in_period(world, invoice[\"issued_at\"])",
       "return invoice[\"posted_at\"] is None or not in_period(world, invoice[\"issued_at\"])",
       "and -> or in the cutoff target test"),
    _m("I04", "inject.py", "                  and not any(p[\"amount_minor\"] < 0 for p in payments_of(world, key))]\n    if not candidates:",
       "                  ]\n    if not candidates:", "refunded invoices may be duplicated"),
    _m("I05", "inject.py", "or record[\"invoice_refs\"] is None", "or False", "an advance payment may carry an fx_error"),
    _m("I06", "inject.py", "if closing < 2:\n            continue\n        gap = _gap(rng, world, payment, closing)",
       "if closing < 1:\n            continue\n        gap = _gap(rng, world, payment, closing)",
       "allocation_gap on a closing of 1 minor unit"),
    _m("I07", "inject.py", "return max(1, min(closing - 1, share))", "return max(1, min(closing, share))",
       "the gap may take the whole receivable credit"),
    _m("I08", "inject.py", 'or refs[0] in used or payment["rate_override"] is not None):',
       'or refs[0] in used):', "allocation_gap on a payment that already carries an fx_error"),
    _m("I09", "inject.py", 'or [p["key"] for p in payments_of(world, invoice["key"])] != [key]):',
       "):", "allocation_gap on an invoice with other payments"),
    _m("I10", "inject.py", "WRONG_DATE_DAYS = 5", "WRONG_DATE_DAYS = 6", "wrong-date window 5 -> 6 days"),
    # --- generate.py: episode assembly and policy
    _m("G01", "generate.py", "if len(ledger) > MAX_LEDGER_ROWS:", "if len(ledger) >= MAX_LEDGER_ROWS:",
       "ledger size boundary > -> >="),
    _m("G02", "generate.py", '"fx_error": 1, "period_cutoff": 0}', '"fx_error": 2, "period_cutoff": 0}',
       "published fx tolerance 1 -> 2"),
    _m("G03", "generate.py", '"evidence_superset_allowed": True,', '"evidence_superset_allowed": False,',
       "published evidence_superset_allowed flipped"),
    _m("G04", "generate.py", 'if name.endswith("_id") and isinstance(value, str)',
       'if name.endswith("_idx") and isinstance(value, str)', "an internal id in provenance params is not translated"),
    _m("G05", "generate.py", "    problems = verify(episode)\n    if problems:", "    problems = []\n    if problems:",
       "generate() no longer verifies its own episode"),
    _m("G06", "generate.py", "hexdigest()[:12]", "hexdigest()[:11]", "episode id one character shorter"),
    _m("G07", "generate.py", "return docs + rows + sorted(key[3:] for key in group if key.startswith(\"FX:\"))",
       "return rows + docs + sorted(key[3:] for key in group if key.startswith(\"FX:\"))",
       "evidence order: rows before documents"),
    # --- verify.py: books, evidence, truth
    _m("V01", "verify.py", "if sum(r[\"debit_minor\"] for r in books.ledger) != sum(r[\"credit_minor\"] for r in books.ledger):",
       "if False:", "whole-book debit = credit check skipped",
       equivalent="every group is checked to balance on its own right after it, and the whole book is the "
                  "disjoint union of the groups, so total balance follows"),
    _m("V02", "verify.py", "        if balance:\n            problems.append(f\"{name}: group {key} is unbalanced by {balance}\")",
       "        if False:\n            problems.append(f\"{name}: group {key} is unbalanced by {balance}\")",
       "group balance check skipped (debit = credit per event group)"),
    _m("V03", "verify.py", "or (debit > 0) == (credit > 0):", "or (debit > 0) and (credit > 0):",
       "a row with zero on both sides is accepted"),
    _m("V04", "verify.py", "if deviation > tolerance:", "if deviation >= tolerance:", "fx invoice predicate > -> >="),
    _m("V05", "verify.py", "if abs(cash - correct) > tolerance:", "if abs(cash - correct) >= tolerance:",
       "fx payment predicate > -> >="),
    _m("V06", "verify.py", "if len(group) > 2:", "if len(group) > 3:", "undefined case for 3 invoices with a signature -> 4"),
    _m("V07", "verify.py", "if not group or len(group) > MAX_EVIDENCE or len(set(group)) != len(group):",
       "if not group and len(group) > MAX_EVIDENCE or len(set(group)) != len(group):",
       "or -> and in the evidence-set shape check"),
    _m("V08", "verify.py", "if not group or len(group) > MAX_EVIDENCE or len(set(group)) != len(group):",
       "if not group or len(group) >= MAX_EVIDENCE or len(set(group)) != len(group):",
       "evidence set size boundary > -> >="),
    _m("V09", "verify.py", "            if not set(group) <= open_ids:\n                problems.append(f\"{tag}: evidence: an id is not in the open records\")",
       "            if False:\n                problems.append(f\"{tag}: evidence: an id is not in the open records\")",
       "evidence ids are not checked against the open records"),
    _m("V10", "verify.py", "or issue[\"expected_amount_minor\"] < FX_MIN_MINOR", "or issue[\"expected_amount_minor\"] <= FX_MIN_MINOR",
       "fx minimum boundary < -> <="),
    _m("V11", "verify.py", "if abs(moment - edge) >= dt.timedelta(days=CUTOFF_MAX_DAYS):",
       "if abs(moment - edge) > dt.timedelta(days=CUTOFF_MAX_DAYS):", "cutoff distance boundary >= -> >"),
    _m("V12", "verify.py", 'if private["truth"]["episode_id"] != public["manifest"]["episode_id"]:', "if False:",
       "truth episode_id is not compared with the manifest"),
    _m("V13", "verify.py", "            if record[ident[table]] in state[table]:\n                problems.append(f\"provenance: step {step['issue_id']} adds an id that already exists\")",
       "            if False:\n                problems.append(f\"provenance: step {step['issue_id']} adds an id that already exists\")",
       "replay does not notice an added id that already exists"),
    _m("V14", "verify.py", "if parse_time(invoice[\"due_at\"]) <= parse_time(invoice[\"issued_at\"]):",
       "if parse_time(invoice[\"due_at\"]) < parse_time(invoice[\"issued_at\"]):", "due_at boundary <= -> <"),
    _m("V15", "verify.py", 'or item["quantity"] <= 0 or item["quantity"]', 'or item["quantity"] < 0 or item["quantity"]',
       "a line item with quantity 0 is accepted"),
    _m("V16", "verify.py", "FX_MARGIN_DAYS = 5", "FX_MARGIN_DAYS = 4", "fx margin 5 -> 4 days"),
    _m("V17", "verify.py", "MIN_RATE_DIGITS = 6", "MIN_RATE_DIGITS = 5", "minimum rate digits 6 -> 5"),
    _m("V18", "verify.py", "if len(by_account[acc[\"unallocated\"]]) > 1 or unallocated != expected[\"unallocated\"]:",
       "if unallocated != expected[\"unallocated\"]:", "several unallocated rows are accepted on a clean payment"),
]


def _check_table() -> None:
    ids = [m.id for m in MUTATIONS]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate mutation id")
    for m in MUTATIONS:
        text = (ROOT / m.file).read_text(encoding="utf-8")
        if text.count(m.old) != 1:
            raise SystemExit(f"{m.id}: the original snippet occurs {text.count(m.old)} times in {m.file}, need 1")
        if m.old == m.new:
            raise SystemExit(f"{m.id}: the mutation changes nothing")


def _prepare(directory: Path) -> Path:
    shutil.copytree(ROOT, directory, ignore=COPY_IGNORE)
    return directory


def _run_suites(directory: Path, suites: tuple, timeout: int) -> tuple[bool, str]:
    """(green, why): every suite command must exit 0 within the timeout (the total budget is shared)."""
    deadline = time.monotonic() + timeout
    for suite in suites:
        flags = ["-f", "-q"]
        args = [sys.executable, "-m", "unittest", *suite, *flags] if suite[0] == "discover" \
            else [sys.executable, "-m", "unittest", *flags, *suite]
        left = deadline - time.monotonic()
        if left <= 0:
            return False, "timeout"
        try:
            done = subprocess.run(args, cwd=directory, capture_output=True, text=True, timeout=left,
                                  env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if done.returncode != 0:
            lines = [line for line in done.stderr.splitlines() if line.startswith(("FAIL:", "ERROR:"))]
            return False, (lines[0] if lines else f"exit {done.returncode}")[:110]
    return True, ""


def _apply(directory: Path, m: Mutation) -> None:
    path = directory / m.file
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(m.old, m.new, 1), encoding="utf-8")


def run_one(m: Mutation, scratch: Path, timeout: int) -> dict:
    started = time.monotonic()
    directory = _prepare(scratch / m.id)
    try:
        _apply(directory, m)
        green, why = _run_suites(directory, m.suites, timeout)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    return {"id": m.id, "file": m.file, "what": m.what, "status": "survived" if green else "killed",
            "by": why, "equivalent": m.equivalent, "seconds": round(time.monotonic() - started)}


def baseline(scratch: Path, timeout: int) -> None:
    """The unmutated copy must be green on every suite, otherwise 'killed' means nothing."""
    directory = _prepare(scratch / "baseline")
    try:
        for suites in (BOOKS,):
            green, why = _run_suites(directory, suites, timeout)
            if not green:
                raise SystemExit(f"the unmutated code fails its tests ({why}); fix that first")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m tools.mutate_gen", description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=6, help="mutations run in parallel (default 6)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per mutation (default 900)")
    parser.add_argument("--only", nargs="*", default=[], help="run only these mutation ids")
    parser.add_argument("--json", default="", help="write the result table to this file")
    parser.add_argument("--scratch", default="", help="directory for the copies (default: a temporary one)")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args(argv)
    _check_table()
    chosen = [m for m in MUTATIONS if not args.only or m.id in args.only]
    base = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="auditgym-mutate-"))
    base.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not args.skip_baseline:
        baseline(base, args.timeout)
        print(f"unmutated suites are green ({time.monotonic() - started:.0f} s)", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(run_one, m, base, args.timeout) for m in chosen]
        for future in futures:
            result = future.result()
            results.append(result)
            print(f"{result['id']} {result['status']:<8} {result['seconds']:>4} s  {result['what']}"
                  + (f"  [{result['by']}]" if result["status"] == "killed" else ""), flush=True)
    killed = [r for r in results if r["status"] == "killed"]
    survived = [r for r in results if r["status"] == "survived"]
    equivalent = [r for r in survived if r["equivalent"]]
    holes = [r for r in survived if not r["equivalent"]]
    stale = [r for r in killed if r["equivalent"]]
    print(f"\nmutations: {len(results)}, killed: {len(killed)}, survived: {len(survived)} "
          f"(equivalent: {len(equivalent)}, unexplained: {len(holes)}), time {time.monotonic() - started:.0f} s")
    for r in equivalent:
        print(f"  equivalent {r['id']}: {r['equivalent']}")
    for r in holes:
        print(f"  SURVIVED {r['id']} ({r['file']}): {r['what']}")
    for r in stale:
        print(f"  KILLED BUT MARKED EQUIVALENT {r['id']}: fix the table")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(base, ignore_errors=True) if not args.scratch else None
    return 1 if holes or stale else 0


if __name__ == "__main__":
    sys.exit(main())
