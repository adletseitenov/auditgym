"""Generator checks for tools/score.py, criterion "Данные и эталон".

CHECKS = [(name, function)]; each function returns None on success or a failure message.

The first two checks do not trust the package: each episode is written to disk with
auditgym.storage, read back as plain CSV/JSON and recomputed here with the standard library only
(no verify, books, inject, money): book and group balance, manifest counts, every declared amount
from the published definitions and policy formula, and every evidence id (docs/AUDIT.md Q5).
generate() already rejects what verify() finds, so checks that only call verify() on generated
episodes cannot fail; these can. The rest call auditgym.verify for the remaining properties.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import tempfile
from collections import defaultdict
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Callable, Optional

SEEDS = range(25)
ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
SMALL = {"invoices": [32, 48], "background_pairs": [0, 40]}
CONFIGS = [
    {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1}, "currencies": ["USD"]},
    {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
     "currencies": ["USD", "EUR"], "timezone": "-11:00", "size": SMALL},
    {"errors": {"fx_error": 2, "period_cutoff": 2}, "currencies": ["CNY"], "timezone": "+14:00", "size": SMALL},
    {"errors": {"duplicate_invoice": 1, "allocation_gap": 1}, "currencies": ["RUB"], "base_currency": "USD",
     "timezone": "+05:45", "size": SMALL, "distractors": False},
]
INTERACTION_CONFIGS = [
    {"errors": {}, "interaction": "I1", "currencies": ["USD"], "size": SMALL},
    {"errors": {"fx_error": 1}, "interaction": "I1", "currencies": ["EUR", "CNY"], "timezone": "-09:30", "size": SMALL},
    {"errors": {}, "interaction": "I2", "currencies": [], "size": SMALL},
    {"errors": {"duplicate_invoice": 1}, "interaction": "I2", "currencies": ["USD"], "timezone": "+14:00", "size": SMALL},
]


def _first_problem(match: Callable[[str], bool], configs=CONFIGS, seeds=SEEDS) -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.verify import verify

    for index, config in enumerate(configs):
        for seed in seeds:
            problems = [p for p in verify(generate(seed, config)) if match(p)]
            if problems:
                return f"config {index}, seed {seed}: {problems[0]}"
    return None


def _read(root: pathlib.Path) -> dict:
    def ledger(name: str) -> list[dict]:
        with open(root / name, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["debit_minor"], row["credit_minor"] = int(row["debit_minor"]), int(row["credit_minor"])
            for key in ("entity_id", "invoice_id", "payment_id"):
                row[key] = row[key] or None
        return rows

    def lines(name: str) -> list[dict]:
        return [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines() if line]

    with open(root / "public/fx.csv", encoding="utf-8", newline="") as handle:
        fx = list(csv.DictReader(handle))
    return {"manifest": json.loads((root / "public/manifest.json").read_text(encoding="utf-8")),
            "policy": json.loads((root / "public/policy.json").read_text(encoding="utf-8")),
            "truth": json.loads((root / "private/truth.json").read_text(encoding="utf-8")),
            "ledger": ledger("public/ledger.csv"), "pre_ledger": ledger("private/pre_ledger.csv"),
            "invoices": lines("public/invoices.jsonl"), "payments": lines("public/payments.jsonl"), "fx": fx}


def _balance_problems(rows: list[dict], name: str) -> list[str]:
    problems = []
    if sum(r["debit_minor"] for r in rows) != sum(r["credit_minor"] for r in rows):
        problems.append(f"{name}: debits and credits differ")
    groups: dict = defaultdict(int)
    for r in rows:
        key = r["payment_id"] or (r["invoice_id"] and "i" + r["invoice_id"]) or (r["timestamp"], r["entity_id"])
        groups[key] += r["debit_minor"] - r["credit_minor"]
    if any(groups.values()):
        problems.append(f"{name}: an event group is unbalanced")
    return problems


def _recompute(files: dict) -> list[str]:
    """Problems found by recomputing an episode from its files alone."""
    policy, ledger = files["policy"], files["ledger"]
    problems = _balance_problems(ledger, "ledger") + _balance_problems(files["pre_ledger"], "pre_ledger")
    for table in ("ledger", "invoices", "payments", "fx"):
        if files["manifest"]["counts"][table] != len(files[table]):
            problems.append(f"manifest: count of {table} differs from the file")
    base, exponents = policy["base_currency"], policy["currency_exponents"]
    sign = -1 if policy["timezone"][0] == "-" else 1
    zone = dt.timezone(sign * dt.timedelta(hours=int(policy["timezone"][1:3]), minutes=int(policy["timezone"][4:6])))
    start, end = (dt.datetime.fromisoformat(policy[key]) for key in ("period_start", "period_end"))
    rates = {f"{r['date']}|{r['source_currency']}|{r['target_currency']}": Decimal(r["rate"]) for r in files["fx"]}
    docs = {d["doc_id"]: d for d in files["invoices"] + files["payments"]}
    own = defaultdict(list)
    for row in ledger:
        if row["payment_id"]:
            own["p" + row["payment_id"]].append(row)
        elif row["invoice_id"]:
            own["i" + row["invoice_id"]].append(row)
    known = set(docs) | {r["row_id"] for r in ledger} | set(rates)

    def rows_of(doc: dict) -> list[dict]:
        return own["i" + doc["invoice_id"]] if "line_items" in doc else own["p" + doc["payment_id"]]

    def correct(doc: dict) -> int:
        amount, moment = doc.get("total_minor", doc.get("amount_minor")), doc.get("issued_at", doc.get("timestamp"))
        date = dt.datetime.fromisoformat(moment).astimezone(zone).date().isoformat()
        scaled = Decimal(amount) * Decimal(10) ** (exponents[base] - exponents[doc["currency"]])
        return int((scaled * rates[f"{date}|{doc['currency']}|{base}"]).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))

    for issue in files["truth"]["issues"]:
        kind, affected = issue["type"], [docs[d] for d in issue["affected_ids"]]
        if kind in ("duplicate_invoice", "period_cutoff"):
            measured, currency = affected[0]["total_minor"], affected[0]["currency"]
            if kind == "duplicate_invoice":
                a, b = affected
                same = all(a[k] == b[k] for k in ("entity_id", "currency", "total_minor")) and \
                    sorted(map(json.dumps, a["line_items"])) == sorted(map(json.dumps, b["line_items"]))
                if not (same and rows_of(a) and rows_of(b)):
                    problems.append(f"{kind}: the two invoices are not both posted with one signature")
            elif not rows_of(affected[0]) or start <= dt.datetime.fromisoformat(affected[0]["issued_at"]) < end:
                problems.append(f"{kind}: the document is not dated outside the period and posted")
        elif kind == "allocation_gap":
            payment = next(d for d in affected if "payment_id" in d)
            measured = sum(r["credit_minor"] for r in rows_of(payment) if r["account"] == "unallocated")
            currency = base
        else:
            doc = affected[0]
            if "line_items" in doc:
                values = [r["debit_minor"] + r["credit_minor"] for r in rows_of(doc) if r["account"] in ("receivable", "revenue")]
            else:
                values = [sum(r["debit_minor"] - r["credit_minor"] for r in rows_of(doc) if r["account"] == "cash")]
            measured, currency = max(abs(v - correct(doc)) for v in values), base
        tolerance = policy["amount_tolerance_minor"][kind]
        if abs(measured - issue["expected_amount_minor"]) > tolerance or currency != issue["currency"]:
            problems.append(f"{kind}: recomputed {measured} {currency}, declared {issue['expected_amount_minor']} "
                            f"{issue['currency']}")
        for group in issue["evidence_sets"]:
            if not group or len(set(group)) > 10 or not set(group) <= known or not set(issue["affected_ids"]) <= set(group):
                problems.append(f"{kind}: an evidence set is empty, too long, cites an unknown id or lacks a document")
            elif not _sufficient(kind, affected, set(group) - set(issue["affected_ids"]), rows_of, rates, base, zone):
                problems.append(f"{kind}: an evidence set does not hold what docs/CONTRACTS.md 3.1 lists")
    return problems


def _sufficient(kind: str, affected: list, extra: set, rows_of, rates: dict, base: str, zone) -> bool:
    """The non-document ids of one evidence set, by CONTRACTS 3.1 (after docs/AUDIT.md D1)."""
    def ids(doc: dict, account: str, side: str, invoice_id=None) -> set:
        return {r["row_id"] for r in rows_of(doc) if r["account"] == account and r[side] > 0
                and (invoice_id is None or r["invoice_id"] == invoice_id)}

    if kind == "duplicate_invoice":
        return len(extra) == 2 and all(len(extra & {r["row_id"] for r in rows_of(doc)}) == 1 for doc in affected)
    if kind == "period_cutoff":
        return len(extra) == 1 and extra <= {r["row_id"] for r in rows_of(affected[0])}
    if kind == "allocation_gap":
        payment = next(d for d in affected if "payment_id" in d)
        invoice = next(d for d in affected if "line_items" in d)
        want = ids(payment, "unallocated", "credit_minor") | ids(invoice, "receivable", "debit_minor") \
            | ids(payment, "receivable", "credit_minor", invoice["invoice_id"])
        return len(want) == 3 and extra == want
    doc = affected[0]
    moment = doc.get("issued_at", doc.get("timestamp"))
    key = f"{dt.datetime.fromisoformat(moment).astimezone(zone).date().isoformat()}|{doc['currency']}|{base}"
    conversion = ids(doc, "cash", "debit_minor") | ids(doc, "cash", "credit_minor") if "payment_id" in doc else \
        ids(doc, "receivable", "debit_minor") | ids(doc, "revenue", "credit_minor")
    return key in rates and len(extra) == 2 and key in extra and len(extra & conversion) == 1


def _first_file_problem(configs=None, seeds=SEEDS, problem=lambda p: True) -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.storage import write_episode

    with tempfile.TemporaryDirectory() as directory:
        for index, config in enumerate(configs or CONFIGS + INTERACTION_CONFIGS):
            for seed in seeds:
                root = pathlib.Path(directory) / f"c{index}s{seed}"
                write_episode(generate(seed, config), root)
                problems = [p for p in _recompute(_read(root)) if problem(p)]
                if problems:
                    return f"config {index}, seed {seed}: {problems[0]}"
    return None


def recomputed_from_files_deviation_equals_declared() -> Optional[str]:
    """200 episodes (8 configs x 25 seeds), recomputed from their files."""
    return _first_file_problem(problem=lambda p: ": recomputed" in p or "not both posted" in p
                               or "not dated outside" in p)


def recomputed_from_files_evidence_suffices() -> Optional[str]:
    """200 episodes (8 configs x 25 seeds): every evidence id exists in the files, a set has at most
    10 ids, and it holds what CONTRACTS 3.1 lists, recomputed from the files."""
    return _first_file_problem(problem=lambda p: "evidence set" in p)


def recomputed_from_files_books_balance() -> Optional[str]:
    """Clean and dense companies: balance and counts recomputed from the files (no truth to lean on)."""
    return _first_file_problem([{"errors": {}}, {"errors": {}, "currencies": ["USD", "EUR", "CNY"]},
                                {"errors": {"duplicate_invoice": 6, "allocation_gap": 6, "fx_error": 6,
                                            "period_cutoff": 6}, "currencies": ["USD", "EUR"]}], range(10))


def each_error_type_configurable_alone() -> Optional[str]:
    from auditgym.generate import generate
    from auditgym.verify import verify

    for name in ERROR_TYPES:
        for count in (1, 3):
            config = {"errors": {name: count}, "currencies": ["USD"]}
            episode = generate(7, config)
            issues = episode["private"]["truth"]["issues"]
            if len(issues) != count or any(issue["type"] != name for issue in issues):
                return f"{name} x{count}: truth has {len(issues)} issues of other types or counts"
            problems = verify(episode)
            if problems:
                return f"{name} x{count}: {problems[0]}"
    return None


def any_seed_and_valid_config_gives_valid_episode() -> Optional[str]:
    from auditgym.config import ConfigError, normalize_config

    configs = CONFIGS + INTERACTION_CONFIGS + [
        None, {},
        {"errors": {"period_cutoff": 1}, "currencies": [], "size": {"invoices": [10, 10], "background_pairs": [0, 5]}},
        {"errors": {}, "currencies": ["RUB"], "base_currency": "USD"},
        {"errors": {"duplicate_invoice": 1}, "period": {"start": "2027-02-01", "months": 3}},
    ]
    problem = _first_problem(lambda p: True, configs, range(8))
    if problem:
        return problem
    for bad in ({"seed": 1}, {"errors": {"fx_error": "1"}}, {"timezone": "+05:00\n"}, {"currencies": [["USD"]]},
                {"period": {"start": "1999-01-01", "months": 1}}, [], {"errors": {"fx_error": 31}}):
        try:
            normalize_config(bad)
        except ConfigError:
            continue
        except Exception as error:  # noqa: BLE001
            return f"invalid config {bad!r} raised {type(error).__name__}"
        return f"invalid config {bad!r} was accepted"
    return None


def distractors_satisfy_no_definition() -> Optional[str]:
    """Clean companies full of distractors: no predicate fires; every distractor kind occurs."""
    from auditgym.generate import generate
    from auditgym.verify import predicate_issues, verify

    seen = {"refund": 0, "advance": 0, "overpayment": 0, "partial": 0, "outside": 0}
    for seed in range(30):
        config = {"errors": {}, "currencies": [["USD"], ["EUR", "CNY"], []][seed % 3],
                  "timezone": ["+05:00", "-11:00", "+14:00"][seed % 3]}
        episode = generate(seed, config)
        found, undefined = predicate_issues(episode["public"])
        if found or undefined or verify(episode):
            return f"seed {seed}: {(found or undefined or verify(episode))[0]}"
        public = episode["public"]
        invoices = {i["invoice_id"]: i for i in public["invoices"]}
        for payment in public["payments"]:
            refs = payment["invoice_refs"]
            seen["refund"] += payment["amount_minor"] < 0
            seen["advance"] += refs is None
            if refs and len(refs) == 1 and payment["amount_minor"] > 0:
                total = invoices[refs[0]]["total_minor"]
                seen["overpayment"] += payment["amount_minor"] > total
                seen["partial"] += payment["amount_minor"] < total
        posted = {r["invoice_id"] for r in public["ledger"] if r["invoice_id"]}
        seen["outside"] += sum(i["invoice_id"] not in posted for i in public["invoices"])
    missing = [kind for kind, count in seen.items() if not count]
    return f"distractors never generated: {', '.join(missing)}" if missing else None


def interactions_are_unambiguous() -> Optional[str]:
    """I1 and I2 give exactly their two issues on top of config.errors and nothing else holds."""
    from auditgym.generate import generate
    from auditgym.verify import verify

    expected = {"I1": ("allocation_gap", "fx_error"), "I2": ("duplicate_invoice", "period_cutoff")}
    for index, config in enumerate(INTERACTION_CONFIGS):
        for seed in SEEDS:
            episode = generate(seed, config)
            problems = verify(episode)
            if problems:
                return f"config {index}, seed {seed}: {problems[0]}"
            steps = [s for s in episode["private"]["provenance"]["steps"] if s["operator"] == config["interaction"]]
            issues = {i["issue_id"]: i for i in episode["private"]["truth"]["issues"]}
            kinds = tuple(sorted(issues[s["issue_id"]]["type"] for s in steps))
            if kinds != expected[config["interaction"]]:
                return f"config {index}, seed {seed}: scenario issues {kinds}"
            docs = [set(issues[s["issue_id"]]["affected_ids"]) for s in steps]
            if not docs[0] & docs[1]:
                return f"config {index}, seed {seed}: the two scenario issues share no document"
    return None


def fuzz_solver_equals_truth() -> Optional[str]:
    """tools.fuzz on 4 seeds per grid config: verify is empty and the independent solver equals the truth."""
    import io

    from tools import fuzz

    result = fuzz.run(4, out=io.StringIO())
    reasons = sorted(result["failures"])
    if not reasons:
        return None
    seed, name = min(result["failures"][reasons[0]])
    return f"{reasons[0]} (seed {seed}, {name})"


CHECKS: list[tuple[str, Callable[[], Optional[str]]]] = [
    # The first two names are planned in tools/score.py; both now recompute from the files of 200 episodes.
    ("after injection the deviation equals expected_amount_minor of each issue",
     recomputed_from_files_deviation_equals_declared),
    ("every evidence set exists, has at most 10 ids and is sufficient", recomputed_from_files_evidence_suffices),
    ("recomputed from the files of 30 clean and dense episodes: books, groups and counts balance",
     recomputed_from_files_books_balance),
    ("each of the four error types can be configured on its own", each_error_type_configurable_alone),
    ("any seed and any valid config gives a valid episode", any_seed_and_valid_config_gives_valid_episode),
    ("distractors do not satisfy any error definition", distractors_satisfy_no_definition),
    ("both interaction scenarios give an unambiguous truth", interactions_are_unambiguous),
    ("fuzz grid (18 configs x 4 seeds): verify is empty and the solver equals the truth", fuzz_solver_equals_truth),
]


if __name__ == "__main__":
    import sys

    failed = 0
    for check_name, function in CHECKS:
        message = function()
        failed += message is not None and not message.startswith("pending:")
        print(f"[{'PASS' if message is None else 'FAIL'}] {check_name}" + ("" if message is None else f" -- {message}"))
    sys.exit(1 if failed else 0)
