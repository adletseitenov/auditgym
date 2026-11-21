"""The D01-D20 matrix and the 20 paired checks of Block 4 through `judge_adapter`.

    python -m tools.adapter_pairs [--only P03 P11 ...] [--verbose]

For every D01-D20 (the matrix below: what each episode is, in terms of the generator's documented
config) the jury's flow is played: `generate_episode` once; a copy transformed on disk by
`tools.transform` (P01-P05 reorder, P06-P10 rename ids, P11-P20 both; the transform has fixed
parameters, the original is never rewritten); both loaded through `reset` with
{"mode": "closed", "episode_dir": ...}. Then:

  repeat     the same actions on the original twice (two resets of one env, and one fresh env):
             records, transitions and rewards are equal as JSON values;
  oracle     on the transformed copy the oracle (built from the transformed truth, jury side) submits
             through step with the same number of inspection steps; after the inverse id mapping the
             findings equal the original ones, and TP/FP/FN/P/R/F1/reward are identical, both by
             score_episode and by the evaluator journal;
  records    the whole open data read through list_records, mapped back, equals the original's;
  control    for a renamed copy the ORIGINAL ids submitted there are not accepted (tp == 0), so the
             invariance is not a reward that ignores the evidence.

Exits 1 on any failure. Episodes are found by a seed search with a predicate over the generated
episode (`discover`), so every claim about a D-episode ("a split payment", "wrong date") is checked
on the data, not assumed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALL4 = {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1}
TABLES = ("ledger", "invoices", "payments", "fx")
EXPLORATION = (  # the same actions on every side of a pair: 5 inspection steps, one of them an error
    {"name": "list_records", "args": {"table": "invoices", "cursor": None, "limit": 10}},
    {"name": "list_records", "args": {"table": "payments", "cursor": None, "limit": 10}},
    {"name": "list_records", "args": {"table": "fx", "cursor": None, "limit": 5}},
    {"name": "calculator", "args": {"expression": "1000*0.002"}},
    {"name": "get_record", "args": {"table": "ledger", "id": "no-such-row"}},
)


# ------------------------------------------------------------------ predicates over an episode
def _clean(episode) -> bool:
    return not episode["private"]["truth"]["issues"]


def _split_payment(episode) -> bool:
    paid: dict = {}
    for payment in episode["public"]["payments"]:
        if payment["amount_minor"] > 0:
            for ref in payment["invoice_refs"] or []:
                paid[ref] = paid.get(ref, 0) + 1
    return _clean(episode) and any(count >= 2 for count in paid.values())


def _refund(episode) -> bool:
    return _clean(episode) and any(p["amount_minor"] < 0 for p in episode["public"]["payments"])


def _repeated_sums(episode) -> bool:
    totals = [(i["currency"], i["total_minor"]) for i in episode["public"]["invoices"]]
    return _clean(episode) and len(set(totals)) < len(totals)


def _zone_boundary(episode) -> bool:
    """A document written in another offset whose date in the company timezone differs from its naive date."""
    sign = -1 if episode["public"]["manifest"]["timezone"][0] == "-" else 1
    hours, minutes = episode["public"]["manifest"]["timezone"][1:].split(":")
    zone = dt.timezone(sign * dt.timedelta(hours=int(hours), minutes=int(minutes)))
    for invoice in episode["public"]["invoices"]:
        moment = dt.datetime.fromisoformat(invoice["issued_at"])
        if moment.date() != moment.astimezone(zone).date():
            return _clean(episode)
    return False


def _issue_types(*types):
    return lambda episode: sorted(i["type"] for i in episode["private"]["truth"]["issues"]) == sorted(types)


def _fx_mode(mode):
    def check(episode) -> bool:
        steps = episode["private"]["provenance"]["steps"]
        return sorted(i["type"] for i in episode["private"]["truth"]["issues"]) == ["fx_error"] and \
            steps[0]["params"].get("mode") == mode
    return check


def _gap_variant(variant):
    """An allocation_gap whose payment allocation is entirely missing ("missing") or only partly moved to
    unallocated ("partial"); the generator records the variant of a missing one in the provenance params."""
    def check(episode) -> bool:
        steps = episode["private"]["provenance"]["steps"]
        return sorted(i["type"] for i in episode["private"]["truth"]["issues"]) == ["allocation_gap"] and \
            steps[0]["params"].get("variant", "partial") == variant
    return check


def _both(*checks):
    return lambda episode: all(check(episode) for check in checks)


# id, what it is, config, predicate. The config is the team config of auditgym/config.py.
MATRIX = (
    ("D01", "clean, plain reconciliation (no distractors)", {"errors": {}, "distractors": False}, _clean),
    ("D02", "clean, a correctly split payment", {"errors": {}, "currencies": ["USD", "EUR"], "distractors": {"split_payments": True}}, _split_payment),
    ("D03", "clean, a correct refund", {"errors": {}, "currencies": ["EUR"], "distractors": {"refunds": True}}, _refund),
    ("D04", "clean, legitimately repeated amounts", {"errors": {}, "currencies": ["USD"], "distractors": {"repeated_amounts": True}}, _repeated_sums),
    ("D05", "clean, a period/offset boundary by policy",
     {"errors": {}, "timezone": "-11:00", "currencies": ["USD"],
      "distractors": {"boundary_dates": True, "weekend_fx": True, "other_offsets": True}}, _zone_boundary),
    ("D06", "duplicate invoice", {"errors": {"duplicate_invoice": 1}}, _issue_types("duplicate_invoice")),
    ("D07", "duplicate invoice, EUR", {"errors": {"duplicate_invoice": 1}, "currencies": ["EUR"]},
     _issue_types("duplicate_invoice")),
    ("D08", "duplicate invoice, two currencies, another offset",
     {"errors": {"duplicate_invoice": 1}, "currencies": ["USD", "RUB"], "timezone": "+03:00"},
     _issue_types("duplicate_invoice")),
    ("D09", "allocation gap, payment allocation entirely missing",
     {"errors": {"allocation_gap": 1}, "allocation_gap_mode": "missing"}, _gap_variant("missing")),
    ("D10", "allocation gap, partial allocation, EUR",
     {"errors": {"allocation_gap": 1}, "allocation_gap_mode": "partial", "currencies": ["EUR"]},
     _gap_variant("partial")),
    ("D11", "allocation gap, entirely missing, CNY, another offset",
     {"errors": {"allocation_gap": 1}, "allocation_gap_mode": "missing", "currencies": ["CNY"],
      "timezone": "+09:00"}, _gap_variant("missing")),
    ("D12", "fx error, wrong rate date", {"errors": {"fx_error": 1}, "currencies": ["USD"]}, _fx_mode("wrong_date")),
    ("D13", "fx error, wrong rate value", {"errors": {"fx_error": 1}, "currencies": ["EUR"]}, _fx_mode("wrong_rate")),
    ("D14", "period cutoff", {"errors": {"period_cutoff": 1}}, _issue_types("period_cutoff")),
    ("D15", "period cutoff, far-west offset", {"errors": {"period_cutoff": 1}, "timezone": "-11:00"},
     _issue_types("period_cutoff")),
    ("D16", "interaction I1", {"errors": {}, "interaction": "I1"}, _issue_types("allocation_gap", "fx_error")),
    ("D17", "interaction I1, CNY, another offset",
     {"errors": {}, "interaction": "I1", "currencies": ["CNY"], "timezone": "+09:00"},
     _issue_types("allocation_gap", "fx_error")),
    ("D18", "interaction I1 with other errors, two currencies",
     {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}, "interaction": "I1", "currencies": ["USD", "EUR"]},
     _issue_types("allocation_gap", "fx_error", "duplicate_invoice", "period_cutoff")),
    ("D19", "interaction I2, EUR", {"errors": {}, "interaction": "I2", "currencies": ["EUR"]},
     _issue_types("duplicate_invoice", "period_cutoff")),
    ("D20", "interaction I2, USD and RUB, another offset",
     {"errors": {}, "interaction": "I2", "timezone": "+14:00", "currencies": ["USD", "RUB"]},
     _issue_types("duplicate_invoice", "period_cutoff")),
)
# P01-P05 reorder, P06-P10 rename, P11-P20 both (docs/ADAPTER.md)
PAIR_KIND = {f"D{number:02d}": ("order" if number <= 5 else "ids" if number <= 10 else "both")
             for number in range(1, 21)}
MAX_SEED_SEARCH = 400


def discover(index: int):
    """(seed, config) of the first seed from 1000*(index+1) on whose episode satisfies the predicate of
    MATRIX[index]. The search is deterministic; a matrix entry the generator cannot produce raises."""
    from auditgym.config import ConfigError
    from auditgym.generate import generate

    label, _, config, predicate = MATRIX[index]
    for seed in range(1000 * (index + 1), 1000 * (index + 1) + MAX_SEED_SEARCH):
        try:
            episode = generate(seed, config)
        except ConfigError:
            continue
        if predicate(episode):
            return seed, config
    raise RuntimeError(f"{label}: no episode within {MAX_SEED_SEARCH} seeds")


# ------------------------------------------------------------------ helpers
def tree_digest(directory) -> str:
    root = pathlib.Path(directory)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def oracle_findings(directory) -> dict:
    """The jury-side oracle: the findings.json object built from the truth of `directory`."""
    from auditgym.storage import load_episode
    from policies.oracle import Oracle

    episode = load_episode(directory)
    oracle = Oracle(episode["private"]["truth"])
    oracle.reset({"manifest": {"episode_id": episode["public"]["manifest"]["episode_id"]}})
    return oracle.act({})["findings"]


def map_findings(findings: dict, mapping: dict) -> dict:
    """The same findings with every id (and the episode id) replaced through `mapping`."""
    def one(item: dict) -> dict:
        return dict(item, affected_ids=[mapping.get(i, i) for i in item["affected_ids"]],
                    evidence_ids=[mapping.get(i, i) for i in item["evidence_ids"]])

    return {"episode_id": mapping.get(findings["episode_id"], findings["episode_id"]),
            "findings": [one(item) for item in findings["findings"]]}


def _truth(directory) -> dict:
    return json.loads((pathlib.Path(directory) / "private" / "truth.json").read_text())


def play(env, directory, script, findings):
    """reset on `directory` (closed mode), the script, then the submit. Returns the transcript."""
    first, info = env.reset(0, {"mode": "closed", "episode_dir": str(directory)})
    transcript = [("reset", first, info)]
    for action in script:
        transcript.append((action, *env.step(action)))
    transcript.append(("submit", *env.step({"name": "submit", "args": {"findings": findings}})))
    return transcript


def read_all(env, directory) -> dict:
    """Every record of every table through list_records (resetting when the budget runs low)."""
    out = {}
    env.reset(0, {"mode": "closed", "episode_dir": str(directory)})
    used = 0
    for table in TABLES:
        records, cursor = [], None
        while True:
            if used >= 95:
                env.reset(0, {"mode": "closed", "episode_dir": str(directory)})
                used = 0
            observation, _, terminated, truncated, _ = env.step(
                {"name": "list_records", "args": {"table": table, "cursor": cursor, "limit": 50}})
            used += 1
            if observation["error"] is not None or terminated or truncated:
                raise AssertionError(f"reading {table} failed")
            records += observation["result"]["records"]
            cursor = observation["result"]["next_cursor"]
            if cursor is None:
                break
        out[table] = records
    return out


def _canonical(table: str, record: dict, back: dict) -> str:
    def walk(value):
        if isinstance(value, str):
            return back.get(value, value)
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    record = walk(record)
    if isinstance(record.get("line_items"), list):
        record["line_items"] = sorted(json.dumps(item, sort_keys=True) for item in record["line_items"])
    if isinstance(record.get("invoice_refs"), list):
        record["invoice_refs"] = sorted(record["invoice_refs"])
    return json.dumps(record, sort_keys=True)


def _finding_key(finding: dict, back: dict) -> str:
    return json.dumps({"type": finding["type"], "affected": sorted(back.get(i, i) for i in finding["affected_ids"]),
                       "amount": finding["claimed_amount_minor"], "currency": finding["currency"],
                       "evidence": sorted(back.get(i, i) for i in finding["evidence_ids"])}, sort_keys=True)


def run_pair(index: int, workdir, verbose: bool = False) -> list[str]:
    """Play pair P{index+1}; returns the list of failures (empty: the pair passes)."""
    import judge_adapter
    from tools import transform

    label, what, _, _ = MATRIX[index]
    kind = PAIR_KIND[label]
    failures: list[str] = []
    work = pathlib.Path(workdir) / f"P{index + 1:02d}"
    seed, config = discover(index)
    original, copy = work / "original", work / "transformed"
    judge_adapter.generate_episode(seed, config, str(original))
    before = tree_digest(original)
    document = transform.transform(original, copy, seed=7000 + index, shuffle=kind in ("order", "both"),
                                   rename=kind in ("ids", "both"), id_style=transform.ID_STYLES[index % 4])
    if tree_digest(original) != before:
        failures.append("the original episode was rewritten")
    mapping = transform.load_mapping(copy)
    back = {new: old for old, new in mapping.items()}
    journal = work / "results.jsonl"

    with judge_adapter.make_env(journal_path=journal) as env:
        findings_original = oracle_findings(original)
        findings_own = oracle_findings(copy)  # the oracle of the copy's own truth (it may pick another valid set)
        findings_copy = map_findings(findings_original, mapping)  # the same findings, new ids
        # repeat on the original: same env twice, then a fresh env
        first = play(env, original, EXPLORATION, findings_original)
        steps_first = env.inspection_steps
        second = play(env, original, EXPLORATION, findings_original)
        with judge_adapter.make_env(journal_path=work / "fresh.jsonl") as fresh:
            third = play(fresh, original, EXPLORATION, findings_original)
        if json.loads(json.dumps(first)) != json.loads(json.dumps(second)):
            failures.append("repeat: second run on the same env differs")
        if json.loads(json.dumps(first)) != json.loads(json.dumps(third)):
            failures.append("repeat: a fresh env differs")
        rewards = [row[2] for row in first[1:]]
        if any(reward != 0 for reward in rewards):
            failures.append("closed mode: a non-zero reward reached the caller")
        rows_original = [row for row in _journal(journal) if row["episode_id"] == _episode_id(original)
                         and row["event"] == "finished"]
        # transformed run: same exploration steps, oracle with the new ids
        play(env, copy, EXPLORATION, findings_copy)
        steps_copy = env.inspection_steps
        rows_copy = [row for row in _journal(journal) if row["episode_id"] == _episode_id(copy)
                     and row["event"] == "finished"]
        if steps_first != steps_copy:
            failures.append(f"inspection steps differ: {steps_first} vs {steps_copy}")
        scored_original = judge_adapter.score_episode(str(original), findings_original, steps_first,
                                                      journal_path=work / "score.log")
        scored_copy = judge_adapter.score_episode(str(copy), findings_copy, steps_copy, journal_path=work / "score.log")
        if scored_original != scored_copy:
            failures.append(f"score_episode differs: {scored_original} vs {scored_copy}")
        for name in ("tp", "fp", "fn", "precision", "recall", "f1"):
            if not (rows_original and rows_copy) or rows_original[-1][name] != rows_copy[-1][name]:
                failures.append(f"journal {name} differs")
        if not rows_original or not rows_copy or rows_original[-1]["reward"] != rows_copy[-1]["reward"] \
                or rows_original[-1]["reward"] != scored_original["return"]:
            failures.append("journal reward differs from score_episode or between the sides")
        if scored_original["fn"] != 0 or scored_original["fp"] != 0 \
                or scored_original["tp"] != len(findings_original["findings"]):
            failures.append(f"the oracle does not score a full match: {scored_original}")
        # the findings after the inverse mapping equal the original ones
        left = sorted(_finding_key(f, back) for f in findings_copy["findings"])
        right = sorted(_finding_key(f, {}) for f in findings_original["findings"])
        if left != right:
            failures.append("findings differ after the inverse id mapping")
        # the oracle of the copy's truth: same types, records, sums, currencies; evidence is one of the valid sets
        truth_sets = {issue["issue_id"]: [sorted(group) for group in issue["evidence_sets"]]
                      for issue in _truth(original)["issues"]}
        own = judge_adapter.score_episode(str(copy), findings_own, steps_copy, journal_path=work / "score.log")
        if own != scored_original:
            failures.append(f"the oracle of the copy scores differently: {own}")
        valid_sets = [group for sets in truth_sets.values() for group in sets]
        for finding in findings_own["findings"]:
            if sorted(back.get(i, i) for i in finding["evidence_ids"]) not in valid_sets:
                failures.append("the oracle of the copy cites a set that is not valid for the original")
        a_core = sorted(_finding_key(dict(f, evidence_ids=[]), back) for f in findings_own["findings"])
        b_core = sorted(_finding_key(dict(f, evidence_ids=[]), {}) for f in findings_original["findings"])
        if a_core != b_core:
            failures.append("the oracle of the copy differs in type, records, sum or currency")
        # the open data, mapped back
        data_original, data_copy = read_all(env, original), read_all(env, copy)
        for table in TABLES:
            a = sorted(_canonical(table, r, {}) for r in data_original[table])
            b = sorted(_canonical(table, r, back) for r in data_copy[table])
            if a != b:
                failures.append(f"records of {table} differ after the inverse mapping")
        if kind == "order" and any(data_original[t] == data_copy[t] for t in ("ledger", "invoices")):
            failures.append("order pair: a table was not reordered")
        # control: the original ids on a renamed copy are not a correct answer
        if kind != "order" and findings_original["findings"]:
            wrong = dict(findings_original, episode_id=document["episode_id"]["new"])
            control = judge_adapter.score_episode(str(copy), wrong, steps_copy, journal_path=work / "score.log")
            if control["tp"] != 0:
                failures.append("control: the original ids are accepted on the renamed copy")
    if verbose:
        print(f"P{index + 1:02d} {label} {kind:<5} {what}: " + ("ok" if not failures else "; ".join(failures)))
    return failures


def _journal(path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _episode_id(directory) -> str:
    return json.loads((pathlib.Path(directory) / "public" / "manifest.json").read_text())["episode_id"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.adapter_pairs")
    parser.add_argument("--only", nargs="*", default=None, help="pair ids such as P03 P11")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)
    wanted = {name.upper() for name in arguments.only} if arguments.only else None
    failed = 0
    count = 0
    with tempfile.TemporaryDirectory() as scratch:
        for index in range(len(MATRIX)):
            if wanted is not None and f"P{index + 1:02d}" not in wanted:
                continue
            count += 1
            problems = run_pair(index, scratch, verbose=arguments.verbose or True)
            failed += bool(problems)
    print(f"adapter pairs: {count - failed}/{count} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
