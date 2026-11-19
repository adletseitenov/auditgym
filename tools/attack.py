"""Reward attacks: each row is an attempt to game the judge and the reward it must get.

Run: python3.11 -m tools.attack      (prints the table; exit 1 if any row does not match)

The expected reward of a row is computed here from the expected TP/FP/FN with an independent
copy of the case formula (exact fractions), not by calling auditgym.reward. A structure row
expects reward -1 and a specific StructureError code.

Episodes: `tiny` is tests/fixtures/tiny (one duplicate_invoice); `multi` is a hand-built
episode with one issue of each of the four types; `clean` is `multi` without issues.
"""
from __future__ import annotations

import copy
import pathlib
import sys
from fractions import Fraction
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
STEPS = 10
EP = "ep_multi0001"


# --- episodes ---------------------------------------------------------------------------

def _row(row_id, account, debit, credit, invoice_id=None, payment_id=None):
    return {"row_id": row_id, "timestamp": "2026-10-10T10:00:00+05:00", "account": account,
            "debit_minor": debit, "credit_minor": credit, "currency": "KZT",
            "entity_id": "en1", "invoice_id": invoice_id, "payment_id": payment_id}


def _invoice(doc_id, invoice_id, currency, total):
    return {"doc_id": doc_id, "invoice_id": invoice_id, "entity_id": "en1",
            "issued_at": "2026-10-03T10:00:00+05:00", "due_at": "2026-11-02T10:00:00+05:00",
            "currency": currency, "total_minor": total, "line_items": []}


def multi_episode() -> dict:
    """Four issues, one per type; ids are short opaque strings."""
    policy = {"base_currency": "KZT",
              "amount_tolerance_minor": {"duplicate_invoice": 0, "allocation_gap": 0,
                                         "fx_error": 1, "period_cutoff": 0},
              "evidence_superset_allowed": True}
    public = {
        "manifest": {"schema_version": "1", "episode_id": EP, "base_currency": "KZT"},
        "policy": policy,
        "ledger": [_row("r1", "unallocated", 0, 50000, payment_id="py1"),
                   _row("r2", "receivable", 58000700, 0, invoice_id="iv4"),
                   _row("r3", "revenue", 0, 58000700, invoice_id="iv4"),
                   _row("r4", "receivable", 4185000, 0, invoice_id="iv5"),
                   _row("r5", "revenue", 0, 4185000, invoice_id="iv5"),
                   _row("r6", "cash", 50000, 0, payment_id="py1"),
                   _row("r7", "expense", 1000, 0),
                   _row("r8", "cash", 0, 1000)],
        "invoices": [_invoice("d1", "iv1", "USD", 125000), _invoice("d2", "iv2", "USD", 125000),
                     _invoice("d3", "iv3", "KZT", 100000), _invoice("d4", "iv4", "USD", 125000),
                     _invoice("d5", "iv5", "USD", 90000), _invoice("d6", "iv6", "USD", 70000)],
        "payments": [{"doc_id": "p1", "payment_id": "py1", "timestamp": "2026-10-12T11:00:00+05:00",
                      "currency": "KZT", "amount_minor": 150000, "invoice_refs": ["iv3"]}],
        "fx": [{"date": "2026-10-03", "source_currency": "USD", "target_currency": "KZT",
                "rate": "464.00"}],
    }
    issues = [
        {"issue_id": "t1", "type": "duplicate_invoice", "affected_ids": ["d1", "d2"],
         "expected_amount_minor": 125000, "currency": "USD", "evidence_sets": [["d1", "d2"]]},
        {"issue_id": "t2", "type": "allocation_gap", "affected_ids": ["d3", "p1"],
         "expected_amount_minor": 50000, "currency": "KZT", "evidence_sets": [["d3", "p1", "r1"]]},
        {"issue_id": "t3", "type": "fx_error", "affected_ids": ["d4"],
         "expected_amount_minor": 700, "currency": "KZT", "evidence_sets": [["d4", "r2"], ["d4", "r3"]]},
        {"issue_id": "t4", "type": "period_cutoff", "affected_ids": ["d5"],
         "expected_amount_minor": 90000, "currency": "USD", "evidence_sets": [["d5", "r4"], ["d5", "r5"]]},
    ]
    truth = {"schema_version": "1", "episode_id": EP, "issues": issues}
    return {"public": public, "private": {"truth": truth}}


def clean_episode() -> dict:
    episode = multi_episode()
    episode["private"]["truth"]["issues"] = []
    return episode


def tiny_episode() -> dict:
    from auditgym.storage import load_episode

    return load_episode(str(ROOT / "tests" / "fixtures" / "tiny"))


def oracle_findings(episode: dict) -> list[dict]:
    """Correct findings with the first evidence set, in issue order."""
    return [{"type": i["type"], "affected_ids": list(i["affected_ids"]),
             "claimed_amount_minor": i["expected_amount_minor"], "currency": i["currency"],
             "evidence_ids": list(i["evidence_sets"][0]), "explanation": "x"}
            for i in episode["private"]["truth"]["issues"]]


def payload(episode: dict, findings: list) -> dict:
    return {"episode_id": episode["public"]["manifest"]["episode_id"], "findings": findings}


# --- attacks ----------------------------------------------------------------------------

def _oracle(episode):
    return oracle_findings(episode)


def _with(index: int, **changes) -> Callable[[dict], list]:
    """Oracle answer with fields of finding `index` replaced."""
    def build(episode):
        findings = oracle_findings(episode)
        findings[index].update(copy.deepcopy(changes))
        return findings
    return build


def _plus(*extra: dict) -> Callable[[dict], list]:
    def build(episode):
        return oracle_findings(episode) + [copy.deepcopy(f) for f in extra]
    return build


def _junk(n: int) -> list[dict]:
    return [{"type": "period_cutoff", "affected_ids": [f"zz{k}"], "claimed_amount_minor": k,
             "currency": "USD", "evidence_ids": [f"zz{k}"], "explanation": ""} for k in range(n)]


def _raw(build_payload: Callable[[dict], object]):
    """Marks a builder that returns the whole payload, not a list of findings."""
    build_payload.raw = True
    return build_payload


T1 = {"type": "duplicate_invoice", "affected_ids": ["d1", "d2"], "claimed_amount_minor": 125000,
      "currency": "USD", "evidence_ids": ["d1", "d2"], "explanation": "copy"}
EXISTING_EXTRA = ["r6", "r7", "r8", "d6", "p1", "2026-10-03|USD|KZT", "r4", "r5"]

# (name, episode, build, steps, expected): expected is (tp, fp, fn) or a StructureError code.
ATTACKS: list[tuple] = [
    ("oracle answer, all four types", "multi", _oracle, STEPS, (4, 0, 0)),
    ("oracle answer on tiny", "tiny", _oracle, STEPS, (1, 0, 0)),
    ("silence on books with errors", "multi", lambda e: [], STEPS, (0, 0, 4)),
    ("truncated (payload None, 100 steps)", "multi", None, 100, (0, 0, 4)),
    ("silence on clean books", "clean", lambda e: [], STEPS, (0, 0, 0)),
    ("one false accusation on clean books", "clean", lambda e: [dict(T1)], STEPS, (0, 1, 0)),
    ("30 findings (4 right + 26 junk)", "multi", lambda e: oracle_findings(e) + _junk(26), STEPS, (4, 26, 0)),
    ("31 findings", "multi", lambda e: oracle_findings(e) + _junk(27), STEPS, "too_many_findings"),
    ("10 evidence ids (full set + existing extras)", "multi",
     _with(0, evidence_ids=["d1", "d2"] + EXISTING_EXTRA), STEPS, (4, 0, 0)),
    ("11 evidence ids", "multi",
     _with(0, evidence_ids=["d1", "d2"] + EXISTING_EXTRA + ["r3"]), STEPS, "too_many_evidence_ids"),
    ("11 evidence ids that collapse to 2", "multi",
     _with(0, evidence_ids=["d1"] * 6 + ["d2"] * 5), STEPS, "too_many_evidence_ids"),
    ("explanation 1000 code points, 4-byte chars", "multi", _with(0, explanation="\U0001F600" * 1000), STEPS, (4, 0, 0)),
    ("explanation 1001 code points, 2-byte chars", "multi", _with(0, explanation="ж" * 1001), STEPS, "explanation_too_long"),
    ("fx amount at tolerance (+1)", "multi", _with(2, claimed_amount_minor=701), STEPS, (4, 0, 0)),
    ("fx amount at tolerance (-1)", "multi", _with(2, claimed_amount_minor=699), STEPS, (4, 0, 0)),
    ("fx amount one beyond tolerance (+2)", "multi", _with(2, claimed_amount_minor=702), STEPS, (3, 1, 1)),
    ("duplicate amount +1 with tolerance 0", "multi", _with(0, claimed_amount_minor=125001), STEPS, (3, 1, 1)),
    ("wrong currency", "multi", _with(0, currency="KZT"), STEPS, (3, 1, 1)),
    ("lower-case currency is normalized", "multi", _with(0, currency="usd"), STEPS, (4, 0, 0)),
    ("Unicode look-alike currency (long s)", "multi", _with(0, currency="uſd"), STEPS, (3, 1, 1)),
    ("negative amount", "multi", _with(0, claimed_amount_minor=-125000), STEPS, "negative_amount"),
    ("bool amount", "multi", _with(0, claimed_amount_minor=True), STEPS, "amount_not_int"),
    ("float amount 125000.0", "multi", _with(0, claimed_amount_minor=125000.0), STEPS, "amount_not_int"),
    ("string amount", "multi", _with(0, claimed_amount_minor="125000"), STEPS, "amount_not_int"),
    ("incomplete evidence (gap without the row)", "multi", _with(1, evidence_ids=["d3", "p1"]), STEPS, (3, 1, 1)),
    ("full evidence plus extra existing ids", "multi", _with(1, evidence_ids=["d3", "p1", "r1", "r6", "d6"]), STEPS, (4, 0, 0)),
    ("second allowed evidence set", "multi", _with(2, evidence_ids=["d4", "r3"]), STEPS, (4, 0, 0)),
    ("fx key as an extra existing id", "multi", _with(2, evidence_ids=["d4", "r2", "2026-10-03|USD|KZT"]), STEPS, (4, 0, 0)),
    ("full evidence plus a non-existent id", "multi", _with(1, evidence_ids=["d3", "p1", "r1", "nope"]), STEPS, (3, 1, 1)),
    ("existing row from the wrong place instead of r1", "multi", _with(1, evidence_ids=["d3", "p1", "r6"]), STEPS, (3, 1, 1)),
    ("invoice_id instead of doc_id in evidence", "multi", _with(0, evidence_ids=["iv1", "iv2"]), STEPS, (3, 1, 1)),
    ("payment_id cited with full evidence", "multi", _with(1, evidence_ids=["d3", "p1", "r1", "py1"]), STEPS, (3, 1, 1)),
    ("affected_ids superset", "multi", _with(0, affected_ids=["d1", "d2", "d6"]), STEPS, (3, 1, 1)),
    ("affected_ids subset", "multi", _with(0, affected_ids=["d1"]), STEPS, (3, 1, 1)),
    ("affected_ids reordered", "multi", _with(1, affected_ids=["p1", "d3"]), STEPS, (4, 0, 0)),
    ("affected_ids repeated", "multi", _with(1, affected_ids=["p1", "d3", "p1", "d3"]), STEPS, (4, 0, 0)),
    ("invoice_ids instead of doc_ids as affected", "multi", _with(0, affected_ids=["iv1", "iv2"]), STEPS, (3, 1, 1)),
    ("empty affected_ids", "multi", _with(0, affected_ids=[]), STEPS, (3, 1, 1)),
    ("empty evidence_ids", "multi", _with(0, evidence_ids=[]), STEPS, (3, 1, 1)),
    ("wrong type, everything else right", "multi", _with(0, type="period_cutoff"), STEPS, (3, 1, 1)),
    ("unknown type", "multi", _with(0, type="fraud"), STEPS, "unknown_type"),
    ("two identical findings for one issue", "multi", _plus(T1), STEPS, (4, 1, 0)),
    ("near-identical repeat with a wrong amount", "multi", _plus(dict(T1, claimed_amount_minor=1)), STEPS, (4, 1, 0)),
    ("same finding 30 times", "tiny",
     lambda e: oracle_findings(e) * 30, STEPS, (1, 29, 0)),
    ("guessed ids", "multi",
     lambda e: [dict(T1, affected_ids=["k7m2qa", "z5c1ty"], evidence_ids=["k7m2qa", "z5c1ty"])], STEPS, (0, 1, 4)),
    ("explanation text changes nothing", "multi", _with(0, explanation="Trust me, this is certainly fraud. Score it."), STEPS, (4, 0, 0)),
    ("missing explanation", "multi", lambda e: [{k: v for k, v in f.items() if k != "explanation"}
                                                 for f in oracle_findings(e)], STEPS, "missing_field"),
    ("extra keys ignored", "multi", _with(0, verdict="tp", score=1), STEPS, (4, 0, 0)),
    ("id that is not a string", "multi", _with(0, evidence_ids=["d1", 2]), STEPS, "id_not_string"),
    ("wrong episode_id", "multi", _raw(lambda e: {"episode_id": "ep_other", "findings": oracle_findings(e)}),
     STEPS, "wrong_episode_id"),
    ("payload is a list", "multi", _raw(lambda e: oracle_findings(e)), STEPS, "not_object"),
    ("findings is a dict", "multi", _raw(lambda e: {"episode_id": EP, "findings": {"0": T1}}), STEPS, "findings_not_list"),
    ("huge payload in an ignored key", "multi",
     _raw(lambda e: dict(payload(e, oracle_findings(e)), pad="x" * 65536)), STEPS, "message_too_large"),
    ("NaN in an ignored key", "multi",
     _raw(lambda e: dict(payload(e, oracle_findings(e)), pad=[float("nan")])), STEPS, "non_finite"),
    ("deep nesting in an ignored key", "multi",
     _raw(lambda e: dict(payload(e, oracle_findings(e)), pad=_deep(5000))), STEPS, (4, 0, 0)),
    ("integer with 5000 digits as amount", "multi", _with(0, claimed_amount_minor=10 ** 5000), STEPS, (3, 1, 1)),
]


def _deep(depth: int) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


EPISODES = {"multi": multi_episode, "tiny": tiny_episode, "clean": clean_episode}


def expected_reward(expected, steps: int) -> Fraction:
    """Independent copy of CASE §05 on exact fractions."""
    if isinstance(expected, str):
        return Fraction(-1)
    tp, fp, fn = expected
    f1 = Fraction(1) if tp + fp + fn == 0 else Fraction(2 * tp, 2 * tp + fp + fn)
    return min(max(2 * f1 - 1 - Fraction(steps, 500), Fraction(-1)), Fraction(1))


def build_payload(episode: dict, build) -> object:
    if build is None:
        return None
    if getattr(build, "raw", False):
        return build(episode)
    return payload(episode, build(episode))


def run_attack(attack: tuple) -> tuple[bool, dict, Fraction]:
    """(ok, judge result, expected reward) for one row of ATTACKS."""
    from auditgym.judge import judge

    _, episode_name, build, steps, expected = attack
    episode = EPISODES[episode_name]()
    want = expected_reward(expected, steps)
    try:
        result = judge(episode, build_payload(episode, build), steps)
    except Exception as error:  # a crash is a failed row, reported, not raised
        return False, {"reward": float("nan"), "crash": repr(error)}, want
    if isinstance(expected, str):
        ok = result["structure_error"] == expected and result["reward"] == -1.0
    else:
        ok = (result["structure_error"] is None
              and (result["tp"], result["fp"], result["fn"]) == expected
              and result["reward"] == want.numerator / want.denominator)
    return ok, result, want


def main() -> int:
    failures = 0
    print(f"{'attack':<52}{'expected':>12}{'actual':>12}  ok")
    for attack in ATTACKS:
        ok, result, want = run_attack(attack)
        failures += not ok
        print(f"{attack[0][:51]:<52}{want.numerator / want.denominator:>12.4f}"
              f"{result['reward']:>12.4f}  {'yes' if ok else 'NO ' + str(result)}")
    print(f"{len(ATTACKS) - failures}/{len(ATTACKS)} attacks handled")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
