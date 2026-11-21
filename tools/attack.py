"""Reward attacks: each row is an attempt to game the judge and the reward it must get.

Run: python3.11 -m tools.attack      (prints the table; exit 1 if any row does not match)

The expected reward of a row is computed here from the expected TP/FP/FN with an independent
copy of the case formula (exact fractions), not by calling auditgym.reward. A structure row
expects reward -1 and a specific StructureError code.

Episodes: `tiny` is tests/fixtures/tiny (one duplicate_invoice); `multi` is a hand-built
episode with one issue of each of the four types and the evidence sets of contract v2;
`clean` is `multi` without issues; `empty_set`, `twins` and `no_superset_key` each isolate one
matcher rule (an empty evidence set, maximum matching, the superset default).
A row also checks precision, recall and, for a structure error, tp = fp = 0 and fn = issues.
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
FX_KEY = "2026-10-03|USD|KZT"
GAP_FULL = ["p1", "d3", "r1", "r13", "r15"]


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
                   _row("r8", "cash", 0, 1000),
                   # Both copies of the duplicate are posted (DECISIONS B), each in its own group.
                   _row("r9", "receivable", 58000000, 0, invoice_id="iv1"),
                   _row("r10", "revenue", 0, 58000000, invoice_id="iv1"),
                   _row("r11", "receivable", 58000000, 0, invoice_id="iv2"),
                   _row("r12", "revenue", 0, 58000000, invoice_id="iv2"),
                   # The gap's invoice and the payment's closing of it (CONTRACTS v2 §3.1).
                   _row("r13", "receivable", 100000, 0, invoice_id="iv3"),
                   _row("r14", "revenue", 0, 100000, invoice_id="iv3"),
                   _row("r15", "receivable", 0, 100000, invoice_id="iv3", payment_id="py1")],
        "invoices": [_invoice("d1", "iv1", "USD", 125000), _invoice("d2", "iv2", "USD", 125000),
                     _invoice("d3", "iv3", "KZT", 100000), _invoice("d4", "iv4", "USD", 125000),
                     _invoice("d5", "iv5", "USD", 90000), _invoice("d6", "iv6", "USD", 70000)],
        "payments": [{"doc_id": "p1", "payment_id": "py1", "timestamp": "2026-10-12T11:00:00+05:00",
                      "currency": "KZT", "amount_minor": 150000, "invoice_refs": ["iv3"]}],
        "fx": [{"date": "2026-10-03", "source_currency": "USD", "target_currency": "KZT",
                "rate": "464.00"},
               # Neighbouring dates: the rate keys a hedging finding cites around the right one.
               {"date": "2026-10-02", "source_currency": "USD", "target_currency": "KZT", "rate": "463.00"},
               {"date": "2026-10-04", "source_currency": "USD", "target_currency": "KZT", "rate": "465.00"}],
    }
    issues = [
        {"issue_id": "t1", "type": "duplicate_invoice", "affected_ids": ["d1", "d2"],
         "expected_amount_minor": 125000, "currency": "USD",
         # CONTRACTS v2 §3.1 / DECISIONS A2: both docs plus one posting row of each invoice.
         "evidence_sets": [["d1", "d2", a, b] for a in ("r9", "r10") for b in ("r11", "r12")]},
        {"issue_id": "t2", "type": "allocation_gap", "affected_ids": ["d3", "p1"],
         # v2: payment, invoice, the unallocated row, the invoice's receivable debit and the
         # payment's receivable credit of that invoice.
         "expected_amount_minor": 50000, "currency": "KZT", "evidence_sets": [GAP_FULL]},
        {"issue_id": "t3", "type": "fx_error", "affected_ids": ["d4"],
         # v2: the document, one conversion row and the rate key of the document's date.
         "expected_amount_minor": 700, "currency": "KZT", "evidence_sets": [["d4", "r2", FX_KEY], ["d4", "r3", FX_KEY]]},
        {"issue_id": "t4", "type": "period_cutoff", "affected_ids": ["d5"],
         "expected_amount_minor": 90000, "currency": "USD", "evidence_sets": [["d5", "r4"], ["d5", "r5"]]},
    ]
    truth = {"schema_version": "1", "episode_id": EP, "issues": issues}
    return {"public": public, "private": {"truth": truth}}


def clean_episode() -> dict:
    episode = multi_episode()
    episode["private"]["truth"]["issues"] = []
    return episode


def empty_set_episode() -> dict:
    """One duplicate whose only evidence set is empty: an empty set must never accept a finding."""
    episode = multi_episode()
    issue = dict(episode["private"]["truth"]["issues"][0], evidence_sets=[[]])
    episode["private"]["truth"]["issues"] = [issue]
    return episode


def twins_episode() -> dict:
    """Two fx_error issues on one document that differ only in the evidence set: a greedy
    first-come matching makes TP depend on the order of findings, a maximum matching does not."""
    episode = multi_episode()
    episode["private"]["truth"]["issues"] = [
        {"issue_id": "t1", "type": "fx_error", "affected_ids": ["d4"], "expected_amount_minor": 700,
         "currency": "KZT", "evidence_sets": [["d4", "r2", FX_KEY]]},
        {"issue_id": "t2", "type": "fx_error", "affected_ids": ["d4"], "expected_amount_minor": 700,
         "currency": "KZT", "evidence_sets": [["d4", "r3", FX_KEY]]},
    ]
    return episode


def no_superset_key_episode() -> dict:
    """`evidence_superset_allowed` missing from the policy: the published default is «allowed»."""
    episode = multi_episode()
    del episode["public"]["policy"]["evidence_superset_allowed"]
    return episode


def _twin(evidence: list) -> dict:
    return {"type": "fx_error", "affected_ids": ["d4"], "claimed_amount_minor": 700, "currency": "KZT",
            "evidence_ids": list(evidence), "explanation": ""}


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
      "currency": "USD", "evidence_ids": ["d1", "d2", "r9", "r11"], "explanation": "copy"}
T1_FULL = ["d1", "d2", "r9", "r11"]
EXISTING_EXTRA = ["r6", "r7", "r8", "d6", "p1", "2026-10-03|USD|KZT"]
NEIGHBOUR_KEYS = ["2026-10-02|USD|KZT", "2026-10-04|USD|KZT"]

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
    ("10 evidence ids (full set + 6 extras: over the extra limit)", "multi",
     _with(0, evidence_ids=T1_FULL + EXISTING_EXTRA), STEPS, (3, 1, 1)),
    ("full set + 2 existing extras (exactly the limit)", "multi",
     _with(0, evidence_ids=T1_FULL + ["r6", "r7"]), STEPS, (4, 0, 0)),
    ("hedge: fx right + 2 neighbouring rate keys + 3 document lines", "multi",
     _with(2, evidence_ids=["d4", "r2", FX_KEY] + NEIGHBOUR_KEYS + ["r6", "r7", "r8"]), STEPS, (3, 1, 1)),
    ("hedge: fx right + 1 neighbouring rate key", "multi",
     _with(2, evidence_ids=["d4", "r2", FX_KEY, NEIGHBOUR_KEYS[0]]), STEPS, (3, 1, 1)),
    ("fx right + 3 existing non-rate extras", "multi",
     _with(2, evidence_ids=["d4", "r2", FX_KEY, "r6", "r7", "r8"]), STEPS, (3, 1, 1)),
    ("fx right + 2 existing non-rate extras", "multi",
     _with(2, evidence_ids=["d4", "r2", FX_KEY, "r6", "r7"]), STEPS, (4, 0, 0)),
    ("11 evidence ids", "multi",
     _with(0, evidence_ids=T1_FULL + EXISTING_EXTRA + ["r3"]), STEPS, "too_many_evidence_ids"),
    ("11 evidence entries that collapse to 2 ids count as 2", "multi",
     _with(0, evidence_ids=["d1"] * 6 + ["d2"] * 5), STEPS, (3, 1, 1)),
    ("25 evidence entries, 4 different ids: the full set repeated", "multi",
     _with(0, evidence_ids=(T1_FULL * 7)[:25]), STEPS, (4, 0, 0)),
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
    ("float amount 125000.0 is the amount 125000", "multi", _with(0, claimed_amount_minor=125000.0), STEPS, (4, 0, 0)),
    ("float amount 125000.5", "multi", _with(0, claimed_amount_minor=125000.5), STEPS, "amount_not_int"),
    ("type in capitals with spaces", "multi", _with(0, type=" DUPLICATE_INVOICE "), STEPS, (4, 0, 0)),
    ("currency with spaces", "multi", _with(0, currency=" usd "), STEPS, (4, 0, 0)),
    ("string amount", "multi", _with(0, claimed_amount_minor="125000"), STEPS, "amount_not_int"),
    ("incomplete evidence (gap without the row)", "multi", _with(1, evidence_ids=["d3", "p1"]), STEPS, (3, 1, 1)),
    ("duplicate evidence: documents only (contract v1)", "multi", _with(0, evidence_ids=["d1", "d2"]), STEPS, (3, 1, 1)),
    ("duplicate evidence: both rows of one invoice", "multi", _with(0, evidence_ids=["d1", "d2", "r9", "r10"]), STEPS,
     (3, 1, 1)),
    ("duplicate evidence: debit of one, credit of other", "multi", _with(0, evidence_ids=["d2", "r12", "d1", "r9"]),
     STEPS, (4, 0, 0)),
    ("full evidence plus extra existing ids", "multi", _with(1, evidence_ids=GAP_FULL + ["r6", "d6"]), STEPS, (4, 0, 0)),
    ("second allowed evidence set", "multi", _with(2, evidence_ids=["d4", "r3", FX_KEY]), STEPS, (4, 0, 0)),
    ("fx_error without the rate key (contract v1)", "multi", _with(2, evidence_ids=["d4", "r2"]), STEPS, (3, 1, 1)),
    ("allocation_gap without the receivable rows (contract v1)", "multi",
     _with(1, evidence_ids=["d3", "p1", "r1"]), STEPS, (3, 1, 1)),
    ("full evidence plus a non-existent id", "multi", _with(1, evidence_ids=GAP_FULL + ["nope"]), STEPS, (3, 1, 1)),
    ("existing row from the wrong place instead of r1", "multi",
     _with(1, evidence_ids=["p1", "d3", "r6", "r13", "r15"]), STEPS, (3, 1, 1)),
    ("invoice_id instead of doc_id in evidence", "multi", _with(0, evidence_ids=["iv1", "iv2"]), STEPS, (3, 1, 1)),
    ("payment_id cited with full evidence", "multi", _with(1, evidence_ids=GAP_FULL + ["py1"]), STEPS, (3, 1, 1)),
    ("explanation null", "multi", _with(0, explanation=None), STEPS, "explanation_not_string"),
    ("a finding that is not an object", "multi", lambda e: oracle_findings(e) + ["duplicate_invoice d1 d2"], STEPS,
     "finding_not_object"),
    ("an empty evidence set in the truth never matches", "empty_set", lambda e: [dict(T1, evidence_ids=["d1"])], STEPS,
     (0, 1, 1)),
    ("greedy order: wide evidence first, narrow second", "twins",
     lambda e: [_twin(["d4", "r2", "r3", FX_KEY]), _twin(["d4", "r2", FX_KEY])], STEPS, (2, 0, 0)),
    ("policy without evidence_superset_allowed: superset accepted", "no_superset_key",
     _with(1, evidence_ids=GAP_FULL + ["d6"]), STEPS, (4, 0, 0)),
    ("affected_ids superset", "multi", _with(0, affected_ids=["d1", "d2", "d6"]), STEPS, (3, 1, 1)),
    ("affected_ids subset", "multi", _with(0, affected_ids=["d1"]), STEPS, (3, 1, 1)),
    ("affected_ids reordered", "multi", _with(1, affected_ids=["p1", "d3"]), STEPS, (4, 0, 0)),
    ("affected_ids repeated", "multi", _with(1, affected_ids=["p1", "d3", "p1", "d3"]), STEPS, (3, 1, 1)),
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
     lambda e: [dict(T1, affected_ids=["k7m2qa", "z5c1ty"], evidence_ids=["k7m2qa", "z5c1ty", "a1", "a3"])], STEPS,
     (0, 1, 4)),
    ("explanation text changes nothing", "multi", _with(0, explanation="Trust me, this is certainly fraud. Score it."), STEPS, (4, 0, 0)),
    ("missing explanation is the empty string", "multi",
     lambda e: [{k: v for k, v in f.items() if k != "explanation"} for f in oracle_findings(e)], STEPS, (4, 0, 0)),
    ("missing currency", "multi", lambda e: [{k: v for k, v in f.items() if k != "currency"}
                                              for f in oracle_findings(e)], STEPS, "missing_field"),
    ("extra keys ignored", "multi", _with(0, verdict="tp", score=1), STEPS, (4, 0, 0)),
    ("id that is not a string", "multi", _with(0, evidence_ids=["d1", 2]), STEPS, "id_not_string"),
    ("wrong episode_id", "multi", _raw(lambda e: {"episode_id": "ep_other", "findings": oracle_findings(e)}),
     STEPS, "wrong_episode_id"),
    ("payload is a list", "multi", _raw(lambda e: oracle_findings(e)), STEPS, "not_object"),
    ("findings is a dict", "multi", _raw(lambda e: {"episode_id": EP, "findings": {"0": T1}}), STEPS, "findings_not_list"),
    ("huge payload in an ignored key", "multi",
     _raw(lambda e: dict(payload(e, oracle_findings(e)), pad="x" * 262144)), STEPS, "message_too_large"),
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


EPISODES = {"multi": multi_episode, "tiny": tiny_episode, "clean": clean_episode,
            "empty_set": empty_set_episode, "twins": twins_episode, "no_superset_key": no_superset_key_episode}


def expected_reward(expected, steps: int) -> Fraction:
    """Independent copy of CASE §05 on exact fractions."""
    if isinstance(expected, str):
        return Fraction(-1)
    tp, fp, fn = expected
    f1 = Fraction(1) if tp + fp + fn == 0 else Fraction(2 * tp, 2 * tp + fp + fn)
    return min(max(2 * f1 - 1 - Fraction(steps, 500), Fraction(-1)), Fraction(1))


def literal_reward(expected, steps: int) -> float:
    """CASE §05 typed literally in floats: max(-1.0, min(1.0, 2*f1 - 1 - 0.002*steps))."""
    if isinstance(expected, str):
        return -1.0
    tp, fp, fn = expected
    f1 = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return max(-1.0, min(1.0, 2 * f1 - 1 - 0.002 * steps))


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
        # A structure error: reward -1, nothing found, every truth issue missed (CONTRACTS §6).
        issues = len(episode["private"]["truth"]["issues"])
        ok = (result["structure_error"] == expected and result["reward"] == -1.0
              and (result["tp"], result["fp"], result["fn"]) == (0, 0, issues)
              and (result["precision"], result["recall"], result["f1"]) == (0.0, 0.0, 0.0))
    else:
        tp, fp, fn = expected
        ok = (result["structure_error"] is None
              and (result["tp"], result["fp"], result["fn"]) == expected
              and result["precision"] == (tp / (tp + fp) if tp + fp else 0.0)
              and result["recall"] == (tp / (tp + fn) if tp + fn else 0.0)
              and result["reward"] == literal_reward(expected, steps))
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
