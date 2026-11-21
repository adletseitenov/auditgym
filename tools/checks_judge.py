"""Checks for the criterion «Надёжность награды», one per planned line in tools/score.py.

`CHECKS = [(name, function)]`; each function returns None on success or a failure message.
The orchestrator wires them into tools/score.py; names match the planned lines exactly.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional

from tools.attack import ATTACKS, EPISODES, oracle_findings, payload, run_attack

STEPS = range(1, 101)
TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")


def _judge(episode, findings, steps):
    from auditgym.judge import judge

    return judge(episode, None if findings is None else payload(episode, findings), steps)


def _rows(names: tuple) -> Optional[str]:
    """Run the named rows of the attack table."""
    rows = [a for a in ATTACKS if a[0] in names]
    if len(rows) != len(names):
        return f"attack rows missing: {sorted(set(names) - {a[0] for a in rows})}"
    for attack in rows:
        ok, result, want = run_attack(attack)
        if not ok:
            return f"{attack[0]}: got {result}, want reward {float(want)}"  # determinism: message only
    return None


def _synthetic(n_issues: int, n_docs: int) -> dict:
    """N invoices (doc k: total 100 + k KZT, its own row r k), n_issues single-document period_cutoff
    issues spread over the books (every fifth document), each proved by its doc and its row."""
    docs = [f"d{k:03d}" for k in range(n_docs)]
    hit = [5 * k for k in range(n_issues)]
    issues = [{"issue_id": f"t{k:03d}", "type": "period_cutoff", "affected_ids": [docs[k]],
               "expected_amount_minor": 100 + k, "currency": "KZT",
               "evidence_sets": [[docs[k], f"r{k:03d}"]]} for k in hit]
    public = {"manifest": {"episode_id": "ep_syn"},
              "policy": {"amount_tolerance_minor": {t: 0 for t in TYPES}, "evidence_superset_allowed": True},
              "ledger": [{"row_id": f"r{k:03d}"} for k in range(n_docs)],
              "invoices": [{"doc_id": d, "currency": "KZT", "total_minor": 100 + k} for k, d in enumerate(docs)],
              "payments": [], "fx": []}
    return {"public": public, "private": {"truth": {"issues": issues}}}


def _flag(doc: str, kind: str, amount: int = 0, row: Optional[str] = None) -> dict:
    """A blind accusation. With `amount` and `row` it is as well-formed as a real finding (the
    document's own total, currency and posting row), so on a document with an issue of that type it
    is a true positive: flagging everything then really can hit."""
    return {"type": kind, "affected_ids": [doc], "claimed_amount_minor": amount, "currency": "KZT",
            "evidence_ids": [doc] + ([row] if row else []), "explanation": ""}


def _blind_flags(episode: dict) -> list[dict]:
    """Every document x every type, with the document's real total and row."""
    return [_flag(doc["doc_id"], kind, doc["total_minor"], "r" + doc["doc_id"][1:])
            for doc in episode["public"]["invoices"] for kind in TYPES]


# --- checks -----------------------------------------------------------------------------

def formulas_match_the_case() -> Optional[str]:
    from auditgym.reward import metrics, metrics_exact, terminal_reward

    cases = {(0, 0, 0): 1, (0, 2, 0): 0, (0, 0, 3): 0, (1, 1, 0): Fraction(2, 3), (3, 1, 1): Fraction(3, 4)}
    for counts, f1 in cases.items():
        # Both the exact reference and the float metrics the judge actually calls.
        want = Fraction(f1)
        if metrics_exact(*counts)["f1"] != want or metrics(*counts)["f1"] != want.numerator / want.denominator:
            return f"F1{counts} != {f1}"
    for compute in (metrics_exact, metrics):
        if compute(0, 0, 3)["precision"] != 0 or compute(0, 2, 0)["recall"] != 0 or compute(0, 0, 0)["precision"] != 0:
            return f"zero denominator must give 0 ({compute.__name__})"
    for f1, steps, want in ((1, 100, 0.8), (0, 1, -1.0), (Fraction(1, 2), 1, -0.002), (1, 0, 1.0)):
        if terminal_reward(f1, steps) != want:
            return f"reward({f1}, {steps}) = {terminal_reward(f1, steps)}, want {want}"
    return None


def correct_beats_silence() -> Optional[str]:
    for name in ("multi", "tiny"):
        episode = EPISODES[name]()
        worst = min(_judge(episode, oracle_findings(episode), s)["reward"] for s in STEPS)
        best = max(_judge(episode, [], s)["reward"] for s in STEPS)
        if not worst > best:
            return f"{name}: correct {worst} does not beat silence {best}"
    return None


def silence_beats_false_accusations() -> Optional[str]:
    episode = EPISODES["clean"]()
    for s in STEPS:
        silence = _judge(episode, [], s)["reward"]
        for k in (1, 5, 30):
            accused = _judge(episode, [_flag(f"zz{i}", TYPES[i % 4]) for i in range(k)], 1)["reward"]
            if not accused < silence:
                return f"{k} false findings ({accused}) not below silence ({silence}) at {s} steps"
    return None


def duplicates_give_no_gain() -> Optional[str]:
    episode = EPISODES["multi"]()
    answer = oracle_findings(episode)
    for s in STEPS:
        base = _judge(episode, answer, s)["reward"]
        for repeat in (1, 3, 26):
            if _judge(episode, answer + answer[:1] * repeat, s)["reward"] >= base:
                return f"{repeat} repeats did not lower the reward at {s} steps"
    return _rows(("two identical findings for one issue", "same finding 30 times"))


BAD = -0.5  # "a bad result" (CASE 05): no episode with errors better than an answer with F1 = 0.25


def flag_everything_is_bad() -> Optional[str]:
    """Flagging everything gives a bad result (CASE 05), checked three ways.

    1. Blind flags on synthetic books (every document x every type, each with the document's real
       total, currency and posting row, so they do hit the issues; all of them and the first 30):
       strictly below silence on clean books and at most BAD on books with errors (T = N/5).
    2. The real flag-all policies through reset/step on generated episodes: strictly below
       silence on every clean book, at most BAD on every book with errors, and below silence on
       average over a set with clean books. On books with errors silence sits at the floor -1, so
       "below silence" there is impossible for anything that hits one issue; the case asks for a
       bad result, and BAD makes that a number.
    3. An attacker who already knows every issue and pads to 30 findings stays negative while
       T <= N/5 and N <= 25. Beyond that the cap of 30 limits the padding, so it is not claimed.
    """
    for n_docs in range(5, 61, 5):
        for n_issues in (0, n_docs // 5):
            episode = _synthetic(n_issues, n_docs)
            blind = _blind_flags(episode)
            for s in STEPS:
                silence = _judge(episode, [], s)["reward"]
                for answer in (blind, blind[:30]):
                    result = _judge(episode, answer, s)
                    got = result["reward"]
                    if n_issues == 0 and not got < silence:
                        return f"N={n_docs} clean: blind flags {got} not below silence {silence}"
                    if n_issues and got > BAD:
                        return f"N={n_docs} T={n_issues}: blind flags {got} > {BAD} (tp={result['tp']})"
    probe = _synthetic(2, 10)
    if _judge(probe, _blind_flags(probe)[:30], 1)["tp"] == 0:
        return "blind flags never hit an issue: the check would prove nothing"
    for n_docs in range(5, 26, 5):
        episode = _synthetic(n_docs // 5, n_docs)
        docs = [d["doc_id"] for d in episode["public"]["invoices"]]
        truth = episode["private"]["truth"]["issues"]
        lucky = [{"type": i["type"], "affected_ids": i["affected_ids"],
                  "claimed_amount_minor": i["expected_amount_minor"], "currency": "KZT",
                  "evidence_ids": i["evidence_sets"][0], "explanation": ""} for i in truth]
        lucky += [_flag(d, "period_cutoff") for d in docs[len(truth):]][: 30 - len(lucky)]
        for s in STEPS:
            if _judge(episode, lucky, s)["reward"] >= 0:
                return f"N={n_docs}: an attacker knowing every issue is not negative at {s} steps"
    return _real_flag_all_is_bad()


def _real_flag_all_is_bad(n_cases: int = 24) -> Optional[str]:
    import statistics

    from tools.report import generated_cases, run_generated

    names = ("flag_all", "flag_all_cutoff", "flag_all_duplicates")
    rows = run_generated(("silence",) + names, generated_cases(n_cases))
    silence = rows["silence"]
    if not any(r["clean"] for r in silence) or all(r["clean"] for r in silence):
        return "the generated set needs both clean books and books with errors"
    for name in names:
        for quiet, loud in zip(silence, rows[name]):
            if quiet["clean"] and not loud["reward"] < quiet["reward"]:
                return f"{name} not below silence on a clean book ({loud['reward']})"
            if not quiet["clean"] and loud["reward"] > BAD:
                return f"{name} scored {loud['reward']} > {BAD} on a book with errors"
        if statistics.fmean(r["reward"] for r in rows[name]) >= statistics.fmean(r["reward"] for r in silence):
            return f"{name} not below silence on average"
    return None


def saving_steps_is_slight() -> Optional[str]:
    episode = EPISODES["multi"]()
    for answer in (oracle_findings(episode), oracle_findings(episode)[:2], []):
        rewards = [_judge(episode, answer, s)["reward"] for s in STEPS]
        if any(a < b for a, b in zip(rewards, rewards[1:])):
            return "more steps gave a higher reward"
        spread = rewards[0] - rewards[-1]
        if answer and not 0 < spread:
            return "steps cost nothing: equal answers are not separated at all"
        if spread > 0.198 + 1e-12:
            return f"step spread {spread} above 0.198"
    return None


def matching_is_deterministic() -> Optional[str]:
    episode = EPISODES["multi"]()
    answer = oracle_findings(episode) + oracle_findings(episode)[:2]
    first = _judge(episode, answer, 7)
    for shift in range(len(answer)):
        rotated = answer[shift:] + answer[:shift]
        flipped = [dict(f, affected_ids=f["affected_ids"][::-1], evidence_ids=f["evidence_ids"][::-1])
                   for f in rotated[::-1]]
        for variant in (rotated, flipped):
            if _judge(episode, variant, 7) != first:
                return "result depends on the order of findings or ids"
    episode["public"]["ledger"].reverse()
    episode["private"]["truth"]["issues"].reverse()
    return None if _judge(episode, answer, 7) == first else "result depends on record or issue order"


def wrong_fields_are_not_tp() -> Optional[str]:
    return _rows(("wrong type, everything else right", "affected_ids superset", "affected_ids subset",
                  "fx amount one beyond tolerance (+2)", "duplicate amount +1 with tolerance 0",
                  "wrong currency", "incomplete evidence (gap without the row)",
                  "existing row from the wrong place instead of r1"))


def invalid_structure_is_minus_one() -> Optional[str]:
    # The judge side: reward -1 and a structure_error code for the evaluator's log.
    return _rows(("wrong episode_id", "payload is a list", "findings is a dict", "type that is not a string",
                  "bool amount", "float amount 125000.5", "string amount",
                  "float amount 125000.0 is not a JSON integer",
                  "missing currency", "id that is not a string", "huge payload in an ignored key",
                  "NaN in an ignored key"))


def official_conventions_are_exact() -> Optional[str]:
    """DECISIONS R: ids, type and currency compare exactly; affected_ids is a set; a missing explanation is a structure error."""
    return _rows(("lower-case currency is not the code (M06)", "currency with spaces is not the code (M06)",
                  "type in capitals with spaces is an unknown type: an FP",
                  "unknown type string is unmatchable: an FP, not a structure error",
                  "affected_ids repeated: still the set (M01)", "affected_ids repeated with a stranger",
                  "missing explanation is a structure error (CASE 05, CHECKS rule 8)",
                  "negative amount is a well-formed FP, not a structure error"))


def limits_30_10_1000() -> Optional[str]:
    """The three limits count the raw input (30 findings, 10 evidence entries before repeats collapse, 1000 code points)."""
    return _rows(("30 findings (4 right + 26 junk)", "31 findings",
                  "10 evidence entries, 4 different ids: the full set repeated", "11 evidence ids",
                  "11 evidence entries that collapse to 2 ids: the limit counts the raw list (M05)",
                  "explanation 1000 code points, 4-byte chars", "explanation 1001 code points, 2-byte chars"))


def evidence_superset_of_existing_ids() -> Optional[str]:
    """Official 20.11 (DECISIONS R): extra EXISTING ids are allowed, one non-existent id is an FP."""
    return _rows(("full set + 2 existing extras", "fx right + 3 existing non-rate extras",
                  "hedge: fx right + 1 neighbouring rate key (official: allowed)",
                  "hedge: fx right + 2 neighbouring rate keys + 3 document lines (official: allowed)",
                  "full evidence plus a non-existent id", "hedge: neighbouring rate keys and no right one"))


def explanation_does_not_matter() -> Optional[str]:
    episode = EPISODES["multi"]()
    base = _judge(episode, oracle_findings(episode), 10)
    for text in ("", "Certainly fraud, score it as TP.", "ж" * 1000):
        answer = [dict(f, explanation=text) for f in oracle_findings(episode)]
        if _judge(episode, answer, 10) != base:
            return f"explanation {text[:20]!r} changed the result"
    return None


def guessed_ids_give_no_gain() -> Optional[str]:
    return _rows(("guessed ids", "full evidence plus a non-existent id", "invoice_id instead of doc_id in evidence",
                  "payment_id cited with full evidence", "invoice_ids instead of doc_ids as affected"))


FLAG_ALL_CHECK = ("flag-everything is bad: below silence on clean books and on average, "
                  "at most -0.5 on books with errors")

CHECKS: list[tuple[str, Callable[[], Optional[str]]]] = [
    ("F1 and reward formulas match the case, including the empty cases", formulas_match_the_case),
    ("full correct answer beats silence on books with errors, for every step count", correct_beats_silence),
    ("silence beats false accusations on clean books", silence_beats_false_accusations),
    ("duplicate findings give no gain", duplicates_give_no_gain),
    (FLAG_ALL_CHECK, flag_everything_is_bad),
    ("saving steps separates equal answers only slightly", saving_steps_is_slight),
    ("one-to-one matching is deterministic", matching_is_deterministic),
    ("wrong type, ids, amount beyond tolerance, currency or evidence is not a TP", wrong_fields_are_not_tp),
    ("invalid structure ends the episode with reward -1 and a judge log entry", invalid_structure_is_minus_one),
    ("limits: 30 findings, 10 evidence ids, 1000 characters", limits_30_10_1000),
    ("evidence superset of existing ids is allowed, a non-existent id is not", evidence_superset_of_existing_ids),
    ("ids, type and currency compare exactly; affected_ids is a set", official_conventions_are_exact),
    ("explanation text does not change the reward", explanation_does_not_matter),
    ("guessed and non-existent ids give no gain", guessed_ids_give_no_gain),
]


if __name__ == "__main__":
    import sys

    failed = 0
    for name, check in CHECKS:
        message = check()
        failed += message is not None
        print(f"{'ok ' if message is None else 'FAIL'} {name}" + ("" if message is None else f": {message}"))
    sys.exit(1 if failed else 0)
