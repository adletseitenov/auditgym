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
    """N invoices, n_issues single-document period_cutoff issues (doc + its own row)."""
    docs = [f"d{k:03d}" for k in range(n_docs)]
    issues = [{"issue_id": f"t{k:03d}", "type": "period_cutoff", "affected_ids": [docs[k]],
               "expected_amount_minor": 100 + k, "currency": "KZT",
               "evidence_sets": [[docs[k], f"r{k:03d}"]]} for k in range(n_issues)]
    public = {"manifest": {"episode_id": "ep_syn"},
              "policy": {"amount_tolerance_minor": {t: 0 for t in TYPES}, "evidence_superset_allowed": True},
              "ledger": [{"row_id": f"r{k:03d}"} for k in range(n_docs)],
              "invoices": [{"doc_id": d} for d in docs], "payments": [], "fx": []}
    return {"public": public, "private": {"truth": {"issues": issues}}}


def _flag(doc: str, kind: str) -> dict:
    return {"type": kind, "affected_ids": [doc], "claimed_amount_minor": 0, "currency": "KZT",
            "evidence_ids": [doc], "explanation": ""}


# --- checks -----------------------------------------------------------------------------

def formulas_match_the_case() -> Optional[str]:
    from auditgym.reward import metrics_exact, terminal_reward

    cases = {(0, 0, 0): 1, (0, 2, 0): 0, (0, 0, 3): 0, (1, 1, 0): Fraction(2, 3), (3, 1, 1): Fraction(3, 4)}
    for counts, f1 in cases.items():
        if metrics_exact(*counts)["f1"] != f1:
            return f"F1{counts} != {f1}"
    if metrics_exact(0, 0, 3)["precision"] != 0 or metrics_exact(0, 2, 0)["recall"] != 0:
        return "zero denominator must give 0"
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


def flag_everything_below_silence() -> Optional[str]:
    # Blind flagging (every document, every type, capped at 30) never beats silence, and is
    # strictly below it on clean books. Even an attacker whose flags contain every issue
    # perfectly (TP = T) stays negative under the density limit T <= N/5 (threshold 501/1499).
    for n_docs in range(5, 61, 5):
        for n_issues in (0, n_docs // 5):
            episode = _synthetic(n_issues, n_docs)
            docs = [d["doc_id"] for d in episode["public"]["invoices"]]
            blind = [_flag(d, t) for d in docs for t in TYPES]
            capped = blind[:30]
            for s in STEPS:
                silence = _judge(episode, [], s)["reward"]
                for answer in (blind, capped):
                    got = _judge(episode, answer, s)["reward"]
                    if got > silence or (n_issues == 0 and got >= silence):
                        return f"N={n_docs} T={n_issues}: flag-all {got} vs silence {silence}"
                truth = episode["private"]["truth"]["issues"]
                lucky = [{"type": i["type"], "affected_ids": i["affected_ids"],
                          "claimed_amount_minor": i["expected_amount_minor"], "currency": "KZT",
                          "evidence_ids": i["evidence_sets"][0], "explanation": ""} for i in truth]
                lucky += [_flag(d, "period_cutoff") for d in docs[n_issues:]][: 30 - len(lucky)]
                if n_issues and len(lucky) < 30 and _judge(episode, lucky, s)["reward"] >= 0:
                    return f"N={n_docs} T={n_issues}: optimistic flag-all is not negative"
    return None


def saving_steps_is_slight() -> Optional[str]:
    episode = EPISODES["multi"]()
    for answer in (oracle_findings(episode), oracle_findings(episode)[:2], []):
        rewards = [_judge(episode, answer, s)["reward"] for s in STEPS]
        if any(a < b for a, b in zip(rewards, rewards[1:])):
            return "more steps gave a higher reward"
        if rewards[0] - rewards[-1] > 0.198 + 1e-12:
            return f"step spread {rewards[0] - rewards[-1]} above 0.198"
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
    return _rows(("wrong episode_id", "payload is a list", "findings is a dict", "unknown type",
                  "negative amount", "bool amount", "float amount 125000.0", "string amount",
                  "missing explanation", "id that is not a string", "huge payload in an ignored key",
                  "NaN in an ignored key"))


def limits_30_10_1000() -> Optional[str]:
    return _rows(("30 findings (4 right + 26 junk)", "31 findings",
                  "10 evidence ids (full set + existing extras)", "11 evidence ids",
                  "explanation 1000 code points, 4-byte chars", "explanation 1001 code points, 2-byte chars"))


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


CHECKS: list[tuple[str, Callable[[], Optional[str]]]] = [
    ("F1 and reward formulas match the case, including the empty cases", formulas_match_the_case),
    ("full correct answer beats silence on books with errors, for every step count", correct_beats_silence),
    ("silence beats false accusations on clean books", silence_beats_false_accusations),
    ("duplicate findings give no gain", duplicates_give_no_gain),
    ("flag-everything scores below silence on every valid config", flag_everything_below_silence),
    ("saving steps separates equal answers only slightly", saving_steps_is_slight),
    ("one-to-one matching is deterministic", matching_is_deterministic),
    ("wrong type, ids, amount beyond tolerance, currency or evidence is not a TP", wrong_fields_are_not_tp),
    ("invalid structure ends the episode with reward -1 and a judge log entry", invalid_structure_is_minus_one),
    ("limits: 30 findings, 10 evidence ids, 1000 characters", limits_30_10_1000),
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
