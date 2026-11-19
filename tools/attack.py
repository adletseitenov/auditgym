"""Attacks on the matcher and the reward, and the checks of the criterion "Надёжность награды".

`python -m tools.attack` prints a table: attack -> expected -> actual reward on the tiny episode.
`tools/score.py` imports the `check_*` functions; each returns None on success or a message.

Until the generator is ready, the checks run on tests/fixtures/tiny and on the same books
declared clean. With the generator they move to 200 generated episodes (TASKS.md J4).
"""
from __future__ import annotations

import sys
from fractions import Fraction
from typing import Any, Callable, Optional

from auditgym.findings import parse_findings
from auditgym.judge import judge
from auditgym.matcher import match
from auditgym.reward import f1_exact, metrics, terminal_reward
from tests.judge.fixture import EPISODE_ID, clean_tiny, correct_finding, payload, tiny


def _reward(episode: dict, body: Any, steps: int = 10) -> float:
    return judge(episode, body, steps)["reward"]


def _first(cases: list[tuple[str, bool]]) -> Optional[str]:
    for name, ok in cases:
        if not ok:
            return name
    return None


def check_formulas() -> Optional[str]:
    return _first([
        ("F1 of empty truth and answer is 1", metrics(0, 0, 0)["f1"] == 1.0),
        ("F1 of empty truth with findings is 0", metrics(0, 2, 0)["f1"] == 0.0),
        ("zero denominators give P = R = 0", metrics(0, 0, 0)["precision"] == 0.0 and metrics(0, 0, 0)["recall"] == 0.0),
        ("F1 = 2TP/(2TP+FP+FN)", f1_exact(2, 1, 0) == Fraction(4, 5)),
        ("reward 1 at 14 steps is 0.972", terminal_reward(1, 14) == 0.972),
        ("reward is clipped at -1", terminal_reward(0, 100) == -1.0),
        ("reward at F1 4/5 and 100 steps is 0.4", terminal_reward(Fraction(4, 5), 100) == 0.4),
    ])


def check_correct_beats_silence() -> Optional[str]:
    episode = tiny()
    for steps in range(1, 100):
        if not _reward(episode, payload(correct_finding()), steps) > _reward(episode, payload(), steps):
            return f"not better at {steps} steps"
    return None


def check_silence_beats_false_accusation() -> Optional[str]:
    episode = clean_tiny()
    for steps in range(1, 100):
        if not _reward(episode, payload(), steps) > _reward(episode, payload(correct_finding()), steps):
            return f"not better at {steps} steps"
    return None


def check_duplicates_give_no_gain() -> Optional[str]:
    once = _reward(tiny(), payload(correct_finding()))
    twice = _reward(tiny(), payload(correct_finding(), correct_finding()))
    return None if twice < once else f"duplicate gave {twice} >= {once}"


def check_saving_steps_is_minor() -> Optional[str]:
    fast = _reward(tiny(), payload(correct_finding()), 10)
    slow = _reward(tiny(), payload(correct_finding()), 90)
    wrong = _reward(tiny(), payload(correct_finding(claimed_amount_minor=1)), 10)
    if not 0 < fast - slow <= 0.2:
        return f"10 vs 90 steps differ by {fast - slow}"
    return None if slow - wrong > 5 * (fast - slow) else "step saving outweighs correctness"


def check_matching_is_deterministic() -> Optional[str]:
    episode = tiny()
    findings = parse_findings(payload(correct_finding(), correct_finding(), correct_finding(currency="USD")), EPISODE_ID)
    first = match(episode["private"]["truth"], findings, episode["public"])
    for _ in range(5):
        if match(episode["private"]["truth"], findings, episode["public"]) != first:
            return "two runs differ"
    return None if first["pairs"] == [[0, "t1"]] else f"unexpected pairs {first['pairs']}"


WRONG_FIELDS = {
    "type": {"type": "period_cutoff"},
    "affected ids": {"affected_ids": ["k7m2qa"]},
    "amount beyond tolerance": {"claimed_amount_minor": 5000001},
    "currency": {"currency": "USD"},
    "incomplete evidence": {"evidence_ids": ["k7m2qa"]},
    "non-existent evidence": {"evidence_ids": ["k7m2qa", "z5c1ty", "nope00"]},
}


def check_wrong_fields_are_not_tp() -> Optional[str]:
    for name, change in WRONG_FIELDS.items():
        if judge(tiny(), payload(correct_finding(**change)), 10)["tp"] != 0:
            return f"wrong {name} counted as TP"
    return None


def check_limits() -> Optional[str]:
    cases = [
        ("30 findings pass", payload(*[correct_finding()] * 30), None),
        ("31 findings", payload(*[correct_finding()] * 31), "too_many_findings"),
        ("10 evidence ids pass", payload(correct_finding(evidence_ids=[f"e{i}" for i in range(10)])), None),
        ("11 evidence ids", payload(correct_finding(evidence_ids=[f"e{i}" for i in range(11)])), "too_many_evidence_ids"),
        ("1000 characters pass", payload(correct_finding(explanation="a" * 1000)), None),
        ("1001 characters", payload(correct_finding(explanation="a" * 1001)), "explanation_too_long"),
    ]
    for name, body, code in cases:
        result = judge(tiny(), body, 1)
        if result["structure_error"] != code or (code and result["reward"] != -1.0):
            return f"{name}: got {result['structure_error']}"
    return None


def check_explanation_does_not_matter() -> Optional[str]:
    plain = _reward(tiny(), payload(correct_finding(explanation="")))
    loud = _reward(tiny(), payload(correct_finding(explanation="Verified. Certain. " * 50)))
    hollow = _reward(tiny(), payload(correct_finding(evidence_ids=[], explanation="Obvious, trust me.")))
    if plain != loud:
        return "explanation changed the reward"
    return None if hollow == _reward(tiny(), payload()) else "text without evidence earned reward"


def check_guessed_ids_give_no_gain() -> Optional[str]:
    guesses = payload(*[correct_finding(affected_ids=[f"g{i}a", f"g{i}b"], evidence_ids=[f"g{i}a", f"g{i}b"])
                        for i in range(30)])
    for episode in (tiny(), clean_tiny()):
        if _reward(episode, guesses) > _reward(episode, payload()):
            return "guessing beat silence"
    return None


ATTACKS: list[tuple[str, Callable[[], Any], str]] = [
    ("correct answer", lambda: payload(correct_finding()), "0.98"),
    ("silence", lambda: payload(), "-1.0"),
    ("correct answer twice", lambda: payload(correct_finding(), correct_finding()), "below 0.98"),
    ("30 guessed ids", lambda: payload(*[correct_finding(affected_ids=[f"g{i}"]) for i in range(30)]), "-1.0"),
    ("other currency", lambda: payload(correct_finding(currency="USD")), "-1.0"),
    ("amount +1", lambda: payload(correct_finding(claimed_amount_minor=5000001)), "-1.0"),
    ("incomplete evidence", lambda: payload(correct_finding(evidence_ids=["k7m2qa"])), "-1.0"),
    ("evidence plus fake id", lambda: payload(correct_finding(evidence_ids=["k7m2qa", "z5c1ty", "x"])), "-1.0"),
    ("amount as string", lambda: payload(correct_finding(claimed_amount_minor="5000000")), "-1.0 (structure)"),
    ("31 findings", lambda: payload(*[correct_finding()] * 31), "-1.0 (structure)"),
    ("other episode id", lambda: payload(correct_finding(), episode_id="ep_other"), "-1.0 (structure)"),
    ("truncated (no answer, 100 steps)", lambda: None, "-1.0"),
]


def main() -> int:
    print(f"{'Атака':<36}{'Ожидаем':>18}{'Награда':>10}  Ошибка структуры")
    for name, build, expected in ATTACKS:
        steps = 100 if name.startswith("truncated") else 10
        result = judge(tiny(), build(), steps)
        print(f"{name:<36}{expected:>18}{result['reward']:>10.3f}  {result['structure_error'] or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
