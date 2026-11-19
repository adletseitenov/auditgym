"""The five reward properties of CASE §05 «Тесты должны показать», one test each.

Checked through `judge` for every step count 1..100 and truth sizes 0, 1, 3, 12.
"""
import unittest
from fractions import Fraction

from auditgym.judge import judge
from auditgym.reward import terminal_reward_exact
from tests.judge import helpers

STEPS = range(1, 101)
SIZES_WITH_ERRORS = (1, 3, 12)


def reward(ep, findings, steps):
    result = judge(ep, helpers.submit(ep, findings), steps)
    assert result["structure_error"] is None, result
    return result["reward"]


def flag_everything_threshold_holds(t: int, f: int) -> bool:
    """Reward < 0 for every step count 1..100 when F findings cover all T issues (TP = T)."""
    return 1499 * t < 501 * f


class FivePropertiesTest(unittest.TestCase):
    def test_1_full_correct_answer_beats_silence_on_books_with_errors(self):
        for n in SIZES_WITH_ERRORS:
            ep = helpers.episode(n)
            correct = [reward(ep, helpers.oracle(ep), s) for s in STEPS]
            silence = [reward(ep, [], s) for s in STEPS]
            # Worst correct (slowest) beats best silence (fastest).
            self.assertGreater(min(correct), max(silence), n)
            self.assertEqual(min(correct), 0.8)
            self.assertEqual(max(silence), -1.0)
            # Truncation (no submit, 100 steps) is silence too.
            self.assertEqual(judge(ep, None, 100)["reward"], -1.0)

    def test_2_silence_beats_false_accusations_on_clean_books(self):
        ep = helpers.episode(0)
        for s in STEPS:
            silence = reward(ep, [], s)
            self.assertEqual(silence, (500 - s) / 500)
            for k in (1, 2, 5, 30):
                accused = reward(ep, [helpers.false_accusation(i) for i in range(k)], s)
                self.assertEqual(accused, -1.0)
                for s_fast in (1, s):  # even the fastest false accusation loses
                    self.assertLess(reward(ep, [helpers.false_accusation(0)], s_fast), silence)
        self.assertEqual(judge(ep, None, 100)["reward"], 0.8)  # truncation on clean books

    def test_3_duplicates_give_no_gain(self):
        for n in (0,) + SIZES_WITH_ERRORS:
            ep = helpers.episode(n)
            oracle = helpers.oracle(ep)
            answers = [oracle, oracle[: max(n - 1, 0)], oracle[:1] + [helpers.false_accusation(0)], []]
            for answer in answers:
                if not answer:
                    continue
                for repeat in (1, 2, 5):
                    duplicated = answer + answer[:1] * repeat
                    if len(duplicated) > 30:
                        continue
                    for s in STEPS:
                        base = judge(ep, helpers.submit(ep, answer), s)
                        dup = judge(ep, helpers.submit(ep, duplicated), s)
                        self.assertEqual(dup["tp"], base["tp"])
                        self.assertLessEqual(dup["reward"], base["reward"])
                        if base["tp"] > 0 and base["reward"] > -1.0:
                            self.assertLess(dup["reward"], base["reward"])

    def test_4_flag_everything_is_bad(self):
        # Synthetic company with N invoices and a density limit stricter than the documented one
        # (<= 1 issue per 5 invoices; DECISIONS A3 allows only 1 per 8). "Flag everything": one
        # finding per invoice per type, capped at 30 by the case. Optimistic attacker: every issue among the flags is perfectly correct (TP = T).
        for n_docs in range(5, 31):
            n_issues = n_docs // 5
            ep = helpers.episode(n_issues, n_filler_docs=n_docs - 2 * n_issues)
            docs = [d["doc_id"] for d in ep["public"]["invoices"]]
            self.assertEqual(len(docs), n_docs)
            for types_per_doc in (1, 4):
                flags = helpers.oracle(ep)
                covered = {i for f in flags for i in f["affected_ids"]}
                for doc in docs:
                    for k in range(types_per_doc):
                        if doc not in covered:
                            flags.append(dict(helpers.false_accusation(k), affected_ids=[doc], evidence_ids=[doc]))
                for s in STEPS:
                    silence = judge(ep, helpers.submit(ep, []), s)["reward"]
                    result = judge(ep, helpers.submit(ep, flags), s)
                    if len(flags) > 30:
                        self.assertEqual(result["structure_error"], "too_many_findings")
                    self.assertLess(result["reward"], 0.0, (n_docs, types_per_doc, s))
                    # Blind flagging (no TP) never beats silence; on clean books it loses.
                    blind = [f for f in flags if f not in helpers.oracle(ep)][:30]
                    self.assertLessEqual(judge(ep, helpers.submit(ep, blind), s)["reward"], silence)
        clean = helpers.episode(0, n_filler_docs=30)
        for s in STEPS:
            flags = [dict(helpers.false_accusation(k), affected_ids=[d["doc_id"]])
                     for k, d in enumerate(clean["public"]["invoices"])]
            self.assertLess(reward(clean, flags, s), reward(clean, [], s))

    def test_4b_exact_density_threshold(self):
        # F findings that include all T issues correctly: F1 = 2T/(T+F), and reward < 0 for every
        # step count 1..100 iff 1499*T < 501*F, i.e. T/F < 501/1499 (just above 1/3).
        for f in range(1, 31):
            for t in range(0, f + 1):
                worst = max(terminal_reward_exact(Fraction(2 * t, t + f), s) for s in STEPS)
                self.assertEqual(worst < 0, flag_everything_threshold_holds(t, f), (t, f))
        # Under the density limit (T <= N/5) with one flag per invoice (F >= N): T/F <= 1/5.
        self.assertLess(Fraction(1, 5), Fraction(501, 1499))
        # And it stops at T/F = 501/1499, e.g. T = 501, F = 1499 (beyond the 30-finding cap).
        self.assertFalse(flag_everything_threshold_holds(501, 1499))
        self.assertTrue(flag_everything_threshold_holds(500, 1499))
        # Inside the 30-finding cap the boundary is crossed e.g. between T=9, F=28 and T=10, F=29.
        self.assertFalse(flag_everything_threshold_holds(10, 29))
        self.assertTrue(flag_everything_threshold_holds(9, 28))

    def test_5_saving_steps_separates_equal_answers_only_slightly(self):
        for n in (0,) + SIZES_WITH_ERRORS:
            ep = helpers.episode(n)
            answers = [helpers.oracle(ep), helpers.oracle(ep)[:1], [helpers.false_accusation(0)], []]
            for answer in answers:
                rewards = [reward(ep, answer, s) for s in STEPS]
                for faster, slower in zip(rewards, rewards[1:]):
                    self.assertGreaterEqual(faster, slower)
                    self.assertLessEqual(faster - slower, 0.002 + 1e-12)
                spread = rewards[0] - rewards[-1]
                self.assertLessEqual(spread, 0.198 + 1e-12)
                f1 = judge(ep, helpers.submit(ep, answer), 1)["f1"]
                if f1 == 0.0:
                    self.assertEqual(spread, 0.0)  # both clipped at -1
        # The whole step budget is worth less than the gap between correct and silence.
        ep = helpers.episode(3)
        self.assertGreater(reward(ep, helpers.oracle(ep), 100) - reward(ep, [], 1), 1.7)


if __name__ == "__main__":
    unittest.main()
