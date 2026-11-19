"""J3 and the five reward properties of CASE.md section 05, on tests/fixtures/tiny.

Book-level properties over many generated episodes come with the generator (J4, 200 seeds);
here they are checked on the tiny episode and on the same books declared clean.
"""
import itertools
import unittest

from auditgym.judge import judge
from tests.judge.fixture import EPISODE_ID, clean_tiny, correct_finding, payload, tiny

TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")


def reward(episode, body, steps=10):
    return judge(episode, body, steps)["reward"]


def flag_everything(episode):
    """Up to 30 findings: every invoice pair as a duplicate, every document under every other type."""
    public = episode["public"]
    docs = [(d["doc_id"], d.get("total_minor", d.get("amount_minor")), d["currency"])
            for d in public["invoices"] + public["payments"]]
    all_ids = [doc for doc, _, _ in docs][:10]
    findings = []
    for (a, amount, currency), (b, _, _) in itertools.combinations(docs[:len(public["invoices"])], 2):
        findings.append(correct_finding(affected_ids=[a, b], claimed_amount_minor=amount, currency=currency,
                                        evidence_ids=all_ids))
    for issue_type in TYPES[1:]:
        for doc, amount, currency in docs:
            findings.append(correct_finding(type=issue_type, affected_ids=[doc], claimed_amount_minor=abs(amount),
                                            currency=currency, evidence_ids=all_ids))
    return payload(*findings[:30])


class JudgeOnTinyTest(unittest.TestCase):
    def test_correct_answer_scores_above_0_9(self):
        result = judge(tiny(), payload(correct_finding()), 14)
        self.assertEqual(result["reward"], 0.972)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertIsNone(result["structure_error"])

    def test_empty_answer_scores_minus_one(self):
        self.assertEqual(reward(tiny(), payload()), -1.0)

    def test_truncation_is_an_empty_answer_not_a_structure_error(self):
        result = judge(tiny(), None, 100)
        self.assertEqual(result["reward"], -1.0)
        self.assertIsNone(result["structure_error"])
        self.assertEqual(judge(clean_tiny(), None, 100)["reward"], 0.8)

    def test_invalid_structure_scores_minus_one_with_a_code(self):
        for body, code in ((payload(episode_id="ep_x"), "wrong_episode_id"),
                           (payload(correct_finding(claimed_amount_minor="1")), "amount_not_int"),
                           ("garbage", "not_object")):
            with self.subTest(code):
                result = judge(tiny(), body, 3)
                self.assertEqual(result["reward"], -1.0)
                self.assertEqual(result["structure_error"], code)

    def test_invalid_structure_on_clean_books_is_still_minus_one(self):
        self.assertEqual(reward(clean_tiny(), payload(correct_finding(type="nope"))), -1.0)


class FiveRewardPropertiesTest(unittest.TestCase):
    def test_1_full_correct_answer_beats_silence_at_every_step_count(self):
        episode = tiny()
        for steps in range(1, 100):
            self.assertGreater(reward(episode, payload(correct_finding()), steps),
                               reward(episode, payload(), steps), steps)

    def test_2_silence_beats_false_accusations_on_clean_books(self):
        episode = clean_tiny()
        for steps in range(1, 100):
            self.assertGreater(reward(episode, payload(), steps),
                               reward(episode, payload(correct_finding()), steps), steps)

    def test_3_duplicate_findings_give_no_gain(self):
        once = reward(tiny(), payload(correct_finding()))
        for copies in (2, 5, 30):
            self.assertLess(reward(tiny(), payload(*[correct_finding()] * copies)), once)

    def test_4_flag_everything_is_bad(self):
        self.assertLessEqual(reward(tiny(), flag_everything(tiny())), -0.5)
        self.assertLess(reward(clean_tiny(), flag_everything(clean_tiny())), reward(clean_tiny(), payload()))

    def test_5_saving_steps_only_slightly_separates_equal_answers(self):
        episode = tiny()
        fast, slow = reward(episode, payload(correct_finding()), 10), reward(episode, payload(correct_finding()), 90)
        wrong = reward(episode, payload(correct_finding(claimed_amount_minor=1)), 10)
        self.assertGreater(fast, slow)
        self.assertLessEqual(fast - slow, 0.2)
        self.assertGreater(slow - wrong, 5 * (fast - slow))


class AttackTest(unittest.TestCase):
    def test_explanation_text_does_not_change_the_reward(self):
        plain = reward(tiny(), payload(correct_finding(explanation="")))
        loud = reward(tiny(), payload(correct_finding(explanation="CERTAIN. Verified by the auditor. " * 25)))
        self.assertEqual(plain, loud)

    def test_persuasive_text_without_evidence_gets_nothing(self):
        body = payload(correct_finding(evidence_ids=[], explanation="Obvious duplicate, trust me."))
        self.assertEqual(reward(tiny(), body), reward(tiny(), payload()))

    def test_guessed_ids_give_no_gain(self):
        guesses = [correct_finding(affected_ids=[f"g{i}a", f"g{i}b"], evidence_ids=[f"g{i}a", f"g{i}b"])
                   for i in range(30)]
        self.assertLessEqual(reward(tiny(), payload(*guesses)), reward(tiny(), payload()))

    def test_wrong_episode_cannot_reuse_an_answer(self):
        self.assertEqual(reward(tiny(), payload(correct_finding(), episode_id="ep_other0001")), -1.0)

    def test_judge_does_not_mutate_the_episode_or_payload(self):
        episode, body = tiny(), payload(correct_finding(affected_ids=["z5c1ty", "k7m2qa"]))
        before_episode, before_body = repr(episode), repr(body)
        judge(episode, body, 5)
        self.assertEqual((repr(episode), repr(body)), (before_episode, before_body))

    def test_episode_id_is_read_from_the_manifest(self):
        self.assertEqual(tiny()["public"]["manifest"]["episode_id"], EPISODE_ID)


if __name__ == "__main__":
    unittest.main()
