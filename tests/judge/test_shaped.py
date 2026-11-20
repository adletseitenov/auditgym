"""The experimental shaped reward: formula, one-to-one credit, and where it may appear."""
import tempfile
import unittest
from fractions import Fraction

from auditgym import Env
from auditgym.generate import generate
from auditgym.judge import judge
from auditgym.shaped import INFO_KEY, WEIGHTS, pair_credit, shaped_reward
from auditgym.matcher import known_ids
from policies.runner import make_policy, read_results, run_episode
from tests.judge.helpers import correct, episode, oracle, submit
from tests.env.helpers import WITH_ERRORS

CONFIG = {"errors": {"duplicate_invoice": 1, "allocation_gap": 1}, "currencies": ["USD"]}


class FormulaTest(unittest.TestCase):
    def setUp(self):
        self.ep = episode(4)
        self.issues = self.ep["private"]["truth"]["issues"]

    def shaped(self, findings, steps=0):
        return shaped_reward(self.ep, submit(self.ep, findings), steps)

    def test_weights_sum_to_one(self):
        self.assertEqual(sum(WEIGHTS.values()), 10)

    def test_equals_official_when_credit_is_binary(self):
        cases = [oracle(self.ep), oracle(self.ep)[:2], [], oracle(self.ep) + [dict(correct(self.issues[0]), type="fx_error")]]
        for findings in cases:
            for steps in (0, 7, 100):
                official = judge(self.ep, submit(self.ep, findings), steps)["reward"]
                with self.subTest(n=len(findings), steps=steps):
                    self.assertEqual(self.shaped(findings, steps), official)

    def test_oracle_is_one_and_steps_cost_the_same(self):
        self.assertEqual(self.shaped(oracle(self.ep)), 1.0)
        self.assertEqual(self.shaped(oracle(self.ep), 10), judge(self.ep, submit(self.ep, oracle(self.ep)), 10)["reward"])

    def test_partial_credit_by_component(self):
        policy, known = self.ep["public"]["policy"], known_ids(self.ep["public"])
        issue = self.issues[0]
        good = correct(issue)
        self.assertEqual(pair_credit(good, issue, policy, known), 1)
        self.assertEqual(pair_credit(dict(good, evidence_ids=["x00"]), issue, policy, known), Fraction(8, 10))
        self.assertEqual(pair_credit(dict(good, claimed_amount_minor=1), issue, policy, known), Fraction(8, 10))
        self.assertEqual(pair_credit(dict(good, affected_ids=[issue["affected_ids"][0]]), issue, policy, known),
                         Fraction(6, 10))
        self.assertEqual(pair_credit(dict(good, affected_ids=[issue["affected_ids"][0]], claimed_amount_minor=1,
                                          evidence_ids=["x00"]), issue, policy, known), Fraction(2, 10))
        self.assertEqual(pair_credit(dict(good, type="fx_error"), issue, policy, known), 0)  # wrong type
        self.assertEqual(pair_credit(dict(good, affected_ids=["f00"]), issue, policy, known), 0)  # unrelated

    def test_partial_answer_lies_between_empty_and_oracle(self):
        empty = self.shaped([])
        partial = self.shaped([dict(correct(self.issues[0]), claimed_amount_minor=1)])
        better = self.shaped([correct(self.issues[0])])
        self.assertEqual(empty, -1.0)
        self.assertLess(empty, partial)
        self.assertLess(partial, better)
        self.assertLess(better, self.shaped(oracle(self.ep)))
        # the official reward cannot tell the empty answer from the near miss
        official = judge(self.ep, submit(self.ep, [dict(correct(self.issues[0]), claimed_amount_minor=1)]), 0)["reward"]
        self.assertEqual(official, -1.0)
        self.assertLess(official, partial)

    def test_one_to_one_and_false_findings_dilute(self):
        issue = self.issues[0]
        twice = [dict(correct(issue), claimed_amount_minor=1)] * 2  # one issue cannot be paid twice
        once = [dict(correct(issue), claimed_amount_minor=1)]
        self.assertLess(self.shaped(twice), self.shaped(once))

    def test_invalid_structure_is_minus_one(self):
        self.assertEqual(shaped_reward(self.ep, {}, 0), -1.0)
        self.assertEqual(shaped_reward(self.ep, submit(self.ep, [{"type": "fx_error"}]), 0), -1.0)

    def test_truncated_and_clean(self):
        self.assertEqual(shaped_reward(self.ep, None, 100), judge(self.ep, None, 100)["reward"])
        clean = episode(0)
        self.assertEqual(shaped_reward(clean, submit(clean, []), 0), 1.0)
        self.assertEqual(shaped_reward(clean, submit(clean, [correct(self.issues[0])]), 0), -1.0)

    def test_bad_steps(self):
        with self.assertRaises(ValueError):
            shaped_reward(self.ep, None, -1)


class ThroughEnvTest(unittest.TestCase):
    def run_oracle(self, **options):
        mode = options.pop("mode", "train")
        episode_data = generate(5, CONFIG)
        with Env(mode=mode, **options) as env:
            return run_episode(env, make_policy("oracle", truth=episode_data["private"]["truth"]), 5, CONFIG)

    def test_off_by_default_in_train(self):
        info = self.run_oracle()["info"]
        self.assertNotIn(INFO_KEY, info)
        self.assertEqual(set(info), {"reward", "precision", "recall", "f1", "inspection_steps"})

    def test_on_in_train_when_asked(self):
        result = self.run_oracle(shaped_reward=True)
        self.assertEqual(result["info"][INFO_KEY], 1.0)
        self.assertEqual(result["reward"], 1.0)  # the official reward is untouched

    def test_never_in_eval(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = scratch + "/results.jsonl"
            result = self.run_oracle(mode="eval", results_path=path, shaped_reward=True)
            self.assertEqual(result["info"], {})
            self.assertEqual(result["reward"], 0.0)
            logged = read_results(path)
        self.assertEqual(len(logged), 1)
        self.assertNotIn(INFO_KEY, logged[0])
        self.assertNotIn("shaped", "".join(logged[0]))

    def test_train_journal_is_unchanged_by_the_option(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = scratch + "/results.jsonl"
            self.run_oracle(results_path=path, shaped_reward=True)
            self.assertNotIn("shaped", "".join(read_results(path)[0]))

    def test_order_on_a_book_with_errors(self):
        data = generate(8, WITH_ERRORS)
        values = {}
        with Env(shaped_reward=True) as env:
            for name in ("oracle", "silence"):
                result = run_episode(env, make_policy(name, truth=data["private"]["truth"]), 8, WITH_ERRORS)
                values[name] = (result["reward"], result["info"][INFO_KEY])
        self.assertEqual(values["oracle"], (1.0, 1.0))
        self.assertEqual(values["silence"][1], values["silence"][0])
        self.assertLess(values["silence"][1], 0.0)
        # a partly right answer (every finding right except its evidence) sits strictly between
        truth = {"issues": [dict(issue, evidence_sets=[["no_such_id"]]) for issue in data["private"]["truth"]["issues"]]}
        with Env(shaped_reward=True) as env:
            partial = run_episode(env, make_policy("oracle", truth=truth), 8, WITH_ERRORS)
        self.assertEqual(partial["reward"], -1.0)  # the official reward sees nothing
        self.assertLess(values["silence"][1], partial["info"][INFO_KEY])
        self.assertLess(partial["info"][INFO_KEY], 1.0)


if __name__ == "__main__":
    unittest.main()
