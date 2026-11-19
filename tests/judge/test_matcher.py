"""J3: matcher and judge on the tiny fixture and hand-built episodes."""
import copy
import pathlib
import unittest

from auditgym import matcher
from auditgym.findings import parse_findings
from auditgym.judge import judge
from auditgym.rng import Rng
from auditgym.storage import load_episode
from tests.judge import helpers

TINY = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "tiny"
KEYS = {"reward", "precision", "recall", "f1", "tp", "fp", "fn", "structure_error"}


def normalized(ep, findings):
    return parse_findings(helpers.submit(ep, findings), ep["public"]["manifest"]["episode_id"])


class TinyFixtureTest(unittest.TestCase):
    def setUp(self):
        self.ep = load_episode(str(TINY))
        self.answer = helpers.submit(self.ep, helpers.oracle(self.ep))

    def test_correct_answer_above_0_9(self):
        for steps in range(1, 50):  # 50 steps give exactly 0.9
            result = judge(self.ep, self.answer, steps)
            self.assertEqual(set(result), KEYS)
            self.assertGreater(result["reward"], 0.9)
            self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))

    def test_empty_answer_is_minus_one(self):
        for steps in (1, 50, 100):
            self.assertEqual(judge(self.ep, helpers.submit(self.ep, []), steps)["reward"], -1.0)
        truncated = judge(self.ep, None, 100)
        self.assertEqual((truncated["reward"], truncated["fn"], truncated["structure_error"]), (-1.0, 1, None))

    def test_structure_error_shape(self):
        result = judge(self.ep, {"episode_id": "nope", "findings": []}, 5)
        self.assertEqual(result, {"reward": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                                  "tp": 0, "fp": 0, "fn": 1, "structure_error": "wrong_episode_id"})

    def test_judge_does_not_mutate_inputs(self):
        before = copy.deepcopy((self.ep, self.answer))
        judge(self.ep, self.answer, 3)
        self.assertEqual((self.ep, self.answer), before)

    def test_rejects_bad_step_count(self):
        for steps in (-1, 1.0, True, None):
            with self.assertRaises(ValueError):
                judge(self.ep, self.answer, steps)


class KeyingTest(unittest.TestCase):
    """With canonical affected_ids a finding can satisfy at most one issue."""

    def test_truth_keys_unique_and_at_most_one_candidate(self):
        ep = helpers.episode(12)
        issues = ep["private"]["truth"]["issues"]
        keys = [matcher.truth_key(i) for i in issues]
        self.assertEqual(len(keys), len(set(keys)))
        known = matcher.known_ids(ep["public"])
        policy = ep["public"]["policy"]
        for finding in normalized(ep, helpers.oracle(ep)):
            candidates = [i for i in issues if matcher.is_match(finding, i, policy, known)]
            self.assertEqual(len(candidates), 1)
            # Same type and same id set already pins the issue down.
            pinned = [i for i in issues if matcher.same_type(finding, i) and matcher.same_affected_ids(finding, i)]
            self.assertEqual(pinned, candidates)

    def test_overlapping_issues_split_by_key(self):
        # Two issues share a document but differ in the id set or the type: no cross matches.
        ep = helpers.episode(0)
        ep["public"]["invoices"] += [helpers._doc(d) for d in ("p", "q", "s")]
        ep["public"]["ledger"] += [helpers._row("rr")]
        ep["private"]["truth"]["issues"] = [
            {"issue_id": "u1", "type": "duplicate_invoice", "affected_ids": ["p", "q"],
             "expected_amount_minor": 5, "currency": "KZT", "evidence_sets": [["p", "q"]]},
            {"issue_id": "u2", "type": "period_cutoff", "affected_ids": ["p"],
             "expected_amount_minor": 5, "currency": "KZT", "evidence_sets": [["p", "rr"]]},
            {"issue_id": "u3", "type": "duplicate_invoice", "affected_ids": ["p", "s"],
             "expected_amount_minor": 5, "currency": "KZT", "evidence_sets": [["p", "s"]]},
        ]
        # Evidence that covers u1, u2 and u3 at once.
        greedy = {"type": "duplicate_invoice", "affected_ids": ["p", "q"], "claimed_amount_minor": 5,
                  "currency": "KZT", "evidence_ids": ["p", "q", "s", "rr"], "explanation": ""}
        result = matcher.match(ep["private"]["truth"], normalized(ep, [greedy]), ep["public"])
        self.assertEqual(result, {"tp": 1, "fp": 0, "fn": 2, "pairs": [[0, "u1"]]})

    def test_first_of_two_identical_findings_wins(self):
        ep = helpers.episode(1)
        f = helpers.oracle(ep)[0]
        result = matcher.match(ep["private"]["truth"], normalized(ep, [helpers.false_accusation(0), f, f]),
                               ep["public"])
        self.assertEqual(result["pairs"], [[1, "t00"]])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 0))


class MaximumMatchingTest(unittest.TestCase):
    """Review finding: with two truth issues on one key, a greedy pass made TP depend on order."""

    def setUp(self):
        self.ep = helpers.episode(0)
        self.ep["public"]["invoices"] += [helpers._doc("p")]
        self.ep["public"]["ledger"] += [helpers._row("r1"), helpers._row("r2")]
        self.ep["private"]["truth"]["issues"] = [
            {"issue_id": "t1", "type": "fx_error", "affected_ids": ["p"], "expected_amount_minor": 5,
             "currency": "KZT", "evidence_sets": [["p", "r1"]]},
            {"issue_id": "t2", "type": "fx_error", "affected_ids": ["p"], "expected_amount_minor": 5,
             "currency": "KZT", "evidence_sets": [["p", "r2"]]},
        ]

    def finding(self, evidence):
        return {"type": "fx_error", "affected_ids": ["p"], "claimed_amount_minor": 5, "currency": "KZT",
                "evidence_ids": evidence, "explanation": ""}

    def test_counts_do_not_depend_on_order_when_keys_repeat(self):
        wide, narrow = self.finding(["p", "r1", "r2"]), self.finding(["p", "r1"])
        truth = self.ep["private"]["truth"]
        forward = matcher.match(truth, normalized(self.ep, [wide, narrow]), self.ep["public"])
        backward = matcher.match(truth, normalized(self.ep, [narrow, wide]), self.ep["public"])
        self.assertEqual((forward["tp"], forward["fp"], forward["fn"]), (2, 0, 0))
        self.assertEqual((backward["tp"], backward["fp"], backward["fn"]), (2, 0, 0))
        self.assertEqual(forward["pairs"], [[0, "t2"], [1, "t1"]])

    def test_one_finding_still_takes_one_issue(self):
        result = matcher.match(self.ep["private"]["truth"],
                               normalized(self.ep, [self.finding(["p", "r1", "r2"])] * 3), self.ep["public"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 0))


class SupersetDefaultTest(unittest.TestCase):
    def test_missing_policy_key_means_allowed_as_the_task_text_says(self):
        ep = helpers.episode(1)
        del ep["public"]["policy"]["evidence_superset_allowed"]
        f = helpers.oracle(ep)[0]
        f["evidence_ids"] = f["evidence_ids"] + ["f00"]
        self.assertEqual(judge(ep, helpers.submit(ep, [f]), 1)["tp"], 1)


class OrderIndependenceTest(unittest.TestCase):
    """TP/FP/FN and reward do not depend on the order of findings, issues, records or ids."""

    def test_permutations(self):
        rng = Rng(11, "matcher-order")
        for n in (1, 3, 12):
            ep = helpers.episode(n)
            findings = helpers.oracle(ep)[: n - 1 if n > 1 else 1]
            findings += [helpers.false_accusation(k) for k in range(3)] + [helpers.oracle(ep)[0]]
            base = judge(ep, helpers.submit(ep, findings), 17)
            for _ in range(30):
                shuffled = copy.deepcopy(ep)
                for table in ("ledger", "invoices"):
                    rng.shuffle(shuffled["public"][table])
                rng.shuffle(shuffled["private"]["truth"]["issues"])
                for issue in shuffled["private"]["truth"]["issues"]:
                    rng.shuffle(issue["affected_ids"])
                    for evidence in issue["evidence_sets"]:
                        rng.shuffle(evidence)
                answer = copy.deepcopy(findings)
                rng.shuffle(answer)
                for f in answer:
                    rng.shuffle(f["affected_ids"])
                    rng.shuffle(f["evidence_ids"])
                result = judge(shuffled, helpers.submit(shuffled, answer), 17)
                self.assertEqual(result, base)

    def test_id_renaming_keeps_reward(self):
        ep = helpers.episode(3)
        answer = helpers.oracle(ep) + [helpers.false_accusation(1)]
        base = judge(ep, helpers.submit(ep, answer), 9)
        rename = lambda s: "zz" + s[::-1]  # noqa: E731
        renamed = copy.deepcopy(ep)
        for row in renamed["public"]["ledger"]:
            row["row_id"] = rename(row["row_id"])
        for doc in renamed["public"]["invoices"]:
            doc["doc_id"] = rename(doc["doc_id"])
        for issue in renamed["private"]["truth"]["issues"]:
            issue["affected_ids"] = [rename(i) for i in issue["affected_ids"]]
            issue["evidence_sets"] = [[rename(i) for i in s] for s in issue["evidence_sets"]]
        moved = copy.deepcopy(answer)
        for f in moved:
            f["affected_ids"] = [rename(i) for i in f["affected_ids"]]
            f["evidence_ids"] = [rename(i) for i in f["evidence_ids"]]
        self.assertEqual(judge(renamed, helpers.submit(renamed, moved), 9), base)


class PolicyTest(unittest.TestCase):
    def test_superset_switch(self):
        for allowed, tp in ((True, 1), (False, 0)):
            ep = helpers.episode(1, superset_allowed=allowed)
            f = helpers.oracle(ep)[0]
            f["evidence_ids"] = f["evidence_ids"] + ["f00"]
            self.assertEqual(judge(ep, helpers.submit(ep, [f]), 1)["tp"], tp)
            exact = helpers.oracle(ep)[0]
            self.assertEqual(judge(ep, helpers.submit(ep, [exact]), 1)["tp"], 1)

    def test_known_ids_by_table(self):
        ep = helpers.episode(1)
        ep["public"]["payments"] = [{"doc_id": "pay1", "payment_id": "pid"}]
        ep["public"]["fx"] = [{"date": "2026-10-01", "source_currency": "USD", "target_currency": "KZT", "rate": "1"}]
        known = matcher.known_ids(ep["public"])
        for present in ("r00", "a00", "pay1", "2026-10-01|USD|KZT"):
            self.assertIn(present, known)
        for absent in ("pid", "inva00", "e", "2026-10-01", "USD"):
            self.assertNotIn(absent, known)


if __name__ == "__main__":
    unittest.main()
