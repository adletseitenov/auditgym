"""J3: one-to-one matching rules on tests/fixtures/tiny and on small hand-written truths."""
import unittest

from auditgym.findings import parse_findings
from auditgym.matcher import known_ids, match
from tests.judge.fixture import DUP_A, DUP_B, EPISODE_ID, OTHER_INVOICE, correct_finding, payload, tiny


def run(*findings, episode=None):
    episode = episode or tiny()
    parsed = parse_findings(payload(*findings), EPISODE_ID)
    return match(episode["private"]["truth"], parsed, episode["public"])


def counts(result):
    return result["tp"], result["fp"], result["fn"]


class TinyMatchTest(unittest.TestCase):
    def test_correct_finding_is_a_true_positive(self):
        result = run(correct_finding())
        self.assertEqual(counts(result), (1, 0, 0))
        self.assertEqual(result["pairs"], [[0, "t1"]])

    def test_order_of_ids_does_not_matter(self):
        self.assertEqual(counts(run(correct_finding(affected_ids=[DUP_B, DUP_A], evidence_ids=[DUP_B, DUP_A]))), (1, 0, 0))

    def test_silence(self):
        self.assertEqual(counts(run()), (0, 0, 1))

    def test_each_wrong_field_is_a_false_positive(self):
        wrong = {
            "type": correct_finding(type="period_cutoff"),
            "affected superset": correct_finding(affected_ids=[DUP_A, DUP_B, OTHER_INVOICE]),
            "affected subset": correct_finding(affected_ids=[DUP_A]),
            "currency": correct_finding(currency="USD"),
            "amount +1 (tolerance 0)": correct_finding(claimed_amount_minor=5000001),
            "amount -1 (tolerance 0)": correct_finding(claimed_amount_minor=4999999),
            "incomplete evidence": correct_finding(evidence_ids=[DUP_A]),
            "no evidence": correct_finding(evidence_ids=[]),
            "non-existent evidence id": correct_finding(evidence_ids=[DUP_A, DUP_B, "zzzzzz"]),
            "id from another episode": correct_finding(evidence_ids=[DUP_A, DUP_B, "ep_other"]),
            "invoice_id instead of doc_id": correct_finding(affected_ids=["p3x9wd", "n8v4js"]),
        }
        for name, finding in wrong.items():
            with self.subTest(name):
                self.assertEqual(counts(run(finding)), (0, 1, 1))

    def test_complete_set_plus_existing_extra_ids_is_accepted(self):
        finding = correct_finding(evidence_ids=[DUP_A, DUP_B, "a3", "a1", "2026-10-03|USD|KZT"])
        self.assertEqual(counts(run(finding)), (1, 0, 0))

    def test_superset_is_rejected_when_policy_forbids_it(self):
        episode = tiny()
        episode["public"]["policy"]["evidence_superset_allowed"] = False
        self.assertEqual(counts(run(correct_finding(evidence_ids=[DUP_A, DUP_B, "a3"]), episode=episode)), (0, 1, 1))
        self.assertEqual(counts(run(correct_finding(), episode=episode)), (1, 0, 0))

    def test_repeat_of_a_matched_finding_is_a_false_positive(self):
        result = run(correct_finding(), correct_finding(explanation="again"))
        self.assertEqual(counts(result), (1, 1, 0))
        self.assertEqual(result["pairs"], [[0, "t1"]])
        self.assertEqual(result["reasons"], ["tp", "already_matched"])

    def test_first_passing_finding_takes_the_issue(self):
        result = run(correct_finding(claimed_amount_minor=1), correct_finding())
        self.assertEqual(counts(result), (1, 1, 0))
        self.assertEqual(result["pairs"], [[1, "t1"]])

    def test_known_ids_cover_every_id_space(self):
        ids = known_ids(tiny()["public"])
        for expected in ("a1", "k7m2qa", "p3x9wd", "h3n8vx", "y6d2ck", "e4h8rn", "2026-10-31|USD|KZT"):
            self.assertIn(expected, ids)
        self.assertNotIn("t1", ids)  # issue ids are not open records


def _episode_with(issues, tolerance=None):
    episode = tiny()
    episode["private"]["truth"]["issues"] = issues
    if tolerance is not None:
        episode["public"]["policy"]["amount_tolerance_minor"] = tolerance
    return episode


FX_ISSUE = {"issue_id": "t9", "type": "fx_error", "affected_ids": ["h3n8vx"], "expected_amount_minor": 908750,
            "currency": "KZT", "evidence_sets": [["h3n8vx", "a7"]]}


class ToleranceAndAmbiguityTest(unittest.TestCase):
    def fx(self, amount):
        return correct_finding(type="fx_error", affected_ids=["h3n8vx"], claimed_amount_minor=amount,
                               evidence_ids=["h3n8vx", "a7"])

    def test_tolerance_edge(self):
        episode = _episode_with([FX_ISSUE])
        for amount, tp in ((908750, 1), (908749, 1), (908751, 1), (908748, 0), (908752, 0)):
            with self.subTest(amount):
                self.assertEqual(run(self.fx(amount), episode=episode)["tp"], tp)

    def test_missing_tolerance_entry_means_zero(self):
        episode = _episode_with([FX_ISSUE], tolerance={})
        self.assertEqual(run(self.fx(908751), episode=episode)["tp"], 0)

    def test_truth_with_two_issues_on_one_key_is_rejected(self):
        twin = dict(FX_ISSUE, issue_id="t10")
        with self.assertRaises(ValueError):
            run(self.fx(908750), episode=_episode_with([FX_ISSUE, twin]))

    def test_one_finding_never_takes_two_issues(self):
        dup = tiny()["private"]["truth"]["issues"][0]
        result = run(correct_finding(), episode=_episode_with([dup, FX_ISSUE]))
        self.assertEqual(counts(result), (1, 0, 1))

    def test_matching_is_deterministic_and_order_independent_in_counts(self):
        dup = tiny()["private"]["truth"]["issues"][0]
        episode = _episode_with([dup, FX_ISSUE])
        findings = [correct_finding(), self.fx(908750), correct_finding(), self.fx(1)]
        first = run(*findings, episode=episode)
        self.assertEqual(first, run(*findings, episode=episode))
        self.assertEqual(counts(run(*reversed(findings), episode=episode)), counts(first))


if __name__ == "__main__":
    unittest.main()
