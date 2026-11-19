"""J4 mutation check: switching off any single matcher condition is caught by the attack table.

Each mutation replaces one predicate of auditgym.matcher with an always-true version (the
condition is dropped). The suite must notice: at least one attack row changes its result.
"""
import unittest
from unittest import mock

from auditgym import matcher
from tools.attack import ATTACKS, run_attack

ALWAYS = lambda *args, **kwargs: True  # noqa: E731

MUTATIONS = {
    "drop type check": "same_type",
    "drop affected-ids check": "same_affected_ids",
    "drop currency check": "same_currency",
    "drop amount check": "amount_within_tolerance",
    "drop evidence-exists check": "evidence_exists",
    "drop evidence-covers check": "evidence_covers",
    "drop one-to-one rule": "issue_available",
}


def caught_by(predicate: str) -> list[str]:
    with mock.patch.object(matcher, predicate, ALWAYS):
        return [attack[0] for attack in ATTACKS if not run_attack(attack)[0]]


class MutationTest(unittest.TestCase):
    def test_unmutated_matcher_passes_every_attack(self):
        self.assertEqual([a[0] for a in ATTACKS if not run_attack(a)[0]], [])

    def test_each_mutation_is_detected(self):
        for name, predicate in MUTATIONS.items():
            with self.subTest(name):
                self.assertTrue(caught_by(predicate), f"no attack detects: {name}")

    def test_superset_rule_mutation_is_detected(self):
        # Ignoring evidence_superset_allowed=false must be caught too.
        from tests.judge import helpers
        from auditgym.judge import judge

        ep = helpers.episode(1, superset_allowed=False)
        f = helpers.oracle(ep)[0]
        f["evidence_ids"] = f["evidence_ids"] + ["f00"]
        original = matcher.evidence_covers
        with mock.patch.object(matcher, "evidence_covers",
                               lambda finding, issue, allowed: original(finding, issue, True)):
            self.assertEqual(judge(ep, helpers.submit(ep, [f]), 1)["tp"], 1)
        self.assertEqual(judge(ep, helpers.submit(ep, [f]), 1)["tp"], 0)


if __name__ == "__main__":
    unittest.main()
