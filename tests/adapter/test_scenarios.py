"""Block 2 smoke: generate_episode for every D01-D20 scenario type, checked by the data and the oracle.

This does not replace the jury's independent reconciliation (it cannot: the check that the injection is the
declared discrepancy is `auditgym.verify`, which is ours); it shows that every scenario type can be produced
by the documented config through generate_episode, loads, is consistent before and after the injection by
verify, is deterministic, and that the oracle through the adapter earns the maximum.
"""
from __future__ import annotations

import tempfile
import unittest

import judge_adapter
from auditgym.storage import load_episode
from auditgym.verify import verify
from policies.oracle import Oracle
from tools import adapter_pairs
from tools.adapter_run import PolicyBridge


class ScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.env = judge_adapter.make_env(journal_path=f"{cls.scratch.name}/results.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.scratch.cleanup()


def _make(index: int):
    label, what, _, _ = adapter_pairs.MATRIX[index]

    def test(self):
        seed, config = adapter_pairs.discover(index)
        directory = f"{self.scratch.name}/{label}"
        judge_adapter.generate_episode(seed, config, directory)
        again = f"{self.scratch.name}/{label}-again"
        judge_adapter.generate_episode(seed, config, again)
        self.assertEqual(adapter_pairs.tree_digest(directory), adapter_pairs.tree_digest(again))  # deterministic
        episode = load_episode(directory)
        self.assertEqual(verify(episode), [], what)  # books consistent before the injection, the injection is the declared one
        truth = episode["private"]["truth"]["issues"]
        self.assertEqual(bool(truth), index >= 5, what)  # D01-D05 are clean, the rest have an error
        for issue in truth:
            self.assertTrue(issue["affected_ids"] and issue["evidence_sets"])
        findings = adapter_pairs.oracle_findings(directory)
        scored = judge_adapter.score_episode(directory, findings, 0, journal_path=f"{self.scratch.name}/score.log")
        self.assertEqual((scored["tp"], scored["fp"], scored["fn"], scored["f1"], scored["return"]),
                         (len(truth), 0, 0, 1.0, 1.0), what)
        # and through the adapter, closed mode: the oracle submits by step, the evaluator's journal agrees
        bridge = PolicyBridge(Oracle(episode["private"]["truth"]))
        seen = bridge.run(self.env, 0, {"mode": "closed", "episode_dir": directory})
        self.assertEqual((seen["reward"], seen["info"], seen["terminated"]), (0, {}, True))
        self.assertEqual(self.env.last_result()["reward"], 1.0)
    return test


for _index, _row in enumerate(adapter_pairs.MATRIX):
    setattr(ScenarioTest, f"test_{_row[0]}", _make(_index))


class ScenarioTypesTest(unittest.TestCase):
    def test_all_documented_error_types_and_both_interactions_are_covered(self):
        from auditgym.generate import generate

        seen_types = set()
        for index in (5, 8, 11, 13, 15, 18):
            seed, config = adapter_pairs.discover(index)
            seen_types |= {i["type"] for i in generate(seed, config)["private"]["truth"]["issues"]}
        self.assertEqual(seen_types, {"duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff"})
        modes = set()
        for index in (11, 12):  # fx: both the wrong date and the wrong rate variant exist
            seed, config = adapter_pairs.discover(index)
            modes |= {s["params"]["mode"] for s in generate(seed, config)["private"]["provenance"]["steps"]}
        self.assertEqual(modes, {"wrong_date", "wrong_rate"})


if __name__ == "__main__":
    unittest.main()
