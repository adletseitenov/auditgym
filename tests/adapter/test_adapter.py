"""judge_adapter itself: generate_episode, reset modes, score_episode, the budget edge cases, the baselines."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import judge_adapter
from tests.adapter import helpers
from tests.adapter.helpers import closed, fixture, list_action, new_env, submit_action
from tools import adapter_pairs
from tools.adapter_run import BASELINES, run_baselines

ROOT = pathlib.Path(__file__).resolve().parents[2]


def tearDownModule():
    helpers.cleanup()


class GenerateTest(unittest.TestCase):
    def test_same_arguments_give_the_same_files_and_other_seed_another_episode(self):
        with tempfile.TemporaryDirectory() as scratch:
            for name, seed in (("a", 5), ("b", 5), ("c", 6)):
                judge_adapter.generate_episode(seed, {"errors": {"period_cutoff": 1}}, f"{scratch}/{name}")
            digest = adapter_pairs.tree_digest
            self.assertEqual(digest(f"{scratch}/a"), digest(f"{scratch}/b"))
            self.assertNotEqual(digest(f"{scratch}/a"), digest(f"{scratch}/c"))
            files = sorted(p.relative_to(scratch + "/a").as_posix() for p in pathlib.Path(scratch, "a").rglob("*")
                           if p.is_file())
            for needed in ("public/manifest.json", "public/ledger.csv", "public/invoices.jsonl",
                           "public/payments.jsonl", "public/fx.csv", "public/policy.json",
                           "private/truth.json", "private/provenance.json"):
                self.assertIn(needed, files)

    def test_bad_config_is_refused_with_the_generators_error(self):
        from auditgym.config import ConfigError

        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaises(ConfigError):
                judge_adapter.generate_episode(1, {"errors": {"no_such_type": 1}}, scratch + "/x")


class ResetTest(unittest.TestCase):
    def test_closed_episode_dir_loads_exactly_that_episode_and_restarts(self):
        fx = fixture("E")
        env, journal = new_env("reset")
        self.addCleanup(env.close)
        first, info = env.reset(123, closed(fx["dir"]))  # the seed and any team_config do not matter
        self.assertEqual(first["manifest"]["episode_id"], fx["episode"]["public"]["manifest"]["episode_id"])
        a = env.step(list_action("invoices", None, 7))
        env.step({"name": "get_record", "args": {"table": "ledger", "id": "nope"}})
        again, _ = env.reset(999, dict(closed(fx["dir"]), team_config={"errors": {"fx_error": 1}}))
        self.assertEqual(again, first)  # restarted, not regenerated; the budget is full again
        self.assertEqual(env.step(list_action("invoices", None, 7)), a)
        # ids were not changed: the records are the ones of the files
        public = fx["episode"]["public"]
        self.assertEqual(a[0]["result"]["records"][0], public["invoices"][0])
        # the abandoned run is journaled, not hidden
        self.assertIn("abandoned", {row["event"] for row in helpers.journal(journal)})

    def test_train_generates_with_seed_and_team_config_and_is_reproducible(self):
        env = judge_adapter.make_env()
        self.addCleanup(env.close)
        config = {"errors": {"period_cutoff": 1}}
        a, _ = env.reset(11, {"mode": "train", "team_config": config})
        b, _ = env.reset(11, {"mode": "train", "team_config": config})
        c, _ = env.reset(12, {"mode": "train", "team_config": config})
        self.assertEqual(a, b)
        self.assertNotEqual(a["manifest"]["episode_id"], c["manifest"]["episode_id"])
        with tempfile.TemporaryDirectory() as scratch:
            judge_adapter.generate_episode(11, config, scratch + "/e")
            self.assertEqual(json.loads((pathlib.Path(scratch) / "e/public/manifest.json").read_text())["episode_id"],
                             a["manifest"]["episode_id"])  # the same generator as generate_episode

    def test_train_with_episode_dir_loads_it_and_gives_the_terminal_reward(self):
        fx = fixture("E")
        env = judge_adapter.make_env()
        self.addCleanup(env.close)
        first, _ = env.reset(0, {"mode": "train", "episode_dir": fx["dir"]})
        self.assertEqual(first["manifest"]["episode_id"], fx["episode"]["public"]["manifest"]["episode_id"])
        observation, reward, terminated, truncated, info = env.step(submit_action(helpers.empty_findings(fx)))
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)  # one issue missed, empty answer
        self.assertEqual(info["reward"], -1.0)

    def test_closed_mode_does_not_generate_when_the_dir_is_missing(self):
        env, _ = new_env("missing")
        self.addCleanup(env.close)
        with self.assertRaises(judge_adapter.AdapterError) as caught:
            env.reset(0, closed("/nonexistent/episode"))
        self.assertEqual(caught.exception.code, "bad_episode")
        self.assertNotIn("/nonexistent", str(caught.exception))
        with self.assertRaises(judge_adapter.AdapterError):
            env.reset(0, {"mode": "dev"})
        with self.assertRaises(judge_adapter.AdapterError):
            env.reset(0, ["not", "a", "dict"])

    def test_step_before_reset_is_a_bounded_error(self):
        env, _ = new_env("noreset")
        self.addCleanup(env.close)
        observation, reward, terminated, truncated, info = env.step(list_action())
        self.assertEqual(observation["error"]["code"], "not_reset")
        self.assertEqual((reward, terminated, truncated, info), (0.0, False, False, {}))

    def test_a_secret_in_the_environment_never_reaches_the_evaluator(self):
        import os
        fx = fixture("E")
        env, _ = new_env("secret")
        self.addCleanup(env.close)
        os.environ["AUDITGYM_SECRET"] = "not-a-real-secret-for-the-test"
        try:
            first, _ = env.reset(0, closed(fx["dir"]))  # eval Env would refuse it; the adapter hands none on
        finally:
            del os.environ["AUDITGYM_SECRET"]
        self.assertEqual(first["remaining_steps"], 100)


class SubmitTest(unittest.TestCase):
    def setUp(self):
        self.fx = fixture("E")
        self.env, self.journal = new_env("submit")
        self.addCleanup(self.env.close)

    def submit_bad(self, mutate):
        findings = helpers.oracle_findings(self.fx)
        mutate(findings)
        self.env.reset(0, closed(self.fx["dir"]))
        observation, reward, terminated, truncated, info = self.env.step(submit_action(findings))
        self.assertTrue(terminated)
        self.assertEqual((reward, info), (0, {}))
        self.assertIsNone(observation["result"])
        self.assertEqual(observation["error"]["code"], "invalid_findings")
        self.assertEqual(helpers.finished(self.journal)[-1]["reward"], -1.0)
        self.assertNotIn(helpers.oracle_findings(self.fx)["findings"][0]["evidence_ids"][0], json.dumps(observation))

    def test_block3_18_to_20_limits_are_structure_errors(self):  # published unit tests 18, 19, 20
        self.submit_bad(lambda f: f["findings"].extend(dict(f["findings"][0]) for _ in range(30)))
        self.submit_bad(lambda f: f["findings"][0].update(evidence_ids=[f"x{n}" for n in range(11)]))
        self.submit_bad(lambda f: f["findings"][0].update(explanation="a" * 1001))

    def test_missing_or_wrong_typed_fields_are_structure_errors(self):
        self.submit_bad(lambda f: f["findings"][0].update(claimed_amount_minor="100"))
        self.submit_bad(lambda f: f["findings"][0].update(claimed_amount_minor=True))
        self.submit_bad(lambda f: f["findings"][0].pop("currency"))
        self.env.reset(0, closed(self.fx["dir"]))
        observation, _, terminated, _, _ = self.env.step({"name": "submit", "args": {}})  # no findings at all
        self.assertTrue(terminated)
        self.assertEqual(observation["error"]["code"], "invalid_findings")

    def test_a_wrong_but_well_formed_answer_is_accepted_and_says_nothing_about_correctness(self):
        findings = helpers.oracle_findings(self.fx)
        findings["findings"][0]["type"] = "fx_error"
        findings["findings"][0]["claimed_amount_minor"] += 1
        self.env.reset(0, closed(self.fx["dir"]))
        observation, reward, terminated, truncated, info = self.env.step(submit_action(findings))
        right = helpers.oracle_findings(self.fx)
        self.env.reset(0, closed(self.fx["dir"]))
        other, *_ = self.env.step(submit_action(right))
        self.assertEqual(observation, other)  # the same reply for a wrong and a right answer
        self.assertEqual(observation["result"], {"accepted": True})
        self.assertEqual(helpers.finished(self.journal)[-2]["tp"], 0)
        self.assertEqual(helpers.finished(self.journal)[-1]["tp"], 1)


class ScoreEpisodeTest(unittest.TestCase):
    def setUp(self):
        self.fx = fixture("E")
        self.log = helpers.scratch() / "score-test.log"

    def score(self, findings, steps=10):
        return judge_adapter.score_episode(self.fx["dir"], findings, steps, journal_path=self.log)

    def test_published_reward_examples_through_score_episode(self):  # unit tests 1, 2, 5, 15, 16, 17
        right = helpers.oracle_findings(self.fx)
        empty = helpers.empty_findings(self.fx)
        one = self.score(right)
        self.assertEqual({k: one[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                         {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertAlmostEqual(one["return"], 0.98, places=9)
        self.assertEqual(set(one), {"tp", "fp", "fn", "precision", "recall", "f1", "return"})
        missed = self.score(empty)
        self.assertEqual((missed["tp"], missed["fp"], missed["fn"], missed["f1"], missed["return"]), (0, 0, 1, 0.0, -1))
        twice = self.score(dict(right, findings=right["findings"] * 2))
        self.assertEqual((twice["tp"], twice["fp"], twice["fn"]), (1, 1, 0))
        self.assertAlmostEqual(twice["precision"], 0.5, places=9)
        self.assertAlmostEqual(twice["f1"], 2 / 3, places=9)
        self.assertAlmostEqual(twice["return"], 47 / 150, places=9)
        for steps, expected in ((0, 1.0), (50, 0.90), (100, 0.80)):
            self.assertAlmostEqual(self.score(right, steps)["return"], expected, places=9)

    def test_invalid_structure_is_minus_one_with_null_metrics_and_a_log_line(self):
        findings = helpers.oracle_findings(self.fx)
        findings["findings"][0]["explanation"] = "a" * 1001
        result = self.score(findings)
        self.assertEqual(result, {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None,
                                  "f1": None, "return": -1})
        lines = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(lines[-1]["event"], "score_error")
        self.assertEqual(lines[-1]["episode_id"], self.fx["episode"]["public"]["manifest"]["episode_id"])
        self.assertNotIn("a" * 10, json.dumps(lines[-1]))  # a code, not the submitted data
        self.assertEqual(self.score({"findings": []})["return"], -1)  # no episode_id: invalid structure

    def test_truncated_episode_is_scored_as_empty(self):
        scored = self.score(None, 100)
        self.assertEqual((scored["tp"], scored["fp"], scored["fn"], scored["return"]), (0, 0, 1, -1))


class BaselineTest(unittest.TestCase):
    def test_four_baselines_act_only_through_the_adapter(self):
        fx = fixture("E")
        with tempfile.TemporaryDirectory() as scratch:
            rows = run_baselines(fx["dir"], pathlib.Path(scratch) / "results.jsonl")
        self.assertEqual([row["policy"] for row in rows], list(BASELINES))
        for row in rows:
            self.assertEqual(row["policy_saw_reward"], 0)
            self.assertTrue(row["terminated"])
            self.assertEqual(row["journal_reward"], row["return"], row["policy"])
        oracle = rows[-1]
        self.assertEqual((oracle["tp"], oracle["fp"], oracle["fn"], oracle["return"]), (1, 0, 0, 1.0))
        self.assertLess(max(r["return"] for r in rows[:3]), oracle["return"])

    def test_random_explorer_with_a_fixed_seed_repeats_exactly(self):
        fx = fixture("E")
        with tempfile.TemporaryDirectory() as scratch:
            a = run_baselines(fx["dir"], pathlib.Path(scratch) / "a.jsonl", ("random_explorer",))
            b = run_baselines(fx["dir"], pathlib.Path(scratch) / "b.jsonl", ("random_explorer",))
        self.assertEqual(a, b)

    def test_demo_command_runs(self):
        done = subprocess.run([sys.executable, "judge_adapter.py", "--demo"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("demo: ok", done.stdout)


class ImportTest(unittest.TestCase):
    def test_entry_points_and_the_container_files(self):
        for name in ("generate_episode", "make_env", "score_episode"):
            self.assertTrue(callable(getattr(judge_adapter, name)))
        ignored = {line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()}
        for path in ("judge_adapter.py", "tools", "policies", "auditgym"):
            self.assertNotIn(path, ignored)
            self.assertNotIn(path + "/", ignored)
        self.assertTrue((ROOT / "judge_adapter.py").is_file())
        self.assertIn("COPY . /app", (ROOT / "Dockerfile").read_text())


if __name__ == "__main__":
    unittest.main()
