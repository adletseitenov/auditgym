"""Baseline policies through a real Env process: completion, modes, limits and the reward order."""
from __future__ import annotations

import ast
import pathlib
import statistics
import tempfile
import unittest

from auditgym import Env
from auditgym.generate import generate
from auditgym.storage import write_episode
from policies.runner import MAX_STEPS, make_policy, read_results, run_episode
from tools import episodes as episodes_tool
from tools.report import generated_cases, ordering_problems, run_generated

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ALL = ("random_explorer", "reconcile", "flag_all", "flag_all_cutoff", "flag_all_duplicates",
       "silence", "solver_policy", "oracle")
ORDERED = ("oracle", "solver_policy", "reconcile", "silence", "flag_all", "flag_all_cutoff", "flag_all_duplicates")


def _policy(name: str, episode: dict):
    return make_policy(name, truth=episode["private"]["truth"] if name == "oracle" else None)


def _case() -> tuple[int, object]:
    return 3, episodes_tool.supported_pool(episodes_tool.ERROR_POOL, quiet=True)[0]


class CompletionTest(unittest.TestCase):
    def test_every_policy_completes_in_train_mode(self):
        seed, config = _case()
        episode = generate(seed, config)
        with Env(mode="train") as env:
            for name in ALL:
                with self.subTest(policy=name):
                    result = run_episode(env, _policy(name, episode), seed, config)
                    self.assertTrue(result["terminated"] and not result["truncated"])
                    self.assertLessEqual(result["steps"], MAX_STEPS)
                    self.assertEqual(set(result["info"]), {"reward", "precision", "recall", "f1", "inspection_steps"})
                    self.assertEqual(result["reward"], result["info"]["reward"])
                    self.assertIsInstance(result["findings"], list)

    def test_every_policy_completes_in_eval_mode_and_the_log_holds_the_reward(self):
        seed, config = _case()
        episode = generate(seed, config)
        with tempfile.TemporaryDirectory() as scratch:
            path = pathlib.Path(scratch) / "results.jsonl"
            with Env(mode="eval", results_path=path) as env:
                for name in ALL:
                    result = run_episode(env, _policy(name, episode), seed, config)
                    self.assertEqual((result["reward"], result["info"]), (0.0, {}), name)
                    self.assertTrue(result["terminated"], name)
            logged = read_results(path)
        self.assertEqual(len(logged), len(ALL))
        with Env(mode="train") as env:
            for name, line in zip(ALL, logged):
                self.assertEqual(run_episode(env, _policy(name, episode), seed, config)["reward"], line["reward"], name)

    def test_policies_run_on_an_episode_loaded_from_files(self):
        seed, config = _case()
        episode = generate(seed, config)
        with tempfile.TemporaryDirectory() as scratch:
            write_episode(episode, scratch)
            for name in ("solver_policy", "oracle", "reconcile"):
                env = Env.from_files(scratch)
                try:
                    result = run_episode(env, _policy(name, episode))
                finally:
                    env.close()
                self.assertTrue(result["terminated"], name)
                self.assertEqual(result["episode_id"], episode["public"]["manifest"]["episode_id"])

    def test_a_policy_that_never_submits_is_truncated_at_the_step_limit(self):
        class NeverSubmits:
            def reset(self, observation):
                pass

            def act(self, observation):
                return {"action": "calculator", "expression": "1 + 1"}

        seed, config = _case()
        with Env() as env:
            result = run_episode(env, NeverSubmits(), seed, config)
        self.assertTrue(result["truncated"] and not result["terminated"])
        self.assertEqual(result["steps"], MAX_STEPS)
        self.assertEqual(result["reward"], -1.0)

    def test_random_explorer_is_reproducible(self):
        seed, config = _case()
        runs = []
        for _ in range(2):
            with Env() as env:
                runs.append(run_episode(env, make_policy("random_explorer", seed=11, guess=True), seed, config,
                                        record=True))
        self.assertEqual([row["action"] for row in runs[0]["trajectory"]],
                         [row["action"] for row in runs[1]["trajectory"]])
        self.assertTrue(runs[0]["findings"])

    def test_trajectory_rows_have_the_hand_in_fields(self):
        seed, config = _case()
        with Env() as env:
            result = run_episode(env, make_policy("solver_policy"), seed, config, record=True)
        self.assertEqual(len(result["trajectory"]), result["steps"])
        for row in result["trajectory"]:
            self.assertEqual(set(row), {"episode_id", "step", "action", "observation", "reward",
                                        "terminated", "truncated"})


class OrderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        errors = [case for case in generated_cases(40) if case[1] in
                  episodes_tool.supported_pool(episodes_tool.ERROR_POOL, quiet=True)][:30]
        cls.error_cases = errors
        cls.errors = run_generated(ORDERED, errors)
        clean = [(70_000 + k, config) for k, config in
                 enumerate(episodes_tool.supported_pool(episodes_tool.CLEAN_POOL, quiet=True) * 2)]
        cls.clean = run_generated(("silence", "solver_policy", "flag_all", "flag_all_cutoff", "reconcile"), clean)
        cls.clean_cases = clean

    def mean(self, rows, name):
        return statistics.fmean(row["reward"] for row in rows[name])

    def test_at_least_thirty_episodes_with_errors(self):
        self.assertGreaterEqual(len(self.error_cases), 30)
        self.assertTrue(all(not row["clean"] for row in self.errors["oracle"]))

    def test_order_of_mean_rewards_on_books_with_errors(self):
        self.assertEqual(ordering_problems(self.errors), [])
        m = {name: self.mean(self.errors, name) for name in ORDERED}
        self.assertGreaterEqual(m["oracle"], m["solver_policy"])
        self.assertGreater(m["solver_policy"], m["reconcile"])
        for name in ("flag_all", "flag_all_cutoff", "flag_all_duplicates"):
            self.assertGreater(m["reconcile"], m[name], name)
            self.assertLessEqual(m[name], m["silence"] + 0.15, name)

    def test_clean_books_reward_silence_and_the_solver_and_punish_accusations(self):
        m = {name: self.mean(self.clean, name) for name in self.clean}
        self.assertEqual(m["silence"], 1.0)
        self.assertGreater(m["solver_policy"], 0.9)
        self.assertLess(m["flag_all"], -0.9)
        self.assertLess(m["flag_all_cutoff"], -0.9)

    def test_random_guesses_score_low_on_clean_and_error_books(self):
        cases = self.clean_cases + self.error_cases[:8]
        with Env() as env:
            rewards = [run_episode(env, make_policy("random_explorer", seed=5, guess=True), seed, config)["reward"]
                       for seed, config in cases]
        self.assertLess(statistics.fmean(rewards), -0.5)

    def test_every_policy_stays_within_the_step_limit(self):
        for rows in (self.errors, self.clean):
            for name, policy_rows in rows.items():
                self.assertTrue(all(row["steps"] <= MAX_STEPS and not row["truncated"] for row in policy_rows), name)


class IsolationTest(unittest.TestCase):
    FORBIDDEN = ("auditgym.generate", "auditgym.inject", "auditgym.verify", "auditgym.storage",
                 "auditgym.evaluator", "auditgym.judge", "auditgym.matcher", "random", "uuid", "secrets")

    def test_policies_import_nothing_from_the_evaluator_side(self):
        for path in sorted((ROOT / "policies").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    self.assertFalse(any(name == bad or name.startswith(bad + ".") for bad in self.FORBIDDEN),
                                     f"{path.name} imports {name}")
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('["private"]', text, path.name)
            self.assertNotIn("private/", text, path.name)


if __name__ == "__main__":
    unittest.main()
