"""Generator -> environment process -> solver policy -> judge, through the public contract only."""
import unittest

from auditgym import Env
from auditgym.generate import generate
from auditgym.solver import solve

CONFIG = {"errors": {"duplicate_invoice": 2, "period_cutoff": 2}, "currencies": ["USD", "EUR"]}
TABLES = ("ledger", "invoices", "payments", "fx")


def read_everything(env, observation):
    """An honest policy: page through every table, then rebuild the open records."""
    public = {"manifest": observation["manifest"], "policy": observation["policy"]}
    steps = 0
    for table in TABLES:
        records, cursor = [], None
        while True:
            observation, reward, terminated, truncated, info = env.step(
                {"action": "list_records", "table": table, "cursor": cursor, "limit": 50})
            steps += 1
            assert observation["type"] == "records" and reward == 0.0 and info == {}
            records += observation["records"]
            cursor = observation["next_cursor"]
            if cursor is None:
                break
        public[table] = records
    return public, steps


class EndToEndTest(unittest.TestCase):
    def test_solver_policy_gets_the_full_reward(self):
        for seed in (1, 2, 3):
            with self.subTest(seed=seed), Env() as env:
                observation, info = env.reset(seed, CONFIG)
                public, steps = read_everything(env, observation)
                self.assertEqual(public, {k: v for k, v in generate(seed, CONFIG)["public"].items()})
                answer = {"episode_id": public["manifest"]["episode_id"], "findings": solve(public)}
                observation, reward, terminated, truncated, info = env.step({"action": "submit", "findings": answer})
                self.assertTrue(terminated and not truncated)
                self.assertEqual(info["f1"], 1.0)
                self.assertEqual(info["inspection_steps"], steps)
                self.assertAlmostEqual(reward, 1 - 0.002 * steps)

    def test_silence_is_punished_on_books_with_errors_and_rewarded_on_clean_ones(self):
        for config, expected in ((CONFIG, -1.0), ({"errors": {}}, 1.0)):
            with self.subTest(config=config), Env() as env:
                observation, _ = env.reset(5, config)
                answer = {"episode_id": observation["manifest"]["episode_id"], "findings": []}
                _, reward, terminated, _, _ = env.step({"action": "submit", "findings": answer})
                self.assertTrue(terminated)
                self.assertEqual(reward, expected)

    def test_submit_without_findings_is_an_invalid_structure_not_silence(self):
        for findings in (None, "x", [], 7):
            with self.subTest(findings=findings), Env() as env:
                env.reset(5, {"errors": {}})  # clean books: silence would score 1.0
                _, reward, terminated, _, _ = env.step({"action": "submit", "findings": findings})
                self.assertTrue(terminated)
                self.assertEqual(reward, -1.0)
        with Env() as env:
            env.reset(5, {"errors": {}})
            _, reward, _, _, _ = env.step({"action": "submit"})
            self.assertEqual(reward, -1.0)


if __name__ == "__main__":
    unittest.main()
