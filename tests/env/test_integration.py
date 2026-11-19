"""Generator -> environment process -> solver policy -> judge, through the public contract only."""
import unittest

from auditgym import Env
from auditgym.evaluator import order_key
from auditgym.generate import generate
from auditgym.solver import solve
from policies.runner import make_policy, run_episode

CONFIG = {"errors": {"duplicate_invoice": 2, "period_cutoff": 2}, "currencies": ["USD", "EUR"]}
TABLES = ("ledger", "invoices", "payments", "fx")
TYPES = {"duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff"}
# The default config (all four types, one each, distractors on) and the same with each interaction.
# An interaction adds two issues to the four, so the invoice floor rises from 40 to 48 to keep one
# error per 8 invoices (CONTRACTS §4); everything else stays default.
WIDER = {"invoices": [48, 100], "background_pairs": [150, 300]}
DEFAULT_CONFIGS = {"default": None, "default+I1": {"interaction": "I1", "size": WIDER},
                   "default+I2": {"interaction": "I2", "size": WIDER}}


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
                expected = generate(seed, CONFIG)["public"]
                # Pages follow evaluator.order_key, not the file order (E8): same records, other order.
                for table in TABLES:
                    expected[table] = sorted(expected[table], key=lambda record, table=table: order_key(table, record))
                self.assertEqual(public, expected)
                answer = {"episode_id": public["manifest"]["episode_id"], "findings": solve(public)}
                observation, reward, terminated, truncated, info = env.step({"action": "submit", "findings": answer})
                self.assertTrue(terminated and not truncated)
                self.assertEqual(info["f1"], 1.0)
                self.assertEqual(info["inspection_steps"], steps)
                self.assertAlmostEqual(reward, 1 - 0.002 * steps)

    def test_honest_solver_policy_gets_f1_one_on_default_configs_with_interactions(self):
        """The real policy (policies/solver_policy.py) through the real environment process, on the
        default config with distractors and with I1 and I2, over several seeds."""
        for label, config in DEFAULT_CONFIGS.items():
            for seed in (11, 12, 13, 14):
                truth = generate(seed, config)["private"]["truth"]["issues"]
                kinds = {issue["type"] for issue in truth}
                self.assertEqual(kinds, TYPES, f"{label} seed {seed}")
                if config is not None:
                    self.assertEqual(len(truth), 6, f"{label} seed {seed}: four errors plus the interaction pair")
                with self.subTest(config=label, seed=seed), Env() as env:
                    result = run_episode(env, make_policy("solver_policy"), seed, config)
                    self.assertTrue(result["terminated"] and not result["truncated"])
                    self.assertEqual(result["info"]["f1"], 1.0, result["info"])
                    self.assertEqual(result["info"]["inspection_steps"], result["steps"] - 1)
                    self.assertAlmostEqual(result["reward"], 1 - 0.002 * (result["steps"] - 1))

    def test_silence_is_punished_on_books_with_errors_and_rewarded_on_clean_ones(self):
        for config, expected in ((CONFIG, -1.0), ({"errors": {}}, 1.0)):
            with self.subTest(config=config), Env() as env:
                observation, _ = env.reset(5, config)
                answer = {"episode_id": observation["manifest"]["episode_id"], "findings": []}
                _, reward, terminated, _, _ = env.step({"action": "submit", "findings": answer})
                self.assertTrue(terminated)
                self.assertEqual(reward, expected)

    def test_submit_without_findings_is_an_invalid_structure_not_silence(self):
        # A bare list is the findings list itself (AUDIT C1): [] is an explicit empty answer.
        for findings in (None, "x", 7, "[", "null", {}):
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
