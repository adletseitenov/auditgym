"""Difficulty levels (configs/levels.json): valid, deterministic, the oracle and the solver reach 1.0."""
from __future__ import annotations

import unittest

from auditgym import Env, levels
from auditgym.config import normalize_config
from auditgym.generate import generate
from auditgym.limits import MAX_PAGES, MAX_STEPS, pages
from policies.runner import make_policy, run_episode

SEEDS = [9000 + k for k in range(20)]


class LevelsTest(unittest.TestCase):
    def test_names_and_load(self) -> None:
        self.assertEqual(levels.names(), ["easy", "medium", "hard", "max"])
        config = levels.load("hard")
        config["errors"]["fx_error"] = 99  # a copy: the file's levels are not changed by callers
        self.assertNotEqual(levels.load("hard")["errors"]["fx_error"], 99)
        with self.assertRaises(KeyError):
            levels.load("nope")

    def test_valid_and_deterministic(self) -> None:
        for name in levels.names():
            with self.subTest(level=name):
                normalize_config(levels.load(name))
                first, second = generate(SEEDS[0], levels.load(name)), generate(SEEDS[0], levels.load(name))
                self.assertEqual(first["public"], second["public"])
                self.assertEqual(first["private"]["truth"], second["private"]["truth"])

    def test_full_read_fits_the_budget(self) -> None:
        for name in levels.names():
            public = generate(SEEDS[1], levels.load(name))["public"]
            total = sum(pages(len(public[table])) for table in ("ledger", "invoices", "payments", "fx"))
            self.assertLessEqual(total, MAX_PAGES)
            self.assertLess(total + 1, MAX_STEPS)

    def test_levels_get_harder(self) -> None:
        sizes = [sum(pages(len(generate(SEEDS[2], levels.load(n))["public"][t])) for t in ("ledger", "invoices", "payments", "fx"))
                 for n in levels.names()]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_oracle_and_solver(self) -> None:
        for name in levels.names():
            config = levels.load(name)
            for policy, floor in (("oracle", 0.99), ("solver_policy", 0.85)):
                with self.subTest(level=name, policy=policy), Env(mode="train") as env:
                    for seed in SEEDS if name in ("easy", "medium") else SEEDS[:5]:  # keeps the fast check fast
                        truth = generate(seed, config)["private"]["truth"]
                        result = run_episode(env, make_policy(policy, truth=truth if policy == "oracle" else None), seed, config)
                        self.assertFalse(result["truncated"])
                        self.assertTrue(truth["issues"])
                        self.assertEqual(result["info"]["f1"], 1.0)
                        self.assertGreaterEqual(result["reward"], floor)


if __name__ == "__main__":
    unittest.main()
