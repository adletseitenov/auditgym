"""fx_error_mode (docs/DECISIONS.md V): another date's rate, a perturbed rate, or a per-issue mix."""
import unittest
from decimal import Decimal

from auditgym import Env
from auditgym.config import ConfigError, normalize_config
from auditgym.generate import generate
from auditgym.solver import solve
from auditgym.verify import verify
from policies.runner import make_policy, run_episode
from tools import fuzz

SIZE = {"invoices": [40, 56], "background_pairs": [10, 40]}
SEEDS = range(30)
VARIANT = {"date": "wrong_date", "rate": "wrong_rate"}


def make(mode=None, count=2, **extra):
    config = {"errors": {"fx_error": count}, "size": SIZE, "currencies": ["USD", "EUR"], **extra}
    if mode is not None:
        config["fx_error_mode"] = mode
    return config


def modes_of(episode):
    return [step["params"]["mode"] for step in episode["private"]["provenance"]["steps"]]


class ConfigTest(unittest.TestCase):
    def test_default_is_mixed_and_not_part_of_the_normalized_config(self):
        self.assertEqual(normalize_config({}), normalize_config({"fx_error_mode": "mixed"}))
        self.assertNotIn("fx_error_mode", normalize_config({"fx_error_mode": "mixed"}))

    def test_other_modes_are_kept(self):
        for mode in ("date", "rate"):
            self.assertEqual(normalize_config({"fx_error_mode": mode})["fx_error_mode"], mode)

    def test_bad_values_are_config_errors(self):
        for bad in ("wrong_date", "", "Date", None, 1, True, ["date"], {"a": 1}):
            with self.assertRaises(ConfigError) as caught:
                normalize_config({"fx_error_mode": bad})
            self.assertEqual(caught.exception.code, "bad_fx_mode")

    def test_mixed_is_the_old_episode_byte_for_byte(self):
        for seed in range(8):
            for extra in ({}, {"errors": {"fx_error": 2, "allocation_gap": 1, "duplicate_invoice": 1}}):
                self.assertEqual(generate(seed, make(**extra)), generate(seed, make("mixed", **extra)))

    def test_other_error_types_do_not_move(self):
        """The mode only changes the fx_error draws: the other issues are the same in every mode."""
        config = {"errors": {"fx_error": 1, "duplicate_invoice": 1, "period_cutoff": 1}, "size": SIZE,
                  "currencies": ["USD"]}
        for seed in range(6):
            base = [i["type"] for i in generate(seed, config)["private"]["truth"]["issues"]]
            for mode in ("date", "rate"):
                got = [i["type"] for i in generate(seed, dict(config, fx_error_mode=mode))["private"]["truth"]["issues"]]
                self.assertEqual(sorted(base), sorted(got))


class ModeTest(unittest.TestCase):
    def test_each_mode_gives_only_its_variant(self):
        for mode, variant in VARIANT.items():
            for seed in SEEDS:
                episode = generate(seed, make(mode, 2))
                self.assertEqual(verify(episode), [], (mode, seed))
                self.assertEqual(modes_of(episode), [variant, variant], (mode, seed))

    def test_rate_value_matches_the_variant(self):
        for mode in VARIANT:
            for seed in range(12):
                episode = generate(seed, make(mode, 1))
                params = episode["private"]["provenance"]["steps"][0]["params"]
                if mode == "date":
                    self.assertIsNotNone(params["rate_date"])
                else:
                    self.assertIsNone(params["rate_date"])
                self.assertNotEqual(Decimal(params["used_rate"]), Decimal(params["policy_rate"]))

    def test_mixed_gives_both(self):
        seen = set()
        for seed in SEEDS:
            episode = generate(seed, make("mixed", 3))
            self.assertEqual(verify(episode), [])
            seen.update(modes_of(episode))
        self.assertEqual(seen, {"wrong_date", "wrong_rate"})

    def test_deterministic(self):
        for mode in VARIANT:
            self.assertEqual(generate(4, make(mode, 2)), generate(4, make(mode, 2)))

    def test_solver_finds_it(self):
        for mode in VARIANT:
            for seed in SEEDS:
                episode = generate(seed, make(mode, 1))
                truth = episode["private"]["truth"]["issues"][0]
                hits = [f for f in solve(episode["public"]) if f["type"] == "fx_error"]
                self.assertEqual([f["affected_ids"] for f in hits], [truth["affected_ids"]], (mode, seed))
                self.assertEqual(hits[0]["claimed_amount_minor"], truth["expected_amount_minor"])

    def test_oracle_and_solver_score_in_the_environment(self):
        for mode in VARIANT:
            config = make(mode, 3)
            for seed in (1, 2, 3):
                episode = generate(seed, config)
                with Env(mode="train") as env:
                    self.assertEqual(run_episode(env, make_policy("oracle", truth=episode["private"]["truth"]),
                                                 seed, config)["reward"], 1.0)
                    info = run_episode(env, make_policy("solver_policy"), seed, config)["info"]
                    self.assertEqual((info["precision"], info["recall"], info["f1"]), (1.0, 1.0, 1.0))

    def test_other_error_types_alongside_still_verify(self):
        for mode in VARIANT:
            config = make(mode, 2, size={"invoices": [64, 80], "background_pairs": [50, 100]})
            config["errors"] = {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 2, "period_cutoff": 1}
            for seed in range(10):
                self.assertEqual(verify(generate(seed, config)), [])

    def test_i1_verifies_in_every_mode(self):
        for seed in range(8):
            for mode in VARIANT:
                episode = generate(seed, make(mode, 0, interaction="I1"))
                self.assertEqual(verify(episode), [])
                self.assertEqual(sorted(i["type"] for i in episode["private"]["truth"]["issues"]),
                                 ["allocation_gap", "fx_error"])

    def test_fuzz_grid_covers_the_modes(self):
        modes = {config.get("fx_error_mode") for _, config in fuzz.grid()}
        self.assertTrue({"date", "rate", "mixed"} <= modes)


if __name__ == "__main__":
    unittest.main()
