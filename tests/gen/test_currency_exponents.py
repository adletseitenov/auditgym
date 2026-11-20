"""JPY (exponent 0) and KWD (exponent 3): conversion, generation, the independent verifier, the
solver and the oracle (docs/DECISIONS.md M). The default config and every shipped episode keep
KZT/USD/EUR/RUB/CNY, all with exponent 2, and are byte-identical to before."""
from __future__ import annotations

import unittest
from decimal import Decimal
from fractions import Fraction

from auditgym import Env
from auditgym.config import CURRENCIES, CURRENCY_EXPONENTS, ConfigError, normalize_config
from auditgym.generate import generate
from auditgym.money import convert
from auditgym.reward import STEP_COST
from auditgym.solver import solve
from auditgym.verify import verify
from policies.runner import make_policy, run_episode

NEW = ("JPY", "KWD")
BASES = ("KZT", "USD", "EUR", "RUB", "CNY", "JPY", "KWD")
FUZZ_SEEDS = 50


def episode_config(currencies, base="KZT", **extra) -> dict:
    return {"errors": {"fx_error": 2, "duplicate_invoice": 1}, "currencies": list(currencies), "base_currency": base,
            "size": {"invoices": [30, 34], "background_pairs": [0, 5]}, **extra}


class ConvertTest(unittest.TestCase):
    def test_exponents_table(self) -> None:
        self.assertEqual(CURRENCY_EXPONENTS["JPY"], 0)
        self.assertEqual(CURRENCY_EXPONENTS["KWD"], 3)
        self.assertEqual(set(CURRENCY_EXPONENTS), set(CURRENCIES))
        self.assertEqual(CURRENCIES[:5], ("KZT", "USD", "EUR", "RUB", "CNY"))  # the default depends on this order

    def test_jpy_to_two_digit_base(self) -> None:
        # 12 345 yen at 3.3 KZT per yen: 12345 * 10^(2-0) * 3.3 = 4 073 850 tiyn (40 738.50 KZT).
        self.assertEqual(convert(12345, Decimal("3.3"), 0, 2), 4073850)
        # half-even at the tie: 1 yen * 100 * 3.005 = 300.5 -> 300 ; 3.015 -> 301.5 -> 302
        self.assertEqual(convert(1, Decimal("3.005"), 0, 2), 300)
        self.assertEqual(convert(1, Decimal("3.015"), 0, 2), 302)

    def test_kwd_to_two_digit_base(self) -> None:
        # 2 500 fils (2.500 KWD) at 1600 KZT per dinar: 2500 * 10^(2-3) * 1600 = 400 000 tiyn (4 000.00 KZT).
        self.assertEqual(convert(2500, Decimal("1600"), 3, 2), 400000)
        # 1234 fils at 1612.5: 1234 / 10 * 1612.5 = 198 982.5 -> half-even 198 982
        self.assertEqual(convert(1234, Decimal("1612.5"), 3, 2), 198982)
        self.assertEqual(convert(1236, Decimal("1612.5"), 3, 2), 199305)  # 199 305.0 exactly
        # 1 fil at 1605 = 160.5 tiyn -> half-even 160
        self.assertEqual(convert(1, Decimal("1605"), 3, 2), 160)

    def test_into_new_bases(self) -> None:
        # USD cents -> JPY: 1000 cents (10.00 USD) at 143.2 JPY per dollar: 1000 / 100 * 143.2 = 1432 yen.
        self.assertEqual(convert(1000, Decimal("143.2"), 2, 0), 1432)
        # USD cents -> KWD fils: 1000 cents at 0.307 KWD per dollar: 1000 * 10 * 0.307 = 3070 fils.
        self.assertEqual(convert(1000, Decimal("0.307"), 2, 3), 3070)


class GenerateTest(unittest.TestCase):
    def check(self, seed: int, config: dict) -> dict:
        episode = generate(seed, config)
        self.assertEqual(verify(episode), [], (seed, config))
        return episode

    def test_new_currencies_verify(self) -> None:
        for base in ("KZT", "USD"):
            for currencies in (["JPY"], ["KWD"], ["JPY", "KWD"], ["JPY", "KWD", "EUR"]):
                for seed in range(3):
                    episode = self.check(300 + seed, episode_config(currencies, base))
                    exponents = episode["public"]["policy"]["currency_exponents"]
                    self.assertEqual({c: exponents[c] for c in currencies},
                                     {c: CURRENCY_EXPONENTS[c] for c in currencies})
                    self.assertEqual({r["source_currency"] for r in episode["public"]["fx"]}, set(currencies))

    def test_rates_are_plausible(self) -> None:
        public = generate(5, episode_config(["JPY", "KWD"]))["public"]
        for row in public["fx"]:
            rate = Decimal(row["rate"])
            low, high = (2.8, 4.2) if row["source_currency"] == "JPY" else (1300, 1900)  # level +-10 %, then a daily walk
            self.assertTrue(Decimal(low) < rate < Decimal(high), row)

    def test_new_bases(self) -> None:
        for base in NEW:
            for seed in range(3):
                self.check(400 + seed, episode_config(["USD", "EUR"], base))
                self.check(410 + seed, episode_config(["KZT", "JPY" if base == "KWD" else "KWD"], base))

    def test_config_rules(self) -> None:
        self.assertEqual(normalize_config({"currencies": ["JPY", "KWD"]})["currencies"], ["JPY", "KWD"])
        with self.assertRaises(ConfigError):
            normalize_config({"base_currency": "JPY", "currencies": ["JPY"]})
        with self.assertRaises(ConfigError):
            normalize_config({"currencies": ["GBP"]})


class SolveTest(unittest.TestCase):
    def test_fx_error_across_exponents(self) -> None:
        for currencies in (["JPY"], ["KWD"], ["JPY", "KWD"]):
            for base in ("KZT", "JPY", "KWD"):
                currencies_ = [c for c in currencies if c != base] or ["USD"]
                for seed in range(3):
                    config = {"errors": {"fx_error": 3}, "currencies": currencies_, "base_currency": base,
                              "size": {"invoices": [30, 40], "background_pairs": [0, 5]}}
                    episode = generate(700 + seed, config)
                    issues = episode["private"]["truth"]["issues"]
                    self.assertEqual({i["type"] for i in issues}, {"fx_error"})
                    self.assertEqual(len(issues), 3)
                    with Env(mode="train") as env:
                        result = run_episode(env, make_policy("solver_policy"), 700 + seed, config)
                    self.assertEqual(result["info"]["f1"], 1.0, (currencies_, base, seed))

    def test_fx_error_between_different_exponents_is_injected(self) -> None:
        config = {"errors": {"fx_error": 3}, "currencies": ["JPY", "KWD"], "size": {"invoices": [40, 40], "background_pairs": [0, 5]}}
        seen = set()
        for seed in range(6):
            episode = generate(800 + seed, config)
            by_id = {d["doc_id"]: d["currency"] for d in episode["public"]["invoices"] + episode["public"]["payments"]}
            for issue in episode["private"]["truth"]["issues"]:
                seen |= {by_id[i] for i in issue["affected_ids"] if i in by_id}
        self.assertEqual(seen, {"JPY", "KWD"})

    def test_solver_function_matches_truth(self) -> None:
        episode = generate(11, episode_config(["JPY", "KWD"]))
        found = {frozenset(f["affected_ids"]) for f in solve(episode["public"])}
        truth = {frozenset(i["affected_ids"]) for i in episode["private"]["truth"]["issues"]}
        self.assertEqual(found, truth)


class OracleTest(unittest.TestCase):
    def test_oracle_reward(self) -> None:
        for currencies in (["JPY"], ["KWD"], ["JPY", "KWD"]):
            config = episode_config(currencies)
            truth = generate(21, config)["private"]["truth"]
            with Env(mode="train") as env:
                result = run_episode(env, make_policy("oracle", truth=truth), 21, config)
            self.assertEqual(result["info"]["f1"], 1.0)
            self.assertFalse(result["truncated"])
            inspection = result["steps"] - 1  # every step before submit is an inspection step
            self.assertAlmostEqual(result["reward"], float(1 - STEP_COST * inspection), places=9)
            self.assertEqual(Fraction(result["reward"]).limit_denominator(10**6), 1 - STEP_COST * inspection)


class FuzzTest(unittest.TestCase):
    def test_fuzz_each_new_currency(self) -> None:
        for currency in NEW:
            for seed in range(FUZZ_SEEDS):
                base = ("KZT", "USD", "JPY" if currency == "KWD" else "KWD")[seed % 3]
                config = {"errors": {"fx_error": 1 + seed % 2, "period_cutoff": 1}, "currencies": [currency, "EUR"],
                          "base_currency": base, "size": {"invoices": [30, 40], "background_pairs": [0, 8]},
                          "distractors": seed % 4 != 0}
                config["currencies"] = [c for c in config["currencies"] if c != base]
                with self.subTest(currency=currency, seed=seed):
                    self.assertEqual(verify(generate(5000 + seed, config)), [])


if __name__ == "__main__":
    unittest.main()
