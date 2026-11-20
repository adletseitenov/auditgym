"""Small hardening of the generator zone after the independent review (item 4): config offsets,
long seeds, Rng.below beyond 64 bits, the Decimal context, verify's manifest/policy checks, the
determinism lint and the fuzz verdict on bad retries."""
import ast
import io
import copy
import decimal
import unittest
from unittest import mock

from auditgym import SCHEMA_VERSION
from auditgym.config import ConfigError, normalize_config, valid_offset
from auditgym.generate import generate
from auditgym.limits import MAX_SEED_BITS
from auditgym.rng import Rng
from auditgym.storage import episode_files
from auditgym.verify import POLICY_KEYS, verify
from tools import fuzz, lint_determinism
from tools.schema_lite import load_schema, validate

SMALL = {"size": {"invoices": [16, 24], "background_pairs": [0, 10]}, "currencies": ["USD", "EUR"],
         "errors": {"fx_error": 1, "allocation_gap": 1}}


class OffsetTest(unittest.TestCase):
    def test_minus_zero_is_refused(self):
        self.assertFalse(valid_offset("-00:00"))
        with self.assertRaises(ConfigError) as caught:
            normalize_config({"timezone": "-00:00"})
        self.assertEqual(caught.exception.code, "bad_timezone")
        normalize_config({"timezone": "+00:00"})
        normalize_config({"timezone": "-00:30"})

    def test_schema_agrees_with_valid_offset(self):
        schema = load_schema("config")
        for sign in "+-":
            for hours in range(16):
                for minutes in ("00", "15", "30", "45"):
                    text = f"{sign}{hours:02d}:{minutes}"
                    with self.subTest(offset=text):
                        self.assertEqual(validate({"timezone": text}, schema) == [], valid_offset(text))


class ConfigSchemaTest(unittest.TestCase):
    def test_limits_agree_with_normalize_config(self):
        schema = load_schema("config")
        limit = (1 << MAX_SEED_BITS) - 1
        for value in (limit, -limit, 0, None):
            self.assertEqual(validate({"surface_seed": value}, schema), [])
            normalize_config({"surface_seed": value})
        for value in (limit + 1, -limit - 1):
            self.assertNotEqual(validate({"surface_seed": value}, schema), [])
            with self.assertRaises(ConfigError):
                normalize_config({"surface_seed": value})
        self.assertEqual(validate({"errors": {"fx_error": 25}}, schema), [])
        self.assertNotEqual(validate({"errors": {"fx_error": 26}}, schema), [])
        # The schema bounds one count (25); the episode total is bounded by normalize_config (12, CASE 03).
        with self.assertRaises(ConfigError):
            normalize_config({"errors": {"duplicate_invoice": 13}, "size": {"invoices": [200, 200]}})
        normalize_config({"errors": {"duplicate_invoice": 12}, "size": {"invoices": [200, 200],
                                                                        "background_pairs": [0, 0]}})


class SeedTest(unittest.TestCase):
    def test_seed_longer_than_the_limit_is_a_config_error(self):
        for seed in (1 << MAX_SEED_BITS, -(1 << MAX_SEED_BITS), 10 ** 5000):
            with self.assertRaises(ConfigError) as caught:
                generate(seed, {"errors": {}, "size": {"invoices": [10, 12], "background_pairs": [0, 0]}})
            self.assertEqual(caught.exception.code, "bad_seed")
        generate((1 << MAX_SEED_BITS) - 1, {"errors": {}, "size": {"invoices": [10, 12], "background_pairs": [0, 0]}})


class RngBelowTest(unittest.TestCase):
    def test_up_to_64_bits_the_stream_is_unchanged(self):
        def old_below(rng, n):
            limit = (1 << 64) - ((1 << 64) % n)
            while True:
                value = rng.u64()
                if value < limit:
                    return value % n
        for n in (1, 2, 7, 1000, (1 << 63) + 5, 1 << 64):
            new, old = Rng(5, "below"), Rng(5, "below")
            self.assertEqual([new.below(n) for _ in range(20)], [old_below(old, n) for _ in range(20)])

    def test_beyond_64_bits_it_returns(self):
        rng = Rng(1, "big")
        for n in (1 << 66, 3 << 200, (1 << 1000) - 1):
            values = [rng.below(n) for _ in range(50)]
            self.assertTrue(all(0 <= v < n for v in values))
            self.assertGreater(max(values), 1 << 64)
        self.assertIn(Rng(2, "edge").below((1 << 64) + 1), range((1 << 64) + 1))


class DecimalContextTest(unittest.TestCase):
    CONFIG = {"currencies": ["USD", "EUR", "CNY"], "period": {"start": "2026-10-01", "months": 3},
              "errors": {"fx_error": 3}}

    def test_caller_context_does_not_change_the_episode(self):
        expected = [episode_files(generate(seed, self.CONFIG)) for seed in range(3)]
        for context in (decimal.Context(prec=9), decimal.Context(prec=16),
                        decimal.Context(prec=20, rounding=decimal.ROUND_DOWN)):
            with decimal.localcontext(context):
                for seed in range(3):
                    episode = generate(seed, self.CONFIG)
                    self.assertEqual(episode_files(episode), expected[seed])
                    self.assertEqual(verify(episode), [])


class VerifyManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = generate(3, SMALL)
        assert verify(cls.episode) == []

    def problems(self, change):
        episode = copy.deepcopy(self.episode)
        change(episode)
        return verify(episode)

    def test_manifest_must_repeat_the_policy(self):
        for key, value in (("base_currency", "EUR"), ("timezone", "+03:00"),
                           ("period_start", "2026-09-01T00:00:00+05:00"), ("period_end", "2026-12-01T00:00:00+05:00")):
            with self.subTest(key=key):
                problems = self.problems(lambda e: e["public"]["manifest"].__setitem__(key, value))
                self.assertIn(f"manifest: {key} differs from the policy", problems)

    def test_policy_keys_are_the_whitelist(self):
        self.assertEqual(tuple(self.episode["public"]["policy"]), POLICY_KEYS)
        problems = self.problems(lambda e: e["public"]["policy"].__setitem__("error_count", 2))
        self.assertIn("policy: keys differ from the published whitelist", problems)
        problems = self.problems(lambda e: e["public"]["policy"].pop("payment_rule"))
        self.assertIn("policy: keys differ from the published whitelist", problems)

    def test_schema_version(self):
        for where in ("manifest", "truth"):
            def change(episode, where=where):
                part = episode["public"]["manifest"] if where == "manifest" else episode["private"]["truth"]
                part["schema_version"] = SCHEMA_VERSION + "0"
            with self.subTest(where=where):
                self.assertTrue(any(p.startswith("manifest: schema_version") for p in self.problems(change)))

    def test_rate_must_be_a_plain_decimal_string(self):
        for bad in ("4.7E+2", "+470.1234", " 470.1234", "470.1234 ", "4_70.1234", "-470.1234"):
            def change(episode, bad=bad):
                episode["public"]["fx"][0]["rate"] = bad
            with self.subTest(rate=bad):
                self.assertTrue(any("is not a plain decimal string" in p for p in self.problems(change)))


class LintTest(unittest.TestCase):
    def found(self, source):
        return [why for _, why in lint_determinism.ast_problems(ast.parse(source))]

    def test_forbidden_constructs(self):
        for source in ("import datetime\nx = datetime.date.today()", "from datetime import date\nx = date.today()",
                       "import time\nx = time.time_ns()", "from time import monotonic", "from os import urandom",
                       "x = hash('a')", "x = 7 / 2", "x = n / 3", "r = __import__('random')",
                       "import importlib\nr = importlib.import_module('random')", "from random import choice"):
            with self.subTest(source=source):
                self.assertTrue(self.found(source), source)

    def test_allowed_constructs(self):
        for source in ("from decimal import Decimal\nx = Decimal(7) / 2", "x = level * Decimal(3) / 1000",
                       "x = 7 // 2", "import hashlib\nx = hashlib.sha256(b'').hexdigest()",
                       "from fractions import Fraction\nx = Fraction(1) / 3", "x = a / b",
                       "import importlib\ng = importlib.import_module('gymnasium')"):
            with self.subTest(source=source):
                self.assertEqual(self.found(source), [], source)

    def test_core_is_clean(self):
        self.assertEqual(lint_determinism.main(), 0)


class FuzzRetryVerdictTest(unittest.TestCase):
    def test_internal_and_verify_retries_are_failures(self):
        for reason, failing in (("internal: KeyError", True), ("verify: ledger: x", True), ("no target: I1", False)):
            episode = {"private": {"provenance": {"attempt": 1, "retries": [{"attempt": 0, "reason": reason}]}}}
            with self.subTest(reason=reason), mock.patch.object(fuzz, "check_episode", return_value=([], episode)), \
                    mock.patch.object(fuzz, "grid", return_value=[("clean", {})]):
                result = fuzz.run(1, out=io.StringIO())
                self.assertEqual(bool(result["failures"]), failing, result["failures"])


if __name__ == "__main__":
    unittest.main()
