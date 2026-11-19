import json
import pathlib
import tempfile
import unittest

from auditgym.config import INVOICES_PER_ERROR, ConfigError, normalize_config
from auditgym.generate import generate
from auditgym.rng import Rng
from auditgym.storage import episode_files, load_episode, write_episode

TINY = pathlib.Path(__file__).parent.parent / "fixtures" / "tiny"
SMALL = {"invoices": [16, 24], "background_pairs": [0, 10]}


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        for empty in (None, {}):
            config = normalize_config(empty)
            self.assertEqual(config["errors"], {"duplicate_invoice": 1, "allocation_gap": 1,
                                                "fx_error": 1, "period_cutoff": 1})
            self.assertEqual(config["interaction"], "none")
            self.assertTrue(config["distractors"])

    def test_density_is_one_error_per_eight_invoices(self):
        self.assertEqual(INVOICES_PER_ERROR, 8)
        normalize_config({"size": {"invoices": [16, 20]}, "errors": {"fx_error": 2}})
        with self.assertRaises(ConfigError) as caught:
            normalize_config({"size": {"invoices": [15, 20]}, "errors": {"fx_error": 2}})
        self.assertEqual(caught.exception.code, "too_dense")
        with self.assertRaises(ConfigError):  # an interaction counts as two issues
            normalize_config({"size": {"invoices": [16, 20]}, "errors": {"fx_error": 1}, "interaction": "I2"})

    def test_input_is_not_modified(self):
        given = {"errors": {"fx_error": 2}}
        normalize_config(given)
        self.assertEqual(given, {"errors": {"fx_error": 2}})

    def test_partial_errors_fill_with_zero(self):
        self.assertEqual(normalize_config({"errors": {"fx_error": 2}})["errors"],
                         {"duplicate_invoice": 0, "allocation_gap": 0, "fx_error": 2, "period_cutoff": 0})

    def test_clean_company_needs_no_foreign_currency(self):
        config = normalize_config({"errors": {}, "currencies": []})
        self.assertEqual(sum(config["errors"].values()), 0)

    def test_rejections(self):
        cases = {
            "unknown_key": {"seed": 1},
            "bad_config": [],
            "bad_version": {"config_version": "2"},
            "bad_size": {"size": {"invoices": [100, 40]}},
            "bad_currency": {"currencies": ["KZT"]},
            "bad_timezone": {"timezone": "Asia/Almaty"},
            "bad_period": {"period": {"start": "2026-10-05", "months": 1}},
            "bad_errors": {"errors": {"fx_error": -1}},
            "bad_interaction": {"interaction": "I3"},
            "bad_distractors": {"distractors": 1},
            "bad_surface_seed": {"surface_seed": "7"},
            "needs_foreign_currency": {"currencies": [], "errors": {"fx_error": 1}},
            "too_many_errors": {"errors": {"fx_error": 31}},
            "too_dense": {"errors": {"duplicate_invoice": 12}},
            "too_large": {"size": {"invoices": [200, 200], "background_pairs": [900, 900]}},
        }
        for code, config in cases.items():
            with self.subTest(code=code):
                with self.assertRaises(ConfigError) as caught:
                    normalize_config(config)
                self.assertEqual(caught.exception.code, code)

    def test_hardening(self):
        """Typed checks before hashing, full matches with ASCII digits, sane years."""
        cases = [
            {1: "x"}, {"currencies": [["USD"]]}, {"currencies": [{"a": 1}]}, {"base_currency": ["KZT"]},
            {"interaction": ["I1"]}, {"interaction": {}}, {"errors": {"fx_error": [1]}}, {"errors": [1]},
            {"errors": {1: 1}}, {"size": {"invoices": [1.5, 2]}}, {"size": [1, 2]}, {"size": {2: [1, 2]}},
            {"timezone": "+05:00\n"}, {"timezone": "+０5:00"}, {"timezone": "+14:30"}, {"timezone": "-12:30"},
            {"timezone": "+05:00x"}, {"period": {"start": "2026-10-01\n", "months": 1}},
            {"period": {"start": "２026-10-01", "months": 1}}, {"period": {"start": "1999-12-01", "months": 1}},
            {"period": {"start": "2101-01-01", "months": 1}}, {"period": {"start": "2026-10-01", "months": True}},
            {"period": {"start": 202610, "months": 1}}, {"period": {"start": "2026-10-01"}},
            {"period": ["2026-10-01", 1]}, {"config_version": 1}, {"config_version": ["1"]},
            {"surface_seed": 1.0}, {"surface_seed": True}, {"distractors": "true"},
        ]
        for config in cases:
            with self.subTest(config=repr(config)):
                with self.assertRaises(ConfigError):
                    normalize_config(config)
        for good in ({"period": {"start": "2000-01-01", "months": 1}}, {"period": {"start": "2100-12-01", "months": 3}},
                     {"timezone": "+14:00"}, {"timezone": "-12:00"}, {"timezone": "+05:45"}):
            normalize_config(dict(good, errors={}))

    def test_booleans_are_not_integers(self):
        with self.assertRaises(ConfigError):
            normalize_config({"errors": {"fx_error": True}})

    def test_twelve_errors_fit_when_the_company_is_large_enough(self):
        config = normalize_config({"size": {"invoices": [96, 120]},
                                   "errors": {"duplicate_invoice": 3, "allocation_gap": 3,
                                              "fx_error": 3, "period_cutoff": 3}})
        self.assertEqual(sum(config["errors"].values()), 12)


def _junk(rng: Rng, depth: int = 0):
    """A random JSON-compatible value (floats come from json, as an outside caller would send)."""
    kind = rng.below(9 if depth < 4 else 6)
    if kind == 0:
        return None
    if kind == 1:
        return rng.chance(1, 2)
    if kind == 2:
        return rng.randint(-10 ** 12, 10 ** 12) if rng.chance(1, 3) else rng.randint(-3, 250)
    if kind == 3:
        return json.loads(rng.choice(["1.5", "-0.0", "1e308", "2.0", "1e-9"]))
    if kind == 4:
        return rng.choice(["", "USD", "KZT", "I1", "I2", "none", "+05:00", "+05:00\n", "2026-10-01", "1",
                           "٣", "２０２６-10-01", "\x00", "a" * 300, "-11:00", "+14:00", "2026-02-01"])
    if kind == 5:
        return rng.choice([[], [40, 100], [10, 10], [200, 200], ["USD"], ["USD", "EUR", "CNY"], [0, 900]])
    if kind == 6:
        return [_junk(rng, depth + 1) for _ in range(rng.below(4))]
    keys = ["invoices", "background_pairs", "start", "months", "duplicate_invoice", "allocation_gap", "fx_error",
            "period_cutoff", "x", ""]
    return {rng.choice(keys): _junk(rng, depth + 1) for _ in range(rng.below(4))}


class JunkConfigTest(unittest.TestCase):
    """Nothing but ConfigError escapes normalize_config or generate, for any JSON-compatible input."""

    BASE = {"size": SMALL, "errors": {"duplicate_invoice": 1, "fx_error": 1}, "currencies": ["USD"],
            "period": {"start": "2026-10-01", "months": 1}}

    def mutate(self, rng: Rng) -> object:
        if rng.chance(1, 10):
            return _junk(rng)
        config = json.loads(json.dumps(self.BASE))
        for _ in range(rng.randint(1, 3)):
            key = rng.choice(["config_version", "size", "currencies", "base_currency", "timezone", "period",
                              "errors", "interaction", "distractors", "surface_seed", "extra"])
            target = config
            if key in ("size", "period", "errors") and rng.chance(1, 2) and isinstance(config.get(key), dict):
                target = config[key]
                key = rng.choice(sorted(target) + ["zz"])
            target[key] = _junk(rng)
        return config

    def test_random_configs(self):
        rng = Rng(20261119, "junk")
        accepted = rejected = 0
        for case in range(400):
            config = self.mutate(rng)
            try:
                normalize_config(config)
            except ConfigError:
                rejected += 1
                continue
            except Exception as error:  # noqa: BLE001 - the point of the test
                self.fail(f"case {case}: {type(error).__name__} escaped normalize_config for {config!r:.200}")
            try:
                generate(case, config)
                accepted += 1
            except ConfigError:
                rejected += 1
            except Exception as error:  # noqa: BLE001
                self.fail(f"case {case}: {type(error).__name__} escaped generate for {config!r:.200}")
        self.assertGreater(accepted, 5)  # mostly junk; valid ones must generate
        self.assertGreater(rejected, 100)

    def test_deep_nesting(self):
        value: object = 1
        for _ in range(5000):
            value = [value]
        for config in ({"errors": value}, {"size": {"invoices": value}}, value, {"currencies": value}):
            with self.assertRaises(ConfigError):
                normalize_config(config)


class StorageTest(unittest.TestCase):
    def test_fixture_round_trip_is_byte_identical(self):
        episode = load_episode(TINY)
        for relative, content in episode_files(episode).items():
            with self.subTest(file=relative):
                self.assertEqual(content, (TINY / relative).read_text(encoding="utf-8"))

    def test_private_snapshot_files(self):
        files = episode_files(generate(3, {"size": SMALL, "errors": {"duplicate_invoice": 1}}))
        for name in ("private/pre_ledger.csv", "private/pre_invoices.jsonl", "private/pre_payments.jsonl"):
            self.assertIn(name, files)

    def test_write_then_load_gives_the_same_episode(self):
        for episode in (load_episode(TINY), generate(5, {"size": SMALL, "errors": {"period_cutoff": 1}})):
            with tempfile.TemporaryDirectory() as directory:
                write_episode(episode, directory)
                self.assertEqual(load_episode(directory), episode)

    def test_missing_links_are_none_and_money_is_int(self):
        row = load_episode(TINY)["public"]["ledger"][0]
        self.assertIsNone(row["invoice_id"])
        self.assertIsInstance(row["debit_minor"], int)


if __name__ == "__main__":
    unittest.main()
