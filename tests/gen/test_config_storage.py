import json
import pathlib
import tempfile
import unittest

from auditgym import limits
from auditgym.config import INVOICES_PER_ERROR, ConfigError, full_read_pages, ledger_budget, normalize_config
from auditgym.generate import generate
from auditgym.verify import verify
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
            "too_dense": {"size": {"invoices": [40, 60]}, "errors": {"duplicate_invoice": 12}},
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


ALL_ONE = {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1}
TWELVE = {"duplicate_invoice": 3, "allocation_gap": 3, "fx_error": 3, "period_cutoff": 3}


class NaturalConfigTest(unittest.TestCase):
    """docs/AUDIT.md D5: configs a reader of the case would write are accepted and generate."""

    NATURAL = [
        {"base_currency": "USD"}, {"currencies": []}, {"size": {"invoices": [20, 40]}}, {"errors": TWELVE},
        {"interaction": "I1"}, {"interaction": "I2"}, {"interaction": "I2", "currencies": []},
        {"base_currency": "EUR", "interaction": "I1"}, {"size": {"invoices": [10, 12]}},
        {"size": {"background_pairs": [0, 0]}}, {"errors": {"fx_error": 2}, "base_currency": "USD"},
        {"period": {"start": "2026-12-01", "months": 3}}, {"timezone": "-11:00", "currencies": ["EUR", "CNY", "RUB"]},
    ]

    def test_natural_configs_are_accepted_and_generate(self):
        for config in self.NATURAL:
            with self.subTest(config=config):
                normalized = normalize_config(config)
                self.assertEqual(verify(generate(1, config)), [])
                issues = sum(normalized["errors"].values()) + (normalized["interaction"] != "none") * 2
                self.assertLessEqual(issues * INVOICES_PER_ERROR, normalized["size"]["invoices"][0])

    def test_default_foreign_currency_is_the_first_supported_other_than_base(self):
        self.assertEqual(normalize_config({})["currencies"], ["USD"])
        self.assertEqual(normalize_config({"base_currency": "USD"})["currencies"], ["KZT"])
        self.assertEqual(normalize_config({"base_currency": "KZT", "currencies": ["EUR"]})["currencies"], ["EUR"])

    def test_default_errors_shrink_to_the_size(self):
        self.assertEqual(normalize_config({})["errors"], ALL_ONE)
        self.assertEqual(normalize_config({"size": {"invoices": [20, 40]}})["errors"],
                         {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 0, "period_cutoff": 0})
        self.assertEqual(sum(normalize_config({"size": {"invoices": [10, 12]}})["errors"].values()), 1)
        self.assertEqual(normalize_config({"currencies": []})["errors"]["fx_error"], 0)
        self.assertEqual(sum(normalize_config({"interaction": "I1", "size": {"invoices": [20, 40]}})["errors"].values()), 0)

    def test_default_size_grows_to_hold_explicit_errors(self):
        config = normalize_config({"errors": TWELVE})
        self.assertEqual(config["size"]["invoices"], [96, 100])
        self.assertEqual(normalize_config({"errors": {}})["size"]["invoices"], [48, 100])
        with self.assertRaises(ConfigError) as caught:  # an explicit size is never changed
            normalize_config({"errors": TWELVE, "size": {"invoices": [48, 100]}})
        self.assertEqual(caught.exception.code, "too_dense")

    def test_case_ranges_are_reachable(self):
        """Up to 12 errors, up to 2 000 ledger rows, 20 to 200 documents."""
        self.assertEqual(len(generate(2, {"errors": TWELVE})["private"]["truth"]["issues"]), 12)
        big = generate(3, {"errors": {}, "size": {"invoices": [10, 10], "background_pairs": [975, 1000]}})["public"]
        self.assertTrue(1950 <= len(big["ledger"]) <= limits.MAX_LEDGER_ROWS, len(big["ledger"]))
        many = generate(4, {"size": {"invoices": [200, 200], "background_pairs": [0, 1000]}})["public"]
        self.assertGreaterEqual(len(many["invoices"]) + len(many["payments"]), 200)
        self.assertLessEqual(len(many["ledger"]), limits.MAX_LEDGER_ROWS)
        few = generate(5, {"errors": {}, "size": {"invoices": [10, 10], "background_pairs": [0, 0]}})["public"]
        self.assertLessEqual(len(few["invoices"]) + len(few["payments"]), 20)


class LimitBoundaryTest(unittest.TestCase):
    """docs/AUDIT.md Q4: every limit holds exactly at its value and fails one past it."""

    def code(self, config):
        try:
            normalize_config(config)
        except ConfigError as error:
            return error.code
        return None

    def test_one_constant_per_limit(self):
        from auditgym import config, solver, verify as verify_module
        self.assertIs(config.MAX_LEDGER_ROWS, limits.MAX_LEDGER_ROWS)
        self.assertEqual(verify_module.MAX_LEDGER_ROWS, limits.MAX_LEDGER_ROWS)
        self.assertEqual(verify_module.MAX_EVIDENCE, limits.MAX_EVIDENCE_IDS)
        self.assertEqual(solver.MAX_EVIDENCE, limits.MAX_EVIDENCE_IDS)
        self.assertEqual(config.MAX_FINDINGS, limits.MAX_FINDINGS)

    def test_ledger_rows(self):
        at = {"errors": {}, "size": {"invoices": [10, 10], "background_pairs": [975, 980]}}
        past = {"errors": {}, "size": {"invoices": [11, 11], "background_pairs": [973, 980]}}
        self.assertEqual(ledger_budget(10, 975, 0), 2000)
        self.assertEqual(ledger_budget(11, 973, 0), 2001)
        self.assertIsNone(self.code(at))
        self.assertEqual(self.code(past), "too_large")

    def test_pages(self):
        self.assertEqual(full_read_pages(2000, 250, 500), limits.MAX_PAGES)
        self.assertEqual(full_read_pages(2000, 250, 501), limits.MAX_PAGES + 1)
        # The largest accepted config still reads in at most MAX_PAGES pages.
        worst = normalize_config({"errors": {"duplicate_invoice": 25}, "currencies": ["USD", "EUR", "CNY"],
                                  "size": {"invoices": [200, 200], "background_pairs": [0, 1000]},
                                  "period": {"start": "2026-10-01", "months": 3}})
        documents = worst["size"]["invoices"][1] + 25
        self.assertLessEqual(full_read_pages(limits.MAX_LEDGER_ROWS, documents, 3 * (31 * 3 + 10)), limits.MAX_PAGES)

    def test_months(self):
        self.assertIsNone(self.code({"period": {"start": "2026-10-01", "months": limits.MAX_PERIOD_MONTHS}}))
        self.assertEqual(self.code({"period": {"start": "2026-10-01", "months": limits.MAX_PERIOD_MONTHS + 1}}),
                         "bad_period")

    def test_density(self):
        self.assertIsNone(self.code({"size": {"invoices": [32, 40]}, "errors": ALL_ONE}))
        self.assertEqual(self.code({"size": {"invoices": [31, 40]}, "errors": ALL_ONE}), "too_dense")
        self.assertIsNone(self.code({"size": {"invoices": [200, 200]}, "errors": {"duplicate_invoice": 25}}))
        self.assertEqual(self.code({"size": {"invoices": [200, 200]}, "errors": {"duplicate_invoice": 26}}), "too_dense")

    def test_findings(self):
        self.assertEqual(self.code({"errors": {"duplicate_invoice": limits.MAX_FINDINGS + 1}}), "too_many_errors")
        self.assertNotEqual(self.code({"errors": {"duplicate_invoice": limits.MAX_FINDINGS}}), "too_many_errors")

    def test_size_bounds(self):
        for name, (low, high) in (("invoices", (limits.MIN_INVOICES, limits.MAX_INVOICES)),
                                  ("background_pairs", (limits.MIN_BACKGROUND_PAIRS, limits.MAX_BACKGROUND_PAIRS))):
            base = {"errors": {}, "size": {"invoices": [10, 10], "background_pairs": [0, 0]}}
            at = dict(base, size=dict(base["size"], **{name: [low, high]}))
            self.assertIsNone(self.code(at), name)
            for bad in ([low - 1, high], [low, high + 1]):
                self.assertEqual(self.code(dict(base, size=dict(base["size"], **{name: bad}))), "bad_size", (name, bad))


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
