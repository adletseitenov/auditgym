import pathlib
import tempfile
import unittest

from auditgym.config import ConfigError, normalize_config
from auditgym.storage import episode_files, load_episode, write_episode

TINY = pathlib.Path(__file__).parent.parent / "fixtures" / "tiny"


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        for empty in (None, {}):
            config = normalize_config(empty)
            self.assertEqual(config["errors"], {"duplicate_invoice": 1, "allocation_gap": 1,
                                                "fx_error": 1, "period_cutoff": 1})
            self.assertEqual(config["interaction"], "none")

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

    def test_booleans_are_not_integers(self):
        with self.assertRaises(ConfigError):
            normalize_config({"errors": {"fx_error": True}})

    def test_twelve_errors_fit_when_the_company_is_large_enough(self):
        config = normalize_config({"size": {"invoices": [60, 120]},
                                   "errors": {"duplicate_invoice": 3, "allocation_gap": 3,
                                              "fx_error": 3, "period_cutoff": 3}})
        self.assertEqual(sum(config["errors"].values()), 12)


class StorageTest(unittest.TestCase):
    def test_fixture_round_trip_is_byte_identical(self):
        episode = load_episode(TINY)
        for relative, content in episode_files(episode).items():
            with self.subTest(file=relative):
                self.assertEqual(content, (TINY / relative).read_text(encoding="utf-8"))

    def test_write_then_load_gives_the_same_episode(self):
        episode = load_episode(TINY)
        with tempfile.TemporaryDirectory() as directory:
            write_episode(episode, directory)
            self.assertEqual(load_episode(directory), episode)

    def test_missing_links_are_none_and_money_is_int(self):
        row = load_episode(TINY)["public"]["ledger"][0]
        self.assertIsNone(row["invoice_id"])
        self.assertIsInstance(row["debit_minor"], int)


if __name__ == "__main__":
    unittest.main()
