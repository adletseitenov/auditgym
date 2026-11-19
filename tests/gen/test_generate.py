import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

from auditgym.config import ConfigError
from auditgym.generate import generate
from auditgym.storage import episode_files, load_episode, write_episode
from auditgym.verify import verify

CLEAN = {"errors": {}}
GRID = [
    {"errors": {}, "currencies": []},
    {"errors": {}, "currencies": ["USD"]},
    {"errors": {}, "currencies": ["USD", "EUR", "CNY"], "timezone": "-11:00"},
    {"errors": {}, "currencies": ["RUB"], "timezone": "+14:00", "period": {"start": "2026-12-01", "months": 3}},
    {"errors": {}, "base_currency": "USD", "currencies": ["KZT", "EUR"], "timezone": "+05:30",
     "period": {"start": "2028-02-01", "months": 1}},
    {"errors": {}, "size": {"invoices": [10, 12], "background_pairs": [0, 5]}},
    {"errors": {}, "size": {"invoices": [200, 200], "background_pairs": [480, 490]}},
]


class CleanCompanyTest(unittest.TestCase):
    def test_books_are_well_formed_on_many_seeds(self):
        for index, config in enumerate(GRID):
            for seed in range(30):
                with self.subTest(config=index, seed=seed):
                    episode = generate(seed, config)
                    self.assertEqual(verify(episode), [])
                    self.assertEqual(episode["private"]["truth"]["issues"], [])

    def test_sizes_follow_the_config(self):
        episode = generate(5, {"errors": {}, "size": {"invoices": [20, 25], "background_pairs": [40, 60]}})
        counts = episode["public"]["manifest"]["counts"]
        self.assertTrue(20 <= counts["invoices"] <= 25)
        self.assertLessEqual(counts["ledger"], 2000)

    def test_ledger_is_in_the_period_and_in_the_base_currency(self):
        episode = generate(3, GRID[2])
        manifest = episode["public"]["manifest"]
        for row in episode["public"]["ledger"]:
            self.assertEqual(row["currency"], manifest["base_currency"])
            self.assertTrue(row["timestamp"].endswith("-11:00"))

    def test_no_two_invoices_share_a_signature(self):
        episode = generate(11, GRID[6])
        signatures = {(inv["entity_id"], inv["currency"], inv["total_minor"],
                       json.dumps(sorted(inv["line_items"], key=json.dumps), sort_keys=True))
                      for inv in episode["public"]["invoices"]}
        self.assertEqual(len(signatures), len(episode["public"]["invoices"]))

    def test_seed_and_config_do_not_appear_in_open_records(self):
        episode = generate(918273645, CLEAN)
        text = json.dumps(episode["public"])
        self.assertNotIn("918273645", text)
        self.assertNotIn("surface_seed", text)

    def test_errors_are_refused_until_they_are_implemented(self):
        with self.assertRaises(ConfigError) as caught:
            generate(1, None)
        self.assertEqual(caught.exception.code, "not_implemented")

    def test_bad_seed(self):
        for seed in ("1", 1.0, True, None):
            with self.assertRaises(ConfigError):
                generate(seed, CLEAN)


class VerifyCatchesBrokenBooksTest(unittest.TestCase):
    """verify() must be able to fail: break the books by hand and expect a problem."""

    def setUp(self):
        self.episode = generate(2, {"errors": {}, "currencies": ["USD"]})

    def broken(self, change):
        episode = copy.deepcopy(self.episode)
        change(episode)
        return verify(episode)

    def test_unbalanced_row(self):
        def change(e):
            e["public"]["ledger"][0]["debit_minor"] += 1 if e["public"]["ledger"][0]["debit_minor"] else 0
            e["public"]["ledger"][0]["credit_minor"] += 1 if e["public"]["ledger"][0]["credit_minor"] else 0
        self.assertTrue(self.broken(change))

    def test_missing_link(self):
        def change(e):
            row = next(r for r in e["public"]["ledger"] if r["invoice_id"])
            row["invoice_id"] = "nosuchid00"
        self.assertTrue(self.broken(change))

    def test_wrong_amount_before_injection(self):
        def change(e):
            invoice = next(i for i in e["public"]["invoices"] if i["currency"] != "KZT")
            for row in e["private"]["pre_ledger"]:
                if row["invoice_id"] == invoice["invoice_id"] and row["payment_id"] is None:
                    side = "debit_minor" if row["debit_minor"] else "credit_minor"
                    row[side] += 100
        self.assertTrue(any("policy amount" in p for p in self.broken(change)))

    def test_row_outside_the_period(self):
        def change(e):
            e["public"]["ledger"][0]["timestamp"] = "2026-11-01T00:00:00+05:00"
        self.assertTrue(any("outside the period" in p for p in self.broken(change)))

    def test_manifest_count(self):
        def change(e):
            e["public"]["manifest"]["counts"]["ledger"] += 1
        self.assertTrue(self.broken(change))


class ReproducibilityTest(unittest.TestCase):
    def test_same_seed_same_bytes(self):
        config = GRID[2]
        self.assertEqual(episode_files(generate(7, config)), episode_files(generate(7, config)))

    def test_different_seeds_differ(self):
        self.assertNotEqual(episode_files(generate(7, CLEAN)), episode_files(generate(8, CLEAN)))

    def test_independent_of_process_environment(self):
        code = ("import hashlib, json; from auditgym.generate import generate; "
                "from auditgym.storage import episode_files; "
                "f = episode_files(generate(7, {'errors': {}, 'currencies': ['USD', 'EUR']})); "
                "print(hashlib.sha256(json.dumps(f, sort_keys=True).encode()).hexdigest())")
        digests = set()
        for hash_seed, zone, locale in (("0", "UTC", "C"), ("4242", "Asia/Tokyo", "ru_RU.UTF-8")):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, TZ=zone, LC_ALL=locale)
            digests.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                       text=True, check=True).stdout)
        self.assertEqual(len(digests), 1)

    def test_written_episode_loads_back(self):
        episode = generate(4, GRID[1])
        with tempfile.TemporaryDirectory() as directory:
            write_episode(episode, directory)
            self.assertEqual(load_episode(directory), episode)


class SurfaceTest(unittest.TestCase):
    """A different surface seed gives the same episode under a renaming of ids."""

    def canonical(self, episode):
        back = {public: internal for internal, public in episode["private"]["provenance"]["id_map"].items()}

        def rename(value):
            if isinstance(value, dict):
                return {key: rename(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rename(item) for item in value]
            return back.get(value, value) if isinstance(value, str) else value

        tables = {}
        for name in ("ledger", "invoices", "payments"):
            records = rename(copy.deepcopy(episode["public"][name]))
            for record in records:
                if record.get("invoice_refs"):
                    record["invoice_refs"].sort()
            tables[name] = sorted(json.dumps(record, sort_keys=True) for record in records)
        tables["fx"] = episode["public"]["fx"]
        return tables

    def test_same_story_under_other_ids(self):
        first = generate(9, {"errors": {}, "surface_seed": 1})
        second = generate(9, {"errors": {}, "surface_seed": 2})
        self.assertNotEqual(first["public"]["ledger"][0]["row_id"], second["public"]["ledger"][0]["row_id"])
        self.assertEqual(self.canonical(first), self.canonical(second))
        self.assertEqual(verify(second), [])

    def test_ids_are_opaque_and_unique(self):
        episode = generate(9, CLEAN)
        ids = list(episode["private"]["provenance"]["id_map"].values())
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(len(i) == 10 and i.isalnum() and i.islower() for i in ids))


if __name__ == "__main__":
    unittest.main()
