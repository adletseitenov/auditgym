"""tools/episodes.py: the open train and dev sets."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from auditgym.storage import load_episode
from tools import episodes as episodes_tool

PUBLIC = {"manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"}


class EpisodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.first = pathlib.Path(cls.scratch.name) / "a"
        cls.second = pathlib.Path(cls.scratch.name) / "b"
        cls.entries = episodes_tool.build(cls.first, quiet=True)
        episodes_tool.build(cls.second, quiet=True)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_rebuild_is_byte_identical(self):
        first, second = episodes_tool.tree_bytes(self.first), episodes_tool.tree_bytes(self.second)
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_rebuild_over_an_existing_tree_gives_the_same_bytes(self):
        before = episodes_tool.tree_bytes(self.first)
        (self.first / "train" / "ep_99").mkdir()  # a stale episode from an older plan
        episodes_tool.build(self.first, quiet=True)
        self.assertEqual(episodes_tool.tree_bytes(self.first), before)

    def test_ten_per_split_with_public_and_private_apart(self):
        for split in ("train", "dev"):
            names = sorted(p.name for p in (self.first / split).iterdir())
            self.assertEqual(names, [f"ep_{k:02d}" for k in range(1, 11)])
            for name in names:
                directory = self.first / split / name
                self.assertEqual({p.name for p in directory.iterdir()}, {"public", "private"})
                self.assertEqual({p.name for p in (directory / "public").iterdir()}, PUBLIC)
                self.assertIn("truth.json", {p.name for p in (directory / "private").iterdir()})

    def test_names_and_public_files_carry_no_seed(self):
        for entry in self.entries:
            seed = str(entry["seed"])
            directory = self.first / entry["split"] / entry["name"]
            self.assertNotIn(seed, str(directory.relative_to(self.first)))
            for path in (directory / "public").iterdir():
                self.assertNotIn(f'"seed"', path.read_text(encoding="utf-8"), path)

    def test_coverage_of_the_sets(self):
        plan = json.loads(episodes_tool.PLAN.read_text(encoding="utf-8"))
        for split in ("train", "dev"):
            episodes = [load_episode(self.first / split / f"ep_{k:02d}") for k in range(1, 11)]
            clean = [e for e in episodes if not e["private"]["truth"]["issues"]]
            self.assertGreaterEqual(len(clean), 2, split)
            offsets = {e["public"]["manifest"]["timezone"] for e in episodes}
            currencies = {tuple(sorted({inv["currency"] for inv in e["public"]["invoices"]})) for e in episodes}
            sizes = {e["public"]["manifest"]["counts"]["invoices"] for e in episodes}
            self.assertGreaterEqual(len(offsets), 3, split)
            self.assertGreaterEqual(len(currencies), 3, split)
            self.assertGreaterEqual(len(sizes), 5, split)
            types = {issue["type"] for e in episodes for issue in e["private"]["truth"]["issues"]}
            supported = {t for t in ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
                         if episodes_tool.unsupported({"errors": {t: 1}, "currencies": ["USD"]}) is None}
            self.assertEqual(types & supported, supported, split)
            for scenario in ("I1", "I2"):
                if episodes_tool.unsupported({"interaction": scenario, "errors": {}, "currencies": ["USD"]}) is None:
                    self.assertTrue(any((e["config"] or {}).get("interaction") == scenario
                                        for e in self.entries if e["split"] == split), (split, scenario))
        train_seeds = {slot["seed"] for slot in plan["train"]}
        dev_seeds = {slot["seed"] for slot in plan["dev"]}
        self.assertFalse(train_seeds & dev_seeds)
        train_configs = {json.dumps(e["config"], sort_keys=True) for e in self.entries if e["split"] == "train"}
        dev_configs = {json.dumps(e["config"], sort_keys=True) for e in self.entries if e["split"] == "dev"}
        self.assertTrue(dev_configs - train_configs)


if __name__ == "__main__":
    unittest.main()
