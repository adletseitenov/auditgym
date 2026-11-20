"""Case ranges (docs/CASE.md section 03): what we ship and generate by default stays inside them."""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from auditgym.generate import generate
from tools import case_ranges


class MeasureTest(unittest.TestCase):
    def test_definitions_follow_the_open_files(self) -> None:
        episode = generate(7, {"errors": {"duplicate_invoice": 2}, "currencies": ["USD"]})
        sizes = case_ranges.measure(episode)
        public = episode["public"]
        self.assertEqual(sizes["postings"], public["manifest"]["counts"]["ledger"])
        self.assertEqual(sizes["documents"], public["manifest"]["counts"]["invoices"] + public["manifest"]["counts"]["payments"])
        self.assertEqual(sizes["errors"], 2)

    def test_in_ranges_edges(self) -> None:
        self.assertTrue(case_ranges.in_ranges({"postings": 200, "documents": 20, "errors": 0}))
        self.assertTrue(case_ranges.in_ranges({"postings": 2000, "documents": 200, "errors": 12}))
        for bad in ({"postings": 199, "documents": 20, "errors": 0}, {"postings": 2001, "documents": 20, "errors": 0},
                    {"postings": 200, "documents": 19, "errors": 0}, {"postings": 200, "documents": 201, "errors": 0},
                    {"postings": 200, "documents": 20, "errors": 13}):
            self.assertFalse(case_ranges.in_ranges(bad), bad)

    def test_a_config_outside_the_ranges_is_still_accepted(self) -> None:
        sizes = case_ranges.measure(generate(3, {"size": {"invoices": [10, 10], "background_pairs": [0, 0]}, "errors": {}}))
        self.assertLess(sizes["postings"], case_ranges.POSTINGS[0])  # allowed for a caller, never a shipped default

    def test_shipped_subjects_pass(self) -> None:
        rows = case_ranges.run(2)
        labels = [row["label"] for row in rows]
        self.assertEqual(labels[:2], ["default (config omitted)", "default ({})"])
        self.assertEqual(len([label for label in labels if label.startswith(("train/", "dev/"))]), 20)
        self.assertEqual(len([label for label in labels if label.startswith("level ")]), 4)
        self.assertEqual([row["label"] for row in rows if not row["ok"]], [])

    def test_main_reports_pass_and_fail(self) -> None:
        rows = [{"label": "a", "n": 1, "ok": True, "postings": (200, 300), "documents": (20, 20), "errors": (0, 0)},
                {"label": "b", "n": 1, "ok": False, "postings": (10, 10), "documents": (20, 20), "errors": (0, 0)}]
        self.assertIn("FAIL (1 of 2: b)", case_ranges.summary(rows))
        self.assertIn("PASS (1 subjects)", case_ranges.summary(rows[:1]))
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(case_ranges.main(["--seeds", "1"]), 0)
        self.assertIn("case ranges: PASS", out.getvalue())


if __name__ == "__main__":
    unittest.main()
