"""policies.runner reads a submit and the results journal the way the environment and the evaluator do."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from auditgym.journal import seal
from policies.runner import _submitted, read_results

MESSAGE = {"episode_id": "ep_1", "findings": [{"type": "fx_error", "affected_ids": ["d1"],
                                             "claimed_amount_minor": 5, "currency": "KZT",
                                             "evidence_ids": ["d1", "r1"], "explanation": ""}]}
FINDINGS = MESSAGE["findings"]


class SubmittedFormsTest(unittest.TestCase):
    """Review note: the runner understood only the canonical submit, so the report counted no
    findings for a policy that used another accepted form."""

    def test_every_accepted_form_gives_the_same_message(self):
        forms = {
            "canonical": {"action": "submit", "findings": MESSAGE},
            "nested args": {"tool": "submit", "args": {"findings": MESSAGE}},
            "upper-case name": {"action": "SUBMIT", "findings": MESSAGE},
            "bare list": {"action": "submit", "findings": FINDINGS},
            "flat episode_id": {"action": "submit", "episode_id": "ep_1", "findings": FINDINGS},
            "JSON string": {"action": "submit", "findings": json.dumps(MESSAGE)},
            "findings.json": {"action": "submit", "findings.json": MESSAGE},
            "whole action as JSON text": json.dumps({"action": "submit", "findings": MESSAGE}),
        }
        for name, action in forms.items():
            with self.subTest(name):
                self.assertEqual(_submitted(action, "ep_1"), MESSAGE)

    def test_not_a_submit(self):
        for action in ({"action": "list_records", "table": "ledger"}, {"action": "nope"}, None, "text", 3):
            with self.subTest(action=action):
                self.assertIsNone(_submitted(action, "ep_1"))


class SignedResultsTest(unittest.TestCase):
    KEY = b"k" * 32

    def _journal(self, directory: str, rows: list[dict]) -> pathlib.Path:
        path = pathlib.Path(directory) / "results.jsonl"
        tail = b""
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                line = json.dumps(seal(self.KEY, row, tail), ensure_ascii=False) + "\n"
                handle.write(line)
                tail += line.encode("utf-8")
        return path

    def test_signed_journal_reads_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._journal(directory, [{"episode_id": "a", "reward": -1.0}, {"episode_id": "b", "reward": 0.9}])
            rows = read_results(path, journal_key=self.KEY)
            self.assertEqual([(r["episode_id"], r["reward"]) for r in rows], [("a", -1.0), ("b", 0.9)])

    def test_a_forged_or_edited_row_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._journal(directory, [{"episode_id": "a", "reward": -1.0}])
            honest = path.read_text(encoding="utf-8")
            forged = honest + json.dumps({"episode_id": "a", "reward": 1.0}) + "\n"
            edited = honest.replace('"reward": -1.0', '"reward": 1.0')
            for name, text in (("appended without the key", forged), ("edited", edited)):
                with self.subTest(name):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_results(path, journal_key=self.KEY)
                    self.assertEqual(len(read_results(path)), text.count("\n"))  # without a key: as before


if __name__ == "__main__":
    unittest.main()
