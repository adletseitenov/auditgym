"""Contract clarifications from the second jury review: journal counters, task text, limit clamp,
bare-list submit and the secret flag. Reward behaviour is untouched; these tests pin what is said."""
import json
import pathlib
import tempfile
import unittest

from auditgym.evaluator import Evaluator
from tests.env.helpers import TINY, FakeClock
from tests.env.test_evaluator import CORRECT_TINY
from tools.schema_lite import load_schema, validate


def task_text() -> str:
    return Evaluator(clock=FakeClock()).load(TINY)[0]["task"]["text"]


def journal_rows(path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]


class StepsCountersTest(unittest.TestCase):
    def test_task_text_says_inspection_steps_stop_before_submit(self):
        text = task_text()
        self.assertIn("inspection_steps — число действий до submit; сам submit в него не считается", text)

    def test_journal_has_both_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, clock=FakeClock())
            evaluator.load(TINY)
            for _ in range(3):
                evaluator.step({"action": "list_records", "table": "fx"})
            evaluator.step({"action": "submit", "findings": {"episode_id": "x", "findings": []}})
            (row,) = journal_rows(path)
        self.assertEqual((row["inspection_steps"], row["steps_total"]), (3, 4))
        self.assertEqual(validate(row, load_schema("journal_row")), [], row)

    def test_abandoned_row_counts_steps_the_same_in_both(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, clock=FakeClock())
            evaluator.reset(5, {})
            evaluator.step({"action": "calculator", "expression": "1"})
            evaluator.reset(6, {})
            row = journal_rows(path)[0]
        self.assertEqual((row["event"], row["inspection_steps"], row["steps_total"]), ("abandoned", 1, 1))
        self.assertEqual(validate(row, load_schema("journal_row")), [], row)

    def test_truncated_row_has_the_steps_really_taken(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, clock=FakeClock())
            evaluator.load(TINY)
            for _ in range(100):
                evaluator.step({"action": "list_records", "table": "fx"})
            (row,) = journal_rows(path)
        self.assertEqual((row["truncated"], row["inspection_steps"], row["steps_total"]), (True, 100, 100))

    def test_signature_covers_steps_total(self):
        from auditgym.journal import verify_lines

        key = b"k" * 32
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, clock=FakeClock(), journal_key=key, secret="")
            evaluator.load(TINY)
            evaluator.step({"action": "submit", "findings": {"episode_id": "x", "findings": []}})
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(verify_lines(lines, key)[0], "ok")
        row = json.loads(lines[0])
        self.assertEqual(row["steps_total"], 1)
        row["steps_total"] = 0
        self.assertEqual(verify_lines([json.dumps(row, sort_keys=True)], key)[0], "bad")


class LimitClampTest(unittest.TestCase):
    def page(self, **arguments):
        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset(5, {})
        observation, *_ = evaluator.step({"action": "list_records", "table": "ledger", **arguments})
        return observation

    def test_task_text_states_the_range_and_the_clamp(self):
        text = task_text()
        self.assertIn("limit от 1 до 50, больше — обрезается до 50", text)
        self.assertIn("limit_clamped", text)

    def test_limit_above_50_is_cut_and_flagged(self):
        for limit in (51, 1000, "999", 51.0):
            observation = self.page(limit=limit)
            self.assertEqual(len(observation["records"]), 50, limit)
            self.assertIs(observation["limit_clamped"], True, limit)
            self.assertEqual(validate(observation, load_schema("observation-records")), [], limit)

    def test_no_flag_without_a_clamp(self):
        for arguments in ({}, {"limit": 50}, {"limit": 1}, {"limit": None}):
            observation = self.page(**arguments)
            self.assertNotIn("limit_clamped", observation, arguments)
            self.assertEqual(validate(observation, load_schema("observation-records")), [], arguments)

    def test_the_clamp_does_not_change_the_page(self):
        self.assertEqual(self.page(limit=500)["records"], self.page(limit=50)["records"])


class SubmitFormTest(unittest.TestCase):
    def test_task_text_states_both_submit_forms(self):
        text = task_text()
        self.assertIn("объект {episode_id, findings}", text)
        self.assertIn("голый список findings принимается как findings текущего эпизода", text)

    def test_what_the_text_says_is_what_happens(self):
        results = []
        for findings in (CORRECT_TINY, CORRECT_TINY["findings"]):
            evaluator = Evaluator(clock=FakeClock())
            evaluator.load(TINY)
            _, reward, terminated, _, info = evaluator.step({"action": "submit", "findings": findings})
            self.assertTrue(terminated)
            results.append((reward, info["f1"]))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][1], 1.0)


if __name__ == "__main__":
    unittest.main()
