"""DECISIONS Y: the docs say `explanation` is required, the answer-format section of docs/ADAPTER.md is the
output of tools/answer_examples.py, and every example in it is a true positive in its own episode."""
from __future__ import annotations

import json
import pathlib
import re
import tempfile
import unittest
import warnings

import judge_adapter
from tools import answer_examples

ROOT = pathlib.Path(__file__).resolve().parents[1]
HISTORY = re.compile(r"ОТМЕНЕН|отменён|отменен|оставался|раньше|было необязательн|у нас необязателен.*\(ОТМЕН", re.I)


class ExplanationRequiredInDocsTest(unittest.TestCase):
    def test_no_current_doc_calls_explanation_optional(self):
        stale = []
        files = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
        for path in files:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for match in re.finditer(r"explanation", line):
                    near = line[max(0, match.start() - 80):match.end() + 80]
                    if re.search(r"необязател|optional", near, re.I) and not HISTORY.search(line):
                        stale.append(f"{path.name}:{number}")
        self.assertEqual(stale, [])


class AnswerFormatSectionTest(unittest.TestCase):
    def test_the_section_is_what_the_script_prints(self):
        self.assertEqual(answer_examples.main(["--check"]), 0)

    def test_every_example_scores_true_positive_in_its_episode(self):
        doc = (ROOT / "docs" / "ADAPTER.md").read_text(encoding="utf-8")
        block = doc[doc.index(answer_examples.BEGIN):doc.index(answer_examples.END)]
        examples = [json.loads(text) for text in re.findall(r"```json\n(.*?)\n```", block, re.S)]
        self.assertEqual([e["type"] for e in examples],
                         ["duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff"])
        self.assertTrue(all(set(e) == {"type", "affected_ids", "claimed_amount_minor", "currency",
                                       "evidence_ids", "explanation"} for e in examples))
        with tempfile.TemporaryDirectory() as directory:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="generate_episode: no secret", category=RuntimeWarning)
                judge_adapter.generate_episode(answer_examples.SEED, answer_examples.CONFIG, directory)
            from auditgym.storage import load_episode

            episode_id = load_episode(directory)["public"]["manifest"]["episode_id"]
            self.assertIn(episode_id, block)
            result = judge_adapter.score_episode(directory, {"episode_id": episode_id, "findings": examples}, 3,
                                                 journal_path=str(pathlib.Path(directory) / "log.jsonl"))
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (4, 0, 0))


if __name__ == "__main__":
    unittest.main()
