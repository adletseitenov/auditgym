"""Fifteenth blind jury (DECISIONS AD): a malformed submit step leaves the same structure_error code in
the evaluator journal as score_episode logs for the same answer."""
from __future__ import annotations

import json
import unittest

import judge_adapter
from tests.adapter import helpers
from tests.adapter.helpers import closed, new_env


def tearDownModule():
    helpers.cleanup()


class JournalCodeTest(unittest.TestCase):
    def test_journal_code_equals_score_episode_code(self):
        fx = helpers.fixture("C")
        good = helpers.empty_findings(fx)
        cases = {
            "not_object": [[], json.dumps(good), "x"],
            "findings_not_list": [{"episode_id": good["episode_id"], "findings": "[]"}],
            "wrong_episode_id": [{"episode_id": 5, "findings": []}],
            "missing_field": [{"findings": []}, {}],
        }
        for expected, answers in cases.items():
            for answer in answers:
                env, path = new_env("j15")
                with env:
                    env.reset(0, closed(fx["dir"]))
                    observation, *_ = env.step({"name": "submit", "args": {"findings": answer}})
                row = helpers.finished(path)[-1]
                log = str(helpers.scratch() / "j15.log")
                judge_adapter.score_episode(fx["dir"], answer, 0, journal_path=log)
                with open(log, encoding="utf-8") as handle:
                    logged = json.loads(handle.read().splitlines()[-1])["structure_error"]
                self.assertEqual(observation["error"]["code"], "invalid_findings", answer)
                self.assertEqual(row["reward"], -1.0, answer)
                self.assertEqual(logged, expected, answer)
                self.assertEqual(row["structure_error"], logged, answer)


if __name__ == "__main__":
    unittest.main()
