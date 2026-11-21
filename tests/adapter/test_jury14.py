"""Fourteenth blind jury (DECISIONS AC): the adapter takes only the findings.json object; the task text
the adapter shows says so; the internal Env keeps its own forms."""
from __future__ import annotations

import json
import unittest

import judge_adapter
from tests.adapter import helpers
from tests.adapter.helpers import closed, new_env


def tearDownModule():
    helpers.cleanup()


class StrictAdapterSubmitTest(unittest.TestCase):
    def setUp(self):
        self.fx = helpers.fixture("C")
        self.answer = helpers.empty_findings(self.fx)

    def run_submit(self, args: dict):
        env, path = new_env("j14")
        with env:
            env.reset(0, closed(self.fx["dir"]))
            observation, *_ = env.step({"name": "submit", "args": args})
        return observation, helpers.finished(path)[-1]

    def test_the_object_is_accepted(self):
        observation, row = self.run_submit({"findings": self.answer})
        self.assertEqual(observation["result"], {"accepted": True})
        self.assertEqual(row["reward"], 1.0)

    def test_every_other_shape_is_a_structure_error(self):
        forms = {
            "json text": {"findings": json.dumps(self.answer)},
            "bare list": {"findings": []},
            "json text list": {"findings": "[]"},
            "no episode_id": {"findings": {"findings": []}},
            "flat episode_id does not fill it in": {"findings": {"findings": []}, "episode_id": self.answer["episode_id"]},
            "flat episode_id with a bare list": {"findings": [], "episode_id": self.answer["episode_id"]},
            "inner json text": {"findings": {"episode_id": self.answer["episode_id"], "findings": "[]"}},
            "episode_id not a string": {"findings": {"episode_id": 5, "findings": []}},
            "nothing": {},
        }
        for label, args in forms.items():
            observation, row = self.run_submit(args)
            self.assertEqual(observation["error"]["code"], "invalid_findings", label)
            self.assertEqual(row["reward"], -1.0, label)
            self.assertIsNotNone(row["structure_error"], label)
            if args.get("findings") is not None:  # score_episode gives the same verdict
                log = str(helpers.scratch() / "j14.log")
                self.assertEqual(judge_adapter.score_episode(self.fx["dir"], args["findings"], 0, journal_path=log)["return"],
                                 -1 if args["findings"] != self.answer else 1.0, label)

    def test_the_first_observation_does_not_promise_a_bare_list(self):
        env, _ = new_env("j14-task")
        with env:
            task = env.reset(0, closed(self.fx["dir"]))[0]["task"]
        self.assertNotIn("голый список findings принимается", task)
        self.assertIn("неверная структура", task)

    def test_the_internal_env_keeps_the_bare_list(self):
        from auditgym import actions

        payload = actions.submit_payload({"findings": []}, "ep_x")
        self.assertEqual(payload, {"episode_id": "ep_x", "findings": []})


if __name__ == "__main__":
    unittest.main()
