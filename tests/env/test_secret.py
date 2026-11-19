"""The evaluator mixes AUDITGYM_SECRET into generation: same seed, different secret, different episode."""
import os
import unittest
from unittest import mock

from auditgym.evaluator import SECRET_ENV, Evaluator
from auditgym.generate import generate


def first_page(evaluator):
    observation, *_ = evaluator.step({"action": "list_records", "table": "ledger"})
    return observation["records"]


class SecretTest(unittest.TestCase):
    def test_without_secret_the_episode_is_the_public_one(self):
        with mock.patch.dict(os.environ, {SECRET_ENV: ""}):
            evaluator = Evaluator()
            observation, _ = evaluator.reset(7, {})
        self.assertEqual(observation["manifest"]["episode_id"], generate(7, {})["public"]["manifest"]["episode_id"])

    def test_secret_changes_the_episode_and_stays_reproducible(self):
        with mock.patch.dict(os.environ, {SECRET_ENV: "evaluator-only"}):
            first, second = Evaluator(), Evaluator()
            first.reset(7, {})
            second.reset(7, {})
            self.assertEqual(first_page(first), first_page(second))
            with_secret = first._episode["public"]["manifest"]["episode_id"]
        self.assertNotEqual(with_secret, generate(7, {})["public"]["manifest"]["episode_id"])


if __name__ == "__main__":
    unittest.main()
