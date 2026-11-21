"""Twelfth blind jury (DECISIONS AA): the limits of CHECKS are checked on the raw submitted lists, before any
renaming of FX keys; the policy-visible verdict equals the journal; no invoice_id alias; closed needs a secret."""
from __future__ import annotations

import copy
import json
import unittest

import judge_adapter
from tests.adapter import helpers
from tests.adapter.helpers import closed, new_env, submit_action


def tearDownModule():
    helpers.cleanup()


def _colon(value: str) -> str:
    return value.replace("|", ":") if value.count("|") == 2 and value[:4].isdigit() else value


class RawLimitsBeforeTranslationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = helpers.fixture("E")
        cls.public = cls.fx["episode"]["public"]

    def fx_finding(self):
        """A true fx_error finding of an episode and the id of its fx key."""
        from auditgym.storage import load_episode
        import pathlib

        if not hasattr(type(self), "_ep"):
            path = pathlib.Path(helpers.scratch()) / "fxep"
            helpers.generate_quietly(94112, {"errors": {"fx_error": 1}}, str(path))
            type(self)._ep = {"dir": str(path), "episode": load_episode(path)}
        ep = type(self)._ep
        message = helpers.oracle_findings(ep)
        finding = message["findings"][0]
        key = next(i for i in finding["evidence_ids"] if i.count("|") == 2)
        return ep, message, finding, key

    def padded(self, finding, size, spelling):
        """evidence_ids with the fx key in `spelling` (list of strings) padded with existing ledger rows to `size`."""
        ep = type(self)._ep
        base = [i for i in finding["evidence_ids"] if i.count("|") != 2]
        pad = [r["row_id"] for r in ep["episode"]["public"]["ledger"] if r["row_id"] not in base]
        ids = base + spelling
        ids += [p for p in pad if p not in ids][: size - len(ids)]
        self.assertEqual(len(ids), size)
        self.assertEqual(len(set(ids)), size)
        return ids

    def counts(self, ep, message):
        env, _ = new_env("raw")
        with env:
            env.reset(0, closed(ep["dir"]))
            observation, *_ = env.step(submit_action(message))
            row = env.last_result()
        score = judge_adapter.score_episode(ep["dir"], message, 0, journal_path=str(helpers.scratch() / "raw-log.jsonl"))
        self.assertEqual(score["return"], row["reward"])
        return observation, row, score

    def test_eleven_raw_ids_two_spellings_of_one_fx_key_is_a_structure_error(self):
        ep, message, finding, key = self.fx_finding()
        message = copy.deepcopy(message)
        message["findings"][0]["evidence_ids"] = self.padded(finding, 11, [key, _colon(key)])
        observation, row, score = self.counts(ep, message)
        self.assertEqual(row["reward"], -1.0)
        self.assertIsNotNone(row["structure_error"])
        self.assertEqual(row["structure_error"], "too_many_evidence_ids")
        self.assertEqual(score["return"], -1)
        self.assertEqual(observation["error"]["code"], "invalid_findings")

    def test_eleven_raw_ids_with_a_colon_key_and_the_same_key_twice_is_still_a_structure_error(self):
        ep, message, finding, key = self.fx_finding()
        message = copy.deepcopy(message)
        message["findings"][0]["evidence_ids"] = self.padded(finding, 11, [key, _colon(key)])
        for form in (message, message["findings"], json.dumps(message)):
            _, _, score = self.counts(ep, form)
            self.assertEqual(score["return"], -1)

    def test_ten_raw_ids_with_a_colon_fx_key_is_a_true_positive(self):
        ep, message, finding, key = self.fx_finding()
        message = copy.deepcopy(message)
        message["findings"][0]["evidence_ids"] = self.padded(finding, 10, [_colon(key)])
        observation, row, score = self.counts(ep, message)
        self.assertIsNone(row["structure_error"])
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 0, 0))
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 0, 0))
        self.assertEqual(observation["result"], {"accepted": True})

    def test_translation_keeps_list_lengths(self):
        message = {"findings": [{"evidence_ids": ["2026-01-02:USD:KZT", "2026-01-02|USD|KZT", "x"],
                                 "affected_ids": ["2026-01-02:USD:KZT", "2026-01-02|USD|KZT"]}]}
        translated = judge_adapter._translate_fx_message(message)["findings"][0]
        self.assertEqual(translated["evidence_ids"], ["2026-01-02|USD|KZT", "2026-01-02|USD|KZT", "x"])
        self.assertEqual(len(translated["affected_ids"]), 2)


class PolicyVerdictEqualsJournalTest(unittest.TestCase):
    def test_accepted_iff_no_structure_error(self):
        fx = helpers.fixture("E")
        good = helpers.oracle_findings(fx)["findings"]
        episode_id = fx["episode"]["public"]["manifest"]["episode_id"]

        def one(**changes):
            finding = dict(good[0], **changes)
            return {"episode_id": episode_id, "findings": [finding]}

        key = "2026-01-02:USD:KZT"
        many = [f"id{n}" for n in range(11)]
        payloads = [
            {"episode_id": episode_id, "findings": good},
            {"episode_id": episode_id, "findings": []},
            good,
            json.dumps({"episode_id": episode_id, "findings": good}),
            one(evidence_ids=many),
            one(evidence_ids=many[:10]),
            one(evidence_ids=many[:9] + [key, key.replace(":", "|")]),
            one(evidence_ids=many[:8] + [key, key.replace(":", "|")]),
            one(evidence_ids=[key] * 11),
            one(evidence_ids=["a", "a", "a"]),
            one(explanation="x" * 1001),
            one(explanation="я" * 1000),
            one(explanation=5),
            one(type=3),
            one(claimed_amount_minor=1.0),
            one(claimed_amount_minor=True),
            one(affected_ids=[key, 5]),
            one(evidence_ids=[5]),
            {"episode_id": episode_id, "findings": [good[0]] * 31},
            {"episode_id": episode_id, "findings": [good[0]] * 30},
            {"episode_id": "other", "findings": good},
            {"findings": good},
            {"episode_id": episode_id, "findings": "nope"},
            {"episode_id": episode_id, "findings": [1]},
            "{bad json",
            5,
        ]
        env, journal = new_env("verdict")
        with env:
            for payload in payloads:
                env.reset(0, closed(fx["dir"]))
                observation, *_ = env.step({"name": "submit", "args": {"findings": payload}})
                row = env.last_result()
                accepted = observation["error"] is None and observation["result"] == {"accepted": True}
                self.assertEqual(accepted, row["structure_error"] is None, repr(payload)[:120])
            # malformed envelopes: arguments that are not an object, no findings, two different answers
            for action in ({"name": "submit", "args": []}, {"name": "submit", "args": {}},
                           {"name": "submit", "args": {"findings": good, "findings.json": []}},
                           {"name": "submit", "args": {"findings": {1, 2}}}):
                env.reset(0, closed(fx["dir"]))
                observation, *_ = env.step(action)
                row = env.last_result()
                accepted = observation["error"] is None
                self.assertEqual(accepted, row["structure_error"] is None, repr(action)[:120])


class ClosedNeedsSecretTest(unittest.TestCase):
    def test_closed_without_directory_and_secret_is_refused(self):
        import os
        from unittest import mock

        env = judge_adapter.make_env(journal_path=helpers.scratch() / "nosecret.jsonl")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(judge_adapter.ADAPTER_SECRET_FILE_ENV, None)
            env = judge_adapter.make_env(journal_path=helpers.scratch() / "nosecret.jsonl")
            with self.assertRaises(judge_adapter.AdapterError) as caught:
                env.reset(1, {"mode": "closed", "team_config": None})
            self.assertEqual(str(caught.exception), "secret_required")
            self.assertIsNone(env._env)  # nothing was started

    def test_closed_with_a_secret_file_generates(self):
        import pathlib

        secret = pathlib.Path(helpers.scratch()) / "secret.txt"
        secret.write_text("a-test-secret-value-0123456789abcdef\n", encoding="utf-8")
        secret.chmod(0o600)
        env = judge_adapter.make_env(journal_path=helpers.scratch() / "withsecret.jsonl", secret_file=secret)
        with env:
            observation, _ = env.reset(1, {"mode": "closed"})
            self.assertIn("task", observation)

    def test_train_without_directory_still_generates_by_seed(self):
        env, _ = new_env("train-seed")
        with env:
            observation, _ = env.reset(1, {"mode": "train"})
            self.assertIn("task", observation)


if __name__ == "__main__":
    unittest.main()
