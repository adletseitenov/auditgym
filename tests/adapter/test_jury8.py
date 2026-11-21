"""Eighth blind jury (DECISIONS W): a rewritten episode_dir, colon FX keys, non-JSON arguments, the score log."""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import shutil
import stat
import tempfile
import unittest
from unittest import mock

import judge_adapter
from auditgym import actions
from auditgym.storage import load_episode
from tests.adapter import helpers
from tests.adapter.helpers import closed, new_env, submit_action

FX_SEED, FX_CONFIG = 93011, {"errors": {"fx_error": 1}, "currencies": ["USD"]}


def tearDownModule():
    helpers.cleanup()


def oracle_of(directory) -> dict:
    episode = load_episode(directory)
    return helpers.oracle_findings({"episode": episode})


class RewrittenEpisodeDirTest(unittest.TestCase):
    def test_reset_serves_what_is_on_disk_now(self):
        base = helpers.scratch() / "rewrite"
        base.mkdir()
        target = base / "episode"
        source_a, source_b = base / "a", base / "b"
        helpers.generate_quietly(81001, {"errors": {"duplicate_invoice": 1}}, str(source_a))
        helpers.generate_quietly(81002, {"errors": {"period_cutoff": 1}}, str(source_b))
        id_a = load_episode(source_a)["public"]["manifest"]["episode_id"]
        id_b = load_episode(source_b)["public"]["manifest"]["episode_id"]
        self.assertNotEqual(id_a, id_b)
        env, journal = new_env("rewrite")
        with env:
            shutil.copytree(source_a, target)
            first, _ = env.reset(0, closed(str(target)))
            self.assertEqual(first["manifest"]["episode_id"], id_a)
            shutil.rmtree(target)
            shutil.copytree(source_b, target)  # the same path, another episode
            second, _ = env.reset(0, closed(str(target)))
            self.assertEqual(second["manifest"]["episode_id"], id_b)
            observation, *_ = env.step(helpers.list_action())
            ids = {row["row_id"] for row in observation["result"]["records"]}
            self.assertTrue(ids <= {r["row_id"] for r in load_episode(source_b)["public"]["ledger"]})
            answer = oracle_of(target)
            env.step(submit_action(answer))
            row = env.last_result()
            self.assertEqual((row["episode_id"], row["tp"], row["fp"], row["fn"]), (id_b, len(answer["findings"]), 0, 0))
            # and back to A, and an unchanged directory is still served without a restart
            shutil.rmtree(target)
            shutil.copytree(source_a, target)
            third, _ = env.reset(0, closed(str(target)))
            self.assertEqual(third["manifest"]["episode_id"], id_a)
            process = env._env._process
            again, _ = env.reset(0, closed(str(target)))
            self.assertIs(env._env._process, process)
            self.assertEqual(again["manifest"]["episode_id"], id_a)


class ColonFxKeyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = str(helpers.scratch() / "fx")
        helpers.generate_quietly(FX_SEED, FX_CONFIG, cls.dir)
        cls.answer = oracle_of(cls.dir)
        cls.colon = json.loads(json.dumps(cls.answer))
        cls.changed = 0
        for finding in cls.colon["findings"]:
            for field in ("evidence_ids", "affected_ids"):
                new = [i.replace("|", ":") for i in finding[field]]
                cls.changed += new != finding[field]
                finding[field] = new

    def test_the_oracle_uses_fx_keys(self):
        self.assertGreater(self.changed, 0)
        self.assertIn(":", json.dumps(self.colon))
        self.assertNotIn("|", json.dumps(self.colon))

    def test_score_episode_reads_colon_keys(self):
        with tempfile.TemporaryDirectory() as scratch:
            plain = judge_adapter.score_episode(self.dir, self.answer, 5, journal_path=scratch + "/log")
            result = judge_adapter.score_episode(self.dir, self.colon, 5, journal_path=scratch + "/log")
        self.assertEqual(result, plain)
        self.assertEqual((result["fp"], result["fn"]), (0, 0))
        self.assertGreater(result["tp"], 0)

    def test_step_reads_colon_keys_in_every_submit_form(self):
        forms = {"message": self.colon, "bare list": self.colon["findings"],
                 "json string": json.dumps(self.colon)}
        for name, payload in forms.items():
            env, _ = new_env("fx")
            with env:
                env.reset(0, closed(self.dir))
                observation, *_ = env.step(submit_action(payload))
                self.assertEqual(observation["result"], {"accepted": True}, name)
                row = env.last_result()
                self.assertEqual((row["tp"], row["fp"], row["fn"]), (len(self.answer["findings"]), 0, 0), name)

    def test_only_exact_fx_keys_are_translated_and_other_ids_stay_exact(self):
        fx_keys = [i for f in self.answer["findings"] for i in f["evidence_ids"] if "|" in i]
        self.assertTrue(fx_keys)
        message = json.loads(json.dumps(self.colon))
        message["findings"][0]["evidence_ids"].append(fx_keys[0].replace("|", ":") + " ")  # not exact
        message["findings"][0]["evidence_ids"].append(fx_keys[0].lower().replace("|", ":"))
        out = judge_adapter._translate_fx_message(message)
        added = out["findings"][0]["evidence_ids"][-2:]
        self.assertEqual(added, message["findings"][0]["evidence_ids"][-2:])
        self.assertEqual(judge_adapter._fx_text("L-000001"), "L-000001")
        self.assertEqual(judge_adapter._fx_text("2026-01-02:USD:KZT"), "2026-01-02|USD|KZT")
        self.assertEqual(judge_adapter._fx_text("2026-01-02:USD"), "2026-01-02:USD")
        self.assertEqual(judge_adapter._fx_text(5), 5)

    def test_a_nonexistent_colon_key_is_still_a_false_positive(self):
        env, _ = new_env("fx2")
        message = json.loads(json.dumps(self.colon))
        message["findings"][0]["evidence_ids"].append("1999-01-01:USD:KZT")
        with env:
            env.reset(0, closed(self.dir))
            env.step(submit_action(message))
            row = env.last_result()
        self.assertEqual(row["structure_error"], None)
        self.assertEqual((row["tp"], row["fp"]), (0, len(self.answer["findings"])))  # unknown evidence id: FP


class NonJsonArgumentsTest(unittest.TestCase):
    def test_known_action_with_non_json_args_is_bad_arguments_and_a_spent_step(self):
        fx = helpers.fixture("C")
        hostile = [{"table": {1, 2}}, {"table": float("nan")}, {"expression": object()}]
        deep: object = "x"
        for _ in range(200):
            deep = [deep]
        hostile.append({"table": deep})
        for name in ("list_records", "get_record", "get_document", "calculator"):
            for args in hostile:
                env, _ = new_env("hostile")
                with env:
                    env.reset(0, closed(fx["dir"]))
                    observation, reward, terminated, truncated, info = env.step({"name": name, "args": args})
                    self.assertEqual(observation["error"]["code"], "bad_arguments", (name, args))
                    self.assertEqual(observation["remaining_steps"], 99)
                    self.assertEqual((reward, terminated, truncated), (0.0, False, False))
                    self.assertTrue(helpers.is_json(observation))

    def test_submit_with_non_json_args_is_an_invalid_structure(self):
        fx = helpers.fixture("C")
        env, journal = new_env("hostile-submit")
        with env:
            env.reset(0, closed(fx["dir"]))
            observation, _, terminated, *_ = env.step({"name": "submit", "args": {"findings": {1, 2}}})
            self.assertTrue(terminated)
            self.assertEqual(observation["error"]["code"], "invalid_findings")
            row = env.last_result()
            self.assertEqual((row["reward"], row["tp"], row["fn"]), (-1.0, None, None))

    def test_unknown_name_stays_unknown_action(self):
        fx = helpers.fixture("C")
        env, _ = new_env("unknown")
        with env:
            env.reset(0, closed(fx["dir"]))
            observation, *_ = env.step({"name": "nope", "args": {"table": {1}}})
            self.assertEqual(observation["error"]["code"], "unknown_action")


class ScoreLogTest(unittest.TestCase):
    def bad_payload(self, fx):
        return {"episode_id": "wrong", "findings": []}

    def test_unwritable_explicit_log_is_loud(self):
        fx = helpers.fixture("C")
        with tempfile.TemporaryDirectory() as scratch:
            blocker = pathlib.Path(scratch) / "file"
            blocker.write_text("x")
            noise = io.StringIO()
            with contextlib.redirect_stderr(noise), self.assertRaises(judge_adapter.AdapterError) as caught:
                judge_adapter.score_episode(fx["dir"], self.bad_payload(fx), 3, journal_path=str(blocker / "log.jsonl"))
            self.assertEqual(caught.exception.code, "score_log_unwritable")
            # the diagnostic is expected here; captured so that it does not look like a failure in check output
            self.assertIn("cannot write the score log", noise.getvalue())

    def test_a_valid_answer_needs_no_log(self):
        fx = helpers.fixture("C")
        result = judge_adapter.score_episode(fx["dir"], helpers.empty_findings(fx), 3,
                                             journal_path="/nonexistent-root-dir/x/log")
        self.assertEqual(result["tp"], 0)

    def test_default_path_is_absolute_and_falls_back_to_a_writable_temp_dir(self):
        fx = helpers.fixture("C")
        with tempfile.TemporaryDirectory() as scratch, tempfile.TemporaryDirectory() as tmp:
            blocker = pathlib.Path(scratch) / "file"
            blocker.write_text("x")
            previous = os.getcwd()
            os.chdir(scratch)
            try:
                with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
                        judge_adapter, "DEFAULT_SCORE_LOG", "file/runs/score_errors.jsonl"), \
                        mock.patch("tempfile.gettempdir", return_value=tmp):
                    os.environ.pop(judge_adapter.SCORE_LOG_ENV, None)
                    path = judge_adapter._score_log_path(None)
                    self.assertTrue(os.path.isabs(path))
                    self.assertEqual(path, os.path.join(tmp, "auditgym", "score_errors.jsonl"))
                    result = judge_adapter.score_episode(fx["dir"], self.bad_payload(fx), 3)
            finally:
                os.chdir(previous)
            self.assertEqual(result["return"], -1)
            row = json.loads(pathlib.Path(path).read_text().splitlines()[-1])
            self.assertEqual(row["event"], "score_error")

    def test_default_path_in_a_writable_cwd_is_runs_score_errors(self):
        with tempfile.TemporaryDirectory() as scratch:
            previous = os.getcwd()
            os.chdir(scratch)
            try:
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop(judge_adapter.SCORE_LOG_ENV, None)
                    path = judge_adapter._score_log_path(None)
            finally:
                os.chdir(previous)
            self.assertEqual(os.path.realpath(path), os.path.realpath(os.path.join(scratch, "runs", "score_errors.jsonl")))


class FindingsSchemaTest(unittest.TestCase):
    def test_first_observation_schema_accepts_negative_amounts(self):
        schema = actions.FINDINGS_SCHEMA["properties"]["findings"]["items"]["properties"]
        self.assertEqual(schema["claimed_amount_minor"], {"type": "integer"})
        published = json.loads((pathlib.Path(__file__).resolve().parents[2] / "schemas" / "findings.schema.json").read_text())
        self.assertEqual(published["properties"]["findings"]["items"]["properties"]["claimed_amount_minor"],
                         {"type": "integer"})


if __name__ == "__main__":
    unittest.main()
