"""Ninth blind jury (DECISIONS X): the reset-style wrapper of generate_episode, a required explanation,
one reading of a findings answer for score_episode and submit, a submit line over the line limit."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import judge_adapter
from auditgym import actions
from auditgym.limits import MAX_LINE_BYTES
from auditgym.secret import SECRET_ENV, SECRET_FILE_ENV
from auditgym.findings import StructureError, parse_findings
from tests.adapter import helpers
from tests.adapter.helpers import closed, new_env, submit_action

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEED, CONFIG = 94011, {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}}
FIELDS = {"type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"}


def tearDownModule():
    helpers.cleanup()


_own: dict = {}


def episode_with_errors() -> dict:
    """An episode that has issues (fixture C is clean): {"dir", "episode"} as helpers.fixture."""
    if not _own:
        from auditgym.storage import load_episode

        directory = helpers.scratch() / "with-errors"
        helpers.generate_quietly(SEED, CONFIG, str(directory))
        _own.update(dir=str(directory), episode=load_episode(directory))
    return _own


def tree(directory) -> dict:
    base = pathlib.Path(directory)
    return {str(p.relative_to(base)): p.read_bytes() for p in sorted(base.rglob("*")) if p.is_file()}


class GenerateWrapperTest(unittest.TestCase):
    def test_reset_style_wrappers_give_the_same_episode_as_the_plain_config(self):
        base = helpers.scratch() / "wrap"
        plain = base / "plain"
        helpers.generate_quietly(SEED, CONFIG, str(plain))
        expected = tree(plain)
        self.assertTrue(expected)
        for name, config in (("a", {"team_config": CONFIG}),
                             ("b", {"mode": "closed", "team_config": CONFIG}),
                             ("c", {"mode": "closed", "episode_dir": "/nonexistent/x", "team_config": CONFIG})):
            target = base / name
            helpers.generate_quietly(SEED, config, str(target))
            self.assertEqual(tree(target), expected, name)

    def test_wrapper_without_team_config_value_is_the_default_config(self):
        base = helpers.scratch() / "wrap-default"
        helpers.generate_quietly(SEED, None, str(base / "none"))
        helpers.generate_quietly(SEED, {"team_config": None}, str(base / "wrapped"))
        self.assertEqual(tree(base / "wrapped"), tree(base / "none"))

    def test_unknown_key_without_the_wrapper_is_still_refused(self):
        from auditgym.config import ConfigError

        with self.assertRaises(ConfigError) as caught:
            helpers.generate_quietly(SEED, {"mode": "closed"}, str(helpers.scratch() / "refused"))
        self.assertEqual(caught.exception.code, "unknown_key")


class GenerateWithoutSecretWarningTest(unittest.TestCase):
    def generate(self, name, **kwargs):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            judge_adapter.generate_episode(SEED, CONFIG, str(helpers.scratch() / name), **kwargs)
        return [w for w in caught if issubclass(w.category, RuntimeWarning)]

    def test_no_secret_warns_and_leaves_the_output_unchanged(self):
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=False):
            for name in (SECRET_ENV, SECRET_FILE_ENV):
                os.environ.pop(name, None)
            seen = self.generate("nosecret-a")
            self.generate("nosecret-b")
        self.assertEqual(len(seen), 1)
        self.assertIn("reproducible from the seed", str(seen[0].message))
        self.assertIn("secret_file", str(seen[0].message))
        self.assertEqual(seen[0].filename, __file__)  # points at the caller
        self.assertEqual(tree(helpers.scratch() / "nosecret-a"), tree(helpers.scratch() / "nosecret-b"))

    def test_secret_file_or_environment_silences_it(self):
        from unittest import mock

        secret = helpers.scratch() / "secret.txt"
        secret.write_text("s" * 40)
        self.assertEqual(self.generate("withsecret", secret_file=str(secret)), [])
        with mock.patch.dict(os.environ, {SECRET_FILE_ENV: str(secret)}):
            self.assertEqual(self.generate("envsecret"), [])


class ExplanationRequiredTest(unittest.TestCase):
    def message(self, **changes):
        fx = episode_with_errors()
        answer = helpers.oracle_findings(fx)
        self.assertTrue(answer["findings"])
        findings = [dict(f) for f in answer["findings"]]
        for finding in findings:
            finding.update(changes)
        return fx, {"episode_id": answer["episode_id"], "findings": findings}

    def test_our_oracle_sends_explanation(self):
        fx = episode_with_errors()
        self.assertTrue(all("explanation" in f for f in helpers.oracle_findings(fx)["findings"]))

    def test_missing_explanation_is_a_structure_error_of_the_whole_submit(self):
        fx, message = self.message()
        for finding in message["findings"]:
            del finding["explanation"]
        result = judge_adapter.score_episode(fx["dir"], message, 3, journal_path=str(helpers.scratch() / "log9.jsonl"))
        self.assertEqual(result["return"], -1)
        self.assertIsNone(result["tp"])
        with self.assertRaises(StructureError) as caught:
            parse_findings(message, message["episode_id"])
        self.assertEqual(caught.exception.code, "missing_field")

    def test_one_finding_without_explanation_spoils_the_message(self):
        fx, message = self.message()
        message["findings"].append({k: v for k, v in message["findings"][0].items() if k != "explanation"})
        with self.assertRaises(StructureError):
            parse_findings(message, message["episode_id"])

    def test_non_string_explanation_is_a_structure_error_empty_and_long_are_fine(self):
        fx, message = self.message(explanation=5)
        with self.assertRaises(StructureError) as caught:
            parse_findings(message, message["episode_id"])
        self.assertEqual(caught.exception.code, "explanation_not_string")
        for text in ("", "я" * 1000):
            _, message = self.message(explanation=text)
            self.assertEqual(len(parse_findings(message, message["episode_id"])), len(message["findings"]))

    def test_step_submit_without_explanation_is_invalid_findings_and_minus_one(self):
        fx, message = self.message()
        for finding in message["findings"]:
            del finding["explanation"]
        env, path = new_env("noexpl")
        with env:
            env.reset(0, closed(fx["dir"]))
            observation, *_ = env.step(submit_action(message))
        self.assertEqual(observation["error"]["code"], "invalid_findings")
        row = helpers.finished(path)[-1]
        self.assertEqual((row["reward"], row["structure_error"]), (-1.0, "missing_field"))

    def test_both_published_schemas_require_explanation(self):
        inner = actions.FINDINGS_SCHEMA["properties"]["findings"]["items"]
        self.assertEqual(set(inner["required"]), FIELDS)
        published = json.loads((ROOT / "schemas" / "findings.schema.json").read_text())
        self.assertEqual(set(published["properties"]["findings"]["items"]["required"]), FIELDS)
        self.assertIn("explanation", json.dumps(actions.build_task({})["answer_rules"], ensure_ascii=False))


class ScoreEpisodeShapesTest(unittest.TestCase):
    """DECISIONS AC: the adapter accepts only the findings.json object; score_episode and submit agree."""

    def test_only_the_object_is_valid_in_score_episode_and_submit(self):
        fx = episode_with_errors()
        answer = helpers.oracle_findings(fx)
        log = str(helpers.scratch() / "log9b.jsonl")
        steps = 4
        good = judge_adapter.score_episode(fx["dir"], answer, steps, journal_path=log)
        self.assertEqual(good["fn"], 0)
        bad = {"list": answer["findings"], "json text": json.dumps(answer["findings"]),
               "json text object": json.dumps(answer), "no episode_id": {"findings": answer["findings"]},
               "inner json text": {"episode_id": answer["episode_id"], "findings": json.dumps(answer["findings"])}}
        for label, payload in bad.items():
            result = judge_adapter.score_episode(fx["dir"], payload, steps, journal_path=log)
            self.assertEqual(result["return"], -1, label)
            self.assertIsNone(result["tp"], label)
        for label, payload in (("object", answer),) + tuple(bad.items()):
            env, path = new_env("shape-" + label.replace(" ", "-"))
            with env:
                env.reset(0, closed(fx["dir"]))
                for _ in range(steps):
                    env.step(helpers.list_action())
                observation, *_ = env.step(submit_action(payload))
            row = helpers.finished(path)[-1]
            if label == "object":
                self.assertEqual(observation["result"], {"accepted": True})
                self.assertEqual((row["tp"], row["fp"], row["fn"], row["reward"]),
                                 (good["tp"], good["fp"], good["fn"], good["return"]))
            else:  # the policy and the journal get the same verdict as score_episode
                self.assertEqual(observation["error"]["code"], "invalid_findings", label)
                self.assertEqual(row["reward"], -1.0, label)
                self.assertIsNotNone(row["structure_error"], label)

    def test_empty_bare_list_and_bad_shapes(self):
        fx = helpers.fixture("C")
        log = str(helpers.scratch() / "log9c.jsonl")
        self.assertEqual(judge_adapter.score_episode(fx["dir"], [], 0, journal_path=log)["return"], -1)
        self.assertEqual(judge_adapter.score_episode(fx["dir"], helpers.empty_findings(fx), 0, journal_path=log)["return"], 1.0)
        self.assertEqual(judge_adapter.score_episode(fx["dir"], {"findings": []}, 0, journal_path=log)["return"], -1)
        self.assertEqual(judge_adapter.score_episode(fx["dir"], 7, 0, journal_path=log)["return"], -1)


class LongIdsWithinCaseLimitsTest(unittest.TestCase):
    """DECISIONS Y: 30 findings x 10 evidence ids of 1 000 characters respect the limits 30 / 10 / 1 000 and
    are false positives, not message_too_large (the cap is 8 MiB, ADAPTER.md item 14)."""

    @staticmethod
    def message(fx) -> dict:
        episode_id = fx["episode"]["public"]["manifest"]["episode_id"]
        findings = [{"type": "duplicate_invoice", "affected_ids": ["no-such-doc-" + "я" * 1000],
                     "claimed_amount_minor": 1, "currency": "KZT",
                     "evidence_ids": [f"{n:02d}-{i}-" + "x" * 990 for i in range(10)], "explanation": "я" * 1000}
                    for n in range(30)]
        return {"episode_id": episode_id, "findings": findings}

    def test_the_message_is_larger_than_the_old_cap_and_smaller_than_the_new_one(self):
        from auditgym.findings import message_size
        from auditgym.limits import MAX_MESSAGE_BYTES

        fx = helpers.fixture("C")
        self.assertGreater(message_size(self.message(fx)), 256 * 1024)
        self.assertLess(message_size(self.message(fx)), MAX_MESSAGE_BYTES)

    def test_score_episode_counts_them_as_false_positives(self):
        fx = episode_with_errors()
        message = self.message(fx)
        message["episode_id"] = fx["episode"]["public"]["manifest"]["episode_id"]
        result = judge_adapter.score_episode(fx["dir"], message, 3, journal_path=str(helpers.scratch() / "long.jsonl"))
        self.assertEqual((result["tp"], result["fp"]), (0, 30))  # counts, not None: no structure error

    def test_step_submit_agrees_with_score_episode(self):
        fx = episode_with_errors()
        message = self.message(fx)
        message["episode_id"] = fx["episode"]["public"]["manifest"]["episode_id"]
        env, path = new_env("long-ids")
        with env:
            env.reset(0, closed(fx["dir"]))
            observation, *_ = env.step(submit_action(message))
        self.assertEqual(observation["result"], {"accepted": True})
        row = helpers.finished(path)[-1]
        self.assertEqual((row["tp"], row["fp"], row.get("structure_error")), (0, 30, None))


class AdapterTaskTextTest(unittest.TestCase):
    """DECISIONS Y: the adapter result of list_records is {records, next_cursor}; its task text does not
    promise limit_clamped."""

    def test_task_does_not_promise_limit_clamped_and_the_result_has_no_such_key(self):
        fx = helpers.fixture("C")
        env, _ = new_env("clamp")
        with env:
            first, _ = env.reset(0, closed(fx["dir"]))
            observation, *_ = env.step(helpers.list_action(limit=500))
        self.assertNotIn("limit_clamped", first["task"])
        self.assertIn("обрезается до 50", first["task"])
        self.assertEqual(set(observation["result"]), {"records", "next_cursor"})
        self.assertEqual(len(observation["result"]["records"]), 50)


class OversizedSubmitTest(unittest.TestCase):
    def test_a_submit_line_over_the_line_limit_is_message_too_large(self):
        fx = helpers.fixture("C")
        journal = pathlib.Path(tempfile.mkdtemp(dir=helpers.scratch())) / "results.jsonl"
        pad = "x" * (MAX_LINE_BYTES + 10)
        lines = [json.dumps({"op": "load", "directory": fx["dir"]}),
                 json.dumps({"op": "step", "action": {"action": "submit", "findings": [], "pad": pad}})]
        data = b"".join(line.encode() + b"\n" for line in lines)
        result = subprocess.run([sys.executable, "-m", "auditgym", "serve", "--mode", "eval", "--allow-load",
                                 "--results", str(journal)], input=data, capture_output=True,
                                env=dict(os.environ, PYTHONPATH=str(ROOT)), cwd=str(ROOT), timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-200:])
        row = helpers.finished(journal)[-1]
        self.assertEqual((row["reward"], row["structure_error"]), (-1.0, "message_too_large"))

    def test_an_unreadable_submit_that_is_small_keeps_its_old_code(self):
        from auditgym.evaluator import Evaluator

        evaluator = Evaluator(mode="train")
        evaluator.reset(5, {})
        _, reward, terminated, _, _ = evaluator.reject(submit=True)
        self.assertEqual((reward, terminated), (-1.0, True))


if __name__ == "__main__":
    unittest.main()
