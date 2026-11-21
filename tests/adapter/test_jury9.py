"""Ninth blind jury (DECISIONS X): the reset-style wrapper of generate_episode, a required explanation,
one reading of a findings answer for score_episode and submit, a submit over 1 MiB."""
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
        judge_adapter.generate_episode(SEED, CONFIG, str(directory))
        _own.update(dir=str(directory), episode=load_episode(directory))
    return _own


def tree(directory) -> dict:
    base = pathlib.Path(directory)
    return {str(p.relative_to(base)): p.read_bytes() for p in sorted(base.rglob("*")) if p.is_file()}


class GenerateWrapperTest(unittest.TestCase):
    def test_reset_style_wrappers_give_the_same_episode_as_the_plain_config(self):
        base = helpers.scratch() / "wrap"
        plain = base / "plain"
        judge_adapter.generate_episode(SEED, CONFIG, str(plain))
        expected = tree(plain)
        self.assertTrue(expected)
        for name, config in (("a", {"team_config": CONFIG}),
                             ("b", {"mode": "closed", "team_config": CONFIG}),
                             ("c", {"mode": "closed", "episode_dir": "/nonexistent/x", "team_config": CONFIG})):
            target = base / name
            judge_adapter.generate_episode(SEED, config, str(target))
            self.assertEqual(tree(target), expected, name)

    def test_wrapper_without_team_config_value_is_the_default_config(self):
        base = helpers.scratch() / "wrap-default"
        judge_adapter.generate_episode(SEED, None, str(base / "none"))
        judge_adapter.generate_episode(SEED, {"team_config": None}, str(base / "wrapped"))
        self.assertEqual(tree(base / "wrapped"), tree(base / "none"))

    def test_unknown_key_without_the_wrapper_is_still_refused(self):
        from auditgym.config import ConfigError

        with self.assertRaises(ConfigError) as caught:
            judge_adapter.generate_episode(SEED, {"mode": "closed"}, str(helpers.scratch() / "refused"))
        self.assertEqual(caught.exception.code, "unknown_key")


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
    def test_bare_list_scores_like_the_adapter_submit(self):
        fx = episode_with_errors()
        answer = helpers.oracle_findings(fx)
        log = str(helpers.scratch() / "log9b.jsonl")
        steps = 4
        by_score = {
            "object": judge_adapter.score_episode(fx["dir"], answer, steps, journal_path=log),
            "list": judge_adapter.score_episode(fx["dir"], answer["findings"], steps, journal_path=log),
            "json text": judge_adapter.score_episode(fx["dir"], json.dumps(answer["findings"]), steps, journal_path=log),
        }
        self.assertEqual(by_score["object"]["fn"], 0)
        self.assertEqual(by_score["list"], by_score["object"])
        self.assertEqual(by_score["json text"], by_score["object"])
        for label, payload in (("object", answer), ("list", answer["findings"])):
            env, path = new_env("shape-" + label)
            with env:
                env.reset(0, closed(fx["dir"]))
                for _ in range(steps):
                    env.step(helpers.list_action())
                observation, *_ = env.step(submit_action(payload))
            self.assertEqual(observation["result"], {"accepted": True}, label)
            row = helpers.finished(path)[-1]
            expected = by_score["object"]
            self.assertEqual((row["tp"], row["fp"], row["fn"], row["reward"]),
                             (expected["tp"], expected["fp"], expected["fn"], expected["return"]), label)

    def test_empty_bare_list_and_bad_shapes(self):
        fx = helpers.fixture("C")
        log = str(helpers.scratch() / "log9c.jsonl")
        empty = judge_adapter.score_episode(fx["dir"], [], 0, journal_path=log)
        self.assertEqual((empty["tp"], empty["fp"]), (0, 0))
        self.assertEqual(judge_adapter.score_episode(fx["dir"], {"findings": []}, 0, journal_path=log)["return"], -1)
        self.assertEqual(judge_adapter.score_episode(fx["dir"], 7, 0, journal_path=log)["return"], -1)


class OversizedSubmitTest(unittest.TestCase):
    def test_a_submit_line_over_one_mebibyte_is_message_too_large(self):
        fx = helpers.fixture("C")
        journal = pathlib.Path(tempfile.mkdtemp(dir=helpers.scratch())) / "results.jsonl"
        pad = "x" * ((1 << 20) + 10)
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
