"""Audit A3: the eval journal is an HMAC-SHA256 chain; verify-journal names the first bad line."""
import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from auditgym import Env
from auditgym.__main__ import main as cli
from auditgym.evaluator import Evaluator
from auditgym.journal import GENESIS, verify_lines
from auditgym.secret import JOURNAL_KEY_ENV, JOURNAL_KEY_FILE_ENV, SECRET_ENV, SECRET_FILE_ENV
from tools.schema_lite import load_schema, validate

KEY = b"journal-test-key"
CLEAN_ENV = {SECRET_ENV: "", SECRET_FILE_ENV: "", JOURNAL_KEY_ENV: "", JOURNAL_KEY_FILE_ENV: ""}


def play(evaluator, seed, steps=1):
    observation, _ = evaluator.reset(seed, {})
    for _ in range(steps):
        evaluator.step({"action": "calculator", "expression": "1"})
    evaluator.step({"action": "submit", "findings": {
        "episode_id": observation["manifest"]["episode_id"], "findings": []}})


def verify_cli(path, environ, *extra):
    out = io.StringIO()
    with mock.patch.dict(os.environ, {**CLEAN_ENV, **environ}), contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(out):
        code = cli(["verify-journal", str(path), *extra])
    return code, out.getvalue()


class SignedJournalTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.path = pathlib.Path(scratch.name) / "results.jsonl"
        # Two sessions append to one file: the second continues the chain of the first.
        first = Evaluator(mode="eval", results_path=self.path, journal_key=KEY, secret="")
        play(first, 1)
        first.reset(2, {})
        first.step({"action": "calculator", "expression": "1"})
        first.reset(3, {})  # abandons episode 2 after one step: an 'abandoned' line is signed too
        first.step({"action": "submit", "findings": {"episode_id": "x", "findings": []}})
        second = Evaluator(mode="eval", results_path=self.path, journal_key=KEY, secret="")
        play(second, 4, steps=2)
        self.lines = self.path.read_text(encoding="utf-8").splitlines()

    def check(self, lines):
        return verify_lines(lines, KEY)

    def test_an_intact_journal_verifies(self):
        self.assertEqual(len(self.lines), 4)
        rows = [json.loads(line) for line in self.lines]
        self.assertEqual([row["seq"] for row in rows], [0, 1, 2, 3])
        self.assertEqual(rows[0]["prev"], GENESIS)
        self.assertEqual([row["event"] for row in rows], ["finished", "abandoned", "finished", "finished"])
        for row in rows:
            self.assertEqual(validate(row, load_schema("journal_row")), [], row)
        self.assertEqual(self.check(self.lines)[0], "ok")
        code, text = verify_cli(self.path, {JOURNAL_KEY_ENV: KEY.decode()})
        self.assertEqual(code, 0, text)
        self.assertIn("journal OK", text)

    def test_a_changed_line_is_caught(self):
        row = json.loads(self.lines[2])
        row["reward"] = 1.0
        lines = self.lines[:2] + [json.dumps(row, sort_keys=True)] + self.lines[3:]
        self.assertEqual(self.check(lines)[:2], ("bad", 3))
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        code, text = verify_cli(self.path, {JOURNAL_KEY_ENV: KEY.decode()})
        self.assertEqual(code, 1)
        self.assertIn("line 3", text)

    def test_an_appended_line_is_caught(self):
        forged = json.loads(self.lines[-1])
        forged.update(seq=4, prev=forged["mac"], mac="0" * 64, reward=1.0)
        self.assertEqual(self.check(self.lines + [json.dumps(forged)])[:2], ("bad", 5))
        unsigned = {key: value for key, value in forged.items() if key not in ("seq", "prev", "mac")}
        self.assertEqual(self.check(self.lines + [json.dumps(unsigned)])[:2], ("bad", 5))

    def test_a_deleted_or_moved_line_is_caught(self):
        self.assertEqual(self.check(self.lines[1:])[:2], ("bad", 1))
        self.assertEqual(self.check(self.lines[:1] + self.lines[2:])[:2], ("bad", 2))
        swapped = [self.lines[0], self.lines[2], self.lines[1], self.lines[3]]
        self.assertEqual(self.check(swapped)[:2], ("bad", 2))

    def test_a_wrong_key_fails_and_no_key_cannot_verify(self):
        self.assertEqual(verify_lines(self.lines, b"other-key")[:2], ("bad", 1))
        code, text = verify_cli(self.path, {})
        self.assertEqual(code, 2)
        self.assertIn("no journal key", text)

    def test_cutting_the_tail_is_the_documented_blind_spot(self):
        self.assertEqual(self.check(self.lines[:3])[0], "ok")

    def test_expect_episodes_catches_a_cut_tail(self):
        env = {JOURNAL_KEY_ENV: KEY.decode()}
        self.assertEqual(verify_cli(self.path, env, "--expect-episodes", "4")[0], 0)
        self.path.write_text("\n".join(self.lines[:3]) + "\n", encoding="utf-8")
        self.assertEqual(verify_cli(self.path, env)[0], 0)  # still a valid chain
        code, text = verify_cli(self.path, env, "--expect-episodes", "4")
        self.assertEqual(code, 1)
        self.assertIn("expected at least 4", text)

    def test_expect_episodes_counts_abandoned_rows_too(self):
        events = [json.loads(line)["event"] for line in self.lines]
        self.assertEqual(sorted(events), ["abandoned", "finished", "finished", "finished"])
        self.assertEqual(verify_cli(self.path, {JOURNAL_KEY_ENV: KEY.decode()}, "--expect-episodes", "5")[0], 1)


class ShapedRewardStaysOutOfEvalTest(unittest.TestCase):
    def test_eval_with_shaped_reward_and_a_signed_journal_has_no_shaped_key(self):
        from auditgym import shaped

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, shaped_reward=True, journal_key=KEY, secret="")
            observation, _ = evaluator.reset(3, {})
            result = evaluator.step({"action": "submit", "findings": {
                "episode_id": observation["manifest"]["episode_id"], "findings": []}})
            text = path.read_text(encoding="utf-8")
        self.assertEqual(result[1:], (0.0, True, False, {}))
        row = json.loads(text)
        self.assertIn("mac", row)
        self.assertNotIn(shaped.INFO_KEY, row)
        self.assertNotIn("shaped", text)
        self.assertEqual(verify_lines(text.splitlines(), KEY)[0], "ok")


class UnsignedJournalTest(unittest.TestCase):
    def test_without_a_key_nothing_is_signed_and_the_command_says_so(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, CLEAN_ENV):
            path = pathlib.Path(directory) / "results.jsonl"
            play(Evaluator(mode="eval", results_path=path), 1)
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse({"seq", "prev", "mac"} & set(row))
            code, text = verify_cli(path, {})
        self.assertEqual(code, 2)
        self.assertIn("not signed", text)


class KeyedVerifierTest(unittest.TestCase):
    """With a key the verifier expects a signed journal: nothing to check is a failure (1), not a skip (2)."""

    def run_on(self, text, environ, *extra):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            path.write_text(text, encoding="utf-8")
            return verify_cli(path, environ, *extra)

    def test_an_empty_journal_fails_with_a_key_and_is_nothing_without_one(self):
        code, text = self.run_on("", {JOURNAL_KEY_ENV: KEY.decode()})
        self.assertEqual(code, 1)
        self.assertIn("FAILED", text)
        self.assertEqual(self.run_on("", {})[0], 2)

    def test_an_unsigned_journal_fails_with_a_key_and_is_nothing_without_one(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, CLEAN_ENV):
            path = pathlib.Path(directory) / "results.jsonl"
            play(Evaluator(mode="eval", results_path=path), 1)
            unsigned = path.read_text(encoding="utf-8")
        code, text = self.run_on(unsigned, {JOURNAL_KEY_ENV: KEY.decode()})
        self.assertEqual(code, 1)
        self.assertIn("unsigned", text)
        self.assertEqual(self.run_on(unsigned, {})[0], 2)

    def test_a_secret_file_counts_as_a_key(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = pathlib.Path(directory) / "secret"
            secret.write_text("some-secret\n", encoding="utf-8")
            self.assertEqual(self.run_on("", {SECRET_FILE_ENV: str(secret)})[0], 1)

    def test_expect_episodes_without_a_key_still_fails_on_too_few_lines(self):
        self.assertEqual(self.run_on("", {}, "--expect-episodes", "1")[0], 1)


class ProcessJournalTest(unittest.TestCase):
    """Through Env: the journal descriptor is handed to the server, which reads its tail to chain."""

    def test_env_sessions_under_a_secret_write_one_verifiable_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            secret = root / "secret"
            secret.write_text("process-journal-secret\n", encoding="utf-8")
            path = root / "results.jsonl"
            with mock.patch.dict(os.environ, CLEAN_ENV):
                for seed in (1, 2):
                    with Env(mode="eval", results_path=path, secret_file=secret) as env:
                        observation, _ = env.reset(seed, {})
                        env.step({"action": "submit", "findings": {
                            "episode_id": observation["manifest"]["episode_id"], "findings": []}})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["seq"] for row in rows], [0, 1])
            self.assertNotIn("process-journal-secret", path.read_text(encoding="utf-8"))
            code, text = verify_cli(path, {SECRET_FILE_ENV: str(secret)})
            self.assertEqual(code, 0, text)
            code, _ = verify_cli(path, {SECRET_ENV: "some-other-secret"})
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
