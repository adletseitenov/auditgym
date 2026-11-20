"""The evaluator mixes AUDITGYM_SECRET into generation: same seed, different secret, different episode.
Audit A1: the secret can come from a file the policy never reads ($AUDITGYM_SECRET_FILE,
--secret-file, Env(secret_file=...)), and an evaluator can run behind Env(server_command=...)."""
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from auditgym import Env
from auditgym.__main__ import main as cli
from auditgym.client import EnvError
from auditgym.evaluator import SECRET_ENV, Evaluator
from auditgym.generate import generate
from auditgym.secret import SECRET_FILE_ENV, SecretError, journal_key, read_secret
from tests.env.helpers import ROOT

FILE_SECRET = "file-only-secret-7f3a"


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



class SecretFileTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = pathlib.Path(scratch.name)
        self.secret_file = self.root / "secret"
        self.secret_file.write_text(FILE_SECRET + "\n", encoding="utf-8")
        self.expected = generate(7, {}, secret=FILE_SECRET)["public"]["manifest"]["episode_id"]

    def test_file_wins_over_the_value_and_drops_the_line_break(self):
        environ = {SECRET_ENV: "value-secret", SECRET_FILE_ENV: str(self.secret_file)}
        self.assertEqual(read_secret(environ), FILE_SECRET)
        self.assertEqual(read_secret({SECRET_ENV: "value-secret"}), "value-secret")
        self.assertEqual(read_secret({}), "")
        self.assertEqual(read_secret({}, secret_file=self.secret_file), FILE_SECRET)

    def test_an_unusable_file_is_an_error_without_path_or_value(self):
        empty = self.root / "empty"
        empty.write_text("\n", encoding="utf-8")
        for path, code in ((self.root / "missing", "secret_file_unreadable"), (empty, "secret_file_empty"),
                           (self.root, "secret_file_unreadable")):
            with self.assertRaises(SecretError) as raised:
                read_secret({SECRET_FILE_ENV: str(path)})
            self.assertEqual(str(raised.exception), code)

    def test_evaluator_reads_the_file(self):
        with mock.patch.dict(os.environ, {SECRET_FILE_ENV: str(self.secret_file), SECRET_ENV: ""}):
            observation, _ = Evaluator().reset(7, {})
        self.assertEqual(observation["manifest"]["episode_id"], self.expected)

    def test_env_passes_the_path_and_the_file_beats_a_value_in_the_environment(self):
        with mock.patch.dict(os.environ, {SECRET_ENV: "policy-side-value"}):
            with Env(secret_file=self.secret_file) as env:
                observation, _ = env.reset(7, {})
                self.assertNotIn(FILE_SECRET, repr(vars(env)))
        self.assertEqual(observation["manifest"]["episode_id"], self.expected)

    def test_serve_refuses_to_start_with_an_unusable_secret_file(self):
        missing = self.root / "no-such-secret"
        done = subprocess.run([sys.executable, "-m", "auditgym", "serve", "--secret-file", str(missing)],
                              input=b'{"op":"reset","seed":1}\n', capture_output=True, cwd=ROOT, timeout=60)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, b"")
        self.assertNotIn(str(missing).encode(), done.stderr)
        self.assertIn(b"secret_file_unreadable", done.stderr)

    def test_server_command_runs_an_evaluator_that_owns_its_secret_and_journal(self):
        journal = self.root / "results.jsonl"
        command = [sys.executable, "-m", "auditgym", "serve", "--secret-file", str(self.secret_file),
                   "--results", str(journal)]
        with mock.patch.dict(os.environ, {SECRET_ENV: ""}), Env(mode="eval", server_command=command) as env:
            observation, _ = env.reset(7, {})
            final = env.step({"action": "submit", "findings": {
                "episode_id": observation["manifest"]["episode_id"], "findings": []}})
        self.assertEqual(observation["manifest"]["episode_id"], self.expected)
        self.assertEqual(final[1:], (0.0, True, False, {}))
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["episode_id"] for row in rows], [self.expected])

    def test_server_command_refuses_client_side_journal_and_secret(self):
        command = [sys.executable, "-m", "auditgym", "serve"]
        for kwargs in ({"results_path": self.root / "r.jsonl"}, {"secret_file": self.secret_file}):
            with self.assertRaises(EnvError) as raised:
                Env(mode="eval", server_command=command, **kwargs)
            self.assertEqual(raised.exception.code, "bad_server_command")
        for bad in ("python -m auditgym serve", [], [sys.executable, 1]):
            with self.assertRaises(EnvError):
                Env(server_command=bad)

    def test_export_reads_the_secret_file(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {SECRET_ENV: "other"}), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            code = cli(["export", "--seed", "7", "--config", "{}", "--out", str(self.root / "ep"),
                        "--secret-file", str(self.secret_file)])
        self.assertEqual(code, 0)
        manifest = json.loads((self.root / "ep" / "public" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["episode_id"], self.expected)
        self.assertNotIn(FILE_SECRET, out.getvalue() + err.getvalue())

    def test_journal_key_is_derived_and_never_the_secret(self):
        self.assertIsNone(journal_key({}))
        derived = journal_key({SECRET_FILE_ENV: str(self.secret_file)})
        self.assertEqual(len(derived), 32)
        self.assertNotIn(FILE_SECRET.encode(), derived)
        self.assertEqual(journal_key({"AUDITGYM_JOURNAL_KEY": "k"}), b"k")


if __name__ == "__main__":
    unittest.main()
