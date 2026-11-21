"""Eval mode refuses `load` unless `--allow-load`; every journal row says where its episode came from
(`source`, `source_sha256`); verify-journal fails on eval rows of loaded episodes (DECISIONS P)."""
import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from auditgym import Env
from auditgym.__main__ import main as cli
from auditgym.evaluator import Evaluator
from auditgym.journal import loaded_eval_rows
from auditgym.secret import JOURNAL_KEY_ENV, JOURNAL_KEY_FILE_ENV, SECRET_ENV, SECRET_FILE_ENV
from tests.env.helpers import ROOT, TINY
from tests.env.test_process import CORRECT_TINY
from tools.schema_lite import load_schema, validate

SECRET = "load-policy-secret"
CLEAN_ENV = {SECRET_ENV: "", SECRET_FILE_ENV: "", JOURNAL_KEY_ENV: "", JOURNAL_KEY_FILE_ENV: ""}
LOAD = json.dumps({"op": "load", "directory": str(TINY)}) + "\n"
EMPTY = {"episode_id": "ep_tiny0001", "findings": []}
SUBMIT = json.dumps({"op": "step", "action": {"action": "submit", "findings": EMPTY}}) + "\n"


def serve(arguments, lines, results, secret_file=None):
    environment = {key: value for key, value in os.environ.items() if key not in CLEAN_ENV}
    environment["PYTHONPATH"] = str(ROOT)
    command = [sys.executable, "-m", "auditgym", "serve", *arguments, "--results", str(results)]
    if secret_file is not None:
        command += ["--secret-file", str(secret_file)]
    done = subprocess.run(command, input="".join(lines).encode(), capture_output=True, env=environment,
                          cwd=str(ROOT), timeout=60)
    return [json.loads(line) for line in done.stdout.splitlines()]


def verify(path, secret_file):
    out = io.StringIO()
    with mock.patch.dict(os.environ, CLEAN_ENV), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = cli(["verify-journal", str(path), "--secret-file", str(secret_file)])
    return code, out.getvalue()


class LoadPolicyTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.dir = pathlib.Path(scratch.name)
        self.results = self.dir / "results.jsonl"
        self.secret = self.dir / "secret"
        self.secret.write_text(SECRET, encoding="utf-8")
        os.chmod(self.secret, 0o600)

    def rows(self):
        if not self.results.exists():
            return []
        return [json.loads(line) for line in self.results.read_text(encoding="utf-8").splitlines()]

    def test_eval_refuses_load_by_default(self):
        replies = serve(["--mode", "eval"], [LOAD, SUBMIT], self.results, self.secret)
        self.assertEqual(replies[0]["observation"]["error"],
                         {"code": "bad_request", "message": "Bad request."})
        self.assertEqual(self.rows(), [])  # nothing was loaded, so nothing was journaled

    def test_eval_no_load_is_still_accepted_and_refuses(self):
        replies = serve(["--mode", "eval", "--no-load"], [LOAD], self.results, self.secret)
        self.assertEqual(replies[0]["observation"]["error"]["code"], "bad_request")

    def test_eval_allow_load_works(self):
        replies = serve(["--mode", "eval", "--allow-load"], [LOAD], self.results, self.secret)
        self.assertEqual(replies[0]["observation"]["type"], "reset")

    def test_no_load_wins_over_allow_load(self):
        replies = serve(["--mode", "eval", "--allow-load", "--no-load"], [LOAD], self.results, self.secret)
        self.assertEqual(replies[0]["observation"]["error"]["code"], "bad_request")

    def test_train_keeps_load_by_default_and_no_load_refuses(self):
        self.assertEqual(serve(["--mode", "train"], [LOAD], self.results)[0]["observation"]["type"], "reset")
        self.assertEqual(serve(["--mode", "train", "--no-load"], [LOAD], self.results)[0]["observation"]["error"]["code"],
                         "bad_request")

    def test_env_from_files_in_eval_still_works_and_is_marked(self):
        with Env.from_files(TINY, mode="eval", results_path=self.results, secret_file=self.secret) as env:
            env.reset()
            env.step({"action": "submit", "findings": CORRECT_TINY})
        row = self.rows()[0]
        self.assertEqual(row["source"], "load")
        self.assertEqual(len(row["source_sha256"]), 64)

    def test_env_without_files_refuses_load_in_eval(self):
        with Env(mode="eval", results_path=self.results, secret_file=self.secret) as env:
            with self.assertRaises(Exception):
                env._request({"op": "load", "directory": str(TINY)})  # a policy holding the pipe

    def test_journal_row_of_a_generated_episode_says_seed(self):
        with Env(mode="eval", results_path=self.results, secret_file=self.secret) as env:
            env.reset(5, {})
            env.step({"action": "submit", "findings": {"episode_id": "x", "findings": []}})
        row = self.rows()[0]
        self.assertEqual(row["source"], "seed")
        self.assertNotIn("source_sha256", row)
        self.assertEqual(validate(row, load_schema("journal_row")), [])

    def test_loaded_row_matches_the_schema_and_the_hash_follows_the_content(self):
        def run(directory):
            path = self.dir / f"r-{directory.name}.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, secret=SECRET)
            evaluator.load(str(directory))
            evaluator.step({"action": "submit", "findings": CORRECT_TINY})
            return json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        copy = self.dir / "copy"
        shutil.copytree(TINY, copy)
        first = run(copy)
        self.assertEqual(validate(first, load_schema("journal_row")), [])
        self.assertEqual(run(copy)["source_sha256"], first["source_sha256"])
        other = self.dir / "other"
        shutil.copytree(TINY, other)
        with open(other / "public" / "fx.csv", "a", encoding="utf-8") as handle:
            handle.write("\n")  # one byte more: still loads, another hash
        self.assertNotEqual(run(other)["source_sha256"], first["source_sha256"])

    def test_abandoned_loaded_episode_is_marked_too(self):
        evaluator = Evaluator(mode="eval", results_path=self.results, secret=SECRET)
        evaluator.load(str(TINY))
        evaluator.step({"action": "calculator", "expression": "1"})
        evaluator.reset(3, {})
        evaluator.step({"action": "submit", "findings": EMPTY})
        abandoned, finished = self.rows()
        self.assertEqual((abandoned["event"], abandoned["source"]), ("abandoned", "load"))
        self.assertEqual((finished["event"], finished["source"]), ("finished", "seed"))

    def test_verify_journal_fails_on_eval_rows_with_source_load(self):
        evaluator = Evaluator(mode="eval", results_path=self.results, secret=SECRET)
        evaluator.load(str(TINY))
        evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        code, text = verify(self.results, self.secret)
        self.assertEqual(code, 1, text)
        self.assertIn('source "load"', text)

    def test_verify_journal_accepts_seed_rows_and_old_rows_without_source(self):
        evaluator = Evaluator(mode="eval", results_path=self.results, secret=SECRET)
        evaluator.reset(5, {})
        evaluator.step({"action": "submit", "findings": {"episode_id": "x", "findings": []}})
        self.assertEqual(verify(self.results, self.secret)[0], 0)
        old = json.dumps({"mode": "eval", "secret": "set"})
        self.assertEqual(loaded_eval_rows([old]), 0)  # a row from before the field: counted as seed

    def test_train_rows_of_loaded_episodes_are_not_flagged(self):
        row = json.dumps({"mode": "train", "source": "load"})
        self.assertEqual(loaded_eval_rows([row]), 0)

    def test_forged_directory_does_not_give_an_accepted_eval_row(self):
        """The jury's attack: the first line to the evaluator is a load of a directory with a hidden
        episode_id and no truth, then an empty submit. The default evaluator refuses it; and when load
        is allowed (--allow-load) the row is marked and verify-journal fails."""
        forged = self.dir / "forged"
        shutil.copytree(TINY, forged)
        truth = json.loads((forged / "private" / "truth.json").read_text(encoding="utf-8"))
        truth["issues"] = []
        (forged / "private" / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
        lines = [json.dumps({"op": "load", "directory": str(forged)}) + "\n", SUBMIT]

        serve(["--mode", "eval"], lines, self.results, self.secret)
        self.assertEqual(self.rows(), [])  # no row at all
        self.results.write_text("", encoding="utf-8")

        replies = serve(["--mode", "eval", "--allow-load"], lines, self.results, self.secret)
        self.assertEqual(replies[0]["observation"]["type"], "reset")
        rows = self.rows()
        self.assertEqual((rows[0]["source"], rows[0]["reward"]), ("load", 1.0))
        code, text = verify(self.results, self.secret)
        self.assertEqual(code, 1, text)


if __name__ == "__main__":
    unittest.main()
