"""python -m auditgym export: the episode directory a jury with AUDITGYM_SECRET gets for its oracle."""
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
from auditgym.evaluator import SECRET_ENV, Evaluator
from auditgym.generate import generate
from auditgym.storage import episode_files, load_episode
from policies.runner import make_policy, run_episode
from tests.env.helpers import ROOT, WITH_ERRORS

SECRET = "jury-only-secret-value"
SEED = 11


def export(directory, secret, *extra, seed=SEED, config=WITH_ERRORS):
    environment = {SECRET_ENV: secret}
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, environment), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli(["export", "--seed", str(seed), "--config", json.dumps(config), "--out", str(directory), *extra])
    return code, out.getvalue() + err.getvalue()


class ExportTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = pathlib.Path(scratch.name)

    def oracle_reward(self, directory):
        truth = load_episode(directory)["private"]["truth"]
        with mock.patch.dict(os.environ, {SECRET_ENV: ""}), Env.from_files(directory) as env:
            return run_episode(env, make_policy("oracle", truth))["reward"]

    def test_with_secret_oracle_gets_one_through_from_files(self):
        code, text = export(self.root / "ep", SECRET)
        self.assertEqual(code, 0, text)
        self.assertEqual(self.oracle_reward(self.root / "ep"), 1.0)
        self.assertTrue(load_episode(self.root / "ep")["private"]["truth"]["issues"])

    def test_the_open_generate_gives_the_oracle_minus_one_under_a_secret(self):
        """The failure this command removes: a reference built without the secret."""
        open_episode = generate(SEED, WITH_ERRORS)
        secret_episode = generate(SEED, WITH_ERRORS, secret=SECRET)
        self.assertNotEqual(open_episode["public"]["manifest"]["episode_id"],
                            secret_episode["public"]["manifest"]["episode_id"])
        from auditgym.storage import write_episode
        write_episode(secret_episode, self.root / "secret")
        truth = open_episode["private"]["truth"]
        with Env.from_files(self.root / "secret") as env:
            self.assertEqual(run_episode(env, make_policy("oracle", truth))["reward"], -1.0)

    def test_export_with_secret_equals_the_evaluator_reset(self):
        code, text = export(self.root / "ep", SECRET)
        self.assertEqual(code, 0, text)
        exported = load_episode(self.root / "ep")
        with mock.patch.dict(os.environ, {SECRET_ENV: SECRET}):
            evaluator = Evaluator()
            observation, _ = evaluator.reset(SEED, WITH_ERRORS)
            with Env() as env:
                env_observation, _ = env.reset(SEED, WITH_ERRORS)
        self.assertEqual(exported["public"], evaluator._episode["public"])
        self.assertEqual(exported["private"]["truth"], evaluator._episode["private"]["truth"])
        self.assertEqual(exported["public"]["manifest"], evaluator._episode["public"]["manifest"])
        self.assertEqual(exported["public"]["manifest"]["episode_id"], observation["manifest"]["episode_id"])
        self.assertEqual(exported["public"]["manifest"]["episode_id"], env_observation["manifest"]["episode_id"])
        self.assertEqual(episode_files(exported), episode_files(evaluator._episode))

    def test_without_secret_equals_generate(self):
        code, text = export(self.root / "ep", "")
        self.assertEqual(code, 0, text)
        self.assertEqual(episode_files(load_episode(self.root / "ep")), episode_files(generate(SEED, WITH_ERRORS)))
        self.assertEqual(self.oracle_reward(self.root / "ep"), 1.0)

    def test_the_secret_is_not_printed_or_written(self):
        code, text = export(self.root / "ep", SECRET)
        self.assertNotIn(SECRET, text)
        for path in (self.root / "ep").rglob("*"):
            if path.is_file():
                self.assertNotIn(SECRET, path.read_text(encoding="utf-8"), path)

    def test_config_file_equals_inline_config(self):
        path = self.root / "config.json"
        path.write_text(json.dumps(WITH_ERRORS), encoding="utf-8")
        out = io.StringIO()
        with mock.patch.dict(os.environ, {SECRET_ENV: SECRET}), contextlib.redirect_stdout(out):
            self.assertEqual(cli(["export", "--seed", str(SEED), "--config-file", str(path),
                                  "--out", str(self.root / "b")]), 0)
        export(self.root / "a", SECRET)
        self.assertEqual(episode_files(load_episode(self.root / "a")), episode_files(load_episode(self.root / "b")))

    def test_bad_input_is_a_short_message_and_a_nonzero_code(self):
        export(self.root / "full", SECRET)
        for extra, directory in ((["--config", "{"], "x1"), ([], "full")):
            args = ["export", "--seed", "1", "--out", str(self.root / directory)] + (extra or [])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(cli(args), 2)
            self.assertIn("export:", err.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli(["export", "--seed", "1", "--config", '{"errors": {"nope": 1}}', "--out", str(self.root / "y")])
        self.assertEqual(code, 2)
        self.assertIn("bad configuration", err.getvalue())
        self.assertFalse((self.root / "y").exists())

    def test_module_entry_point(self):
        environment = dict(os.environ, **{SECRET_ENV: SECRET})
        done = subprocess.run([sys.executable, "-m", "auditgym", "export", "--seed", "5", "--out", str(self.root / "m")],
                              cwd=ROOT, env=environment, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn(SECRET, done.stdout + done.stderr)
        self.assertEqual(self.oracle_reward(self.root / "m"), 1.0)


if __name__ == "__main__":
    unittest.main()
