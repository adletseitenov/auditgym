"""DECISIONS AC: an eval Env without an explicit journal path falls back to <tmp>/auditgym/results.jsonl when the
default path cannot be opened (non-root user, read-only file system); an explicit path stays strict."""
from __future__ import annotations

import os
import pathlib
import stat
import tempfile
import unittest
from unittest import mock

from auditgym import Env
from auditgym.actions import default_results_with_fallback as _default_eval_results
from auditgym.client import EnvError


@unittest.skipIf(os.name != "posix" or os.geteuid() == 0, "needs permission checks that root bypasses")
class ClientResultsFallbackTest(unittest.TestCase):
    def setUp(self):
        self._scratch = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._scratch.name)
        self.readonly = root / "ro"
        self.readonly.mkdir()
        self.readonly.chmod(stat.S_IRUSR | stat.S_IXUSR)
        self.tmp = root / "tmp"
        self.tmp.mkdir()
        self.addCleanup(self._cleanup)
        env = {key: value for key, value in os.environ.items() if key != "AUDITGYM_RESULTS"}
        patches = [mock.patch.dict(os.environ, env, clear=True), mock.patch.object(tempfile, "tempdir", str(self.tmp))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.previous = os.getcwd()
        os.chdir(self.readonly)
        self.addCleanup(os.chdir, self.previous)

    def _cleanup(self):
        self.readonly.chmod(stat.S_IRWXU)
        self._scratch.cleanup()

    def test_unwritable_default_falls_back_to_the_temporary_directory(self):
        path = _default_eval_results()
        self.assertEqual(pathlib.Path(path), self.tmp / "auditgym" / "results.jsonl")
        self.assertTrue(os.path.exists(path))

    def test_an_eval_env_starts_and_journals_in_the_fallback(self):
        with Env(mode="eval") as env:
            env.reset(1, {})
        self.assertTrue((self.tmp / "auditgym" / "results.jsonl").exists())
        self.assertFalse((self.readonly / "runs").exists())

    def test_the_eval_server_started_directly_uses_the_fallback_too(self):
        """`python -m auditgym serve --mode eval` (tools.leakscan starts it so) in a read-only directory."""
        import json
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(root), TMPDIR=str(self.tmp))
        environment.pop("AUDITGYM_RESULTS", None)
        reply = subprocess.run([sys.executable, "-m", "auditgym", "serve", "--mode", "eval"], cwd=self.readonly,
                               input=json.dumps({"op": "reset", "seed": 3, "config": {}}) + "\n", text=True,
                               capture_output=True, env=environment, timeout=60)
        self.assertEqual(reply.returncode, 0, reply.stderr)
        self.assertIn("observation", json.loads(reply.stdout.splitlines()[0]))
        self.assertTrue((self.tmp / "auditgym" / "results.jsonl").exists())

    def test_a_writable_default_is_used_when_it_works(self):
        self.readonly.chmod(stat.S_IRWXU)
        self.assertEqual(pathlib.Path(_default_eval_results()).resolve(), (self.readonly / "runs" / "results.jsonl").resolve())

    def test_neither_writable_is_bad_results_path(self):
        self.tmp.chmod(stat.S_IRUSR | stat.S_IXUSR)
        self.addCleanup(self.tmp.chmod, stat.S_IRWXU)
        with self.assertRaises(EnvError) as caught:
            Env(mode="eval")
        self.assertEqual(caught.exception.code, "bad_results_path")

    def test_an_explicit_path_never_falls_back(self):
        with self.assertRaises(EnvError) as caught:
            Env(mode="eval", results_path=self.readonly / "runs" / "results.jsonl")
        self.assertEqual(caught.exception.code, "bad_results_path")
        self.assertFalse((self.tmp / "auditgym").exists())

    def test_the_environment_path_is_the_primary_one(self):
        with mock.patch.dict(os.environ, {"AUDITGYM_RESULTS": str(self.readonly / "x" / "named.jsonl")}):
            path = _default_eval_results()
        self.assertEqual(pathlib.Path(path), self.tmp / "auditgym" / "named.jsonl")


if __name__ == "__main__":
    unittest.main()
