"""Process-boundary hardening found by review: the results path is not in the process list, the
pipe gives a policy no more power than reset/step, and the server's stderr does not reach the
policy process."""
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from auditgym import Env
from tests.env.helpers import TINY

# DECISIONS A2 / CONTRACTS v2 §3.1: duplicate evidence = both docs + one posting row of each
# invoice (a1 = debit of k7m2qa, a3 = debit of z5c1ty); the two docs alone are a FP.
CORRECT_TINY = {"episode_id": "ep_tiny0001", "findings": [{
    "type": "duplicate_invoice", "affected_ids": ["k7m2qa", "z5c1ty"], "claimed_amount_minor": 5000000,
    "currency": "KZT", "evidence_ids": ["k7m2qa", "z5c1ty", "a1", "a3"], "explanation": "x"}]}


def _raw(env, message: dict) -> dict:
    """What a policy could do by writing to the pipe directly."""
    process = env._process
    process.stdin.write(json.dumps(message).encode() + b"\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def _open_descriptors_to(path: pathlib.Path) -> int:
    target = path.stat()
    count = 0
    for name in os.listdir("/dev/fd"):
        try:
            info = os.fstat(int(name))
        except OSError:
            continue
        count += (info.st_ino, info.st_dev) == (target.st_ino, target.st_dev)
    return count


def _command_line(pid: int) -> str:
    """The process's argv as the OS shows it; slim containers have /proc but no `ps`."""
    proc = pathlib.Path(f"/proc/{pid}/cmdline")
    if proc.exists():
        return proc.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    return subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)],
                          capture_output=True, text=True, check=True).stdout


@unittest.skipUnless(os.name == "posix", "descriptor passing is POSIX only")
class ResultsPathTest(unittest.TestCase):
    def test_results_path_is_not_in_the_process_list_and_results_are_written(self):
        with tempfile.TemporaryDirectory(prefix="secret_results_dir_") as directory:
            results = pathlib.Path(directory) / "results.jsonl"
            with Env.from_files(TINY, mode="eval", results_path=results) as env:
                listing = _command_line(env._process.pid)
                self.assertIn("auditgym", listing)
                self.assertNotIn("secret_results_dir_", listing)
                self.assertNotIn(str(results), listing)
                self.assertEqual(_open_descriptors_to(results), 0, "the client keeps the results file open")
                env.reset()
                env.step({"action": "submit", "findings": CORRECT_TINY})
            lines = results.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["reward"], 1.0)

    def test_results_from_two_envs_append(self):
        with tempfile.TemporaryDirectory() as directory:
            results = pathlib.Path(directory) / "results.jsonl"
            for _ in range(2):
                with Env(mode="eval", results_path=results) as env:
                    first, _ = env.reset(5, {"errors": {}})
                    env.step({"action": "submit", "findings": {"episode_id": first["manifest"]["episode_id"],
                                                               "findings": []}})
            self.assertEqual(len(results.read_text(encoding="utf-8").splitlines()), 2)


class PipeInjectionTest(unittest.TestCase):
    def test_load_through_the_pipe_after_reset_is_refused(self):
        with Env.from_files(TINY) as env:
            env.reset()
            env.step({"action": "list_records", "table": "fx"})
            reply = _raw(env, {"op": "load", "directory": "/"})
            self.assertEqual(reply["observation"]["error"]["code"], "bad_request")
            self.assertEqual(reply["observation"]["steps_remaining"], 99)
            observation, reward, terminated, _, info = env.step({"action": "submit", "findings": CORRECT_TINY})
            self.assertEqual((reward, terminated), (0.998, True))  # one step spent, not two

    def test_env_without_files_refuses_load_even_first(self):
        with Env() as env:
            reply = _raw(env, {"op": "load", "directory": str(TINY)})
            self.assertEqual(reply["observation"]["error"]["code"], "bad_request")
            reply = _raw(env, {"op": "reset"})
            self.assertEqual(reply["observation"]["type"], "reset")  # default seed (C7), not a load

    def test_server_stderr_is_not_connected_to_the_client(self):
        seen = {}
        real = subprocess.Popen

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        with mock.patch("auditgym.client.subprocess.Popen", side_effect=spy):
            with Env() as env:
                self.assertIsNone(env._process.stderr)
        self.assertIs(seen.get("stderr"), subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
