"""Isolation holes the env review reproduced on main 638f1a0 (attack scripts a1, a2, a3, a5):

- A3: Env copied the whole environment into the server, so AUDITGYM_SECRET had to live in the
  policy's process and a policy could start its own Env(mode="train") with the same secret and
  find a hidden seed by comparing episode_id;
- A2: an eval episode ended by end of input, close or a silent 60 s left no journal row;
- A1: in eval, reset() restarted the loaded episode with a fresh 100 steps and 60 s;
- A5: Env.step let a hostile action object raise something other than EnvError.
"""
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
import warnings
from unittest import mock

from auditgym import Env, EnvError
from auditgym import serve
from auditgym.evaluator import Evaluator
from auditgym.secret import SECRET_ENV, SECRET_FILE_ENV
from tests.env.helpers import CLEAN, TINY, FakeClock, tiny

FX_PAGE = {"action": "list_records", "table": "fx"}


def rows(path) -> list[dict]:
    text = pathlib.Path(path).read_text(encoding="utf-8") if pathlib.Path(path).exists() else ""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class SecretNotInheritedTest(unittest.TestCase):
    def episode_id(self, environ, **kwargs) -> str:
        with mock.patch.dict(os.environ, environ), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with Env(mode="train", **kwargs) as env:
                return env.reset(5, CLEAN)[0]["manifest"]["episode_id"]

    def test_a_secret_in_the_environment_does_not_reach_the_server_env_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = pathlib.Path(directory) / "secret"
            secret.write_text("jury-only\n", encoding="utf-8")
            plain = self.episode_id({SECRET_ENV: "", SECRET_FILE_ENV: ""})
            self.assertEqual(self.episode_id({SECRET_ENV: "jury-only", SECRET_FILE_ENV: ""}), plain)
            self.assertEqual(self.episode_id({SECRET_ENV: "", SECRET_FILE_ENV: str(secret)}), plain)
            explicit = self.episode_id({SECRET_ENV: "", SECRET_FILE_ENV: ""}, secret_file=secret)
        self.assertNotEqual(explicit, plain)  # only an explicit secret_file sets it

    def test_eval_refuses_when_the_secret_is_only_in_the_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            file = pathlib.Path(directory) / "secret"
            file.write_text("jury-only-value\n", encoding="utf-8")
            for environ in ({SECRET_ENV: "jury-only-value", SECRET_FILE_ENV: ""},
                            {SECRET_ENV: "", SECRET_FILE_ENV: str(file)}):
                with mock.patch.dict(os.environ, environ), self.subTest(environ=sorted(environ)):
                    results = pathlib.Path(directory) / "r.jsonl"
                    with self.assertRaises(EnvError) as caught:
                        Env(mode="eval", results_path=results)
                    self.assertEqual(caught.exception.code, "bad_secret_file")
                    self.assertNotIn("jury-only-value", str(caught.exception))
                    self.assertNotIn(str(file), str(caught.exception))
                    with self.assertRaises(EnvError):
                        Env.from_files(TINY, mode="eval", results_path=results)

    def test_eval_with_an_explicit_secret_file_or_server_command_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            file = pathlib.Path(directory) / "secret"
            file.write_text("jury-only-value\n", encoding="utf-8")
            results = pathlib.Path(directory) / "r.jsonl"
            with mock.patch.dict(os.environ, {SECRET_ENV: "", SECRET_FILE_ENV: str(file)}):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with Env(mode="eval", results_path=results, secret_file=file) as env:
                        explicit = env.reset(5, CLEAN)[0]["manifest"]["episode_id"]
                    command = [sys.executable, "-m", "auditgym", "serve", "--results", str(results)]
                    with Env(mode="eval", server_command=command) as env:
                        env.reset(5, CLEAN)
            with mock.patch.dict(os.environ, {SECRET_ENV: "", SECRET_FILE_ENV: ""}):
                with Env(mode="eval", results_path=results) as env:
                    plain = env.reset(5, CLEAN)[0]["manifest"]["episode_id"]
        self.assertNotEqual(explicit, plain)

    def test_train_warns_when_the_secret_is_only_in_the_environment(self):
        with mock.patch.dict(os.environ, {SECRET_ENV: "jury-only", SECRET_FILE_ENV: ""}):
            with self.assertWarns(RuntimeWarning):
                Env(mode="train").close()


class EndOfInputTest(unittest.TestCase):
    def test_eof_in_the_middle_of_an_eval_episode_leaves_one_truncated_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            env = Env.from_files(TINY, mode="eval", results_path=path)
            env.reset()
            env.step(FX_PAGE)
            env._process.stdin.close()  # end of input, no close request
            env._process.wait(timeout=30)
            env._process.stdout.close()
            env._process = None
            [row] = rows(path)
        self.assertEqual((row["event"], row["truncated"], row["terminated"]), ("finished", True, False))
        self.assertEqual((row["inspection_steps"], row["ended_by"]), (100, "close"))
        self.assertEqual(row["episode_id"], "ep_tiny0001")
        self.assertEqual(row["reward"], -1.0)  # the tiny episode has an error: silence is -1

    def test_close_in_the_middle_leaves_one_row_and_a_reset_only_episode_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            with Env.from_files(TINY, mode="eval", results_path=path) as env:
                env.reset()
                env.step(FX_PAGE)
            self.assertEqual(len(rows(path)), 1)
            with Env.from_files(TINY, mode="eval", results_path=path) as env:
                env.reset()  # no step: the policy saw only the reset observation
            self.assertEqual(len(rows(path)), 1)

    def test_a_silent_minute_is_a_truncation_found_at_close_and_at_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            clock = FakeClock()
            evaluator = Evaluator(mode="eval", results_path=path, clock=clock)
            first = evaluator.reset(1, CLEAN)[0]["manifest"]["episode_id"]
            clock.t += 61  # no request for 61 s, no step at all
            second = evaluator.reset(2, CLEAN)[0]["manifest"]["episode_id"]
            clock.t += 61
            serve.serve(evaluator, stdin=io.BytesIO(b""), stdout=io.StringIO())  # end of input
            found = rows(path)
        self.assertEqual([(row["episode_id"], row["event"], row["truncated"], row["inspection_steps"])
                          for row in found],
                         [(first, "finished", True, 0), (second, "finished", True, 0)])
        self.assertNotIn("ended_by", found[0])  # out of time, found at reset

    def test_a_loaded_episode_restarted_before_its_first_step_is_not_journaled(self):
        """A harness may load long before the policy calls reset(): that is not a timeout."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            clock = FakeClock()
            evaluator = tiny(mode="eval", results_path=path, clock=clock)
            clock.t += 120
            evaluator.reset()
            evaluator.step({"action": "submit", "findings": {"episode_id": "ep_tiny0001", "findings": []}})
            [row] = rows(path)
        self.assertEqual((row["truncated"], row["inspection_steps"]), (False, 0))


class EvalRestartTest(unittest.TestCase):
    """One rule for reset in the middle of an eval episode, by seed or for a loaded episode: it is
    allowed (the jury may test that reset isolates the episode), the spent run is journaled as
    `abandoned`, and the new run starts with the full budget (DECISIONS H, A1/A2)."""

    def test_eval_restarts_the_loaded_episode_and_journals_the_abandoned_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            with Env.from_files(TINY, mode="eval", results_path=path) as env:
                first = env.reset()[0]
                second = env.reset()[0]  # before the first step: the same episode, no journal row
                self.assertEqual(first, second)
                self.assertEqual(env.step(FX_PAGE)[0]["steps_remaining"], 99)
                self.assertEqual(env.step(FX_PAGE)[0]["steps_remaining"], 98)
                self.assertEqual(env.reset()[0], first)  # mid-episode: allowed, a full budget again
                self.assertEqual(rows(path)[0]["event"], "abandoned")
                self.assertEqual(env.step(FX_PAGE)[0]["steps_remaining"], 99)
                env.step({"action": "submit", "findings": {"episode_id": "ep_tiny0001", "findings": []}})
                self.assertEqual(env.reset()[0], first)  # after the end: a new run of the same episode
            found = rows(path)
        self.assertEqual([(row["event"], row["episode_run"], row["inspection_steps"]) for row in found],
                         [("abandoned", 1, 2), ("finished", 2, 1)])
        self.assertIsNone(found[0]["reward"])
        self.assertEqual(found[1]["reward"], -1.0)  # silence on a book with an error

    def test_a_restarted_loaded_episode_closed_untouched_is_not_a_minus_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            with Env.from_files(TINY, mode="eval", results_path=path) as env:
                env.reset()
                env.step(FX_PAGE)
                env.reset()  # abandons the first run; the new one has no step
            found = rows(path)
        self.assertEqual([(row["event"], row["inspection_steps"]) for row in found], [("abandoned", 1)])

    def test_the_same_rule_for_an_episode_by_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            with Env(mode="eval", results_path=path) as env:
                first = env.reset(5, CLEAN)[0]
                env.step(FX_PAGE)
                self.assertEqual(env.reset(5, CLEAN)[0], first)
                self.assertEqual(env.step(FX_PAGE)[0]["steps_remaining"], 99)
            found = rows(path)
        self.assertEqual([(row["event"], row["inspection_steps"], row["ended_by"] if "ended_by" in row else None)
                          for row in found], [("abandoned", 1, None), ("finished", 100, "close")])

    def test_train_still_restarts_the_loaded_episode(self):
        with Env.from_files(TINY) as env:
            env.reset()
            env.step(FX_PAGE)
            self.assertEqual(env.reset()[0]["steps_remaining"], 100)

    def test_a_replayed_seed_is_numbered_in_the_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, clock=FakeClock())
            for _ in range(2):
                episode_id = evaluator.reset(7, CLEAN)[0]["manifest"]["episode_id"]
                evaluator.step(FX_PAGE)
            evaluator.step({"action": "submit", "findings": {"episode_id": episode_id, "findings": []}})
            found = rows(path)
        self.assertEqual([(row["event"], row["episode_run"]) for row in found], [("abandoned", 1), ("finished", 2)])


class HostileActionTest(unittest.TestCase):
    def test_env_step_raises_nothing_but_env_error(self):
        class BadValues(dict):
            def values(self):
                raise LookupError("values")

        class BadItems(dict):
            def items(self):
                raise ZeroDivisionError("items")

        class BadList(list):
            def __iter__(self):
                raise KeyError("iter")

        with Env.from_files(TINY) as env:
            env.reset()
            for action in (BadValues(action="calculator", expression="1"),
                           BadItems(action="calculator", expression="1"),
                           {"action": "calculator", "expression": "1", "args": [BadList([1])]}):
                observation = env.step(action)[0]
                self.assertEqual(observation["error"]["code"], "unknown_action", type(action))


if __name__ == "__main__":
    unittest.main()
