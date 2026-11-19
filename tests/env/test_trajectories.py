"""CASE §09: trajectories.jsonl with episode_id, step, action, observation, reward, terminated,
truncated; no truth in training trajectories. Oracle trajectories live in a separate file."""
import contextlib
import io
import json
import pathlib
import shutil
import tempfile
import unittest

from auditgym import Env
from tools import trajectories
from tools.trajectories import (FIELDS, OracleSubmitPolicy, Recorder, SolverPolicy, TrajectoryError,
                                TrajectoryWriter, record, run_policy, scan_file)

CONFIG = {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}}


class TrajectoriesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="trajectories_test_"))
        cls.written = record(cls.tmp / "trajectories.jsonl", seeds=(4001, 5002), config=CONFIG,
                             policies=("random", "solver", "oracle"))
        cls.lines = [json.loads(line) for line in cls.written["train"].read_text(encoding="utf-8").splitlines()]
        cls.oracle_lines = cls.written["oracle"].read_text(encoding="utf-8").splitlines()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _scan_lines(self, lines, training=True):
        path = self.tmp / "edited.jsonl"
        path.write_text("".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines),
                        encoding="utf-8")
        return scan_file(path, self.written["episodes"], training=training)

    def test_exact_fields_and_steps(self):
        self.assertTrue(self.lines)
        for line in self.lines:
            self.assertEqual(tuple(line), FIELDS)
        starts = [line for line in self.lines if line["step"] == 0]
        self.assertEqual(len(starts), 4)  # random and solver on two seeds
        self.assertTrue(all(line["action"] is None for line in starts))
        ends = [line for line in self.lines if line["terminated"] or line["truncated"]]
        self.assertEqual(len(ends), 4)

    def test_written_files_are_clean(self):
        self.assertEqual(scan_file(self.written["train"], self.written["episodes"]), [])
        self.assertEqual(scan_file(self.written["oracle"], self.written["episodes"], training=False), [])

    def test_oracle_lines_are_separate_and_caught_in_a_training_file(self):
        self.assertIn(".oracle.", self.written["oracle"].name)
        train_text = self.written["train"].read_text(encoding="utf-8")
        for line in self.oracle_lines:
            if json.loads(line)["step"] > 0:  # the reset line is the same for every policy
                self.assertNotIn(line, train_text)
        problems = self._scan_lines([json.dumps(line) for line in self.lines] + self.oracle_lines)
        self.assertTrue(any("never shown" in p for p in problems), problems)

    def test_writer_refuses_mixing(self):
        with self.assertRaises(TrajectoryError):
            TrajectoryWriter(self.tmp / "oracle_runs.jsonl")
        with self.assertRaises(TrajectoryError):
            TrajectoryWriter(self.tmp / "runs.jsonl", privileged=True)
        with TrajectoryWriter(self.tmp / "runs.jsonl") as writer, Env() as env:
            with self.assertRaises(TrajectoryError):
                Recorder(env, writer, OracleSubmitPolicy({}))
        with TrajectoryWriter(self.tmp / "w.jsonl") as writer:
            with self.assertRaises(TrajectoryError):
                writer.write({"episode_id": "x", "step": 0, "action": None, "observation": {}, "reward": 0.0,
                              "terminated": False, "truncated": False, "info": {}})

    def test_scan_finds_planted_truth(self):
        episode_id, episode = next(iter(self.written["episodes"].items()))
        issue = episode["private"]["truth"]["issues"][0]
        line = next(l for l in self.lines if l["episode_id"] == episode_id and l["observation"]["type"] == "records")
        planted = [
            dict(line, observation={"type": "error", "steps_remaining": 5,
                                    "error": {"code": "unknown_id", "message": issue["affected_ids"][0]}}),
            dict(line, observation=dict(line["observation"], records=[], next_cursor=issue["issue_id"])),
            dict(line, observation=dict(line["observation"], steps_remaining=episode["private"]["provenance"]["seed"])),
            dict(line, observation={"type": "final", "steps_remaining": 1, "evidence_sets": []}),
            {key: line[key] for key in FIELDS if key != "action"},
        ]
        for bad in planted:
            self.assertTrue(self._scan_lines([bad]), f"not found: {bad['observation']}")

    def test_recorder_wraps_env_from_files(self):
        from tests.env.helpers import TINY

        path = self.tmp / "tiny.jsonl"
        with Env.from_files(TINY) as env, TrajectoryWriter(path) as writer:
            reward = run_policy(Recorder(env, writer, SolverPolicy()), SolverPolicy())
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertGreater(reward, 0.9)
        self.assertEqual(json.loads(lines[-1])["terminated"], True)

    def test_cli(self):
        out = self.tmp / "cli" / "trajectories.jsonl"
        out.parent.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            code = trajectories.main(["--out", str(out), "--seeds", "6003", "--config", json.dumps(CONFIG),
                                      "--policies", "solver,oracle"])
        self.assertEqual(code, 0)
        self.assertTrue((out.parent / "trajectories.oracle.jsonl").is_file())

    def test_output_directory_is_created(self):
        out = self.tmp / "does" / "not" / "exist" / "trajectories.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            code = trajectories.main(["--out", str(out), "--seeds", "6004", "--config", json.dumps(CONFIG),
                                      "--policies", "solver,oracle"])
        self.assertEqual(code, 0)
        self.assertTrue(out.is_file())
        self.assertTrue(out.with_name("trajectories.oracle.jsonl").is_file())
        with TrajectoryWriter(self.tmp / "deep" / "er" / "w.jsonl"):
            pass
        self.assertTrue((self.tmp / "deep" / "er" / "w.jsonl").is_file())

    def test_no_arguments_record_the_shipped_train_episodes(self):
        """`make trajectories` used to die on the missing --episodes/--seeds."""
        from tests.env.helpers import TINY
        from unittest import mock

        default = self.tmp / "shipped"
        shutil.copytree(TINY, default)
        out = self.tmp / "nodefault" / "trajectories.jsonl"
        with mock.patch.object(trajectories, "DEFAULT_EPISODES", default), \
                contextlib.redirect_stdout(io.StringIO()):
            code = trajectories.main(["--out", str(out), "--policies", "solver"])
        self.assertEqual(code, 0)
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({line["episode_id"] for line in lines}, {"ep_tiny0001"})

    def test_the_default_points_at_the_train_episodes_of_the_repository(self):
        self.assertEqual(trajectories.DEFAULT_EPISODES, trajectories.ROOT / "episodes" / "train")
        self.assertTrue((trajectories.DEFAULT_EPISODES / "ep_01" / "public").is_dir())
        self.assertEqual(trajectories.DEFAULT_OUT, "trajectories.jsonl")

    def test_command_line_without_arguments_runs_in_another_directory(self):
        import os
        import subprocess
        import sys

        elsewhere = self.tmp / "elsewhere"
        elsewhere.mkdir()
        environment = dict(os.environ, PYTHONPATH=str(trajectories.ROOT))
        result = subprocess.run([sys.executable, "-m", "tools.trajectories", "--policies", "solver"],
                                cwd=elsewhere, env=environment, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = (elsewhere / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        episodes = {json.loads(line)["episode_id"] for line in lines}
        self.assertEqual(len(episodes), len(list(trajectories.DEFAULT_EPISODES.iterdir())))


if __name__ == "__main__":
    unittest.main()
