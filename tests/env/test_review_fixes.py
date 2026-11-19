"""Regression tests for what the code review of the env zone found (contract, process boundary,
blind attack E8), reproduced on main before the fix:

- unreadable lines: a long reset spent a step of the running episode, the client handed a step
  reply back as a reset, a submit the server could not decode left the episode running;
- submit: an explicit null episode_id was replaced, two different answers were not refused;
- reset: a loaded episode ignored a bad config, the default seeds broke at the seed bound,
  config and options together were not refused;
- an episode replaced after a step left no journal line (main journals it as "abandoned" since
  827b6ca; the test below checks that format);
- the journal: a symlink was followed, a FIFO hung the client, a closed stdout made the journal
  descriptor 1 (the protocol channel);
- the evaluator secret of generate() was never passed by the environment (L1; main took the
  secret file in a27fa64 and tests it in tests/env/test_secret.py, so the PR's own test is dropped);
- E8: a policy that picks documents by position finds errors only at chance level.
"""
import json
import math
import os
import pathlib
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unittest

from auditgym import Env, EnvError
from auditgym.evaluator import DEFAULT_SEED, MAX_SEED_BITS, Evaluator, EvaluatorError
from auditgym.generate import generate
from tests.env.helpers import CLEAN, TINY, FakeClock, tiny
from tests.env.test_evaluator import CORRECT_TINY
from tools import metamorph

ROOT = pathlib.Path(__file__).resolve().parents[2]
FX_PAGE = {"action": "list_records", "table": "fx"}
PAD = "x" * ((1 << 20) + 10)


def serve(lines, *args, cwd=None, code=False):
    data = b"".join((line if isinstance(line, bytes) else line.encode()) + b"\n" for line in lines)
    result = subprocess.run([sys.executable, "-m", "auditgym", "serve", *args], input=data, capture_output=True,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)), cwd=cwd or str(ROOT), timeout=120)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    return (result.returncode, replies) if code else replies


def journal(path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]


class UnreadableLineTest(unittest.TestCase):
    def test_a_long_reset_spends_nothing_and_the_client_refuses_it(self):
        step = json.dumps({"op": "step", "action": FX_PAGE})
        replies = serve([json.dumps({"op": "load", "directory": str(TINY)}), step,
                         json.dumps({"op": "reset", "config": {"note": "submit " + PAD}}), step])
        self.assertEqual(replies[2]["observation"]["error"]["code"], "bad_request")
        self.assertEqual(replies[3]["observation"]["steps_remaining"], 98)  # the long reset cost nothing
        with Env.from_files(TINY) as env:
            env.reset()
            env.step(FX_PAGE)
            with self.assertRaises(EnvError) as caught:
                env.reset(4, {"errors": {}, "note": "submit " + PAD})
            self.assertEqual(caught.exception.code, "bad_config")
            self.assertEqual(env.step(FX_PAGE)[0]["steps_remaining"], 98)

    def test_the_client_never_returns_a_step_reply_from_reset(self):
        with Env() as env:
            env.reset(3, CLEAN)
            # A request written past the client: its reply (records) is what the next reset reads.
            env._process.stdin.write(json.dumps({"op": "step", "action": FX_PAGE}).encode() + b"\n")
            env._process.stdin.flush()
            with self.assertRaises(EnvError) as caught:
                env.reset(4, CLEAN)
            self.assertEqual(caught.exception.code, "bad_reply")

    def test_a_submit_the_server_cannot_decode_still_ends_the_episode(self):
        deep = '{"op": "step", "action": {"action": "submit", "findings": ' + "[" * 100000 + "]" * 100000 + "}}"
        cut = '{"op": "step", "action": {"action": "list_records", "table": "fx"'
        for line, kind, terminated in ((deep, "final", True), (cut, "error", False)):
            replies = serve([json.dumps({"op": "load", "directory": str(TINY)}), line])
            self.assertEqual((replies[1]["observation"]["type"], replies[1]["terminated"]), (kind, terminated), kind)
            self.assertEqual(replies[1]["observation"]["steps_remaining"], 99)
        # a line that is no step at all stays a free bad request
        replies = serve([json.dumps({"op": "load", "directory": str(TINY)}), "not json"])
        self.assertEqual((replies[1]["observation"]["error"]["code"], replies[1]["observation"]["steps_remaining"]),
                         ("bad_request", 100))

    def test_an_action_string_that_names_a_submit_but_does_not_decode(self):
        evaluator = tiny()
        observation, reward, terminated, _, _ = evaluator.step(
            '{"action": "submit", "findings": {"episode_id": "ep_tiny0001", "findings": [')
        self.assertEqual((observation["type"], reward, terminated), ("final", -1.0, True))
        observation = tiny().step('{"action": "list_records", "table": "fx"')[0]
        self.assertEqual(observation["error"]["code"], "unknown_action")

    def test_a_submit_named_after_the_first_mebibyte_is_found(self):
        line = json.dumps({"op": "step", "findings": PAD, "action": "SUBMIT"})
        replies = serve([json.dumps({"op": "load", "directory": str(TINY)}), line])
        self.assertEqual((replies[1]["observation"]["type"], replies[1]["reward"]), ("final", -1.0))

    def test_a_submit_name_split_between_two_reads_is_found(self):
        head = '{"op": "step", "pad": "'
        cut = '", "action": "sub'
        pad = "x" * ((1 << 20) + 1 - len(head) - len(cut))
        replies = serve([json.dumps({"op": "load", "directory": str(TINY)}),
                         head + pad + '", "action": "submit", "findings": {}}'])
        self.assertEqual((replies[1]["observation"]["type"], replies[1]["reward"]), ("final", -1.0))


class SubmitAndResetTest(unittest.TestCase):
    def test_flat_null_episode_id_is_kept_and_two_answers_are_refused(self):
        findings = CORRECT_TINY["findings"]
        for action in ({"action": "submit", "episode_id": None, "findings": findings},
                       {"action": "submit", "findings": CORRECT_TINY,
                        "findings.json": {"episode_id": "ep_tiny0001", "findings": []}}):
            _, reward, terminated, _, _ = tiny().step(action)
            self.assertEqual((reward, terminated), (-1.0, True), action)
        _, reward, _, _, _ = tiny().step({"action": "submit", "findings": CORRECT_TINY, "findings.json": CORRECT_TINY})
        self.assertGreater(reward, 0.9)

    def test_loaded_episode_refuses_a_config_without_a_seed(self):
        evaluator = tiny()
        for config in ({"nope": 1}, {"errors": {}}):
            with self.assertRaises(EvaluatorError):
                evaluator.reset(None, config)
        self.assertEqual(evaluator.reset(None, {})[0]["manifest"]["episode_id"], "ep_tiny0001")

    def test_default_seeds_start_again_after_the_largest_seed(self):
        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset((1 << MAX_SEED_BITS) - 1, CLEAN)
        self.assertEqual(evaluator.reset(None, CLEAN)[0]["manifest"]["episode_id"],
                         Evaluator(clock=FakeClock()).reset(DEFAULT_SEED, CLEAN)[0]["manifest"]["episode_id"])

    def test_options_is_config_in_the_raw_protocol(self):
        by_config, by_options = (serve([json.dumps({"op": "reset", "seed": 4, key: {"errors": {}}})])[0]
                                 for key in ("config", "options"))
        self.assertEqual(by_config, by_options)
        self.assertEqual(by_options["observation"]["manifest"]["episode_id"],
                         generate(4, {"errors": {}})["public"]["manifest"]["episode_id"])

    def test_config_and_options_together_are_refused(self):
        replies = serve([json.dumps({"op": "reset", "seed": 4, "config": CLEAN, "options": CLEAN})])
        self.assertEqual(replies[0]["observation"]["error"]["code"], "bad_config")
        with Env() as env:
            with self.assertRaises(EnvError) as caught:
                env.reset(4, CLEAN, options=CLEAN)
            self.assertEqual(caught.exception.code, "bad_config")


class JournalTest(unittest.TestCase):
    def test_an_abandoned_episode_leaves_one_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "r.jsonl")
            evaluator = Evaluator(mode="eval", results_path=path, clock=FakeClock())
            first = evaluator.reset(1, CLEAN)[0]["manifest"]["episode_id"]
            evaluator.step(FX_PAGE)
            second = evaluator.reset(2, CLEAN)[0]["manifest"]["episode_id"]
            evaluator.step({"action": "submit", "findings": []})
            evaluator.reset(3, CLEAN)  # after the end: nothing to abandon
            evaluator.reset(4, CLEAN)  # no step in episode 3: nothing to abandon either
            lines = journal(path)
        self.assertEqual([(line["episode_id"], line["event"]) for line in lines],
                         [(first, "abandoned"), (second, "finished")])
        self.assertEqual((lines[0]["inspection_steps"], lines[0]["reward"]), (1, None))

    @unittest.skipUnless(os.name == "posix", "POSIX file types")
    def test_symlink_and_fifo_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            victim = root / "victim.txt"
            victim.write_text("keep\n")
            (root / "link.jsonl").symlink_to(victim)
            os.mkfifo(root / "fifo.jsonl")
            for name in ("link.jsonl", "fifo.jsonl"):
                with self.assertRaises(EnvError, msg=name):
                    Env(mode="eval", results_path=root / name)
                with self.assertRaises(OSError, msg=name):
                    Evaluator(mode="eval", results_path=root / name, clock=FakeClock())
            self.assertEqual(victim.read_text(), "keep\n")

    @unittest.skipUnless(os.name == "posix", "descriptor numbers")
    def test_the_journal_is_never_a_standard_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            results = pathlib.Path(directory) / "r.jsonl"
            script = ("import os, sys\nos.close(1)\nos.close(0)\nfrom auditgym import Env\n"
                      "env = Env.from_files(sys.argv[1], mode='eval', results_path=sys.argv[2])\nenv.reset()\n"
                      "o, r, t, u, i = env.step({'action': 'submit', 'findings': {'episode_id': 'ep_tiny0001',"
                      " 'findings': []}})\nsys.stderr.write(repr((o['type'], r, t)))\nenv.close()\n")
            done = subprocess.run([sys.executable, "-c", script, str(TINY), str(results)], capture_output=True,
                                  env=dict(os.environ, PYTHONPATH=str(ROOT)), timeout=120)
            self.assertEqual(done.stderr.decode().strip().splitlines()[-1], "('final', 0.0, True)")
            self.assertEqual(len(journal(results)), 1)
            self.assertEqual(stat.S_IMODE(results.stat().st_mode), 0o600)
        code, _ = serve([], "--results-fd", "1", code=True)
        self.assertEqual(code, 2)


class TrajectoryFileTest(unittest.TestCase):
    def test_shipped_trajectories_are_what_the_environment_records_now(self):
        """CASE §09: trajectories.jsonl must belong to the shipped train episodes and the current
        environment. The episode ids alone do not show a stale file; the bytes do (recording is
        deterministic). After `make episodes` or a change of the environment: `make trajectories`."""
        import contextlib
        import io

        from tools import trajectories

        shipped = ROOT / trajectories.DEFAULT_OUT
        self.assertTrue(shipped.is_file(), "trajectories.jsonl is part of the hand-in")
        with tempfile.TemporaryDirectory() as directory:
            fresh = pathlib.Path(directory) / "trajectories.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(trajectories.main(["--out", str(fresh)]), 0)
            self.assertEqual(fresh.read_bytes(), shipped.read_bytes(), "run `make trajectories`")

    def test_training_scan_finds_truth_ids_in_every_submit_form(self):
        from tools.trajectories import _submitted_ids

        findings = CORRECT_TINY["findings"]
        expected = set(findings[0]["affected_ids"]) | set(findings[0]["evidence_ids"])
        forms = [{"action": "submit", "findings": CORRECT_TINY},
                 {"action": "submit", "findings": findings},
                 {"action": "submit", "episode_id": "ep_tiny0001", "findings": findings},
                 {"action": "submit", "findings": json.dumps(CORRECT_TINY)},
                 {"tool": "submit", "arguments": {"findings.json": CORRECT_TINY}},
                 json.dumps({"action": "submit", "findings": findings})]
        for form in forms:
            self.assertEqual(_submitted_ids(form), expected, form)


class BlindPositionTest(unittest.TestCase):
    """E8: documents picked by position, by id order or at random hit errors equally often. Pages
    follow the order of the files, which the generator sorts by public id (AUDIT R6); before that
    they followed the document date, and the first and last invoices were the cut-off errors."""

    SEEDS = range(8_000_001, 8_000_005)

    @staticmethod
    def _pages(evaluator, table):
        records, cursor = [], None
        while True:
            observation = evaluator.step({"action": "list_records", "table": table, "cursor": cursor})[0]
            records += observation["records"]
            cursor = observation["next_cursor"]
            if cursor is None:
                return records

    def test_position_finds_errors_only_by_chance(self):
        """The first two and the last two invoices of the pages hit affected documents no more often
        than four invoices drawn at random: compared with the exact expectation of a random draw
        (hypergeometric mean and variance per episode), three standard deviations of slack."""
        hits, expected, variance = 0, 0.0, 0.0
        for label, config in metamorph.supported_configs():
            for seed in self.SEEDS:
                evaluator = Evaluator(clock=FakeClock())
                evaluator.reset(seed, config)
                affected = {i for issue in evaluator._episode["private"]["truth"]["issues"]
                            for i in issue["affected_ids"]}
                invoices = self._pages(evaluator, "invoices")
                n, k = len(invoices), min(4, len(invoices))
                share = sum(invoice["doc_id"] in affected for invoice in invoices) / n
                chosen = invoices[:2] + invoices[-2:] if n > 4 else invoices
                hits += sum(invoice["doc_id"] in affected for invoice in chosen)
                expected += k * share
                variance += k * share * (1 - share) * (n - k) / max(n - 1, 1)
        self.assertLessEqual(hits, expected + 3 * math.sqrt(variance) + 1,
                             f"{hits} affected among positional picks, {expected:.1f} expected by chance")

if __name__ == "__main__":
    unittest.main()
