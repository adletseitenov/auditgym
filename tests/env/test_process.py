"""CASE §03: «Оценщик работает вне процесса агента; записи доступны только на чтение через узкий
интерфейс инструментов. Эталон не должен утекать через наблюдения, пути, ошибки, debug info или
объяснения награды». CASE §04: воспроизводимость и изоляция через процесс-оценщик."""
import gc
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

from auditgym import Env, EnvError
from auditgym.evaluator import Evaluator
from auditgym.generate import generate
from auditgym.storage import write_episode
from tests.env.helpers import (CLEAN, ROOT, TINY, FakeClock, assert_no_leak, full_walk, strict_json)

CORRECT_TINY = {"episode_id": "ep_tiny0001", "findings": [{
    "type": "duplicate_invoice", "affected_ids": ["k7m2qa", "z5c1ty"], "claimed_amount_minor": 5000000,
    "currency": "KZT", "evidence_ids": ["k7m2qa", "z5c1ty"], "explanation": "Same customer and total."}]}


class _EnvAdapter:
    """Lets helpers.full_walk drive an Env; it needs the episode id for the final submit."""

    def __init__(self, env, episode_id):
        self.env = env
        self._episode = {"public": {"manifest": {"episode_id": episode_id}}}

    def step(self, action):
        return self.env.step(action)


def env_walk(env, first):
    return full_walk(_EnvAdapter(env, first["manifest"]["episode_id"]))


def serve_transcript(lines, hashseed):
    environment = dict(os.environ, PYTHONHASHSEED=str(hashseed), PYTHONPATH=str(ROOT))
    data = b"".join(line if isinstance(line, bytes) else line.encode() for line in lines)
    result = subprocess.run([sys.executable, "-m", "auditgym", "serve"], input=data,
                            capture_output=True, env=environment, cwd=str(ROOT), timeout=60)
    return result.stdout


def reachable(root, limit=200000):
    """Objects reachable from `root`, not following modules, classes, functions or code."""
    skip = (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType, types.CodeType,
            types.MethodType, types.FrameType)
    seen, stack, found = set(), [root], []
    while stack and len(seen) < limit:
        current = stack.pop()
        if id(current) in seen or isinstance(current, skip):
            continue
        seen.add(id(current))
        found.append(current)
        stack.extend(gc.get_referents(current))
        if hasattr(current, "__dict__") and not isinstance(current, skip):
            stack.append(vars(current))
    return found


class ProcessTest(unittest.TestCase):
    def test_env_matches_the_in_process_evaluator(self):
        evaluator = Evaluator(clock=FakeClock())
        expected = [evaluator.load(TINY)] + full_walk(evaluator)
        with Env.from_files(TINY) as env:
            first = env.reset()
            got = [first] + env_walk(env, first[0])
        self.assertEqual(strict_json(got), strict_json(expected))

    def test_generated_episode_through_the_process(self):
        evaluator = Evaluator(clock=FakeClock())
        expected = [evaluator.reset(21, CLEAN)] + full_walk(evaluator)
        with Env() as env:
            first = env.reset(21, CLEAN)
            got = [first] + env_walk(env, first[0])
        self.assertEqual(strict_json(got), strict_json(expected))

    def test_reset_returns_two_and_step_five_values(self):
        with Env.from_files(TINY) as env:
            self.assertEqual(len(env.reset()), 2)
            self.assertEqual(len(env.step({"action": "list_records", "table": "fx"})), 5)

    def test_step_before_reset_through_the_process(self):
        with Env() as env:
            observation, reward, terminated, truncated, info = env.step({"action": "list_records", "table": "fx"})
            self.assertEqual(observation["error"]["code"], "not_reset")
            self.assertEqual((reward, terminated, info), (0.0, True, {}))

    def test_refused_reset_raises_a_fixed_code(self):
        with Env() as env:
            for seed, config in ((1, {"nope": 1}), ("1", None), (1, {"mode": "eval"})):
                with self.assertRaises(EnvError) as caught:
                    env.reset(seed, config)
                self.assertEqual(caught.exception.code, "bad_config")
            with self.assertRaises(EnvError) as caught:
                env.reset()
            self.assertEqual(caught.exception.code, "not_loaded")

    def test_bad_directory_is_a_fixed_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(EnvError) as caught:
                Env.from_files(directory)
            self.assertEqual(caught.exception.code, "bad_episode")

    def test_non_json_action_is_an_unknown_action(self):
        with Env.from_files(TINY) as env:
            env.reset()
            observation = env.step({"action": object()})[0]
            self.assertEqual((observation["error"]["code"], observation["steps_remaining"]), ("unknown_action", 99))

    def test_close_stops_the_server(self):
        env = Env()
        process = env._process
        env.close()
        self.assertIsNotNone(process.returncode)
        with self.assertRaises(EnvError):
            env.step({"action": "list_records"})
        env.close()


class EvalModeProcessTest(unittest.TestCase):
    """«Только в открытом режиме обучения терминальная награда возвращается политике».
    Режим задаётся при запуске оценщика и не меняется ни конфигурацией, ни действиями."""

    def test_eval_returns_zero_and_writes_results(self):
        with tempfile.TemporaryDirectory() as directory:
            results = pathlib.Path(directory) / "results.jsonl"
            with Env.from_files(TINY, mode="eval", results_path=results) as env:
                env.reset()
                env.step({"action": "list_records", "table": "ledger"})
                self.assertEqual(env.step({"action": "submit", "findings": CORRECT_TINY, "mode": "train"}),
                                 ({"type": "final", "steps_remaining": 98}, 0.0, True, False, {}))
                with self.assertRaises(EnvError):
                    env.reset(1, {"mode": "train", "errors": {}})
            lines = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(lines[0]["reward"], 1 - 0.002)
        self.assertEqual(lines[0]["mode"], "eval")

    def test_train_returns_the_terminal_reward(self):
        with Env.from_files(TINY) as env:
            env.reset()
            _, reward, terminated, _, info = env.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual((reward, terminated), (1.0, True))
        self.assertEqual(set(info), {"reward", "precision", "recall", "f1", "inspection_steps"})


class ProtocolTest(unittest.TestCase):
    """CONTRACTS §5.1: stdout carries only protocol lines; malformed lines never crash the server."""

    def test_malformed_lines_get_an_error_reply_and_the_server_survives(self):
        lines = [
            b"not json\n", b"[1, 2]\n", b'{"op": 5}\n', b'{"op": "explode"}\n', b'{"no_op": true}\n',
            b'"string"\n', b"\xff\xfe{}\n", b"[" * 100000 + b"\n",
            b'{"op": "step", "action": ' + b"1" * 5000 + b"}\n",
            b"x" * (1 << 21) + b"\n", b'{"op": "load", "directory": "/"}\n',
            b'{"op": "reset", "seed": 2, "config": {"errors": {}}}\n',
            b'{"op": "step", "action": {"action": "list_records", "table": "fx", "limit": 2}}\n',
            b'{"op": "reset", "seed": 2, "config": {"errors": {}}, "extra": NaN}\n',
            b'{"op": "close"}\n', b'{"op": "step"}\n',
        ]
        output = serve_transcript(lines, 0).decode("ascii").splitlines()
        replies = [json.loads(line) for line in output]
        self.assertEqual(len(replies), 14)  # one per request up to close
        for reply in replies[:11]:
            self.assertEqual(reply["observation"], {"type": "error", "steps_remaining": 0,
                                                    "error": {"code": "bad_request", "message": "Bad request."}})
            self.assertEqual((reply["reward"], reply["terminated"], reply["info"]), (0.0, False, {}))
        self.assertEqual(replies[11]["observation"]["type"], "reset")
        self.assertEqual(replies[12]["observation"]["type"], "records")
        self.assertEqual(replies[12]["observation"]["steps_remaining"], 99)
        self.assertEqual(replies[13]["observation"]["type"], "reset")

    def test_load_is_accepted_only_as_the_first_request(self):
        lines = ['{"op": "reset", "seed": 2, "config": {"errors": {}}}\n',
                 json.dumps({"op": "load", "directory": str(TINY)}) + "\n"]
        replies = [json.loads(line) for line in serve_transcript(lines, 0).splitlines()]
        self.assertEqual(replies[1]["observation"]["error"]["code"], "bad_request")

    def test_two_processes_with_different_hash_seeds_give_identical_bytes(self):
        lines = [json.dumps({"op": "reset", "seed": 31, "config": CLEAN}) + "\n"]
        cursor_steps = [{"action": "list_records", "table": t, "limit": 50} for t in ("ledger", "invoices", "payments", "fx")]
        for action in cursor_steps + [{"action": "get_record", "table": "fx", "id": "2026-10-01|USD|KZT"},
                                      {"action": "calculator", "expression": "1/7"},
                                      {"tool": "nope"}, {"action": "submit", "findings": {"episode_id": "x"}}]:
            lines.append(json.dumps({"op": "step", "action": action}) + "\n")
        transcripts = {serve_transcript(lines, hashseed) for hashseed in (0, 1, 4242, "random")}
        self.assertEqual(len(transcripts), 1)
        self.assertEqual(len(next(iter(transcripts)).splitlines()), len(lines))


class NoLeakTest(unittest.TestCase):
    """«Агент не видит эти файлы и seed.» «Эталон, provenance и seed живут только на стороне оценщика:
    не в наблюдениях, info, текстах ошибок, именах и путях, траекториях.»"""

    def test_full_walk_on_tiny_leaks_nothing(self):
        from auditgym.storage import load_episode

        episode = load_episode(TINY)
        with Env.from_files(TINY) as env:
            first = env.reset()
            values = [first] + env_walk(env, first[0])
        assert_no_leak(self, values, episode, extra_words=(str(TINY), "fixtures/"))

    def test_full_walk_on_generated_episodes_leaks_nothing(self):
        for seed in (48151623, 90210377, 3141592653):
            episode = generate(seed, CLEAN)
            with Env() as env:
                first = env.reset(seed, CLEAN)
                values = [first] + env_walk(env, first[0])
            assert_no_leak(self, values, episode)

    def test_files_episode_leaks_no_path(self):
        episode = generate(48151623, CLEAN)
        with tempfile.TemporaryDirectory(prefix="secretdir_") as directory:
            write_episode(episode, directory)
            with Env.from_files(directory) as env:
                first = env.reset()
                values = [first] + env_walk(env, first[0])
        assert_no_leak(self, values, episode, extra_words=(directory, pathlib.Path(directory).name, "secretdir"))

    def test_eval_mode_leaks_nothing_either(self):
        episode = generate(48151623, CLEAN)
        with tempfile.TemporaryDirectory() as directory:
            with Env(mode="eval", results_path=pathlib.Path(directory) / "r.jsonl") as env:
                first = env.reset(48151623, CLEAN)
                values = [first] + env_walk(env, first[0])
        assert_no_leak(self, values, episode, extra_words=(directory, "r.jsonl"))
        for _, reward, _, _, info in values[1:]:
            self.assertEqual((reward, info), (0.0, {}))

    def test_client_object_holds_no_private_data(self):
        seed = 77777331
        episode = generate(seed, CLEAN)
        with tempfile.TemporaryDirectory(prefix="hiddenresults_") as directory:
            results = pathlib.Path(directory) / "results.jsonl"
            with tempfile.TemporaryDirectory(prefix="hiddenepisode_") as episode_dir:
                write_episode(episode, episode_dir)
                for env in (Env(mode="eval", results_path=results), Env.from_files(episode_dir, results_path=results)):
                    with env:
                        if env.mode == "eval":
                            env.reset(seed, CLEAN)
                        else:
                            env.reset()
                        env.step({"action": "list_records", "table": "ledger"})
                        self.assertEqual(set(vars(env)), {"mode", "_process"})
                        objects = reachable(env)
                        strings = [o for o in objects if isinstance(o, (str, bytes))]
                        joined = "\n".join(s if isinstance(s, str) else s.decode("latin-1") for s in strings)
                        for word in ("hiddenresults_", "hiddenepisode_", "results.jsonl", "truth", "provenance",
                                     "pre_ledger", "private"):
                            self.assertNotIn(word, joined, word)
                        self.assertNotIn(seed, [o for o in objects if type(o) is int])
                        self.assertFalse(any(isinstance(o, Evaluator) for o in objects))
        # And nothing in this process ever loaded the episode's truth through the client.
        self.assertFalse(any(isinstance(o, Evaluator) and o._episode is not None and
                             o._episode["private"]["provenance"].get("seed") == seed for o in gc.get_objects()))


if __name__ == "__main__":
    unittest.main()
