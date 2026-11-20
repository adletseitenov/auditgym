"""Audit fixes for the environment (docs/AUDIT.md C1-C10, R1, R9, R10, L3, L4, C5, Q2) and the
survivors of the mutation run (Q1: N1-N9, A2). Each test pins a behaviour that was wrong, or
that no test noticed when it was changed."""
import io
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from auditgym import Env, EnvError, actions, serve
from auditgym.actions import ActionError, parse_action
from auditgym.evaluator import DEFAULT_SEED, Evaluator, EvaluatorError, MonotonicClock
from tests.env.helpers import TINY, FakeClock, strict_json, tiny
from tests.env.test_evaluator import CORRECT_TINY

FX_PAGE = {"action": "list_records", "table": "fx"}


def final(result, least=0.9):
    """The step result is the terminal one of a finished episode with a full-credit answer."""
    observation, got, terminated, truncated, _ = result
    return observation["type"] == "final" and terminated and not truncated and got >= least


class SubmitFormsTest(unittest.TestCase):
    """C1, R7: every reasonable way to submit the same answer is scored the same."""

    FINDINGS = CORRECT_TINY["findings"]

    def forms(self):
        wrapper = CORRECT_TINY
        return {
            "documented": {"action": "submit", "findings": wrapper},
            "bare list": {"action": "submit", "findings": self.FINDINGS},
            "flat episode_id and findings": {"action": "submit", "episode_id": "ep_tiny0001",
                                             "findings": self.FINDINGS},
            "findings as a JSON string (list)": {"action": "submit", "findings": json.dumps(self.FINDINGS)},
            "findings as a JSON string (message)": {"action": "submit", "findings": json.dumps(wrapper)},
            "inner findings as a JSON string": {"action": "submit", "findings": {
                "episode_id": "ep_tiny0001", "findings": json.dumps(self.FINDINGS)}},
            "flat with findings as a JSON string": {"action": "submit", "episode_id": "ep_tiny0001",
                                                    "findings": json.dumps(self.FINDINGS)},
            "findings.json under arguments": {"action": "submit", "arguments": {"findings.json": wrapper}},
            "findings.json as a string under args": {"tool": "submit", "args": {
                "findings.json": json.dumps(wrapper)}},
            "arguments with the flat form": {"name": "submit", "arguments": {
                "episode_id": "ep_tiny0001", "findings": self.FINDINGS}},
            "input container": {"type": "submit", "input": {"findings": wrapper}},
            "upper-case name": {"action": "SUBMIT", "findings": wrapper},
            "action as a JSON string": json.dumps({"action": "submit", "findings": wrapper}),
        }

    def test_every_form_is_scored_as_the_documented_one(self):
        for name, action in self.forms().items():
            with self.subTest(name):
                self.assertTrue(final(tiny().step(action)), name)

    def test_forms_agree_through_the_process(self):
        with Env.from_files(TINY) as env:
            for name, action in self.forms().items():
                with self.subTest(name):
                    env.reset()
                    self.assertTrue(final(env.step(action)), name)

    def test_a_wrong_episode_id_is_still_minus_one(self):
        for action in ({"action": "submit", "episode_id": "ep_other", "findings": self.FINDINGS},
                       {"action": "submit", "findings": {"episode_id": "ep_other", "findings": self.FINDINGS}},
                       {"action": "submit", "findings": {"findings": self.FINDINGS}}):
            _, reward, terminated, _, _ = tiny().step(action)
            self.assertEqual((reward, terminated), (-1.0, True), action)

    def test_bare_list_is_the_current_episode_but_an_object_without_episode_id_is_minus_one(self):
        """Documented in CONTRACTS §5.2 and README: [..] is findings of this episode (also []); an object
        without `episode_id` is a message missing a required field (CASE: invalid structure, -1)."""
        bare = tiny().step({"action": "submit", "findings": self.FINDINGS})
        documented = tiny().step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual((bare[1], bare[2]), (documented[1], documented[2]))
        self.assertGreater(bare[1], 0.9)
        no_id = tiny().step({"action": "submit", "findings": {"findings": self.FINDINGS}})
        self.assertEqual((no_id[1], no_id[2]), (-1.0, True))
        for answer, reward in (([], 1.0), ({"findings": []}, -1.0)):  # clean books: silence vs a broken message
            clean = Evaluator(clock=FakeClock())
            clean.reset(3, {"errors": {}})
            self.assertEqual(clean.step({"action": "submit", "findings": answer})[1], reward, answer)
        with Env.from_files(TINY) as env:  # the same through the process
            self.assertGreater(env.step({"action": "submit", "findings": self.FINDINGS})[1], 0.9)
            env.reset()
            self.assertEqual(env.step({"action": "submit", "findings": {"findings": self.FINDINGS}})[1], -1.0)

    def test_broken_forms_are_invalid_structure(self):
        for findings in ("not json", "[1, 2", 7, True, {"episode_id": "ep_tiny0001"}, [5],
                         json.dumps({"episode_id": "ep_tiny0001", "findings": "[oops"})):
            _, reward, terminated, _, _ = tiny().step({"action": "submit", "findings": findings})
            self.assertEqual((reward, terminated), (-1.0, True), findings)

    def test_empty_bare_list_is_silence_not_a_structure_error(self):
        evaluator = tiny()
        _, reward, terminated, _, info = evaluator.step({"action": "submit", "findings": []})
        self.assertEqual((terminated, info["f1"], info["recall"]), (True, 0.0, 0.0))
        clean = Evaluator(clock=FakeClock())
        clean.reset(3, {"errors": {}})
        self.assertEqual(clean.step({"action": "submit", "findings": []})[1], 1.0)  # silence on clean books

    def test_deeply_nested_string_does_not_crash(self):
        deep = "[" * 100000
        for findings in (deep, '{"findings": ' + deep):
            self.assertEqual(tiny().step({"action": "submit", "findings": findings})[1], -1.0)


class TableAndArgumentTolerance(unittest.TestCase):
    """C2, C3: names of tables and arguments are read generously; unknown ids and actions stay strict."""

    def test_table_names(self):
        for name in ("ledger", "Ledger", "LEDGER", "ledger.csv", "Ledger.CSV", " ledger ", "invoice",
                     "invoices.jsonl", "Invoice.jsonl", "payment", "payments", "PAYMENTS.jsonl",
                     "fx", "FX", "fx.csv"):
            observation = tiny().step({"action": "list_records", "table": name})[0]
            self.assertEqual(observation["type"], "records", name)
        for name in ("ledger.json", "ledgers.csv", "invoice.csv.csv", "", " ", "fx_rates", "../ledger",
                     "ledger/", None, 5, ["ledger"]):
            observation = tiny().step({"action": "list_records", "table": name})[0]
            self.assertEqual(observation["error"]["code"], "unknown_table", name)

    def test_table_name_in_get_record(self):
        evaluator = tiny()
        self.assertEqual(evaluator.step({"action": "get_record", "table": "Ledger.csv", "id": "a5"})[0]["table"],
                         "ledger")
        self.assertEqual(evaluator.step({"action": "get_record", "table": "invoice", "id": "k7m2qa"})[0]["table"],
                         "invoices")

    def test_get_record_id_synonyms(self):
        for key in ("id", "record_id", "row_id", "doc_id", "invoice_id", "payment_id"):
            observation = tiny().step({"action": "get_record", "table": "ledger", key: "a5"})[0]
            self.assertEqual(observation["record"]["row_id"], "a5", key)
        # the canonical key wins over a synonym; a null value does not count
        observation = tiny().step({"action": "get_record", "table": "ledger", "id": "a5", "row_id": "b2"})[0]
        self.assertEqual(observation["record"]["row_id"], "a5")
        observation = tiny().step({"action": "get_record", "table": "ledger", "id": None, "row_id": "b2"})[0]
        self.assertEqual(observation["record"]["row_id"], "b2")

    def test_get_document_id_synonyms(self):
        for key in ("doc_id", "id", "document_id"):
            observation = tiny().step({"action": "get_document", key: "k7m2qa"})[0]
            self.assertEqual(observation["record"]["doc_id"], "k7m2qa", key)
        self.assertEqual(tiny().step({"action": "get_document", "invoice_id": "p3x9wd"})[0]["error"]["code"],
                         "bad_arguments")

    def test_unknown_ids_and_actions_stay_strict_and_spend_a_step(self):
        evaluator = tiny()
        for action, code in (({"action": "get_record", "table": "ledger", "record_id": "nope"}, "unknown_id"),
                             ({"action": "get_document", "document_id": "nope"}, "unknown_id"),
                             ({"action": "get_document", "id": "a5"}, "unknown_id"),
                             ({"action": "list_recordz"}, "unknown_action"),
                             ({"action": "list records", "table": "ledger"}, "unknown_action"),
                             ({"action": "get_record", "table": "ledger", "id": 5}, "bad_arguments")):
            observation = evaluator.step(action)[0]
            self.assertEqual(observation["error"]["code"], code, action)
        self.assertEqual(evaluator.steps_remaining(), 94)

    def test_cursor_forms_for_the_first_page(self):
        reference = tiny().step({"action": "list_records", "table": "ledger", "limit": 5})[0]
        for cursor in (None, "", 0, "0"):
            observation = tiny().step({"action": "list_records", "table": "ledger", "limit": 5,
                                       "cursor": cursor})[0]
            self.assertEqual(observation, reference, cursor)
        for cursor in (False, 0.0, "00", " 0", [], {}, 1, -1):
            observation = tiny().step({"action": "list_records", "table": "ledger", "cursor": cursor})[0]
            self.assertEqual(observation["error"]["code"], "bad_cursor", cursor)

    def test_limit_forms(self):
        for limit, count in (("5", 5), (5.0, 5), ("5.0", 5), (" 7 ", 7), ("50", 12), (50.0, 12), (None, 12),
                             (1, 1), ("1", 1), (1000, 12), ("9" * 5000, 12)):
            observation = tiny().step({"action": "list_records", "table": "ledger", "limit": limit})[0]
            self.assertEqual(len(observation["records"]), count, limit)

    def test_action_as_a_json_string(self):
        evaluator = tiny()
        text = json.dumps({"action": "get_record", "table": "ledger", "id": "a5"})
        self.assertEqual(evaluator.step(text)[0]["record"]["row_id"], "a5")
        for bad in ("{", "[]", "null", "5", '"list_records"', "list_records"):
            self.assertEqual(evaluator.step(bad)[0]["error"]["code"], "unknown_action", bad)

    def test_action_name_ignores_case_and_spaces(self):
        for name in ("LIST_RECORDS", "List_Records", " list_records "):
            for key in ("action", "tool", "name", "type"):
                self.assertEqual(tiny().step({key: name, "table": "fx"})[0]["type"], "records", (key, name))

    def test_argument_containers(self):
        expected = ("get_record", {"table": "ledger", "id": "a5"})
        for container in ("args", "arguments", "params", "input", "parameters"):
            self.assertEqual(parse_action({"action": "get_record", container: {"table": "ledger", "id": "a5"}}),
                             expected, container)
            self.assertEqual(parse_action({"action": "get_record",
                                           container: json.dumps({"table": "ledger", "id": "a5"})}), expected)

    def test_tool_use_block_of_an_llm_api(self):
        action = {"type": "tool_use", "id": "toolu_1", "name": "get_record",
                  "input": {"table": "ledger", "id": "a5"}}
        self.assertEqual(tiny().step(action)[0]["record"]["row_id"], "a5")

    def test_nested_arguments_win_over_flat_ones(self):
        """A2: the documented precedence is flat < args < arguments < params < input < parameters."""
        name, arguments = parse_action({"action": "get_record", "table": "ledger", "id": "flat",
                                        "args": {"id": "args"}})
        self.assertEqual(arguments["id"], "args")
        name, arguments = parse_action({"action": "get_record", "table": "ledger", "id": "flat",
                                        "args": {"id": "args"}, "arguments": {"id": "arguments"},
                                        "params": {"id": "params"}, "input": {"id": "input"},
                                        "parameters": {"id": "parameters"}})
        self.assertEqual(arguments["id"], "parameters")
        for winner, containers in (("arguments", ("args", "arguments")), ("params", ("params", "arguments")),
                                   ("input", ("input", "params")), ("parameters", ("parameters", "input"))):
            action = {"action": "get_record", "table": "ledger"}
            action.update({c: {"id": c} for c in containers})
            self.assertEqual(parse_action(action)[1]["id"], winner, containers)
        observation = tiny().step({"action": "get_record", "table": "ledger", "id": "b2", "args": {"id": "a5"}})[0]
        self.assertEqual(observation["record"]["row_id"], "a5")

    def test_name_must_be_a_known_action_in_some_key(self):
        self.assertEqual(parse_action({"type": "function", "name": "calculator", "expression": "1+1"})[0],
                         "calculator")
        with self.assertRaises(ActionError) as caught:
            parse_action({"type": "function", "name": "rm"})
        self.assertEqual(caught.exception.code, "unknown_action")


class MalformedSubmitTest(unittest.TestCase):
    """C9: a submit whose wrapper is broken ends the episode as an invalid structure."""

    def test_malformed_wrapper_ends_the_episode(self):
        for action in ({"action": "submit", "args": 5}, {"action": "submit", "arguments": "not json"},
                       {"tool": "submit", "params": [1, 2]}, {"name": "SUBMIT", "input": "7"},
                       {"action": "submit", "parameters": '["findings"]'}):
            evaluator = tiny()
            observation, reward, terminated, truncated, info = evaluator.step(action)
            self.assertEqual((observation["type"], reward, terminated, truncated), ("final", -1.0, True, False), action)
            self.assertEqual(info["inspection_steps"], 0)
            self.assertEqual(evaluator.step(FX_PAGE)[0]["error"]["code"], "episode_over")

    def test_malformed_wrapper_is_in_the_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, clock=FakeClock())
            evaluator.load(TINY)
            evaluator.step({"action": "submit", "args": 5})
            line = json.loads(path.read_text())
        self.assertEqual((line["reward"], line["terminated"], line["structure_error"]), (-1.0, True, "missing_field"))

    def test_other_malformed_wrappers_stay_an_error_step(self):
        evaluator = tiny()
        observation = evaluator.step({"action": "list_records", "args": 5})[0]
        self.assertEqual((observation["error"]["code"], observation["steps_remaining"]), ("bad_arguments", 99))


class TruncationConsistencyTest(unittest.TestCase):
    """C10, N4, N5."""

    def truncate_by_steps(self, evaluator):
        for _ in range(100):
            result = evaluator.step(FX_PAGE)
        return result

    def test_every_step_after_a_truncation_says_truncated(self):
        evaluator = tiny()
        self.assertEqual(self.truncate_by_steps(evaluator)[2:4], (False, True))
        for action in (FX_PAGE, {"action": "submit", "findings": CORRECT_TINY}, None, "x"):
            observation, reward, terminated, truncated, info = evaluator.step(action)
            self.assertEqual(observation["error"]["code"], "episode_over")
            self.assertEqual((reward, terminated, truncated, info), (0.0, False, True, {}), action)
            self.assertEqual(observation["steps_remaining"], 0)

    def test_after_a_submit_the_steps_still_say_terminated(self):
        evaluator = tiny()
        evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual(evaluator.step(FX_PAGE)[2:4], (True, False))

    def test_time_truncation_is_consistent_too(self):
        clock = FakeClock()
        evaluator = Evaluator(clock=clock)
        evaluator.load(TINY)
        evaluator.step(FX_PAGE)
        clock.t += 60
        observation, reward, terminated, truncated, info = evaluator.step(FX_PAGE)
        self.assertEqual(observation, {"type": "final", "steps_remaining": 0})
        self.assertEqual((reward, terminated, truncated), (-1.0, False, True))
        observation, reward, terminated, truncated, _ = evaluator.step(FX_PAGE)
        self.assertEqual((observation["steps_remaining"], terminated, truncated), (0, False, True))

    def test_eval_mode_time_truncation_hides_the_reward_and_logs_it(self):
        """N4: truncation by time in eval mode returns zero, no info, and writes the true result."""
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, clock=clock)
            evaluator.load(TINY)
            clock.t += 61
            result = evaluator.step({"action": "submit", "findings": CORRECT_TINY})
            line = json.loads(path.read_text())
        self.assertEqual(result, ({"type": "final", "steps_remaining": 0}, 0.0, False, True, {}))
        self.assertEqual((line["reward"], line["truncated"], line["terminated"], line["inspection_steps"]),
                         (-1.0, True, False, 0))

    def test_an_error_action_on_step_100_truncates(self):
        """N5: the budget is spent by any action, an error included."""
        evaluator = tiny()
        for _ in range(99):
            evaluator.step({"action": "nope"})
        observation, reward, terminated, truncated, info = evaluator.step({"action": "nope"})
        self.assertEqual((observation["type"], terminated, truncated, reward), ("final", False, True, -1.0))
        self.assertEqual(info["inspection_steps"], 100)

    def test_failed_reset_keeps_the_episode_through_the_process(self):
        with Env.from_files(TINY) as env:
            env.reset()
            env.step(FX_PAGE)
            with self.assertRaises(EnvError):
                env.reset(1, {"nope": 1})
            with self.assertRaises(EnvError):
                env.reset(True)
            observation = env.step(FX_PAGE)[0]
            self.assertEqual(observation["steps_remaining"], 98)


class ClockTest(unittest.TestCase):
    """N1: the default clock is the monotonic one, and the limit is read from the injected clock."""

    def test_default_clock_is_time_monotonic(self):
        with mock.patch("time.monotonic", side_effect=[1000.0, 1001.0, 1059.9, 1060.0]) as monotonic:
            evaluator = Evaluator()  # public constructor, no clock
            self.assertIsInstance(evaluator._clock, MonotonicClock)
            evaluator.load(TINY)                                    # started_at = 1000.0
            self.assertFalse(evaluator.step(FX_PAGE)[3])            # 1001.0
            self.assertFalse(evaluator.step(FX_PAGE)[3])            # 1059.9: 59.9 s elapsed
            self.assertTrue(evaluator.step(FX_PAGE)[3])             # 1060.0: 60 s elapsed
            self.assertEqual(monotonic.call_count, 4)

    def test_time_limit_is_a_constructor_argument(self):
        clock = FakeClock()
        evaluator = Evaluator(clock=clock, time_limit=5)
        evaluator.load(TINY)
        clock.t += 4
        self.assertFalse(evaluator.step(FX_PAGE)[3])
        clock.t += 1
        self.assertTrue(evaluator.step(FX_PAGE)[3])

    def test_default_time_limit_is_60_seconds(self):
        from auditgym.evaluator import DEFAULT_TIME_LIMIT

        self.assertEqual(DEFAULT_TIME_LIMIT, 60)

    def test_real_clock_runs_forward(self):
        clock = MonotonicClock()
        first = clock.now()
        self.assertGreaterEqual(clock.now(), first)


class LimitBoundariesTest(unittest.TestCase):
    """N2, N3, N7."""

    def test_limit_one_is_accepted(self):
        observation = tiny().step({"action": "list_records", "table": "ledger", "limit": 1})[0]
        self.assertEqual((observation["type"], len(observation["records"])), ("records", 1))
        self.assertIsNotNone(observation["next_cursor"])

    def test_limits_between_51_and_99_are_clipped_to_50(self):
        from auditgym.generate import generate

        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset(5, {"errors": {}})
        self.assertGreater(len(evaluator._tables["ledger"]), 99)
        for limit in (49, 50, 51, 52, 75, 99, 100, 101):
            observation = evaluator.step({"action": "list_records", "table": "ledger", "limit": limit})[0]
            self.assertEqual(len(observation["records"]), min(limit, 50), limit)

    def test_null_limit_is_the_default_page(self):
        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset(5, {"errors": {}})
        for action in ({"action": "list_records", "table": "ledger", "limit": None},
                       {"action": "list_records", "table": "ledger"}):
            self.assertEqual(len(evaluator.step(action)[0]["records"]), 50)

    def test_null_cursor_is_the_first_page(self):
        evaluator = tiny()
        self.assertEqual(evaluator.step({"action": "list_records", "table": "fx", "cursor": None})[0]["type"],
                         "records")


class JournalTest(unittest.TestCase):
    """N6, N8, L3, C5."""

    def journal(self, mode="eval"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = pathlib.Path(directory.name) / "r.jsonl"
        evaluator = Evaluator(mode=mode, results_path=path, clock=FakeClock())
        evaluator.load(TINY)
        return evaluator, path

    def test_terminated_and_truncated_are_in_the_journal(self):
        evaluator, path = self.journal()
        evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        again = Evaluator(mode="eval", results_path=path, clock=FakeClock())
        again.load(TINY)
        for _ in range(100):
            again.step(FX_PAGE)
        first, second = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual((first["terminated"], first["truncated"]), (True, False))
        self.assertEqual((second["terminated"], second["truncated"]), (False, True))
        self.assertEqual(set(first), {"episode_id", "mode", "event", "episode_run", "terminated", "truncated",
                                      "inspection_steps", "steps_total", "secret", "reward", "precision", "recall",
                                      "f1", "tp", "fp", "fn", "structure_error"})

    def test_judge_exception_is_minus_one_and_logged_never_a_win(self):
        for mode in ("eval", "train"):
            evaluator, path = self.journal(mode)
            with mock.patch("auditgym.judge.judge", side_effect=RuntimeError("boom")):
                observation, reward, terminated, truncated, info = evaluator.step(
                    {"action": "submit", "findings": CORRECT_TINY})
            line = json.loads(path.read_text())
            self.assertEqual(line["structure_error"], "judge_exception")
            self.assertEqual((line["reward"], line["tp"], line["fn"]), (-1.0, 0, 1))
            self.assertEqual((observation["type"], terminated, truncated), ("final", True, False))
            self.assertEqual(reward, -1.0 if mode == "train" else 0.0)
            if mode == "train":
                self.assertEqual(info["reward"], -1.0)
                self.assertNotIn("boom", strict_json([observation, info]))

    def test_judge_exception_on_truncation_and_on_a_broken_truth(self):
        evaluator, path = self.journal()
        evaluator._episode["private"] = {}  # even a broken episode cannot give a free win
        _, reward, terminated, _, _ = evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertTrue(terminated)
        self.assertEqual(json.loads(path.read_text())["reward"], -1.0)

    def test_results_file_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            Evaluator(mode="eval", results_path=path, clock=FakeClock())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            path.chmod(0o644)  # an old file with loose rights is tightened
            Evaluator(mode="eval", results_path=path, clock=FakeClock())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_results_file_through_the_client_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sub" / "r.jsonl"
            with Env(mode="eval", results_path=path) as env:
                env.reset(5, {"errors": {}})
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_a_write_failure_does_not_break_the_episode(self):
        evaluator, path = self.journal()
        os_path = str(path)
        os.remove(os_path)
        os.mkdir(os_path)  # the journal cannot be appended to any more
        observation, reward, terminated, _, _ = evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual((observation["type"], terminated), ("final", True))

    def test_default_journal_path_in_eval_mode(self):
        self.assertEqual(actions.DEFAULT_RESULTS_PATH, "runs/results.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            before = os.getcwd()
            os.chdir(directory)
            try:
                with Env(mode="eval") as env:
                    first, _ = env.reset(5, {"errors": {}})
                    env.step({"action": "submit", "findings": {"episode_id": first["manifest"]["episode_id"],
                                                               "findings": "broken"}})
                with Env(mode="train") as env:
                    env.reset(5, {"errors": {}})
                journal = pathlib.Path(directory, "runs", "results.jsonl")
                lines = [json.loads(line) for line in journal.read_text().splitlines()]
                mode = stat.S_IMODE(journal.stat().st_mode)
            finally:
                os.chdir(before)
        self.assertEqual(len(lines), 1)
        self.assertEqual((lines[0]["structure_error"], lines[0]["reward"], lines[0]["mode"]),
                         ("findings_not_list", -1.0, "eval"))
        self.assertEqual(mode, 0o600)

    def test_default_journal_path_for_the_serve_command(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).resolve().parents[2]))
            lines = [json.dumps({"op": "reset", "seed": 5, "config": {"errors": {}}}),
                     json.dumps({"op": "step", "action": {"action": "submit", "args": 3}})]
            subprocess.run([sys.executable, "-m", "auditgym", "serve", "--mode", "eval"],
                           input=("\n".join(lines) + "\n").encode(), cwd=directory, env=env,
                           capture_output=True, timeout=60, check=True)
            journal = pathlib.Path(directory, "runs", "results.jsonl")
            self.assertEqual(json.loads(journal.read_text())["reward"], -1.0)
            self.assertEqual(stat.S_IMODE(journal.stat().st_mode), 0o600)


class DefaultJournalEnvVarTest(unittest.TestCase):
    """C5: $AUDITGYM_RESULTS overrides the default journal path, for the client and for the server."""

    def test_env_var_wins_over_the_default(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory, "elsewhere", "journal.jsonl")
            before = os.environ.get(actions.RESULTS_ENV)
            os.environ[actions.RESULTS_ENV] = str(target)
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                self.assertEqual(actions.default_results_path(), str(target))
                with Env(mode="eval") as env:
                    first, _ = env.reset(5, {"errors": {}})
                    env.step({"action": "submit", "args": 3})
            finally:
                os.chdir(cwd)
                if before is None:
                    del os.environ[actions.RESULTS_ENV]
                else:
                    os.environ[actions.RESULTS_ENV] = before
            self.assertEqual(json.loads(target.read_text())["reward"], -1.0)
            self.assertFalse(pathlib.Path(directory, "runs").exists())


class ServerLineLimitTest(unittest.TestCase):
    """L4, N9: the line limit of the server is exact, and a long line during an episode costs a step."""

    LIMIT = serve.MAX_LINE_BYTES

    def padded(self, size, **fields):
        base = {"op": "step", "action": {"action": "calculator", "expression": "1+1", "pad": ""}}
        base["action"].update(fields)
        empty = len(json.dumps(base, separators=(",", ":")))
        base["action"]["pad"] = "x" * (size - empty)
        text = json.dumps(base, separators=(",", ":"))
        assert len(text) == size
        return text.encode()

    def run_lines(self, evaluator, lines):
        data = b"".join(line if line.endswith(b"\n") else line + b"\n" for line in lines)
        out = io.StringIO()
        serve.serve(evaluator, stdin=io.BytesIO(data), stdout=out)
        return [json.loads(line) for line in out.getvalue().splitlines()]

    def test_a_line_of_exactly_the_limit_is_served_and_one_more_byte_is_not(self):
        evaluator = tiny()
        replies = self.run_lines(evaluator, [self.padded(self.LIMIT), self.padded(self.LIMIT + 1),
                                             self.padded(self.LIMIT - 1)])
        self.assertEqual([r["observation"]["type"] for r in replies], ["calculation", "error", "calculation"])
        self.assertEqual(replies[0]["observation"]["result"], "2")
        # the long line is an error step during the episode: 3 requests, 3 steps
        self.assertEqual(replies[1]["observation"]["error"]["code"], "bad_arguments")
        self.assertEqual([r["observation"]["steps_remaining"] for r in replies], [99, 98, 97])
        self.assertFalse(evaluator.running)  # the end of input truncated the episode (audit A2)

    def test_a_long_line_without_an_episode_is_a_bad_request_and_spends_nothing(self):
        evaluator = Evaluator(clock=FakeClock())
        [reply] = self.run_lines(evaluator, [self.padded(self.LIMIT + 1)])
        self.assertEqual(reply["observation"]["error"]["code"], "bad_request")
        self.assertEqual(reply["observation"]["steps_remaining"], 0)

    def test_a_long_line_after_the_end_is_episode_over_as_a_step(self):
        evaluator = tiny()
        evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        [reply] = self.run_lines(evaluator, [self.padded(self.LIMIT + 5)])
        [step] = self.run_lines(evaluator, [json.dumps({"op": "step", "action": FX_PAGE}).encode()])
        self.assertEqual(reply, step)
        self.assertEqual(reply["observation"]["error"]["code"], "episode_over")

    def test_a_long_submit_is_an_invalid_structure(self):
        evaluator = tiny()
        long_submit = json.dumps({"op": "step", "action": {"action": "submit", "findings": {
            "episode_id": "ep_tiny0001", "findings": [], "pad": "x" * self.LIMIT}}}).encode()
        replies = self.run_lines(evaluator, [long_submit, json.dumps({"op": "step", "action": FX_PAGE}).encode()])
        self.assertEqual((replies[0]["observation"]["type"], replies[0]["reward"], replies[0]["terminated"]),
                         ("final", -1.0, True))
        self.assertEqual(replies[0]["info"]["inspection_steps"], 0)
        self.assertEqual(replies[1]["observation"]["error"]["code"], "episode_over")

    def test_a_submit_after_the_first_mebibyte_is_still_an_invalid_structure(self):
        # D2: the action name comes after a filler longer than the limit, also split by a chunk border
        for gap in (self.LIMIT + 10, 3 * self.LIMIT, 2 * self.LIMIT - 12):
            for hint in ('"action":"submit"', '"action" \t:   " submit "'):
                evaluator = tiny()
                line = ('{"op":"step","action":{"pad":"' + "x" * gap + '",' + hint + '}}').encode()
                [reply] = self.run_lines(evaluator, [line])
                self.assertEqual((reply["observation"]["type"], reply["reward"], reply["terminated"]),
                                 ("final", -1.0, True), (gap, hint))

    def test_whitespace_around_the_hint_may_cross_a_chunk_border(self):
        prefix = b'{"op":"step","action":{"action"' + b" " * (self.LIMIT + 5) + b':' + b" " * 7 + b'"submit"}}'
        [reply] = self.run_lines(tiny(), [prefix])
        self.assertEqual((reply["reward"], reply["terminated"]), (-1.0, True))

    def test_a_long_line_without_submit_anywhere_stays_an_error_step(self):
        line = ('{"op":"step","action":{"pad":"' + "x" * (3 * self.LIMIT) + '","note":"submitted"}}').encode()
        [reply] = self.run_lines(tiny(), [line])
        self.assertEqual((reply["observation"]["error"]["code"], reply["terminated"]), ("bad_arguments", False))

    def test_a_long_submit_with_the_name_in_another_key_and_case(self):
        for head in ('{"op":"step","action":{"TOOL":"x","name":"Submit",', '{"action":{"type" : " submit "},'):
            evaluator = tiny()
            line = (head + '"pad":"' + "x" * self.LIMIT + '"}}').encode()
            [reply] = self.run_lines(evaluator, [line])
            self.assertEqual(reply["reward"], -1.0, head)

    def test_a_long_line_in_eval_mode_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.jsonl"
            evaluator = Evaluator(mode="eval", results_path=path, clock=FakeClock())
            evaluator.load(TINY)
            long_submit = json.dumps({"op": "step", "action": {"action": "submit", "pad": "x" * self.LIMIT}})
            [reply] = self.run_lines(evaluator, [long_submit.encode()])
            self.assertEqual((reply["reward"], reply["info"]), (0.0, {}))
            self.assertEqual(json.loads(path.read_text())["reward"], -1.0)

    def test_the_process_survives_a_long_line_and_keeps_counting(self):
        with Env.from_files(TINY) as env:
            env.reset()
            process = env._process
            process.stdin.write(self.padded(self.LIMIT + 10) + b"\n")
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())
            self.assertEqual((reply["observation"]["error"]["code"], reply["observation"]["steps_remaining"]),
                             ("bad_arguments", 99))
            self.assertEqual(env.step(FX_PAGE)[0]["steps_remaining"], 98)


class ClientRobustnessTest(unittest.TestCase):
    """C7, C8, R1, C10: the client raises only EnvError and ends a broken submit as an invalid structure."""

    def test_reset_without_arguments_and_with_options(self):
        with Env() as env:
            default, info = env.reset()
            self.assertEqual((default["type"], info), ("reset", {}))
            by_seed = env.reset(DEFAULT_SEED)[0]
            self.assertEqual(strict_json(default), strict_json(by_seed))
        with Env() as env:
            by_config = env.reset(5, {"errors": {}})[0]
            by_options = env.reset(5, options={"errors": {}})[0]
            by_gym_style = env.reset(seed=5, options={"errors": {}})[0]
            self.assertEqual(strict_json(by_config), strict_json(by_options))
            self.assertEqual(strict_json(by_config), strict_json(by_gym_style))
            default_config = env.reset(5)[0]
            self.assertNotEqual(strict_json(default_config), strict_json(by_config))
            # no seed: the next default seed (previous seed + 1 = 6) with the given options
            clean_default = env.reset(options={"errors": {}})[0]
            self.assertEqual(strict_json(clean_default), strict_json(env.reset(6, {"errors": {}})[0]))

    def test_options_on_the_wire(self):
        lines = [json.dumps({"op": "reset", "seed": 5, "options": {"errors": {}}}),
                 json.dumps({"op": "reset", "seed": 5, "config": {"errors": {}}}),
                 json.dumps({"op": "reset", "seed": 5, "config": None, "options": {"errors": {}}})]
        out = io.StringIO()
        serve.serve(Evaluator(clock=FakeClock()), stdin=io.BytesIO(("\n".join(lines) + "\n").encode()), stdout=out)
        replies = [r["observation"]["manifest"]["episode_id"] for r in map(json.loads, out.getvalue().splitlines())]
        self.assertEqual(len(set(replies)), 1)

    def test_bad_input_only_raises_env_error(self):
        deep = []
        for _ in range(5000):
            deep = [deep]
        cycle = []
        cycle.append(cycle)
        hostile = [float("nan"), float("inf"), 10 ** 5000, object(), {1, 2}, b"x", deep, cycle,
                   {"a": float("nan")}, {(1, 2): 3}, "\ud800", -(10 ** 5000)]
        with Env() as env:
            for value in hostile:
                for call in (lambda: env.reset(value), lambda: env.reset(1, value), lambda: env.reset(1, options=value)):
                    try:
                        call()
                    except EnvError:
                        pass
                    except Exception as error:  # noqa: BLE001
                        self.fail(f"{type(error).__name__} for {str(value)[:30]!r}")
            env.reset(5, {"errors": {}})
            for value in hostile:
                try:
                    observation = env.step(value)[0]
                except EnvError:
                    continue
                except Exception as error:  # noqa: BLE001
                    self.fail(f"{type(error).__name__} for step {str(value)[:30]!r}")
                self.assertEqual(observation["error"]["code"], "unknown_action")
            self.assertEqual(env.step(FX_PAGE)[0]["type"], "records")

    def test_constructor_and_loader_raise_env_error(self):
        for call in (lambda: Env(mode="debug"), lambda: Env(results_path=object()),
                     lambda: Env.from_files(object()), lambda: Env.from_files("/nonexistent/dir")):
            with self.assertRaises(EnvError):
                call()

    def test_a_submit_that_cannot_be_serialised_is_an_invalid_structure(self):
        """R1: -1 and the episode ends, as it does for a malformed wrapper."""
        cycle = []
        cycle.append(cycle)
        deep = []
        for _ in range(100000):  # too deep to encode on every Python (3.12+ encodes 5 000 levels)
            deep = [deep]
        good = CORRECT_TINY["findings"][0]
        for label, extra in (("nan", float("nan")), ("inf", float("-inf")), ("huge int", 10 ** 5000),
                             ("cycle", cycle), ("deep", deep), ("object", object())):
            with self.subTest(label), Env.from_files(TINY) as env:
                env.reset()
                env.step(FX_PAGE)
                action = {"action": "submit", "findings": {"episode_id": "ep_tiny0001", "findings": [good],
                                                           "extra": extra}}
                observation, reward, terminated, truncated, info = env.step(action)
                self.assertEqual((observation["type"], reward, terminated, truncated), ("final", -1.0, True, False))
                self.assertEqual(info["inspection_steps"], 1)
                self.assertEqual(env.step(FX_PAGE)[0]["error"]["code"], "episode_over")

    def test_a_submit_in_any_spelling_is_recognised_when_unserialisable(self):
        for action in ({"tool": "Submit", "args": {"findings": float("nan")}},
                       {"name": " SUBMIT ", "arguments": {"findings": [float("inf")]}}):
            with Env.from_files(TINY) as env:
                env.reset()
                self.assertEqual(env.step(action)[1:3], (-1.0, True), action)

    def test_an_unserialisable_non_submit_is_an_unknown_action_and_spends_a_step(self):
        with Env.from_files(TINY) as env:
            env.reset()
            observation = env.step({"action": "calculator", "expression": float("nan")})[0]
            self.assertEqual((observation["error"]["code"], observation["steps_remaining"]), ("unknown_action", 99))

    def test_huge_integer_reaches_the_server_in_a_raw_submit(self):
        """The server process has no limit on the digits of an integer: such a submit is judged (FP)."""
        line = ('{"op":"step","action":{"action":"submit","findings":{"episode_id":"ep_tiny0001","findings":['
                '{"type":"duplicate_invoice","affected_ids":["k7m2qa","z5c1ty"],"claimed_amount_minor":'
                + "9" * 6000 + ',"currency":"KZT","evidence_ids":["k7m2qa"]}]}}}\n')
        with Env.from_files(TINY) as env:
            env.reset()
            process = env._process
            process.stdin.write(line.encode())
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())
        self.assertEqual((reply["observation"]["type"], reply["terminated"], reply["info"]["f1"]),
                         ("final", True, 0.0))  # judged as a false finding, not refused as bad JSON

    def test_timeout_is_a_clear_env_error(self):
        env = Env(timeout=0.5)
        stalled = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        real, env._process = env._process, stalled
        real.kill()
        real.wait()
        for stream in (real.stdin, real.stdout):  # the replaced server's pipes (no ResourceWarning)
            stream.close()
        try:
            with self.assertRaises(EnvError) as caught:
                env.reset(1, {})
            self.assertEqual(caught.exception.code, "timeout")
            self.assertEqual(stalled.wait(timeout=10) is not None, True)  # the stuck server was stopped
            with self.assertRaises(EnvError) as again:
                env.step(FX_PAGE)
            self.assertEqual(again.exception.code, "closed")
        finally:
            stalled.kill()
            stalled.wait()
            for stream in (stalled.stdin, stalled.stdout):
                stream.close()

    def test_timeout_none_waits(self):
        with Env.from_files(TINY, timeout=None) as env:
            self.assertEqual(env.reset()[0]["type"], "reset")


class StorageToleranceTest(unittest.TestCase):
    """R9, R10: optional private files, a byte order mark, doc_id before invoice_id."""

    def copy_tiny(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        target = pathlib.Path(directory.name) / "episode"
        shutil.copytree(TINY, target)
        return target

    def test_episode_without_provenance_and_pre_files_loads(self):
        target = self.copy_tiny()
        for name in ("provenance.json", "pre_ledger.csv", "pre_invoices.jsonl", "pre_payments.jsonl"):
            (target / "private" / name).unlink()
        reference = Evaluator(clock=FakeClock())
        reference_first = reference.load(TINY)
        evaluator = Evaluator(clock=FakeClock())
        self.assertEqual(strict_json(evaluator.load(target)), strict_json(reference_first))
        self.assertTrue(final(evaluator.step({"action": "submit", "findings": CORRECT_TINY})))
        with Env.from_files(target) as env:
            env.reset()
            self.assertTrue(final(env.step({"action": "submit", "findings": CORRECT_TINY})))

    def test_truth_and_public_files_are_still_required(self):
        for relative in ("private/truth.json", "public/ledger.csv", "public/manifest.json"):
            target = self.copy_tiny()
            (target / relative).unlink()
            with self.assertRaises(EvaluatorError) as caught:
                Evaluator(clock=FakeClock()).load(target)
            self.assertEqual(caught.exception.code, "bad_episode")

    def test_byte_order_mark_is_tolerated(self):
        target = self.copy_tiny()
        for path in list((target / "public").iterdir()) + list((target / "private").iterdir()):
            path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
        reference = Evaluator(clock=FakeClock()).load(TINY)
        evaluator = Evaluator(clock=FakeClock())
        self.assertEqual(strict_json(evaluator.load(target)), strict_json(reference))
        self.assertEqual(evaluator.step({"action": "get_record", "table": "ledger", "id": "a5"})[0]["record"]["row_id"],
                         "a5")
        self.assertTrue(final(evaluator.step({"action": "submit", "findings": CORRECT_TINY})))

    def test_get_record_finds_doc_id_before_invoice_id(self):
        """R10: after a foreign renumbering one invoice's invoice_id may equal another's doc_id."""
        target = self.copy_tiny()
        path = target / "public" / "invoices.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["invoice_id"] = rows[1]["doc_id"]  # the first record's invoice_id collides with the second's doc_id
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        evaluator = Evaluator(clock=FakeClock())
        evaluator.load(target)
        observation = evaluator.step({"action": "get_record", "table": "invoices", "id": rows[1]["doc_id"]})[0]
        self.assertEqual(observation["record"]["doc_id"], rows[1]["doc_id"])
        observation = evaluator.step({"action": "get_record", "table": "invoices", "id": rows[0]["doc_id"]})[0]
        self.assertEqual(observation["record"]["doc_id"], rows[0]["doc_id"])
        observation = evaluator.step({"action": "get_record", "table": "invoices", "id": rows[2]["invoice_id"]})[0]
        self.assertEqual(observation["record"]["doc_id"], rows[2]["doc_id"])  # the invoice_id still finds it
        observation = evaluator.step({"action": "get_document", "doc_id": rows[1]["doc_id"]})[0]
        self.assertEqual(observation["record"]["doc_id"], rows[1]["doc_id"])


class PolicyWhitelistTest(unittest.TestCase):
    """Q2: the agent sees only whitelisted keys of the open policy."""

    def test_extra_keys_do_not_reach_the_agent(self):
        evaluator = Evaluator(clock=FakeClock())
        evaluator.load(TINY)
        episode = evaluator._episode
        episode["public"]["policy"].update({"error_count": 3, "issues": ["x"], "scenario": "I1",
                                            "seed": 7, "distractors": True, "errors": {"fx_error": 1}})
        observation, _ = evaluator._start(episode)
        self.assertTrue(set(observation["policy"]) <= set(actions.POLICY_KEYS + actions.POLICY_KEYS_ADDED))
        for word in ("error_count", "issues", "scenario", "seed", "distractors", "errors"):
            self.assertNotIn(word, strict_json(observation))
        self.assertEqual(observation["policy"]["amount_tolerance_minor"]["fx_error"], 1)

    def test_every_contract_key_passes_and_the_rest_is_documented(self):
        policy = json.loads((TINY / "public" / "policy.json").read_text())
        self.assertEqual(set(actions.open_policy(policy)), set(policy))  # all of CONTRACTS 2.2 survives
        self.assertEqual(actions.open_policy({"period_interval": "[start, end)", "x": 1}),
                         {"period_interval": "[start, end)"})

    def test_the_generated_policy_is_fully_whitelisted(self):
        """A new rule key of the generator must be added to POLICY_KEYS_ADDED, or the agent never sees it."""
        from auditgym.generate import generate

        allowed = set(actions.POLICY_KEYS + actions.POLICY_KEYS_ADDED)
        for config in (None, {"errors": {}}, {"interaction": "I1", "size": {"invoices": [48, 100],
                                                                            "background_pairs": [150, 300]}}):
            policy = generate(4, config)["public"]["policy"]
            self.assertEqual(set(policy) - allowed, set(), config)

    def test_task_tolerances_come_from_the_policy(self):
        observation, _ = tiny().__class__(clock=FakeClock()).load(TINY)
        self.assertEqual(observation["task"]["tolerances"]["amount_tolerance_minor"]["fx_error"], 1)


class CalculatorRoundingTest(unittest.TestCase):
    """C2 (mutation): the calculator rounds half to even."""

    def test_half_even(self):
        from auditgym.calc import calculate

        # 34 significant digits: the tie below is resolved to the even neighbour, not up.
        self.assertEqual(calculate("1000000000000000000000000000000000 + 0.5"), "1000000000000000000000000000000000")
        self.assertEqual(calculate("1000000000000000000000000000000001 + 0.5"), "1000000000000000000000000000000002")
        self.assertEqual(calculate("1000000000000000000000000000000000 + 1.5"), "1000000000000000000000000000000002")
        self.assertEqual(calculate("1000000000000000000000000000000002 - 0.5"), "1000000000000000000000000000000002")
        self.assertEqual(calculate("1000000000000000000000000000000000 + 2.5"), "1000000000000000000000000000000002")
        self.assertEqual(calculate("-1000000000000000000000000000000000 - 0.5"), "-1000000000000000000000000000000000")


if __name__ == "__main__":
    unittest.main()
