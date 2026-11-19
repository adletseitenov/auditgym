"""Regression tests for the env-zone findings of docs/AUDIT.md (one class per finding group).

C1, R7  submit accepts findings.json as an object, a JSON string, a bare list, flat in the action
        and under "findings.json"; action names in any letter case
C2      table names: any letter case, a file extension, the singular
C3      argument synonyms, start cursors "" / 0 / "0", limits "50" / 50.0, action as JSON text
C5, L3  a judge log by default, created owner-only
C7      reset() without a seed, reset(options=...), Env(directory)
C8      the client raises only EnvError
C9      a submit whose arguments cannot be read ends the episode with -1
C10     flags and step counts after the end, a refused reset keeps the episode, client timeout
L1      evaluator secret mixed into generated seeds
L4      a line over 1 MiB during an episode spends a step (a submit ends with -1)
R1      NaN, Infinity, huge integers and unserializable submits end the episode with -1
R9, R10 loading without provenance and pre_* files, with byte order marks, and doc_id-first lookup
"""
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

from auditgym import Env, EnvError
from auditgym.evaluator import Evaluator, EvaluatorError, generator_seed
from auditgym.generate import generate
from tests.env.helpers import CLEAN, ROOT, TINY, FakeClock, full_walk, strict_json, tiny

CORRECT_TINY = {"episode_id": "ep_tiny0001", "findings": [{
    "type": "duplicate_invoice", "affected_ids": ["k7m2qa", "z5c1ty"], "claimed_amount_minor": 5000000,
    "currency": "KZT", "evidence_ids": ["k7m2qa", "z5c1ty", "a1", "a3"], "explanation": "Same customer and total."}]}
FINDINGS = CORRECT_TINY["findings"]


def results_lines(path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]


def evaluator_with_log(directory: str) -> tuple[Evaluator, str]:
    path = os.path.join(directory, "results.jsonl")
    evaluator = Evaluator(results_path=path, clock=FakeClock())
    evaluator.load(TINY)
    return evaluator, path


def serve(lines, *args, cwd=None, env=None) -> list[dict]:
    environment = dict(os.environ, PYTHONPATH=str(ROOT), **(env or {}))
    data = b"".join(line if isinstance(line, bytes) else (line + "\n").encode() for line in lines)
    result = subprocess.run([sys.executable, "-m", "auditgym", "serve", *args], input=data, capture_output=True,
                            env=environment, cwd=cwd or str(ROOT), timeout=120)
    return [json.loads(line) for line in result.stdout.splitlines()]


class SubmitFormsTest(unittest.TestCase):
    """C1, R7: every accepted form scores like the canonical one; the structure rules still apply."""

    FORMS = [
        {"action": "submit", "findings": CORRECT_TINY},
        {"action": "submit", "findings": json.dumps(CORRECT_TINY)},
        {"action": "submit", "findings": FINDINGS},
        {"action": "submit", "findings": json.dumps(FINDINGS)},
        {"action": "submit", "episode_id": "ep_tiny0001", "findings": FINDINGS},
        {"tool": "submit", "arguments": {"findings.json": CORRECT_TINY}},
        {"tool": "submit", "arguments": json.dumps({"findings.json": CORRECT_TINY})},
        {"tool": "submit", "arguments": CORRECT_TINY},
        {"name": "Submit", "params": {"findings": FINDINGS}},
        {"type": " SUBMIT ", "findings": CORRECT_TINY},
        json.dumps({"action": "submit", "findings": CORRECT_TINY}),
    ]

    def test_every_form_gets_the_full_reward(self):
        for form in self.FORMS:
            evaluator = tiny()
            observation, reward, terminated, truncated, info = evaluator.step(form)
            self.assertEqual((observation["type"], reward, terminated, truncated), ("final", 1.0, True, False), form)

    def test_the_rules_still_apply_inside_every_form(self):
        wrong_episode = {"episode_id": "ep_other", "findings": FINDINGS}
        cases = [
            ({"action": "submit", "episode_id": "ep_other", "findings": FINDINGS}, "wrong_episode_id"),
            ({"action": "submit", "findings": json.dumps(wrong_episode)}, "wrong_episode_id"),
            ({"action": "submit", "findings": [1, 2]}, "finding_not_object"),
            ({"action": "submit", "findings": "[{\"type\": \"nope\"}]"}, "missing_field"),
            ({"action": "submit", "findings": "{not json"}, "not_json"),
            ({"action": "submit", "findings": "[" * 100000}, "not_json"),
            ({"action": "submit", "findings": FINDINGS * 31}, "too_many_findings"),
            ({"action": "submit", "findings": None}, "missing_field"),
            ({"action": "submit"}, "missing_field"),
            ({"action": "submit", "findings": 7}, "not_object"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for action, code in cases:
                evaluator, path = evaluator_with_log(directory)
                _, reward, terminated, _, _ = evaluator.step(action)
                self.assertEqual((reward, terminated), (-1.0, True), action)
                self.assertEqual(results_lines(path)[-1]["structure_error"], code, action)

    def test_bare_list_takes_the_running_episode_id(self):
        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset(5, CLEAN)
        _, reward, terminated, _, info = evaluator.step({"action": "submit", "findings": []})
        self.assertEqual((reward, terminated, info["f1"]), (1.0, True, 1.0))  # silence on clean books


class BadSubmitArgumentsTest(unittest.TestCase):
    """C9: a submit whose arguments cannot be read is an invalid structure, not an error step."""

    def test_unreadable_submit_arguments_end_the_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            for action in ({"action": "submit", "args": 5}, {"action": "submit", "arguments": "not json"},
                           {"tool": "submit", "params": "[1]"}, {"action": "SUBMIT", "args": []}):
                evaluator, path = evaluator_with_log(directory)
                observation, reward, terminated, truncated, info = evaluator.step(action)
                self.assertEqual((observation["type"], reward, terminated, truncated), ("final", -1.0, True, False))
                self.assertEqual(results_lines(path)[-1]["structure_error"], "bad_submit_arguments")

    def test_other_actions_with_unreadable_arguments_are_error_steps(self):
        evaluator = tiny()
        observation, _, terminated, _, _ = evaluator.step({"action": "get_record", "args": 5})
        self.assertEqual((observation["error"]["code"], terminated, observation["steps_remaining"]),
                         ("bad_arguments", False, 99))


class TableNamesTest(unittest.TestCase):
    """C2: letter case, a file extension and the singular are accepted; anything else is unknown."""

    def test_table_name_variants(self):
        evaluator = tiny()
        for given, table in (("Ledger", "ledger"), ("ledger.csv", "ledger"), (" LEDGER.CSV ", "ledger"),
                             ("invoice", "invoices"), ("Invoices.jsonl", "invoices"), ("payment", "payments"),
                             ("payments.json", "payments"), ("FX", "fx"), ("fx.csv", "fx")):
            observation = evaluator.step({"action": "list_records", "table": given})[0]
            self.assertEqual((observation["type"], observation["table"]), ("records", table), given)

    def test_unknown_tables_stay_unknown(self):
        evaluator = tiny()
        for given in ("ledgers", "public/ledger.csv", "../private/truth.json", "truth", "truth.json",
                      "pre_ledger.csv", "ledger.txt", "", 5, None, ["ledger"]):
            observation = evaluator.step({"action": "list_records", "table": given})[0]
            self.assertEqual(observation["error"]["code"], "unknown_table", given)


class ArgumentSynonymsTest(unittest.TestCase):
    """C3: synonyms of argument names; conflicting or non-string values are bad arguments."""

    def test_get_record_synonyms(self):
        evaluator = tiny()
        cases = [({"table": "ledger", "record_id": "a5"}, "row_id", "a5"),
                 ({"table": "ledger", "row_id": "a5"}, "row_id", "a5"),
                 ({"table": "ledger", "key": "a5"}, "row_id", "a5"),
                 ({"table": "invoices", "doc_id": "k7m2qa"}, "doc_id", "k7m2qa"),
                 ({"table": "invoices", "document_id": "k7m2qa"}, "doc_id", "k7m2qa"),
                 ({"table": "invoices", "invoice_id": "p3x9wd"}, "doc_id", "k7m2qa"),
                 ({"table": "invoices", "id": "p3x9wd"}, "doc_id", "k7m2qa"),
                 ({"table": "payments", "payment_id": "y6d2ck"}, "doc_id", "h3n8vx"),
                 ({"table": "fx", "date": "2026-10-01", "source_currency": "USD", "target_currency": "KZT"},
                  "date", "2026-10-01"),
                 ({"table": "fx", "id": "2026-10-01|USD|KZT", "date": "1999-01-01"}, "date", "2026-10-01")]
        for arguments, field, value in cases:
            observation = evaluator.step(dict(arguments, action="get_record"))[0]
            self.assertEqual((observation["type"], observation["record"][field]), ("record", value), arguments)

    def test_named_field_is_searched_alone(self):
        evaluator = tiny()
        # p3x9wd is an invoice_id, not a doc_id: by doc_id it is unknown.
        for arguments in ({"table": "invoices", "doc_id": "p3x9wd"},
                          {"table": "invoices", "invoice_id": "k7m2qa"}):
            observation = evaluator.step(dict(arguments, action="get_record"))[0]
            self.assertEqual(observation["error"]["code"], "unknown_id", arguments)

    def test_bad_synonym_values(self):
        evaluator = tiny()
        for arguments in ({"table": "ledger", "id": "a5", "row_id": "a1"}, {"table": "ledger", "record_id": 5},
                          {"table": "fx", "date": "2026-10-01", "source_currency": "USD"},
                          {"table": "ledger", "id": ["a5"]}):
            observation = evaluator.step(dict(arguments, action="get_record"))[0]
            self.assertEqual(observation["error"]["code"], "bad_arguments", arguments)
        for arguments in ({"id": "k7m2qa", "doc_id": "z5c1ty"}, {"document_id": 7}, {}):
            observation = evaluator.step(dict(arguments, action="get_document"))[0]
            self.assertEqual(observation["error"]["code"], "bad_arguments", arguments)

    def test_get_document_and_calculator_synonyms(self):
        evaluator = tiny()
        for arguments in ({"id": "h3n8vx"}, {"document_id": "h3n8vx"}, {"doc_id": "h3n8vx", "id": "h3n8vx"}):
            observation = evaluator.step(dict(arguments, action="get_document"))[0]
            self.assertEqual(observation["record"]["payment_id"], "y6d2ck", arguments)
        observation = evaluator.step({"action": "calculator", "expr": "2 * 3"})[0]
        self.assertEqual(observation["result"], "6")

    def test_action_as_json_text_and_any_case(self):
        evaluator = tiny()
        for action in ('{"action": "get_record", "table": "ledger", "id": "a5"}',
                       {"action": " Get_Record ", "table": "Ledger", "id": "a5"},
                       {"tool": "GET_RECORD", "args": '{"table": "LEDGER.CSV", "id": "a5"}'}):
            observation = evaluator.step(action)[0]
            self.assertEqual(observation["record"]["row_id"], "a5", action)


class LoadingTest(unittest.TestCase):
    """R9, R10: optional evaluator-unused files, byte order marks, doc_id-first lookup."""

    def _copy(self, directory: str) -> pathlib.Path:
        target = pathlib.Path(directory) / "episode"
        shutil.copytree(TINY, target)
        return target

    def _transcript(self, directory) -> str:
        evaluator = Evaluator(clock=FakeClock())
        return strict_json([evaluator.load(directory)] + full_walk(evaluator))

    def test_without_provenance_and_pre_files(self):
        expected = self._transcript(TINY)
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(directory)
            for name in ("provenance.json", "pre_ledger.csv", "pre_invoices.jsonl", "pre_payments.jsonl"):
                (target / "private" / name).unlink()
            self.assertEqual(self._transcript(target), expected)
            with Env.from_files(target) as env:
                env.reset()
                self.assertEqual(env.step({"action": "submit", "findings": CORRECT_TINY})[1], 1.0)

    def test_truth_is_still_required(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(directory)
            (target / "private" / "truth.json").unlink()
            with self.assertRaises(EvaluatorError) as caught:
                Evaluator(clock=FakeClock()).load(target)
            self.assertEqual(caught.exception.code, "bad_episode")

    def test_byte_order_marks(self):
        expected = self._transcript(TINY)
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(directory)
            for path in sorted(target.rglob("*")):
                if path.is_file() and path.suffix in (".json", ".jsonl", ".csv"):
                    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
            self.assertEqual(self._transcript(target), expected)

    def test_doc_id_wins_over_a_colliding_invoice_id(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._copy(directory)
            invoices = target / "public" / "invoices.jsonl"
            records = [json.loads(line) for line in invoices.read_text(encoding="utf-8").splitlines()]
            records[0]["invoice_id"] = records[1]["doc_id"]  # the first record now collides with the second
            invoices.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            evaluator = Evaluator(clock=FakeClock())
            evaluator.load(target)
            by_id = evaluator.step({"action": "get_record", "table": "invoices", "id": "z5c1ty"})[0]["record"]
            by_invoice = evaluator.step({"action": "get_record", "table": "invoices",
                                         "invoice_id": "z5c1ty"})[0]["record"]
        self.assertEqual(by_id["doc_id"], "z5c1ty")
        self.assertEqual(by_invoice["doc_id"], "k7m2qa")


class ResetTest(unittest.TestCase):
    """C7, C10: reset without a seed, options, Env(directory); a refused reset changes nothing."""

    def test_env_reset_variants(self):
        default = generate(0, None)["public"]["manifest"]["episode_id"]
        configured = generate(4, CLEAN)["public"]["manifest"]["episode_id"]

        class Index:
            def __index__(self):
                return 4

        with Env() as env:
            self.assertEqual(env.reset()[0]["manifest"]["episode_id"], default)
            self.assertEqual(env.reset(seed=4, options=CLEAN)[0]["manifest"]["episode_id"], configured)
            self.assertEqual(env.reset(Index(), CLEAN)[0]["manifest"]["episode_id"], configured)
            with self.assertRaises(EnvError) as caught:
                env.reset(4, CLEAN, options=CLEAN)
            self.assertEqual(caught.exception.code, "bad_config")

    def test_env_directory_in_place_of_the_mode(self):
        with Env(str(TINY)) as env:
            self.assertEqual(env.mode, "train")
            self.assertEqual(env.reset()[0]["manifest"]["episode_id"], "ep_tiny0001")
        with Env(TINY) as env:
            self.assertEqual(env.reset()[0]["manifest"]["episode_id"], "ep_tiny0001")

    def test_refused_reset_through_the_process_keeps_the_episode(self):
        with Env.from_files(TINY) as env:
            env.reset()
            env.step({"action": "list_records", "table": "fx"})
            with self.assertRaises(EnvError):
                env.reset(1, {"nope": 1})
            observation = env.step({"action": "list_records", "table": "fx"})[0]
            self.assertEqual(observation["steps_remaining"], 98)

    def test_episode_over_repeats_the_final_flags(self):
        evaluator = tiny()
        for _ in range(100):
            final = evaluator.step({"action": "list_records", "table": "fx"})
        self.assertEqual(final[2:4], (False, True))
        after = evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual(after, ({"type": "error", "error": {"code": "episode_over", "message": "Episode is over."},
                                  "steps_remaining": 0}, 0.0, False, True, {}))


class ClientErrorsTest(unittest.TestCase):
    """C8, R1: hostile values never raise anything but EnvError; bad submits end with -1."""

    def test_only_env_error_escapes(self):
        cycle = {}
        cycle["self"] = cycle
        deep = []
        for _ in range(100000):
            deep = [deep]
        with Env() as env:
            for seed, config in ((object(), None), (1, {"x": {1, 2}}), (1, cycle), (float("nan"), None),
                                 (10 ** 5000, None), (1, deep)):
                with self.assertRaises(EnvError) as caught:
                    env.reset(seed, config)
                self.assertEqual(caught.exception.code, "bad_config")
            env.reset(3, CLEAN)
            remaining = 100
            for action in (object(), cycle, deep, {"action": "list_records", "table": "fx", "limit": {1}},
                           {"action": "calculator", "expression": b"1"}, {"action": object()}):
                observation, reward, terminated, truncated, info = env.step(action)
                remaining -= 1
                self.assertEqual((observation["type"], terminated, observation["steps_remaining"]),
                                 ("error", False, remaining), type(action).__name__)
            self.assertEqual(env.step({"action": "list_records", "table": "fx", "limit": {1}})[0]["error"]["code"],
                             "bad_arguments")
        for mode in ("debug", None, 5):
            with self.assertRaises(EnvError) as caught:
                Env(mode)
            self.assertEqual(caught.exception.code, "bad_mode")
        with self.assertRaises(EnvError) as caught:
            Env(timeout=0)
        self.assertEqual(caught.exception.code, "bad_timeout")

    def test_unsendable_submit_ends_the_episode_with_minus_one(self):
        cycle = []
        cycle.append(cycle)
        huge = dict(FINDINGS[0], claimed_amount_minor=10 ** 5000)
        nan = dict(FINDINGS[0], claimed_amount_minor=float("nan"))
        infinite = dict(FINDINGS[0], explanation=float("inf"))
        cases = [({"action": "submit", "findings": {"episode_id": "ep_tiny0001", "findings": [huge]}}, "not_json"),
                 ({"action": "submit", "findings": {"episode_id": "ep_tiny0001", "findings": cycle}}, "not_json"),
                 ({"action": "submit", "findings": {"episode_id": "ep_tiny0001", "findings": [nan]}}, "non_finite"),
                 ({"tool": "submit", "args": {"findings": [infinite]}}, "non_finite"),
                 ({"action": "submit", "findings": object()}, "not_json")]
        with tempfile.TemporaryDirectory() as directory:
            results = pathlib.Path(directory) / "r.jsonl"
            for action, code in cases:
                with Env.from_files(TINY, mode="eval", results_path=results) as env:
                    env.reset()
                    observation, reward, terminated, truncated, info = env.step(action)
                    self.assertEqual((observation["type"], terminated, truncated), ("final", True, False))
                self.assertEqual(results_lines(results)[-1]["structure_error"], code, code)

    def test_timeout_stops_the_server(self):
        env = Env(timeout=0.001)
        process = env._process
        with self.assertRaises(EnvError) as caught:
            env.reset(5, None)  # generating takes far longer than a millisecond
        self.assertEqual(caught.exception.code, "timeout")
        self.assertIsNotNone(process.poll())
        with self.assertRaises(EnvError) as caught:
            env.step({"action": "list_records", "table": "fx"})
        self.assertEqual(caught.exception.code, "closed")
        env.close()

    def test_bad_results_path_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = pathlib.Path(directory) / "file"
            blocker.write_text("x")
            with self.assertRaises(EnvError) as caught:
                Env(results_path=blocker / "results.jsonl")
            self.assertEqual(caught.exception.code, "bad_results")


class ServerIntegersTest(unittest.TestCase):
    """R1: a JSON integer of any length is JSON; as an amount it is well-formed and never matches."""

    def test_huge_amount_through_the_protocol_is_judged(self):
        finding = dict(FINDINGS[0])
        line = json.dumps({"op": "step", "action": {"action": "submit", "findings": {
            "episode_id": "ep_tiny0001", "findings": [finding]}}})
        line = line.replace('"claimed_amount_minor": 5000000', '"claimed_amount_minor": ' + "9" * 6000)
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "r.jsonl")
            replies = serve([json.dumps({"op": "load", "directory": str(TINY)}), line], "--results", results)
            self.assertEqual((replies[1]["terminated"], replies[1]["reward"]), (True, -1.0))
            entry = results_lines(results)[-1]
        self.assertEqual((entry["structure_error"], entry["fp"], entry["fn"]), (None, 1, 1))

    def test_huge_seed_is_a_refused_reset(self):
        replies = serve(['{"op": "reset", "seed": ' + "7" * 5000 + "}"], "--no-results")
        self.assertEqual(replies[0]["observation"]["error"]["code"], "bad_config")


class OversizedLineTest(unittest.TestCase):
    """L4: a line over 1 MiB is a spent error step during an episode; one mentioning submit ends it."""

    def test_oversized_lines(self):
        padding = "x" * ((1 << 20) + 10)
        lines = [json.dumps({"op": "load", "directory": str(TINY)}),
                 json.dumps({"op": "step", "action": {"action": "calculator", "expression": padding}}),
                 json.dumps({"op": "step", "action": {"action": "list_records", "table": "fx"}}),
                 json.dumps({"op": "step", "action": {"action": "submit", "findings": padding + "sub"}})[:-5]
                 + "mit\"}}",
                 json.dumps({"op": "step", "action": {"action": "list_records", "table": "fx"}})]
        with tempfile.TemporaryDirectory() as directory:
            results = os.path.join(directory, "r.jsonl")
            replies = serve(lines, "--results", results)
            entry = results_lines(results)[-1]
        self.assertEqual(replies[1]["observation"], {"type": "error", "steps_remaining": 99,
                                                     "error": {"code": "bad_arguments", "message": "Bad arguments."}})
        self.assertEqual(replies[2]["observation"]["steps_remaining"], 98)
        self.assertEqual((replies[3]["observation"]["type"], replies[3]["terminated"], replies[3]["reward"]),
                         ("final", True, -1.0))
        self.assertEqual(entry["structure_error"], "message_too_large")
        self.assertEqual(replies[4]["observation"]["error"]["code"], "episode_over")


class JudgeLogTest(unittest.TestCase):
    """C5, L3: the judge log exists by default, owner-only; --no-results keeps none."""

    def test_client_default_log_from_the_environment_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "logs" / "judge.jsonl"
            with mock.patch.dict(os.environ, {"AUDITGYM_RESULTS": str(path)}):
                with Env.from_files(TINY) as env:
                    env.reset()
                    env.step({"action": "submit", "findings": {"episode_id": "ep_tiny0001"}})
            entry = results_lines(path)[-1]
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual((entry["reward"], entry["structure_error"]), (-1.0, "missing_field"))

    def test_client_default_log_under_the_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            here = os.getcwd()
            os.chdir(directory)
            try:
                with mock.patch.dict(os.environ, {"AUDITGYM_RESULTS": ""}):
                    with Env.from_files(TINY) as env:
                        env.reset()
                        env.step({"action": "submit", "findings": CORRECT_TINY})
            finally:
                os.chdir(here)
            self.assertEqual(results_lines(pathlib.Path(directory) / "runs" / "results.jsonl")[-1]["reward"], 1.0)

    def test_unwritable_default_location_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = pathlib.Path(directory) / "file"
            blocker.write_text("x")
            with mock.patch.dict(os.environ, {"AUDITGYM_RESULTS": str(blocker / "r.jsonl")}):
                with Env.from_files(TINY) as env:
                    env.reset()
                    self.assertEqual(env.step({"action": "submit", "findings": CORRECT_TINY})[1], 1.0)

    def test_server_default_log_and_no_results(self):
        lines = [json.dumps({"op": "load", "directory": str(TINY)}),
                 json.dumps({"op": "step", "action": {"action": "submit", "findings": "nope"}})]
        with tempfile.TemporaryDirectory() as directory:
            serve(lines, cwd=directory, env={"AUDITGYM_RESULTS": ""})
            path = pathlib.Path(directory) / "runs" / "results.jsonl"
            self.assertEqual(results_lines(path)[-1]["structure_error"], "not_json")
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        with tempfile.TemporaryDirectory() as directory:
            serve(lines, "--no-results", cwd=directory, env={"AUDITGYM_RESULTS": ""})
            self.assertFalse((pathlib.Path(directory) / "runs").exists())

    def test_evaluator_results_file_is_owner_only(self):
        if os.name != "posix":
            self.skipTest("POSIX permissions")
        with tempfile.TemporaryDirectory() as directory:
            evaluator, path = evaluator_with_log(directory)
            evaluator.step({"action": "submit", "findings": CORRECT_TINY})
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)


class SecretTest(unittest.TestCase):
    """L1: with a secret the generator gets a derived seed; without one, the seed itself."""

    def _episode_id(self, **kwargs) -> str:
        return Evaluator(clock=FakeClock(), **kwargs).reset(5, CLEAN)[0]["manifest"]["episode_id"]

    def test_secret_changes_the_episode_and_is_reproducible(self):
        plain = generate(5, CLEAN)["public"]["manifest"]["episode_id"]
        self.assertEqual(self._episode_id(), plain)
        self.assertEqual(self._episode_id(secret=b""), plain)
        first = self._episode_id(secret=b"organizers")
        self.assertNotEqual(first, plain)
        self.assertEqual(self._episode_id(secret="organizers"), first)
        self.assertNotEqual(self._episode_id(secret=b"other"), first)
        self.assertEqual(generator_seed(5, b""), 5)
        self.assertNotEqual(generator_seed(5, b"organizers"), generator_seed(6, b"organizers"))

    def test_server_reads_the_secret_from_the_environment_or_a_file(self):
        lines = [json.dumps({"op": "reset", "seed": 5, "config": CLEAN})]
        expected = Evaluator(clock=FakeClock(), secret=b"organizers").reset(5, CLEAN)[0]["manifest"]["episode_id"]
        from_env = serve(lines, "--no-results", env={"AUDITGYM_SECRET": "organizers"})
        with tempfile.TemporaryDirectory() as directory:
            secret_file = pathlib.Path(directory) / "secret"
            secret_file.write_text("organizers\n")
            from_file = serve(lines, "--no-results", "--secret-file", str(secret_file), env={"AUDITGYM_SECRET": ""})
        for replies in (from_env, from_file):
            self.assertEqual(replies[0]["observation"]["manifest"]["episode_id"], expected)
            self.assertNotIn("organizers", json.dumps(replies))


if __name__ == "__main__":
    unittest.main()
