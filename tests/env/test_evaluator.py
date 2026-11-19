"""CASE §04 sentence by sentence, on the evaluator inside one process (the process split is
tested in test_process.py). Fixture: tests/fixtures/tiny and generated clean companies."""
import json
import pathlib
import tempfile
import unittest

from auditgym.actions import ERRORS, parse_action, ActionError
from auditgym.evaluator import Evaluator, EvaluatorError
from tests.env.helpers import CLEAN, TABLES, TINY, FakeClock, full_walk, generated, strict_json, tiny

RESET_KEYS = {"type", "task", "manifest", "policy", "steps_remaining"}
TASK_KEYS = {"text", "error_types", "findings_schema", "limits", "tolerances", "accounts"}
KEYS = {
    "records": {"type", "table", "records", "next_cursor", "steps_remaining"},
    "record": {"type", "table", "record", "steps_remaining"},
    "document": {"type", "table", "record", "steps_remaining"},
    "calculation": {"type", "result", "steps_remaining"},
    "error": {"type", "error", "steps_remaining"},
    "final": {"type", "steps_remaining"},
}
# DECISIONS A2 / CONTRACTS v2 §3.1: duplicate evidence = both docs + one posting row of each
# invoice (a1 = debit of k7m2qa, a3 = debit of z5c1ty); the two docs alone are a FP.
CORRECT_TINY = {"episode_id": "ep_tiny0001", "findings": [{
    "type": "duplicate_invoice", "affected_ids": ["z5c1ty", "k7m2qa"], "claimed_amount_minor": 5000000,
    "currency": "KZT", "evidence_ids": ["k7m2qa", "z5c1ty", "a1", "a3"], "explanation": "Same customer and total."}]}


def walk_table(evaluator, table, limit=None):
    ids, cursor = [], None
    while True:
        action = {"action": "list_records", "table": table, "cursor": cursor}
        if limit is not None:
            action["limit"] = limit
        observation = evaluator.step(action)[0]
        assert observation["type"] == "records", observation
        ids.extend(json.dumps(r, sort_keys=True) for r in observation["records"])
        cursor = observation["next_cursor"]
        if cursor is None:
            return ids


class ResetTest(unittest.TestCase):
    """«reset(seed, config) возвращает (observation, info). Первое наблюдение: задача, manifest,
    краткая политика и остаток бюджета.»"""

    def test_reset_returns_observation_and_info(self):
        result = Evaluator(clock=FakeClock()).reset(3, CLEAN)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        observation, info = result
        self.assertEqual(info, {})
        strict_json(observation)

    def test_first_observation_has_exactly_task_manifest_policy_budget(self):
        observation, _ = Evaluator(clock=FakeClock()).load(TINY)
        self.assertEqual(set(observation), RESET_KEYS)
        self.assertEqual(observation["type"], "reset")
        self.assertEqual(observation["steps_remaining"], 100)
        self.assertEqual(set(observation["task"]), TASK_KEYS)

    def test_task_lists_types_schema_limits_tolerances(self):
        observation, _ = Evaluator(clock=FakeClock()).load(TINY)
        task = observation["task"]
        self.assertEqual(task["error_types"], ["duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff"])
        self.assertEqual(task["limits"]["max_findings"], 30)
        self.assertEqual(task["limits"]["max_evidence_ids"], 10)
        self.assertEqual(task["limits"]["max_explanation_chars"], 1000)
        self.assertEqual(task["limits"]["max_steps"], 100)
        self.assertEqual(task["limits"]["page_size"], 50)
        self.assertEqual(task["tolerances"]["amount_tolerance_minor"],
                         {"duplicate_invoice": 0, "allocation_gap": 0, "fx_error": 1, "period_cutoff": 0})
        self.assertIn("episode_id", task["findings_schema"]["required"])
        self.assertTrue(task["text"])

    def test_task_text_says_the_budget_includes_submit_and_matches_the_behaviour(self):
        text = Evaluator(clock=FakeClock()).load(TINY)[0]["task"]["text"]
        self.assertIn("Бюджет — 100 шагов, и submit входит в него", text)
        self.assertIn("последним шагом должен быть submit, не позже 100-го", text)
        self.assertIn("Если 100 шагов прошли без submit, эпизод обрезается", text)
        # What the text says is what happens: submit as step 100 counts, step 100 without it truncates.
        evaluator = tiny()
        for _ in range(99):
            evaluator.step({"action": "list_records", "table": "fx"})
        _, reward, terminated, truncated, info = evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual((terminated, truncated, info["f1"], info["inspection_steps"]), (True, False, 1.0, 99))
        evaluator = tiny()
        for _ in range(100):
            result = evaluator.step({"action": "list_records", "table": "fx"})
        self.assertEqual(result[2:4], (False, True))
        self.assertEqual(evaluator.step({"action": "submit", "findings": CORRECT_TINY})[0]["error"]["code"],
                         "episode_over")

    def test_manifest_and_policy_are_the_open_files(self):
        evaluator = Evaluator(clock=FakeClock())
        observation, _ = evaluator.load(TINY)
        self.assertEqual(observation["manifest"], json.loads((TINY / "public/manifest.json").read_text()))
        self.assertEqual(observation["policy"], json.loads((TINY / "public/policy.json").read_text()))

    def test_manifest_keeps_only_contract_fields(self):
        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset(3, CLEAN)
        episode = evaluator._episode
        episode["public"]["manifest"]["seed"] = 3
        observation, _ = evaluator._start(episode)
        self.assertNotIn("seed", observation["manifest"])

    def test_bad_seed_and_config_are_refused(self):
        evaluator = Evaluator(clock=FakeClock())
        for seed, config in ((True, CLEAN), ("7", CLEAN), (1.0, CLEAN), (1, {"nope": 1}), (1, []),
                             (1, {"mode": "train"})):
            with self.assertRaises(EvaluatorError) as caught:
                evaluator.reset(seed, config)
            self.assertEqual(caught.exception.code, "bad_config")
        self.assertEqual(evaluator.step({"action": "list_records", "table": "fx"})[0]["error"]["code"], "not_reset")

    def test_reset_without_seed_uses_the_default_seeds(self):
        """AUDIT C7: reset() without a seed and without files is seed 0, then the previous seed + 1."""
        def episode_id(evaluator, *args):
            return evaluator.reset(*args)[0]["manifest"]["episode_id"]

        evaluator = Evaluator(clock=FakeClock())
        seeds = [episode_id(Evaluator(clock=FakeClock()), seed, None) for seed in (0, 1, 8)]
        self.assertEqual([episode_id(evaluator), episode_id(evaluator)], seeds[:2])
        episode_id(evaluator, 7, None)
        self.assertEqual(episode_id(evaluator), seeds[2])
        with self.assertRaises(EvaluatorError):
            evaluator.reset(None, {"nope": 1})
        self.assertEqual(episode_id(evaluator, None, CLEAN), episode_id(Evaluator(clock=FakeClock()), 9, CLEAN))

    def test_bad_directory_is_one_fixed_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(EvaluatorError) as caught:
                Evaluator(clock=FakeClock()).load(directory)
            self.assertEqual(caught.exception.code, "bad_episode")
            self.assertNotIn(directory, str(caught.exception))

    def test_mode_is_fixed_at_construction(self):
        with self.assertRaises(ValueError):
            Evaluator(mode="debug")
        evaluator = Evaluator(mode="eval", clock=FakeClock())
        with self.assertRaises(AttributeError):
            evaluator.mode = "train"


class StepShapeTest(unittest.TestCase):
    """«step(action) возвращает (observation, reward, terminated, truncated, info). Это JSON-совместимый
    контракт». «Наблюдение содержит только запрошенные записи, результат/ошибку и остаток шагов.»"""

    def test_every_return_value_is_json_and_has_only_contract_keys(self):
        for evaluator in (tiny(), generated(4)):
            for result in full_walk(evaluator):
                self.assertEqual(len(result), 5)
                observation, reward, terminated, truncated, info = result
                strict_json([observation, reward, terminated, truncated, info])
                self.assertIs(type(reward), float)
                self.assertIs(type(terminated), bool)
                self.assertIs(type(truncated), bool)
                self.assertEqual(set(observation), KEYS[observation["type"]], observation)

    def test_record_observations_return_only_the_requested_record(self):
        evaluator = tiny()
        observation = evaluator.step({"action": "get_record", "table": "ledger", "id": "a5"})[0]
        self.assertEqual(observation["record"]["row_id"], "a5")
        self.assertEqual(observation["table"], "ledger")
        observation = evaluator.step({"action": "get_document", "doc_id": "h3n8vx"})[0]
        self.assertEqual((observation["type"], observation["table"]), ("document", "payments"))
        self.assertEqual(observation["record"]["payment_id"], "y6d2ck")
        observation = evaluator.step({"action": "get_document", "doc_id": "k7m2qa"})[0]
        self.assertEqual(observation["table"], "invoices")
        self.assertEqual(len(observation["record"]["line_items"]), 1)

    def test_get_record_accepts_table_ids(self):
        evaluator = tiny()
        by_doc = evaluator.step({"action": "get_record", "table": "invoices", "id": "k7m2qa"})[0]["record"]
        by_invoice = evaluator.step({"action": "get_record", "table": "invoices", "id": "p3x9wd"})[0]["record"]
        self.assertEqual(by_doc, by_invoice)
        by_payment = evaluator.step({"action": "get_record", "table": "payments", "id": "y6d2ck"})[0]["record"]
        self.assertEqual(by_payment["doc_id"], "h3n8vx")

    def test_fx_record_by_composite_key(self):
        observation = tiny().step({"action": "get_record", "table": "fx", "id": "2026-10-01|USD|KZT"})[0]
        self.assertEqual(observation["record"], {"date": "2026-10-01", "source_currency": "USD",
                                                 "target_currency": "KZT", "rate": "466.2100"})

    def test_missing_links_are_null(self):
        observation = tiny().step({"action": "get_record", "table": "ledger", "id": "b2"})[0]
        self.assertIsNone(observation["record"]["invoice_id"])
        self.assertIsNone(observation["record"]["payment_id"])

    def test_returned_records_are_copies(self):
        evaluator = tiny()
        record = evaluator.step({"action": "get_record", "table": "ledger", "id": "a5"})[0]["record"]
        record["debit_minor"] = 1
        again = evaluator.step({"action": "get_record", "table": "ledger", "id": "a5"})[0]["record"]
        self.assertEqual(again["debit_minor"], 1200000)

    def test_calculator_action(self):
        evaluator = tiny()
        observation = evaluator.step({"action": "calculator", "expression": "125000 * 465.37"})[0]
        self.assertEqual(observation, {"type": "calculation", "result": "58171250.00", "steps_remaining": 99})


class PagingTest(unittest.TestCase):
    """«Страница — максимум 50 строк, курсоры стабильны.»"""

    def test_full_walk_gives_every_record_once_in_order(self):
        evaluator = generated(5)
        for table in TABLES:
            for limit in (None, 50, 7, 1000):
                if limit == 7 and table == "ledger":
                    continue  # keep within 100 steps
                ids = walk_table(evaluator, table, limit)
                expected = [json.dumps(r, sort_keys=True) for r in evaluator._state["tables"][table]]
                self.assertEqual(ids, expected, (table, limit))
            evaluator.reset(5, CLEAN)

    def test_page_is_at_most_50_and_limit_is_clipped(self):
        evaluator = generated(5)
        observation = evaluator.step({"action": "list_records", "table": "ledger", "limit": 1000})[0]
        self.assertEqual(len(observation["records"]), 50)
        observation = evaluator.step({"action": "list_records", "table": "ledger"})[0]
        self.assertEqual(len(observation["records"]), 50)
        observation = evaluator.step({"action": "list_records", "table": "ledger", "limit": 3})[0]
        self.assertEqual(len(observation["records"]), 3)

    def test_last_page_has_null_cursor(self):
        evaluator = tiny()
        observation = evaluator.step({"action": "list_records", "table": "ledger"})[0]
        self.assertEqual(len(observation["records"]), 12)
        self.assertIsNone(observation["next_cursor"])

    def test_cursors_are_stable(self):
        evaluator = generated(5)
        first = evaluator.step({"action": "list_records", "table": "ledger", "limit": 10})[0]
        again = evaluator.step({"action": "list_records", "table": "ledger", "limit": 10})[0]
        self.assertEqual(first["next_cursor"], again["next_cursor"])
        page = evaluator.step({"action": "list_records", "table": "ledger", "cursor": first["next_cursor"]})[0]
        page_again = evaluator.step({"action": "list_records", "table": "ledger", "cursor": first["next_cursor"]})[0]
        self.assertEqual(page["records"], page_again["records"])
        other = generated(5)
        self.assertEqual(other.step({"action": "list_records", "table": "ledger", "limit": 10})[0]["next_cursor"],
                         first["next_cursor"])

    def test_interleaved_walks_each_see_every_record_once(self):
        evaluator = generated(6)
        cursors = {table: None for table in TABLES}
        cursors["ledger#2"] = None
        seen = {key: [] for key in cursors}
        done = set()
        limits = {"ledger": 50, "invoices": 13, "payments": 9, "fx": 11, "ledger#2": 50}
        while len(done) < len(cursors):
            for key in sorted(cursors):
                if key in done:
                    continue
                table = key.split("#")[0]
                observation = evaluator.step({"action": "list_records", "table": table, "cursor": cursors[key],
                                              "limit": limits[key]})[0]
                self.assertEqual(observation["type"], "records", observation)
                seen[key].extend(r.get("row_id") or r.get("doc_id") or r["date"] for r in observation["records"])
                cursors[key] = observation["next_cursor"]
                if cursors[key] is None:
                    done.add(key)
        for key, ids in seen.items():
            table = key.split("#")[0]
            expected = [r.get("row_id") or r.get("doc_id") or r["date"] for r in evaluator._state["tables"][table]]
            self.assertEqual(ids, expected, key)
            self.assertEqual(len(ids), len(set(ids)), key)

    def test_foreign_and_garbled_cursors_are_errors_that_spend_a_step(self):
        evaluator = generated(5)
        ledger_cursor = evaluator.step({"action": "list_records", "table": "ledger", "limit": 10})[0]["next_cursor"]
        other_episode_cursor = generated(7).step(
            {"action": "list_records", "table": "ledger", "limit": 10})[0]["next_cursor"]
        self.assertNotEqual(ledger_cursor, other_episode_cursor)
        bad = [("invoices", ledger_cursor), ("ledger", other_episode_cursor), ("ledger", ledger_cursor[:-1] + "x"),
               ("ledger", 10), ("ledger", ["x"]), ("ledger", ledger_cursor.upper()), ("ledger", "50"),
               ("ledger", " "), ("ledger", 0.0), ("ledger", False), ("ledger", {})]
        remaining = evaluator.steps_remaining()
        for table, cursor in bad:
            observation = evaluator.step({"action": "list_records", "table": table, "cursor": cursor})[0]
            remaining -= 1
            self.assertEqual(observation["error"]["code"], "bad_cursor", (table, cursor))
            self.assertEqual(observation["steps_remaining"], remaining)

    def test_start_cursors_give_the_first_page(self):
        """AUDIT C3: null, "", 0 and "0" all mean the first page."""
        evaluator = generated(5)
        first = evaluator.step({"action": "list_records", "table": "ledger", "limit": 10})[0]
        for cursor in ("", 0, "0"):
            observation = evaluator.step({"action": "list_records", "table": "ledger", "limit": 10,
                                          "cursor": cursor})[0]
            self.assertEqual(observation, dict(first, steps_remaining=observation["steps_remaining"]), cursor)

    def test_bad_limits_are_bad_arguments(self):
        evaluator = tiny()
        bad = (0, -1, True, 1.5, [], {}, "x", "1.5", "-1", "0", "", "٣", float("nan"), float("inf"))
        for limit in bad:
            observation = evaluator.step({"action": "list_records", "table": "ledger", "limit": limit})[0]
            self.assertEqual(observation["error"]["code"], "bad_arguments", limit)
        self.assertEqual(evaluator.steps_remaining(), 100 - len(bad))

    def test_limits_as_text_or_integral_float(self):
        """AUDIT C3: "50" and 50.0 are numbers; above 50 they are clipped like integers."""
        evaluator = generated(5)
        for limit, size in (("10", 10), (" 7 ", 7), (10.0, 10), ("500", 50), (1e300, 50)):
            observation = evaluator.step({"action": "list_records", "table": "ledger", "limit": limit})[0]
            self.assertEqual(len(observation["records"]), size, limit)

    def test_cursor_says_nothing_about_the_offset(self):
        evaluator = generated(5)
        cursor = evaluator.step({"action": "list_records", "table": "ledger", "limit": 10})[0]["next_cursor"]
        self.assertRegex(cursor, r"^[0-9a-f]{20}$")


class ErrorsTest(unittest.TestCase):
    """«Неизвестный ID или действие даёт ограниченную ошибку и расходует шаг.»"""

    CASES = [
        ("unknown_action", {"action": "delete_records", "table": "ledger"}),
        ("unknown_action", {"action": "list_records()"}),
        ("unknown_action", {"table": "ledger"}),
        ("unknown_action", "list_records"),
        ("unknown_action", None),
        ("unknown_action", {"action": 5}),
        ("unknown_table", {"action": "list_records", "table": "truth"}),
        ("unknown_table", {"action": "get_record", "table": "../private/truth.json", "id": "x"}),
        ("unknown_table", {"action": "list_records"}),
        ("unknown_id", {"action": "get_record", "table": "ledger", "id": "zzzz"}),
        ("unknown_id", {"action": "get_record", "table": "payments", "id": "k7m2qa"}),
        ("unknown_id", {"action": "get_record", "table": "ledger", "id": "k7m2qa"}),
        ("unknown_id", {"action": "get_record", "table": "fx", "id": "2026-10-01|KZT|USD"}),
        ("unknown_id", {"action": "get_document", "doc_id": "a5"}),
        ("unknown_id", {"action": "get_document", "doc_id": "p3x9wd"}),
        ("bad_arguments", {"action": "get_record", "table": "ledger"}),
        ("bad_arguments", {"action": "get_document", "doc_id": 5}),
        ("bad_arguments", {"action": "calculator"}),
        ("bad_arguments", {"action": "list_records", "table": "ledger", "args": [1]}),
        ("bad_cursor", {"action": "list_records", "table": "ledger", "cursor": "abc"}),
        ("bad_expression", {"action": "calculator", "expression": "__import__('os').system('id')"}),
        ("bad_expression", {"action": "calculator", "expression": "1/0"}),
    ]

    def test_each_error_code_spends_a_step_and_has_fixed_text(self):
        evaluator = tiny()
        for number, (code, action) in enumerate(self.CASES, 1):
            observation, reward, terminated, truncated, info = evaluator.step(action)
            self.assertEqual(observation, {"type": "error", "error": {"code": code, "message": ERRORS[code]},
                                           "steps_remaining": 100 - number}, action)
            self.assertEqual((reward, terminated, truncated, info), (0.0, False, False, {}))

    def test_unknown_ids_are_indistinguishable(self):
        evaluator = tiny()
        replies = [evaluator.step({"action": "get_record", "table": "ledger", "id": i})[0]["error"]
                   for i in ("a3x", "t1", "zzzz", "", "k7m2qa", "../../etc/passwd")]
        self.assertEqual(len({json.dumps(r) for r in replies}), 1)

    def test_episode_over_and_not_reset_do_not_spend_steps(self):
        evaluator = Evaluator(clock=FakeClock())
        for _ in range(3):
            observation, reward, terminated, truncated, info = evaluator.step({"action": "list_records"})
            self.assertEqual(observation["error"]["code"], "not_reset")
            self.assertEqual((reward, terminated, truncated, info), (0.0, True, False, {}))
        observation, _ = evaluator.load(TINY)
        self.assertEqual(observation["steps_remaining"], 100)
        self.assertEqual(evaluator.step({"action": "list_records", "table": "fx"})[0]["steps_remaining"], 99)


class ParseActionTest(unittest.TestCase):
    """CONTRACTS §5.2: name under action/tool/name/type, arguments flat or in args/arguments/params."""

    def test_shapes(self):
        expected = ("get_record", {"table": "ledger", "id": "a5"})
        shapes = [
            {"action": "get_record", "table": "ledger", "id": "a5"},
            {"tool": "get_record", "args": {"table": "ledger", "id": "a5"}},
            {"name": "get_record", "arguments": {"table": "ledger", "id": "a5"}},
            {"type": "get_record", "params": {"table": "ledger", "id": "a5"}},
            {"name": "get_record", "arguments": json.dumps({"table": "ledger", "id": "a5"})},
            {"tool": "get_record", "table": "ledger", "args": {"id": "a5"}},
            {"action": "get_record", "table": "ledger", "id": "a5", "args": None},
        ]
        for shape in shapes:
            self.assertEqual(parse_action(shape), expected, shape)

    def test_all_shapes_work_in_the_evaluator(self):
        evaluator = tiny()
        for shape in ({"tool": "get_record", "args": {"table": "ledger", "id": "a5"}},
                      {"name": "get_record", "arguments": '{"table": "ledger", "id": "a5"}'},
                      {"type": "get_record", "params": {"table": "ledger", "id": "a5"}}):
            self.assertEqual(evaluator.step(shape)[0]["record"]["row_id"], "a5")

    def test_bad_shapes(self):
        for shape, code in (({"tool": "get_record", "args": "not json"}, "bad_arguments"),
                            ({"tool": "get_record", "args": "[1, 2]"}, "bad_arguments"),
                            ({"tool": ["get_record"]}, "unknown_action"),
                            ({"tool": "get record"}, "unknown_action"),
                            ({"tool": "getrecord"}, "unknown_action"),
                            ('{"tool": "nope"}', "unknown_action"), ("[1]", "unknown_action"),
                            ("get_record", "unknown_action"), ({}, "unknown_action")):
            with self.assertRaises(ActionError) as caught:
                parse_action(shape)
            self.assertEqual(caught.exception.code, code)

    def test_submit_through_nested_arguments(self):
        evaluator = tiny()
        result = evaluator.step({"tool": "submit", "arguments": {"findings": CORRECT_TINY}})
        self.assertTrue(result[2])
        self.assertGreater(result[1], 0.9)


class TerminationTest(unittest.TestCase):
    """«Максимум 100 шагов, включая отправку. submit завершает эпизод с terminated=true. Лимит шагов
    или времени даёт truncated=true и пустой итоговый ответ.»"""

    def test_submit_terminates(self):
        evaluator = tiny()
        observation, reward, terminated, truncated, info = evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual(observation, {"type": "final", "steps_remaining": 99})
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(set(info), {"reward", "precision", "recall", "f1", "inspection_steps"})
        self.assertEqual(info["inspection_steps"], 0)
        self.assertEqual(reward, info["reward"])
        self.assertEqual(reward, 1.0)

    def test_submit_on_step_100_is_terminated(self):
        evaluator = tiny()
        for _ in range(99):
            self.assertFalse(evaluator.step({"action": "list_records", "table": "fx"})[2])
        observation, reward, terminated, truncated, info = evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual((terminated, truncated), (True, False))
        self.assertEqual(info["inspection_steps"], 99)
        self.assertEqual(observation["steps_remaining"], 0)

    def test_step_100_without_submit_truncates_with_empty_answer(self):
        evaluator = tiny()
        for _ in range(99):
            evaluator.step({"action": "nope"})
        observation, reward, terminated, truncated, info = evaluator.step({"action": "list_records", "table": "fx"})
        self.assertEqual(observation, {"type": "final", "steps_remaining": 0})
        self.assertEqual((terminated, truncated), (False, True))
        self.assertEqual(info["inspection_steps"], 100)
        self.assertEqual(info["f1"], 0.0)  # one true issue, empty answer
        self.assertEqual(reward, -1.0)

    def test_truncation_on_clean_books_scores_the_empty_answer(self):
        evaluator = generated(3)
        for _ in range(100):
            result = evaluator.step({"action": "calculator", "expression": "1+1"})
        _, reward, terminated, truncated, info = result
        self.assertTrue(truncated)
        self.assertEqual(info["f1"], 1.0)
        self.assertAlmostEqual(reward, 0.8)

    def test_time_limit_truncates_through_the_injected_clock(self):
        clock = FakeClock()
        evaluator = Evaluator(clock=clock)
        evaluator.load(TINY)
        self.assertEqual(evaluator.step({"action": "list_records", "table": "fx"})[2:4], (False, False))
        clock.t += 59
        self.assertEqual(evaluator.step({"action": "list_records", "table": "fx"})[2:4], (False, False))
        clock.t += 1
        observation, reward, terminated, truncated, info = evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        self.assertEqual((terminated, truncated), (False, True))
        # AUDIT C10: a truncated episode has no steps left, whether the steps or the time ran out.
        self.assertEqual(observation, {"type": "final", "steps_remaining": 0})
        self.assertEqual(info["inspection_steps"], 100)
        self.assertEqual(reward, -1.0)
        self.assertEqual(evaluator.step({"action": "list_records", "table": "fx"}),
                         ({"type": "error", "error": {"code": "episode_over", "message": "Episode is over."},
                           "steps_remaining": 0}, 0.0, False, True, {}))
        self.assertEqual(evaluator.steps_remaining(), 0)

    def test_reset_restarts_the_clock(self):
        clock = FakeClock()
        evaluator = Evaluator(clock=clock)
        evaluator.load(TINY)
        clock.t += 1000
        evaluator.reset()
        self.assertFalse(evaluator.step({"action": "list_records", "table": "fx"})[3])

    def test_time_never_reaches_the_policy(self):
        clock = FakeClock()
        clock.t = 987654321
        evaluator = Evaluator(clock=clock)
        values = [evaluator.load(TINY)] + full_walk(evaluator)
        self.assertNotIn("987654321", strict_json(values))

    def test_step_after_the_end_changes_nothing(self):
        evaluator = tiny()
        evaluator.step({"action": "list_records", "table": "fx"})
        evaluator.step({"action": "submit", "findings": CORRECT_TINY})
        state = (evaluator._steps, evaluator._done)
        for action in ({"action": "submit", "findings": CORRECT_TINY}, {"action": "list_records", "table": "fx"}, None):
            observation, reward, terminated, truncated, info = evaluator.step(action)
            self.assertEqual(observation, {"type": "error", "error": {"code": "episode_over",
                                                                      "message": "Episode is over."},
                                           "steps_remaining": 98})
            self.assertEqual((reward, terminated, truncated, info), (0.0, True, False, {}))
        self.assertEqual((evaluator._steps, evaluator._done), state)

    def test_invalid_structure_terminates_with_minus_one(self):
        for payload in (None, [], "x", {"episode_id": "ep_other", "findings": []}, {"episode_id": "ep_tiny0001"}):
            evaluator = tiny()
            observation, reward, terminated, truncated, info = evaluator.step({"action": "submit", "findings": payload})
            self.assertEqual((reward, terminated, truncated), (-1.0, True, False), payload)
            self.assertEqual(observation["type"], "final")

    def test_missing_findings_argument_is_invalid_structure(self):
        _, reward, terminated, _, _ = tiny().step({"action": "submit"})
        self.assertEqual((reward, terminated), (-1.0, True))


class RewardTest(unittest.TestCase):
    """«За исследование награда нулевая. Только в открытом режиме обучения терминальная награда
    возвращается политике; при проверке награда и диагностика остаются у оценщика.»"""

    def test_reward_is_zero_and_info_empty_before_the_terminal_step(self):
        for evaluator in (tiny(), generated(8)):
            transcript = full_walk(evaluator)
            terminal = [i for i, r in enumerate(transcript) if r[2] or r[3]]
            first_terminal = terminal[0]
            for observation, reward, terminated, truncated, info in transcript[:first_terminal]:
                self.assertEqual((reward, info), (0.0, {}))

    def test_train_info_has_no_per_finding_verdicts(self):
        evaluator = tiny()
        payload = json.loads(json.dumps(CORRECT_TINY))
        payload["findings"].append({**payload["findings"][0], "affected_ids": ["a1", "a2"]})
        _, reward, _, _, info = evaluator.step({"action": "submit", "findings": payload})
        self.assertEqual(set(info), {"reward", "precision", "recall", "f1", "inspection_steps"})
        self.assertEqual((info["precision"], info["recall"]), (0.5, 1.0))
        self.assertNotIn("t1", strict_json(info))

    def test_eval_hides_reward_and_writes_one_result_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/results.jsonl"
            evaluator = tiny(mode="eval", results_path=path)
            for _ in range(3):
                observation, reward, terminated, truncated, info = evaluator.step(
                    {"action": "list_records", "table": "ledger"})
                self.assertEqual((reward, info), (0.0, {}))
            observation, reward, terminated, truncated, info = evaluator.step(
                {"action": "submit", "findings": CORRECT_TINY, "mode": "train"})
            self.assertEqual((observation, reward, terminated, truncated, info),
                             ({"type": "final", "steps_remaining": 96}, 0.0, True, False, {}))
            evaluator.step({"action": "submit", "findings": CORRECT_TINY})
            evaluator.reset()
            evaluator.step({"action": "submit", "findings": None})
            lines = [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["episode_id"], "ep_tiny0001")
        self.assertAlmostEqual(lines[0]["reward"], 1 - 0.002 * 3)
        self.assertEqual((lines[0]["tp"], lines[0]["fp"], lines[0]["fn"]), (1, 0, 0))
        self.assertEqual(lines[0]["inspection_steps"], 3)
        self.assertEqual(lines[1]["reward"], -1.0)
        self.assertIsNotNone(lines[1]["structure_error"])

    def test_eval_truncation_is_logged_and_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/results.jsonl"
            evaluator = tiny(mode="eval", results_path=path)
            for _ in range(100):
                result = evaluator.step({"action": "calculator", "expression": "1"})
            self.assertEqual(result, ({"type": "final", "steps_remaining": 0}, 0.0, False, True, {}))
            lines = [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines[0]["truncated"], True)
        self.assertEqual(lines[0]["inspection_steps"], 100)
        self.assertEqual(lines[0]["reward"], -1.0)

    def test_silence_on_clean_books_beats_a_false_accusation(self):
        evaluator = generated(3)
        episode_id = evaluator._episode["public"]["manifest"]["episode_id"]
        silent = evaluator.step({"action": "submit", "findings": {"episode_id": episode_id, "findings": []}})[1]
        evaluator.reset(3, CLEAN)
        doc = evaluator._state["tables"]["invoices"][0]["doc_id"]
        accusing = evaluator.step({"action": "submit", "findings": {"episode_id": episode_id, "findings": [{
            "type": "duplicate_invoice", "affected_ids": [doc], "claimed_amount_minor": 1, "currency": "KZT",
            "evidence_ids": [doc], "explanation": ""}]}})[1]
        self.assertEqual(silent, 1.0)
        self.assertLess(accusing, silent)


class IsolationTest(unittest.TestCase):
    """«Одинаковые seed, config, версия и действия должны воспроизводить записи, переходы и награды.
    Reset изолирует состояние эпизода.»"""

    def transcript(self, evaluator, seed):
        first = evaluator.reset(seed, CLEAN)
        return strict_json([first] + full_walk(evaluator))

    def test_same_seed_config_actions_give_identical_transitions(self):
        self.assertEqual(self.transcript(Evaluator(clock=FakeClock()), 11),
                         self.transcript(Evaluator(clock=FakeClock()), 11))

    def test_reset_isolation_a_b_a(self):
        evaluator = Evaluator(clock=FakeClock())
        a = self.transcript(evaluator, 11)
        b = self.transcript(evaluator, 12)
        a_again = self.transcript(evaluator, 11)
        self.assertEqual(a, a_again)
        self.assertNotEqual(a, b)

    def test_reset_mid_episode_starts_clean(self):
        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset(11, CLEAN)
        for _ in range(30):
            evaluator.step({"action": "list_records", "table": "ledger"})
        reference = Evaluator(clock=FakeClock())
        self.assertEqual(strict_json(evaluator.reset(11, CLEAN)), strict_json(reference.reset(11, CLEAN)))
        self.assertEqual(strict_json(full_walk(evaluator)), strict_json(full_walk(reference)))

    def test_loaded_episode_restarts_identically(self):
        evaluator = Evaluator(clock=FakeClock())
        first = strict_json([evaluator.load(TINY)] + full_walk(evaluator))
        second = strict_json([evaluator.reset()] + full_walk(evaluator))
        self.assertEqual(first, second)

    def test_from_files_equals_generated(self):
        from auditgym.generate import generate
        from auditgym.storage import write_episode

        with tempfile.TemporaryDirectory() as directory:
            write_episode(generate(13, CLEAN), directory)
            loaded = Evaluator(clock=FakeClock())
            from_files = strict_json([loaded.load(directory)] + full_walk(loaded))
        self.assertEqual(from_files, self.transcript(Evaluator(clock=FakeClock()), 13))

    def test_failed_reset_keeps_the_running_episode(self):
        """AUDIT C10: a refused reset or load changes nothing."""
        evaluator = tiny()
        evaluator.step({"action": "list_records", "table": "fx"})
        for bad in (lambda: evaluator.reset(1, {"nope": 1}), lambda: evaluator.reset("1"),
                    lambda: evaluator.reset(1 << 300), lambda: evaluator.load("/nonexistent/episode")):
            with self.assertRaises(EvaluatorError):
                bad()
        observation = evaluator.step({"action": "get_record", "table": "ledger", "id": "a5"})[0]
        self.assertEqual((observation["record"]["row_id"], observation["steps_remaining"]), ("a5", 98))
        evaluator.reset()  # still restarts the loaded episode
        self.assertEqual(evaluator.steps_remaining(), 100)


if __name__ == "__main__":
    unittest.main()
