"""Block 1 of the published checks: C01-C20 against judge_adapter, on the fixtures C, E, L, U.

Each test is named after the check it implements; docs/ADAPTER.md has the table check -> test.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import unittest

import judge_adapter
from tests.adapter import helpers
from tests.adapter.helpers import closed, fixture, list_action, new_env, submit_action

TIMED_SERVER = pathlib.Path(__file__).resolve().parent / "timed_server.py"
FORBIDDEN_KEYS = {"truth", "provenance", "seed", "issues", "issue_id", "evidence_sets", "expected_amount_minor",
                  "id_map", "pre_ledger", "reward", "f1", "precision", "recall", "tp", "fp", "fn",
                  "inspection_steps", "structure_error"}


def tearDownModule():
    helpers.cleanup()


def all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_keys(item)


class ContractCase(unittest.TestCase):
    def setUp(self):
        self.env, self.journal = new_env(self.id().rsplit(".", 1)[-1][:24])
        self.addCleanup(self.env.close)

    def start(self, name: str):
        return self.env.reset(0, closed(fixture(name)["dir"]))

    def step(self, action):
        return self.env.step(action)


class C01to04(ContractCase):
    def test_C01_reset_returns_observation_and_info_json(self):
        result = self.start("C")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        observation, info = result
        self.assertIsInstance(observation, dict)
        self.assertIsInstance(info, dict)
        self.assertTrue(helpers.is_json(observation) and helpers.is_json(info))
        exchange = self.step(list_action())
        self.assertTrue(helpers.is_json(list(exchange)))

    def test_C02_first_observation_has_task_manifest_policy_budget_and_no_secret(self):
        observation, info = self.env.reset(0, closed(fixture("S")["dir"]))
        self.assertEqual(set(observation), {"task", "manifest", "policy", "remaining_steps"})
        self.assertIsInstance(observation["task"], str)
        self.assertTrue(observation["task"].strip())
        self.assertIsInstance(observation["manifest"], dict)
        self.assertTrue(observation["policy"])
        self.assertEqual(observation["remaining_steps"], 100)
        text = json.dumps([observation, info])
        self.assertNotIn(str(helpers.SEED_VALUE), text)
        self.assertFalse(FORBIDDEN_KEYS & set(all_keys(observation)))
        self.assertFalse(FORBIDDEN_KEYS & set(all_keys(info)))
        truth = fixture("S")["episode"]["private"]["truth"]["issues"][0]
        for hidden in truth["evidence_sets"][0] + truth["affected_ids"]:
            self.assertNotIn(hidden, text)  # not even as an id in the task or the policy
        self.assertNotIn(fixture("S")["dir"], text)

    def test_C03_manifest_matches_the_data(self):
        manifest = self.start("C")[0]["manifest"]
        for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                    "counts", "rounding"):
            self.assertIn(key, manifest)
        public = fixture("C")["episode"]["public"]
        self.assertEqual(manifest["episode_id"], public["manifest"]["episode_id"])
        self.assertEqual(manifest["timezone"], public["manifest"]["timezone"])
        self.assertTrue(manifest["period_start"] < manifest["period_end"])
        self.assertEqual(manifest["rounding"], public["policy"]["rounding"])
        for table in helpers_tables():
            seen, cursor = 0, None
            while True:
                observation, *_ = self.step(list_action(table, cursor))
                seen += len(observation["result"]["records"])
                cursor = observation["result"]["next_cursor"]
                if cursor is None:
                    break
            self.assertEqual(manifest["counts"][table], seen, table)
            self.assertEqual(seen, len(public[table]), table)

    def test_C04_list_records_returns_the_five_tuple(self):
        self.start("C")
        result = self.step(list_action("invoices", None, 5))
        self.assertEqual(len(result), 5)
        observation, reward, terminated, truncated, info = result
        self.assertIs(type(terminated), bool)
        self.assertIs(type(truncated), bool)
        self.assertTrue(helpers.is_json(list(result)))
        self.assertEqual(set(observation), {"result", "error", "remaining_steps"})
        self.assertIsNone(observation["error"])
        self.assertEqual(set(observation["result"]), {"records", "next_cursor"})
        self.assertEqual(observation["remaining_steps"], 99)
        self.assertEqual(len(observation["result"]["records"]), 5)


def helpers_tables():
    return judge_adapter.TABLES


class C05to08(ContractCase):
    def test_C05_pagination_of_L_has_no_gaps_and_no_repeats(self):
        self.start("L")
        expected = [row["row_id"] for row in fixture("L")["episode"]["public"]["ledger"]]
        self.assertGreaterEqual(len(expected), 101)
        seen, cursor, pages = [], None, 0
        while True:
            observation, *_ = self.step(list_action("ledger", cursor, 50))
            records = observation["result"]["records"]
            self.assertLessEqual(len(records), 50)
            seen += [row["row_id"] for row in records]
            cursor, pages = observation["result"]["next_cursor"], pages + 1
            if cursor is None:
                break
        self.assertGreaterEqual(pages, 3)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(sorted(seen), sorted(expected))
        self.assertEqual(seen, expected)  # and in the order of the table

    def test_C06_same_cursor_twice_gives_the_same_page(self):
        self.start("L")
        first, *_ = self.step(list_action("ledger", None, 50))
        cursor = first["result"]["next_cursor"]
        a, *_ = self.step(list_action("ledger", cursor, 50))
        b, *_ = self.step(list_action("ledger", cursor, 50))
        self.assertEqual(a["result"], b["result"])
        self.assertEqual(a["remaining_steps"] - 1, b["remaining_steps"])

    def test_C07_get_record_returns_the_right_record_only(self):
        fx = fixture("L")
        self.start("L")
        observation, *_ = self.step({"name": "get_record", "args": {"table": "ledger", "id": fx["R1"]}})
        expected = next(r for r in fx["episode"]["public"]["ledger"] if r["row_id"] == fx["R1"])
        self.assertIsNone(observation["error"])
        self.assertEqual(observation["result"], expected)
        self.assertFalse(FORBIDDEN_KEYS & set(all_keys(observation)))

    def test_C08_get_document_returns_the_right_document(self):
        fx = fixture("C")
        self.start("C")
        observation, *_ = self.step({"name": "get_document", "args": {"doc_id": fx["D1"]}})
        expected = next(r for r in fx["episode"]["public"]["invoices"] if r["doc_id"] == fx["D1"])
        self.assertIsNone(observation["error"])
        self.assertEqual(observation["result"], expected)
        # a payment document is a document too
        payment = fx["episode"]["public"]["payments"][0]
        observation, *_ = self.step({"name": "get_document", "args": {"doc_id": payment["doc_id"]}})
        self.assertEqual(observation["result"], payment)
        self.assertFalse(FORBIDDEN_KEYS & set(all_keys(observation)))


class C09to14(ContractCase):
    def test_C09_calculator_is_decimal_arithmetic(self):
        self.start("C")
        observation, *_ = self.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
        self.assertIsNone(observation["error"])
        self.assertEqual(observation["result"], "0.3")
        self.assertIsInstance(observation["result"], str)

    def test_C10_calculator_refuses_anything_but_arithmetic(self):
        self.start("C")
        marker = pathlib.Path(helpers.scratch()) / "secret_marker.txt"
        marker.write_text("TOP-SECRET-CONTENT")
        spent = 0
        for expression in (f"open('{marker}').read()", "__import__('os').system('id')", "1+1; import os",
                           "(lambda: 1)()", "'a'*10", str(marker), "[x for x in range(3)]", "2**10000000", "1/0"):
            observation, reward, terminated, truncated, info = self.step(
                {"name": "calculator", "args": {"expression": expression}})
            spent += 1
            self.assertIsNone(observation["result"], expression)
            self.assertIn("code", observation["error"])
            self.assertNotIn("TOP-SECRET", json.dumps(observation))
            self.assertFalse(terminated or truncated)
            self.assertEqual(observation["remaining_steps"], 100 - spent)

    def test_C11_get_record_of_U_is_a_bounded_error(self):
        fx = fixture("C")
        self.start("C")
        observation, reward, terminated, truncated, info = self.step(
            {"name": "get_record", "args": {"table": "invoices", "id": fx["U"]}})
        self.assertIsNone(observation["result"])
        self.assertEqual(observation["error"]["code"], "unknown_id")
        self.assertNotIn(fx["U"], json.dumps(observation))
        self.assertLess(len(observation["error"]["message"]), 80)
        self.assertEqual(observation["remaining_steps"], 99)
        self.assertFalse(terminated or truncated)
        again, *_ = self.step({"name": "get_record", "args": {"table": "ledger", "id": fx["R1"]}})  # alive
        self.assertIsNone(again["error"])

    def test_C12_get_document_of_U_is_a_bounded_error(self):
        fx = fixture("C")
        self.start("C")
        observation, *_ = self.step({"name": "get_document", "args": {"doc_id": fx["U"]}})
        self.assertIsNone(observation["result"])
        self.assertEqual(observation["error"]["code"], "unknown_id")
        self.assertEqual(observation["remaining_steps"], 99)
        text = json.dumps(observation)
        for invoice in fx["episode"]["public"]["invoices"][:20]:
            self.assertNotIn(invoice["doc_id"], text)
        self.assertNotIn(fx["U"], text)

    def test_C13_unknown_action_name_is_a_bounded_error_and_a_step(self):
        self.start("C")
        spent = 0
        for action in ({"name": "delete_everything", "args": {}}, {"name": "List_Records", "args": {}},
                       {"args": {}}, "list_records", None, {"name": 5}, ["list_records"]):
            observation, reward, terminated, truncated, info = self.step(action)
            spent += 1
            self.assertEqual(observation["error"]["code"], "unknown_action", repr(action))
            self.assertIsNone(observation["result"])
            self.assertEqual(observation["remaining_steps"], 100 - spent)
            self.assertFalse(terminated or truncated)
        observation, *_ = self.step({"name": "get_record", "args": "not an object"})  # a known name, bad args
        self.assertEqual(observation["error"]["code"], "bad_arguments")
        self.assertEqual(observation["remaining_steps"], 100 - spent - 1)

    def test_C14_errors_do_not_give_free_actions_and_do_not_spoil_access(self):
        fx = fixture("C")
        self.start("C")
        read = {"name": "get_record", "args": {"table": "ledger", "id": fx["R1"]}}
        first, *_ = self.step(read)
        bad, *_ = self.step({"name": "get_record", "args": {"table": "ledger", "id": fx["U"]}})
        last, *_ = self.step(read)
        self.assertEqual([first["remaining_steps"], bad["remaining_steps"], last["remaining_steps"]], [99, 98, 97])
        self.assertEqual(first["result"], last["result"])


class C15to19(ContractCase):
    def test_C15_submit_ends_the_episode_and_later_actions_change_nothing(self):
        fx = fixture("E")
        self.start("E")
        self.step(list_action("invoices", None, 5))
        observation, reward, terminated, truncated, info = self.step(submit_action(helpers.empty_findings(fx)))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(observation["result"], {"accepted": True})
        rows = helpers.finished(self.journal)
        self.assertEqual(len(rows), 1)
        decided = {key: rows[0][key] for key in ("tp", "fp", "fn", "reward", "inspection_steps")}
        self.assertEqual(decided["inspection_steps"], 1)
        for action in (submit_action(helpers.oracle_findings(fx)), list_action(), {"name": "nope"}):
            observation, reward, terminated, truncated, info = self.step(action)
            self.assertEqual(observation["error"]["code"], "episode_over")
            self.assertTrue(terminated)
            self.assertEqual(reward, 0)
        after = helpers.finished(self.journal)
        self.assertEqual(len(after), 1)
        self.assertEqual({key: after[0][key] for key in decided}, decided)  # the empty answer stays: fn = 1
        self.assertEqual((decided["tp"], decided["fn"]), (0, 1))

    def test_C16_submit_as_action_100_is_accepted(self):
        fx = fixture("E")
        self.start("E")
        for _ in range(99):
            observation, _, terminated, truncated, _ = self.step(list_action("invoices", None, 1))
            self.assertFalse(terminated or truncated)
        self.assertEqual(observation["remaining_steps"], 1)
        observation, reward, terminated, truncated, info = self.step(submit_action(helpers.oracle_findings(fx)))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(observation["result"], {"accepted": True})
        self.assertEqual(observation["remaining_steps"], 0)
        row = helpers.finished(self.journal)[-1]
        self.assertEqual((row["inspection_steps"], row["steps_total"], row["tp"]), (99, 100, 1))
        self.assertFalse(row["truncated"])

    def test_C17_one_hundred_inspections_truncate_with_empty_findings(self):
        fx = fixture("E")
        self.start("E")
        for number in range(1, 101):
            observation, reward, terminated, truncated, info = self.step(list_action("invoices", None, 1))
            self.assertEqual(truncated, number == 100)
            self.assertFalse(terminated)
        self.assertEqual(observation["remaining_steps"], 0)
        rows = helpers.finished(self.journal)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["truncated"])
        self.assertEqual((rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]), (0, 0, 1))  # the answer is empty
        self.assertEqual(rows[0]["inspection_steps"], 100)
        self.assertEqual(rows[0]["reward"], -1.0)
        later, reward, terminated, truncated, info = self.step(submit_action(helpers.oracle_findings(fx)))
        self.assertTrue(truncated)
        self.assertEqual(later["error"]["code"], "episode_over")
        self.assertEqual(len(helpers.finished(self.journal)), 1)  # nothing added afterwards

    def test_C18_time_limit_truncates_with_empty_findings(self):
        """The evaluator is started with a 1 s limit instead of 60 (tests/adapter/timed_server.py); the
        mechanism is the same one the 60 s limit uses. The real 60 s run is RealTimeLimitTest (--full)."""
        journal = helpers.scratch() / "time.jsonl"
        env = judge_adapter.make_env(server_command=[sys.executable, str(TIMED_SERVER), "--time-limit", "1.0",
                                                     "--results", str(journal)])
        self.addCleanup(env.close)
        fx = fixture("E")
        env.reset(0, closed(fx["dir"]))
        env.step(list_action("invoices", None, 3))
        env.step(list_action("payments", None, 3))
        time.sleep(1.3)  # the policy is slow: its time counts
        observation, reward, terminated, truncated, info = env.step(submit_action(helpers.oracle_findings(fx)))
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0)
        row = helpers.finished(journal)[-1]
        self.assertTrue(row["truncated"])
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (0, 0, 1))  # empty findings, not the late oracle
        self.assertEqual(row["inspection_steps"], 2)
        later, _, _, still, _ = env.step(list_action())
        self.assertTrue(still)
        self.assertEqual(len(helpers.finished(journal)), 1)

    def test_C19_closed_mode_gives_the_policy_nothing_of_the_reward(self):
        fx = fixture("E")
        self.start("E")
        seen = []
        for table in ("invoices", "payments", "ledger", "fx"):
            seen.append(self.step(list_action(table, None, 20)))
        seen.append(self.step({"name": "get_record", "args": {"table": "ledger", "id": fx["U"]}}))
        seen.append(self.step(submit_action(helpers.oracle_findings(fx))))
        for observation, reward, terminated, truncated, info in seen:
            self.assertEqual(reward, 0)
            self.assertIn(type(reward), (int, float))
            self.assertEqual(info, {})
            self.assertFalse(FORBIDDEN_KEYS & set(all_keys(observation)))
        self.assertTrue(seen[-1][2])
        self.assertEqual(seen[-1][0]["result"], {"accepted": True})
        row = helpers.finished(self.journal)[-1]  # the evaluator holds the truth
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["inspection_steps"]), (1, 0, 0, 5))
        self.assertAlmostEqual(row["reward"], 1 - 0.002 * 5, places=9)
        self.assertEqual(row["reward"], self.env.last_result()["reward"])
        scored = judge_adapter.score_episode(fx["dir"], helpers.oracle_findings(fx), 5)
        self.assertEqual(scored["return"], row["reward"])
        # train mode, by contrast: the terminal reward is the policy's
        train = judge_adapter.make_env()
        self.addCleanup(train.close)
        train.reset(0, {"mode": "train", "episode_dir": fx["dir"]})
        train.step(list_action("invoices", None, 5))
        observation, reward, terminated, truncated, info = train.step(submit_action(helpers.oracle_findings(fx)))
        self.assertTrue(terminated)
        self.assertAlmostEqual(reward, 1 - 0.002 * 1, places=9)
        self.assertEqual(info["reward"], reward)


class C20(ContractCase):
    def test_C20_nothing_leaks_and_the_tools_open_no_door(self):
        for name in ("C", "S"):
            fx = fixture(name)
            public = fx["episode"]["public"]
            truth_ids = {i for issue in fx["episode"]["private"]["truth"]["issues"]
                         for i in issue["affected_ids"] + [e for s in issue["evidence_sets"] for e in s]}
            provenance = fx["episode"]["private"]["provenance"]
            outputs = []
            used = 0

            def act(action):
                nonlocal used
                if used >= 90:
                    outputs.append(self.env.reset(0, closed(fx["dir"])))
                    used = 0
                used += 1
                result = self.env.step(action)
                outputs.append(result)
                return result

            outputs.append(self.env.reset(0, closed(fx["dir"])))
            for table in ("ledger", "invoices", "payments", "fx"):
                cursor = None
                while True:
                    observation, *_ = act(list_action(table, cursor, 50))
                    cursor = observation["result"]["next_cursor"]
                    if cursor is None:
                        break
            for invoice in public["invoices"]:
                act({"name": "get_record", "args": {"table": "invoices", "id": invoice["invoice_id"]}})
                act({"name": "get_document", "args": {"doc_id": invoice["doc_id"]}})
            for payment in public["payments"]:
                act({"name": "get_document", "args": {"doc_id": payment["doc_id"]}})
            for row in public["ledger"][:60]:
                act({"name": "get_record", "args": {"table": "ledger", "id": row["row_id"]}})
            for bad in ({"name": "get_document", "args": {"doc_id": "../private/truth.json"}},
                        {"name": "get_document", "args": {"doc_id": "/etc/passwd"}},
                        {"name": "get_record", "args": {"table": "../private/truth", "id": "x"}},
                        {"name": "get_record", "args": {"table": "truth", "id": "t1"}},
                        {"name": "get_record", "args": {"table": "ledger; DROP TABLE ledger", "id": "1"}},
                        {"name": "get_record", "args": {"table": "ledger", "id": "' OR '1'='1"}},
                        {"name": "get_document", "args": {"doc_id": "file:///etc/passwd"}},
                        {"name": "get_document", "args": {"doc_id": "http://127.0.0.1/"}},
                        {"name": "list_records", "args": {"table": "ledger", "cursor": "../../x", "limit": 5}},
                        {"name": "calculator", "args": {"expression": "open('/etc/passwd').read()"}},
                        {"name": "shell", "args": {"command": "id"}},
                        {"name": "sql", "args": {"query": "select 1"}},
                        {"name": "fetch", "args": {"url": "http://example.com"}}):
                act(bad)
            text = json.dumps(outputs, ensure_ascii=False)
            keys = set(all_keys(outputs))
            self.assertFalse(FORBIDDEN_KEYS & keys, FORBIDDEN_KEYS & keys)
            self.assertNotIn(str(provenance["seed"]), text)
            self.assertNotIn(fx["dir"], text)
            self.assertNotIn("/private/", text)
            self.assertNotIn("root:", text)
            self.assertNotIn("Traceback", text)
            self.assertNotIn("truth.json", text.replace("../private/truth.json", ""))
            for _, reward, _, _, info in (o for o in outputs if len(o) == 5):
                self.assertEqual((reward, info), (0, {}))
            # truth-only ids must not be extra: any id of the truth is an open id of the data
            open_ids = {v for t in ("ledger", "invoices", "payments") for r in public[t] for k, v in r.items()
                        if isinstance(v, str)} | {
                f"{r['date']}|{r['source_currency']}|{r['target_currency']}" for r in public["fx"]}
            self.assertLessEqual(truth_ids, open_ids)
        # the evaluator lives in another process and the object holds no episode data
        process = self.env._env._process
        self.assertNotEqual(process.pid, __import__("os").getpid())
        held = json.dumps({k: v for k, v in vars(self.env).items() if k not in ("_env",)}, default=str)
        self.assertNotIn("truth", held)
        self.assertNotIn(str(provenance["seed"]), held)


class RealTimeLimitTest(unittest.TestCase):
    """C18 at the real 60 s (the full check runs it; a fast run uses the 1 s variant above)."""

    def test_C18_real_sixty_seconds(self):
        fx = fixture("E")
        env, journal = new_env("real-time")
        self.addCleanup(env.close)
        env.reset(0, closed(fx["dir"]))
        env.step(list_action("invoices", None, 3))
        time.sleep(60.5)
        observation, reward, terminated, truncated, info = env.step(submit_action(helpers.oracle_findings(fx)))
        self.assertTrue(truncated)
        self.assertEqual(reward, 0)
        row = helpers.finished(journal)[-1]
        self.assertEqual((row["tp"], row["fn"], row["inspection_steps"]), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
