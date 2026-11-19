"""schemas/ (docs/AUDIT.md C6): the JSON Schema files describe what the environment really produces.

The checks use tools/schema_lite.py, a validator for the small subset of JSON Schema the files use, so
there is no third-party package. They cover the shipped episodes (episodes/train and episodes/dev),
generated episodes (every error type, both scenarios, clean books, two currencies), a full walk of the
environment through its process boundary (every action, every observation type, the final info, the
eval journal) and trajectories.jsonl.
"""
import json
import pathlib
import tempfile
import unittest

from auditgym import Env
from auditgym.generate import generate
from auditgym.storage import load_episode
from policies.runner import make_policy
from tools.schema_lite import SCHEMA_DIR, check_schema, load_schema, validate

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE_SCHEMA = {"ledger": "ledger_row", "invoices": "invoice", "payments": "payment", "fx": "fx_row"}
ACTION_SCHEMA = {name: f"action-{name}" for name in ("list_records", "get_record", "get_document", "calculator", "submit")}
USED = set(TABLE_SCHEMA.values()) | set(ACTION_SCHEMA.values()) | {
    "config", "findings", "manifest", "policy", "truth", "journal_row", "trajectory_row", "info-empty", "info-final",
    "observation-reset", "observation-records", "observation-record", "observation-document",
    "observation-calculation", "observation-error", "observation-final"}
_CACHE: dict = {}


def schema(name: str) -> dict:
    if name not in _CACHE:
        _CACHE[name] = load_schema(name)
    return _CACHE[name]


class Checker(unittest.TestCase):
    def ok(self, instance, name: str, where: str = "") -> None:
        problems = validate(instance, schema(name))
        self.assertEqual(problems, [], f"{where} {name}: {problems[:3]}")

    def bad(self, instance, name: str) -> None:
        self.assertNotEqual(validate(instance, schema(name)), [], f"{name} accepted {instance!r}")

    def public(self, episode: dict, where: str) -> None:
        public = episode["public"]
        self.ok(public["manifest"], "manifest", where)
        self.ok(public["policy"], "policy", where)
        for table, name in TABLE_SCHEMA.items():
            for record in public[table]:
                self.ok(record, name, where)
        self.assertEqual(public["manifest"]["counts"], {t: len(public[t]) for t in TABLE_SCHEMA}, where)

    def private(self, episode: dict, where: str) -> None:
        self.ok(episode["private"]["truth"], "truth", where)
        config = episode["private"]["provenance"].get("config")
        if config is not None:
            self.ok(config, "config", where)

    def observation(self, observation: dict, where: str = "") -> None:
        kind = observation["type"]
        self.ok(observation, f"observation-{kind}", where)
        if kind == "reset":
            self.ok(observation["manifest"], "manifest", where)
            self.ok(observation["policy"], "policy", where)
        elif kind == "records":
            for record in observation["records"]:
                self.ok(record, TABLE_SCHEMA[observation["table"]], where)
        elif kind in ("record", "document"):
            self.ok(observation["record"], TABLE_SCHEMA[observation["table"]], where)

    def action(self, action: dict, where: str = "") -> None:
        self.ok(action, ACTION_SCHEMA[action["action"]], where)
        if action["action"] == "submit":
            self.ok(action["findings"], "findings", where)


class ValidatorTest(Checker):
    """The validator itself: each keyword accepts and rejects, and an unknown keyword is refused."""

    def check(self, schema_, good, bad):
        for item in good:
            self.assertEqual(validate(item, schema_), [], item)
        for item in bad:
            self.assertNotEqual(validate(item, schema_), [], item)

    def test_type(self):
        self.check({"type": "integer"}, [0, -3], [1.5, 1.0, True, "1", None])
        self.check({"type": "number"}, [0, 1.5], [True, "1"])
        self.check({"type": ["string", "null"]}, ["a", None], [1, []])
        self.check({"type": "object"}, [{}], [[], "x"])
        self.check({"type": "array"}, [[]], [{}, "x"])
        self.check({"type": "boolean"}, [True, False], [0, "true"])

    def test_numbers_strings_enum(self):
        self.check({"type": "integer", "minimum": 1, "maximum": 3}, [1, 3], [0, 4])
        self.check({"type": "string", "pattern": "^a[0-9]$"}, ["a1"], ["a", "a12", "b1"])
        self.check({"enum": ["x", 1]}, ["x", 1], ["y", True, 2])

    def test_arrays(self):
        self.check({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
                   [[1], [1, 2]], [[], [1, 2, 3], ["a"]])

    def test_objects(self):
        closed = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
                  "required": ["a"], "additionalProperties": False}
        self.check(closed, [{"a": 1}, {"a": 1, "b": "x"}], [{}, {"a": "1"}, {"a": 1, "c": 2}])
        self.check({"type": "object", "additionalProperties": {"type": "integer"}}, [{"x": 1}], [{"x": "1"}])
        self.check({"type": "object", "properties": {"a": {"type": "integer"}}}, [{"a": 1, "z": []}], [{"a": []}])

    def test_paths_are_reported(self):
        problems = validate({"rows": [{"n": "x"}]}, {"type": "object", "properties": {"rows": {
            "type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}}}}}})
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("$.rows[0].n:"), problems)

    def test_an_unsupported_keyword_is_refused(self):
        self.assertEqual(check_schema({"type": "object", "properties": {"a": {"oneOf": []}}}),
                         ["$.a: unsupported keyword 'oneOf'"])
        self.assertEqual(check_schema({"$ref": "x.json"}), ["$: unsupported keyword '$ref'"])
        self.assertEqual(check_schema({"type": "integr"}), ["$: unknown type 'integr'"])
        self.assertEqual(check_schema({"pattern": "("}), ["$: bad pattern"])


class SchemaFilesTest(Checker):
    def test_every_file_is_inside_the_subset_and_is_used(self):
        files = {path.name[:-len(".schema.json")] for path in SCHEMA_DIR.glob("*.schema.json")}
        self.assertEqual(files, USED, "a schema file is not covered by these tests, or a covered one is missing")
        for name in sorted(files):
            body = schema(name)
            self.assertEqual(check_schema(body), [], name)
            self.assertEqual(body["$schema"], "https://json-schema.org/draft/2020-12/schema", name)
            self.assertTrue(body.get("title") and body.get("description"), name)

    def test_the_schemas_reject_what_they_should(self):
        self.bad({"row_id": "a"}, "ledger_row")
        self.bad({"doc_id": "a", "payment_id": "b", "timestamp": "2026-10-01 10:00", "currency": "USD",
                  "amount_minor": 1, "invoice_refs": None}, "payment")
        self.bad({"doc_id": "a", "payment_id": "b", "timestamp": "2026-10-01T10:00:00+05:00", "currency": "USD",
                  "amount_minor": 1.5, "invoice_refs": None}, "payment")
        self.bad({"type": "final", "steps_remaining": 5, "seed": 1}, "observation-final")
        self.bad({"type": "final", "steps_remaining": 101}, "observation-final")
        self.bad({"reward": 0.0}, "info-empty")
        self.bad({"errors": {"duplicate_invoice": -1}}, "config")
        self.bad({"errors": {"made_up": 1}}, "config")
        self.bad({"surface_seed": "1"}, "config")
        self.bad({"episode_id": "x", "findings": [{"type": "other", "affected_ids": [], "claimed_amount_minor": 1,
                                                   "currency": "USD", "evidence_ids": []}]}, "findings")
        self.bad({"action": "list_records", "table": "ledger", "limit": 51}, "action-list_records")
        self.ok({}, "config")
        self.ok({"errors": {}, "interaction": "I1", "surface_seed": None, "size": {"invoices": [48, 100]}}, "config")


class EpisodeFilesTest(Checker):
    def test_shipped_episodes(self):
        directories = sorted((ROOT / "episodes").glob("*/ep_*"))
        self.assertEqual(len(directories), 20)
        for directory in directories:
            with self.subTest(episode=directory.name):
                episode = load_episode(directory)
                where = f"{directory.parent.name}/{directory.name}"
                self.public(episode, where)
                self.private(episode, where)
                self.assertEqual(episode["private"]["truth"]["episode_id"], episode["public"]["manifest"]["episode_id"])

    def test_fixture(self):
        # Hand-written before the generator existed: its policy.json predates the published rules, so only
        # the manifest, the tables and the truth are held to the schemas.
        episode = load_episode(ROOT / "tests" / "fixtures" / "tiny")
        self.ok(episode["public"]["manifest"], "manifest", "tiny")
        for table, name in TABLE_SCHEMA.items():
            for record in episode["public"][table]:
                self.ok(record, name, "tiny")
        self.ok(episode["private"]["truth"], "truth", "tiny")

    def test_generated_episodes(self):
        both = {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1}}
        configs = [None, {"errors": {}}, {"errors": {}, "distractors": False},
                   dict(both, currencies=["USD", "EUR"], timezone="-11:00"),
                   {"errors": {"duplicate_invoice": 2}, "interaction": "I2", "base_currency": "USD",
                    "currencies": ["EUR"], "period": {"start": "2027-02-01", "months": 2}},
                   {"errors": {"fx_error": 1, "allocation_gap": 1}, "interaction": "I1",
                    "currencies": ["CNY", "RUB"], "surface_seed": 5}]
        for index, config in enumerate(configs):
            for seed in (1, 2):
                with self.subTest(config=index, seed=seed):
                    episode = generate(seed, config)
                    where = f"config {index} seed {seed}"
                    self.public(episode, where)
                    self.private(episode, where)

    def test_the_four_types_and_both_scenarios_appear_in_the_shipped_truth(self):
        seen = set()
        for directory in (ROOT / "episodes").glob("*/ep_*"):
            seen |= {issue["type"] for issue in load_episode(directory)["private"]["truth"]["issues"]}
        self.assertEqual(seen, {"duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff"})


class WalkTest(Checker):
    CONFIG = {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
              "interaction": "I1", "currencies": ["USD", "EUR"]}

    def walk(self, env: Env, seed: int, final_info: str) -> int:
        """A solver policy through the process boundary; every action, observation and info is checked."""
        policy = make_policy("solver_policy")
        observation, info = env.reset(seed, self.CONFIG)
        self.observation(observation, "reset")
        self.ok(info, "info-empty", "reset")
        policy.reset(observation)
        kinds, steps = {observation["type"]}, 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy.act(observation)
            self.action(action, f"step {steps + 1}")
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            self.observation(observation, f"step {steps}")
            kinds.add(observation["type"])
            self.assertIs(type(reward), float)
            self.ok(info, final_info if terminated else "info-empty", f"step {steps}")
        self.assertTrue({"reset", "records", "final"} <= kinds, kinds)
        return steps

    def test_train_walk(self):
        with Env(mode="train") as env:
            self.assertGreater(self.walk(env, 11, "info-final"), 3)

    def test_eval_walk_and_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory) / "results.jsonl"
            with Env(mode="eval", results_path=journal) as env:
                self.walk(env, 12, "info-empty")
            lines = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 1)
        self.ok(lines[0], "journal_row", "journal")
        self.assertEqual(lines[0]["mode"], "eval")

    def test_single_records_documents_errors_and_the_calculator(self):
        with Env(mode="train") as env:
            observation, _ = env.reset(13, self.CONFIG)
            self.observation(observation)
            episode_id = observation["manifest"]["episode_id"]
            page = env.step({"action": "list_records", "table": "payments"})[0]
            self.observation(page)
            payment = page["records"][0]
            invoices = env.step({"action": "list_records", "table": "invoices", "limit": 5})[0]
            self.observation(invoices)
            fx = env.step({"action": "list_records", "table": "fx", "limit": 1})[0]
            key = f"{fx['records'][0]['date']}|{fx['records'][0]['source_currency']}|{fx['records'][0]['target_currency']}"
            sent = [{"action": "get_record", "table": "invoices", "id": invoices["records"][0]["doc_id"]},
                    {"action": "get_record", "table": "ledger", "id": "no-such-id"},
                    {"action": "get_record", "table": "fx", "id": key},
                    {"action": "get_document", "doc_id": payment["doc_id"]},
                    {"action": "calculator", "expression": "125000 * 465.37"},
                    {"action": "calculator", "expression": "1 +"},
                    {"action": "list_records", "table": "ledger", "cursor": "not-a-cursor"},
                    {"action": "launch_rockets"}]
            kinds = []
            for action in sent:
                if action["action"] != "launch_rockets":
                    self.action(action)
                observation, _, terminated, truncated, info = env.step(action)
                self.observation(observation, str(action))
                self.assertFalse(terminated or truncated)
                self.ok(info, "info-empty")
                kinds.append(observation["type"])
            self.assertEqual(kinds, ["record", "error", "record", "document", "calculation", "error", "error", "error"])
            answer = {"action": "submit", "findings": {"episode_id": episode_id, "findings": []}}
            self.action(answer)
            final = env.step(answer)
            self.observation(final[0])
            self.ok(final[4], "info-final")
            self.assertEqual((final[2], final[3]), (True, False))
            over = env.step({"action": "calculator", "expression": "1"})
            self.observation(over[0])
            self.assertEqual(over[0]["error"]["code"], "episode_over")

    def test_truncation_reply(self):
        with Env(mode="train") as env:
            env.reset(14, self.CONFIG)
            reply = None
            for number in range(100):
                reply = env.step({"action": "calculator", "expression": "1"})
                self.observation(reply[0])
                if number < 99:
                    self.ok(reply[4], "info-empty")
            self.assertEqual((reply[2], reply[3], reply[0]["steps_remaining"]), (False, True, 0))
            self.ok(reply[4], "info-final")


class TrajectoriesTest(Checker):
    def test_every_row_of_the_shipped_file(self):
        rows = [json.loads(line) for line in (ROOT / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertGreater(len(rows), 100)
        for number, row in enumerate(rows, 1):
            self.ok(row, "trajectory_row", f"line {number}")
            self.observation(row["observation"], f"line {number}")
            if row["action"] is not None:
                self.action(row["action"], f"line {number}")
            if not (row["terminated"] or row["truncated"]):
                self.assertEqual(row["reward"], 0.0, f"line {number}")
                self.assertNotIn("metrics", row, f"line {number}")

    def test_every_baseline_policy_plays_and_every_episode_ends_with_metrics(self):
        rows = [json.loads(line) for line in (ROOT / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["policy"] for row in rows}, {"random_explorer", "reconcile", "flag_all"})
        final = [row for row in rows if row["terminated"] or row["truncated"]]
        starts = [row for row in rows if row["step"] == 0]
        self.assertEqual(len(final), len(starts))
        self.assertEqual({(r["episode_id"], r["policy"]) for r in final},
                         {(r["episode_id"], r["policy"]) for r in starts})
        episodes = {row["episode_id"] for row in rows}
        self.assertGreaterEqual(len(episodes), 12)  # train and dev books
        for row in final:
            metrics = row["metrics"]
            self.ok(row, "trajectory_row")
            # reward = clip(2 F1 - 1 - 0.002 steps, -1, 1): the metrics are the ones of that reward
            self.assertAlmostEqual(row["reward"], max(-1.0, min(1.0, 2 * metrics["f1"] - 1 - 0.002 * (row["step"] - 1))),
                                   places=6, msg=row["episode_id"])

    def test_the_row_schema_wants_policy_and_closed_metrics(self):
        row = {"episode_id": "e", "policy": "p", "step": 0, "action": None, "observation": {}, "reward": 0.0,
               "terminated": False, "truncated": False}
        self.ok(row, "trajectory_row")
        self.bad({key: value for key, value in row.items() if key != "policy"}, "trajectory_row")
        self.bad(dict(row, metrics={"precision": 1.0, "recall": 1.0}), "trajectory_row")
        self.bad(dict(row, metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1}), "trajectory_row")
        self.ok(dict(row, metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0}), "trajectory_row")


if __name__ == "__main__":
    unittest.main()
