"""Checks of the criterion «Контракт и изоляция» (CASE.md §03-§04) plus the reproducibility
checks the env zone can prove and the trajectory-log check of «Инженерия и объяснение». Wired into tools/score.py by the orchestrator.

CHECKS = [(name, function)]; a function returns None on success or a failure message.
Names match the planned entries in tools/score.py. A message starting with "pending:" means
the check cannot be proven yet because another zone has not landed (it still counts as not
passed).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Callable, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
TINY = ROOT / "tests" / "fixtures" / "tiny"
CLEAN = {"errors": {}}
# Leak checks run on companies WITH errors: with an empty truth nothing can leak (audit Q7).
WITH_ERRORS = {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
               "interaction": "I1", "currencies": ["USD", "EUR"]}
TABLES = ("ledger", "invoices", "payments", "fx")
# DECISIONS A2 / CONTRACTS v2 §3.1: duplicate evidence = both docs + one posting row of each
# invoice (a1 = debit of k7m2qa, a3 = debit of z5c1ty); the two docs alone are a FP.
TINY_ANSWER = {"episode_id": "ep_tiny0001", "findings": [{
    "type": "duplicate_invoice", "affected_ids": ["k7m2qa", "z5c1ty"], "claimed_amount_minor": 5000000,
    "currency": "KZT", "evidence_ids": ["k7m2qa", "z5c1ty", "a1", "a3"], "explanation": "Same customer and total."}]}
OBS_KEYS = {
    "reset": {"type", "task", "manifest", "policy", "steps_remaining"},
    "records": {"type", "table", "records", "next_cursor", "steps_remaining"},
    "record": {"type", "table", "record", "steps_remaining"},
    "document": {"type", "table", "record", "steps_remaining"},
    "calculation": {"type", "result", "steps_remaining"},
    "error": {"type", "error", "steps_remaining"},
    "final": {"type", "steps_remaining"},
}
PRIVATE_WORDS = ("truth", "provenance", "pre_ledger", "issue_id", "evidence_sets", "expected_amount_minor",
                 "id_map", "generator_version", "private", "seed")


class _Clock:
    def __init__(self) -> None:
        self.t = 0

    def now(self):
        return self.t


def _evaluator(**kwargs):
    from auditgym.evaluator import Evaluator

    return Evaluator(clock=_Clock(), **kwargs)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _walk(step, episode_id: str) -> list:
    """Read every table, touch each error path, submit, then step once more."""
    out = []
    first = {}
    for table in TABLES:
        cursor = None
        while True:
            result = step({"action": "list_records", "table": table, "cursor": cursor})
            out.append(result)
            observation = result[0]
            if observation["type"] != "records":
                return out
            if observation["records"]:
                first.setdefault(table, observation["records"][0])
            cursor = observation["next_cursor"]
            if cursor is None:
                break
    actions = [{"action": "get_record", "table": "ledger", "id": first["ledger"]["row_id"]},
               {"action": "nope"}, {"action": "list_records", "table": "nope"},
               {"action": "get_record", "table": "ledger", "id": "nope"},
               {"action": "list_records", "table": "ledger", "cursor": "nope"},
               {"action": "calculator", "expression": "__import__('os')"},
               {"action": "calculator", "expression": "125000 * 465.37"},
               {"action": "submit", "findings": {"episode_id": episode_id, "findings": []}},
               {"action": "list_records", "table": "ledger"}]
    if "invoices" in first:
        actions.insert(0, {"action": "get_document", "doc_id": first["invoices"]["doc_id"]})
    for action in actions:
        out.append(step(action))
    return out


def _env_transcript(env, seed=None, config=None) -> list:
    first = env.reset(seed, config) if seed is not None else env.reset()
    return [first] + _walk(env.step, first[0]["manifest"]["episode_id"])


def _guard(function: Callable[[], Optional[str]]) -> Callable[[], Optional[str]]:
    def run() -> Optional[str]:
        try:
            return function()
        except Exception as exc:  # a crash is a failed check, not a crashed scorer
            return f"raised {type(exc).__name__}: {exc}"
    run.__name__ = function.__name__
    return run


# --- the 15 checks of «Контракт и изоляция» -------------------------------------------------
def reset_contract() -> Optional[str]:
    from auditgym import Env

    with Env() as env:
        result = env.reset(3, CLEAN)
    if not isinstance(result, tuple) or len(result) != 2:
        return "reset must return (observation, info)"
    observation, info = result
    if set(observation) != OBS_KEYS["reset"] or info != {}:
        return f"first observation keys {sorted(observation)}, info {info}"
    task = observation["task"]
    if observation["steps_remaining"] != 100 or not task.get("text") or len(task.get("error_types", [])) != 5:
        return "task text, error types or budget missing"
    limits = task.get("limits", {})
    if (limits.get("max_findings"), limits.get("max_evidence_ids"), limits.get("max_explanation_chars"),
            limits.get("max_steps")) != (30, 10, 1000, 100):
        return f"limits are {limits}"
    if "amount_tolerance_minor" not in task.get("tolerances", {}) or "findings_schema" not in task:
        return "tolerances or findings schema missing"
    return None


def step_contract() -> Optional[str]:
    from auditgym import Env

    with Env.from_files(TINY) as env:
        transcript = _env_transcript(env)
    for result in transcript[1:]:
        if not isinstance(result, tuple) or len(result) != 5:
            return "step must return 5 values"
        observation, reward, terminated, truncated, info = result
        if type(reward) is not float or type(terminated) is not bool or type(truncated) is not bool:
            return "reward must be float and flags bool"
        if set(observation) != OBS_KEYS.get(observation.get("type"), set()):
            return f"observation {observation.get('type')} has keys {sorted(observation)}"
        if json.loads(_json(list(result))) != list(result):
            return "step result does not round-trip through JSON"
    return None


def paging() -> Optional[str]:
    from auditgym.generate import generate

    episode = generate(5, CLEAN)
    evaluator = _evaluator()
    evaluator.reset(5, CLEAN)
    cursors = {table: None for table in TABLES}
    seen = {table: [] for table in TABLES}
    limits = {"ledger": 1000, "invoices": 7, "payments": 13, "fx": 50}
    pending = set(TABLES)
    while pending:
        for table in TABLES:
            if table not in pending:
                continue
            observation = evaluator.step({"action": "list_records", "table": table, "cursor": cursors[table],
                                          "limit": limits[table]})[0]
            if observation["type"] != "records" or len(observation["records"]) > 50:
                return f"{table}: bad page {observation.get('error')}"
            if table == "ledger" and observation["next_cursor"] is not None and len(observation["records"]) != 50:
                return "limit above 50 is not clipped to 50"
            seen[table].extend(observation["records"])
            cursors[table] = observation["next_cursor"]
            if cursors[table] is None:
                pending.discard(table)
    for table in TABLES:
        if _json(seen[table]) != _json(episode["public"][table]):
            return f"{table}: a full interleaved walk does not give every record exactly once"
    again = _evaluator()
    again.reset(5, CLEAN)
    one = again.step({"action": "list_records", "table": "ledger", "limit": 10})[0]["next_cursor"]
    two = again.step({"action": "list_records", "table": "ledger", "limit": 10})[0]["next_cursor"]
    if one != two:
        return "cursor is not stable"
    if again.step({"action": "list_records", "table": "invoices", "cursor": one})[0].get("error", {}).get("code") \
            != "bad_cursor":
        return "a cursor of another table is accepted"
    return None


def get_record_and_document() -> Optional[str]:
    evaluator = _evaluator()
    evaluator.load(TINY)
    checks = [
        ({"action": "get_record", "table": "ledger", "id": "a5"}, "record", "row_id", "a5"),
        ({"action": "get_record", "table": "invoices", "id": "k7m2qa"}, "record", "doc_id", "k7m2qa"),
        ({"action": "get_record", "table": "fx", "id": "2026-10-01|USD|KZT"}, "record", "date", "2026-10-01"),
        ({"action": "get_document", "doc_id": "h3n8vx"}, "document", "payment_id", "y6d2ck"),
        ({"action": "get_document", "doc_id": "z5c1ty"}, "document", "invoice_id", "n8v4js"),
    ]
    for action, kind, field, value in checks:
        observation = evaluator.step(action)[0]
        if observation["type"] != kind or set(observation) != OBS_KEYS[kind] or observation["record"][field] != value:
            return f"{action} returned {observation}"
    return None


def calculator() -> Optional[str]:
    from auditgym.calc import CalcError, calculate

    if calculate("125000 * 465.37") != "58171250.00" or calculate("(1 + 2) * -3 / 4") != "-2.25":
        return "arithmetic is wrong"
    for expression in ("__import__('os')", "open('x')", "9**9**9", "1e999", "x", "lambda: 1", "(1).real",
                       "1/0", "٣", "1" * 201, "'1'"):
        try:
            calculate(expression)
            return f"accepted {expression[:30]!r}"
        except CalcError:
            pass
    evaluator = _evaluator()
    evaluator.load(TINY)
    observation = evaluator.step({"action": "calculator", "expression": "__import__('os').system('id')"})[0]
    if observation.get("error", {}).get("code") != "bad_expression":
        return "code in the calculator is not a bad_expression"
    return None


def errors_spend_a_step() -> Optional[str]:
    evaluator = _evaluator()
    evaluator.load(TINY)
    cases = [("unknown_action", {"action": "rm"}), ("unknown_table", {"action": "list_records", "table": "truth"}),
             ("unknown_id", {"action": "get_record", "table": "ledger", "id": "x"}),
             ("unknown_id", {"action": "get_document", "doc_id": "x"}),
             ("bad_cursor", {"action": "list_records", "table": "ledger", "cursor": "x"}),
             ("bad_arguments", {"action": "list_records", "table": "ledger", "limit": 0}),
             ("bad_expression", {"action": "calculator", "expression": "x"})]
    for number, (code, action) in enumerate(cases, 1):
        observation, reward, terminated, truncated, info = evaluator.step(action)
        if observation.get("error", {}).get("code") != code or observation["steps_remaining"] != 100 - number:
            return f"{action}: got {observation}"
        if (reward, terminated, truncated, info) != (0.0, False, False, {}):
            return "an error step must not end the episode"
    return None


def submit_and_step_budget() -> Optional[str]:
    evaluator = _evaluator()
    evaluator.load(TINY)
    for _ in range(99):
        if evaluator.step({"action": "list_records", "table": "fx"})[2:4] != (False, False):
            return "episode ended before 100 steps"
    _, _, terminated, truncated, info = evaluator.step({"action": "submit", "findings": TINY_ANSWER})
    if (terminated, truncated) != (True, False) or info.get("inspection_steps") != 99:
        return "submit on step 100 must terminate"
    return None


def truncation() -> Optional[str]:
    evaluator = _evaluator()
    evaluator.load(TINY)
    for _ in range(100):
        result = evaluator.step({"action": "list_records", "table": "fx"})
    observation, reward, terminated, truncated, info = result
    if (terminated, truncated) != (False, True) or info.get("inspection_steps") != 100 or info.get("f1") != 0.0:
        return f"step limit: {result}"
    from auditgym.evaluator import Evaluator

    clock = _Clock()
    timed = Evaluator(clock=clock)
    timed.load(TINY)
    timed.step({"action": "list_records", "table": "fx"})
    clock.t += 60
    observation, reward, terminated, truncated, info = timed.step({"action": "submit", "findings": TINY_ANSWER})
    if (terminated, truncated) != (False, True) or info.get("inspection_steps") != 100 or reward != -1.0:
        return "time limit does not truncate with an empty answer"
    return None


def zero_reward_before_end() -> Optional[str]:
    from auditgym import Env

    with Env.from_files(TINY) as env:
        transcript = _env_transcript(env)
    for observation, reward, terminated, truncated, info in transcript[1:]:
        if terminated or truncated:
            return None
        if reward != 0.0 or info != {}:
            return "non-zero reward or non-empty info before the terminal step"
    return "episode never ended"


def eval_mode() -> Optional[str]:
    from auditgym import Env

    with tempfile.TemporaryDirectory() as directory:
        results = pathlib.Path(directory) / "results.jsonl"
        with Env.from_files(TINY, mode="eval", results_path=results) as env:
            transcript = _env_transcript(env)
        lines = results.read_text(encoding="utf-8").splitlines()
    if any(reward != 0.0 or info != {} for _, reward, _, _, info in transcript[1:]):
        return "eval mode returned a reward or info"
    if len(lines) != 1 or "reward" not in json.loads(lines[0]):
        return "eval mode must write one result line per episode"
    return None


def evaluator_out_of_process() -> Optional[str]:
    import gc
    import types

    from auditgym import Env
    from auditgym.evaluator import Evaluator

    skip = (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType, types.CodeType, types.MethodType)
    with Env.from_files(TINY) as env:
        env.reset()
        env.step({"action": "list_records", "table": "ledger"})
        if env._process.pid == os.getpid():
            return "evaluator runs in the agent process"
        seen, stack, strings = set(), [env], []
        while stack:
            current = stack.pop()
            if id(current) in seen or isinstance(current, skip):
                continue
            seen.add(id(current))
            if isinstance(current, Evaluator):
                return "an Evaluator is reachable from the client"
            if isinstance(current, str):
                strings.append(current)
            stack.extend(gc.get_referents(current))
            if hasattr(current, "__dict__"):
                stack.append(vars(current))
    text = "\n".join(strings)
    for word in ("truth", "provenance", "pre_ledger", "private", str(TINY)):
        if word in text:
            return f"{word!r} is reachable from the client object"
    return None


def no_leak() -> Optional[str]:
    from auditgym import Env
    from auditgym.generate import generate
    from auditgym.storage import load_episode, write_episode

    runs = []
    episode = load_episode(TINY)
    with Env.from_files(TINY) as env:
        runs.append((episode, _env_transcript(env), (str(TINY), "fixtures")))
    for seed in (48151623, 90210377):
        episode = generate(seed, WITH_ERRORS)
        if not episode["private"]["truth"]["issues"]:
            return "the leak check needs episodes with errors"
        with Env() as env:
            runs.append((episode, _env_transcript(env, seed, WITH_ERRORS), ()))
        with tempfile.TemporaryDirectory(prefix="episodedir_") as directory:
            write_episode(episode, directory)
            with Env.from_files(directory) as env:
                runs.append((episode, _env_transcript(env), (directory, "episodedir_")))
    for episode, transcript, extra in runs:
        private = episode["private"]
        forbidden = {issue["issue_id"] for issue in private["truth"]["issues"]}
        forbidden |= set(private["provenance"].get("id_map", {})) | set(private["provenance"])
        text = _json([list(item) for item in transcript])
        tokens = set(re.findall(r'"([^"\\]*)"', text))
        if forbidden & tokens:
            return f"private value reached the policy: {sorted(forbidden & tokens)[:3]}"
        lowered = text.lower()
        for word in PRIVATE_WORDS + extra:
            if word.lower() in lowered:
                return f"{word!r} reached the policy"
        seed = private["provenance"].get("seed")
        if isinstance(seed, int) and re.search(rf"(?<![0-9]){seed}(?![0-9])", text):
            return "the seed reached the policy"
    return None


def ids_do_not_separate_affected() -> Optional[str]:
    from auditgym.config import ConfigError
    from auditgym.generate import generate

    try:
        episode = generate(1, {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}})
    except ConfigError:
        return "pending: needs injected episodes from generate (G4)"
    public = episode["public"]
    affected = {i for issue in episode["private"]["truth"]["issues"] for i in issue["affected_ids"]}
    shapes = {(len(record["doc_id"]), set(record["doc_id"]) <= set("0123456789abcdefghjkmnpqrstvwxyz"))
              for table in ("invoices", "payments") for record in public[table]}
    if len(shapes) != 1:
        return "document ids differ in shape"
    if any(public["manifest"]["counts"][table] != len(public[table]) for table in TABLES):
        return "manifest counts differ from the tables"
    if not affected:
        return "no affected documents to compare"
    return None


def _is_int(value) -> bool:
    return type(value) is int


def files_follow_schemas() -> Optional[str]:
    offset = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
    date = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    rate = re.compile(r"^\d+(\.\d+)?$")
    evaluator = _evaluator()
    for source in ("tiny", 3, 9):
        if source == "tiny":
            evaluator.load(TINY)
        else:
            evaluator.reset(source, {"errors": {}, "currencies": ["USD", "EUR"]})
        tables = {}
        for table in TABLES:
            records, cursor = [], None
            while True:
                observation = evaluator.step({"action": "list_records", "table": table, "cursor": cursor})[0]
                records += observation["records"]
                cursor = observation["next_cursor"]
                if cursor is None:
                    break
            tables[table] = records
        for row in tables["ledger"]:
            if not offset.match(row["timestamp"]) or not _is_int(row["debit_minor"]) or not _is_int(row["credit_minor"]):
                return f"ledger row {row['row_id']} breaks the schema"
            if any(row[key] == "" for key in ("entity_id", "invoice_id", "payment_id")):
                return "missing link is not null"
        for invoice in tables["invoices"]:
            if not offset.match(invoice["issued_at"]) or not offset.match(invoice["due_at"]) \
                    or not _is_int(invoice["total_minor"]) \
                    or any(not _is_int(item["amount_minor"]) for item in invoice["line_items"]):
                return f"invoice {invoice['doc_id']} breaks the schema"
        for payment in tables["payments"]:
            if not offset.match(payment["timestamp"]) or not _is_int(payment["amount_minor"]) \
                    or not (payment["invoice_refs"] is None or isinstance(payment["invoice_refs"], list)):
                return f"payment {payment['doc_id']} breaks the schema"
        for fx in tables["fx"]:
            if not date.match(fx["date"]) or not isinstance(fx["rate"], str) or not rate.match(fx["rate"]):
                return f"fx {fx['date']} breaks the schema"
    return None


def fixed_error_outside_episode() -> Optional[str]:
    evaluator = _evaluator()
    if evaluator.step({"action": "list_records", "table": "fx"}) != (
            {"type": "error", "error": {"code": "not_reset", "message": "Call reset first."}, "steps_remaining": 0},
            0.0, False, False, {}):
        return "step before reset"
    evaluator.load(TINY)
    evaluator.step({"action": "submit", "findings": TINY_ANSWER})
    for action in ({"action": "submit", "findings": TINY_ANSWER}, {"action": "list_records", "table": "fx"}):
        result = evaluator.step(action)
        if result != ({"type": "error", "error": {"code": "episode_over", "message": "Episode is over."},
                       "steps_remaining": 99}, 0.0, True, False, {}):
            return f"step after the end: {result}"
    return None


# --- reproducibility --------------------------------------------------------------------------
def identical_transitions() -> Optional[str]:
    from auditgym import Env

    transcripts = set()
    for _ in range(2):
        with Env() as env:
            transcripts.add(_json([list(item) for item in _env_transcript(env, 17, CLEAN)]))
    return None if len(transcripts) == 1 else "two processes gave different transitions"


def reset_isolation() -> Optional[str]:
    from auditgym import Env

    with Env() as env:
        a = _json([list(item) for item in _env_transcript(env, 17, CLEAN)])
        b = _json([list(item) for item in _env_transcript(env, 18, CLEAN)])
        a_again = _json([list(item) for item in _env_transcript(env, 17, CLEAN)])
    if a != a_again:
        return "A after B differs from the first A"
    return None if a != b else "different seeds gave the same episode"


def environment_independent() -> Optional[str]:
    requests = [json.dumps({"op": "reset", "seed": 31, "config": CLEAN})]
    for action in ({"action": "list_records", "table": "ledger"}, {"action": "list_records", "table": "invoices"},
                   {"action": "get_record", "table": "fx", "id": "2026-10-01|USD|KZT"},
                   {"action": "calculator", "expression": "1/7"}, {"tool": "nope"},
                   {"action": "submit", "findings": {"episode_id": "x", "findings": []}}):
        requests.append(json.dumps({"op": "step", "action": action}))
    data = ("\n".join(requests) + "\n").encode()
    outputs = set()
    with tempfile.TemporaryDirectory() as elsewhere:
        for hashseed, tz, locale, cwd in (("0", "UTC", "C", str(ROOT)), ("4242", "Asia/Tokyo", "C.UTF-8", elsewhere),
                                          ("random", "America/Los_Angeles", "en_US.UTF-8", elsewhere)):
            environment = dict(os.environ, PYTHONHASHSEED=hashseed, TZ=tz, LC_ALL=locale, PYTHONPATH=str(ROOT))
            result = subprocess.run([sys.executable, "-m", "auditgym", "serve"], input=data, capture_output=True,
                                    env=environment, cwd=cwd, timeout=120)
            if result.returncode != 0 or len(result.stdout.splitlines()) != len(requests):
                return "server did not answer every request"
            outputs.add(result.stdout)
    return None if len(outputs) == 1 else "transcripts depend on hash seed, time zone, locale or cwd"


def from_files_equals_generated() -> Optional[str]:
    from auditgym import Env
    from auditgym.generate import generate
    from auditgym.storage import write_episode

    with Env() as env:
        generated = _json([list(item) for item in _env_transcript(env, 13, CLEAN)])
    with tempfile.TemporaryDirectory() as directory:
        write_episode(generate(13, CLEAN), directory)
        with Env.from_files(directory) as env:
            loaded = _json([list(item) for item in _env_transcript(env)])
    return None if generated == loaded else "Env.from_files differs from the generated episode"


def surface_seed_renaming() -> Optional[str]:
    from tools import metamorph

    configs = metamorph.supported_configs()
    for number, (label, config) in enumerate(configs):
        message = metamorph.check_surface(2_000_003 + 101 * number, config, 977 + number)
        if message is not None:
            return f"{label}: {message}"
    return None


def external_transform() -> Optional[str]:
    from tools import metamorph

    configs = metamorph.supported_configs()
    for number, (label, config) in enumerate(configs):
        with tempfile.TemporaryDirectory(prefix="transform_check_") as directory:
            message = metamorph.check_transform(3_000_017 + 103 * number, config, number, pathlib.Path(directory))
        if message is not None:
            return f"{label}: {message}"
    return None


def trajectory_logs() -> Optional[str]:
    from tools import metamorph
    from tools.trajectories import FIELDS, FINAL_FIELDS, scan_file, record

    configs = [config for _, config in metamorph.supported_configs()]
    with tempfile.TemporaryDirectory(prefix="trajectory_check_") as directory:
        out = pathlib.Path(directory) / "trajectories.jsonl"
        written = record(out, seeds=(4_000_037,), config=configs[-1], policies=("random", "solver", "oracle"))
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        if not lines or any(tuple(line) != (FINAL_FIELDS if line["terminated"] or line["truncated"] else FIELDS)
                            for line in lines):
            return "trajectory lines do not have exactly the case fields, the policy and, last, the metrics"
        if sum(1 for line in lines if line["terminated"] or line["truncated"]) != 2:
            return "every recorded episode must end"
        problems = scan_file(out, written["episodes"], (directory,))
        if problems:
            return problems[0]
        oracle = written["oracle"]
        if oracle is None or oracle == out or scan_file(oracle, written["episodes"], (directory,), training=False):
            return "oracle trajectories are not in their own clean file"
        mixed = pathlib.Path(directory) / "mixed.jsonl"
        mixed.write_text(out.read_text(encoding="utf-8") + oracle.read_text(encoding="utf-8"), encoding="utf-8")
        if not scan_file(mixed, written["episodes"]):
            return "the scan does not catch oracle lines in a training file"
    return None


CHECKS: list[tuple[str, Callable[[], Optional[str]]]] = [(name, _guard(function)) for name, function in [
    # «Контракт и изоляция» — names match tools/score.py
    ("reset returns (observation, info) with task, manifest, policy, budget", reset_contract),
    ("step returns (observation, reward, terminated, truncated, info), JSON-compatible", step_contract),
    ("list_records: page of at most 50 rows, stable cursors, every row exactly once", paging),
    ("get_record and get_document return only the requested record", get_record_and_document),
    ("calculator: arithmetic and decimals only; code, names and files rejected", calculator),
    ("unknown id or action: bounded error, step is spent", errors_spend_a_step),
    ("100 steps including submit; submit gives terminated=true", submit_and_step_budget),
    ("step or time limit gives truncated=true and an empty final answer", truncation),
    ("reward is zero before the terminal step", zero_reward_before_end),
    ("eval mode: reward and diagnostics stay with the evaluator", eval_mode),
    ("evaluator runs outside the agent process; truth is not on the agent side", evaluator_out_of_process),
    ("no truth, seed, provenance or paths in observations, info, errors", no_leak),
    ("ids, ordering and counts do not separate affected records from the rest", ids_do_not_separate_affected),
    ("episode files follow the schemas: ISO-8601 offsets, integer money, null links", files_follow_schemas),
    ("step after the end and step before reset give a fixed error, no state change", fixed_error_outside_episode),
    # «Воспроизводимость и обобщение»
    ("same seed, config and actions: identical transitions and rewards", identical_transitions),
    ("reset isolates episodes (A, B, A again equals A)", reset_isolation),
    ("independent of PYTHONHASHSEED, locale, system time zone, working directory", environment_independent),
    ("Env.from_files gives the same observations as the generated episode", from_files_equals_generated),
    ("surface_seed change: oracle answer maps by id table, reward is the same", surface_seed_renaming),
    ("external reorder and id rename of episode files: same oracle reward", external_transform),
    # «Инженерия и объяснение»
    ("trajectory logs are written and contain no truth", trajectory_logs),
]]


def main() -> int:
    failed = 0
    for name, function in CHECKS:
        message = function()
        failed += message is not None
        print(f"[{'ok' if message is None else 'FAIL'}] {name}" + ("" if message is None else f" -- {message}"))
    print(f"env checks: {len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
