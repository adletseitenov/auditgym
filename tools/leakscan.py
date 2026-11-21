"""Leak scan of everything that reaches the agent side (CASE §03: «Эталон не должен утекать через
наблюдения, пути, ошибки, debug info или объяснения награды»).

    python -m tools.leakscan [--episodes N] [--seed-base S]

For N episodes (varied supported configs; every other one is loaded from files so paths are in
play) and both modes (train, eval), the server process is driven by two scripts:

  honest   page every table, open documents (affected ones included), calculator, submit the
           solver's answer, step after the end
  probe    malformed protocol lines (not JSON, not an object, unknown op, op not a string, bad
           UTF-8, an overlong line, load after the first request, bad reset seeds and configs),
           unknown actions, tables and ids (issue ids, internal keys, pre-injection-only ids,
           paths), garbled cursors, bad limits, bad expressions, bad nested arguments, every
           submit variant (not an object, missing fields, wrong episode id, too many findings,
           too large, guessing, the oracle, empty), step after the end, truncation by 100 steps

Channels scanned, each reported on its own line:
  stdout   every protocol reply byte (observations, info, error messages)
  client   every string reachable from an Env object after honest episodes
  stderr   the server's stderr (Env sends it to /dev/null; scanned anyway as a separate channel)

What is searched for: truth issue ids, internal keys of provenance id_map, the seed (digit-bounded,
seeds are >= 1 000 000), config-only key names, provenance and truth key names, private file
names, the episode directory and the repository path, values that exist only before injection
(pre_* tables and provenance `before` records: strings absent from the public tables, integers
>= 10 000 absent from the public tables), per-finding verdict keys (tp, fp, fn, pairs,
structure_error), exception texts, and wall-clock time (epoch-like numbers near now that stand on their own and are not served data; digit
runs inside cursors and ids are chance, not a clock; a repeat of the honest walk after 1.1 s must be
byte-identical). Reply shapes are checked too: fixed error
texts, observation keys per type, info == {} before the end, the five metrics at the end in
train, reward 0.0 and info {} everywhere in eval.

Surface statistics on what the agent can read, in three views. For each error type:
  documents  affected documents against documents no issue touches: id rank, table position,
             second / minute / hour of the document time, posting time equals document time,
             paid / unpaid (invoices), line-item count, payment terms in days, reference count
             and refund (payments);
  ledger     the ledger rows an issue cites as evidence against the other rows of the SAME account
             and side (so a definition such as "the gap is on `unallocated`" is not a finding): id
             rank, position, second / minute / hour, round amounts (multiple of 100, of 1 000),
             number of digits, size of the posting group, posting time equals document time;
  pairs      the two invoices of a duplicate_invoice against other pairs of invoices of one entity
             with one total (the legitimate repeats): gap in days, order by position and by id,
             id distance, same time of day, same items, same service period, which one is paid and
             posted. Only pairs of two posted invoices are compared: a pair with an unposted invoice
             (such as a resent copy, decoy A) is ruled out by the definition itself, so its
             features would "find" the definition, not a leak.
The separation measure is the AUC (Mann-Whitney, ties averaged; a boolean is 0/1). A row FAILs only
when it is both large and significant: |AUC - 0.5| >= MIN_EFFECT (0.10) and the two-sided p-value of
the Mann-Whitney z statistic (normal approximation, tie-corrected variance) is below
ALPHA / m, with ALPHA = 0.01 and m the number of rows tested (Bonferroni); at least MIN_AFFECTED (8)
positives are needed, fewer is reported as "few". So 200 samples of AUC 0.55 do not fail (the effect is
small), and 10 samples of AUC 1.0 do (the count is small but the separation cannot be chance).
Known generator defects awaiting a fix are listed in PENDING with their docs/AUDIT.md number: they are
printed as `pending:` and do not fail the run, but they do not pass either; a feature that is the
error's own definition (period_cutoff: table order is by document date) is reported as `explained`.

Side channels: `get_record` on affected vs unaffected documents, reply size and round-trip time,
medians reported. Fail only on a clear separation: |AUC - 0.5| > 0.35 for size, > 0.40 for time.

Exit 1 on any FAIL; `pending:` items do not fail the run, but `--strict` fails it while any is left.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import types

from tools.metamorph import (CANDIDATES, TABLES, oracle_answer, page_all, solver_answer, supported_configs)

ROOT = pathlib.Path(__file__).resolve().parent.parent
MIN_EFFECT = 0.10   # |AUC - 0.5| below this is never a finding, however many samples
ALPHA = 0.01        # family-wise error rate over all the rows tested (Bonferroni)
SIZE_THRESHOLD = 0.35
TIME_THRESHOLD = 0.40
MIN_AFFECTED = 8
INFO_KEYS = {"reward", "precision", "recall", "f1", "inspection_steps"}
OBS_KEYS = {
    "reset": {"type", "task", "manifest", "policy", "steps_remaining"},
    "records": {"type", "table", "records", "next_cursor", "steps_remaining"},
    "record": {"type", "table", "record", "steps_remaining"},
    "document": {"type", "table", "record", "steps_remaining"},
    "calculation": {"type", "result", "steps_remaining"},
    "error": {"type", "error", "steps_remaining"},
    "final": {"type", "steps_remaining"},
}
# JSON keys that never appear in anything the agent receives.
FORBIDDEN_KEYS = {"seed", "config", "generator_version", "attempt", "steps", "id_map", "operator", "params",
                  "before", "after", "issues", "issue_id", "evidence_sets", "expected_amount_minor", "truth",
                  "provenance", "tp", "fp", "fn", "pairs", "verdict", "verdicts", "matched", "structure_error",
                  "pre_ledger", "pre_invoices", "pre_payments", "private", "path", "directory", "results",
                  "surface_seed", "background_pairs", "interaction", "distractors", "size", "trace", "exception"}
# Substrings (case-insensitive) that never appear in the raw bytes.
FORBIDDEN_WORDS = ("truth", "provenance", "pre_ledger", "pre_invoices", "pre_payments", "private", "id_map",
                   "evidence_sets", "expected_amount", "issue_id", "generator_version", "surface_seed",
                   "background_pairs", "traceback", "exception", "error:", ".jsonl", ".json", ".csv",
                   "/users/", "/home/", "/tmp", "/var/folders", "c:\\\\", "auditgym/")
# Known generator defects awaiting a fix are listed here as `pending:` with their docs/AUDIT.md number.
# They do not fail the run and they do not pass it either: `--strict` exits 1 while any is left, and the
# summary line names them. The ledger view found this one: the amount of an allocation gap and the
# amount of an overpayment (both on `unallocated`) have different distributions.
PENDING: dict = {
    ("allocation_gap", "digits"): "pending: AUDIT L6, a gap and an overpayment have different amounts",
    ("allocation_gap", "round_100"): "pending: AUDIT L6, a gap and an overpayment have different amounts",
    ("allocation_gap", "round_1000"): "pending: AUDIT L6, a gap and an overpayment have different amounts",
}
EXPLAINED = {
    ("period_cutoff", "position"): "table order is by document date; a date outside the period is the definition",
    ("allocation_gap", "paid"): "a gap needs a payment, so the affected invoice is paid: the definition",
    ("allocation_gap", "group_size"): "the gap payment's group holds the unallocated row of the definition; "
                                      "its receivable credit is cited as evidence (docs/AUDIT.md D1)",
    ("duplicate_invoice", "same_service_period"): "a duplicate repeats the same service: the definition "
                                                  "(a legitimate repeat bills another period)",
}


# --- raw server --------------------------------------------------------------------------------
class RawServer:
    """The agent side of the pipe, byte for byte. stderr is captured as a separate channel."""

    def __init__(self, mode: str, allow_load: bool) -> None:
        command = [sys.executable, "-m", "auditgym", "serve", "--mode", mode]
        if not allow_load:
            command.append("--no-load")
        environment = dict(os.environ, PYTHONPATH=str(ROOT))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, cwd=str(ROOT), env=environment)
        self.stdout = []  # (request kind, raw reply bytes)

    def send(self, raw: bytes, kind: str = "raw") -> tuple[dict | None, bytes, float]:
        started = time.perf_counter()
        self.process.stdin.write(raw if raw.endswith(b"\n") else raw + b"\n")
        self.process.stdin.flush()
        reply = self.process.stdout.readline()
        elapsed = time.perf_counter() - started
        self.stdout.append((kind, reply))
        try:
            return json.loads(reply), reply, elapsed
        except ValueError:
            return None, reply, elapsed

    def request(self, message: dict, kind: str) -> dict | None:
        return self.send(json.dumps(message, ensure_ascii=True).encode("ascii"), kind)[0]

    def reset(self, seed=None, config=None):
        message = {"op": "reset"} if seed is None else {"op": "reset", "seed": seed, "config": config}
        return self.request(message, "reset")

    def step(self, action):
        reply = self.request({"op": "step", "action": action}, "step")
        return (reply["observation"], reply["reward"], reply["terminated"], reply["truncated"], reply["info"])

    def close(self) -> bytes:
        try:
            self.process.stdin.close()
        except OSError:
            pass
        stderr = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        self.process.wait(timeout=30)
        return stderr


# --- what must never reach the agent ----------------------------------------------------------
def _leaves(value, strings: set, ints: set) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _leaves(item, strings, ints)
    elif isinstance(value, list):
        for item in value:
            _leaves(item, strings, ints)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int):
        ints.add(value)
    elif isinstance(value, str):
        strings.add(value)


def secrets(episode: dict, paths: tuple[str, ...] = ()) -> dict:
    private = episode["private"]
    provenance = private["provenance"]
    public_strings, public_ints = set(), set()
    _leaves(episode["public"], public_strings, public_ints)
    # An all-digit public string (an id) is served data that reads as a number once the digits are bounded.
    public_ints |= {int(s) for s in public_strings if s.isascii() and s.isdigit() and len(s) <= 30}
    pre_strings, pre_ints = set(), set()
    for name in ("pre_ledger", "pre_invoices", "pre_payments"):
        _leaves(private.get(name, []), pre_strings, pre_ints)
    for step in provenance.get("steps", []):
        _leaves(step.get("before", []), pre_strings, pre_ints)
        _leaves(step.get("params", {}), pre_strings, pre_ints)
    exact = {issue["issue_id"] for issue in private["truth"]["issues"]} | set(provenance.get("id_map", {}))
    return {
        "exact": exact - public_strings,
        "strings": {s for s in pre_strings - public_strings if len(s) >= 4 and s not in exact},
        "ints": {i for i in pre_ints - public_ints if i >= 10000},
        # Integers the agent may legitimately read (amounts, ...): one that happens to lie within ten
        # days of today's epoch seconds is data, not the clock. The 1.1 s repeat still catches a clock.
        # (Digit runs inside opaque tokens, such as cursors, are excluded by the boundaries in scan_text.)
        "public_ints": public_ints,
        "seed": provenance.get("seed"),
        "paths": tuple(p for p in paths if p) + (str(ROOT),),
    }


def _keys(value, out: set) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            out.add(key)
            _keys(item, out)
    elif isinstance(value, list):
        for item in value:
            _keys(item, out)


def scan_text(text: str, secret: dict, parsed=None) -> list[str]:
    """Problems in one chunk of agent-side text (parsed: its JSON value when it has one)."""
    problems = []
    lowered = text.lower()
    for word in FORBIDDEN_WORDS:
        if word in lowered:
            problems.append(f"word {word!r}")
    for path in secret["paths"]:
        if path in text:
            problems.append("a path")
    seed = secret["seed"]
    if isinstance(seed, int) and re.search(rf"(?<![0-9]){abs(seed)}(?![0-9])", text):
        problems.append("the seed")
    for value in secret["strings"]:
        if value in text:
            problems.append(f"pre-injection value {value!r}")
    for number in re.findall(r"(?<![0-9.])[0-9]{5,}(?![0-9])", text):
        if int(number) in secret["ints"]:
            problems.append(f"pre-injection amount {number}")
    # Wall clock: a number standing on its own (not a digit run inside a hash-like token such as a cursor
    # or an id, where the calendar window sweeps over chance digits) that is not served data.
    for number in re.findall(r"(?<![0-9A-Za-z_.])[0-9]{5,}(?![0-9A-Za-z_])", text):
        if _near_now(int(number)) and int(number) not in secret.get("public_ints", ()):
            problems.append(f"epoch-like number {number}")
    if parsed is not None:
        strings, ints = set(), set()
        _leaves(parsed, strings, ints)
        hit = strings & secret["exact"]
        if hit:
            problems.append(f"private id {sorted(hit)[0]!r}")
        keys = set()
        _keys(parsed, keys)
        bad = keys & FORBIDDEN_KEYS
        if bad:
            problems.append(f"key {sorted(bad)[0]!r}")
    return problems


def _near_now(number: int) -> bool:
    now = int(time.time())
    return abs(number - now) < 10 * 86400 or abs(number - now * 1000) < 10 * 86400 * 1000


def check_shape(reply: dict | None, kind: str, mode: str) -> list[str]:
    from auditgym.actions import ERRORS
    from auditgym.serve import PROTOCOL_ERRORS

    if not isinstance(reply, dict):
        return ["reply is not a JSON object"]
    observation = reply.get("observation")
    if not isinstance(observation, dict):
        return [f"observation keys {observation}"]
    keys = set(observation)
    if observation.get("type") == "records" and observation.get("limit_clamped") is True:
        keys.discard("limit_clamped")  # the documented flag of a list_records limit above the page size
    if keys != OBS_KEYS.get(observation.get("type"), set()):
        return [f"observation keys {sorted(observation)}"]
    problems = []
    if observation["type"] == "error":
        error = observation["error"]
        texts = dict(ERRORS, **PROTOCOL_ERRORS)
        if set(error) != {"code", "message"} or texts.get(error["code"]) != error["message"]:
            problems.append(f"error is not a fixed text: {error}")
    if observation["type"] == "reset":
        if set(reply) != {"observation", "info"} or reply["info"] != {}:
            problems.append("reset reply shape")
        return problems
    if set(reply) != {"observation", "reward", "terminated", "truncated", "info"}:
        return problems + [f"reply keys {sorted(reply)}"]
    ended = observation["type"] == "final"
    if mode == "eval" or not ended:
        if reply["reward"] != 0.0 or reply["info"] != {}:
            problems.append(f"reward or info before the end or in eval: {reply['reward']} {reply['info']}")
    elif set(reply["info"]) != INFO_KEYS:
        problems.append(f"terminal info keys {sorted(reply['info'])}")
    return problems


# --- scripts ------------------------------------------------------------------------------------
def honest(server: RawServer, seed, config, episode: dict, load_dir: str | None) -> None:
    if load_dir is not None:
        server.request({"op": "load", "directory": load_dir}, "reset")
        first = server.reset()
    else:
        first = server.reset(seed, config)
    first = first["observation"]
    tables, _, _, problem = page_all(server.step)
    if problem:
        raise RuntimeError(problem)
    affected = sorted({i for issue in episode["private"]["truth"]["issues"] for i in issue["affected_ids"]})
    docs = affected + [doc["doc_id"] for doc in (tables["invoices"] + tables["payments"])[:5]]
    for doc_id in docs[:20]:
        server.step({"action": "get_document", "doc_id": doc_id})
    if tables["ledger"]:
        server.step({"action": "get_record", "table": "ledger", "id": tables["ledger"][-1]["row_id"]})
    server.step({"action": "calculator", "expression": "(125000 * 465.37) / 7"})
    server.step({"action": "submit", "findings": solver_answer(first, tables)})
    server.step({"action": "list_records", "table": "ledger"})


def _guess(tables: dict, episode_id: str) -> dict:
    docs = [doc["doc_id"] for doc in tables["invoices"]]
    findings = [{"type": "duplicate_invoice", "affected_ids": [a, b], "claimed_amount_minor": 1, "currency": "KZT",
                 "evidence_ids": [a, b], "explanation": "guess"} for a, b in zip(docs[::2], docs[1::2])][:30]
    return {"episode_id": episode_id, "findings": findings}


def probe(server: RawServer, seed, config, episode: dict, load_dir: str | None, secret: dict) -> None:
    if load_dir is not None:  # load is only accepted as the very first request
        server.request({"op": "load", "directory": load_dir}, "reset")
    for raw in (b"not json", b"[1, 2]", b'{"op": 5}', b'{"op": "debug"}', b'{"op": "truth"}', b"\xff\xfe\xfd",
                b"{}", b'"reset"', b'{"op": "step", "action": {"action": "list_records", "table": "fx"}}',
                b'{"op": "reset", "seed": "abc"}', b'{"op": "reset", "seed": true}',
                b'{"op": "reset", "seed": 1, "config": {"bogus": 1}}',
                b'{"op": "reset", "seed": 1, "config": {"errors": {"duplicate_invoice": 99}}}',
                b'{"op": "reset", "seed": 1, "config": "x"}', b'{"op": "reset"}', b"[" * 100000 + b"]" * 100000):
        server.send(raw, "probe")
    def start():
        if load_dir is not None:
            reply = server.reset()
            if reply["observation"].get("type") != "error":
                return reply
            # A reset of the loaded episode can be refused (for example an unreadable directory); the
            # same episode comes from its seed (load_dir was written from it).
        return server.reset(seed, config)

    first = start()["observation"]
    episode_id = first["manifest"]["episode_id"]
    server.send(json.dumps({"op": "load", "directory": load_dir or str(ROOT)}).encode(), "probe")
    server.send(b"x" * ((1 << 20) + 10), "probe")
    page = server.step({"action": "list_records", "table": "ledger", "limit": 3})[0]
    cursor = page.get("next_cursor") or "0" * 20
    other = server.step({"action": "list_records", "table": "fx", "limit": 1})[0].get("next_cursor") or "x"
    pre_only = sorted(secret["strings"])[:3]
    internal = sorted(episode["private"]["provenance"].get("id_map", {}))[:3]
    issue_ids = [issue["issue_id"] for issue in episode["private"]["truth"]["issues"]][:3]
    actions = [{"action": "nope"}, {"tool": "get_truth"}, {"name": "load"}, {"type": "shell", "cmd": "ls"},
               None, [], "list_records", 5, {"action": None}]
    actions += [{"action": "list_records", "table": table} for table in
                ("truth", "private", "provenance", "../private/truth.json", "LEDGER", "", None, 5)]
    for record_id in issue_ids + internal + pre_only + ["../../etc/passwd", "", "x" * 5000]:
        actions.append({"action": "get_record", "table": "invoices", "id": record_id})
        actions.append({"action": "get_document", "doc_id": record_id})
    actions += [{"action": "get_record", "table": "ledger", "id": 12}, {"action": "get_record", "table": "fx",
                                                                       "id": "2026-10-01|USD|KZT|x"},
                {"action": "get_document", "doc_id": None}]
    actions += [{"action": "list_records", "table": "ledger", "cursor": c} for c in
                (cursor[:-1], cursor + "0", other, 5, {"a": 1}, cursor.upper())]
    actions += [{"action": "list_records", "table": "ledger", "limit": limit} for limit in
                (-1, 0, "50", 10 ** 30, True, 2.5)]
    actions += [{"action": "calculator", "expression": e} for e in
                ("__import__('os')", "1/0", "9**9**9", "1e999", "open('/etc/passwd')", "seed", "truth", "",
                 "1" * 500, None, "2 ** -1074")]
    actions += [{"action": "get_record", "args": "{bad"}, {"action": "get_record", "params": [1]},
                {"action": "calculator", "arguments": {"expression": 5}}]
    for action in actions:
        server.step(action)
    # Submit variants, each in a fresh episode, each followed by a step and a submit after the end.
    tables = None
    variants = ["x", {}, {"episode_id": "ep_wrong", "findings": []}, {"episode_id": episode_id, "findings": "x"},
                {"episode_id": episode_id, "findings": [{}] * 31},
                {"episode_id": episode_id, "findings": [], "pad": "x" * 70000}, "guess",
                oracle_answer(episode), {"episode_id": episode_id, "findings": []}, None]
    for variant in variants:
        start()
        if variant == "guess":
            if tables is None:
                tables, _, _, _ = page_all(server.step)
            variant = _guess(tables, episode_id)
        server.step({"action": "submit", "findings": variant} if variant is not None else {"action": "submit"})
        server.step({"action": "submit", "findings": oracle_answer(episode)})
        server.step({"action": "list_records", "table": "fx"})
    start()
    for _ in range(100):
        server.step({"action": "list_records", "table": "fx"})
    server.step({"action": "list_records", "table": "fx"})


# --- client object ------------------------------------------------------------------------------
def client_strings(env) -> list[str]:
    skip = (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType, types.CodeType, types.MethodType,
            types.FrameType)
    seen, stack, strings = set(), [env], []
    while stack and len(seen) < 200000:
        current = stack.pop()
        if id(current) in seen or isinstance(current, skip):
            continue
        seen.add(id(current))
        if isinstance(current, str):
            strings.append(current)
        stack.extend(gc.get_referents(current))
        if hasattr(current, "__dict__"):
            stack.append(vars(current))
    return strings


def client_run(seed, config, episode, load_dir, mode) -> list[str]:
    from auditgym import Env

    env = Env.from_files(load_dir, mode=mode) if load_dir else Env(mode=mode)
    with env:
        first = env.reset() if load_dir else env.reset(seed, config)
        tables, _, _, _ = page_all(env.step)
        env.step({"action": "submit", "findings": solver_answer(first[0], tables)})
        return client_strings(env)


# --- surface statistics ----------------------------------------------------------------------------
def auc(positive: list, negative: list) -> float | None:
    if not positive or not negative:
        return None
    values = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    ranks, index = {}, 0
    rank_sum = 0.0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[end + 1][0] == values[index][0]:
            end += 1
        average = (index + end) / 2 + 1
        rank_sum += average * sum(1 for k in range(index, end + 1) if values[k][1] == 1)
        index = end + 1
    n1, n0 = len(positive), len(negative)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def separation(positive: list, negative: list) -> dict | None:
    """AUC, z and the two-sided p-value of the Mann-Whitney test (normal approximation with the
    tie-corrected variance). None when one side is empty."""
    score = auc(positive, negative)
    if score is None:
        return None
    n1, n0 = len(positive), len(negative)
    total = n1 + n0
    ties = {}
    for value in positive + negative:
        ties[value] = ties.get(value, 0) + 1
    correction = sum(t ** 3 - t for t in ties.values()) / (total * (total - 1)) if total > 1 else 0.0
    variance = n1 * n0 / 12 * ((total + 1) - correction)
    z = 0.0 if variance <= 0 else (score - 0.5) * n1 * n0 / math.sqrt(variance)
    return {"auc": score, "z": z, "p": math.erfc(abs(z) / math.sqrt(2)), "n1": n1, "n0": n0}


def verdict(stat: dict, tests: int) -> bool:
    """True when the separation is both large and significant after correcting for `tests` rows."""
    return abs(stat["auc"] - 0.5) >= MIN_EFFECT and stat["p"] < ALPHA / max(tests, 1)


def _moment(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


def document_features(public: dict) -> dict[str, tuple[str, dict]]:
    """doc_id -> (table, features) for every document, from the open tables only."""
    groups: dict[tuple, list] = {}
    for row in public["ledger"]:
        key = ("pay", row["payment_id"]) if row["payment_id"] else ("inv", row["invoice_id"]) if row["invoice_id"] else None
        if key:
            groups.setdefault(key, []).append(row)
    paid = {ref for payment in public["payments"] for ref in payment["invoice_refs"] or []}
    out = {}
    for table, time_field in (("invoices", "issued_at"), ("payments", "timestamp")):
        records = public[table]
        ranks = {doc_id: rank for rank, doc_id in enumerate(sorted(r["doc_id"] for r in records))}
        span = max(len(records) - 1, 1)
        for position, record in enumerate(records):
            moment = _moment(record[time_field])
            if table == "invoices":
                own = groups.get(("inv", record["invoice_id"]), [])
            else:
                own = groups.get(("pay", record["payment_id"]), [])
            features = {"id_rank": ranks[record["doc_id"]] / span, "position": position / span,
                        "second": moment.second, "minute": moment.minute, "hour": moment.hour,
                        "posting_equals_document_time": int(any(row["timestamp"] == record[time_field] for row in own))}
            if table == "invoices":
                features.update({"paid": int(record["invoice_id"] in paid), "line_items": len(record["line_items"]),
                                 "terms_days": (_moment(record["due_at"]) - moment).days})
            else:
                features.update({"refs": len(record["invoice_refs"] or []), "refund": int(record["amount_minor"] < 0)})
            out[record["doc_id"]] = (table, features)
    return out


def ledger_features(public: dict) -> dict[str, tuple[str, dict]]:
    """row_id -> (stratum "account/side", features) for every ledger row, from the open tables only."""
    invoice_time = {i["invoice_id"]: i["issued_at"] for i in public["invoices"]}
    payment_time = {p["payment_id"]: p["timestamp"] for p in public["payments"]}
    base = public["manifest"]["base_currency"]
    currency = {("inv", i["invoice_id"]): i["currency"] for i in public["invoices"]}
    currency.update({("pay", p["payment_id"]): p["currency"] for p in public["payments"]})
    sizes: dict[tuple, int] = {}

    def group(row: dict) -> tuple:
        if row["payment_id"]:
            return ("pay", row["payment_id"])
        if row["invoice_id"]:
            return ("inv", row["invoice_id"])
        return ("bg", row["timestamp"], row["entity_id"])

    for row in public["ledger"]:
        sizes[group(row)] = sizes.get(group(row), 0) + 1
    ranks = {row_id: rank for rank, row_id in enumerate(sorted(r["row_id"] for r in public["ledger"]))}
    span = max(len(public["ledger"]) - 1, 1)
    out = {}
    for position, row in enumerate(public["ledger"]):
        amount = row["debit_minor"] or row["credit_minor"]
        moment = _moment(row["timestamp"])
        document = payment_time.get(row["payment_id"]) if row["payment_id"] else invoice_time.get(row["invoice_id"])
        # The stratum is account, side and whether the document is in a foreign currency: a converted
        # amount is not round, so mixing the two would "find" the definition of fx_error.
        foreign = currency.get(group(row), base) != base
        out[row["row_id"]] = (f"{row['account']}/{'Dr' if row['debit_minor'] else 'Cr'}/{'fx' if foreign else 'base'}", {
            "id_rank": ranks[row["row_id"]] / span, "position": position / span,
            "second": moment.second, "minute": moment.minute, "hour": moment.hour,
            "round_100": int(amount % 100 == 0), "round_1000": int(amount % 1000 == 0),
            "digits": len(str(amount)), "group_size": sizes[group(row)],
            "posting_equals_document_time": int(document == row["timestamp"]),
        })
    return out


def pair_features(public: dict) -> dict[frozenset, dict]:
    """For every two invoices of one entity with one total (a possible duplicate): its features."""
    paid = {ref for payment in public["payments"] for ref in payment["invoice_refs"] or []}
    posted = {row["invoice_id"] for row in public["ledger"] if row["invoice_id"]}
    ranks = {doc_id: rank for rank, doc_id in enumerate(sorted(i["doc_id"] for i in public["invoices"]))}
    span = max(len(public["invoices"]) - 1, 1)
    groups: dict[tuple, list] = {}
    for position, invoice in enumerate(public["invoices"]):
        groups.setdefault((invoice["entity_id"], invoice["total_minor"], invoice["currency"]), []).append(
            (position, invoice))
    out = {}
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                (pos_a, one), (pos_b, two) = members[a], members[b]
                (pos_e, early), (pos_l, late) = sorted(((pos_a, one), (pos_b, two)),
                                                       key=lambda item: (item[1]["issued_at"], item[1]["doc_id"]))
                first, second = _moment(early["issued_at"]), _moment(late["issued_at"])
                items = lambda inv: [(x["description"], x["quantity"], x["amount_minor"]) for x in inv["line_items"]]  # noqa: E731
                periods = lambda inv: [(x.get("service_start"), x.get("service_end")) for x in inv["line_items"]]  # noqa: E731
                out[frozenset((one["doc_id"], two["doc_id"]))] = {
                    "gap_days": (second - first).total_seconds() / 86400,
                    "position_order": int(pos_e < pos_l), "id_order": int(ranks[early["doc_id"]] < ranks[late["doc_id"]]),
                    "id_distance": abs(ranks[one["doc_id"]] - ranks[two["doc_id"]]) / span,
                    "position_distance": abs(pos_a - pos_b) / span,
                    "same_time_of_day": int(first.time() == second.time()),
                    "same_items": int(items(early) == items(late)),
                    "same_service_period": int(periods(early) == periods(late)),
                    "earlier_paid": int(early["invoice_id"] in paid), "later_paid": int(late["invoice_id"] in paid),
                    "earlier_posted": int(early["invoice_id"] in posted), "later_posted": int(late["invoice_id"] in posted),
                }
    return out


def _evidence_rows(episode: dict) -> tuple[dict[str, set], set]:
    """Ledger row ids cited as evidence: per error type, and by any issue."""
    rows = {row["row_id"] for row in episode["public"]["ledger"]}
    by_type: dict[str, set] = {}
    for issue in episode["private"]["truth"]["issues"]:
        cited = {i for group in issue["evidence_sets"] for i in group} & rows
        by_type.setdefault(issue["type"], set()).update(cited)
    return by_type, set().union(*by_type.values()) if by_type else set()


def surface_statistics(episodes: list[dict]) -> list[dict]:
    """Rows {type, table, feature, affected, other, auc, z, p, status} over the three views."""
    pools: dict[tuple, tuple[list, list]] = {}

    def add(key: tuple, value, positive: bool) -> None:
        pools.setdefault(key, ([], []))[0 if positive else 1].append(value)

    for episode in episodes:
        features = document_features(episode["public"])
        touched = {i for issue in episode["private"]["truth"]["issues"] for i in issue["affected_ids"]}
        by_type: dict[str, set] = {}
        for issue in episode["private"]["truth"]["issues"]:
            by_type.setdefault(issue["type"], set()).update(issue["affected_ids"])
        for kind, affected in by_type.items():
            for doc_id, (table, values) in features.items():
                if doc_id in touched and doc_id not in affected:
                    continue
                for name, value in values.items():
                    add((kind, table, name), value, doc_id in affected)
        # ledger view: evidence rows against rows of the same account and side
        cited, any_cited = _evidence_rows(episode)
        rows = ledger_features(episode["public"])
        for kind, evidence in cited.items():
            strata = {rows[row_id][0] for row_id in evidence if row_id in rows}
            for row_id, (stratum, values) in rows.items():
                if stratum not in strata or (row_id in any_cited and row_id not in evidence):
                    continue
                for name, value in values.items():
                    add((kind, f"ledger:{stratum}", name), value, row_id in evidence)
        # pair view: duplicate pairs against the other pairs of one entity and one total
        pairs = pair_features(episode["public"])
        duplicates = {frozenset(issue["affected_ids"]) for issue in episode["private"]["truth"]["issues"]
                      if issue["type"] == "duplicate_invoice"}
        for pair, values in pairs.items():
            if pair not in duplicates and (pair & touched or not values["earlier_posted"] or not values["later_posted"]):
                continue
            for name, value in values.items():
                add(("duplicate_invoice", "pairs", name), value, pair in duplicates)
    stats = {}
    for key, (positive, negative) in pools.items():
        if len(positive) >= MIN_AFFECTED and negative:
            stats[key] = separation(positive, negative)
    tests = len(stats)
    rows = []
    for (kind, table, name), (positive, negative) in sorted(pools.items()):
        if not positive:
            continue
        stat = stats.get((kind, table, name)) or separation(positive, negative)
        status = "ok"
        if len(positive) < MIN_AFFECTED or stat is None:
            status = "few"
        elif verdict(stat, tests):
            status = (EXPLAINED.get((kind, name)) and "explained") or PENDING.get((kind, name)) \
                     or PENDING.get(("*", name)) or "FAIL"
        rows.append({"type": kind, "table": table, "feature": name, "affected": len(positive),
                     "other": len(negative), "auc": stat["auc"] if stat else None,
                     "z": stat["z"] if stat else None, "p": stat["p"] if stat else None, "status": status,
                     "tests": tests})
    return rows


def side_channels(seed, config, episode: dict) -> dict:
    """get_record on affected vs other documents: reply size and round-trip time."""
    from auditgym.rng import Rng

    affected = sorted({i for issue in episode["private"]["truth"]["issues"] for i in issue["affected_ids"]})
    if not affected:
        return {"affected": ([], []), "other": ([], [])}
    tables = {doc["doc_id"]: "invoices" for doc in episode["public"]["invoices"]}
    tables.update({doc["doc_id"]: "payments" for doc in episode["public"]["payments"]})
    rng = Rng(seed, "leakscan-side")
    plan = []
    for _ in range(48):  # pairs from the same table: an invoice is compared with an invoice
        doc_id = affected[rng.below(len(affected))]
        others = sorted(d for d, table in tables.items() if table == tables[doc_id] and d not in affected)
        if not others:
            continue
        plan += [("affected", doc_id), ("other", others[rng.below(len(others))])]
    server = RawServer("train", allow_load=False)
    out = {"affected": ([], []), "other": ([], [])}
    try:
        server.reset(seed, config)
        for _ in range(3):  # warm-up
            server.send(json.dumps({"op": "step", "action": {"action": "calculator", "expression": "1"}}).encode())
        for group, doc_id in plan:
            message = {"op": "step", "action": {"action": "get_record", "table": tables[doc_id], "id": doc_id}}
            _, raw, elapsed = server.send(json.dumps(message).encode())
            out[group][0].append(len(raw) - len(doc_id))
            out[group][1].append(elapsed)
    finally:
        server.close()
    return out


def _median(values: list) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


# --- driver -------------------------------------------------------------------------------------------
def _episode_configs() -> list[tuple[str, dict]]:
    supported = supported_configs()
    with_errors = [item for item in supported if any(item[1].get("errors", {}).values()) or
                   item[1].get("interaction", "none") != "none"]
    return with_errors + [item for item in supported if item not in with_errors]


def _dense_configs() -> list[dict]:
    """Configs with as many errors as the density limit allows, for the surface statistics."""
    from auditgym.config import ConfigError
    from auditgym.generate import generate

    out = []
    for name in ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff"):
        for count in (6, 5, 4, 3, 2, 1):
            config = {"errors": {name: count}, "size": {"invoices": [60, 100]}, "currencies": ["USD", "EUR"]}
            try:
                generate(1, config)
            except ConfigError:
                continue
            out.append(config)
            break
    return out


def run(episodes: int, seed_base: int = 7_000_001, out=sys.stdout) -> dict:
    from auditgym.generate import generate
    from auditgym.storage import write_episode

    configs = _episode_configs()
    channels = {"stdout": [0, 0, []], "client": [0, 0, []], "stderr": [0, 0, []]}  # chunks, bytes, problems
    shape_problems, repeat_problems, failures = [], [], []
    side = {"affected": ([], []), "other": ([], [])}
    with tempfile.TemporaryDirectory(prefix="leakscan_episode_") as directory:
        for number in range(episodes):
            label, config = configs[number % len(configs)]
            seed = seed_base + 104729 * number
            episode = generate(seed, config)
            load_dir = None
            if number % 2:
                load_dir = str(pathlib.Path(directory) / f"ep{number}")
                write_episode(episode, load_dir)
            secret = secrets(episode, (load_dir or "", directory, str(pathlib.Path.home())))
            honest_bytes = {}
            for mode in ("train", "eval"):
                for script in ("honest", "probe"):
                    server = RawServer(mode, allow_load=load_dir is not None)
                    try:
                        if script == "honest":
                            honest(server, seed, config, episode, load_dir)
                        else:
                            probe(server, seed, config, episode, load_dir, secret)
                    except Exception as exc:
                        failures.append(f"episode {number} {mode} {script}: driver {type(exc).__name__}: {exc}")
                    stderr = server.close()
                    if script == "honest":
                        honest_bytes[mode] = b"".join(raw for _, raw in server.stdout)
                    for kind, raw in server.stdout:
                        text = raw.decode("utf-8", errors="replace")
                        try:
                            parsed = json.loads(raw)
                        except ValueError:
                            parsed = None
                        channels["stdout"][0] += 1
                        channels["stdout"][1] += len(raw)
                        for problem in scan_text(text, secret, parsed):
                            channels["stdout"][2].append(f"episode {number} {mode} {script}: {problem}")
                        for problem in check_shape(parsed, kind, mode):
                            shape_problems.append(f"episode {number} {mode} {script}: {problem}")
                    channels["stderr"][0] += 1
                    channels["stderr"][1] += len(stderr)
                    for problem in scan_text(stderr.decode("utf-8", errors="replace"), secret):
                        channels["stderr"][2].append(f"episode {number} {mode} {script}: {problem}")
            if number % 10 == 0:
                time.sleep(1.1)
                server = RawServer("train", allow_load=load_dir is not None)
                honest(server, seed, config, episode, load_dir)
                server.close()
                if b"".join(raw for _, raw in server.stdout) != honest_bytes.get("train"):
                    repeat_problems.append(f"episode {number}: the honest walk 1.1 s later is not byte-identical")
            if number % 4 in (0, 1):
                for mode in ("train", "eval"):
                    strings = client_run(seed, config, episode, load_dir, mode)
                    channels["client"][0] += len(strings)
                    channels["client"][1] += sum(len(s) for s in strings)
                    text = "\n".join(strings)
                    problems = scan_text(text, secret)
                    problems += [f"private id {s!r}" for s in strings if s in secret["exact"]]
                    for problem in problems:
                        channels["client"][2].append(f"episode {number} {mode}: {problem}")
            if episode["private"]["truth"]["issues"]:
                measured = side_channels(seed, config, episode)
                for group in side:
                    side[group][0].extend(measured[group][0])
                    side[group][1].extend(measured[group][1])

    dense = [generate(seed_base + 31 * k, config) for k in range(max(episodes, 10)) for config in _dense_configs()]
    surface = surface_statistics(dense)

    print(f"leakscan: {episodes} episodes, configs {', '.join(label for label, _ in configs)}; "
          f"{len(dense)} dense episodes for the surface statistics", file=out)
    print(f"{'channel':<10}{'chunks':>10}{'bytes':>12}{'problems':>10}", file=out)
    for name, (chunks, size, problems) in channels.items():
        print(f"{name:<10}{chunks:>10}{size:>12}{len(problems):>10}", file=out)
    print(f"{'shape':<10}{'':>22}{len(shape_problems):>10}", file=out)
    print(f"{'repeat':<10}{'':>22}{len(repeat_problems):>10}", file=out)
    for name, (_, _, problems) in channels.items():
        failures += [f"{name}: {p}" for p in problems[:20]]
    failures += [f"shape: {p}" for p in shape_problems[:20]] + [f"repeat: {p}" for p in repeat_problems]

    print(f"\nsurface separation: FAIL when |AUC-0.5| >= {MIN_EFFECT} and p < {ALPHA}/m (Bonferroni, m = rows "
          f"tested), at least {MIN_AFFECTED} affected", file=out)
    print(f"{'type':<18}{'table':<30}{'feature':<30}{'aff':>5}{'other':>7}{'AUC':>7}{'p':>10}  status", file=out)
    pending = []
    shown = 0
    for row in surface:
        interesting = row["status"] not in ("ok", "few") or (row["auc"] is not None and abs(row["auc"] - 0.5) >= MIN_EFFECT)
        if not interesting and row["table"].startswith("ledger:"):
            continue  # the ledger view has hundreds of quiet rows; they are counted below
        shown += 1
        auc_text = f"{row['auc']:>7.2f}" if row["auc"] is not None else f"{'-':>7}"
        p_text = f"{row['p']:>10.1e}" if row["p"] is not None else f"{'-':>10}"
        print(f"{row['type']:<18}{row['table']:<30}{row['feature']:<30}{row['affected']:>5}{row['other']:>7}"
              f"{auc_text}{p_text}  {row['status'] if not row['status'].startswith('pending') else 'pending'}",
              file=out)
        if row["status"] == "FAIL":
            failures.append(f"surface: {row['type']} {row['table']}.{row['feature']} AUC {row['auc']:.2f} "
                            f"(p {row['p']:.1e})")
        elif row["status"].startswith("pending"):
            pending.append(f"{row['status']} [{row['type']} {row['table']}.{row['feature']} AUC {row['auc']:.2f}]")
    tested = sum(1 for row in surface if row["status"] != "few")
    print(f"{len(surface)} rows, {tested} tested, {len(surface) - tested} too few samples "
          f"({shown} shown: the document tables and every row with a visible effect)", file=out)

    size_auc = auc(side["affected"][0], side["other"][0])
    time_auc = auc(side["affected"][1], side["other"][1])
    print("\nside channels of get_record (affected vs other documents)", file=out)
    if size_auc is None:
        print("  no affected documents", file=out)
    else:
        print(f"  size  median {_median(side['affected'][0]):.0f} vs {_median(side['other'][0]):.0f} bytes, "
              f"AUC {size_auc:.2f} (fail > {SIZE_THRESHOLD} from 0.5)", file=out)
        print(f"  time  median {_median(side['affected'][1]) * 1e6:.0f} vs {_median(side['other'][1]) * 1e6:.0f} us, "
              f"AUC {time_auc:.2f} (fail > {TIME_THRESHOLD} from 0.5)", file=out)
        if abs(size_auc - 0.5) > SIZE_THRESHOLD:
            failures.append(f"side channel: reply size separates affected documents (AUC {size_auc:.2f})")
        if abs(time_auc - 0.5) > TIME_THRESHOLD:
            failures.append(f"side channel: reply time separates affected documents (AUC {time_auc:.2f})")
    for item in sorted(set(p.split(" [")[0] for p in pending)):
        print(item, file=out)
    for item in pending:
        print("  " + item.split(" [", 1)[1].rstrip("]"), file=out)
    for failure in failures:
        print("FAIL " + failure, file=out)
    print(f"leakscan: {'FAILED' if failures else 'PASSED'} ({len(failures)} failures, {len(pending)} pending)",
          file=out)
    return {"failures": failures, "pending": pending, "surface": surface, "channels": channels,
            "side": {"size_auc": size_auc, "time_auc": time_auc}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.leakscan")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=7_000_001)
    parser.add_argument("--strict", action="store_true", help="exit 1 while a `pending:` defect is left")
    arguments = parser.parse_args(argv)
    summary = run(arguments.episodes, arguments.seed_base)
    return 1 if summary["failures"] or (arguments.strict and summary["pending"]) else 0


if __name__ == "__main__":
    sys.exit(main())
