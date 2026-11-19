"""Evaluator: the episode state on the evaluator side (docs/CONTRACTS.md, sections 5 and 7).

The evaluator is the only place that holds `episode["private"]`. Everything it returns to a
policy is built from the open tables, constants and the step counter. It runs inside the
server process (`python -m auditgym serve`); the client only ever sees JSON.

Episode lifetime:
- reset(seed, config) generates an episode; reset() without a seed restarts the episode given
  to load(), or, when nothing was loaded, generates with the next default seed (0 first, then
  the previous seed + 1, like a Gymnasium env whose generator keeps going). A refused reset or
  load leaves the running episode untouched.
- With a secret (the server's --secret-file or AUDITGYM_SECRET) the generator gets a seed
  derived from the secret and the given seed, so an agent that imports this package cannot find
  a small closed seed by generating candidates. Without one (the default) the seed is used as is.
- After the end (submit, 100 steps, time limit) every step returns `episode_over` with the
  flags and the step count of the final reply, and changes nothing.

Order of the pages: ledger rows, invoices and payments are served in the order of a hash of their
own id (`order_key`), fx rates by their key (date, then currencies). The files keep the order of the
episode, but what a policy pages through says nothing about dates, posting times or injected
errors (served by date, the first and last invoices were the cut-off errors: AUDIT R6, task E8),
and reordering the rows of an episode's files changes no observation.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import pathlib

from . import actions
from .actions import MAX_STEPS, ActionError, error
from .calc import CalcError, calculate

DEFAULT_TIME_LIMIT = 60
DEFAULT_SEED = 0
MAX_SEED_BITS = 256
MODES = ("train", "eval")

# Fields a policy may see (CONTRACTS section 2). Anything else in a record or the manifest is
# dropped, so a generator mistake cannot leak extra keys through the tools.
_FIELDS = {
    "ledger": ("row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
               "entity_id", "invoice_id", "payment_id"),
    "invoices": ("doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency",
                 "total_minor", "line_items"),
    "payments": ("doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"),
    "fx": ("date", "source_currency", "target_currency", "rate"),
}
_MANIFEST_FIELDS = ("schema_version", "episode_id", "base_currency", "period_start", "period_end",
                    "timezone", "counts", "rounding", "config_version")
_CURSOR_SALT = "auditgym-cursor-v1"
_ORDER_SALT = b"auditgym-order-v1"
_SECRET_SALT = b"auditgym-secret-seed-v1"
_RESULTS_FLAGS = os.O_WRONLY | os.O_APPEND | os.O_CREAT
_RESULTS_MODE = 0o600  # the judge log holds rewards: owner only (AUDIT L3)


class MonotonicClock:
    """Default clock. Lives only in the evaluator; its readings never leave it."""

    def now(self):
        import time

        return time.monotonic()  # determinism: wall time only bounds the episode, never reaches data


def fx_id(record: dict) -> str:
    return f"{record['date']}|{record['source_currency']}|{record['target_currency']}"


def order_key(table: str, record: dict) -> bytes:
    """Sort key of a record in the pages of its table: from its own id only (see the module notes)."""
    if table == "fx":
        return fx_id(record).encode("utf-8", "surrogatepass")
    identity = record.get(actions.ID_FIELDS[table][0])
    text = identity if isinstance(identity, str) else ""
    return hashlib.sha256(_ORDER_SALT + b"\x1f" + table.encode("ascii") + b"\x1f"
                          + text.encode("utf-8", "surrogatepass")).digest()


def _project(record: dict, fields: tuple[str, ...]) -> dict:
    return {field: copy.deepcopy(record.get(field)) for field in fields}


def _cursor_token(episode_id: str, table: str, offset: int) -> str:
    body = f"{_CURSOR_SALT}|{episode_id}|{table}|{offset}".encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:20]


def _judge(episode: dict, payload, inspection_steps: int) -> dict:
    from .judge import judge

    try:
        return judge(episode, payload, inspection_steps)
    except Exception as exc:  # a judge defect must not crash the episode; logged by class only
        import sys

        print(f"auditgym evaluator: judge raised {type(exc).__name__}", file=sys.stderr)
        return _structure_failure(episode, "judge_exception")


def _structure_failure(episode: dict, code: str) -> dict:
    """The judge's result for an invalid structure (reward -1), for failures found before it runs."""
    return {"reward": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0,
            "fn": len(episode["private"]["truth"].get("issues", [])), "structure_error": code}


def open_results(path) -> int:
    """An append-only descriptor of the judge log; the file is created owner-only and its folder
    made when missing."""
    path = pathlib.Path(path)
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, _RESULTS_FLAGS, _RESULTS_MODE)


def generator_seed(seed: int, secret: bytes) -> int:
    """The seed the generator gets: `seed` itself without a secret, else a 256-bit digest of both."""
    if not secret:
        return seed
    digest = hashlib.sha256(_SECRET_SALT + b"\x1f" + secret + b"\x1f" + str(seed).encode("ascii")).digest()
    return int.from_bytes(digest, "big")


# --- reading an episode directory -------------------------------------------------------------
def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8-sig")  # a byte order mark from another tool is fine


def _jsonl(text: str) -> list:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _csv_rows(text: str) -> list[dict]:
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def load_files(directory) -> dict:
    """The open tables and the truth of an episode directory, as the evaluator needs them.

    Unlike storage.load_episode, provenance.json and the pre_* files are optional (the evaluator
    never reads them; someone who rewrites an episode's ids may drop them, AUDIT R9) and every file
    may start with a UTF-8 byte order mark (AUDIT R10)."""
    from .storage import _read_ledger

    root = pathlib.Path(directory)
    public, private = root / "public", root / "private"

    def optional(path: pathlib.Path, parse, empty):
        return parse(_text(path)) if path.is_file() else empty

    return {
        "public": {
            "manifest": json.loads(_text(public / "manifest.json")),
            "policy": json.loads(_text(public / "policy.json")),
            "ledger": _read_ledger(_text(public / "ledger.csv")),
            "invoices": _jsonl(_text(public / "invoices.jsonl")),
            "payments": _jsonl(_text(public / "payments.jsonl")),
            "fx": _csv_rows(_text(public / "fx.csv")),
        },
        "private": {
            "truth": json.loads(_text(private / "truth.json")),
            "provenance": optional(private / "provenance.json", json.loads, {}),
            "pre_ledger": optional(private / "pre_ledger.csv", _read_ledger, []),
            "pre_invoices": optional(private / "pre_invoices.jsonl", _jsonl, []),
            "pre_payments": optional(private / "pre_payments.jsonl", _jsonl, []),
        },
    }


class EvaluatorError(Exception):
    """A request the evaluator refuses outright (bad seed or config, nothing loaded)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Evaluator:
    def __init__(self, mode: str = "train", results_path=None, clock=None,
                 time_limit=DEFAULT_TIME_LIMIT, results_stream=None, secret=None) -> None:
        """`results_stream` is an already open text stream (the server gets the results file as an
        inherited descriptor, so its path never appears in the process list); it wins over the path.
        `secret` (bytes or str) is mixed into generated seeds; empty or None leaves them as given."""
        if mode not in MODES:
            raise ValueError("mode must be 'train' or 'eval'")
        self._mode = mode
        self._results_path = None if results_stream is not None else results_path
        self._results_stream = results_stream
        if self._results_path is not None:
            os.close(open_results(self._results_path))  # fail at start, not at the end of an episode
        self._clock = clock if clock is not None else MonotonicClock()
        self._time_limit = time_limit
        self._secret = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret or b"")
        self._next_seed = DEFAULT_SEED
        self._loaded = None   # episode from load(); reset() without seed restarts it
        self._state = None
        self._install(None)

    @property
    def mode(self) -> str:
        return self._mode

    # --- episode setup ---------------------------------------------------------------------
    def reset(self, seed=None, config=None) -> tuple[dict, dict]:
        """With a seed: generate a new episode. Without: restart the episode from load(), or
        generate with the next default seed when nothing was loaded."""
        if seed is None and self._loaded is not None:
            return self._start(copy.deepcopy(self._loaded))
        from .config import ConfigError
        from .generate import generate

        if seed is None:
            seed = self._next_seed
        if not isinstance(seed, int) or isinstance(seed, bool) or seed.bit_length() > MAX_SEED_BITS:
            raise EvaluatorError("bad_config")
        try:
            episode = generate(generator_seed(seed, self._secret), config)
        except ConfigError:
            raise EvaluatorError("bad_config") from None
        observation = self._start(episode)
        self._next_seed = seed + 1
        return observation

    def load(self, directory) -> tuple[dict, dict]:
        try:
            loaded = load_files(directory)
            observation = self._start(copy.deepcopy(loaded))
        except Exception:  # files are outside input: any defect is one fixed error
            raise EvaluatorError("bad_episode") from None
        self._loaded = loaded
        return observation

    def _start(self, episode: dict) -> tuple[dict, dict]:
        """Build the whole state of `episode` first and install it only when nothing failed, so a
        refused reset or load leaves the running episode as it was."""
        public = episode["public"]
        episode_id = public["manifest"]["episode_id"]
        if not isinstance(episode_id, str) or not isinstance(episode["private"]["truth"], dict):
            raise ValueError("episode without an id or a truth")
        tables, index, cursors = {}, {}, {}
        for table, fields in _FIELDS.items():
            records = sorted((_project(record, fields) for record in public[table]),
                             key=lambda record, table=table: order_key(table, record))
            tables[table] = records
            by_field = {field: {} for field in actions.ID_FIELDS[table]}
            for position, record in enumerate(records):
                if table == "fx":
                    by_field["key"].setdefault(fx_id(record), position)
                    continue
                for field in actions.ID_FIELDS[table]:
                    if isinstance(record[field], str):
                        by_field[field].setdefault(record[field], position)
            index[table] = by_field
            cursors[table] = {_cursor_token(episode_id, table, offset): offset for offset in range(1, len(records))}
        documents = {}
        for table in ("invoices", "payments"):
            for position, record in enumerate(tables[table]):
                if isinstance(record["doc_id"], str):
                    documents.setdefault(record["doc_id"], (table, position))
        manifest = {field: copy.deepcopy(public["manifest"][field])
                    for field in _MANIFEST_FIELDS if field in public["manifest"]}
        observation = {
            "type": "reset",
            "task": actions.build_task(public["policy"]),
            "manifest": manifest,
            "policy": copy.deepcopy(public["policy"]),
            "steps_remaining": MAX_STEPS,
        }
        self._install({"episode": episode, "episode_id": episode_id, "tables": tables, "index": index,
                       "documents": documents, "cursors": cursors})
        self._started_at = self._clock.now()
        return observation, {}

    def _install(self, state) -> None:
        self._state = state
        self._episode = None if state is None else state["episode"]
        self._steps = 0
        self._done = False
        self._final = None  # (terminated, truncated, steps_remaining) of the final reply
        self._started_at = None

    # --- stepping ----------------------------------------------------------------------------
    def steps_remaining(self) -> int:
        """Steps left in the episode: 0 before any reset; after the end, what the final reply said."""
        if self._episode is None:
            return 0
        if self._final is not None:
            return self._final[2]
        return MAX_STEPS - self._steps

    def running(self) -> bool:
        """An episode has been reset and has not ended."""
        return self._episode is not None and not self._done

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        return self._advance(lambda: self._act(action))

    def step_unreadable(self, submit: bool) -> tuple[dict, float, bool, bool, dict]:
        """A request too large to read, consumed unread by the server (AUDIT L4). It spends a step
        like any bad action; when it is a submit, the episode ends as an invalid structure."""
        if submit:
            return self._advance(lambda: self._finish(None, truncated=False, structure_error="message_too_large"))
        return self._advance(lambda: {"type": "error", "error": error("bad_arguments")})

    def _advance(self, act) -> tuple[dict, float, bool, bool, dict]:
        if self._episode is None:
            return ({"type": "error", "error": error("not_reset"), "steps_remaining": 0},
                    0.0, True, False, {})
        if self._done:
            terminated, truncated, remaining = self._final
            return ({"type": "error", "error": error("episode_over"), "steps_remaining": remaining},
                    0.0, terminated, truncated, {})
        self._steps += 1
        if self._clock.now() - self._started_at >= self._time_limit:
            return self._finish(None, truncated=True)
        observation = act()
        if isinstance(observation, tuple):  # the episode ended in this step
            return observation
        observation["steps_remaining"] = MAX_STEPS - self._steps
        if self._steps >= MAX_STEPS:
            return self._finish(None, truncated=True)
        return observation, 0.0, False, False, {}

    def _act(self, action):
        try:
            name, arguments = actions.parse_action(action)
            if name == "submit":
                return self._submit(arguments)
            return getattr(self, "_do_" + name)(arguments)
        except ActionError as exc:
            if exc.name == "submit":  # a submit whose arguments cannot be read (AUDIT C9)
                return self._finish(None, truncated=False, structure_error="bad_submit_arguments")
            return {"type": "error", "error": error(exc.code)}

    def _submit(self, arguments: dict):
        try:
            payload = actions.submit_payload(arguments, self._state["episode_id"])
        except ActionError as exc:
            return self._finish(None, truncated=False, structure_error=exc.code)
        return self._finish(payload, truncated=False)

    def _finish(self, payload, truncated: bool, structure_error: str | None = None):
        self._done = True
        if truncated:
            payload = {"episode_id": self._state["episode_id"], "findings": []}
            inspection_steps = MAX_STEPS
            remaining = 0  # the budget is over, whether the steps or the time ran out
        else:
            inspection_steps = self._steps - 1  # steps before submit, error steps included
            remaining = MAX_STEPS - self._steps
        if structure_error is not None:
            result = _structure_failure(self._episode, structure_error)
        else:
            # judge() reads None as "no answer" (a truncated episode); a submit always has a payload.
            result = _judge(self._episode, payload, inspection_steps)
        self._final = (not truncated, truncated, remaining)
        self._write_result(result, inspection_steps, truncated)
        observation = {"type": "final", "steps_remaining": remaining}
        if self._mode == "eval":
            return observation, 0.0, not truncated, truncated, {}
        info = {"reward": result["reward"], "precision": result["precision"], "recall": result["recall"],
                "f1": result["f1"], "inspection_steps": inspection_steps}
        return observation, result["reward"], not truncated, truncated, info

    def _write_result(self, result: dict, inspection_steps: int, truncated: bool) -> None:
        if self._results_path is None and self._results_stream is None:
            return
        line = {"episode_id": self._state["episode_id"], "mode": self._mode,
                "terminated": not truncated, "truncated": truncated, "inspection_steps": inspection_steps}
        for key in ("reward", "precision", "recall", "f1", "tp", "fp", "fn", "structure_error"):
            line[key] = result.get(key)
        text = json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n"
        if self._results_stream is not None:
            self._results_stream.write(text)
            self._results_stream.flush()
            return
        with os.fdopen(open_results(self._results_path), "a", encoding="utf-8") as handle:
            handle.write(text)

    # --- read actions -----------------------------------------------------------------------
    def _do_list_records(self, arguments: dict) -> dict:
        table = actions.table_name(arguments)
        limit = actions.limit(arguments)
        cursor = actions.cursor(arguments)
        if cursor is None:
            offset = 0
        else:
            offset = self._state["cursors"][table].get(cursor)
            if offset is None:
                raise ActionError("bad_cursor")
        records = self._state["tables"][table]
        end = min(offset + limit, len(records))
        next_cursor = None if end >= len(records) else _cursor_token(self._state["episode_id"], table, end)
        return {"type": "records", "table": table,
                "records": copy.deepcopy(records[offset:end]), "next_cursor": next_cursor}

    def _do_get_record(self, arguments: dict) -> dict:
        table = actions.table_name(arguments)
        fields, record_id = actions.record_reference(arguments, table)
        for field in fields:
            position = self._state["index"][table][field].get(record_id)
            if position is not None:
                return {"type": "record", "table": table,
                        "record": copy.deepcopy(self._state["tables"][table][position])}
        raise ActionError("unknown_id")

    def _do_get_document(self, arguments: dict) -> dict:
        found = self._state["documents"].get(actions.document_id(arguments))
        if found is None:
            raise ActionError("unknown_id")
        table, position = found
        return {"type": "document", "table": table, "record": copy.deepcopy(self._state["tables"][table][position])}

    def _do_calculator(self, arguments: dict) -> dict:
        expression = actions.expression(arguments)
        try:
            result = calculate(expression)
        except CalcError:
            raise ActionError("bad_expression") from None
        return {"type": "calculation", "result": result}
