"""Evaluator: the episode state on the evaluator side (docs/CONTRACTS.md, sections 5 and 7).

The evaluator is the only place that holds `episode["private"]`. Everything it returns to a
policy is built from the open tables, constants and the step counter. It runs inside the
server process (`python -m auditgym serve`); the client only ever sees JSON.
"""
from __future__ import annotations

import copy
import hashlib
import json

from . import actions
from .actions import MAX_STEPS, PAGE_SIZE, ActionError, error
from .calc import CalcError, calculate

DEFAULT_TIME_LIMIT = 60
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
# Keys under which get_record finds a record in each table. Ids are globally unique tokens.
_ID_FIELDS = {"ledger": ("row_id",), "invoices": ("doc_id", "invoice_id"),
              "payments": ("doc_id", "payment_id")}
_CURSOR_SALT = "auditgym-cursor-v1"


class MonotonicClock:
    """Default clock. Lives only in the evaluator; its readings never leave it."""

    def now(self):
        import time

        return time.monotonic()  # determinism: wall time only bounds the episode, never reaches data


def fx_id(record: dict) -> str:
    return f"{record['date']}|{record['source_currency']}|{record['target_currency']}"


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
        return {"reward": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0,
                "fn": len(episode["private"]["truth"].get("issues", [])), "structure_error": "judge_exception"}


class EvaluatorError(Exception):
    """A request the evaluator refuses outright (bad seed or config, nothing loaded)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Evaluator:
    def __init__(self, mode: str = "train", results_path=None, clock=None,
                 time_limit=DEFAULT_TIME_LIMIT, results_stream=None) -> None:
        """`results_stream` is an already open text stream (the server gets the results file as an
        inherited descriptor, so its path never appears in the process list); it wins over the path."""
        if mode not in MODES:
            raise ValueError("mode must be 'train' or 'eval'")
        self._mode = mode
        self._results_path = None if results_stream is not None else results_path
        self._results_stream = results_stream
        if self._results_path is not None:
            with open(results_path, "a", encoding="utf-8"):
                pass  # fail at start, not at the end of the first episode
        self._clock = clock if clock is not None else MonotonicClock()
        self._time_limit = time_limit
        self._loaded = None   # episode from load(); reset() without seed restarts it
        self._episode = None  # the running episode, public and private
        self._clear()

    @property
    def mode(self) -> str:
        return self._mode

    def _clear(self) -> None:
        self._episode = None
        self._steps = 0
        self._done = False
        self._started_at = None
        self._tables = {}
        self._index = {}
        self._doc_index = {}
        self._cursors = {}

    # --- episode setup ---------------------------------------------------------------------
    def reset(self, seed=None, config=None) -> tuple[dict, dict]:
        """With a seed: generate a new episode. Without: restart the episode from load()."""
        self._clear()
        if seed is None:
            if self._loaded is None:
                raise EvaluatorError("not_loaded")
            episode = copy.deepcopy(self._loaded)
        else:
            from .config import ConfigError
            from .generate import generate

            if not isinstance(seed, int) or isinstance(seed, bool):
                raise EvaluatorError("bad_config")
            try:
                episode = generate(seed, config)
            except ConfigError:
                raise EvaluatorError("bad_config") from None
        return self._start(episode)

    def load(self, directory) -> tuple[dict, dict]:
        from .storage import load_episode

        self._clear()
        self._loaded = None
        try:
            loaded = load_episode(directory)
            observation = self._start(copy.deepcopy(loaded))
        except Exception:  # files are outside input: any defect is one fixed error
            self._clear()
            raise EvaluatorError("bad_episode") from None
        self._loaded = loaded
        return observation

    def _start(self, episode: dict) -> tuple[dict, dict]:
        public = episode["public"]
        self._episode = episode
        episode_id = public["manifest"]["episode_id"]
        for table, fields in _FIELDS.items():
            records = [_project(record, fields) for record in public[table]]
            self._tables[table] = records
            if table == "fx":
                index = {fx_id(record): position for position, record in enumerate(records)}
            else:
                index = {}
                for position, record in enumerate(records):
                    for field in _ID_FIELDS[table]:
                        if isinstance(record[field], str):
                            index.setdefault(record[field], position)
            self._index[table] = index
            self._cursors[table] = {_cursor_token(episode_id, table, offset): offset
                                    for offset in range(1, len(records))}
        for table in ("invoices", "payments"):
            for position, record in enumerate(self._tables[table]):
                if isinstance(record["doc_id"], str):
                    self._doc_index.setdefault(record["doc_id"], (table, position))
        manifest = {field: copy.deepcopy(public["manifest"][field])
                    for field in _MANIFEST_FIELDS if field in public["manifest"]}
        observation = {
            "type": "reset",
            "task": actions.build_task(public["policy"]),
            "manifest": manifest,
            "policy": copy.deepcopy(public["policy"]),
            "steps_remaining": MAX_STEPS,
        }
        self._started_at = self._clock.now()
        return observation, {}

    # --- stepping ----------------------------------------------------------------------------
    def _remaining(self) -> int:
        return MAX_STEPS - self._steps

    def steps_remaining(self) -> int:
        """Steps left in the running episode; 0 when nothing has been reset."""
        return 0 if self._episode is None else self._remaining()

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if self._episode is None:
            return ({"type": "error", "error": error("not_reset"), "steps_remaining": 0},
                    0.0, True, False, {})
        if self._done:
            return ({"type": "error", "error": error("episode_over"), "steps_remaining": self._remaining()},
                    0.0, True, False, {})
        self._steps += 1
        if self._clock.now() - self._started_at >= self._time_limit:
            return self._finish(None, truncated=True)
        try:
            name, arguments = actions.parse_action(action)
            if name == "submit":
                return self._finish(arguments.get("findings"), truncated=False)
            observation = getattr(self, "_do_" + name)(arguments)
        except ActionError as exc:
            observation = {"type": "error", "error": error(exc.code)}
        observation["steps_remaining"] = self._remaining()
        if self._steps >= MAX_STEPS:
            return self._finish(None, truncated=True)
        return observation, 0.0, False, False, {}

    def _finish(self, payload, truncated: bool):
        self._done = True
        if truncated:
            payload = {"episode_id": self._episode["public"]["manifest"]["episode_id"], "findings": []}
            inspection_steps = MAX_STEPS
        else:
            inspection_steps = self._steps - 1  # steps before submit, error steps included
            if payload is None:
                # judge() reads None as "no answer" (a truncated episode). A submit without
                # findings is an invalid structure instead, so it must not be scored as silence.
                payload = {}
        result = _judge(self._episode, payload, inspection_steps)
        self._write_result(result, inspection_steps, truncated)
        observation = {"type": "final", "steps_remaining": self._remaining()}
        if self._mode == "eval":
            return observation, 0.0, not truncated, truncated, {}
        info = {"reward": result["reward"], "precision": result["precision"], "recall": result["recall"],
                "f1": result["f1"], "inspection_steps": inspection_steps}
        return observation, result["reward"], not truncated, truncated, info

    def _write_result(self, result: dict, inspection_steps: int, truncated: bool) -> None:
        if self._results_path is None and self._results_stream is None:
            return
        line = {"episode_id": self._episode["public"]["manifest"]["episode_id"], "mode": self._mode,
                "terminated": not truncated, "truncated": truncated, "inspection_steps": inspection_steps}
        for key in ("reward", "precision", "recall", "f1", "tp", "fp", "fn", "structure_error"):
            line[key] = result.get(key)
        text = json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n"
        if self._results_stream is not None:
            self._results_stream.write(text)
            self._results_stream.flush()
            return
        with open(self._results_path, "a", encoding="utf-8") as handle:
            handle.write(text)

    # --- read actions -----------------------------------------------------------------------
    @staticmethod
    def _table(arguments: dict) -> str:
        table = arguments.get("table")
        if not isinstance(table, str) or table not in _FIELDS:
            raise ActionError("unknown_table")
        return table

    def _do_list_records(self, arguments: dict) -> dict:
        table = self._table(arguments)
        limit = arguments.get("limit", PAGE_SIZE)
        if limit is None:
            limit = PAGE_SIZE
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ActionError("bad_arguments")
        limit = min(limit, PAGE_SIZE)
        cursor = arguments.get("cursor")
        if cursor is None:
            offset = 0
        elif isinstance(cursor, str) and cursor in self._cursors[table]:
            offset = self._cursors[table][cursor]
        else:
            raise ActionError("bad_cursor")
        records = self._tables[table]
        end = min(offset + limit, len(records))
        next_cursor = None if end >= len(records) else _cursor_token(
            self._episode["public"]["manifest"]["episode_id"], table, end)
        return {"type": "records", "table": table,
                "records": copy.deepcopy(records[offset:end]), "next_cursor": next_cursor}

    def _do_get_record(self, arguments: dict) -> dict:
        table = self._table(arguments)
        record_id = arguments.get("id")
        if not isinstance(record_id, str):
            raise ActionError("bad_arguments")
        position = self._index[table].get(record_id)
        if position is None:
            raise ActionError("unknown_id")
        return {"type": "record", "table": table, "record": copy.deepcopy(self._tables[table][position])}

    def _do_get_document(self, arguments: dict) -> dict:
        doc_id = arguments.get("doc_id")
        if not isinstance(doc_id, str):
            raise ActionError("bad_arguments")
        found = self._doc_index.get(doc_id)
        if found is None:
            raise ActionError("unknown_id")
        table, position = found
        return {"type": "document", "table": table, "record": copy.deepcopy(self._tables[table][position])}

    def _do_calculator(self, arguments: dict) -> dict:
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            raise ActionError("bad_arguments")
        try:
            result = calculate(expression)
        except CalcError:
            raise ActionError("bad_expression") from None
        return {"type": "calculation", "result": result}
