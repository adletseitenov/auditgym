"""Evaluator: the episode state on the evaluator side (docs/CONTRACTS.md, sections 5 and 7).

The evaluator is the only place that holds `episode["private"]`. Everything it returns to a
policy is built from the open tables, constants and the step counter. It runs inside the
server process (`python -m auditgym serve`); the client only ever sees JSON.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys

from . import actions, shaped
from .actions import MAX_STEPS, ActionError, error
from .calc import CalcError, calculate
from .limits import MAX_SEED_BITS  # a longer seed is refused as bad_config; the running episode is untouched

DEFAULT_TIME_LIMIT = 60
DEFAULT_SEED = 0  # first reset() without a seed and without a loaded episode (audit C7)
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
# Keys under which get_record finds a record in each table, in priority order: a doc_id is looked up
# before an invoice_id/payment_id, so a renumbered id of one record can never shadow another
# record's doc_id (audit R10).
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
        print(f"auditgym evaluator: judge raised {type(exc).__name__}", file=sys.stderr)
        # A defect of ours is never a free win: reward -1, no credit, and the journal says why.
        try:
            missed = len(episode["private"]["truth"].get("issues", []))
        except Exception:  # noqa: BLE001 - the episode itself may be what is broken
            missed = 0
        return {"reward": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0,
                "fn": missed, "structure_error": "judge_exception"}


def open_results_file(path) -> None:
    """Create the results file (mode 0600: it holds rewards the policy must not read) or tighten
    an existing one, and fail now if that is impossible (audit L3; actions.open_results)."""
    os.close(actions.open_results(path))


# The generation secret and its sources live in secret.py; SECRET_ENV stays importable from here.
from .secret import SECRET_ENV, journal_key as _journal_key, read_secret  # noqa: E402


def _seed_from_text(text: str) -> int:
    """A seed given as a string of ASCII digits ("42", spaces at the ends allowed) is that number.
    No sign, no separators, no other digits (str.isdigit would accept Arabic-Indic ones); anything
    else, and a number of more than MAX_SEED_BITS bits, is bad_config."""
    text = text.strip(" \t\r\n")
    if not re.fullmatch(r"[0-9]+", text, re.ASCII):
        raise EvaluatorError("bad_config")
    digits = text.lstrip("0")
    if len(digits) > 100:  # far beyond MAX_SEED_BITS; also keeps int() clear of its digit limit
        raise EvaluatorError("bad_config")
    return int(digits or "0")


class EvaluatorError(Exception):
    """A request the evaluator refuses outright (bad seed or config, nothing loaded)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Evaluator:
    def __init__(self, mode: str = "train", results_path=None, clock=None,
                 time_limit=DEFAULT_TIME_LIMIT, results_stream=None, shaped_reward=False,
                 secret: str | None = None, journal_key: bytes | None = None) -> None:
        """`results_stream` is an already open text stream (the server gets the results file as an
        inherited descriptor, so its path never appears in the process list); it wins over the path.
        `shaped_reward` adds the experimental partial-credit reward (auditgym/shaped.py) to the terminal
        `info` in train mode; it is off by default and ignored in eval mode, always.
        `secret` is the generation secret; None reads it once, now, from $AUDITGYM_SECRET_FILE or
        $AUDITGYM_SECRET (auditgym/secret.py; an unusable secret file raises SecretError).
        `journal_key` signs the journal (auditgym/journal.py); None takes $AUDITGYM_JOURNAL_KEY_FILE,
        $AUDITGYM_JOURNAL_KEY or a key derived from the secret, and no secret means no signature."""
        if mode not in MODES:
            raise ValueError("mode must be 'train' or 'eval'")
        self._mode = mode
        self._shaped = bool(shaped_reward) and mode == "train"
        self._results_path = None if results_stream is not None else results_path
        self._results_stream = results_stream
        if self._results_path is not None:
            open_results_file(self._results_path)  # fail at start, not at the end of the first episode
        self._secret = read_secret() if secret is None else secret
        self._journal_key = _journal_key(secret=self._secret) if journal_key is None else journal_key
        self._clock = clock if clock is not None else MonotonicClock()
        self._time_limit = time_limit
        self._loaded = None   # episode from load(); reset() without seed restarts it
        self._next_seed = DEFAULT_SEED  # seed of the next reset() without a seed and without files
        self._episode = None  # the running episode, public and private
        self._resets_abandoned = 0  # episodes of this session replaced by reset/load while running
        self._episode_runs = {}   # episode_id -> journal rows written for it in this session (episode_run)
        self._clear()

    @property
    def mode(self) -> str:
        return self._mode

    def _clear(self) -> None:
        self._episode = None
        self._steps = 0
        self._done = False
        self._truncated = False
        self._started_at = None
        self._from_loaded = False  # the running episode is the one from load()
        self._tables = {}
        self._index = {}
        self._doc_index = {}
        self._cursors = {}

    # --- episode setup ---------------------------------------------------------------------
    def reset(self, seed=None, config=None) -> tuple[dict, dict]:
        """With a seed: generate a new episode. Without: restart the episode from load(), or, when
        none was loaded, generate one from the next default seed with the default configuration:
        DEFAULT_SEED (0) the first time, then the seed of the previous successful reset plus one.
        A refused reset (bad seed, a seed of more than MAX_SEED_BITS bits, bad config) leaves the
        running episode and the default-seed counter as they are."""
        if seed is None:
            if self._loaded is not None:
                if config not in (None, {}):  # a loaded episode has no config to apply (CONTRACTS §4)
                    raise EvaluatorError("bad_config")
                # A restart is allowed in eval as it is for an episode by seed (the jury may test that
                # reset isolates the episode); what was spent before it is journaled as `abandoned`
                # by _swap_in (audit A1), and the new run starts with the full budget.
                return self._swap_in(copy.deepcopy(self._loaded), from_loaded=True)
            seed = self._next_seed
        elif isinstance(seed, str):
            seed = _seed_from_text(seed)
        elif not isinstance(seed, int) or isinstance(seed, bool):
            raise EvaluatorError("bad_config")
        if seed.bit_length() > MAX_SEED_BITS:
            raise EvaluatorError("bad_config")
        observation = self._swap_in(self._generate(seed, config))
        following = seed + 1  # past the largest seed the default sequence starts again at 0
        self._next_seed = following if following.bit_length() <= MAX_SEED_BITS else DEFAULT_SEED
        return observation

    def _generate(self, seed: int, config) -> dict:
        from .config import ConfigError
        from .generate import generate

        try:
            # The evaluator-side secret is mixed into every random stream, so knowing the
            # generator code and the seed is not enough to rebuild the hidden truth. Empty by
            # default: open train/dev episodes stay reproducible from seed and config.
            return generate(seed, config, secret=self._secret)
        except ConfigError:
            raise EvaluatorError("bad_config") from None

    def load(self, directory) -> tuple[dict, dict]:
        from .storage import load_episode

        try:
            loaded = load_episode(directory)
            observation = self._swap_in(copy.deepcopy(loaded), from_loaded=True)
        except EvaluatorError:
            raise
        except Exception:  # files are outside input: any defect is one fixed error
            raise EvaluatorError("bad_episode") from None
        self._loaded = loaded
        return observation

    def _swap_in(self, episode: dict, from_loaded: bool = False) -> tuple[dict, dict]:
        """Start `episode`; if that fails, the previous state is put back untouched."""
        # A running episode whose 60 s are over is truncated now, as the next step would do; the
        # time limit is checked at reset, load and close too, not only at a step (audit A2).
        # Restarting the loaded episode before its first step is not a new attempt: it showed
        # nothing new, and a harness may load long before the policy resets.
        if self.running and self._expired() and (self._steps > 0 or not (from_loaded and self._from_loaded)):
            self._finish(None, truncated=True)
        saved = dict(self.__dict__)
        # A reset of a running episode is allowed (the jury may test that reset isolates the
        # episode), but it is never silent: the abandoned episode goes to the journal with the
        # steps it spent, so a restart cannot hide 99 inspection steps (audit A2). An episode that
        # has not spent a step (Env.from_files loads, then reset() restarts it) showed nothing but
        # the reset observation and is not journaled.
        abandoned = (self._episode, self._steps) if self.running and self._steps > 0 else None
        self._clear()
        try:
            started = self._start(episode)
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(saved)
            raise
        self._from_loaded = from_loaded
        if abandoned is not None:
            self._resets_abandoned += 1
            self._write_abandoned(*abandoned)
        return started

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
                for field in _ID_FIELDS[table]:  # one pass per key: doc_id first (R10)
                    for position, record in enumerate(records):
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
            "policy": actions.open_policy(public["policy"]),
            "steps_remaining": MAX_STEPS,
        }
        self._started_at = self._clock.now()
        return observation, {}

    # --- stepping ----------------------------------------------------------------------------
    def _remaining(self) -> int:
        return MAX_STEPS - self._steps

    def steps_remaining(self) -> int:
        """Steps left in the running episode; 0 when nothing has been reset or after a truncation
        (the same 0 as in the final observation, whichever limit ended the episode)."""
        return 0 if self._episode is None or self._truncated else self._remaining()

    def _expired(self) -> bool:
        return self._started_at is not None and self._clock.now() - self._started_at >= self._time_limit

    def end(self) -> None:
        """The input ended, a close arrived or the server stops (audit A2): a running episode that
        had a step, or whose 60 s are over, is truncated now and leaves its journal row (empty
        answer, 100 inspection steps, "ended_by": "close"). Otherwise a policy could keep an
        episode out of the journal by closing the pipe or stopping to answer."""
        if self.running and (self._steps > 0 or self._expired()):
            self._finish(None, truncated=True, ended_by="close")

    @property
    def running(self) -> bool:
        """An episode is in progress (reset, not finished)."""
        return self._episode is not None and not self._done

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if self._episode is None:
            return ({"type": "error", "error": error("not_reset"), "steps_remaining": 0},
                    0.0, False, False, {})  # a protocol error, not an end of episode
        if self._done:
            # After a truncation every later step says so, too: terminated=false, truncated=true.
            return ({"type": "error", "error": error("episode_over"),
                     "steps_remaining": 0 if self._truncated else self._remaining()},
                    0.0, not self._truncated, self._truncated, {})
        self._count_step()
        if self._expired():
            return self._finish(None, truncated=True)
        try:
            name, arguments = actions.parse_action(action)
            if name == "submit":
                episode_id = self._episode["public"]["manifest"]["episode_id"]
                return self._finish(actions.submit_payload(arguments, episode_id), truncated=False)
            observation = getattr(self, "_do_" + name)(arguments)
        except ActionError as exc:
            if exc.action == "submit":  # a malformed submit wrapper is an invalid structure (C9)
                return self._finish({}, truncated=False)
            observation = {"type": "error", "error": error(exc.code)}
        return self._after_step(observation)

    def _count_step(self) -> None:
        self._steps += 1

    def _after_step(self, observation: dict):
        observation["steps_remaining"] = self._remaining()
        if self._steps >= MAX_STEPS:
            return self._finish(None, truncated=True)
        return observation, 0.0, False, False, {}

    def reject(self, submit: bool = False):
        """A request line the server could not take as an action (too long, audit L4) during an
        episode: a step with an error, or, if it was a submit, an invalid structure. After the end of
        the episode: the same episode_over reply as a step."""
        if self._episode is not None and self._done:
            return self.step(None)
        if not self.running:
            raise EvaluatorError("bad_request")
        self._count_step()
        if self._expired():
            return self._finish(None, truncated=True)
        if submit:
            return self._finish({}, truncated=False)
        return self._after_step({"type": "error", "error": error("bad_arguments")})

    def _finish(self, payload, truncated: bool, ended_by: str | None = None):
        self._done = True
        self._truncated = truncated
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
        self._write_result(result, inspection_steps, truncated, ended_by, steps_total=self._steps)
        # A truncated episode has no steps left, whichever limit ended it (audit C10).
        observation = {"type": "final", "steps_remaining": 0 if truncated else self._remaining()}
        if self._mode == "eval":
            return observation, 0.0, not truncated, truncated, {}
        info = {"reward": result["reward"], "precision": result["precision"], "recall": result["recall"],
                "f1": result["f1"], "inspection_steps": inspection_steps}
        if self._shaped:
            info[shaped.INFO_KEY] = self._shaped_reward(payload, inspection_steps)
        return observation, result["reward"], not truncated, truncated, info

    def _shaped_reward(self, payload, inspection_steps: int) -> float:
        try:
            return shaped.shaped_reward(self._episode, payload, inspection_steps)
        except Exception as exc:  # an experimental number must not crash the episode; -1 as for a judge defect
            print(f"auditgym evaluator: shaped reward raised {type(exc).__name__}", file=sys.stderr)
            return -1.0

    def _write_result(self, result: dict, inspection_steps: int, truncated: bool, ended_by=None,
                      steps_total: int | None = None) -> None:
        line = {"episode_id": self._episode["public"]["manifest"]["episode_id"], "mode": self._mode,
                "event": "finished", "terminated": not truncated, "truncated": truncated,
                "inspection_steps": inspection_steps,
                # every step taken, the submit included (inspection_steps stops before it)
                "steps_total": inspection_steps if steps_total is None else steps_total}
        if ended_by is not None:
            line["ended_by"] = ended_by
        for key in ("reward", "precision", "recall", "f1", "tp", "fp", "fn", "structure_error"):
            line[key] = result.get(key)
        self._write_line(line)

    def _write_abandoned(self, episode: dict, steps: int) -> None:
        """The journal row of an episode replaced by reset/load before it ended: no reward, the
        steps it spent, and how many episodes this session has abandoned so far."""
        line = {"episode_id": episode["public"]["manifest"]["episode_id"], "mode": self._mode,
                "event": "abandoned", "terminated": False, "truncated": False,
                "inspection_steps": steps, "steps_total": steps, "abandoned_in_session": self._resets_abandoned}
        for key in ("reward", "precision", "recall", "f1", "tp", "fp", "fn", "structure_error"):
            line[key] = None
        self._write_line(line)

    def _write_line(self, line: dict) -> None:
        if self._results_path is None and self._results_stream is None:
            return
        # The how-many-th row of this episode_id in this session: a harness sees a replayed episode
        # (the same seed reset again, a loaded episode restarted in train mode) at a glance.
        episode_id = line["episode_id"]
        self._episode_runs[episode_id] = self._episode_runs.get(episode_id, 0) + 1
        line["episode_run"] = self._episode_runs[episode_id]
        if self._journal_key is not None:
            from .journal import seal

            line = seal(self._journal_key, line, self._journal_tail())
        text = json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n"
        try:
            if self._results_stream is not None:
                self._results_stream.write(text)
                self._results_stream.flush()
            else:
                with os.fdopen(actions.open_results(self._results_path), "a", encoding="utf-8") as handle:
                    handle.write(text)
        except OSError:  # a full disk must not turn a finished episode into a server error
            print("auditgym evaluator: cannot write the results file", file=sys.stderr)

    def _journal_tail(self) -> bytes:
        """The last bytes of the journal, read before every signed write so the chain continues
        from whatever line is last now (another session may have appended since)."""
        from .journal import read_tail_fd, read_tail_path

        try:
            if self._results_stream is not None:
                self._results_stream.flush()
                return read_tail_fd(self._results_stream.fileno())
            return read_tail_path(self._results_path)
        except (OSError, AttributeError, ValueError):  # not readable: a new chain, verify shows the break
            return b""

    # --- read actions -----------------------------------------------------------------------
    @staticmethod
    def _table(arguments: dict) -> str:
        table = actions.table_name(arguments.get("table"))
        if table is None:
            raise ActionError("unknown_table")
        return table

    def _do_list_records(self, arguments: dict) -> dict:
        table = self._table(arguments)
        limit, clamped = actions.parse_limit_clamped(arguments.get("limit"))
        cursor = arguments.get("cursor")
        if actions.is_first_page(cursor):
            offset = 0
        elif isinstance(cursor, str) and cursor in self._cursors[table]:
            offset = self._cursors[table][cursor]
        else:
            raise ActionError("bad_cursor")
        records = self._tables[table]
        end = min(offset + limit, len(records))
        next_cursor = None if end >= len(records) else _cursor_token(
            self._episode["public"]["manifest"]["episode_id"], table, end)
        observation = {"type": "records", "table": table,
                       "records": copy.deepcopy(records[offset:end]), "next_cursor": next_cursor}
        if clamped:
            observation["limit_clamped"] = True  # only when a limit above the page size was cut
        return observation

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
