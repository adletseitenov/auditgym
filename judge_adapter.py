"""Thin adapter for the jury of Genesis FinTech (docs/ADAPTER.md).

    cd <repo root>            # the directory with judge_adapter.py and auditgym/ (in the container: /app)
    python judge_adapter.py --demo     # four baseline policies through this adapter, on generated episodes

The jury imports this module and calls the three entry points of the published interface:

    generate_episode(seed, config, output_dir)   write one episode directory (public/ + private/)
    make_env()                                   an object with reset(seed, config) and step(action)
    score_episode(episode_dir, findings, inspection_steps)   the closed evaluator call

Nothing here chooses an action instead of a policy and nothing here changes the task: the adapter
translates the envelope {"name", "args"} into the internal action format and the internal replies
into the observations of the published contract. The environment itself runs in a separate process
(auditgym.client.Env: `python -m auditgym serve`); closed-mode rewards, truth and provenance never
cross that boundary towards the caller of `step`.

Standard library only (the core package has a determinism lint; this file follows its spirit: no
randomness, no clock).
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys
import warnings

from auditgym.actions import ACTIONS, ActionError, build_task, default_results_path, open_results, submit_payload
from auditgym.client import Env, EnvError
from auditgym.findings import StructureError, parse_findings
from auditgym.judge import judge
from auditgym.secret import SECRET_ENV, SECRET_FILE_ENV

__all__ = ["generate_episode", "make_env", "score_episode", "AdapterEnv", "AdapterError", "TABLES"]

TABLES = ("ledger", "invoices", "payments", "fx")  # canonical table names of the contract = ours
MAX_STEPS = 100
SCORE_LOG_ENV = "AUDITGYM_SCORE_LOG"
DEFAULT_SCORE_LOG = "runs/score_errors.jsonl"

ERROR_MESSAGES = {
    "unknown_action": "Unknown action.",
    "unknown_table": "Unknown table.",
    "unknown_id": "Unknown id.",
    "bad_cursor": "Bad cursor.",
    "bad_arguments": "Bad arguments.",
    "bad_expression": "Bad expression.",
    "episode_over": "Episode is over.",
    "not_reset": "Call reset first.",
    "invalid_findings": "The findings structure is invalid.",
    "truncated": "The episode ended: step or time limit.",
    "internal_error": "The environment is not available.",
}


class AdapterError(RuntimeError):
    """A refused reset (bad config, unreadable episode directory). `code` is a short fixed identifier."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# ------------------------------------------------------------------------------------ generation
def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Write the episode of (seed, config) into `output_dir`: public/{manifest,policy,ledger,invoices,
    payments,fx} and private/{truth,provenance,pre_*}. The same generator and writer as
    `python -m auditgym export`; the same arguments always give the same bytes. `config` is the
    team config (auditgym/config.py; None or {} is the default). The evaluator secret of the calling
    environment ($AUDITGYM_SECRET_FILE / $AUDITGYM_SECRET) is mixed in exactly as `export` does it;
    without a secret the episode is reproducible from seed and config."""
    from auditgym.export import build_episode
    from auditgym.storage import write_episode

    write_episode(build_episode(seed, config), output_dir)


# ------------------------------------------------------------------------------------ scoring
def _score_log_path(journal_path) -> str:
    return os.fspath(journal_path) if journal_path is not None else os.path.abspath(
        os.environ.get(SCORE_LOG_ENV) or DEFAULT_SCORE_LOG)


def _log_score_error(path: str, episode_id: str, code: str) -> None:
    line = json.dumps({"event": "score_error", "episode_id": episode_id, "structure_error": code},
                      ensure_ascii=False, sort_keys=True) + "\n"
    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(open_results(path), "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        print("judge_adapter: cannot write the score log", file=sys.stderr)


def score_episode(episode_dir: str, findings: dict, inspection_steps: int, journal_path=None) -> dict:
    """The closed evaluator call. `findings` is the findings.json object {"episode_id", "findings": [...]}
    (None means an empty answer: a truncated episode). A valid structure gives
    {"tp","fp","fn","precision","recall","f1","return"}; an invalid one gives return -1 and None in the
    other fields, and the error code goes to the evaluator's score log (`journal_path`, else
    $AUDITGYM_SCORE_LOG, else runs/score_errors.jsonl); the log holds a fixed code, never agent data."""
    from auditgym.storage import load_episode

    episode = load_episode(episode_dir)
    if findings is None:
        findings = {"episode_id": episode["public"]["manifest"]["episode_id"], "findings": []}
    result = judge(episode, findings, inspection_steps)
    if result.get("structure_error"):
        _log_score_error(_score_log_path(journal_path), episode["public"]["manifest"]["episode_id"],
                         str(result["structure_error"]))
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None,
                "return": -1}
    return {"tp": result["tp"], "fp": result["fp"], "fn": result["fn"], "precision": result["precision"],
            "recall": result["recall"], "f1": result["f1"], "return": result["reward"]}


# ------------------------------------------------------------------------------------ environment
@contextlib.contextmanager
def _without_secret_environment():
    """The adapter never hands a generation secret to the environment it starts (Env refuses an eval
    run with a secret in the environment and warns in train): a loaded episode needs none."""
    saved = {key: os.environ.pop(key) for key in list(os.environ) if key.upper() in (SECRET_ENV, SECRET_FILE_ENV)}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            yield
    finally:
        os.environ.update(saved)


def _error(code: str) -> dict:
    return {"code": code, "message": ERROR_MESSAGES[code]}


class AdapterEnv:
    """The environment object of the contract. `reset(seed, config)` and `step(action)` are the whole
    policy-facing surface; `inspection_steps`, `last_result()` and `close()` are for the jury side.

    Modes (config["mode"]): "closed" with config["episode_dir"] loads exactly that episode in the eval
    evaluator (no regeneration, ids untouched; every reset restarts it); "train" with an episode_dir
    loads it in the train evaluator; "train" without one generates with `seed` and config["team_config"];
    "closed" without an episode_dir generates in the eval evaluator the same way. In closed mode the
    reward is always 0.0 and `info` is {}; the true reward is in the evaluator's journal."""

    def __init__(self, journal_path=None, server_command=None) -> None:
        self._journal = os.path.abspath(os.fspath(journal_path) if journal_path is not None
                                        else default_results_path())
        self._server_command = list(server_command) if server_command is not None else None
        self._env = None
        self._key = None
        self._closed_mode = False
        self._episode_id = ""
        self._running = False
        self._inspection_steps = 0
        self._remaining = 0

    # --- reset ------------------------------------------------------------------------------
    def reset(self, seed, config=None):
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise AdapterError("bad_config")
        mode = config.get("mode", "train")
        if mode not in ("closed", "train"):
            raise AdapterError("bad_mode")
        episode_dir = config.get("episode_dir")
        if episode_dir is not None and not isinstance(episode_dir, (str, os.PathLike)):
            raise AdapterError("bad_config")
        team_config = config.get("team_config")
        if team_config is not None and not isinstance(team_config, dict):
            raise AdapterError("bad_config")
        key = (mode, os.path.abspath(os.fspath(episode_dir)) if episode_dir is not None else None)
        try:
            if self._env is None or key != self._key:
                self._start(key)
            if episode_dir is not None:
                observation, _ = self._env.reset()  # restart of the loaded episode, nothing is generated
            else:
                observation, _ = self._env.reset(seed, team_config)
        except EnvError as error:
            self._drop()
            raise AdapterError(error.code) from None
        self._closed_mode = mode == "closed"
        self._episode_id = observation["manifest"]["episode_id"]
        self._running = True
        self._inspection_steps = 0
        self._remaining = MAX_STEPS
        return self._first_observation(observation), {}

    def _start(self, key) -> None:
        self._drop()
        mode, directory = key
        evaluator_mode = "eval" if mode == "closed" else "train"
        options = {"server_command": self._server_command}
        if self._server_command is None:
            options = {"results_path": self._journal if evaluator_mode == "eval" else None}
        with _without_secret_environment():
            if directory is not None:
                self._env = Env.from_files(directory, mode=evaluator_mode, **options)
            else:
                self._env = Env(mode=evaluator_mode, **options)
        self._key = key

    def _drop(self) -> None:
        env, self._env, self._key = self._env, None, None
        self._running = False
        if env is not None:
            env.close()

    @staticmethod
    def _first_observation(internal: dict) -> dict:
        task = internal["task"]
        text = (task["text"] + "\n\nerror_types: " + json.dumps(task["error_types"])
                + "\nlimits: " + json.dumps(task["limits"], sort_keys=True)
                + "\nfindings_schema: " + json.dumps(task["findings_schema"], ensure_ascii=False, sort_keys=True)
                + "\nanswer_rules: " + json.dumps(task["answer_rules"], ensure_ascii=False, sort_keys=True))
        return {"task": text, "manifest": internal["manifest"], "policy": internal["policy"],
                "remaining_steps": internal["steps_remaining"]}

    # --- step -------------------------------------------------------------------------------
    @staticmethod
    def _internal_action(action) -> tuple[str | None, object]:
        """(name, internal action). An unknown or malformed envelope becomes an action the evaluator
        answers with a bounded error and a spent step (None is read as an unknown action)."""
        if not isinstance(action, dict):
            return None, None
        name = action.get("name")
        if not isinstance(name, str) or name not in ACTIONS:
            return None, None
        args = action.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return name, {"action": name, "args": []}  # not an object: bad_arguments (a submit: invalid structure)
        return name, {"action": name, "args": _fx_id(name, args)}

    def step(self, action):
        if self._env is None:
            return self._plain(_error("not_reset"), 0), 0.0, False, False, {}
        name, internal = self._internal_action(action)
        was_running = self._running
        try:
            observation, reward, terminated, truncated, info = self._env.step(internal)
        except EnvError:
            self._running = False
            return self._plain(_error("internal_error"), 0), 0.0, False, True, {}
        if was_running and name != "submit":
            self._inspection_steps += 1
        remaining = observation.get("steps_remaining", 0)
        if terminated or truncated:
            self._running = False
        self._remaining = remaining
        translated = self._observation(observation, name, action, truncated, was_running)
        if self._closed_mode:
            return translated, 0.0, bool(terminated), bool(truncated), {}
        return translated, reward, bool(terminated), bool(truncated), info

    @staticmethod
    def _plain(error: dict, remaining: int) -> dict:
        return {"result": None, "error": error, "remaining_steps": remaining}

    def _observation(self, internal: dict, name, action, truncated: bool, was_running: bool) -> dict:
        remaining = internal.get("steps_remaining", 0)
        kind = internal.get("type")
        if kind == "error":
            return self._plain(dict(internal["error"]), remaining)
        if kind == "final":
            if truncated:
                return self._plain(_error("truncated"), 0)
            if name == "submit" and self._valid_submit(action):
                return {"result": {"accepted": True}, "error": None, "remaining_steps": remaining}
            return self._plain(_error("invalid_findings"), remaining)
        if kind == "records":
            result = {"records": internal["records"], "next_cursor": internal["next_cursor"]}
        elif kind in ("record", "document"):
            result = internal["record"]
        elif kind == "calculation":
            result = internal["result"]
        else:
            return self._plain(_error("internal_error"), remaining)
        return {"result": result, "error": None, "remaining_steps": remaining}

    def _valid_submit(self, action: dict) -> bool:
        """Whether the findings of this submit have a valid structure (the same parser the evaluator
        uses); nothing about their correctness is known here."""
        args = action.get("args")
        try:
            parse_findings(submit_payload(args if isinstance(args, dict) else {}, self._episode_id), self._episode_id)
        except (ActionError, StructureError):
            return False
        return True

    # --- jury side ----------------------------------------------------------------------------
    @property
    def inspection_steps(self) -> int:
        """Actions before the submit, errors included (a submit is not one). After a time-limit
        truncation the evaluator's journal row is authoritative: it does not charge the step it did
        not run, this counter does."""
        return self._inspection_steps

    def last_result(self) -> dict | None:
        """The last `finished` row of the evaluator journal for the current episode (true reward,
        TP/FP/FN, inspection_steps): the evaluator side, never a policy channel. None when the journal
        is not readable here (a server started with `server_command` keeps its own journal)."""
        if self._server_command is not None:
            return None
        try:
            with open(self._journal, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        except (OSError, ValueError):
            return None
        rows = [row for row in rows if row.get("event") == "finished" and row.get("episode_id") == self._episode_id]
        return rows[-1] if rows else None

    def close(self) -> None:
        self._drop()

    def __enter__(self) -> "AdapterEnv":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _fx_id(name: str, args: dict) -> dict:
    """The fx key of the contract is date:source:target; the environment writes it date|source|target.
    A get_record on fx accepts both."""
    table, record_id = args.get("table"), args.get("id")
    if name == "get_record" and table == "fx" and isinstance(record_id, str) and "|" not in record_id \
            and record_id.count(":") == 2:
        return dict(args, id=record_id.replace(":", "|"))
    return args


def make_env(journal_path=None, server_command=None) -> AdapterEnv:
    """The environment object. `journal_path`: the evaluator journal of closed runs (default
    $AUDITGYM_RESULTS or runs/results.jsonl, as for `python -m auditgym serve --mode eval`);
    `server_command`: argv of an evaluator started elsewhere (README «Закрытая проверка»), which then
    owns its journal."""
    return AdapterEnv(journal_path=journal_path, server_command=server_command)


def main(argv: list[str]) -> int:
    if argv[:1] == ["--demo"]:
        from tools.adapter_run import main as demo_main

        return demo_main(argv[1:])
    print(__doc__.strip().split("\n\n")[0] + "\n\nusage: python judge_adapter.py --demo [--seeds N]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
