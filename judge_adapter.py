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

import sys

if sys.version_info < (3, 11):  # before any other import: older interpreters fail later with obscure errors
    raise SystemExit("%s needs Python 3.11 or newer; this is Python %d.%d. Run it with python3.11 (docs/USAGE.md)."
                     % ("judge_adapter", sys.version_info[0], sys.version_info[1]))

import contextlib
import json
import os
import pathlib
import re
import tempfile
import warnings

from auditgym.actions import ACTIONS, SUBMIT_FINDINGS_KEYS, build_task, default_results_path, open_results
from auditgym.client import Env, EnvError
from auditgym.evaluator import _directory_sha256
from auditgym.findings import StructureError, parse_findings
from auditgym.judge import judge
from auditgym.limits import json_depth_exceeds
from auditgym.secret import SECRET_ENV, SECRET_FILE_ENV

__all__ = ["generate_episode", "make_env", "score_episode", "AdapterEnv", "AdapterError", "TABLES"]

TABLES = ("ledger", "invoices", "payments", "fx")  # canonical table names of the contract = ours
MAX_STEPS = 100
_CLAMP_PROMISE = " (ответ тогда содержит limit_clamped: true)"
# The internal task text promises a bare list; the adapter takes only the findings.json object (DECISIONS AC).
_BARE_LIST_PROMISE = ("submit принимает findings как объект {episode_id, findings}, "
                      "а голый список findings принимается как findings текущего эпизода; ")
_STRICT_SUBMIT = ("submit принимает findings только как JSON-объект {episode_id, findings: [...]}: "
                  "строка JSON, голый список и объект без episode_id — неверная структура, награда −1; ")
ADAPTER_SECRET_FILE_ENV = "AUDITGYM_ADAPTER_SECRET_FILE"  # signing secret for the eval evaluator (make_env)
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
def generate_episode(seed: int, config: dict, output_dir: str, secret_file=None) -> None:
    """Write the episode of (seed, config) into `output_dir`: public/{manifest,policy,ledger,invoices,
    payments,fx} and private/{truth,provenance,pre_*}. The same generator and writer as
    `python -m auditgym export`; the same arguments always give the same bytes. `config` is the
    team config (auditgym/config.py; None or {} is the default), or the reset-style wrapper with a
    `team_config` key (its value is the team config; `mode` and `episode_dir` are ignored). The evaluator secret of the calling
    environment ($AUDITGYM_SECRET_FILE / $AUDITGYM_SECRET), or the explicit optional `secret_file`
    (it wins), is mixed in exactly as `export` does it. Without a secret the episode is reproducible
    from seed and config, and so a policy that has this package can recover a hidden episode's seed
    from its episode_id by trying seeds (measured: seed 7600 takes 7601 candidates, about 3-5 minutes on one core, 27-43 candidates/s; `python3.11 -m tools.seed_recovery --bits 13 --seed 7600`): for hidden episodes the jury
    should pass a secret (docs/ADAPTER.md, docs/LIMITATIONS.md)."""
    from auditgym.export import build_episode
    from auditgym.secret import read_secret
    from auditgym.storage import write_episode

    if secret_file is None and not read_secret():
        warnings.warn("generate_episode: no secret (no secret_file, no $AUDITGYM_SECRET_FILE / $AUDITGYM_SECRET): "
                      "the episode is reproducible from the seed; for hidden runs pass secret_file",
                      RuntimeWarning, stacklevel=2)
    if isinstance(config, dict) and "team_config" in config:
        # the reset-style wrapper {"mode": ..., "episode_dir": ..., "team_config": {...}}: the team
        # config is the value, mode/episode_dir mean nothing for a generation
        config = config["team_config"]
    write_episode(build_episode(seed, config, secret_file=secret_file), output_dir)


# ------------------------------------------------------------------------------------ scoring
def _score_log_path(journal_path) -> str:
    """Explicit `journal_path`, else $AUDITGYM_SCORE_LOG, else <cwd>/runs/score_errors.jsonl made
    absolute at call time, and, when that directory cannot be written, the same file name under the
    system temporary directory (auditgym/ there). Always absolute."""
    if journal_path is not None:
        return os.path.abspath(os.fspath(journal_path))
    if os.environ.get(SCORE_LOG_ENV):
        return os.path.abspath(os.environ[SCORE_LOG_ENV])
    primary = os.path.abspath(DEFAULT_SCORE_LOG)
    directory = os.path.dirname(primary)
    try:
        os.makedirs(directory, exist_ok=True)
        if os.access(directory, os.W_OK):
            return primary
    except OSError:
        pass
    return os.path.join(tempfile.gettempdir(), "auditgym", os.path.basename(primary))


def _default_journal_path() -> str:
    """The evaluator journal of a make_env() without an explicit path: $AUDITGYM_RESULTS or
    <cwd>/runs/results.jsonl made absolute at call time; when that file cannot be opened for append
    (non-root user, read-only file system), the same file name under the system temporary directory
    (auditgym/ there), as for the score log. When neither can be opened the primary path is returned
    and the reset refuses it with bad_results_path. Always absolute."""
    primary = os.path.abspath(default_results_path())
    candidates = [primary, os.path.join(tempfile.gettempdir(), "auditgym", os.path.basename(primary))]
    for path in candidates:
        try:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            os.close(open_results(path))
            return path
        except OSError:
            continue
    return primary


def _log_score_error(path: str, episode_id: str, code: str) -> None:
    """Append the score_error row. A failure is loud: AdapterError("score_log_unwritable") (the log is
    the only place the structure error code goes; losing it silently would hide it from the jury)."""
    line = json.dumps({"event": "score_error", "episode_id": episode_id, "structure_error": code},
                      ensure_ascii=False, sort_keys=True) + "\n"
    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(open_results(path), "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        print(f"judge_adapter: cannot write the score log {path}", file=sys.stderr)
        raise AdapterError("score_log_unwritable") from None


def score_episode(episode_dir: str, findings: dict, inspection_steps: int, journal_path=None) -> dict:
    """The closed evaluator call. `findings` is the findings.json object {"episode_id": str, "findings": [...]},
    the same strict form a submit step accepts: a JSON text, a bare list or a missing episode_id is a
    structure error (-1). None means an empty answer: a truncated episode. A valid structure gives
    {"tp","fp","fn","precision","recall","f1","return"}; an invalid one gives return -1 and None in the
    other fields, and the error code goes to the evaluator's score log (`journal_path`, else
    $AUDITGYM_SCORE_LOG, else an absolute runs/score_errors.jsonl, else the same name under the system
    temporary directory); the log holds a fixed code, never agent data. If the log cannot be written,
    AdapterError("score_log_unwritable") is raised. The structure (30 findings, 10 evidence ids per
    finding, 1 000 code points of explanation, types) is checked on the raw answer first; only then
    FX keys of the contract `date:src:dst` in evidence_ids/affected_ids are read as our `date|src|dst`
    (a renaming: no id is dropped or merged; every other id is compared exactly; docs/ADAPTER.md)."""
    from auditgym.storage import load_episode

    episode = load_episode(episode_dir)
    if findings is None:
        findings = {"episode_id": episode["public"]["manifest"]["episode_id"], "findings": []}
    # strict, as a submit step: only the findings.json object {episode_id, findings: [...]} is a valid
    # structure; a JSON text or a bare list reaches the parser as it is and is refused (not_object)
    episode_id = episode["public"]["manifest"]["episode_id"]
    result = judge(episode, _checked_message(findings, episode_id), inspection_steps)
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


def _content_digest(directory):
    """Hash of the files of an episode directory (the evaluator's own `source_sha256` function), or
    None when it cannot be read: the evaluator then refuses the load with its own error code."""
    if directory is None:
        return None
    try:
        return _directory_sha256(directory)
    except OSError:
        return None


def _error(code: str) -> dict:
    return {"code": code, "message": ERROR_MESSAGES[code]}


class AdapterEnv:
    """The environment object of the contract. `reset(seed, config)` and `step(action)` are the whole
    policy-facing surface; `inspection_steps`, `last_result()` and `close()` are for the jury side.

    Modes (config["mode"]): "closed" with config["episode_dir"] loads exactly that episode in the eval
    evaluator (no regeneration, ids untouched; every reset restarts it); "train" with an episode_dir
    loads it in the train evaluator; "train" without one generates with `seed` and config["team_config"];
    "closed" without an episode_dir generates in the eval evaluator under the signing secret and is
    refused with AdapterError("secret_required") when none is available (no `secret_file`, no
    `server_command`): an episode reproducible from the seed is not closed. In closed mode the
    reward is always 0.0 and `info` is {}; the true reward is in the evaluator's journal."""

    def __init__(self, journal_path=None, server_command=None, secret_file=None) -> None:
        self._journal = os.path.abspath(os.fspath(journal_path)) if journal_path is not None \
            else _default_journal_path()
        self._server_command = list(server_command) if server_command is not None else None
        if secret_file is None and server_command is None:
            secret_file = os.environ.get(ADAPTER_SECRET_FILE_ENV) or None
        self._secret_file = os.fspath(secret_file) if secret_file is not None else None
        self._env = None
        self._key = None
        self._closed_mode = False
        self._episode_id = ""
        self._running = False
        self._inspection_steps = 0
        self._remaining = 0
        self._submit_valid = False  # the structure verdict of the last submit sent (same pipeline as the evaluator)

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
        if episode_dir is not None and "mode" not in config:
            # explicit is safer: a missing mode must not silently pick train (reward visible) for a given episode
            raise AdapterError("bad_mode")
        team_config = config.get("team_config")
        if team_config is not None and not isinstance(team_config, dict):
            raise AdapterError("bad_config")
        if mode == "closed" and episode_dir is None and self._secret_file is None and self._server_command is None:
            # a closed episode generated without a secret is reproducible from the seed: not closed
            raise AdapterError("secret_required")
        directory = os.path.abspath(os.fspath(episode_dir)) if episode_dir is not None else None
        # The content of the directory is part of the key: the jury may write another episode into the
        # same path between two resets, and a reset must serve what is on disk now.
        key = (mode, directory, _content_digest(directory))
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
        mode, directory, _ = key
        evaluator_mode = "eval" if mode == "closed" else "train"
        options = {"server_command": self._server_command}
        if self._server_command is None:
            options = {"results_path": self._journal if evaluator_mode == "eval" else None}
            if evaluator_mode == "eval" and self._secret_file is not None:
                # signing only: a loaded episode is not regenerated from the secret; a reset without
                # episode_dir does generate under it (that is what the secret is for)
                options["secret_file"] = self._secret_file
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
        # The adapter's list_records result is the official {records, next_cursor}: it has no limit_clamped,
        # so the internal task text must not promise it (DECISIONS Y).
        base_text = task["text"].replace(_CLAMP_PROMISE, "").replace(_BARE_LIST_PROMISE, _STRICT_SUBMIT)
        text = (base_text + "\n\nerror_types: " + json.dumps(task["error_types"])
                + "\nlimits: " + json.dumps(task["limits"], sort_keys=True)
                + "\nfindings_schema: " + json.dumps(task["findings_schema"], ensure_ascii=False, sort_keys=True)
                + "\nanswer_rules: " + json.dumps(task["answer_rules"], ensure_ascii=False, sort_keys=True))
        return {"task": text, "manifest": internal["manifest"], "policy": internal["policy"],
                "remaining_steps": internal["steps_remaining"]}

    # --- step -------------------------------------------------------------------------------
    def _internal_action(self, action) -> tuple[str | None, object]:
        """(name, internal action). An unknown or malformed envelope becomes an action the evaluator
        answers with a bounded error and a spent step (None is read as an unknown action)."""
        self._submit_valid = False
        if not isinstance(action, dict):
            return None, None
        name = action.get("name")
        if not isinstance(name, str) or name not in ACTIONS:
            return None, None
        args = action.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict) or not _serialisable(args):
            # not an object, or not JSON (a set, NaN, too deep): bad_arguments (a submit: invalid structure)
            self._submit_valid = False
            return name, {"action": name, "args": []}
        args = _fx_id(name, args)
        if name == "submit":
            args, self._submit_valid = _prepare_submit(args, self._episode_id)
        return name, {"action": name, "args": args}

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
            if name == "submit" and self._submit_valid:
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


def _serialisable(args: dict) -> bool:
    try:
        if json_depth_exceeds(args):
            return False
        json.dumps(args, ensure_ascii=True, allow_nan=False)
    except Exception:  # noqa: BLE001 - a hostile object may raise anything
        return False
    return True


_FX_COLON = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}:[A-Z]{3}:[A-Z]{3}")


def _fx_text(value):
    """`date:src:dst` (exactly the contract's FX key shape) as our `date|src|dst`; anything else as is.
    A key that names no fx record stays a non-existent id after the translation, as before it."""
    if isinstance(value, str) and _FX_COLON.fullmatch(value):
        return value.replace(":", "|")
    return value


def _translate_fx_message(message):
    """The findings message with colon FX keys in evidence_ids and affected_ids renamed; the same object
    when nothing changes. Only a renaming: lists keep their length and order (no id is dropped or
    merged). Shapes the parser would refuse are returned untouched."""
    if not isinstance(message, dict) or not isinstance(message.get("findings"), list):
        return message
    changed, findings = False, []
    for finding in message["findings"]:
        if isinstance(finding, dict):
            update = {}
            for field in ("evidence_ids", "affected_ids"):
                values = finding.get(field)
                if isinstance(values, list):
                    translated = [_fx_text(value) for value in values]
                    if translated != values:
                        update[field] = translated
            if update:
                finding, changed = dict(finding, **update), True
        findings.append(finding)
    return dict(message, findings=findings) if changed else message


def _checked_message(message, episode_id: str):
    """The message to judge. The structure is validated on the RAW message (the limits of CHECKS count
    the submitted lists); a refused structure is returned as it came, so the judge names the same error.
    A valid one is returned with the FX keys renamed (the structure cannot change by a renaming)."""
    try:
        parse_findings(message, episode_id)
    except StructureError:
        return message
    translated = _translate_fx_message(message)
    try:
        parse_findings(translated, episode_id)
    except StructureError:
        return message
    return translated


def _strict_message(args: dict):
    """The submitted findings.json object, or None when the answer is not one. The adapter accepts only
    the documented payload {"findings": {"episode_id": str, "findings": [...]}} (also under the aliases
    `findings.json` / `findings_json`): no JSON text, no bare list, no flat episode_id filling in a
    missing one, no JSON text in the inner `findings`."""
    given = [args[key] for key in SUBMIT_FINDINGS_KEYS if args.get(key) is not None]
    if not given or any(value != given[0] for value in given[1:]):
        return None
    message = given[0]
    if not (isinstance(message, dict) and isinstance(message.get("episode_id"), str)
            and isinstance(message.get("findings"), list)):
        return None
    return message


def _prepare_submit(args: dict, episode_id: str) -> tuple[dict, bool]:
    """(arguments for the evaluator, structure valid). The raw submitted answer is validated first by the
    evaluator's own parser; the same verdict is what the policy is told. An invalid one is passed on
    untouched (the evaluator refuses it with its own code); a valid one with its FX keys renamed."""
    message = _strict_message(args)
    if message is None:
        # a JSON text, a bare list, two different answers or nothing: no repair. The evaluator is given
        # an object without episode_id, which its own parser refuses, so the journal and the policy
        # get the same verdict (a structure error, -1).
        return {"findings": {}}, False
    try:
        parse_findings(message, episode_id)
    except StructureError:
        return args, False
    translated = _checked_message(message, episode_id)
    if translated is message:
        return args, True
    try:
        parse_findings(translated, episode_id)
    except StructureError:
        return args, False
    return {"findings": translated}, True


def _fx_id(name: str, args: dict) -> dict:
    """The fx key of the contract is date:source:target; the environment writes it date|source|target.
    A get_record on fx accepts both."""
    table, record_id = args.get("table"), args.get("id")
    if name == "get_record" and table == "fx" and isinstance(record_id, str) and "|" not in record_id \
            and record_id.count(":") == 2:
        return dict(args, id=record_id.replace(":", "|"))
    return args


def make_env(journal_path=None, server_command=None, secret_file=None) -> AdapterEnv:
    """The environment object. `journal_path`: the evaluator journal of closed runs (default
    $AUDITGYM_RESULTS or runs/results.jsonl, as for `python -m auditgym serve --mode eval`; when that
    is not writable, auditgym/results.jsonl under the system temporary directory; see `AdapterEnv._journal`);
    `server_command`: argv of an evaluator started elsewhere (README «Закрытая проверка»), which then
    owns its journal (and its secret). `secret_file` (else $AUDITGYM_ADAPTER_SECRET_FILE; the generic
    $AUDITGYM_SECRET_FILE is still stripped): the evaluator secret file given to the `eval` evaluator
    only, so that journal rows are HMAC-signed and `verify-journal --secret-file` exits 0. Without it
    official rows are unsigned and verify-journal exits 1 by design (docs/ADAPTER.md)."""
    return AdapterEnv(journal_path=journal_path, server_command=server_command, secret_file=secret_file)


def main(argv: list[str]) -> int:
    if argv[:1] == ["--demo"]:
        from tools.adapter_run import main as demo_main

        return demo_main(argv[1:])
    print(__doc__.strip().split("\n\n")[0] + "\n\nusage: python judge_adapter.py --demo [--seeds N]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
