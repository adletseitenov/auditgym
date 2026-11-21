"""Evaluator process: `python -m auditgym serve [--mode train|eval] [--results PATH]`.

One JSON object per line on stdin, one per line on stdout (docs/CONTRACTS.md, section 5.1):

    -> {"op": "load", "directory": "..."}     only as the very first request (Env.from_files)
    -> {"op": "reset", "seed": 7, "config": {}}   or {"op": "reset"}: restart a loaded episode,
                                                  else the episode of the next default seed
                                                  (0, then the previous seed + 1)
                                                  ("options" is accepted as a synonym of "config")
    <- {"observation": {...}, "info": {}}
    -> {"op": "step", "action": {...}}
    <- {"observation": {...}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}
    -> {"op": "close"}                         (end of input also stops the server)

When the input ends, a close arrives or the output breaks, a running episode that had a step or
whose 60 s are over is truncated and journaled (Evaluator.end); the 60 s are also checked at
reset and load, not only at the next step.

Protocol problems (a line that is not JSON, not an object, an unknown op, a load that is not
the first request) never stop the server and never spend a step; the reply has the shape of a
step reply with an error observation. Lines that cannot be read are the exception while an
episode is running (audit L4): a line longer than MAX_LINE_BYTES (read whole), and a line that
does not decode but says `"op": "step"` (nested deeper than the parser goes, cut short), count
as a step with a `bad_arguments` error, or, if they name a submit action (an action/tool/name/type
key with the value "submit", also inside an action sent as a JSON string), as an invalid
structure that ends the episode with reward -1. A long line that names another op (a reset with
a huge config) is a `bad_request` and spends nothing. With no running episode, or after its end,
it is a `bad_request` like any other bad line; after the end of the episode it gets the same
`episode_over` reply as a step:

    <- {"observation": {"type": "error", "error": {"code": "bad_request", "message": "Bad request."},
        "steps_remaining": N}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}

`load` is allowed by default only in train mode. In `--mode eval` it is refused (`bad_request`) unless
`--allow-load` is given: a policy that writes first to the evaluator's stdin could otherwise load a
forged directory (a hidden episode_id, empty truth) and get a signed row with reward 1.0. Rows of loaded
episodes carry `"source": "load"` and `source_sha256` in the journal, and `verify-journal` fails on them
in eval. `--no-load` refuses `load` in every mode and wins over `--allow-load` (Env() without files
starts the server this way, so a policy writing to the pipe cannot point the server at a directory). `--results-fd N` (N >= 3)
takes the results file as an inherited, already open descriptor instead of a path, so the path is
not in the server's argv (the process list); Env uses it on POSIX. `--secret-file PATH` reads the
generation secret from a file (auditgym/secret.py; else $AUDITGYM_SECRET_FILE, else $AUDITGYM_SECRET);
an unusable secret file stops the server with exit code 2 before the first request.

A refused reset or load replies the same way with code `bad_config` (bad seed or config),
`bad_request` (a load that is not allowed, is not the first request or has no directory string; a reset in the
middle of an episode is not refused in eval: the abandoned run goes to the journal as `abandoned`,
docs/CONTRACTS.md section 5) or `bad_episode` (directory unreadable). `not_loaded` is reserved: since audit C7 a reset without a seed always
starts an episode.
In eval mode without `--results` and `--results-fd` the journal goes to $AUDITGYM_RESULTS or
`runs/results.jsonl` (directory created, file mode 0600). Python's limit on the digits of an int is lifted in this
process, so a submit with a 5 000-digit amount is judged (a well-formed amount that never
matches) instead of being refused as bad JSON.
stdout carries only protocol lines; diagnostics go to stderr and contain no episode data.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from .actions import default_results_path
from .limits import MAX_LINE_BYTES, json_depth_exceeds
from .evaluator import Evaluator, EvaluatorError

PROTOCOL_ERRORS = {
    "bad_request": "Bad request.",
    "bad_config": "Bad seed or configuration.",
    "not_loaded": "No episode is loaded.",  # reserved: not sent since C7
    "bad_episode": "Episode directory cannot be read.",
    "internal_error": "Internal error.",
}


def _dump(message: dict) -> str:
    return json.dumps(message, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def protocol_error(code: str, steps_remaining: int) -> dict:
    return {"observation": {"type": "error", "error": {"code": code, "message": PROTOCOL_ERRORS[code]},
                            "steps_remaining": steps_remaining},
            "reward": 0.0, "terminated": False, "truncated": False, "info": {}}


def _step_reply(result: tuple) -> dict:
    observation, reward, terminated, truncated, info = result
    return {"observation": observation, "reward": reward, "terminated": terminated,
            "truncated": truncated, "info": info}


class Server:
    def __init__(self, evaluator: Evaluator, allow_load: bool = True) -> None:
        self.evaluator = evaluator
        self.allow_load = allow_load
        self.requests = 0

    def _remaining(self) -> int:
        return self.evaluator.steps_remaining()

    def handle(self, line: bytes) -> dict | None:
        """One request line -> one reply. None means: stop serving."""
        first = self.requests == 0
        self.requests += 1
        if isinstance(line, Unreadable):
            return self._unreadable(line)
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            request = None
        if request is None or json_depth_exceeds(request):
            # Not JSON, or nested deeper than MAX_JSON_DEPTH: a step that cannot be read is still a step
            ops: set = set()
            submit = _scan(_WHITESPACE.sub(b" ", line), ops)
            if ops == {b"step"}:
                return self._unreadable(Unreadable(ops, submit))
            return protocol_error("bad_request", self._remaining())
        if not isinstance(request, dict) or not isinstance(request.get("op"), str):
            return protocol_error("bad_request", self._remaining())
        op = request["op"]
        try:
            if op == "step":
                return _step_reply(self.evaluator.step(request.get("action")))
            if op == "reset":
                seed = request.get("seed")
                config, options = request.get("config"), request.get("options")
                if config is not None and options is not None:
                    return protocol_error("bad_config", self._remaining())  # which one is meant?
                if config is None:
                    config = options  # the Gymnasium name for the same thing
                observation, info = self.evaluator.reset(seed, config)
                return {"observation": observation, "info": info}
            if op == "load":
                directory = request.get("directory")
                if not self.allow_load or not first or not isinstance(directory, str):
                    return protocol_error("bad_request", self._remaining())
                observation, info = self.evaluator.load(directory)
                return {"observation": observation, "info": info}
            if op == "close":
                return None
        except EvaluatorError as exc:
            return protocol_error(exc.code, self._remaining())
        except Exception as exc:  # never crash, never echo the exception text
            print(f"auditgym serve: internal error {type(exc).__name__}", file=sys.stderr)
            return protocol_error("internal_error", self._remaining())
        return protocol_error("bad_request", self._remaining())

    def _unreadable(self, line: "Unreadable") -> dict:
        if line.ops - {b"step"}:  # names a reset, load or close: not a step of the episode
            return protocol_error("bad_request", self._remaining())
        try:
            return _step_reply(self.evaluator.reject(submit=line.submit, oversize=line.oversize))
        except EvaluatorError:  # nothing is running: a plain bad request
            return protocol_error("bad_request", self._remaining())


class Unreadable:
    """A request line that could not be taken as JSON: the "op" values seen in it, and whether it
    names a submit action."""

    def __init__(self, ops: set, submit: bool, oversize: bool = False) -> None:
        self.ops = ops
        self.submit = submit
        self.oversize = oversize  # longer than MAX_LINE_BYTES (the structure error is message_too_large)


_OP = re.compile(rb'"op"\s*:\s*"([A-Za-z_]{1,16})"')
# An action named submit, also inside an action sent as a JSON string (there the quotes are \").
_SUBMIT_HINT = re.compile(rb'\\?"(?:action|tool|name|type)\\?"\s*:\s*\\?"\s*submit\s*\\?"', re.IGNORECASE)
_WHITESPACE = re.compile(rb"\s+")
_HINT_TAIL = 64  # bytes carried between reads; the longest hint with whitespace collapsed is ~30


def _scan(chunk: bytes, ops: set) -> bool:
    """Add the "op" values in `chunk` to `ops`; True when it names a submit action."""
    ops.update(match.group(1) for match in _OP.finditer(chunk))
    return _SUBMIT_HINT.search(chunk) is not None


def _read_line(stream):
    """One line of at most MAX_LINE_BYTES (not counting the newline); a longer line is consumed
    whole, scanned on the way (audit D2), and returned as Unreadable. Each read is scanned with runs
    of whitespace collapsed and a short tail of the previous read carried over, so a hint split by a
    read border or padded with megabytes of spaces is still found and memory stays at about one
    read. None at end of input."""
    line = stream.readline(MAX_LINE_BYTES + 1)
    if not line:
        return None
    if len(line) <= MAX_LINE_BYTES or line.endswith(b"\n"):
        return line
    ops, submit, tail, chunk = set(), False, b"", line
    while True:
        window = _WHITESPACE.sub(b" ", tail + chunk)
        submit = _scan(window, ops) or submit
        if chunk.endswith(b"\n"):
            break
        tail = window[-_HINT_TAIL:]
        chunk = stream.readline(MAX_LINE_BYTES)
        if not chunk:
            break
    return Unreadable(ops, submit, oversize=True)


def serve(evaluator: Evaluator, stdin=None, stdout=None, allow_load: bool = True) -> int:
    stdin = stdin if stdin is not None else sys.stdin.buffer
    stdout = stdout if stdout is not None else sys.stdout
    server = Server(evaluator, allow_load=allow_load)
    try:
        while True:
            line = _read_line(stdin)
            if line is None:
                return 0
            if not isinstance(line, Unreadable) and not line.strip():
                continue
            reply = server.handle(line)
            if reply is None:
                return 0
            try:
                text = _dump(reply)
            except (TypeError, ValueError):
                print("auditgym serve: reply is not valid JSON", file=sys.stderr)
                text = _dump(protocol_error("internal_error", server._remaining()))
            stdout.write(text + "\n")
            stdout.flush()
    finally:
        # End of input, close, or a broken stdout: the running episode leaves its row (audit A2).
        try:
            evaluator.end()
        except Exception as exc:  # noqa: BLE001 - stopping must not fail
            print(f"auditgym serve: internal error at the end {type(exc).__name__}", file=sys.stderr)


def main(argv: list[str]) -> int:
    sys.set_int_max_str_digits(0)  # a huge integer is a well-formed amount, not bad JSON
    parser = argparse.ArgumentParser(prog="python -m auditgym serve")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--results", default=None)
    target.add_argument("--results-fd", type=int, default=None)
    parser.add_argument("--no-load", action="store_true",
                        help="refuse load in any mode (wins over --allow-load)")
    parser.add_argument("--allow-load", action="store_true",
                        help="eval mode: accept load (dev only: the journal marks such rows source=load and "
                             "verify-journal fails on them); train mode allows load without it")
    parser.add_argument("--shaped-reward", action="store_true",
                        help="train mode only: add info['shaped_reward'] (experimental) to the terminal info")
    parser.add_argument("--secret-file", default=None,
                        help="file with the generation secret (wins over $AUDITGYM_SECRET_FILE and $AUDITGYM_SECRET)")
    arguments = parser.parse_args(argv)
    from .secret import SecretError, read_secret

    try:
        secret = read_secret(secret_file=arguments.secret_file)
    except SecretError as error:  # fixed code only: no path, no value
        print(f"auditgym serve: cannot use the secret ({error.code})", file=sys.stderr)
        return 2
    stream = None
    if arguments.mode == "eval" and arguments.results is None and arguments.results_fd is None:
        arguments.results = default_results_path()  # the journal of an eval run always exists (C5)
        try:
            os.makedirs(os.path.dirname(arguments.results) or ".", exist_ok=True)
        except OSError:
            pass  # the open below reports it
    try:
        if arguments.results_fd is not None:
            if arguments.results_fd < 3:  # never a standard stream: stdout is the protocol channel
                raise ValueError("results descriptor")
            stream = open(arguments.results_fd, "a", encoding="utf-8", closefd=True)
        evaluator = Evaluator(mode=arguments.mode, results_path=arguments.results, results_stream=stream,
                              shaped_reward=arguments.shaped_reward, secret=secret)
    except (OSError, ValueError):
        print("auditgym serve: cannot open the results file", file=sys.stderr)
        return 2
    try:
        allow_load = (arguments.mode == "train" or arguments.allow_load) and not arguments.no_load
        return serve(evaluator, allow_load=allow_load)
    finally:
        if stream is not None:
            stream.close()
