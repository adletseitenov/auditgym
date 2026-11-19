"""Evaluator process: `python -m auditgym serve [--mode train|eval] [--results PATH]`.

One JSON object per line on stdin, one per line on stdout (docs/CONTRACTS.md, section 5.1):

    -> {"op": "load", "directory": "..."}     only as the very first request (Env.from_files)
    -> {"op": "reset", "seed": 7, "config": {}}   or {"op": "reset"}: restart a loaded episode,
                                                  else the episode of the default seed
                                                  ("options" is accepted as a synonym of "config")
    <- {"observation": {...}, "info": {}}
    -> {"op": "step", "action": {...}}
    <- {"observation": {...}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}
    -> {"op": "close"}                         (end of input also stops the server)

Protocol problems (a line that is not JSON, not an object, an unknown op, a load that is not
the first request) never stop the server and never spend a step; the reply has the shape of a
step reply with an error observation. A line longer than MAX_LINE_BYTES is the exception while
an episode is running (audit L4): it is read whole and counts as a step with a `bad_arguments`
error, or, if it looks like a submit (an action/tool/name/type key with the value "submit"),
as an invalid structure that ends the episode with reward -1. With no running episode it is a
`bad_request` like any other bad line:

    <- {"observation": {"type": "error", "error": {"code": "bad_request", "message": "Bad request."},
        "steps_remaining": N}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}

`--no-load` refuses `load` altogether (Env() without files starts the server this way, so a
policy writing to the pipe cannot point the server at a directory). `--results-fd N` takes the
results file as an inherited, already open descriptor instead of a path, so the path is not in
the server's argv (the process list); Env uses it on POSIX.

A refused reset or load replies the same way with code `bad_config` (bad seed or config),
`not_loaded` (reset without seed and nothing loaded) or `bad_episode` (directory unreadable).
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
from .evaluator import Evaluator, EvaluatorError

MAX_LINE_BYTES = 1 << 20
PROTOCOL_ERRORS = {
    "bad_request": "Bad request.",
    "bad_config": "Bad seed or configuration.",
    "not_loaded": "No episode is loaded.",
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
        if line in (_TOO_LONG, _TOO_LONG_SUBMIT):
            try:
                return _step_reply(self.evaluator.reject(submit=line == _TOO_LONG_SUBMIT))
            except EvaluatorError:  # nothing is running: a plain bad request
                return protocol_error("bad_request", self._remaining())
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return protocol_error("bad_request", self._remaining())
        if not isinstance(request, dict) or not isinstance(request.get("op"), str):
            return protocol_error("bad_request", self._remaining())
        op = request["op"]
        try:
            if op == "step":
                return _step_reply(self.evaluator.step(request.get("action")))
            if op == "reset":
                seed = request.get("seed")
                config = request.get("config")
                if config is None:
                    config = request.get("options")  # the Gymnasium name for the same thing
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


_TOO_LONG = b"\x00too long"  # not JSON; marks a line over the limit
_TOO_LONG_SUBMIT = b"\x00too long submit"
_SUBMIT_HINT = re.compile(rb'"(?:action|tool|name|type)"\s*:\s*"\s*submit\s*"', re.IGNORECASE)


def _read_line(stream) -> bytes | None:
    """One line of at most MAX_LINE_BYTES (not counting the newline); a longer line is consumed
    whole and replaced by _TOO_LONG, or by _TOO_LONG_SUBMIT when the part kept in memory contains
    an action named submit. None at end of input."""
    line = stream.readline(MAX_LINE_BYTES + 1)
    if not line:
        return None
    if len(line) > MAX_LINE_BYTES and not line.endswith(b"\n"):
        submit = _SUBMIT_HINT.search(line) is not None
        while True:
            rest = stream.readline(MAX_LINE_BYTES)
            if not rest or rest.endswith(b"\n"):
                break
        return _TOO_LONG_SUBMIT if submit else _TOO_LONG
    return line


def serve(evaluator: Evaluator, stdin=None, stdout=None, allow_load: bool = True) -> int:
    stdin = stdin if stdin is not None else sys.stdin.buffer
    stdout = stdout if stdout is not None else sys.stdout
    server = Server(evaluator, allow_load=allow_load)
    while True:
        line = _read_line(stdin)
        if line is None:
            return 0
        if not line.strip():
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


def main(argv: list[str]) -> int:
    sys.set_int_max_str_digits(0)  # a huge integer is a well-formed amount, not bad JSON
    parser = argparse.ArgumentParser(prog="python -m auditgym serve")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--results", default=None)
    target.add_argument("--results-fd", type=int, default=None)
    parser.add_argument("--no-load", action="store_true")
    arguments = parser.parse_args(argv)
    stream = None
    if arguments.mode == "eval" and arguments.results is None and arguments.results_fd is None:
        arguments.results = default_results_path()  # the journal of an eval run always exists (C5)
        try:
            os.makedirs(os.path.dirname(arguments.results) or ".", exist_ok=True)
        except OSError:
            pass  # the open below reports it
    try:
        if arguments.results_fd is not None:
            stream = open(arguments.results_fd, "a", encoding="utf-8", closefd=True)
        evaluator = Evaluator(mode=arguments.mode, results_path=arguments.results, results_stream=stream)
    except (OSError, ValueError):
        print("auditgym serve: cannot open the results file", file=sys.stderr)
        return 2
    try:
        return serve(evaluator, allow_load=not arguments.no_load)
    finally:
        if stream is not None:
            stream.close()
