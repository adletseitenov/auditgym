"""Evaluator process: `python -m auditgym serve [--mode train|eval] [--results PATH | --results-fd N |
--no-results] [--no-load] [--secret-file PATH]`.

One JSON object per line on stdin, one per line on stdout (docs/CONTRACTS.md, section 5.1):

    -> {"op": "load", "directory": "..."}     only as the very first request (Env.from_files)
    -> {"op": "reset", "seed": 7, "config": {}}   or {"op": "reset"} (see Evaluator.reset);
                                               "options" is accepted as a synonym of "config"
    <- {"observation": {...}, "info": {}}
    -> {"op": "step", "action": {...}}
    <- {"observation": {...}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}
    -> {"op": "close"}                         (end of input also stops the server)

Protocol problems (a line that is not JSON, not an object, an unknown op, a load that is not
the first request) never stop the server and never spend a step; the reply has the shape of a
step reply with an error observation:

    <- {"observation": {"type": "error", "error": {"code": "bad_request", "message": "Bad request."},
        "steps_remaining": N}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}

A line longer than MAX_LINE_BYTES is consumed unread. Outside a running episode it is a
bad_request; during one it is a step: an error step, or, when the line mentions "submit", a
submit with an invalid structure that ends the episode with reward -1 (AUDIT L4). Integer
literals of any length are accepted (AUDIT R1): one longer than the language's conversion limit
becomes an integer with the same number of digits, which no seed, limit or amount can use.

The judge log (one JSON line per finished episode, with the structure error of an invalid
submit) goes to `--results PATH`, to the inherited descriptor `--results-fd N` (Env uses it on
POSIX, so the path is not in the process list), or by default to $AUDITGYM_RESULTS or
runs/results.jsonl under the working directory, created owner-only; a default location that
cannot be written is skipped. `--no-results` keeps no log.

`--no-load` refuses `load` altogether (Env() without files starts the server this way, so a
policy writing to the pipe cannot point the server at a directory). `--secret-file PATH` (or
$AUDITGYM_SECRET) sets the evaluator secret mixed into generated seeds (see evaluator.py).

A refused reset or load replies with code `bad_config` (bad seed or config) or `bad_episode`
(directory unreadable); `not_loaded` is kept in the table for older clients but no longer sent.
stdout carries only protocol lines; diagnostics go to stderr and contain no episode data.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from .evaluator import Evaluator, EvaluatorError, open_results

MAX_LINE_BYTES = 1 << 20
RESULTS_ENV = "AUDITGYM_RESULTS"
SECRET_ENV = "AUDITGYM_SECRET"
DEFAULT_RESULTS = os.path.join("runs", "results.jsonl")
PROTOCOL_ERRORS = {
    "bad_request": "Bad request.",
    "bad_config": "Bad seed or configuration.",
    "not_loaded": "No episode is loaded.",
    "bad_episode": "Episode directory cannot be read.",
    "internal_error": "Internal error.",
}
# Below the 4 300-digit limit of int(str), with room for a sign.
_EXACT_INT_DIGITS = 4000
_SUBMIT = re.compile(rb"submit", re.IGNORECASE)
_SCAN_OVERLAP = len(b"submit") - 1


def _dump(message: dict) -> str:
    return json.dumps(message, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_int(text: str) -> int:
    digits = len(text) - text.startswith("-")
    if digits <= _EXACT_INT_DIGITS:
        return int(text)
    value = 10 ** (digits - 1)
    return -value if text.startswith("-") else value


def parse_request(line: bytes):
    return json.loads(line.decode("utf-8"), parse_int=_parse_int)


def protocol_error(code: str, steps_remaining: int) -> dict:
    return {"observation": {"type": "error", "error": {"code": code, "message": PROTOCOL_ERRORS[code]},
                            "steps_remaining": steps_remaining},
            "reward": 0.0, "terminated": False, "truncated": False, "info": {}}


def _step_reply(result: tuple) -> dict:
    observation, reward, terminated, truncated, info = result
    return {"observation": observation, "reward": reward, "terminated": terminated, "truncated": truncated,
            "info": info}


class Oversized:
    """A line longer than MAX_LINE_BYTES, consumed unread. `submit`: it mentions "submit"."""

    def __init__(self, submit: bool) -> None:
        self.submit = submit


class Server:
    def __init__(self, evaluator: Evaluator, allow_load: bool = True) -> None:
        self.evaluator = evaluator
        self.allow_load = allow_load
        self.requests = 0

    def _remaining(self) -> int:
        return self.evaluator.steps_remaining()

    def handle(self, line) -> dict | None:
        """One request line (bytes, or Oversized) -> one reply. None means: stop serving."""
        first = self.requests == 0
        self.requests += 1
        if isinstance(line, Oversized):
            if not self.evaluator.running():  # nothing to spend a step in
                return protocol_error("bad_request", self._remaining())
            return _step_reply(self.evaluator.step_unreadable(line.submit))
        try:
            request = parse_request(line)
        except (UnicodeDecodeError, ValueError, RecursionError):
            return protocol_error("bad_request", self._remaining())
        if not isinstance(request, dict) or not isinstance(request.get("op"), str):
            return protocol_error("bad_request", self._remaining())
        op = request["op"]
        try:
            if op == "step":
                return _step_reply(self.evaluator.step(request.get("action")))
            if op == "reset":
                config = request.get("config")
                if config is None:
                    config = request.get("options")
                observation, info = self.evaluator.reset(request.get("seed"), config)
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


def _read_line(stream):
    """One line of at most MAX_LINE_BYTES; a longer line is consumed whole and returned as
    Oversized (scanned for "submit" on the way). None at end of input."""
    line = stream.readline(MAX_LINE_BYTES + 1)
    if not line:
        return None
    if len(line) <= MAX_LINE_BYTES or line.endswith(b"\n"):
        return line
    submit = _SUBMIT.search(line) is not None
    tail = line[-_SCAN_OVERLAP:]
    while True:
        rest = stream.readline(MAX_LINE_BYTES)
        if not submit:  # the overlap catches a "submit" split between two reads
            submit = _SUBMIT.search(tail + rest) is not None
            tail = (tail + rest)[-_SCAN_OVERLAP:]
        if not rest or rest.endswith(b"\n"):
            break
    return Oversized(submit)


def serve(evaluator: Evaluator, stdin=None, stdout=None, allow_load: bool = True) -> int:
    stdin = stdin if stdin is not None else sys.stdin.buffer
    stdout = stdout if stdout is not None else sys.stdout
    server = Server(evaluator, allow_load=allow_load)
    while True:
        line = _read_line(stdin)
        if line is None:
            return 0
        if not isinstance(line, Oversized) and not line.strip():
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


def _default_results():
    """The judge log at the default location, or None when it cannot be opened there."""
    try:
        descriptor = open_results(os.environ.get(RESULTS_ENV) or DEFAULT_RESULTS)
    except (OSError, ValueError):
        print("auditgym serve: no judge log (default location not writable)", file=sys.stderr)
        return None
    return open(descriptor, "a", encoding="utf-8", closefd=True)


def _secret(path) -> bytes:
    if path is not None:
        with open(path, "rb") as handle:
            return handle.read().strip()
    return os.environ.get(SECRET_ENV, "").encode("utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m auditgym serve")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--results", default=None)
    target.add_argument("--results-fd", type=int, default=None)
    target.add_argument("--no-results", action="store_true")
    parser.add_argument("--no-load", action="store_true")
    parser.add_argument("--secret-file", default=None)
    arguments = parser.parse_args(argv)
    stream = None
    try:
        if arguments.results_fd is not None:
            stream = open(arguments.results_fd, "a", encoding="utf-8", closefd=True)
        elif arguments.results is None and not arguments.no_results:
            stream = _default_results()
        secret = _secret(arguments.secret_file)
        evaluator = Evaluator(mode=arguments.mode, results_path=arguments.results, results_stream=stream,
                              secret=secret)
    except (OSError, ValueError):
        print("auditgym serve: cannot open the results or secret file", file=sys.stderr)
        return 2
    try:
        return serve(evaluator, allow_load=not arguments.no_load)
    finally:
        if stream is not None:
            stream.close()
