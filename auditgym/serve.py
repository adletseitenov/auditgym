"""Evaluator process: `python -m auditgym serve [--mode train|eval] [--results PATH]`.

One JSON object per line on stdin, one per line on stdout (docs/CONTRACTS.md, section 5.1):

    -> {"op": "load", "directory": "..."}     only as the very first request (Env.from_files)
    -> {"op": "reset", "seed": 7, "config": {}}   or {"op": "reset"} to restart a loaded episode
    <- {"observation": {...}, "info": {}}
    -> {"op": "step", "action": {...}}
    <- {"observation": {...}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}
    -> {"op": "close"}                         (end of input also stops the server)

Protocol problems (a line that is not JSON, not an object, too long, an unknown op, a load
that is not the first request) never stop the server and never spend a step; the reply has
the shape of a step reply with an error observation:

    <- {"observation": {"type": "error", "error": {"code": "bad_request", "message": "Bad request."},
        "steps_remaining": N}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}

A refused reset or load replies the same way with code `bad_config` (bad seed or config),
`not_loaded` (reset without seed and nothing loaded) or `bad_episode` (directory unreadable).
stdout carries only protocol lines; diagnostics go to stderr and contain no episode data.
"""
from __future__ import annotations

import argparse
import json
import sys

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


class Server:
    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self.requests = 0

    def _remaining(self) -> int:
        return self.evaluator.steps_remaining()

    def handle(self, line: bytes) -> dict | None:
        """One request line -> one reply. None means: stop serving."""
        first = self.requests == 0
        self.requests += 1
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return protocol_error("bad_request", self._remaining())
        if not isinstance(request, dict) or not isinstance(request.get("op"), str):
            return protocol_error("bad_request", self._remaining())
        op = request["op"]
        try:
            if op == "step":
                observation, reward, terminated, truncated, info = self.evaluator.step(request.get("action"))
                return {"observation": observation, "reward": reward, "terminated": terminated,
                        "truncated": truncated, "info": info}
            if op == "reset":
                seed = request.get("seed")
                observation, info = self.evaluator.reset(seed, request.get("config"))
                return {"observation": observation, "info": info}
            if op == "load":
                directory = request.get("directory")
                if not first or not isinstance(directory, str):
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


_TOO_LONG = b"\x00too long"  # not JSON, so it gets the bad_request reply


def _read_line(stream) -> bytes | None:
    """One line of at most MAX_LINE_BYTES; a longer line is consumed whole and replaced by a
    line that fails to parse. None at end of input."""
    line = stream.readline(MAX_LINE_BYTES + 1)
    if not line:
        return None
    if len(line) > MAX_LINE_BYTES and not line.endswith(b"\n"):
        while True:
            rest = stream.readline(MAX_LINE_BYTES)
            if not rest or rest.endswith(b"\n"):
                break
        return _TOO_LONG
    return line


def serve(evaluator: Evaluator, stdin=None, stdout=None) -> int:
    stdin = stdin if stdin is not None else sys.stdin.buffer
    stdout = stdout if stdout is not None else sys.stdout
    server = Server(evaluator)
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
    parser = argparse.ArgumentParser(prog="python -m auditgym serve")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--results", default=None)
    arguments = parser.parse_args(argv)
    try:
        evaluator = Evaluator(mode=arguments.mode, results_path=arguments.results)
    except OSError:
        print("auditgym serve: cannot open the results file", file=sys.stderr)
        return 2
    return serve(evaluator)
