"""Env: the policy-side client. It runs the evaluator in a separate process and talks JSON lines.

    with Env(mode="train") as env:
        observation, info = env.reset(seed, config)
        observation, reward, terminated, truncated, info = env.step(action)

    env = Env.from_files(directory)   # then env.reset() without a seed

The client keeps no episode data beyond the last reply: no truth, no provenance, no seed, no
paths (the episode directory and the results path are handed to the server and dropped).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

_PACKAGE_PARENT = str(pathlib.Path(__file__).resolve().parent.parent)
_REFUSALS = ("bad_request", "bad_config", "not_loaded", "bad_episode", "internal_error")


class EnvError(RuntimeError):
    """The evaluator refused a request or stopped. `code` is a fixed short identifier."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Env:
    def __init__(self, mode: str = "train", results_path=None) -> None:
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        command = [sys.executable, "-m", "auditgym", "serve", "--mode", mode]
        if results_path is not None:
            command += ["--results", os.fspath(pathlib.Path(results_path).resolve())]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [_PACKAGE_PARENT] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []))
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         env=environment, cwd=_PACKAGE_PARENT)
        # Popen keeps its argv; drop it so the results path is not reachable from the client.
        self._process.args = ["auditgym-serve"]
        del command, environment, results_path

    @classmethod
    def from_files(cls, directory, mode: str = "train", results_path=None) -> "Env":
        env = cls(mode=mode, results_path=results_path)
        try:
            env._request({"op": "load", "directory": os.fspath(pathlib.Path(directory).resolve())})
        except BaseException:
            env.close()
            raise
        return env

    # --- protocol ----------------------------------------------------------------------------
    def _request(self, message: dict) -> dict:
        process = self._process
        if process is None or process.poll() is not None:
            raise EnvError("closed")
        try:
            line = json.dumps(message, ensure_ascii=True, allow_nan=False)
        except (TypeError, ValueError):
            if message.get("op") != "step":
                raise
            line = json.dumps({"op": "step", "action": None})  # not JSON: an unknown action
        try:
            process.stdin.write(line.encode("ascii") + b"\n")
            process.stdin.flush()
            reply = process.stdout.readline()
        except (BrokenPipeError, OSError):
            raise EnvError("server_exited") from None
        if not reply:
            raise EnvError("server_exited")
        reply = json.loads(reply)
        observation = reply.get("observation")
        if (message.get("op") in ("reset", "load") and isinstance(observation, dict)
                and observation.get("type") == "error"):
            raise EnvError(observation["error"]["code"])
        return reply

    def reset(self, seed=None, config=None):
        message = {"op": "reset"}
        if seed is not None:
            message["seed"] = seed
            message["config"] = config
        reply = self._request(message)
        return reply["observation"], reply["info"]

    def step(self, action):
        reply = self._request({"op": "step", "action": action})
        return (reply["observation"], reply["reward"], reply["terminated"], reply["truncated"],
                reply["info"])

    # --- lifetime ----------------------------------------------------------------------------
    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.stdin.write(b'{"op":"close"}\n')
                process.stdin.flush()
        except OSError:
            pass
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def __enter__(self) -> "Env":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
