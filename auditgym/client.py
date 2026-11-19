"""Env: the policy-side client. It runs the evaluator in a separate process and talks JSON lines.

    with Env(mode="train") as env:
        observation, info = env.reset(seed, config)
        observation, reward, terminated, truncated, info = env.step(action)

    env = Env.from_files(directory)   # then env.reset() without a seed

The client keeps no episode data beyond the last reply: no truth, no provenance, no seed, no
paths (the episode directory and the results path are handed to the server and dropped).

Hardening of the process boundary:
- The results file is opened here and handed to the server as an inherited descriptor
  (`--results-fd`), so its path is not in the server's argv and not in the process list. On a
  platform without descriptor passing (not POSIX) the path goes on the command line instead.
- The server's stderr goes to /dev/null, not to this process: diagnostics never reach a policy.
- A policy holding this object can still reach the pipes (Python has no private attributes), so
  the server gives the pipe no more power than reset/step: `load` is accepted only as the first
  request and only by a server started for Env.from_files; Env() starts it with `--no-load`.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading

from .actions import default_results_path, open_results

_PACKAGE_PARENT = str(pathlib.Path(__file__).resolve().parent.parent)
_REFUSALS = ("bad_request", "bad_config", "not_loaded", "bad_episode", "internal_error")
_NAME_KEYS = ("action", "tool", "name", "type")
DEFAULT_TIMEOUT = 120.0  # seconds to wait for one reply (generating the largest episode takes a few)
_UNSERIALISABLE = "<not serialisable>"
_MAX_LINE_BYTES = 1 << 20  # serve.MAX_LINE_BYTES; not imported, so the client does not load the evaluator


class EnvError(RuntimeError):
    """The evaluator refused a request or stopped; the only exception Env raises for bad input
    or a broken process. `code` is a fixed short identifier (`bad_config`, `bad_episode`,
    `bad_request`, `server_exited`, `closed`, `timeout`, `bad_reply`, ...)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _is_submit(action: object) -> bool:
    try:
        action = json.loads(action) if isinstance(action, str) else action
        if not isinstance(action, dict):
            return False
        return any(isinstance(action.get(key), str) and action[key].strip().lower() == "submit"
                   for key in _NAME_KEYS)
    except Exception:  # noqa: BLE001 - a hostile object may raise anything; it is not a submit then
        return False


class Env:
    """Client of the evaluator process.

    `Env(mode="train" | "eval", results_path=None, timeout=120.0)`. In eval mode without a
    `results_path` the journal is $AUDITGYM_RESULTS or `runs/results.jsonl` in the current directory (audit C5); it is
    created with mode 0600. `timeout` bounds the wait for one reply (None: no bound); on expiry
    the server is stopped and `EnvError("timeout")` is raised.

    `reset(seed=None, config=None, options=None)`: `options` is a synonym of `config`; without a
    seed the episode of Env.from_files is restarted, or, if there is none, a new one is generated
    from the next default seed: 0 on the first such call, then the previous seed plus one. A refused reset raises `EnvError` and leaves the running episode.
    A `submit` that cannot be sent as JSON (NaN, a reference cycle, a 5 000-digit integer, nesting
    deeper than the recursion limit) is sent as a submit with an invalid structure: reward -1, the
    episode ends. Any other action that cannot be sent is an unknown action (a step is spent).
    """

    def __init__(self, mode: str = "train", results_path=None, *, timeout: float | None = DEFAULT_TIMEOUT,
                 secret_file=None, _files: bool = False) -> None:
        if mode not in ("train", "eval"):
            raise EnvError("bad_mode")
        self.mode = mode
        self._timeout = timeout
        if mode == "eval" and results_path is None:
            results_path = default_results_path()
        command = [sys.executable, "-m", "auditgym", "serve", "--mode", mode]
        if not _files:
            command.append("--no-load")
        if secret_file is not None:  # the evaluator secret of a closed run (audit L1)
            try:
                secret_path = pathlib.Path(secret_file).resolve()
                if not secret_path.is_file():
                    raise OSError("not a file")
            except (TypeError, ValueError, OSError):
                raise EnvError("bad_secret") from None
            command += ["--secret-file", os.fspath(secret_path)]
            del secret_path
        pass_fds = ()
        if results_path is not None:
            try:
                # abspath, not resolve(): a symlink planted at the journal path must be refused
                path = os.path.abspath(os.fspath(results_path))
                pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
                if os.name == "posix":
                    descriptor = open_results(path)  # regular file, 0600, never descriptor 0-2
                    pass_fds = (descriptor,)
                    command += ["--results-fd", str(descriptor)]
                else:
                    command += ["--results", path]
            except (TypeError, ValueError, OSError):
                raise EnvError("bad_results_path") from None
            del path
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [_PACKAGE_PARENT] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []))
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL, env=environment, cwd=_PACKAGE_PARENT,
                                             pass_fds=pass_fds)
        except OSError:
            raise EnvError("server_exited") from None
        finally:
            for descriptor in pass_fds:
                os.close(descriptor)  # the server holds its own copy
        # Popen keeps its argv; drop it so nothing about the server's arguments stays here.
        self._process.args = ["auditgym-serve"]
        del command, environment, results_path, secret_file

    @classmethod
    def from_files(cls, directory, mode: str = "train", results_path=None, *,
                   timeout: float | None = DEFAULT_TIMEOUT, secret_file=None) -> "Env":
        env = cls(mode=mode, results_path=results_path, timeout=timeout, secret_file=secret_file, _files=True)
        try:
            try:
                path = os.fspath(pathlib.Path(directory).resolve())
            except (TypeError, ValueError, OSError):
                raise EnvError("bad_episode") from None
            env._request({"op": "load", "directory": path})
        except BaseException:
            env.close()
            raise
        return env

    # --- protocol ----------------------------------------------------------------------------
    @staticmethod
    def _encode(message: dict) -> str | None:
        try:
            return json.dumps(message, ensure_ascii=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError, OverflowError, MemoryError):
            return None

    def _request(self, message: dict) -> dict:
        process = self._process
        if process is None or process.poll() is not None:
            raise EnvError("closed")
        line = self._encode(message)
        if line is not None and message.get("op") != "step" and len(line) >= _MAX_LINE_BYTES:
            raise EnvError("bad_config")  # the server would not read it as a reset at all
        if line is None:
            if message.get("op") != "step":
                raise EnvError("bad_config")  # a seed or config that is not JSON
            if _is_submit(message.get("action")):
                action = {"action": "submit", "findings": _UNSERIALISABLE}  # invalid structure, ends it
            else:
                action = None  # not JSON: an unknown action, a step is spent
            line = json.dumps({"op": "step", "action": action})
        expired = threading.Event()
        timer = None
        if self._timeout is not None:
            def expire() -> None:
                expired.set()
                try:
                    process.kill()
                except OSError:
                    pass

            timer = threading.Timer(self._timeout, expire)
            timer.daemon = True
            timer.start()
        try:
            try:
                process.stdin.write(line.encode("ascii") + b"\n")
                process.stdin.flush()
                reply = process.stdout.readline()
            except (BrokenPipeError, OSError, ValueError):
                reply = b""
        finally:
            if timer is not None:
                timer.cancel()
        if expired.is_set():
            self.close()
            raise EnvError("timeout")
        if not reply:
            raise EnvError("server_exited")
        try:
            reply = json.loads(reply)
        except (ValueError, RecursionError):
            raise EnvError("bad_reply") from None
        observation = reply.get("observation") if isinstance(reply, dict) else None
        if not isinstance(observation, dict):
            raise EnvError("bad_reply")
        if message.get("op") in ("reset", "load"):
            if observation.get("type") == "error" and isinstance(observation.get("error"), dict):
                raise EnvError(str(observation["error"].get("code")))
            if observation.get("type") != "reset":  # never hand a step reply back as a reset
                raise EnvError("bad_reply")
        return reply

    def reset(self, seed=None, config=None, options=None):
        message = {"op": "reset"}
        if config is not None and options is not None:
            raise EnvError("bad_config")  # two configurations: which one is meant?
        if config is None:
            config = options
        if seed is not None:
            message["seed"] = seed
        if config is not None or seed is not None:
            message["config"] = config
        reply = self._request(message)
        try:
            return reply["observation"], reply["info"]
        except KeyError:
            raise EnvError("bad_reply") from None

    def step(self, action):
        reply = self._request({"op": "step", "action": action})
        try:
            return (reply["observation"], reply["reward"], reply["terminated"], reply["truncated"],
                    reply["info"])
        except KeyError:
            raise EnvError("bad_reply") from None

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
