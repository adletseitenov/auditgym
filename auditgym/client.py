"""Env: the policy-side client. It runs the evaluator in a separate process and talks JSON lines.

    with Env(mode="train") as env:
        observation, info = env.reset(seed, config)        # also reset(seed=7, options=config)
        observation, reward, terminated, truncated, info = env.step(action)

    env = Env.from_files(directory)   # or Env(directory); then env.reset() without a seed
    Env().reset()                     # no seed, no files: seed 0, then the previous seed + 1

Every failure the client reports is an EnvError with a short fixed `code` (`bad_config`,
`bad_episode`, `bad_mode`, `bad_timeout`, `bad_results`, `timeout`, `server_exited`, `bad_reply`,
`closed`): hostile values passed to reset or step never raise anything else (AUDIT C8).
A step whose action cannot be sent as JSON (a cycle, an object, an integer too long to print)
is sent as the same action with unreadable arguments: an error step, or, for a submit, an
invalid structure that ends the episode with reward -1 (AUDIT C9, R1). NaN and Infinity are
sent as such; the judge rejects them inside findings.

Each request waits at most `timeout` seconds (default 120) for the server's reply; a server that
does not answer in time is stopped and the call raises EnvError("timeout").

The judge log of every finished episode goes to `results_path`, else to $AUDITGYM_RESULTS, else
to runs/results.jsonl under the working directory (created owner-only; a default location that
cannot be written is skipped, an explicit one raises EnvError("bad_results")).

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
import math
import operator
import os
import pathlib
import subprocess
import sys

from .actions import action_name

_PACKAGE_PARENT = str(pathlib.Path(__file__).resolve().parent.parent)
MODES = ("train", "eval")
DEFAULT_TIMEOUT = 120.0
RESULTS_ENV = "AUDITGYM_RESULTS"
DEFAULT_RESULTS = os.path.join("runs", "results.jsonl")
_UNREADABLE = "\u0000"  # not JSON: decoding it fails on the server side


class EnvError(RuntimeError):
    """The evaluator refused a request or stopped. `code` is a fixed short identifier."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _dump(message: dict) -> str:
    return json.dumps(message, ensure_ascii=True, allow_nan=True)


def _is_directory(value) -> bool:
    try:
        return isinstance(value, (str, os.PathLike)) and pathlib.Path(value).is_dir()
    except (OSError, ValueError, TypeError):
        return False


def _step_line(action) -> str:
    """The step request for `action`; one that is not JSON-serializable becomes the same action
    with unreadable arguments (see the module docstring)."""
    try:
        return _dump({"op": "step", "action": action})
    except Exception:  # TypeError, ValueError, RecursionError, OverflowError, or a hostile object
        pass
    name = action_name(action)
    if name == "submit":
        substitute = {"action": "submit", "findings": _UNREADABLE}
    elif name is not None:
        substitute = {"action": name, "args": _UNREADABLE}
    else:
        substitute = None
    return _dump({"op": "step", "action": substitute})


def open_results(path) -> int:
    """Append-only descriptor of the judge log, created owner-only, its folder made when missing.
    (Same as evaluator.open_results; kept here so the client does not import the evaluator.)"""
    path = pathlib.Path(path)
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)


def _results_descriptor(results_path):
    """(descriptor or None, path for non-POSIX or None). Explicit paths must work; the default is
    best effort."""
    explicit = results_path is not None
    path = results_path if explicit else (os.environ.get(RESULTS_ENV) or DEFAULT_RESULTS)
    try:
        path = os.fspath(pathlib.Path(path).resolve())
        if os.name != "posix":
            os.close(open_results(path))
            return None, path
        return open_results(path), None
    except (OSError, ValueError, TypeError):
        if explicit:
            raise EnvError("bad_results") from None
        return None, None


class Env:
    def __init__(self, mode="train", results_path=None, *, directory=None, timeout=DEFAULT_TIMEOUT) -> None:
        """`mode` is "train" or "eval"; a directory path in its place loads that episode (train)."""
        self._process = None
        self._pending = b""
        if mode not in MODES:
            if directory is None and _is_directory(mode):
                directory, mode = mode, "train"
            else:
                raise EnvError("bad_mode")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not timeout > 0):
            raise EnvError("bad_timeout")
        if timeout is not None and math.isinf(timeout):
            timeout = None
        self.mode = mode
        self._timeout = timeout
        command = [sys.executable, "-m", "auditgym", "serve", "--mode", mode]
        if directory is None:
            command.append("--no-load")
        descriptor, path = _results_descriptor(results_path)
        pass_fds = ()
        if descriptor is not None:
            pass_fds = (descriptor,)
            command += ["--results-fd", str(descriptor)]
        elif path is not None:
            command += ["--results", path]
        else:
            command.append("--no-results")
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
            for item in pass_fds:
                os.close(item)  # the server holds its own copy
        # Popen keeps its argv; drop it so nothing about the server's arguments stays here.
        self._process.args = ["auditgym-serve"]
        del command, environment, results_path, path
        if directory is not None:
            try:
                location = os.fspath(pathlib.Path(directory).resolve())
            except (OSError, ValueError, TypeError):
                self.close()
                raise EnvError("bad_episode") from None
            try:
                self._request(_dump({"op": "load", "directory": location}), refusals=True)
            except BaseException:
                self.close()
                raise
            del location
        del directory

    @classmethod
    def from_files(cls, directory, mode: str = "train", results_path=None, *, timeout=DEFAULT_TIMEOUT) -> "Env":
        return cls(mode=mode, results_path=results_path, directory=directory, timeout=timeout)

    # --- protocol ----------------------------------------------------------------------------
    def _read_reply(self) -> bytes:
        """One reply line, waiting at most the timeout (POSIX); EnvError on timeout or exit."""
        stream = self._process.stdout
        if self._timeout is None or os.name != "posix":
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise EnvError("server_exited")
            return line
        import select
        import time

        descriptor = stream.fileno()
        buffer = bytearray(self._pending)
        deadline = time.monotonic() + self._timeout  # determinism: a pipe timeout, never reaches data
        while b"\n" not in buffer:
            left = deadline - time.monotonic()  # determinism: a pipe timeout, never reaches data
            if left <= 0 or not select.select([descriptor], [], [], left)[0]:
                self._kill()
                raise EnvError("timeout")
            chunk = os.read(descriptor, 1 << 16)
            if not chunk:
                raise EnvError("server_exited")
            buffer += chunk
        line, _, rest = bytes(buffer).partition(b"\n")
        self._pending = rest
        return line + b"\n"

    def _request(self, line: str, refusals: bool = False) -> dict:
        """Send one request line and return the reply. With `refusals`, an error observation
        (a refused reset or load) raises EnvError with its code."""
        process = self._process
        if process is None or process.poll() is not None:
            raise EnvError("closed")
        try:
            process.stdin.write(line.encode("ascii") + b"\n")
            process.stdin.flush()
            raw = self._read_reply()
        except OSError:
            raise EnvError("server_exited") from None
        try:
            reply = json.loads(raw)
        except (ValueError, RecursionError):
            raise EnvError("bad_reply") from None
        if not isinstance(reply, dict) or not isinstance(reply.get("observation"), dict):
            raise EnvError("bad_reply")
        observation = reply["observation"]
        if refusals and observation.get("type") == "error":
            code = (observation.get("error") or {}).get("code")
            raise EnvError(code if isinstance(code, str) else "bad_reply")
        return reply

    def reset(self, seed=None, config=None, options=None):
        """reset(seed, config); `options` is a synonym of `config` (Gymnasium's name). Without a
        seed: the loaded episode again, or the next default seed (see the module docstring)."""
        if options is not None:
            if config is not None:
                raise EnvError("bad_config")
            config = options
        message = {"op": "reset"}
        if seed is not None:
            if not isinstance(seed, bool):
                try:
                    seed = operator.index(seed)  # numpy integers and the like
                except TypeError:
                    pass
            message["seed"] = seed
        if config is not None:
            message["config"] = config
        try:
            line = _dump(message)
        except Exception:
            raise EnvError("bad_config") from None
        reply = self._request(line, refusals=True)
        if "info" not in reply:
            raise EnvError("bad_reply")
        return reply["observation"], reply["info"]

    def step(self, action):
        reply = self._request(_step_line(action))
        try:
            return (reply["observation"], reply["reward"], reply["terminated"], reply["truncated"],
                    reply["info"])
        except KeyError:
            raise EnvError("bad_reply") from None

    # --- lifetime ----------------------------------------------------------------------------
    def _kill(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.kill()
            process.wait()
        except OSError:
            pass
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass

    def close(self) -> None:
        process, self._process = self._process, None
        self._pending = b""
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
