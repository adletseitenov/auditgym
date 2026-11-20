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
- The client never reads the generation secret. `secret_file=PATH` hands the server the path
  ($AUDITGYM_SECRET_FILE in the server's environment) and drops $AUDITGYM_SECRET from that
  environment. A server started by Env runs as the policy's user, so this is not a security
  boundary by itself; the boundary is `server_command` (README «Закрытая проверка»): the evaluator
  in its own container or under its own user, holding the secret and the journal, while the policy
  gets nothing but the two pipes.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading

from .actions import default_results_path
from .limits import json_depth_exceeds
from .secret import SECRET_ENV, SECRET_FILE_ENV

_PACKAGE_PARENT = str(pathlib.Path(__file__).resolve().parent.parent)
_REFUSALS = ("bad_request", "bad_config", "not_loaded", "bad_episode", "internal_error")
_NAME_KEYS = ("action", "tool", "name", "type")
DEFAULT_TIMEOUT = 120.0  # seconds to wait for one reply (generating the largest episode takes a few)
_UNSERIALISABLE = "<not serialisable>"


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

    `Env(mode="train" | "eval", results_path=None, timeout=120.0, secret_file=None, server_command=None)`.
    `secret_file`: path of the generation secret, passed to the server, never read here.
    `server_command`: argv that starts `python -m auditgym serve ...` elsewhere (docker run -i ...,
    sudo -u ...); the client appends `--mode M` and `--no-load`, and the journal and the secret are
    configured in that command (results_path and secret_file are then refused: bad_server_command).
    In eval mode without a
    `results_path` the journal is $AUDITGYM_RESULTS or `runs/results.jsonl` in the current directory (audit C5); it is
    created with mode 0600. `timeout` bounds the wait for one reply (None: no bound); on expiry
    the server is stopped and `EnvError("timeout")` is raised.

    `reset(seed=None, config=None, options=None)`: `options` is a synonym of `config`; without a
    seed the episode of Env.from_files is restarted, or, if there is none, a new one is generated
    from the next default seed: 0 on the first such call, then the previous seed plus one. A refused reset raises `EnvError` and leaves the running episode.
    A `submit` that cannot be sent as JSON (NaN, a reference cycle, a 5 000-digit integer, nesting
    deeper than the recursion limit) is sent as a submit with an invalid structure: reward -1, the
    episode ends. Any other action that cannot be sent is an unknown action (a step is spent).
    `shaped_reward=True` (train mode only, ignored in eval) asks for the experimental partial-credit
    number `info["shaped_reward"]` in the terminal info (auditgym/shaped.py); by default the reply is unchanged.
    """

    def __init__(self, mode: str = "train", results_path=None, *, timeout: float | None = DEFAULT_TIMEOUT,
                 shaped_reward: bool = False, secret_file=None, server_command=None,
                 _files: bool = False) -> None:
        if mode not in ("train", "eval"):
            raise EnvError("bad_mode")
        self.mode = mode
        self._timeout = timeout
        if server_command is not None:
            # An evaluator started elsewhere (a container, another user): it owns its journal and
            # its secret, the client adds only the mode and the load rule (README «Закрытая проверка»).
            if (results_path is not None or secret_file is not None or isinstance(server_command, (str, bytes))
                    or not all(isinstance(part, str) for part in server_command) or not server_command):
                raise EnvError("bad_server_command")
            command = list(server_command)
        else:
            command = [sys.executable, "-m", "auditgym", "serve"]
            if mode == "eval" and results_path is None:
                results_path = default_results_path()
        command += ["--mode", mode]
        if not _files:
            command.append("--no-load")
        if shaped_reward and mode == "train":  # never in eval: no hints, no extra journal data
            command.append("--shaped-reward")
        pass_fds = ()
        if results_path is not None:
            try:
                path = pathlib.Path(results_path).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                path = os.fspath(path)
                if os.name == "posix":
                    # Read and append: a signed journal continues the chain from its last line.
                    descriptor = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
                    os.fchmod(descriptor, 0o600)  # the results are not for the policy's user group
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
        if secret_file is not None:
            try:
                environment[SECRET_FILE_ENV] = os.fspath(pathlib.Path(secret_file).resolve())
            except (TypeError, ValueError, OSError):
                raise EnvError("bad_secret_file") from None
            environment.pop(SECRET_ENV, None)  # the file wins; the value need not travel further
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
                   timeout: float | None = DEFAULT_TIMEOUT, shaped_reward: bool = False,
                   secret_file=None) -> "Env":
        env = cls(mode=mode, results_path=results_path, timeout=timeout, shaped_reward=shaped_reward,
                  secret_file=secret_file, _files=True)
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
            if json_depth_exceeds(message):
                return None
            return json.dumps(message, ensure_ascii=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError, OverflowError, MemoryError):
            return None

    def _request(self, message: dict) -> dict:
        process = self._process
        if process is None or process.poll() is not None:
            raise EnvError("closed")
        line = self._encode(message)
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
        if (message.get("op") in ("reset", "load") and observation.get("type") == "error"
                and isinstance(observation.get("error"), dict)):
            raise EnvError(str(observation["error"].get("code")))
        return reply

    def reset(self, seed=None, config=None, options=None):
        message = {"op": "reset"}
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
