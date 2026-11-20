"""Gymnasium wrapper over `auditgym.Env` (the client of the evaluator process).

    import auditgym.gym_env                       # registers "AuditGym-v0" when gymnasium is installed
    env = gymnasium.make("AuditGym-v0")           # or AuditGymEnv(mode="train")
    obs, info = env.reset(seed=7, options={"errors": {"duplicate_invoice": 1}})   # options = config
    obs, reward, terminated, truncated, info = env.step('{"action": "calculator", "expression": "1+1"}')

Gymnasium is not a dependency: without it this module still imports and works, the class derives
from `object`, and the spaces are small stand-ins with the same `max_length`/`contains` surface. The
core (`auditgym/`) never imports gymnasium; only this module does, and only when it is available.

Observations and actions are JSON text (a `gymnasium.spaces.Text` of printable ASCII). The
observation is the evaluator's observation dict serialized with sorted keys and `ensure_ascii`; an
action may be that kind of JSON string or a dict, which is passed on as it is. A string that is not
JSON is sent as it is: the evaluator answers with a bounded error and spends a step, as for any
unknown action. The wrapper adds nothing to `reward`, `info` or the episode: it is the same
evaluator, so the train/eval rules apply unchanged (eval: reward 0.0 and info {}).
"""
from __future__ import annotations

import importlib
import json

from .client import DEFAULT_TIMEOUT, Env
from .limits import MAX_MESSAGE_BYTES

try:  # gymnasium is optional: an installed copy only adds the real Env base class and spaces
    # importlib, not an import statement: the determinism lint allows only the standard library in the
    # package, and this one optional, lazily found module is the single exception (see DECISIONS I).
    gymnasium = importlib.import_module("gymnasium")
    _spaces = importlib.import_module("gymnasium.spaces")
except ImportError:  # the usual case in this repository
    gymnasium = None
    _spaces = None

ENV_ID = "AuditGym-v0"
MAX_TEXT_LENGTH = MAX_MESSAGE_BYTES
# json.dumps(..., ensure_ascii=True) only ever writes these characters (controls are escaped).
CHARSET = "".join(chr(code) for code in range(32, 127))


class TextSpace:
    """Stand-in for `gymnasium.spaces.Text` when gymnasium is missing: a string of 1..max_length
    characters from `charset`."""

    def __init__(self, max_length: int = MAX_TEXT_LENGTH, min_length: int = 1, charset: str = CHARSET) -> None:
        self.min_length, self.max_length, self.charset = min_length, max_length, charset
        self.shape = None
        self.dtype = str

    def contains(self, value: object) -> bool:
        return (isinstance(value, str) and self.min_length <= len(value) <= self.max_length
                and all(char in self.charset for char in value))

    def __contains__(self, value: object) -> bool:
        return self.contains(value)

    def __repr__(self) -> str:
        return f"TextSpace({self.min_length}, {self.max_length})"


def _text_space():
    if _spaces is not None:
        return _spaces.Text(min_length=1, max_length=MAX_TEXT_LENGTH, charset=CHARSET)
    return TextSpace()


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_action(action: object) -> object:
    if isinstance(action, str):
        try:
            return json.loads(action)
        except (ValueError, RecursionError):
            return action  # not JSON: the evaluator takes it as an unknown action
    return action


_Base = gymnasium.Env if gymnasium is not None else object


class AuditGymEnv(_Base):
    """`auditgym.Env` with the Gymnasium interface. See the module text."""

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, mode: str = "train", results_path=None, *, shaped_reward: bool = False,
                 timeout: float | None = DEFAULT_TIMEOUT, episode_dir=None) -> None:
        if gymnasium is not None:
            super().__init__()
        self.observation_space = _text_space()
        self.action_space = _text_space()
        if episode_dir is None:
            self._env = Env(mode=mode, results_path=results_path, timeout=timeout, shaped_reward=shaped_reward)
        else:
            self._env = Env.from_files(episode_dir, mode=mode, results_path=results_path, timeout=timeout,
                                       shaped_reward=shaped_reward)

    @classmethod
    def from_files(cls, directory, mode: str = "train", results_path=None, *, shaped_reward: bool = False,
                   timeout: float | None = DEFAULT_TIMEOUT) -> "AuditGymEnv":
        """An environment over an episode directory (`episodes/train/ep_01`); `reset()` restarts it."""
        return cls(mode=mode, results_path=results_path, shaped_reward=shaped_reward, timeout=timeout,
                   episode_dir=directory)

    def reset(self, seed=None, options=None):
        if gymnasium is not None:
            super().reset(seed=seed)  # seeds np_random as the API wants; the episode seed is `seed` itself
        observation, info = self._env.reset(seed, options)
        return _dump(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(_parse_action(action))
        return _dump(observation), reward, terminated, truncated, info

    def render(self):
        return None

    def close(self) -> None:
        self._env.close()


def register() -> None:
    """Register `AuditGym-v0` (idempotent; does nothing without gymnasium)."""
    if gymnasium is None or ENV_ID in gymnasium.registry:
        return
    gymnasium.register(id=ENV_ID, entry_point="auditgym.gym_env:AuditGymEnv")


register()
