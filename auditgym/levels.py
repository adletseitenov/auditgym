"""Difficulty levels: ready-made configs from configs/levels.json.

    from auditgym import Env
    from auditgym.levels import load
    with Env() as env:
        observation, info = env.reset(seed, load("hard"))

The file lives in the repository checkout (configs/), not inside the package: without a checkout
`load` raises FileNotFoundError with the path it looked at. Levels only use documented config keys
(auditgym/config.py), so `normalize_config` validates them like any other config.
"""
from __future__ import annotations

import copy
import json
import pathlib

LEVELS_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "levels.json"


def _read(path: pathlib.Path | str | None = None) -> dict:
    with open(path or LEVELS_PATH, encoding="utf-8") as handle:
        return json.load(handle)["levels"]


def names(path: pathlib.Path | str | None = None) -> list[str]:
    """Level names in file order (easiest first)."""
    return list(_read(path))


def load(name: str, path: pathlib.Path | str | None = None) -> dict:
    """A fresh copy of the config of one level; KeyError for an unknown name."""
    levels = _read(path)
    if name not in levels:
        raise KeyError(f"unknown level {name!r}; known: {', '.join(levels)}")
    return copy.deepcopy(levels[name]["config"])
