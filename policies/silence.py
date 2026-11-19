"""Silence: submit an empty answer at the first step. The reference point for every attack."""
from __future__ import annotations

from .base import submit


class Silence:
    name = "silence"

    def reset(self, observation: dict) -> None:
        self.episode_id = (observation.get("manifest") or {}).get("episode_id", "")

    def act(self, observation: dict) -> dict:
        return submit(self.episode_id, [])
