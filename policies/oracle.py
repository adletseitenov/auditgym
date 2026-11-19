"""Oracle: evaluator-side only. It is handed the episode's truth through its constructor by the
runner or a tool that already holds the private files; it never asks the environment for truth.

At step 1 it submits every issue with the first evidence set listed for it. It measures the
maximum reward of the environment, not the ability of an agent.
"""
from __future__ import annotations

from .base import finding, submit


class Oracle:
    name = "oracle"

    def __init__(self, truth: dict) -> None:
        self.issues = [dict(issue) for issue in truth.get("issues", [])]

    def reset(self, observation: dict) -> None:
        self.episode_id = (observation.get("manifest") or {}).get("episode_id", "")

    def act(self, observation: dict) -> dict:
        findings = [finding(issue["type"], issue["affected_ids"], issue["expected_amount_minor"],
                            issue["currency"], (issue.get("evidence_sets") or [[]])[0], "Known issue.")
                    for issue in self.issues]
        return submit(self.episode_id, findings)
