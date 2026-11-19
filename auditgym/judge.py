"""The judge: structure check, matching and terminal reward in one call.

Runs on the evaluator side only. The returned dict is judge-side diagnostics; the evaluator
decides what reaches the agent (in train mode only reward, precision, recall, f1 and
inspection_steps; in eval mode nothing).
"""
from __future__ import annotations

from typing import Any

from auditgym.findings import StructureError, parse_findings
from auditgym.matcher import match
from auditgym.reward import f1_exact, metrics, terminal_reward

INVALID_STRUCTURE_REWARD = -1.0


def judge(episode: dict[str, Any], payload: Any, inspection_steps: int) -> dict[str, Any]:
    """Score one submission.

    `payload` is the findings object from `submit`; `None` means an empty answer (truncation by
    the step or time limit), which is scored like `{"findings": []}`, not as a structure error.
    """
    public = episode["public"]
    truth = episode["private"]["truth"]
    issue_count = len(truth.get("issues", []))

    if payload is None:
        findings: list[dict[str, Any]] = []
    else:
        try:
            findings = parse_findings(payload, public["manifest"]["episode_id"])
        except StructureError as error:
            return {"reward": INVALID_STRUCTURE_REWARD, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "tp": 0, "fp": 0, "fn": issue_count, "structure_error": error.code,
                    "pairs": [], "reasons": []}

    result = match(truth, findings, public)
    scores = metrics(result["tp"], result["fp"], result["fn"])
    reward = terminal_reward(f1_exact(result["tp"], result["fp"], result["fn"]), inspection_steps)
    return {"reward": reward, **scores, "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
            "structure_error": None, "pairs": result["pairs"], "reasons": result["reasons"]}
