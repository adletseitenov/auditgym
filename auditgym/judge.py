"""Judge: structure check, matching and reward in one pure function (docs/CONTRACTS.md §7).

`payload=None` means the episode was truncated (step or time limit): the answer is empty and
`inspection_steps` is used as given (the evaluator passes 100). Invalid structure gives reward
-1.0, zero metrics, tp = fp = 0, fn = number of truth issues (none was found), and
`structure_error` set to the StructureError code.
"""
from __future__ import annotations

from auditgym.findings import StructureError, parse_findings
from auditgym.matcher import match
from auditgym.reward import metrics, terminal_reward


def judge(episode: dict, payload: object, inspection_steps: int) -> dict:
    if type(inspection_steps) is not int or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative int")
    public = episode["public"]
    truth = episode["private"]["truth"]
    if payload is None:
        findings: list = []
    else:
        try:
            findings = parse_findings(payload, public["manifest"]["episode_id"])
        except StructureError as error:
            return {"reward": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "tp": 0, "fp": 0, "fn": len(truth.get("issues", [])),
                    "structure_error": error.code}
    counts = match(truth, findings, public)
    scores = metrics(counts["tp"], counts["fp"], counts["fn"])
    return {"reward": terminal_reward(scores["f1"], inspection_steps),
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
            "structure_error": None}
