"""The honest strong baseline: read all four tables within the step budget, run the independent
solver on what was read, submit. Uses only list_records and submit.

The solver already returns contract-v2 evidence (a duplicate cites both documents and one own
posting row of each invoice, DECISIONS A2) and fresh lists for `affected_ids` and `evidence_ids`,
so its findings are submitted as they are.
"""
from __future__ import annotations

from auditgym.solver import solve

from .base import MAX_FINDINGS, ReadThenSubmit


class SolverPolicy(ReadThenSubmit):
    name = "solver_policy"

    def decide(self, public: dict) -> list[dict]:
        return solve(public)[:MAX_FINDINGS]
