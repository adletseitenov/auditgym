"""The honest strong baseline: read all four tables within the step budget, run the independent
solver on what was read, submit. Uses only list_records and submit.

The solver already returns contract-v2 evidence that contains a full allowed set (a duplicate
cites both documents and up to four posting rows of each invoice's own group, DECISIONS A2; an
fx_error the conversion rows and the rate key; an allocation gap the unallocated row and the
receivable rows of the invoice and the payment) and fresh lists for `affected_ids` and
`evidence_ids`, so its findings are submitted as they are.
"""
from __future__ import annotations

from auditgym.solver import solve

from .base import MAX_FINDINGS, ReadThenSubmit


class SolverPolicy(ReadThenSubmit):
    name = "solver_policy"

    def decide(self, public: dict) -> list[dict]:
        return solve(public)[:MAX_FINDINGS]
