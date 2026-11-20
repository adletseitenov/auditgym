"""Experimental shaped reward with partial credit (NOT part of the case reward).

The case reward (`auditgym.reward`, CASE section 05) is sparse: a finding scores only if every
predicate of the matcher holds. For research on learning from a sparse signal this module adds a
second number, `shaped_reward`, that gives partial credit to a finding that is close to an issue.
It never replaces or changes the official reward, `reward` in the reply stays the official one.

It is returned only by an evaluator that was started with `shaped_reward=True`, only in train mode
and only in the terminal `info` under the key `info["shaped_reward"]`. In eval mode it is never
computed, returned or journaled (CASE: no hints about correctness during the run).

Credit of a finding for one issue (weights are integers out of 10; the sum is 10):

    gate    the finding has the issue's type and shares at least one affected id with it,
            otherwise the credit is 0 (a wrong-type or unrelated finding earns nothing)
    type              2   (the gate itself)
    affected_ids      4   (the sets are equal)
    amount            2   (same currency and |claimed - expected| within the policy tolerance)
    evidence          2   (every id exists and the set covers an allowed evidence set)

so the credit is 1 exactly when the official matcher accepts the pair. The pairs are chosen
one-to-one: first the pairs of the official maximum matching (credit 1 each), then, greedily by
credit (ties by issue_id, then finding index), the remaining findings and issues.

    soft_tp     = sum of the credits of the chosen pairs          (exact Fraction)
    shaped_f1   = 2*soft_tp / (n_findings + n_issues)             (1 when both are 0)
    shaped      = clip(2*shaped_f1 - 1 - 0.002*inspection_steps, -1, 1)

With binary credits this is the official F1 and the official reward bit for bit. An invalid
structure is -1, as in the official reward.
"""
from __future__ import annotations

from fractions import Fraction

from . import matcher
from .findings import StructureError, parse_findings
from .reward import terminal_reward

WEIGHTS = {"type": 2, "affected_ids": 4, "amount": 2, "evidence": 2}
TOTAL_WEIGHT = sum(WEIGHTS.values())
INFO_KEY = "shaped_reward"


def pair_credit(finding: dict, issue: dict, policy: dict, known: frozenset) -> Fraction:
    """Partial credit of one finding for one issue, a multiple of 1/10 in [0, 1]."""
    if not matcher.same_type(finding, issue) or not set(finding["affected_ids"]) & set(issue["affected_ids"]):
        return Fraction(0)
    points = WEIGHTS["type"]
    if matcher.same_affected_ids(finding, issue):
        points += WEIGHTS["affected_ids"]
    if matcher.same_currency(finding, issue) and matcher.amount_within_tolerance(
            finding, issue, matcher.tolerance(policy, issue["type"])):
        points += WEIGHTS["amount"]
    if matcher.evidence_exists(finding, known) and matcher.evidence_covers(
            finding, issue, matcher.superset_allowed(policy)):
        points += WEIGHTS["evidence"]
    return Fraction(points, TOTAL_WEIGHT)


def soft_true_positives(truth: dict, findings: list, public: dict) -> Fraction:
    policy = public.get("policy", {})
    known = matcher.known_ids(public)
    issues = sorted(truth.get("issues", []), key=lambda issue: issue["issue_id"])
    chosen_findings: set = set()
    chosen_issues: set = set()
    total = Fraction(0)
    for index, issue_id in matcher.match(truth, findings, public)["pairs"]:
        chosen_findings.add(index)
        chosen_issues.add(issue_id)
        total += 1  # the official TP: credit 1 by construction
    candidates = []
    for index, finding in enumerate(findings):
        if index in chosen_findings:
            continue
        for issue in issues:
            if issue["issue_id"] in chosen_issues:
                continue
            credit = pair_credit(finding, issue, policy, known)
            if credit > 0:
                candidates.append((-credit, issue["issue_id"], index))
    for negative, issue_id, index in sorted(candidates):
        if index in chosen_findings or issue_id in chosen_issues:
            continue
        chosen_findings.add(index)
        chosen_issues.add(issue_id)
        total += -negative
    return total


def shaped_reward(episode: dict, payload: object, inspection_steps: int) -> float:
    """The experimental reward for a payload (None: truncated, an empty answer)."""
    if type(inspection_steps) is not int or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative int")
    public = episode["public"]
    truth = episode["private"]["truth"]
    if payload is None:
        findings: list = []
    else:
        try:
            findings = parse_findings(payload, public["manifest"]["episode_id"])
        except StructureError:
            return -1.0
    denominator = len(findings) + len(truth.get("issues", []))
    f1 = Fraction(1) if denominator == 0 else 2 * soft_true_positives(truth, findings, public) / denominator
    return terminal_reward(f1, inspection_steps)
