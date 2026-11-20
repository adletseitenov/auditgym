"""Audit Q4: every limit has one constant and a test on both sides of its boundary, and a generator
that runs out of attempts says so with a plain error instead of a quiet result.

Boundaries: config (density, months, issues per answer, sizes), the evidence set size in verify
(10 pass, 11 fail), the answer size (256 KiB, audit R2; the case's own 68 KB must pass) and the
default amount tolerance (a type the policy does not list has 0, tests/judge/test_matcher.py)."""
from __future__ import annotations

import ast
import copy
import pathlib
import unittest
from unittest import mock

from auditgym import actions, config, findings, generate as generator, inject, limits, solver, verify
from auditgym.config import ConfigError, normalize_config
from auditgym.generate import MAX_ATTEMPTS, generate

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED = 987654321


class OneConstantPerLimitTest(unittest.TestCase):
    NAMES = ("MAX_FINDINGS", "MAX_ISSUES", "MAX_EVIDENCE_IDS", "MAX_EXPLANATION_CHARS", "MAX_MESSAGE_BYTES", "MAX_STEPS",
             "PAGE_SIZE", "MAX_PAGES", "MAX_LEDGER_ROWS", "MAX_PERIOD_MONTHS", "INVOICES_PER_ERROR")

    def test_modules_reuse_the_values_of_limits(self):
        self.assertEqual((findings.MAX_FINDINGS, findings.MAX_EVIDENCE_IDS, findings.MAX_MESSAGE_BYTES,
                          findings.MAX_EXPLANATION_CODE_POINTS),
                         (limits.MAX_FINDINGS, limits.MAX_EVIDENCE_IDS, limits.MAX_MESSAGE_BYTES,
                          limits.MAX_EXPLANATION_CHARS))
        self.assertEqual((actions.MAX_FINDINGS, actions.MAX_EVIDENCE_IDS, actions.MAX_EXPLANATION_CHARS,
                          actions.MAX_SUBMIT_BYTES, actions.MAX_STEPS, actions.PAGE_SIZE),
                         (limits.MAX_FINDINGS, limits.MAX_EVIDENCE_IDS, limits.MAX_EXPLANATION_CHARS,
                          limits.MAX_MESSAGE_BYTES, limits.MAX_STEPS, limits.PAGE_SIZE))
        self.assertEqual(verify.MAX_EVIDENCE, limits.MAX_EVIDENCE_IDS)
        self.assertEqual(solver.MAX_EVIDENCE, limits.MAX_EVIDENCE_IDS)
        self.assertEqual(config.MAX_ISSUES, limits.MAX_ISSUES)

    def test_no_module_of_the_package_assigns_a_limit_a_second_time(self):
        """A limit name bound to a number anywhere but limits.py would be a second source of truth."""
        for path in sorted((ROOT / "auditgym").glob("*.py")):
            if path.name == "limits.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in self.NAMES:
                            self.fail(f"{path.name}: {target.id} is a literal; import it from limits")


class ExhaustedAttemptsTest(unittest.TestCase):
    """CONTRACTS 3.2 (retries) and DECISIONS B: a bounded retry, then ConfigError("no_episode")."""

    def assert_plain_failure(self, error: ConfigError):
        self.assertEqual(error.code, "no_episode")
        text = str(error)
        self.assertIn(str(MAX_ATTEMPTS), text)
        for secret in (str(SEED), "attempt:", "provenance", "truth", "I1", "no target"):
            self.assertNotIn(secret, text)

    def test_an_operator_without_a_target_is_retried_then_refused(self):
        calls = mock.Mock(side_effect=inject.NoTarget("I1"))
        with mock.patch.object(generator, "inject_errors", calls):
            with self.assertRaises(ConfigError) as caught:
                generate(SEED, {"errors": {"duplicate_invoice": 1}})
        self.assertEqual(calls.call_count, MAX_ATTEMPTS)
        self.assert_plain_failure(caught.exception)

    def test_a_rejected_episode_is_retried_then_refused(self):
        with mock.patch.object(generator, "verify", return_value=["truth t1: rule I6 broken"]) as checked:
            with self.assertRaises(ConfigError) as caught:
                generate(SEED, {"errors": {"duplicate_invoice": 1}})
        self.assertEqual(checked.call_count, MAX_ATTEMPTS)
        self.assert_plain_failure(caught.exception)

    def test_an_unexpected_failure_is_not_a_result_either(self):
        with mock.patch.object(generator, "inject_errors", side_effect=KeyError(SEED)):
            with self.assertRaises(ConfigError) as caught:
                generate(SEED, {"errors": {"duplicate_invoice": 1}})
        self.assert_plain_failure(caught.exception)

    def test_one_failed_attempt_is_recorded_and_the_next_one_succeeds(self):
        real = generator.inject_errors
        calls = mock.Mock(side_effect=[inject.NoTarget("I1"), None])

        def flaky(*args, **kwargs):
            calls()
            return real(*args, **kwargs)

        with mock.patch.object(generator, "inject_errors", flaky):
            episode = generate(SEED, {"errors": {"duplicate_invoice": 1}})
        provenance = episode["private"]["provenance"]
        self.assertEqual((provenance["attempt"], len(provenance["retries"])), (1, 1))
        self.assertTrue(provenance["retries"][0]["reason"].startswith("no target"))


class ConfigBoundaryTest(unittest.TestCase):
    """K4-K6: at the limit the config is accepted; past it, ConfigError names the limit."""

    def refused(self, config_: dict) -> ConfigError:
        with self.assertRaises(ConfigError) as caught:
            normalize_config(config_)
        self.assertNotIn(str(SEED), str(caught.exception))
        return caught.exception

    def test_density(self):
        low = limits.INVOICES_PER_ERROR * 4
        errors = {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1}
        normalize_config({"size": {"invoices": [low, low]}, "errors": errors})
        error = self.refused({"size": {"invoices": [low - 1, low]}, "errors": errors})
        self.assertEqual(error.code, "too_dense")
        self.assertIn(str(limits.INVOICES_PER_ERROR), str(error))

    def test_months(self):
        normalize_config({"period": {"start": "2026-10-01", "months": limits.MAX_PERIOD_MONTHS}})
        error = self.refused({"period": {"start": "2026-10-01", "months": limits.MAX_PERIOD_MONTHS + 1}})
        self.assertEqual(error.code, "bad_period")
        self.assertIn(str(limits.MAX_PERIOD_MONTHS), str(error))

    def test_issues_per_episode(self):
        # docs/CASE.md section 03: 0-12 errors. The total counts the 2 issues interaction I1/I2 add.
        normalize_config({"errors": {"duplicate_invoice": limits.MAX_ISSUES}})
        error = self.refused({"errors": {"duplicate_invoice": limits.MAX_ISSUES + 1}})
        self.assertEqual(error.code, "too_many_errors")
        self.assertIn(str(limits.MAX_ISSUES), str(error))

    def test_interaction_issues_count_toward_the_limit(self):
        for interaction in ("I1", "I2"):
            normalize_config({"interaction": interaction, "errors": {"duplicate_invoice": 5, "allocation_gap": 5}})
            error = self.refused({"interaction": interaction,
                                  "errors": {"duplicate_invoice": 4, "allocation_gap": 4, "fx_error": 3}})
            self.assertEqual(error.code, "too_many_errors")
        self.assertEqual(self.refused({"interaction": "I1", "errors": {"duplicate_invoice": 4, "allocation_gap": 3,
                                                                       "fx_error": 3, "period_cutoff": 3}}).code,
                         "too_many_errors")  # 13 + 2 = 15
        self.assertEqual(self.refused({"interaction": "I2", "errors": {"duplicate_invoice": 3, "allocation_gap": 3,
                                                                       "fx_error": 3, "period_cutoff": 3}}).code,
                         "too_many_errors")  # 12 + 2 = 14

    def test_env_reports_too_many_issues_as_bad_config(self):
        from auditgym.evaluator import Evaluator, EvaluatorError  # noqa: PLC0415
        for config_ in ({"errors": {"duplicate_invoice": 13}},
                        {"interaction": "I1", "errors": {"duplicate_invoice": 4, "allocation_gap": 4, "fx_error": 3}},
                        {"interaction": "I2", "errors": {"duplicate_invoice": 3, "allocation_gap": 3, "fx_error": 3,
                                                         "period_cutoff": 3}}):
            with self.assertRaises(EvaluatorError) as caught:
                Evaluator().reset(1, config_)
            self.assertEqual(caught.exception.code, "bad_config")
        self.assertEqual(caught.exception.code, "bad_config")

    def test_sizes(self):
        for name, (low, high) in (("invoices", (limits.MIN_INVOICES, limits.MAX_INVOICES)),
                                  ("background_pairs", (limits.MIN_BACKGROUND_PAIRS, limits.MAX_BACKGROUND_PAIRS))):
            base = {"errors": {}, "size": {"invoices": [10, 10], "background_pairs": [0, 0]}}
            for bad in ([low, high + 1], [low - 1, high]):
                error = self.refused(dict(base, size=dict(base["size"], **{name: bad})))
                self.assertEqual(error.code, "bad_size", (name, bad))
                self.assertIn(str(high), str(error))


class EvidenceSetBoundaryTest(unittest.TestCase):
    """V4: verify accepts a truth evidence set of 10 distinct ids and refuses one of 11."""

    @classmethod
    def setUpClass(cls):
        cls.episode = generate(11, {"errors": {"duplicate_invoice": 1}})

    def problems_with(self, size: int) -> list[str]:
        episode = copy.deepcopy(self.episode)
        group = episode["private"]["truth"]["issues"][0]["evidence_sets"][0]
        for row in episode["public"]["ledger"]:
            if len(group) >= size:
                break
            if row["row_id"] not in group:
                group.append(row["row_id"])
        self.assertEqual(len(group), size)
        return [problem for problem in verify.verify(episode) if "distinct ids" in problem]

    def test_ten_pass_eleven_fail(self):
        self.assertEqual(self.problems_with(limits.MAX_EVIDENCE_IDS), [])
        self.assertEqual(len(self.problems_with(limits.MAX_EVIDENCE_IDS + 1)), 1)


class AnswerSizeBoundaryTest(unittest.TestCase):
    """F5: the limit is MAX_MESSAGE_BYTES (256 KiB, audit R2), not 64 KiB: the largest answer the
    case itself allows is about 68 KB of Cyrillic and must be accepted."""

    @staticmethod
    def payload(count: int, ids: int, text: str) -> dict:
        one = {"type": "period_cutoff", "affected_ids": ["a"], "claimed_amount_minor": 1, "currency": "USD",
               "evidence_ids": [f"evidence-id-{i:07d}" for i in range(ids)], "explanation": text}
        return {"episode_id": "ep", "findings": [dict(one) for _ in range(count)]}

    def test_the_largest_answer_of_the_case_is_accepted(self):
        payload = self.payload(limits.MAX_FINDINGS, limits.MAX_EVIDENCE_IDS, "ж" * limits.MAX_EXPLANATION_CHARS)
        size = findings.message_size(payload)
        self.assertGreater(size, 64 * 1024)
        self.assertLess(size, limits.MAX_MESSAGE_BYTES)
        self.assertEqual(len(findings.parse_findings(payload, "ep")), limits.MAX_FINDINGS)

    def test_one_byte_past_the_limit_is_refused(self):
        payload = self.payload(1, 1, "")
        pad = limits.MAX_MESSAGE_BYTES - findings.message_size(dict(payload, pad=""))
        payload["pad"] = "x" * pad
        self.assertEqual(findings.message_size(payload), limits.MAX_MESSAGE_BYTES)
        findings.parse_findings(payload, "ep")
        payload["pad"] += "x"
        with self.assertRaises(findings.StructureError) as caught:
            findings.parse_findings(payload, "ep")
        self.assertEqual(str(caught.exception), "message_too_large")


if __name__ == "__main__":
    unittest.main()


class JsonDepthTest(unittest.TestCase):
    """The nesting limit is explicit, so Python 3.11 and newer versions answer the same."""

    def test_depth_boundary(self):
        from auditgym.limits import MAX_JSON_DEPTH, json_depth_exceeds
        value = 1
        for _ in range(MAX_JSON_DEPTH):
            value = [value]
        self.assertFalse(json_depth_exceeds(value))
        self.assertTrue(json_depth_exceeds([value]))
        self.assertTrue(json_depth_exceeds({"a": [value]}))
