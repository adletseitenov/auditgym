"""Regressions for the solver review of 19.11: malformed input, offsets, ambiguous keys.

Each test reproduces one finding of the independent review and fails on the solver before the fix.
"""
import copy
import json
import os
import pathlib
import subprocess
import sys
import unittest

from auditgym.generate import generate
from auditgym.solver import solve

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = {"currencies": ["USD"], "errors": {}}


def clean_foreign_episode() -> dict:
    return generate(3, CONFIG)["public"]


class MalformedInputTest(unittest.TestCase):
    def test_unusable_policy_gives_no_findings_instead_of_raising(self):
        public = clean_foreign_episode()
        variants = {
            "no policy": {k: v for k, v in public.items() if k != "policy"},
            "policy is a list": dict(public, policy=[]),
            "no timezone": dict(public, policy={k: v for k, v in public["policy"].items() if k != "timezone"}),
            "period without offset": dict(public, policy=dict(public["policy"], period_start="2026-10-01T00:00:00")),
        }
        for name, broken in variants.items():
            with self.subTest(name):
                self.assertEqual(solve(broken), [])

    def test_odd_field_types_do_not_raise(self):
        public = clean_foreign_episode()
        expected = solve(public)
        broken = copy.deepcopy(public)
        broken["policy"]["accounts"] = ["cash", "receivable"]
        broken["invoices"].append(dict(broken["invoices"][0], doc_id="zz1", invoice_id=["not", "hashable"]))
        broken["payments"].append(dict(broken["payments"][0], doc_id="zz2", payment_id={"x": 1}))
        broken["fx"].append({"date": "2026-10-02", "source_currency": "USD", "target_currency": "KZT", "rate": "NaN"})
        self.assertEqual(solve(broken), expected)


class OffsetTest(unittest.TestCase):
    def _solve_in_time_zone(self, public: dict, zone: str) -> list:
        code = "import json,sys; from auditgym.solver import solve; print(json.dumps(solve(json.load(sys.stdin))))"
        env = dict(os.environ, TZ=zone, PYTHONPATH=str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], input=json.dumps(public), capture_output=True,
                             text=True, env=env, cwd=ROOT, check=True)
        return json.loads(out.stdout)

    def test_timestamp_without_offset_does_not_depend_on_the_machine_time_zone(self):
        public = clean_foreign_episode()
        invoice = next(inv for inv in public["invoices"] if inv["currency"] == "USD")
        invoice["issued_at"] = invoice["issued_at"][:19]  # drop "+05:00"
        answers = {zone: self._solve_in_time_zone(public, zone) for zone in ("UTC", "Pacific/Kiritimati")}
        self.assertEqual(answers["UTC"], answers["Pacific/Kiritimati"])
        self.assertFalse(any(invoice["doc_id"] in f["affected_ids"] for f in answers["UTC"]))


class AmbiguousKeyTest(unittest.TestCase):
    def test_two_rates_for_one_day_do_not_make_the_answer_order_dependent(self):
        public = clean_foreign_episode()
        invoice = next(inv for inv in public["invoices"] if inv["currency"] == "USD")
        day = invoice["issued_at"][:10]
        clash = {"date": day, "source_currency": "USD", "target_currency": "KZT", "rate": "400.0000"}
        at_end = copy.deepcopy(public)
        at_end["fx"].append(clash)
        at_start = copy.deepcopy(public)
        at_start["fx"].insert(0, clash)
        self.assertEqual(solve(at_end), solve(at_start))

    def test_exact_copy_of_a_payment_is_not_a_conflict_but_a_different_one_is(self):
        from auditgym.solver import _unique_by

        record = {"payment_id": "p1", "amount_minor": 5}
        self.assertEqual(list(_unique_by([record, dict(record)], "payment_id")), ["p1"])
        self.assertEqual(list(_unique_by([record, dict(record, amount_minor=6)], "payment_id")), [])

    def test_same_rate_twice_is_not_a_conflict(self):
        public = clean_foreign_episode()
        repeated = copy.deepcopy(public)
        repeated["fx"].append(dict(public["fx"][3]))
        self.assertEqual(solve(repeated), solve(public))


if __name__ == "__main__":
    unittest.main()
