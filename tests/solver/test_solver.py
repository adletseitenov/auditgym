"""The independent solver against hand-built cases (no error generator involved)."""
from __future__ import annotations

import ast
import pathlib
import time
import unittest

from auditgym.generate import generate
from auditgym.solver import solve
from auditgym.storage import load_episode

from .cases import COMPANIES, DISTRACTOR_CASES, ERROR_CASES, build
from .helpers import Company, canonical, rename_and_shuffle, shift

ROOT = pathlib.Path(__file__).resolve().parents[2]
FINDING_KEYS = ["type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"]
OFFSETS = ("+05:00", "-11:00", "+14:00", "+00:00", "-03:30", "+05:45")
CURRENCY_SETS = ((), ("USD",), ("USD", "EUR"), ("CNY", "RUB", "EUR"))


def all_cases(cases: dict):
    for name, builder in sorted(cases.items()):
        for seed, currencies, timezone in COMPANIES:
            built = build(name, builder, seed, currencies, timezone)
            if built is not None:
                yield name, (seed, currencies, timezone), built


class ShapeMixin:
    def assert_shape(self, public: dict, findings: list[dict]) -> None:
        ids = {d["doc_id"] for d in public["invoices"] + public["payments"]} | {r["row_id"] for r in public["ledger"]}
        for finding in findings:
            self.assertEqual(list(finding), FINDING_KEYS)
            self.assertIsInstance(finding["claimed_amount_minor"], int)
            self.assertGreaterEqual(finding["claimed_amount_minor"], 0)
            self.assertLessEqual(len(finding["evidence_ids"]), 10)
            self.assertLessEqual(len(finding["explanation"]), 1000)
            self.assertTrue(set(finding["evidence_ids"]) <= ids, finding)
            self.assertTrue(set(finding["affected_ids"]) <= ids, finding)

    def assert_matches(self, public: dict, findings: list[dict], expected: list[dict], label: str) -> None:
        self.assert_shape(public, findings)
        self.assertEqual(len(findings), len(expected), f"{label}: {findings}")
        unmatched = list(findings)
        for item in expected:
            match = [f for f in unmatched if f["type"] == item["type"] and set(f["affected_ids"]) == item["affected"]]
            self.assertEqual(len(match), 1, f"{label}: no finding for {item['type']} {item['affected']}: {findings}")
            finding = match[0]
            unmatched.remove(finding)
            self.assertEqual(finding["claimed_amount_minor"], item["amount"], label)
            self.assertEqual(finding["currency"], item["currency"], label)
            self.assertTrue(any(s <= set(finding["evidence_ids"]) for s in item["evidence_sets"]),
                            f"{label}: evidence {finding['evidence_ids']} has no full set of {item['evidence_sets']}")


class CleanCompanyTest(unittest.TestCase):
    def test_clean_companies_give_no_findings(self):
        for seed in range(120):
            config = {"errors": {}, "currencies": list(CURRENCY_SETS[seed % 4]), "timezone": OFFSETS[seed % 6]}
            if seed % 5 == 0:
                config["period"] = {"start": "2026-12-01", "months": 2}
            self.assertEqual(solve(generate(seed, config)["public"]), [], f"seed {seed} {config}")

    def test_tiny_fixture_gives_exactly_its_truth(self):
        episode = load_episode(ROOT / "tests" / "fixtures" / "tiny")
        findings = solve(episode["public"])
        (issue,) = episode["private"]["truth"]["issues"]
        (finding,) = findings
        self.assertEqual((finding["type"], sorted(finding["affected_ids"]), finding["claimed_amount_minor"],
                          finding["currency"]),
                         (issue["type"], sorted(issue["affected_ids"]), issue["expected_amount_minor"], issue["currency"]))
        self.assertTrue(any(set(s) <= set(finding["evidence_ids"]) for s in issue["evidence_sets"]))


class ErrorCaseTest(ShapeMixin, unittest.TestCase):
    def test_each_error_gives_exactly_the_expected_findings(self):
        for name in ERROR_CASES:
            with self.subTest(case=name):
                built = 0
                for case, company, (public, expected) in all_cases({name: ERROR_CASES[name]}):
                    self.assert_matches(public, solve(public), expected, f"{case} {company}")
                    built += 1
                self.assertGreaterEqual(built, 2, f"{name}: too few companies fit the edit")


class DistractorTest(ShapeMixin, unittest.TestCase):
    def test_each_distractor_stays_clean(self):
        for name in DISTRACTOR_CASES:
            with self.subTest(case=name):
                built = 0
                for case, company, (public, expected) in all_cases({name: DISTRACTOR_CASES[name]}):
                    self.assertEqual(solve(public), [], f"{case} {company}")
                    built += 1
                self.assertGreaterEqual(built, 2, f"{name}: too few companies fit the edit")


class MetamorphicTest(unittest.TestCase):
    def test_order_and_ids_do_not_matter(self):
        cases = {**ERROR_CASES, **DISTRACTOR_CASES}
        for index, (name, company, (public, _)) in enumerate(all_cases(cases)):
            findings = solve(public)
            for round_ in range(2):
                renamed, mapping = rename_and_shuffle(public, index * 10 + round_)
                self.assertEqual(canonical(solve(renamed)), canonical(findings, mapping), f"{name} {company}")

    def test_output_order_is_deterministic(self):
        for name, company, (public, _) in all_cases({"all_four": ERROR_CASES["all_four"]}):
            renamed, _ = rename_and_shuffle(public, 99)
            self.assertEqual(solve(renamed), solve(rename_and_shuffle(public, 99)[0]))
            again, _ = rename_and_shuffle(renamed, 5)
            self.assertEqual([f["type"] for f in solve(again)], sorted(f["type"] for f in solve(again)))


class ToleranceTest(unittest.TestCase):
    def _foreign_unpaid(self):
        for seed in range(20):
            c = Company(seed, ("USD",))
            invoices = [inv for inv in c.unpaid(foreign=True) if c.invoice_rows(inv)]
            if invoices:
                return c, invoices[0]
        self.fail("no foreign unpaid invoice")

    def test_one_minor_unit_is_within_tolerance_two_is_not(self):
        for delta, expected in ((1, []), (-1, []), (2, [2]), (-2, [2])):
            c, invoice = self._foreign_unpaid()
            for row in c.invoice_rows(invoice):
                row["debit_minor" if row["debit_minor"] else "credit_minor"] += delta
            found = [f["claimed_amount_minor"] for f in solve(c.done())]
            self.assertEqual(found, expected, delta)

    def test_neighbour_day_rate_booking_is_caught(self):
        """The distractor bites: booking at the UTC-date rate instead of the local-date rate is an error."""
        for seed in range(30):
            c = Company(seed, ("USD",), "+05:00")
            for invoice in c.unpaid(foreign=True):
                moment = shift(invoice["issued_at"][:10] + "T02:00:00+05:00", days=0)
                day_before = shift(moment, days=-1)[:10]
                if not c.inside(moment) or day_before < c.policy["period_start"][:10]:
                    continue
                correct = c.to_base(invoice["total_minor"], "USD", moment)
                wrong = c.to_base(invoice["total_minor"], "USD", moment, date=day_before)
                if abs(correct - wrong) <= 1:
                    continue
                invoice["issued_at"] = moment
                for row in c.invoice_rows(invoice):
                    row["timestamp"] = moment
                    row["debit_minor" if row["debit_minor"] else "credit_minor"] = wrong
                (finding,) = solve(c.done())
                self.assertEqual((finding["type"], finding["claimed_amount_minor"]), ("fx_error", abs(correct - wrong)))
                return
        self.fail("no invoice fit the edit")


class ReadingTest(unittest.TestCase):
    """Pins the readings chosen where the definitions are silent (see the module docstring of the solver)."""

    def _company_with_invoice(self, foreign=False):
        for seed in range(20):
            c = Company(seed, ("USD",))
            invoices = [inv for inv in c.unpaid(foreign) if c.invoice_rows(inv)]
            if invoices:
                return c, invoices[0]
        self.fail("no invoice")

    def test_three_identical_posted_invoices_give_every_pair(self):
        c, invoice = self._company_with_invoice()
        c.add_invoice(invoice)
        c.add_invoice(invoice)
        findings = solve(c.done())
        self.assertEqual([f["type"] for f in findings], ["duplicate_invoice"] * 3)

    def test_unposted_copy_is_not_a_duplicate(self):
        c, invoice = self._company_with_invoice()
        c.add_invoice(invoice, post=False)
        self.assertEqual(solve(c.done()), [])

    def test_foreign_document_outside_the_period_is_only_a_cutoff(self):
        c, invoice = self._company_with_invoice(foreign=True)
        invoice["issued_at"] = shift(c.policy["period_end"], days=1)  # its rate now differs from the booking
        findings = solve(c.done())
        self.assertEqual([f["type"] for f in findings], ["period_cutoff"])

    def test_base_currency_invoice_at_a_wrong_amount_is_not_fx_error(self):
        c, invoice = self._company_with_invoice(foreign=False)
        for row in c.invoice_rows(invoice):
            row["debit_minor" if row["debit_minor"] else "credit_minor"] += 500
        self.assertEqual(solve(c.done()), [])

    def test_payment_with_two_unallocated_rows_is_left_unreported(self):
        for seed in range(20):
            c = Company(seed, ("USD",))
            pairs = c.paid_once(foreign=False)
            if pairs:
                break
        invoice, payment = pairs[0]
        booked = c.booked(invoice)
        rows = c.rebook_payment(payment, closes={invoice["invoice_id"]: booked - 300}, unallocated=200)
        c.row(rows, "fx_difference")["account"] = "unallocated"  # the other 100 also lands on unallocated
        self.assertEqual(solve(c.done()), [])

    def test_wrong_cash_absorbed_by_fx_difference_is_one_finding_on_the_payment(self):
        for seed in range(20):
            c = Company(seed, ("USD",))
            pairs = c.paid_once(foreign=True)
            if pairs:
                break
        invoice, payment = pairs[0]
        correct = c.to_base(payment["amount_minor"], "USD", payment["timestamp"])
        c.rebook_payment(payment, closes={invoice["invoice_id"]: c.booked(invoice)}, cash=correct + 5000)
        findings = solve(c.done())
        self.assertEqual([(f["type"], f["affected_ids"], f["claimed_amount_minor"]) for f in findings],
                         [("fx_error", [payment["doc_id"]], 5000)])


class PerformanceTest(unittest.TestCase):
    def test_largest_episode_is_fast(self):
        public = generate(11, {"errors": {}, "currencies": ["USD", "EUR"],
                               "size": {"invoices": [200, 200], "background_pairs": [480, 490]}})["public"]
        self.assertGreaterEqual(len(public["ledger"]), 1500)
        started = time.perf_counter()
        for _ in range(5):
            solve(public)
        self.assertLess((time.perf_counter() - started) / 5, 0.5)


class IndependenceTest(unittest.TestCase):
    def test_solver_imports_neither_generator_nor_private_data(self):
        source = (ROOT / "auditgym" / "solver.py").read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
        self.assertTrue(imported <= {"__future__", "json", "collections", "decimal", "money"}, imported)
        self.assertNotIn("\"private\"", source)


if __name__ == "__main__":
    unittest.main()
