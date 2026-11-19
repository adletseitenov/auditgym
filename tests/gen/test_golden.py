"""Hand-computed examples and verify/solver agreement on hand-edited books (docs/AUDIT.md Q3, Q4, Q8).

Expected amounts below are written out by hand, not computed with auditgym.money.
"""
import copy
import unittest
from decimal import Decimal

from auditgym.generate import generate
from auditgym.money import closed_share, convert
from auditgym.solver import solve
from auditgym.verify import predicate_issues, verify


class GoldenConversionTest(unittest.TestCase):
    def test_half_even_ties(self):
        # 5 * 0.5 = 2.5 -> 2;  15 * 0.5 = 7.5 -> 8;  25 * 0.5 = 12.5 -> 12;  35 * 0.5 = 17.5 -> 18
        self.assertEqual(convert(5, Decimal("0.5"), 2, 2), 2)
        self.assertEqual(convert(15, Decimal("0.5"), 2, 2), 8)
        self.assertEqual(convert(25, Decimal("0.5"), 2, 2), 12)
        self.assertEqual(convert(35, Decimal("0.5"), 2, 2), 18)
        # Negative ties round to even as well: -2.5 -> -2, -7.5 -> -8
        self.assertEqual(convert(-5, Decimal("0.5"), 2, 2), -2)
        self.assertEqual(convert(-15, Decimal("0.5"), 2, 2), -8)

    def test_exponent_zero_two_three(self):
        # 123 units (exp 0) at 4.5 into exp 2: 123 * 100 * 4.5 = 55 350
        self.assertEqual(convert(123, Decimal("4.5"), 0, 2), 55350)
        # 12.345 (exp 3) at 2 into exp 2: 1234.5 * 2 = 2469
        self.assertEqual(convert(12345, Decimal("2"), 3, 2), 2469)
        # 12.345 (exp 3) at 1 into exp 2: 1234.5 -> 1234 (tie, even); 12.355 -> 1235.5 -> 1236
        self.assertEqual(convert(12345, Decimal("1"), 3, 2), 1234)
        self.assertEqual(convert(12355, Decimal("1"), 3, 2), 1236)
        # 2.50 (exp 2) at 1 into exp 0: 2.5 -> 2; 3.50 -> 3.5 -> 4
        self.assertEqual(convert(250, Decimal("1"), 2, 0), 2)
        self.assertEqual(convert(350, Decimal("1"), 2, 0), 4)
        # 1 000.00 USD at 466.21 into KZT tiyn: 100000 * 466.21 = 46 621 000
        self.assertEqual(convert(100000, Decimal("466.21"), 2, 2), 46621000)
        # 7 units (exp 0) at 0.0021437 into exp 3: 7 * 1000 * 0.0021437 = 15.0059 -> 15
        self.assertEqual(convert(7, Decimal("0.0021437"), 0, 3), 15)

    def test_payment_rule_share(self):
        # booked 1 000 001 for a total of 3: one third paid -> 333 333.67 -> 333 334
        self.assertEqual(closed_share(1000001, 3, 1), 333334)
        # half of 5 is a tie: 2.5 -> 2;  half of 7 -> 3.5 -> 4
        self.assertEqual(closed_share(5, 2, 1), 2)
        self.assertEqual(closed_share(7, 2, 1), 4)
        # fully paid closes exactly the booked amount
        self.assertEqual(closed_share(46621000, 100000, 100000), 46621000)


class VerifyCatchesTest(unittest.TestCase):
    def setUp(self):
        self.episode = generate(5, {"errors": {"fx_error": 1, "duplicate_invoice": 1}, "currencies": ["USD"]})
        self.assertEqual(verify(self.episode), [])

    def test_unbalanced_group_with_a_balanced_book(self):
        """V1: two groups out by the same amount in opposite directions keep the whole book balanced."""
        episode = copy.deepcopy(self.episode)
        ledger = episode["public"]["ledger"]
        invoice_row = next(r for r in ledger if r["invoice_id"] and not r["payment_id"] and r["debit_minor"])
        background = next(r for r in ledger if not r["invoice_id"] and not r["payment_id"] and r["debit_minor"] > 7)
        invoice_row["debit_minor"] += 7
        background["debit_minor"] -= 7
        self.assertEqual(sum(r["debit_minor"] for r in ledger), sum(r["credit_minor"] for r in ledger))
        self.assertTrue(any("is unbalanced by" in p for p in verify(episode)))

    def test_dirty_books_before_injection(self):
        """V6: an error already present in the pre-injection books is reported."""
        episode = copy.deepcopy(self.episode)
        pre = episode["private"]["pre_ledger"]
        invoices = {i["invoice_id"]: i for i in episode["private"]["pre_invoices"]}
        target = next(i for i, inv in invoices.items() if inv["currency"] == "USD"
                      and any(r["invoice_id"] == i and not r["payment_id"] for r in pre))
        for row in pre:  # both sides of a foreign invoice moved equally: a balanced fx error
            if row["invoice_id"] == target and not row["payment_id"]:
                row["debit_minor" if row["debit_minor"] else "credit_minor"] += 500
        problems = verify(episode)
        self.assertTrue(any(p.startswith("pre:") for p in problems), problems[:3])


class SolverVerifyAgreementTest(unittest.TestCase):
    """Q8: the solver and verify's predicates read the two hand cases the same way."""

    def fx_findings(self, public):
        found, _ = predicate_issues(public)
        by_verify = sorted((i["affected_ids"][0], i["amount"]) for i in found if i["type"] == "fx_error")
        by_solver = sorted((f["affected_ids"][0], f["claimed_amount_minor"]) for f in solve(public)
                           if f["type"] == "fx_error")
        return by_verify, by_solver

    def clean(self):
        return copy.deepcopy(generate(8, {"errors": {}, "currencies": ["USD"]})["public"])

    def test_foreign_payment_without_a_cash_row(self):
        public = self.clean()
        payments = {p["payment_id"]: p for p in public["payments"]}
        cash = next(r for r in public["ledger"] if r["account"] == "cash" and r["payment_id"]
                    and payments[r["payment_id"]]["currency"] == "USD")
        public["ledger"].remove(cash)
        by_verify, by_solver = self.fx_findings(public)
        self.assertEqual(by_verify, by_solver)
        self.assertNotIn(payments[cash["payment_id"]]["doc_id"], [doc for doc, _ in by_verify])

    def test_extra_receivable_credit_in_an_invoice_group(self):
        public = self.clean()
        own = next(r for r in public["ledger"] if r["invoice_id"] and not r["payment_id"]
                   and r["account"] == "receivable" and r["currency"])
        invoice = next(i for i in public["invoices"] if i["invoice_id"] == own["invoice_id"])
        extra = dict(own, row_id="extra00001", debit_minor=0, credit_minor=1000)
        balance = dict(own, row_id="extra00002", account="revenue", debit_minor=1000, credit_minor=0)
        public["ledger"] += [extra, balance]
        by_verify, by_solver = self.fx_findings(public)
        self.assertEqual(by_verify, by_solver)
        self.assertNotIn(invoice["doc_id"], [doc for doc, _ in by_verify])


class OperatorRulesOnOutputTest(unittest.TestCase):
    """Q4: DECISIONS B prohibitions hold on the generated records, not only through retries."""

    def test_prohibitions(self):
        checked = {"duplicate_invoice": 0, "allocation_gap": 0}
        for seed in range(80):
            episode = generate(seed, {"errors": {"duplicate_invoice": 2, "allocation_gap": 2, "period_cutoff": 1},
                                      "currencies": ["USD"]})
            public = episode["public"]
            docs = {d["doc_id"]: d for d in public["invoices"] + public["payments"]}
            refs = {}
            for payment in public["payments"]:
                for ref in payment["invoice_refs"] or []:
                    refs.setdefault(ref, []).append(payment)
            for issue in episode["private"]["truth"]["issues"]:
                kind = issue["type"]
                if kind == "duplicate_invoice":  # no duplicate of a refunded invoice
                    for doc in issue["affected_ids"]:
                        self.assertFalse(any(p["amount_minor"] < 0 for p in refs.get(docs[doc]["invoice_id"], [])))
                    checked[kind] += 1
                elif kind == "allocation_gap":  # gap only on an invoice with no other payments
                    invoice = next(docs[d] for d in issue["affected_ids"] if "line_items" in docs[d])
                    payment = next(docs[d] for d in issue["affected_ids"] if "payment_id" in docs[d])
                    self.assertEqual(refs[invoice["invoice_id"]], [payment])
                    self.assertEqual(payment["invoice_refs"], [invoice["invoice_id"]])
                    checked[kind] += 1
        self.assertGreater(min(checked.values()), 100)


if __name__ == "__main__":
    unittest.main()
