"""Tests written for the holes that `python3 -m tools.mutate_gen` found: each one kills a mutation that
the rest of the suite let survive (the mutation id is named in the test). All of them pass on the
unmutated code. Hand-built worlds and edited episodes, no random targets.
"""
import copy
import datetime as dt
import hashlib
import json
import re
import unittest
from decimal import Decimal

from unittest import mock

from auditgym import books, generate as generate_module, inject
from auditgym.generate import generate
from auditgym.rng import Rng
from auditgym.verify import Books, _check_rules, predicate_issues, verify
from auditgym.world import new_invoice, new_payment

UTC = dt.timezone.utc
CONFIG = {"errors": {"fx_error": 1, "period_cutoff": 1, "duplicate_invoice": 1, "allocation_gap": 1},
          "currencies": ["USD", "EUR"]}


def episode() -> dict:
    ep = generate(3, CONFIG)
    assert verify(ep) == []
    return copy.deepcopy(ep)


def messages(ep: dict, fragment: str) -> list[str]:
    return [problem for problem in verify(ep) if fragment in problem]


# ----------------------------------------------------------------------------- books.py
def hand_world(*invoices_and_payments) -> dict:
    """A one-currency world: no rates are needed, every amount is already in the base currency."""
    start = dt.datetime(2026, 10, 1, tzinfo=UTC)
    world = {"zone": UTC, "rates": {}, "base_currency": "KZT", "currency_exponents": {"KZT": 2},
             "period_start": start, "period_end": dt.datetime(2026, 11, 1, tzinfo=UTC),
             "invoices": [], "payments": [], "background": [], "distractors": False}
    for record in invoices_and_payments:
        world["invoices" if "line_items" in record else "payments"].append(record)
    return world


def hand_invoice(key: str, total: int) -> dict:
    issued = dt.datetime(2026, 10, 2, tzinfo=UTC)
    item = {"description": "x", "quantity": 1, "unit_price_minor": total, "amount_minor": total}
    invoice = new_invoice({}, key, "E1", "KZT", issued, 30, [item])
    invoice["posted_at"] = issued + dt.timedelta(hours=1)
    return invoice


def hand_payment(key: str, amount: int, day: int, refs: list[str]) -> dict:
    moment = dt.datetime(2026, 10, day, tzinfo=UTC)
    payment = new_payment(key, "E1", "KZT", moment, amount, refs)
    payment["posted_at"] = moment + dt.timedelta(hours=1)
    return payment


class PaymentRuleHandWorldTest(unittest.TestCase):
    def test_multi_reference_payment_closes_only_what_is_left(self):
        """B04: a payment of two invoices closes the rest of an invoice that was already part paid."""
        world = hand_world(hand_invoice("I0", 1000), hand_invoice("I1", 500),
                           hand_payment("P0", 400, 5, ["I0"]), hand_payment("P1", 1100, 6, ["I0", "I1"]))
        parts = books.payment_parts(world)
        self.assertEqual(parts["P0"]["closes"], {"I0": 400})
        self.assertEqual(parts["P1"]["closes"], {"I0": 600, "I1": 500})

    def test_overpayment_after_a_part_payment_closes_the_rest_and_credits_the_excess(self):
        """B09 (and B03): 400 then 800 on a total of 1000 closes 600 and leaves 200 unallocated."""
        world = hand_world(hand_invoice("I0", 1000), hand_payment("P0", 400, 5, ["I0"]),
                           hand_payment("P1", 800, 6, ["I0"]))
        parts = books.payment_parts(world)
        self.assertEqual((parts["P1"]["closes"], parts["P1"]["unallocated"]), ({"I0": 600}, 200))

    def test_exact_payment_has_no_excess(self):
        world = hand_world(hand_invoice("I0", 1000), hand_payment("P0", 400, 5, ["I0"]),
                           hand_payment("P1", 600, 6, ["I0"]))
        parts = books.payment_parts(world)
        self.assertEqual((parts["P1"]["closes"], parts["P1"]["unallocated"]), ({"I0": 600}, 0))


# ----------------------------------------------------------------------------- inject.py
class FarEnoughTest(unittest.TestCase):
    def test_minimum_difference_is_100_minor_units(self):
        """I01, I02: 100 minor units is far enough, 99 is not (on a correct value of 0 the
        per-mille test cannot interfere)."""
        self.assertTrue(inject._far_enough(100, 0))
        self.assertFalse(inject._far_enough(99, 0))
        self.assertTrue(inject._far_enough(0, 100))

    def test_minimum_difference_is_half_a_percent(self):
        self.assertTrue(inject._far_enough(100_500, 100_000))
        self.assertFalse(inject._far_enough(100_499, 100_000))
        self.assertTrue(inject._far_enough(99_500, 100_000))


class CutoffTargetTest(unittest.TestCase):
    def test_only_unposted_invoices_outside_the_period_are_targets(self):
        """I03: unposted and outside (and nothing else) qualifies."""
        world = hand_world()
        outside = dt.datetime(2026, 9, 29, tzinfo=UTC)
        inside = dt.datetime(2026, 10, 15, tzinfo=UTC)
        posted = dt.datetime(2026, 10, 16, tzinfo=UTC)
        cases = [(None, outside, True), (posted, outside, False), (None, inside, False), (posted, inside, False)]
        for posted_at, issued_at, expected in cases:
            with self.subTest(posted=posted_at is not None, outside=issued_at == outside):
                invoice = {"posted_at": posted_at, "issued_at": issued_at}
                self.assertIs(inject._outside_unposted(world, invoice), expected)


class AllocationGapHandWorldTest(unittest.TestCase):
    def target(self, *documents, order=("P0", "I0")):
        return inject.allocation_gap(Rng(1), hand_world(*documents), list(order), set())

    def test_control_a_plain_payment_is_a_target(self):
        _, issue = self.target(hand_invoice("I0", 100_000), hand_payment("P0", 60_000, 5, ["I0"]))
        self.assertEqual(issue["affected"], ["D:I0", "D:P0"])
        self.assertTrue(1 <= issue["amount_minor"] < 60_000)

    def test_a_closing_of_one_minor_unit_is_no_target(self):
        """I06: the gap must leave at least one unit of the receivable credit, so a closing of 1 cannot carry one."""
        with self.assertRaises(inject.NoTarget):
            self.target(hand_invoice("I0", 1000), hand_payment("P0", 1, 5, ["I0"]))

    def test_a_payment_with_a_wrong_rate_is_no_target(self):
        """I08: one error per document."""
        payment = hand_payment("P0", 60_000, 5, ["I0"])
        payment["rate_override"] = Decimal("1.01")
        with self.assertRaises(inject.NoTarget):
            self.target(hand_invoice("I0", 100_000), payment)

    def test_an_invoice_with_another_payment_is_no_target(self):
        """I09: the invoice must be paid by this payment alone."""
        with self.assertRaises(inject.NoTarget):
            self.target(hand_invoice("I0", 100_000), hand_payment("P0", 30_000, 5, ["I0"]),
                        hand_payment("P1", 30_000, 6, ["I0"]), order=("P0", "I0", "P1"))

    def test_gap_leaves_a_unit_of_the_receivable_credit(self):
        """I07: a share of 2-30 % of a large payment exceeds a small closing; the gap is capped below it."""
        payment = hand_payment("P0", 1_000_000, 5, ["I0"])
        for seed in range(12):
            for closing in (2, 3, 10):
                self.assertEqual(inject._gap(Rng(seed), hand_world(), payment, closing), closing - 1)


class WrongDateWindowTest(unittest.TestCase):
    MOMENT = dt.datetime(2026, 10, 15, 12, tzinfo=UTC)

    def world(self, odd_day: int) -> dict:
        world = hand_world()
        world["currency_exponents"] = {"KZT": 2, "USD": 2}
        right = Decimal("450")
        for delta in range(-12, 13):
            date = (self.MOMENT.date() + dt.timedelta(days=delta)).isoformat()
            world["rates"][(date, "USD")] = Decimal("500") if delta == odd_day else right
        return world

    def used_dates(self, odd_day: int) -> set:
        dates = set()
        for seed in range(30):
            chosen = inject.wrong_rate(Rng(seed), self.world(odd_day), 100_000, "USD", self.MOMENT)
            if chosen:
                dates.add(chosen[1]["rate_date"])
        return dates

    def test_a_rate_five_days_away_can_be_used(self):
        self.assertIn((self.MOMENT.date() + dt.timedelta(days=5)).isoformat(), self.used_dates(5))

    def test_a_rate_six_days_away_is_never_used(self):
        """I10: the table has five days of margin; the window is five days."""
        self.assertNotIn((self.MOMENT.date() + dt.timedelta(days=6)).isoformat(), self.used_dates(6))


# ----------------------------------------------------------------------------- verify.py
class VerifyHolesTest(unittest.TestCase):
    def test_a_row_with_zero_on_both_sides_is_rejected(self):
        """V03: a zero-sum extra row keeps every group balanced, only its own shape is wrong."""
        ep = episode()
        row = dict(ep["public"]["ledger"][0], row_id="zzzzzzzzzz", debit_minor=0, credit_minor=0)
        ep["public"]["ledger"].append(row)
        self.assertTrue(messages(ep, "must have exactly one positive integer side"))

    def test_several_unallocated_rows_on_a_clean_payment_are_rejected(self):
        """V18: a split row whose parts add up to the rule's amount is still not what the books produce."""
        ep, pid = None, None
        for seed in range(3, 60):
            ep = generate(seed, CONFIG)
            public = ep["public"]
            gap_docs = {d for issue in ep["private"]["truth"]["issues"] if issue["type"] == "allocation_gap"
                        for d in issue["affected_ids"]}
            doc_of = {p["payment_id"]: p["doc_id"] for p in public["payments"]}
            pid = next((r["payment_id"] for r in public["ledger"] if r["account"] == "unallocated"
                        and r["credit_minor"] >= 2 and doc_of[r["payment_id"]] not in gap_docs), None)
            if pid:
                break
        self.assertIsNotNone(pid, "no clean payment with an unallocated row")
        ep = copy.deepcopy(ep)
        ledger = ep["public"]["ledger"]
        row = next(r for r in ledger if r["payment_id"] == pid and r["account"] == "unallocated")
        first = row["credit_minor"] // 2
        extra = dict(row, row_id="zzzzzzzzzz", credit_minor=row["credit_minor"] - first)
        row["credit_minor"] = first
        ledger.append(extra)
        self.assertTrue(messages(ep, "has an unallocated amount the payment rule does not give"))

    def test_fx_tolerance_is_inclusive_for_an_invoice_and_a_payment(self):
        """V04, V05: a deviation equal to the tolerance (1) is no fx_error, one more is."""
        ep = episode()
        public = ep["public"]
        tolerance = public["policy"]["amount_tolerance_minor"]["fx_error"]
        self.assertEqual(tolerance, 1)
        truth = {d for issue in ep["private"]["truth"]["issues"] for d in issue["affected_ids"]}
        books_ = Books(public)
        invoices = {i["invoice_id"]: i for i in public["invoices"]}
        payments = {p["payment_id"]: p for p in public["payments"]}
        invoice = next(i for i in public["invoices"] if i["currency"] != "KZT" and i["doc_id"] not in truth
                       and books_.own.get(i["invoice_id"]))
        payment = next(p for p in public["payments"] if p["currency"] != "KZT" and p["doc_id"] not in truth
                       and books_.by_payment.get(p["payment_id"]))
        self.assertIn(invoice["invoice_id"], invoices)
        self.assertIn(payment["payment_id"], payments)
        invoice_rows = books_.own[invoice["invoice_id"]]
        cash_row = next(r for r in books_.by_payment[payment["payment_id"]] if r["account"] == "cash")
        correct_invoice = books_.policy_amount(invoice["total_minor"], invoice["currency"], invoice["issued_at"])
        correct_cash = books_.policy_amount(payment["amount_minor"], payment["currency"], payment["timestamp"])

        def fx_docs(shift: int) -> set:
            for row in invoice_rows:
                row["debit_minor" if row["debit_minor"] else "credit_minor"] = correct_invoice + shift
            cash_row["debit_minor"], cash_row["credit_minor"] = correct_cash + shift, 0
            found, _ = predicate_issues(public)
            return {d for i in found if i["type"] == "fx_error" for d in i["affected_ids"]}

        at = fx_docs(tolerance)
        self.assertNotIn(invoice["doc_id"], at)
        self.assertNotIn(payment["doc_id"], at)
        over = fx_docs(tolerance + 1)
        self.assertIn(invoice["doc_id"], over)
        self.assertIn(payment["doc_id"], over)

    def test_three_invoices_with_one_signature_are_an_undefined_case(self):
        """V06: two posted invoices are a duplicate pair, a third makes the definition undefined."""
        ep = episode()
        public = ep["public"]
        original = next(i for i in public["invoices"] if any(r["invoice_id"] == i["invoice_id"] and r["payment_id"] is None
                                                             for r in public["ledger"]))
        _, undefined = predicate_issues(public)
        self.assertEqual([case for case in undefined if "share a signature" in case], [])
        for n in (1, 2):
            clone = dict(copy.deepcopy(original), doc_id=f"cc{n}", invoice_id=f"ci{n}")
            public["invoices"].append(clone)
            public["ledger"] += [dict(r, row_id=f"cr{n}{k}", invoice_id=clone["invoice_id"])
                                 for k, r in enumerate(x for x in public["ledger"]
                                                       if x["invoice_id"] == original["invoice_id"] and x["payment_id"] is None)]
            _, undefined = predicate_issues(public)
            flagged = [case for case in undefined if "share a signature" in case]
            self.assertEqual(bool(flagged), n == 2, (n, flagged))

    def test_evidence_ids_must_be_open_records(self):
        """V09: an evidence id that is not in the files is named as such."""
        ep = episode()
        ep["private"]["truth"]["issues"][0]["evidence_sets"][0][0] = "not-an-open-id"
        self.assertTrue(messages(ep, "an id is not in the open records"))

    def test_truth_episode_id_must_match_the_manifest(self):
        """V12."""
        ep = episode()
        ep["private"]["truth"]["episode_id"] = ep["public"]["manifest"]["episode_id"] + "x"
        self.assertTrue(messages(ep, "episode_id differs from the manifest"))

    def test_replay_notices_a_step_adding_an_existing_id(self):
        """V13: the same record added twice by one step."""
        ep = episode()
        step = ep["private"]["provenance"]["steps"][0]
        step["after"].append(copy.deepcopy(step["after"][0]))
        self.assertTrue(messages(ep, "adds an id that already exists"))

    def test_a_line_item_with_quantity_zero_is_rejected(self):
        """V15: 0 x price = 0 = amount, so only the quantity rule can reject it."""
        ep = episode()
        invoice = ep["public"]["invoices"][0]
        item = invoice["line_items"][0]
        item["quantity"], item["amount_minor"] = 0, 0
        invoice["total_minor"] = sum(i["amount_minor"] for i in invoice["line_items"])
        self.assertTrue(messages(ep, "quantity x unit price differs from the amount"))

    def test_a_rate_needs_six_significant_digits(self):
        """V17: five digits are too few."""
        ep = episode()
        ep["public"]["fx"][0]["rate"] = "535.29"
        self.assertTrue(messages(ep, "needs 6 significant digits"))
        ep["public"]["fx"][0]["rate"] = "535.299"
        self.assertEqual(messages(ep, "significant digits"), [])


class RuleBoundaryTest(unittest.TestCase):
    """V10, V11: the generator rules of docs/DECISIONS.md B at their exact boundaries."""

    def books_with(self, invoice: dict) -> Books:
        public = episode()["public"]
        public["invoices"], public["payments"], public["ledger"] = [invoice], [], []
        return Books(public)

    def invoice(self, issued_at: str, currency: str, total: int) -> dict:
        item = {"description": "x", "quantity": 1, "unit_price_minor": total, "amount_minor": total,
                "service_start": "2026-09-01", "service_end": "2026-09-30"}
        return {"doc_id": "d1", "invoice_id": "i1", "entity_id": "e1", "issued_at": issued_at,
                "due_at": issued_at, "currency": currency, "total_minor": total, "line_items": [item]}

    def test_fx_difference_of_exactly_100_minor_units_is_allowed(self):
        invoice = self.invoice("2026-10-15T12:00:00+05:00", "USD", 10)
        books_ = self.books_with(invoice)
        below = "the injected fx difference is below 100 minor units or 0.5 %"
        correct = books_.policy_amount(10, "USD", invoice["issued_at"])
        self.assertLessEqual(5 * correct, 100 * 1000)  # the per-mille rule is not the one under test
        for amount, expected in ((100, False), (99, True)):
            with self.subTest(amount=amount):
                issue = {"type": "fx_error", "affected_ids": ["d1"], "expected_amount_minor": amount}
                self.assertEqual(_check_rules(books_, issue, "fx_error") == below, expected)

    def test_cutoff_date_four_days_outside_is_too_far(self):
        books_ = self.books_with(self.invoice("2026-09-27T00:00:00+05:00", "KZT", 1000))
        issue = {"type": "period_cutoff", "affected_ids": ["d1"], "expected_amount_minor": 1000}
        self.assertEqual(_check_rules(books_, issue, ""), "a cutoff date is more than four days outside the period")
        books_ = self.books_with(self.invoice("2026-09-27T00:00:01+05:00", "KZT", 1000))
        self.assertIsNone(_check_rules(books_, issue, ""))


class GenerateHolesTest(unittest.TestCase):
    def test_episode_id_is_twelve_hex_of_the_open_records_hash(self):
        """G06: the id is documented as a hash of the open records."""
        public = generate(3, CONFIG)["public"]
        body = json.dumps({name: public[name] for name in ("ledger", "invoices", "payments", "fx")},
                          ensure_ascii=False, sort_keys=True)
        expected = "ep_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
        self.assertEqual(public["manifest"]["episode_id"], expected)
        self.assertRegex(public["manifest"]["episode_id"], r"^ep_[0-9a-f]{12}$")

    def test_provenance_params_name_public_ids(self):
        """G04: original_doc_id is translated from the internal key to the public document id."""
        found = 0
        for seed in range(6):
            ep = generate(seed, {"errors": {"duplicate_invoice": 1}, "interaction": "I2"})
            docs = {d["doc_id"] for d in ep["public"]["invoices"] + ep["public"]["payments"]}
            for step in ep["private"]["provenance"]["steps"]:
                if "original_doc_id" in step["params"]:
                    found += 1
                    self.assertIn(step["params"]["original_doc_id"], docs)
        self.assertGreater(found, 0)

    def test_ledger_of_exactly_the_maximum_size_is_accepted(self):
        """G01: the limit is inclusive; one row less than the ledger has is too few."""
        reference = generate(3, CONFIG)
        rows = len(reference["public"]["ledger"])
        with mock.patch.object(generate_module, "MAX_LEDGER_ROWS", rows):
            same = generate(3, CONFIG)
        self.assertEqual(same["private"]["provenance"]["attempt"], reference["private"]["provenance"]["attempt"])
        self.assertEqual(same["public"], reference["public"])
        with mock.patch.object(generate_module, "MAX_LEDGER_ROWS", rows - 1):
            smaller = generate(3, CONFIG)
        self.assertGreater(smaller["private"]["provenance"]["attempt"], reference["private"]["provenance"]["attempt"])


if __name__ == "__main__":
    unittest.main()
