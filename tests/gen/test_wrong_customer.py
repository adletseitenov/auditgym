"""wrong_customer end to end: operator, verify, solver against the truth, findings, config."""
import copy
import unittest

from auditgym.config import normalize_config
from auditgym.findings import TYPES
from auditgym.generate import generate
from auditgym.judge import judge
from auditgym.solver import solve
from auditgym.verify import predicate_issues, verify

WC = {"errors": {"wrong_customer": 2}, "currencies": ["USD"]}
MIXED = {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1,
                    "wrong_customer": 2}, "currencies": ["USD", "EUR"]}


def wc_issues(episode):
    return [i for i in episode["private"]["truth"]["issues"] if i["type"] == "wrong_customer"]


def key(item, amount):
    return (item["type"], tuple(sorted(item["affected_ids"])), item[amount], item["currency"])


class OperatorTest(unittest.TestCase):
    def test_only_the_receivable_debit_moves_to_another_customer(self):
        for seed in range(40):
            episode = generate(seed, WC)
            public, private = episode["public"], episode["private"]
            invoices = {i["doc_id"]: i for i in public["invoices"]}
            pre_rows = {r["row_id"]: r for r in private["pre_ledger"]}
            customers = {i["entity_id"] for i in public["invoices"]}
            self.assertEqual(len(wc_issues(episode)), 2)
            for issue in wc_issues(episode):
                invoice = invoices[issue["affected_ids"][0]]
                rows = [r for r in public["ledger"] if r["invoice_id"] == invoice["invoice_id"] and not r["payment_id"]]
                debit = next(r for r in rows if r["account"] == "receivable")
                credit = next(r for r in rows if r["account"] == "revenue")
                self.assertNotEqual(debit["entity_id"], invoice["entity_id"])
                self.assertIn(debit["entity_id"], customers)
                self.assertEqual(credit["entity_id"], invoice["entity_id"])
                self.assertEqual(issue["expected_amount_minor"], debit["debit_minor"])
                self.assertEqual(issue["evidence_sets"], [[invoice["doc_id"], debit["row_id"]]])  # documents first, then rows
                before = pre_rows[debit["row_id"]]  # the document and the amount are untouched
                self.assertEqual(before["entity_id"], invoice["entity_id"])
                self.assertEqual(before["debit_minor"], debit["debit_minor"])

    def test_provenance_records_the_changed_row(self):
        episode = generate(3, WC)
        steps = [s for s in episode["private"]["provenance"]["steps"] if s["operator"] == "wrong_customer"]
        self.assertEqual(len(steps), 2)
        for step in steps:
            # the invoice record is listed too (its internal ledger_entity changed) but reads the same in public form
            before = [r for r in step["before"] if "row_id" in r]
            after = [r for r in step["after"] if "row_id" in r]
            self.assertEqual(len(before), 1)
            self.assertEqual(len(after), 1)
            self.assertEqual(before[0]["row_id"], after[0]["row_id"])
            self.assertNotEqual(before[0]["entity_id"], after[0]["entity_id"])
            self.assertEqual([r for r in step["before"] if "row_id" not in r], [r for r in step["after"] if "row_id" not in r])

    def test_one_error_per_document_in_a_mixed_episode(self):
        for seed in range(40):
            episode = generate(seed, MIXED)
            docs = [d for i in episode["private"]["truth"]["issues"] for d in i["affected_ids"]]
            self.assertEqual(len(docs), len(set(docs)))


class VerifyTest(unittest.TestCase):
    def test_predicate_is_false_before_and_true_exactly_on_the_truth_after(self):
        for seed in range(30):
            episode = generate(seed, MIXED)
            public, private = episode["public"], episode["private"]
            pre = dict(public, ledger=private["pre_ledger"], invoices=private["pre_invoices"],
                       payments=private["pre_payments"])
            self.assertEqual([i for i in predicate_issues(pre)[0] if i["type"] == "wrong_customer"], [])
            found = {key(i, "amount") for i in predicate_issues(public)[0] if i["type"] == "wrong_customer"}
            truth = {key(i, "expected_amount_minor") for i in wc_issues(episode)}
            self.assertEqual(found, truth)

    def _tamper(self, edit):
        episode = copy.deepcopy(generate(5, WC))
        issue = wc_issues(episode)[0]
        invoice = next(i for i in episode["public"]["invoices"] if i["doc_id"] == issue["affected_ids"][0])
        rows = [r for r in episode["public"]["ledger"] if r["invoice_id"] == invoice["invoice_id"] and not r["payment_id"]]
        edit(episode, invoice, rows)
        return verify(episode)

    def test_untouched_episode_is_accepted(self):
        self.assertEqual(verify(generate(5, WC)), [])

    def test_a_revenue_row_on_the_wrong_customer_too_is_rejected(self):
        def edit(episode, invoice, rows):
            debit = next(r for r in rows if r["account"] == "receivable")
            next(r for r in rows if r["account"] == "revenue")["entity_id"] = debit["entity_id"]
        self.assertTrue(self._tamper(edit))

    def test_a_vendor_as_the_wrong_customer_is_rejected(self):
        def edit(episode, invoice, rows):
            customers = {i["entity_id"] for i in episode["public"]["invoices"]}
            vendor = next(r["entity_id"] for r in episode["public"]["ledger"] if r["entity_id"] not in customers)
            next(r for r in rows if r["account"] == "receivable")["entity_id"] = vendor
        self.assertTrue(self._tamper(edit))

    def test_wrong_evidence_is_rejected(self):
        def edit(episode, invoice, rows):
            wc_issues(episode)[0]["evidence_sets"] = [[invoice["doc_id"]]]
        self.assertTrue(self._tamper(edit))


class SolverTest(unittest.TestCase):
    def test_solver_equals_truth_on_300_seeds(self):
        for seed in range(300):
            episode = generate(seed, WC if seed % 3 else MIXED)
            findings = solve(episode["public"])
            truth = {key(i, "expected_amount_minor") for i in episode["private"]["truth"]["issues"]}
            self.assertEqual({key(f, "claimed_amount_minor") for f in findings}, truth, seed)
            self.assertEqual(len(findings), len(truth), seed)
            result = judge(episode, {"episode_id": episode["public"]["manifest"]["episode_id"], "findings": findings}, 5)
            self.assertEqual((result["tp"], result["fp"], result["fn"]), (len(truth), 0, 0), seed)

    def test_clean_company_gives_no_wrong_customer(self):
        for seed in range(20):
            findings = solve(generate(seed, {"errors": {}, "currencies": ["USD"]})["public"])
            self.assertEqual([f for f in findings if f["type"] == "wrong_customer"], [])


class ContractTest(unittest.TestCase):
    def test_type_is_known_to_the_findings_parser_and_policy(self):
        self.assertIn("wrong_customer", TYPES)
        policy = generate(1, WC)["public"]["policy"]
        self.assertIn("receivable_entity", policy)
        self.assertEqual(policy["amount_tolerance_minor"]["wrong_customer"], 0)

    def test_policy_does_not_depend_on_the_error_config(self):
        self.assertEqual(generate(1, WC)["public"]["policy"].keys(),
                         generate(1, {"errors": {}, "currencies": ["USD"]})["public"]["policy"].keys())

    def test_config_key_only_when_positive_and_old_configs_keep_their_digest(self):
        self.assertNotIn("wrong_customer", normalize_config({"errors": {"fx_error": 1}})["errors"])
        self.assertNotIn("wrong_customer", normalize_config(None)["errors"])
        self.assertEqual(normalize_config({"errors": {"wrong_customer": 3}})["errors"]["wrong_customer"], 3)


if __name__ == "__main__":
    unittest.main()
