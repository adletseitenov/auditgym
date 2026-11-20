"""Audit B3: a document a posting issue is about is still checked, except for the one deviation its
issue declares. Each test corrupts one such posting, records the change in provenance (so the
replay check stays quiet) and expects verify to name the posting problem."""
import copy
import unittest

from auditgym.generate import generate
from auditgym.verify import verify

FOREIGN = ["USD", "EUR", "RUB"]


def find(config_errors, issue_type, wanted, seeds=range(80)):
    """(episode, issue) for the first issue of `issue_type` whose posting rows satisfy `wanted`."""
    for seed in seeds:
        episode = generate(seed, {"errors": config_errors, "currencies": FOREIGN})
        for issue in episode["private"]["truth"]["issues"]:
            if issue["type"] == issue_type and wanted(episode, issue):
                return episode, issue
    raise AssertionError(f"no {issue_type} target found")


def doc_of(episode, doc_id):
    for table in ("invoices", "payments"):
        for record in episode["public"][table]:
            if record["doc_id"] == doc_id:
                return table, record
    raise KeyError(doc_id)


def payment_rows(episode, payment):
    return [r for r in episode["public"]["ledger"] if r["payment_id"] == payment["payment_id"]]


def account(episode, rows, name):
    target = episode["public"]["policy"]["accounts"][name]
    return [r for r in rows if r["account"] == target]


def rewrite(episode, issue_id, changes):
    """Apply {row_id: new row} to the open ledger and record it in the step of `issue_id`."""
    episode = copy.deepcopy(episode)
    ledger = episode["public"]["ledger"]
    steps = episode["private"]["provenance"]["steps"]
    step = next(s for s in steps if s["issue_id"] == issue_id)
    for row_id, new in changes.items():
        index = next(i for i, r in enumerate(ledger) if r["row_id"] == row_id)
        old = ledger[index]
        ledger[index] = new
        for record in step["after"]:
            if record.get("row_id") == row_id:
                record.clear()
                record.update(new)
                break
        else:
            step["before"].append(copy.deepcopy(old))
            step["after"].append(copy.deepcopy(new))
    return episode


def shifted(row, **delta):
    row = copy.deepcopy(row)
    for key, value in delta.items():
        row[key] += value
    return row


class PostingDeviationTest(unittest.TestCase):
    def assert_caught(self, episode, text):
        problems = verify(episode)
        self.assertFalse([p for p in problems if p.startswith("provenance")], problems)
        self.assertTrue(any(p.startswith("ledger:") and text in p for p in problems), problems)

    def test_fx_error_payment_still_closes_its_invoice_by_the_rule(self):
        def wanted(episode, issue):
            table, payment = doc_of(episode, issue["affected_ids"][0])
            rows = payment_rows(episode, payment) if table == "payments" else []
            return bool(account(episode, rows, "receivable")) and bool(account(episode, rows, "fx_difference"))

        episode, issue = find({"fx_error": 3}, "fx_error", wanted)
        self.assertEqual(verify(episode), [])
        _, payment = doc_of(episode, issue["affected_ids"][0])
        rows = payment_rows(episode, payment)
        receivable, fx = account(episode, rows, "receivable")[0], account(episode, rows, "fx_difference")[0]
        # One more minor unit closed, the fx row absorbs it: balanced, but not the payment rule.
        fx_new = shifted(fx, debit_minor=1) if fx["debit_minor"] else shifted(fx, credit_minor=-1)
        if fx_new["credit_minor"] == 0 and fx_new["debit_minor"] == 0:
            self.skipTest("fx row of one minor unit")
        corrupted = rewrite(episode, issue["issue_id"], {receivable["row_id"]: shifted(receivable, credit_minor=1),
                                                          fx["row_id"]: fx_new})
        self.assert_caught(corrupted, f"payment {payment['doc_id']} does not close its invoices by the payment rule")

    def test_allocation_gap_payment_keeps_the_rule_sum(self):
        def wanted(episode, issue):
            payments = [doc_of(episode, d)[1] for d in issue["affected_ids"] if doc_of(episode, d)[0] == "payments"]
            return bool(payments) and bool(account(episode, payment_rows(episode, payments[0]), "fx_difference"))

        episode, issue = find({"allocation_gap": 2}, "allocation_gap", wanted)
        self.assertEqual(verify(episode), [])
        payment = next(doc_of(episode, d)[1] for d in issue["affected_ids"] if doc_of(episode, d)[0] == "payments")
        rows = payment_rows(episode, payment)
        receivable, fx = account(episode, rows, "receivable")[0], account(episode, rows, "fx_difference")[0]
        fx_new = shifted(fx, credit_minor=1) if fx["credit_minor"] else shifted(fx, debit_minor=-1)
        if fx_new["credit_minor"] == 0 and fx_new["debit_minor"] == 0:
            self.skipTest("fx row of one minor unit")
        corrupted = rewrite(episode, issue["issue_id"], {receivable["row_id"]: shifted(receivable, credit_minor=-1),
                                                          fx["row_id"]: fx_new})
        self.assert_caught(corrupted, f"payment {payment['doc_id']} splits its closing")

    def test_period_cutoff_invoice_is_still_posted_at_the_policy_amount(self):
        episode, issue = find({"period_cutoff": 2}, "period_cutoff", lambda e, i: True)
        self.assertEqual(verify(episode), [])
        _, invoice = doc_of(episode, issue["affected_ids"][0])
        rows = [r for r in episode["public"]["ledger"]
                if r["invoice_id"] == invoice["invoice_id"] and r["payment_id"] is None]
        changes = {r["row_id"]: shifted(r, **({"debit_minor": 1} if r["debit_minor"] else {"credit_minor": 1}))
                   for r in rows}
        corrupted = rewrite(episode, issue["issue_id"], changes)
        self.assert_caught(corrupted, f"invoice {invoice['doc_id']} is not posted once at the policy amount")

    def test_fx_error_invoice_is_still_posted_for_its_entity(self):
        def wanted(episode, issue):
            return doc_of(episode, issue["affected_ids"][0])[0] == "invoices"

        episode, issue = find({"fx_error": 3}, "fx_error", wanted)
        self.assertEqual(verify(episode), [])
        _, invoice = doc_of(episode, issue["affected_ids"][0])
        other = next(i["entity_id"] for i in episode["public"]["invoices"] if i["entity_id"] != invoice["entity_id"])
        changes = {}
        for row in episode["public"]["ledger"]:
            if row["invoice_id"] == invoice["invoice_id"] and row["payment_id"] is None:
                changes[row["row_id"]] = dict(row, entity_id=other)
        corrupted = rewrite(episode, issue["issue_id"], changes)
        self.assert_caught(corrupted, f"invoice {invoice['doc_id']} is posted for another entity")


if __name__ == "__main__":
    unittest.main()
