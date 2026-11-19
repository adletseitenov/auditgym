import csv
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).parent / "fixtures" / "tiny"


def read_ledger(name):
    with open(ROOT / name, newline="") as handle:
        return list(csv.DictReader(handle))


class TinyFixtureTest(unittest.TestCase):
    def test_header_matches_the_case(self):
        with open(ROOT / "public" / "ledger.csv") as handle:
            self.assertEqual(handle.readline().strip(),
                             "row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id")

    def test_books_balance_before_and_after(self):
        for name in ("public/ledger.csv", "private/pre_ledger.csv"):
            rows = read_ledger(name)
            self.assertEqual(sum(int(r["debit_minor"]) for r in rows),
                             sum(int(r["credit_minor"]) for r in rows), name)
            for r in rows:
                self.assertEqual((int(r["debit_minor"]) > 0) + (int(r["credit_minor"]) > 0), 1, r["row_id"])

    def test_manifest_counts_match_files(self):
        manifest = json.loads((ROOT / "public" / "manifest.json").read_text())
        self.assertEqual(manifest["counts"]["ledger"], len(read_ledger("public/ledger.csv")))
        for table in ("invoices", "payments"):
            lines = (ROOT / "public" / f"{table}.jsonl").read_text().splitlines()
            self.assertEqual(manifest["counts"][table], len(lines))

    def test_truth_refers_to_existing_documents(self):
        truth = json.loads((ROOT / "private" / "truth.json").read_text())
        doc_ids = {json.loads(line)["doc_id"]
                   for table in ("invoices", "payments")
                   for line in (ROOT / "public" / f"{table}.jsonl").read_text().splitlines()}
        for issue in truth["issues"]:
            self.assertLessEqual(set(issue["affected_ids"]), doc_ids)
            for evidence in issue["evidence_sets"]:
                self.assertLessEqual(len(evidence), 10)

    def test_injected_deviation_equals_expected_amount(self):
        pre = {r["row_id"] for r in read_ledger("private/pre_ledger.csv")}
        added = [r for r in read_ledger("public/ledger.csv") if r["row_id"] not in pre]
        truth = json.loads((ROOT / "private" / "truth.json").read_text())
        self.assertEqual(sum(int(r["debit_minor"]) for r in added), truth["issues"][0]["expected_amount_minor"])


if __name__ == "__main__":
    unittest.main()
