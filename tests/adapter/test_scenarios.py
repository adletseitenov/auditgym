"""Block 2 smoke: generate_episode for every D01-D20 scenario type, checked by the data and the oracle.

This does not replace the jury's independent reconciliation (it cannot: the check that the injection is the
declared discrepancy is `auditgym.verify`, which is ours); it shows that every scenario type can be produced
by the documented config through generate_episode, loads, is consistent before and after the injection by
verify, is deterministic, and that the oracle through the adapter earns the maximum.
"""
from __future__ import annotations

import datetime as dt
import re
import tempfile
import unittest

import judge_adapter
from auditgym.storage import load_episode
from auditgym.verify import verify
from policies.oracle import Oracle
from tests.adapter import helpers
from tools import adapter_pairs
from tools.adapter_run import PolicyBridge


RATE = re.compile(r"[0-9]+(\.[0-9]+)?")


def _fetch(env, table: str, identifier: str):
    """One evidence id through the adapter actions: ledger/fx by get_record, documents by get_document."""
    if table in ("invoices", "payments"):
        action = {"name": "get_document", "args": {"doc_id": identifier}}
    else:
        action = {"name": "get_record", "args": {"table": table, "id": identifier}}
    observation, _, _, _, _ = env.step(action)
    return observation


def _table_of(public: dict, identifier: str):
    for table, key in (("ledger", "row_id"), ("invoices", "doc_id"), ("payments", "doc_id")):
        if any(record[key] == identifier for record in public[table]):
            return table
    if any(f"{r['date']}|{r['source_currency']}|{r['target_currency']}" == identifier for r in public["fx"]):
        return "fx"
    return None


def evidence_reachable(directory: str, episode: dict, journal: str) -> int:
    """Walk make_env in TRAIN mode on the directory: every affected id and every id of every evidence set of
    every issue must be fetchable through list/get actions, and the record must be the one on disk. Returns
    the number of ids fetched."""
    public, count = episode["public"], 0
    env = judge_adapter.make_env(journal_path=journal)
    try:
        env.reset(0, {"mode": "train", "episode_dir": directory})
        for issue in episode["private"]["truth"]["issues"]:
            ids = list(issue["affected_ids"]) + [i for group in issue["evidence_sets"] for i in group]
            for identifier in dict.fromkeys(ids):
                table = _table_of(public, identifier)
                if table is None:
                    raise AssertionError(f"{issue['issue_id']}: evidence id is in no table")
                observation = _fetch(env, table, identifier)
                if observation["error"] is not None:
                    raise AssertionError(f"{issue['issue_id']}: {table} id is not fetchable: {observation['error']}")
                if table == "fx":
                    fetched = observation["result"]
                    key = f"{fetched['date']}|{fetched['source_currency']}|{fetched['target_currency']}"
                    if key != identifier:
                        raise AssertionError("fx: another record came back")
                    count += 1
                    continue
                key = "row_id" if table == "ledger" else "doc_id"
                expected = next(r for r in public[table] if r[key] == identifier)
                if observation["result"] != expected:
                    raise AssertionError(f"{table}: the fetched record differs from the file")
                count += 1
    finally:
        env.close()
    return count


def _aware(text: str) -> bool:
    return dt.datetime.fromisoformat(text).utcoffset() is not None


def check_formats(tables: dict) -> list[str]:
    """Section 03 formats on the records as the policy reads them: time with an offset, money as integers of
    minimal units, decimal FX strings, explicit nulls for missing links."""
    problems = []
    for row in tables["ledger"]:
        if not _aware(row["timestamp"]):
            problems.append("ledger time without an offset")
        if any(type(row[k]) is not int for k in ("debit_minor", "credit_minor")):
            problems.append("ledger money is not an integer")
        if any(k not in row for k in ("entity_id", "invoice_id", "payment_id")):
            problems.append("ledger link is absent instead of null")
    for doc in tables["invoices"]:
        if not (_aware(doc["issued_at"]) and _aware(doc["due_at"])):
            problems.append("invoice time without an offset")
        if type(doc["total_minor"]) is not int:
            problems.append("invoice total is not an integer")
    for doc in tables["payments"]:
        if not _aware(doc["timestamp"]):
            problems.append("payment time without an offset")
        if type(doc["amount_minor"]) is not int or "invoice_refs" not in doc:
            problems.append("payment amount or refs")
    for rate in tables["fx"]:
        if not isinstance(rate["rate"], str) or not RATE.fullmatch(rate["rate"]):
            problems.append("fx rate is not a decimal string")
    return sorted(set(problems))


def balanced(rows: list[dict]) -> bool:
    return sum(r["debit_minor"] for r in rows) == sum(r["credit_minor"] for r in rows)


def independent_issue_problems(episode: dict) -> list[str]:
    """What the injection did, recomputed from the open rows and the pre-injection snapshot without verify and
    without the matcher: a duplicate is two posted invoices with one signature; a payment allocation gap
    leaves the unallocated credit equal to the truth amount, entirely (no receivable closing at all) or in
    part (closing and unallocated together are the pre-injection closing)."""
    public, private, problems = episode["public"], episode["private"], []
    ledger, pre_ledger = public["ledger"], private["pre_ledger"]
    invoices = {doc["doc_id"]: doc for doc in public["invoices"]}
    payments = {doc["doc_id"]: doc for doc in public["payments"]}
    for issue in private["truth"]["issues"]:
        if issue["type"] == "duplicate_invoice":
            first, second = (invoices[i] for i in issue["affected_ids"])
            same = all(first[k] == second[k] for k in ("entity_id", "currency", "total_minor", "line_items"))
            posted = all(any(r["invoice_id"] == d["invoice_id"] and r["account"] == "receivable"
                             and r["debit_minor"] > 0 for r in ledger) for d in (first, second))
            if not (same and posted and first["doc_id"] != second["doc_id"]):
                problems.append("duplicate: not one obligation reflected twice")
        if issue["type"] == "allocation_gap":
            pay = next(payments[i] for i in issue["affected_ids"] if i in payments)
            inv = next(invoices[i] for i in issue["affected_ids"] if i in invoices)

            def closing(rows, account):
                return sum(r["credit_minor"] for r in rows if r["account"] == account
                           and r["payment_id"] == pay["payment_id"] and (account == "unallocated"
                                                                         or r["invoice_id"] == inv["invoice_id"]))
            gap = closing(ledger, "unallocated")
            before = closing(pre_ledger, "receivable")
            after = closing(ledger, "receivable")
            if gap != issue["expected_amount_minor"] or gap <= 0 or closing(pre_ledger, "unallocated") != 0:
                problems.append("allocation gap: the unallocated credit is not the truth amount")
            created = private["provenance"]["steps"] and any(
                step["operator"] == "I1" and not step["before"] for step in private["provenance"]["steps"])
            if not created and after + gap != before:  # I1 creates the payment: nothing of it is in the snapshot
                problems.append("allocation gap: closing and unallocated are not the pre-injection closing")
            variant = private["provenance"]["steps"][0]["params"].get("variant", "partial")
            if not created and (variant == "missing") != (after == 0):
                problems.append("allocation gap: the variant does not match the rows")
    return problems


class ScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.env = judge_adapter.make_env(journal_path=f"{cls.scratch.name}/results.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.scratch.cleanup()


def _make(index: int):
    label, what, _, _ = adapter_pairs.MATRIX[index]

    def test(self):
        seed, config = adapter_pairs.discover(index)
        directory = f"{self.scratch.name}/{label}"
        helpers.generate_quietly(seed, config, directory)
        again = f"{self.scratch.name}/{label}-again"
        helpers.generate_quietly(seed, config, again)
        self.assertEqual(adapter_pairs.tree_digest(directory), adapter_pairs.tree_digest(again))  # deterministic
        episode = load_episode(directory)
        self.assertEqual(verify(episode), [], what)  # books consistent before the injection, the injection is the declared one
        truth = episode["private"]["truth"]["issues"]
        self.assertEqual(bool(truth), index >= 5, what)  # D01-D05 are clean, the rest have an error
        for issue in truth:
            self.assertTrue(issue["affected_ids"] and issue["evidence_sets"])
        findings = adapter_pairs.oracle_findings(directory)
        scored = judge_adapter.score_episode(directory, findings, 0, journal_path=f"{self.scratch.name}/score.log")
        self.assertEqual((scored["tp"], scored["fp"], scored["fn"], scored["f1"], scored["return"]),
                         (len(truth), 0, 0, 1.0, 1.0), what)
        # and through the adapter, closed mode: the oracle submits by step, the evaluator's journal agrees
        bridge = PolicyBridge(Oracle(episode["private"]["truth"]))
        seen = bridge.run(self.env, 0, {"mode": "closed", "episode_dir": directory})
        self.assertEqual((seen["reward"], seen["info"], seen["terminated"]), (0, {}, True))
        self.assertEqual(self.env.last_result()["reward"], 1.0)
        # condition 1 and 2 of the page, as far as our own tools go: books balance before and after the
        # injection (sums of debits and credits, whole book) and the injection is what the truth says
        self.assertTrue(balanced(episode["private"]["pre_ledger"]) and balanced(episode["public"]["ledger"]))
        self.assertEqual(independent_issue_problems(episode), [], what)
        # condition 4: a full evidence set (all of them) is reachable through adapter actions in train mode
        if truth:
            fetched = evidence_reachable(directory, episode, f"{self.scratch.name}/train.jsonl")
            self.assertGreaterEqual(fetched, 2 * len(truth), what)
        # condition 5: formats of section 03 on what a policy reads through list_records
        tables = adapter_pairs.read_all(self.env, directory)
        self.assertEqual(check_formats(tables), [], what)
    return test


for _index, _row in enumerate(adapter_pairs.MATRIX):
    setattr(ScenarioTest, f"test_{_row[0]}", _make(_index))


class ScenarioTypesTest(unittest.TestCase):
    def test_allocation_gap_has_both_a_missing_and_a_partial_episode(self):
        variants = []
        for index in (8, 9, 10):  # D09-D11
            seed, config = adapter_pairs.discover(index)
            from auditgym.generate import generate

            step = generate(seed, config)["private"]["provenance"]["steps"][0]
            variants.append(step["params"].get("variant", "partial"))
        self.assertIn("missing", variants)
        self.assertIn("partial", variants)

    def test_interactions_are_the_two_documented_scenarios_with_varied_parameters(self):
        from auditgym.generate import generate

        operators, configs = {}, {}
        for index in range(15, 20):  # D16-D20
            seed, config = adapter_pairs.discover(index)
            steps = generate(seed, config)["private"]["provenance"]["steps"]
            operators[adapter_pairs.MATRIX[index][0]] = {s["operator"] for s in steps}
            configs[adapter_pairs.MATRIX[index][0]] = repr(sorted(config.items()))
        self.assertTrue(all("I1" in operators[d] for d in ("D16", "D17", "D18")), operators)
        self.assertTrue(all("I2" in operators[d] for d in ("D19", "D20")), operators)
        self.assertEqual(len(set(configs.values())), 5)  # no two of them share the parameters

    def test_all_documented_error_types_and_both_interactions_are_covered(self):
        from auditgym.generate import generate

        seen_types = set()
        for index in (5, 8, 11, 13, 15, 18):
            seed, config = adapter_pairs.discover(index)
            seen_types |= {i["type"] for i in generate(seed, config)["private"]["truth"]["issues"]}
        self.assertEqual(seen_types, {"duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff"})
        modes = set()
        for index in (11, 12):  # fx: both the wrong date and the wrong rate variant exist
            seed, config = adapter_pairs.discover(index)
            modes |= {s["params"]["mode"] for s in generate(seed, config)["private"]["provenance"]["steps"]}
        self.assertEqual(modes, {"wrong_date", "wrong_rate"})


if __name__ == "__main__":
    unittest.main()
