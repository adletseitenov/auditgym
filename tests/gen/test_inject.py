import copy
import json
import unittest
from collections import Counter

from auditgym.config import ConfigError
from auditgym.generate import generate
from auditgym.money import in_period, local_date
from auditgym.storage import episode_files
from auditgym.verify import invoice_signature, verify

SEEDS = range(200)
SMALL = {"invoices": [30, 45], "background_pairs": [10, 40]}


def config(dup=0, cut=0, **extra):
    return {"errors": {"duplicate_invoice": dup, "period_cutoff": cut}, "size": SMALL, **extra}


# (name, config): each type alone, both together, counts 0/1/3, foreign currencies, extreme offsets.
GRID = [
    ("none", config()),
    ("dup1", config(dup=1, currencies=[])),
    ("cut1", config(cut=1, currencies=[])),
    ("dup3_usd", config(dup=3, currencies=["USD"])),
    ("cut3_eur_cny", config(cut=3, currencies=["EUR", "CNY"])),
    ("both1_m11", config(dup=1, cut=1, currencies=["USD"], timezone="-11:00")),
    ("both3_p14", config(dup=3, cut=3, currencies=["USD", "RUB"], timezone="+14:00",
                         period={"start": "2026-12-01", "months": 2})),
    ("both_usd_base", config(dup=2, cut=2, base_currency="USD", currencies=["KZT", "EUR"], timezone="+05:30")),
]


def by_id(records, key):
    return {record[key]: record for record in records}


class GridTest(unittest.TestCase):
    """verify() is empty and the truth has exactly the configured issues, on 200 seeds per config."""

    def test_grid(self):
        for name, cfg in GRID:
            for seed in SEEDS:
                with self.subTest(config=name, seed=seed):
                    episode = generate(seed, cfg)
                    self.assertEqual(verify(episode), [])
                    counts = Counter(issue["type"] for issue in episode["private"]["truth"]["issues"])
                    expected = {k: v for k, v in cfg["errors"].items() if v}
                    self.assertEqual(dict(counts), expected)
                    self.assert_ledger_diff(episode)

    def assert_ledger_diff(self, episode):
        """pre_ledger and ledger differ exactly by the rows the steps record."""
        pre = by_id(episode["private"]["pre_ledger"], "row_id")
        post = by_id(episode["public"]["ledger"], "row_id")
        added, removed, changed = set(), set(), set()
        for step in episode["private"]["provenance"]["steps"]:
            before = {r["row_id"] for r in step["before"] if "row_id" in r}
            after = {r["row_id"] for r in step["after"] if "row_id" in r}
            added |= after - before
            removed |= before - after
            changed |= {rid for rid in after & before
                        if next(r for r in step["after"] if r.get("row_id") == rid) != pre[rid]}
        self.assertEqual(set(post) - set(pre), added)
        self.assertEqual(set(pre) - set(post), removed)
        self.assertEqual({rid for rid in set(pre) & set(post) if pre[rid] != post[rid]}, changed)


class OperatorTest(unittest.TestCase):
    def test_duplicate_is_a_copy_with_new_ids(self):
        for seed in range(50):
            episode = generate(seed, config(dup=3, currencies=["USD"]))
            invoices = by_id(episode["public"]["invoices"], "doc_id")
            originals = set()
            for issue in episode["private"]["truth"]["issues"]:
                a, b = (invoices[doc] for doc in issue["affected_ids"])
                self.assertNotEqual(a["invoice_id"], b["invoice_id"])
                self.assertNotEqual(a["issued_at"], b["issued_at"])
                self.assertEqual(invoice_signature(a), invoice_signature(b))
                self.assertEqual(issue["expected_amount_minor"], a["total_minor"])
                self.assertEqual(issue["currency"], a["currency"])
                self.assertEqual(issue["evidence_sets"], [issue["affected_ids"]])
                self.assertTrue(originals.isdisjoint(issue["affected_ids"]))  # one copy per original
                originals |= set(issue["affected_ids"])

    def test_cutoff_dates_cover_both_sides_and_the_edge(self):
        directions, near, on_edge = Counter(), 0, 0
        for seed in SEEDS:
            episode = generate(seed, config(cut=1, currencies=["USD"], timezone="-11:00"))
            policy = episode["public"]["policy"]
            step = episode["private"]["provenance"]["steps"][0]
            params = step["params"]
            directions[params["direction"]] += 1
            near += params["seconds_outside"] <= 6 * 3600
            on_edge += params["seconds_outside"] <= 1
            invoice = step["after"][0]
            self.assertFalse(in_period(invoice["issued_at"], policy["period_start"], policy["period_end"]))
            rows = [r for r in episode["public"]["ledger"] if r["invoice_id"] == invoice["invoice_id"]]
            self.assertEqual(len(rows), 2)  # unpaid: only its own posting pair
            for row in rows:
                self.assertTrue(in_period(row["timestamp"], policy["period_start"], policy["period_end"]))
            fx_dates = {r["date"] for r in episode["public"]["fx"]}
            self.assertIn(local_date(invoice["issued_at"], policy["timezone"]), fx_dates)
            issue = episode["private"]["truth"]["issues"][0]
            debit = next(r for r in rows if r["debit_minor"])
            credit = next(r for r in rows if r["credit_minor"])
            self.assertEqual(issue["evidence_sets"], [[invoice["doc_id"], debit["row_id"]],
                                                      [invoice["doc_id"], credit["row_id"]]])
            self.assertEqual(issue["expected_amount_minor"], invoice["total_minor"])
        self.assertGreater(directions["before_start"], 50)
        self.assertGreater(directions["after_end"], 50)
        self.assertGreater(near, 50)
        self.assertGreater(on_edge, 3)

    def test_foreign_cutoff_is_booked_at_the_new_date_rate(self):
        seen = 0
        for seed in SEEDS:
            episode = generate(seed, config(cut=3, currencies=["USD"]))
            for step in episode["private"]["provenance"]["steps"]:
                invoice = step["after"][0]
                if invoice["currency"] == "KZT":
                    continue
                seen += 1
                rate = next(r["rate"] for r in episode["public"]["fx"]
                            if r["date"] == local_date(invoice["issued_at"], "+05:00")
                            and r["source_currency"] == invoice["currency"])
                from decimal import Decimal
                from auditgym.money import convert
                amount = convert(invoice["total_minor"], Decimal(rate), 2, 2)
                self.assertEqual([r.get("debit_minor", 0) + r.get("credit_minor", 0)
                                  for r in step["after"][1:]], [amount, amount])
        self.assertGreater(seen, 20)

    def test_types_do_not_share_invoices_and_streams_are_independent(self):
        moved_same = 0
        for seed in range(100):
            alone = generate(seed, config(cut=2, currencies=["USD"]))
            both = generate(seed, config(dup=2, cut=2, currencies=["USD"]))
            issues = both["private"]["truth"]["issues"]
            dup_docs = {d for i in issues if i["type"] == "duplicate_invoice" for d in i["affected_ids"]}
            cut_docs = {d for i in issues if i["type"] == "period_cutoff" for d in i["affected_ids"]}
            self.assertFalse(dup_docs & cut_docs)
            if alone["private"]["provenance"]["attempt"] or both["private"]["provenance"]["attempt"]:
                continue

            def cutoff_keys(episode):
                back = {v: k for k, v in episode["private"]["provenance"]["id_map"].items()}
                return sorted(back[d] for i in episode["private"]["truth"]["issues"]
                              if i["type"] == "period_cutoff" for d in i["affected_ids"])
            moved_same += cutoff_keys(alone) == cutoff_keys(both)
        self.assertGreater(moved_same, 70)  # differs only when a duplicate took the same invoice

    def test_unimplemented_types_are_refused(self):
        for cfg in ({"errors": {"allocation_gap": 1}}, {"errors": {"fx_error": 1}},
                    {"errors": {}, "interaction": "I2"}):
            with self.assertRaises(ConfigError) as caught:
                generate(1, cfg)
            self.assertEqual(caught.exception.code, "not_implemented")

    def test_attempt_is_recorded_and_retries_are_deterministic(self):
        tight = {"errors": {"period_cutoff": 2}, "size": {"invoices": [10, 10], "background_pairs": [0, 5]}}
        attempts = Counter()
        for seed in SEEDS:
            episode = generate(seed, tight)
            self.assertEqual(verify(episode), [])
            self.assertEqual(len(episode["private"]["truth"]["issues"]), 2)
            attempts[episode["private"]["provenance"]["attempt"]] += 1
        self.assertGreater(sum(n for a, n in attempts.items() if a > 0), 0)  # the retry path is exercised
        seed = next(s for s in SEEDS if generate(s, tight)["private"]["provenance"]["attempt"] > 0)
        self.assertEqual(episode_files(generate(seed, tight)), episode_files(generate(seed, tight)))


class VerifyCatchesBrokenTruthTest(unittest.TestCase):
    def setUp(self):
        self.episode = generate(5, config(dup=1, cut=1, currencies=["USD"]))
        self.assertEqual(verify(self.episode), [])

    def broken(self, change):
        episode = copy.deepcopy(self.episode)
        change(episode)
        return verify(episode)

    def issue(self, episode, kind):
        return next(i for i in episode["private"]["truth"]["issues"] if i["type"] == kind)

    def test_wrong_amount(self):
        for kind in ("duplicate_invoice", "period_cutoff"):
            def change(e, kind=kind):
                self.issue(e, kind)["expected_amount_minor"] += 1
            self.assertTrue(any("deviation" in p for p in self.broken(change)), kind)

    def test_missing_evidence_id(self):
        def change(e):
            self.issue(e, "period_cutoff")["evidence_sets"][0][1] = "nosuchid00"
        self.assertTrue(self.broken(change))

    def test_too_much_evidence(self):
        def change(e):
            rows = [r["row_id"] for r in e["public"]["ledger"][:10]]
            self.issue(e, "duplicate_invoice")["evidence_sets"][0] += rows
        self.assertTrue(any("evidence" in p for p in self.broken(change)))

    def test_incomplete_cutoff_evidence(self):
        def change(e):
            issue = self.issue(e, "period_cutoff")
            issue["evidence_sets"] = issue["evidence_sets"][:1]
        self.assertTrue(self.broken(change))

    def test_unlisted_issue(self):
        def change(e):
            truth = e["private"]["truth"]
            truth["issues"] = [i for i in truth["issues"] if i["type"] != "period_cutoff"]
            e["private"]["provenance"]["steps"] = [s for s in e["private"]["provenance"]["steps"]
                                                   if s["operator"] != "period_cutoff"]
        problems = self.broken(change)
        self.assertTrue(any("outside the period" in p for p in problems))

    def test_pre_ledger_tampered(self):
        def change(e):
            e["private"]["pre_ledger"].pop()
        self.assertTrue(any("provenance" in p for p in self.broken(change)))

    def test_affected_ids_not_a_document(self):
        def change(e):
            self.issue(e, "duplicate_invoice")["affected_ids"][0] = e["public"]["ledger"][0]["row_id"]
        self.assertTrue(self.broken(change))


class FixtureTest(unittest.TestCase):
    def test_hand_written_fixture_passes_verify(self):
        import pathlib
        from auditgym.storage import load_episode
        root = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "tiny"
        self.assertEqual(verify(load_episode(root)), [])


class ReproducibilityTest(unittest.TestCase):
    CFG = config(dup=2, cut=2, currencies=["USD", "EUR"], timezone="+14:00")

    def test_same_seed_same_bytes(self):
        for seed in range(20):
            self.assertEqual(episode_files(generate(seed, self.CFG)), episode_files(generate(seed, self.CFG)))

    def canonical(self, episode):
        """Everything in internal keys, so two surface seeds can be compared."""
        back = {public: internal for internal, public in episode["private"]["provenance"]["id_map"].items()}

        def rename(value):
            if isinstance(value, dict):
                return {key: rename(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rename(item) for item in value]
            return back.get(value, value) if isinstance(value, str) else value

        out = {}
        for name in ("ledger", "invoices", "payments"):
            records = rename(copy.deepcopy(episode["public"][name]))
            for record in records:
                if record.get("invoice_refs"):
                    record["invoice_refs"].sort()
            out[name] = sorted(json.dumps(record, sort_keys=True) for record in records)
        out["pre_ledger"] = sorted(json.dumps(r, sort_keys=True) for r in rename(episode["private"]["pre_ledger"]))
        issues = rename(copy.deepcopy(episode["private"]["truth"]["issues"]))
        for issue in issues:
            issue["affected_ids"].sort()
            issue["evidence_sets"] = [sorted(group) for group in issue["evidence_sets"]]
        out["issues"] = issues
        steps = rename(copy.deepcopy(episode["private"]["provenance"]["steps"]))
        out["steps"] = json.dumps(steps, sort_keys=True)
        out["fx"] = episode["public"]["fx"]
        return out

    def test_surface_seed_renames_episode_and_truth(self):
        for seed in range(20):
            first = generate(seed, dict(self.CFG, surface_seed=1))
            second = generate(seed, dict(self.CFG, surface_seed=2))
            self.assertEqual(verify(second), [])
            self.assertNotEqual(first["private"]["truth"]["issues"][0]["affected_ids"],
                                second["private"]["truth"]["issues"][0]["affected_ids"])
            self.assertEqual(self.canonical(first), self.canonical(second))


class NotSeparableTest(unittest.TestCase):
    """Injected records look like any other record apart from the error itself."""

    def test_surface_features(self):
        ranks, line_counts_injected, line_counts_all = [], Counter(), Counter()
        for seed in SEEDS:
            episode = generate(seed, config(dup=1, cut=1, currencies=["USD"]))
            public = episode["public"]
            invoices = public["invoices"]
            # Order is by document date, then id: nothing else decides the position.
            self.assertEqual(invoices, sorted(invoices, key=lambda r: (r["issued_at"], r["doc_id"])))
            ledger = public["ledger"]
            self.assertEqual(ledger, sorted(ledger, key=lambda r: (r["timestamp"], r["row_id"])))
            ids = sorted(public_id for public_id in episode["private"]["provenance"]["id_map"].values())
            injected = set()
            for step in episode["private"]["provenance"]["steps"]:
                for record in step["after"]:
                    injected |= {record.get("doc_id"), record.get("invoice_id"), record.get("row_id")} - {None}
            for public_id in injected:
                ranks.append(ids.index(public_id) / (len(ids) - 1))
                self.assertEqual(len(public_id), 10)
            for invoice in invoices:
                line_counts_all[len(invoice["line_items"])] += 1
                if invoice["doc_id"] in injected:
                    line_counts_injected[len(invoice["line_items"])] += 1
                    # Same id shape, whole seconds, standard terms.
                    self.assertTrue(invoice["issued_at"][17:19].isdigit())
            for step in episode["private"]["provenance"]["steps"]:
                invoice = step["after"][0]
                from auditgym.money import parse_time
                days = (parse_time(invoice["due_at"]) - parse_time(invoice["issued_at"])).days
                self.assertIn(days, (14, 30, 45))
        # Token rank of injected ids among all ids is uniform: mean near 1/2, both halves used.
        mean = sum(ranks) / len(ranks)
        self.assertTrue(0.45 < mean < 0.55, mean)
        self.assertTrue(0.4 < sum(r < 0.5 for r in ranks) / len(ranks) < 0.6)
        # Injected invoices have the same spread of line-item counts as all invoices.
        self.assertEqual(set(line_counts_injected), set(line_counts_all))

    def test_injected_invoice_position_is_spread(self):
        positions = []
        for seed in SEEDS:
            episode = generate(seed, config(dup=1, currencies=["USD"]))
            copy_doc = episode["private"]["provenance"]["steps"][0]["after"][0]["doc_id"]
            docs = [inv["doc_id"] for inv in episode["public"]["invoices"]]
            positions.append(docs.index(copy_doc) / (len(docs) - 1))
        self.assertTrue(0.4 < sum(positions) / len(positions) < 0.6)
        self.assertGreater(sum(p < 0.25 for p in positions), 20)
        self.assertGreater(sum(p > 0.75 for p in positions), 20)


if __name__ == "__main__":
    unittest.main()
