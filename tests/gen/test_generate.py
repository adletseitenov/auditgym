import copy
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

from auditgym.generate import generate
from auditgym.money import local_date, parse_time, significant_digits
from auditgym.storage import episode_files, load_episode, write_episode
from auditgym.verify import invoice_signature, predicate_issues, verify

CLEAN = {"errors": {}}
GRID = [
    {"errors": {}, "currencies": []},
    {"errors": {}, "currencies": ["USD"]},
    {"errors": {}, "currencies": ["USD", "EUR", "CNY"], "timezone": "-11:00"},
    {"errors": {}, "currencies": ["RUB"], "timezone": "+14:00", "period": {"start": "2026-12-01", "months": 3}},
    {"errors": {}, "base_currency": "USD", "currencies": ["KZT", "EUR"], "timezone": "+05:30",
     "period": {"start": "2028-02-01", "months": 1}},
    {"errors": {}, "size": {"invoices": [10, 12], "background_pairs": [0, 5]}},
    {"errors": {}, "size": {"invoices": [200, 200], "background_pairs": [480, 490]}},
    {"errors": {}, "currencies": ["USD", "EUR"], "distractors": False, "timezone": "-03:00"},
]


def posted_docs(public):
    """doc_id -> posting moment of each posted document."""
    by_invoice = {r["invoice_id"]: r["timestamp"] for r in public["ledger"] if r["invoice_id"] and not r["payment_id"]}
    by_payment = {r["payment_id"]: r["timestamp"] for r in public["ledger"] if r["payment_id"]}
    out = {inv["doc_id"]: (inv["issued_at"], by_invoice[inv["invoice_id"]]) for inv in public["invoices"]
           if inv["invoice_id"] in by_invoice}
    out.update({pay["doc_id"]: (pay["timestamp"], by_payment[pay["payment_id"]]) for pay in public["payments"]
                if pay["payment_id"] in by_payment})
    return out


class CleanCompanyTest(unittest.TestCase):
    def test_books_are_well_formed_on_many_seeds(self):
        for index, config in enumerate(GRID):
            for seed in range(12):
                with self.subTest(config=index, seed=seed):
                    episode = generate(seed, config)
                    self.assertEqual(verify(episode), [])
                    self.assertEqual(episode["private"]["truth"]["issues"], [])
                    self.assertEqual(predicate_issues(episode["public"]), ([], []))

    def test_defaults_generate(self):
        for config in (None, {}):
            episode = generate(1, config)
            kinds = sorted(issue["type"] for issue in episode["private"]["truth"]["issues"])
            self.assertEqual(kinds, ["allocation_gap", "duplicate_invoice", "fx_error", "period_cutoff"])

    def test_sizes_follow_the_config(self):
        episode = generate(5, {"errors": {}, "size": {"invoices": [20, 25], "background_pairs": [40, 60]}})
        counts = episode["public"]["manifest"]["counts"]
        self.assertTrue(20 <= counts["invoices"] <= 25)
        self.assertLessEqual(counts["payments"], counts["invoices"])
        self.assertLessEqual(counts["ledger"], 2000)

    def test_ledger_is_in_the_period_and_in_the_base_currency(self):
        episode = generate(3, GRID[2])
        manifest = episode["public"]["manifest"]
        for row in episode["public"]["ledger"]:
            self.assertEqual(row["currency"], manifest["base_currency"])
            self.assertTrue(row["timestamp"].endswith("-11:00"))

    def test_no_two_invoices_share_a_signature(self):
        episode = generate(11, GRID[6])
        signatures = {invoice_signature(inv) for inv in episode["public"]["invoices"]}
        self.assertEqual(len(signatures), len(episode["public"]["invoices"]))

    def test_seed_and_config_do_not_appear_in_open_records(self):
        episode = generate(918273645, CLEAN)
        text = json.dumps(episode["public"])
        self.assertNotIn("918273645", text)
        self.assertNotIn("surface_seed", text)

    def test_policy_states_the_period_basis(self):
        self.assertEqual(generate(1, CLEAN)["public"]["policy"]["period_basis"], "document_date")

    def test_bad_seed(self):
        from auditgym.config import ConfigError
        for seed in ("1", 1.0, True, None):
            with self.assertRaises(ConfigError):
                generate(seed, CLEAN)

    def test_largest_config_is_fast(self):
        config = {"size": {"invoices": [200, 200], "background_pairs": [430, 430]}, "currencies": ["USD", "EUR", "CNY"],
                  "errors": {"duplicate_invoice": 6, "allocation_gap": 6, "fx_error": 6, "period_cutoff": 5},
                  "interaction": "I1", "period": {"start": "2026-12-01", "months": 3}}
        started = time.perf_counter()
        episode = generate(4, config)
        elapsed = time.perf_counter() - started
        self.assertLessEqual(episode["public"]["manifest"]["counts"]["ledger"], 2000)
        self.assertLess(elapsed, 2.0)


class RatesTest(unittest.TestCase):
    def test_six_significant_digits_and_cheap_currencies_move(self):
        episode = generate(2, {"errors": {}, "base_currency": "USD", "currencies": ["KZT", "RUB"], "distractors": False})
        by_currency = {}
        for row in episode["public"]["fx"]:
            self.assertGreaterEqual(significant_digits(row["rate"]), 6, row)
            by_currency.setdefault(row["source_currency"], []).append(row["rate"])
        for currency, rates in by_currency.items():
            self.assertTrue(all(a != b for a, b in zip(rates, rates[1:])), currency)  # no distractors: daily moves

    def test_weekends_repeat_friday_only_with_distractors(self):
        episode = generate(3, {"errors": {}, "currencies": ["EUR"]})
        rates = {row["date"]: row["rate"] for row in episode["public"]["fx"]}
        for date, rate in rates.items():
            day = dt.date.fromisoformat(date)
            friday = (day - dt.timedelta(days=day.weekday() - 4)).isoformat()
            if day.weekday() >= 5 and friday in rates:
                self.assertEqual(rate, rates[friday])


class TimingTest(unittest.TestCase):
    def test_postings_lag_inside_the_period(self):
        lags = []
        for seed in range(10):
            episode = generate(seed, {"errors": {}, "currencies": ["USD"]})
            policy = episode["public"]["policy"]
            end = parse_time(policy["period_end"])
            for doc, (dated, posted) in posted_docs(episode["public"]).items():
                lag = (parse_time(posted) - parse_time(dated)).total_seconds()
                self.assertGreaterEqual(lag, 60, doc)
                self.assertLess(parse_time(posted), end)
                lags.append(lag)
        self.assertTrue(any(lag < 3600 for lag in lags))
        self.assertTrue(any(lag > 86400 for lag in lags))
        self.assertTrue(any(lag > 7 * 86400 for lag in lags))

    def test_documents_in_other_offsets(self):
        crossing = other = 0
        for seed in range(10):
            public = generate(seed, CLEAN)["public"]
            offset = public["policy"]["timezone"]
            for doc in public["invoices"] + public["payments"]:
                stamp = doc.get("issued_at") or doc["timestamp"]
                if not stamp.endswith(offset):
                    other += 1
                    crossing += stamp[:10] != local_date(stamp, offset)
        self.assertGreater(other, 30)
        self.assertGreater(crossing, 5)


class VerifyCatchesBrokenBooksTest(unittest.TestCase):
    """verify() must be able to fail: break the books by hand and expect a problem."""

    def setUp(self):
        self.episode = generate(2, {"errors": {}, "currencies": ["USD"]})

    def broken(self, change):
        episode = copy.deepcopy(self.episode)
        change(episode)
        return verify(episode)

    def test_unbalanced_row(self):
        def change(e):
            row = e["public"]["ledger"][0]
            row["debit_minor" if row["debit_minor"] else "credit_minor"] += 1
        self.assertTrue(self.broken(change))

    def test_missing_link(self):
        def change(e):
            row = next(r for r in e["public"]["ledger"] if r["invoice_id"])
            row["invoice_id"] = "nosuchid00"
        self.assertTrue(self.broken(change))

    def test_wrong_amount_before_injection(self):
        def change(e):
            posted = {r["invoice_id"] for r in e["private"]["pre_ledger"] if r["invoice_id"]}
            invoice = next(i for i in e["private"]["pre_invoices"] if i["currency"] != "KZT" and i["invoice_id"] in posted)
            for row in e["private"]["pre_ledger"]:
                if row["invoice_id"] == invoice["invoice_id"] and row["payment_id"] is None:
                    row["debit_minor" if row["debit_minor"] else "credit_minor"] += 100
        self.assertTrue(any("policy amount" in p or "fx_error" in p for p in self.broken(change)))

    def test_row_outside_the_period(self):
        def change(e):
            e["public"]["ledger"][0]["timestamp"] = "2026-11-01T00:00:00+05:00"
        self.assertTrue(any("outside the period" in p for p in self.broken(change)))

    def test_posting_before_the_document(self):
        def change(e):
            start = parse_time(e["public"]["policy"]["period_start"])
            invoice = next(i for i in e["public"]["invoices"]
                           if parse_time(i["issued_at"]) - start > dt.timedelta(days=1))
            early = (parse_time(invoice["issued_at"]) - dt.timedelta(minutes=1)).astimezone(start.tzinfo).isoformat()
            for row in e["public"]["ledger"]:
                if row["invoice_id"] == invoice["invoice_id"] and row["payment_id"] is None:
                    row["timestamp"] = early
        self.assertTrue(any("posted before its date" in p for p in self.broken(change)))

    def test_times_compared_as_moments(self):
        def change(e):  # same moment written in another offset: still well formed
            invoice = next(i for i in e["public"]["invoices"] if i["issued_at"].endswith("+05:00"))
            invoice["issued_at"] = parse_time(invoice["issued_at"]).astimezone(dt.timezone.utc).isoformat()
            for record in e["private"]["pre_invoices"]:
                if record["doc_id"] == invoice["doc_id"]:
                    record["issued_at"] = invoice["issued_at"]
        self.assertEqual([p for p in self.broken(change) if "episode_id" not in p], [])

    def test_quantity_times_price(self):
        def change(e):
            e["public"]["invoices"][0]["line_items"][0]["quantity"] += 1
        self.assertTrue(any("unit price" in p for p in self.broken(change)))

    def test_due_before_issue(self):
        def change(e):
            e["public"]["invoices"][0]["due_at"] = e["public"]["invoices"][0]["issued_at"]
        self.assertTrue(any("due_at" in p for p in self.broken(change)))

    def test_missing_rate(self):
        def change(e):
            e["public"]["fx"].pop(len(e["public"]["fx"]) // 2)
        self.assertTrue(any("fx:" in p or "no rate" in p for p in self.broken(change)))

    def test_short_rate(self):
        def change(e):
            e["public"]["fx"][0]["rate"] = "466.2"
        self.assertTrue(any("significant digits" in p for p in self.broken(change)))

    def test_background_pair(self):
        def change(e):
            row = next(r for r in e["public"]["ledger"] if r["invoice_id"] is None and r["payment_id"] is None
                       and r["account"] == "cash")
            row["account"] = "revenue"
        self.assertTrue(any("background" in p for p in self.broken(change)))

    def test_posting_entity(self):
        def change(e):
            posted = {r["invoice_id"] for r in e["public"]["ledger"] if r["invoice_id"]}
            invoice = next(i for i in e["public"]["invoices"] if i["invoice_id"] in posted)
            other = next(i["entity_id"] for i in e["public"]["invoices"] if i["entity_id"] != invoice["entity_id"])
            for row in e["public"]["ledger"]:
                if row["invoice_id"] == invoice["invoice_id"] and row["payment_id"] is None:
                    row["entity_id"] = other
        self.assertTrue(any("another entity" in p for p in self.broken(change)))

    def test_fx_difference_on_a_base_payment(self):
        def change(e):
            base = e["public"]["policy"]["base_currency"]
            payment = next(p for p in e["public"]["payments"] if p["currency"] == base and p["invoice_refs"]
                           and any(r["payment_id"] == p["payment_id"] for r in e["public"]["ledger"]))
            rows = [r for r in e["public"]["ledger"] if r["payment_id"] == payment["payment_id"]]
            cash = next(r for r in rows if r["account"] == "cash")
            cash["debit_minor"] += 100
            e["public"]["ledger"].append(dict(cash, row_id="zzzzzzzzzz", account="fx_difference", debit_minor=0,
                                              credit_minor=100))
            e["public"]["manifest"]["counts"]["ledger"] += 1
        self.assertTrue(self.broken(change))

    def test_manifest_count(self):
        def change(e):
            e["public"]["manifest"]["counts"]["ledger"] += 1
        self.assertTrue(self.broken(change))

    def test_garbage_does_not_crash(self):
        def change(e):
            e["public"]["invoices"][0]["line_items"] = None
        self.assertTrue(self.broken(change))


class ReproducibilityTest(unittest.TestCase):
    def test_same_seed_same_bytes(self):
        config = GRID[2]
        self.assertEqual(episode_files(generate(7, config)), episode_files(generate(7, config)))

    def test_different_seeds_differ(self):
        self.assertNotEqual(episode_files(generate(7, CLEAN)), episode_files(generate(8, CLEAN)))

    def test_same_seed_other_config_is_another_world(self):
        first = generate(7, {"errors": {}, "currencies": ["USD"]})["public"]
        second = generate(7, {"errors": {}, "currencies": ["USD"], "interaction": "none",
                              "size": {"invoices": [40, 101]}})["public"]
        totals = sorted(i["total_minor"] for i in first["invoices"])
        self.assertNotEqual(totals, sorted(i["total_minor"] for i in second["invoices"]))

    def test_ids_depend_on_the_error_config(self):
        size = {"invoices": [24, 30], "background_pairs": [10, 20]}
        first = generate(7, {"errors": {"fx_error": 1}, "size": size})["private"]["provenance"]["id_map"]
        second = generate(7, {"errors": {"period_cutoff": 1}, "size": size})["private"]["provenance"]["id_map"]
        self.assertFalse(set(first.values()) & set(second.values()))

    def test_independent_of_process_environment(self):
        code = ("import hashlib, json; from auditgym.generate import generate; "
                "from auditgym.storage import episode_files; "
                "f = episode_files(generate(7, {'currencies': ['USD', 'EUR'], 'interaction': 'none'})); "
                "print(hashlib.sha256(json.dumps(f, sort_keys=True).encode()).hexdigest())")
        digests = set()
        for hash_seed, zone, locale in (("0", "UTC", "C"), ("4242", "Asia/Tokyo", "ru_RU.UTF-8")):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, TZ=zone, LC_ALL=locale)
            digests.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                       text=True, check=True).stdout)
        self.assertEqual(len(digests), 1)

    def test_written_episode_loads_back(self):
        episode = generate(4, GRID[1])
        with tempfile.TemporaryDirectory() as directory:
            write_episode(episode, directory)
            self.assertEqual(load_episode(directory), episode)


class SurfaceTest(unittest.TestCase):
    """A different surface seed gives the same episode under a renaming of ids."""

    def canonical(self, episode):
        back = {public: internal for internal, public in episode["private"]["provenance"]["id_map"].items()}

        def rename(value):
            if isinstance(value, dict):
                return {key: rename(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rename(item) for item in value]
            return back.get(value, value) if isinstance(value, str) else value

        tables = {}
        for name in ("ledger", "invoices", "payments"):
            records = rename(copy.deepcopy(episode["public"][name]))
            for record in records:
                if record.get("invoice_refs"):
                    record["invoice_refs"].sort()
            tables[name] = sorted(json.dumps(record, sort_keys=True) for record in records)
        tables["fx"] = episode["public"]["fx"]
        return tables

    def test_same_story_under_other_ids(self):
        first = generate(9, {"errors": {}, "surface_seed": 1})
        second = generate(9, {"errors": {}, "surface_seed": 2})
        self.assertNotEqual(first["public"]["ledger"][0]["row_id"], second["public"]["ledger"][0]["row_id"])
        self.assertEqual(self.canonical(first), self.canonical(second))
        self.assertEqual(verify(second), [])

    def test_ids_are_opaque_and_unique(self):
        episode = generate(9, CLEAN)
        ids = list(episode["private"]["provenance"]["id_map"].values())
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(len(i) == 10 and i.isalnum() and i.islower() for i in ids))

    def test_tables_are_ordered_by_public_id(self):
        """R6: position carries no date, so "first and last N records" says nothing."""
        public = generate(9, {"errors": {"period_cutoff": 2}})["public"]
        self.assertEqual([r["row_id"] for r in public["ledger"]], sorted(r["row_id"] for r in public["ledger"]))
        for name in ("invoices", "payments"):
            self.assertEqual([d["doc_id"] for d in public[name]], sorted(d["doc_id"] for d in public[name]))
        self.assertEqual([(r["date"], r["source_currency"]) for r in public["fx"]],
                         sorted((r["date"], r["source_currency"]) for r in public["fx"]))
        moments = [parse_time(i["issued_at"]) for i in public["invoices"]]
        self.assertNotEqual(moments, sorted(moments))


class SecretTest(unittest.TestCase):
    """L1: the evaluator's secret is mixed into the root stream; the default keeps reproducibility."""

    def test_secret_changes_everything_and_default_reproduces(self):
        plain = generate(11, CLEAN)
        self.assertEqual(episode_files(plain), episode_files(generate(11, CLEAN, secret="")))
        hidden = generate(11, CLEAN, secret="evaluator-only")
        self.assertEqual(verify(hidden), [])
        self.assertNotEqual(plain["public"]["fx"], hidden["public"]["fx"])
        self.assertNotEqual(plain["public"]["manifest"]["episode_id"], hidden["public"]["manifest"]["episode_id"])
        self.assertNotIn("evaluator-only", json.dumps(hidden))
        self.assertEqual(episode_files(hidden), episode_files(generate(11, CLEAN, secret="evaluator-only")))

    def test_first_rate_depends_on_the_whole_story(self):
        """The rates are keyed by a digest of the story: two stories from one stream differ in rates."""
        from auditgym.config import normalize_config
        from auditgym.rng import Rng
        from auditgym.world import build_world
        config = normalize_config({"errors": {}})
        first = build_world(Rng(5, "world"), config)
        second = build_world(Rng(5, "world"), dict(config, size={"invoices": [60, 60], "background_pairs": [150, 300]}))
        self.assertNotEqual(first["fx"][0]["rate"], second["fx"][0]["rate"])


class PolicyTest(unittest.TestCase):
    """Q2/S4, D2, D3, D6: policy.json is a whitelist and states the published rules."""

    def test_policy_is_the_whitelist_and_states_the_rules(self):
        from auditgym.generate import POLICY_KEYS
        policy = generate(1, CLEAN)["public"]["policy"]
        self.assertEqual(tuple(policy), POLICY_KEYS)
        self.assertEqual(policy["period_interval"], "[start, end)")
        self.assertIn("booked", policy["payment_rule"])
        self.assertNotIn("\n", policy["payment_rule"])
        self.assertEqual(policy["duplicate_signature"], "entity, currency, total, line items incl. service period")

    def test_no_truth_derived_value_in_public_files(self):
        """Episodes that differ only in errors and interaction expose the same policy, the same manifest
        keys, no error type outside the fixed tolerance table, and no scenario name."""
        configs = [{"errors": {}}, {"errors": {"fx_error": 3}}, {"errors": {"period_cutoff": 2, "duplicate_invoice": 1}},
                   {"interaction": "I1"}, {"interaction": "I2", "errors": {"allocation_gap": 2}}]
        policies, manifests = set(), set()
        for config in configs:
            config = dict(config, size={"invoices": [48, 60], "background_pairs": [10, 20]}, currencies=["USD"])
            episode = generate(3, config)
            files = episode_files(episode)
            policy = json.loads(files["public/policy.json"])
            policies.add(json.dumps(policy, sort_keys=True))
            manifest = json.loads(files["public/manifest.json"])
            manifests.add(tuple(manifest))
            public_text = "".join(text for name, text in files.items() if name.startswith("public/")
                                  and name != "public/policy.json")
            for word in ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff", '"I1"', '"I2"',
                         "scenario", "issue_id", "errors", "interaction"):
                self.assertNotIn(word, public_text, (config, word))
            issues = len(episode["private"]["truth"]["issues"])
            self.assertNotIn(issues, [value for value in manifest.values() if isinstance(value, int)])
        self.assertEqual(len(policies), 1)
        self.assertEqual(len(manifests), 1)


class PaymentRuleTest(unittest.TestCase):
    """D2: receivables close in proportion at the invoice's booked rate; every foreign payment carries its
    own fx_difference row."""

    def test_closings_follow_the_published_rule(self):
        checked_partial = foreign_fx = 0
        for seed in range(30):
            public = generate(seed, {"errors": {}, "currencies": ["USD", "EUR"]})["public"]
            base = public["policy"]["base_currency"]
            invoices = {i["invoice_id"]: i for i in public["invoices"]}
            ledger = public["ledger"]
            booked = {r["invoice_id"]: r["debit_minor"] for r in ledger
                      if r["invoice_id"] and not r["payment_id"] and r["account"] == "receivable"}
            for payment in public["payments"]:
                rows = [r for r in ledger if r["payment_id"] == payment["payment_id"]]
                if not rows or not payment["invoice_refs"] or len(payment["invoice_refs"]) != 1:
                    continue
                invoice = invoices[payment["invoice_refs"][0]]
                closing = sum(r["credit_minor"] - r["debit_minor"] for r in rows if r["account"] == "receivable")
                cash = sum(r["debit_minor"] - r["credit_minor"] for r in rows if r["account"] == "cash")
                unallocated = sum(r["credit_minor"] for r in rows if r["account"] == "unallocated")
                fx_rows = [r for r in rows if r["account"] == "fx_difference"]
                earlier = [p for p in public["payments"] if p["invoice_refs"] == payment["invoice_refs"]
                           and parse_time(p["timestamp"]) < parse_time(payment["timestamp"])]
                if 0 < payment["amount_minor"] < invoice["total_minor"] and payment["currency"] != base and not earlier:
                    # The rule written out for a first partial payment: booked * amount / total, half-even.
                    total = invoice["total_minor"]
                    whole, rest = divmod(booked[invoice["invoice_id"]] * payment["amount_minor"], total)
                    expected = whole + (1 if 2 * rest > total or (2 * rest == total and whole % 2) else 0)
                    self.assertEqual(closing, expected, payment["doc_id"])
                    checked_partial += 1
                if payment["currency"] != base and cash != closing + unallocated:
                    self.assertEqual(len(fx_rows), 1)
                    foreign_fx += 1
                if payment["currency"] == base:
                    self.assertEqual(fx_rows, [])
        self.assertGreater(checked_partial, 5)
        self.assertGreater(foreign_fx, 20)


class SurfaceRegularityTest(unittest.TestCase):
    """D8/L6: no cheap surface rule tells injected records apart."""

    def test_legitimate_postings_lag_by_weeks(self):
        over_four_days = 0
        for seed in range(10):
            for dated, posted in posted_docs(generate(seed, CLEAN)["public"]).values():
                over_four_days += parse_time(posted) - parse_time(dated) > dt.timedelta(days=4)
        self.assertGreater(over_four_days, 30)

    def test_late_posting_does_not_find_cutoffs(self):
        """Rule "posted more than 4 days after the document": its precision on period_cutoff is low."""
        flagged = hits = 0
        for seed in range(60):
            episode = generate(seed, {"errors": {"period_cutoff": 2}, "currencies": ["USD"]})
            cut = {d for i in episode["private"]["truth"]["issues"] for d in i["affected_ids"]}
            for doc, (dated, posted) in posted_docs(episode["public"]).items():
                if parse_time(posted) - parse_time(dated) > dt.timedelta(days=4):
                    flagged += 1
                    hits += doc in cut
        self.assertLess(hits / flagged, 0.25, (hits, flagged))

    def test_overpayment_and_gap_amounts_look_alike(self):
        def zeros(value):
            count = 0
            while value and value % 10 == 0:
                value, count = value // 10, count + 1
            return count

        over, gap = [], []
        for seed in range(150):
            episode = generate(seed, {"errors": {"allocation_gap": 2}, "currencies": ["USD"]})
            gaps = {i["expected_amount_minor"] for i in episode["private"]["truth"]["issues"]}
            payments = {p["payment_id"]: p for p in episode["public"]["payments"]}
            for row in episode["public"]["ledger"]:
                if row["account"] == "unallocated" and payments[row["payment_id"]]["invoice_refs"]:
                    (gap if row["credit_minor"] in gaps else over).append(zeros(row["credit_minor"]))
        self.assertGreater(len(over), 30)
        mean_over, mean_gap = sum(over) / len(over), sum(gap) / len(gap)
        self.assertLess(abs(mean_over - mean_gap), 0.5, (mean_over, mean_gap))
        self.assertLess(abs(sum(z >= 2 for z in over) / len(over) - sum(z >= 2 for z in gap) / len(gap)), 0.15)


if __name__ == "__main__":
    unittest.main()
