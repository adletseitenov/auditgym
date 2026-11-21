"""tools/leakscan.py finds what it claims to find, and the current environment passes it."""
import io
import json
import time
import unittest

from auditgym.generate import generate
from tools import leakscan

CONFIG = {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}}


class ScannerSensitivityTest(unittest.TestCase):
    def setUp(self):
        self.episode = generate(7_123_457, CONFIG)
        self.secret = leakscan.secrets(self.episode, ("/somewhere/leakscan_episode_x",))

    def scan(self, value) -> list:
        text = json.dumps(value)
        return leakscan.scan_text(text, self.secret, json.loads(text))

    def test_clean_reply_passes(self):
        self.assertEqual(self.scan({"observation": {"type": "final", "steps_remaining": 3}}), [])

    def test_each_kind_of_leak_is_found(self):
        issue_id = self.episode["private"]["truth"]["issues"][0]["issue_id"]
        internal = sorted(self.episode["private"]["provenance"]["id_map"])[0]
        cases = [{"x": issue_id}, {"x": internal}, {"tp": 1}, {"evidence_sets": []}, {"note": "see truth.json"},
                 {"n": 7123457}, {"p": "/somewhere/leakscan_episode_x/public"}, {"e": "Traceback (most recent"},
                 {"t": int(time.time())}]
        for case in cases:
            self.assertTrue(self.scan(case), f"not found: {case}")

    def test_a_public_amount_near_today_is_not_a_clock(self):
        """Amounts are 10-digit integers too; one that is a public record value and lies within ten
        days of today's epoch seconds (1794500000 on 19.11.2026 did) is data. The clock is still found."""
        now = int(time.time())
        self.secret["public_ints"] = set(self.secret["public_ints"]) | {now - 86400}
        self.assertEqual(self.scan({"amount_minor": now - 86400}), [])
        self.assertTrue(self.scan({"amount_minor": now - 2 * 86400}))

    def test_a_served_amount_equal_to_the_clock_is_data(self):
        """A record amount that equals int(time.time()) (or its millisecond form) is served data: the
        scan must not call it the clock, whatever the date the check is run on."""
        now = int(time.time())
        episode = generate(7_123_457, CONFIG)
        episode["public"]["ledger"][0]["debit_minor"] = now
        episode["public"]["ledger"][1]["credit_minor"] = now * 1000
        episode["public"]["invoices"][0]["doc_id"] = str(now + 5)  # an all-digit id is data too
        secret = leakscan.secrets(episode, ())
        for value in (now, now * 1000, now + 5):
            text = json.dumps({"observation": {"records": [{"debit_minor": value}]}})
            self.assertEqual(leakscan.scan_text(text, secret, json.loads(text)), [], value)

    def test_digits_inside_an_opaque_token_are_not_a_clock(self):
        """Cursors and ids are hashes: a run of digits in one that lies near today's epoch seconds is
        chance (the window moves with the calendar), not a leak."""
        from unittest import mock

        cursor = "3a9d28a51dc388016583"
        text = json.dumps({"observation": {"next_cursor": cursor, "steps_remaining": 9}})
        with mock.patch.object(leakscan.time, "time", return_value=388016583 + 3600):
            self.assertTrue(leakscan._near_now(388016583))
            self.assertEqual(leakscan.scan_text(text, self.secret, json.loads(text)), [])

    def test_a_real_clock_value_is_still_found(self):
        """Not served data, not inside a token: a bare number, one in a message, in milliseconds."""
        now = int(time.time())
        cases = [{"t": now}, {"t": now * 1000}, {"message": f"retry after {now} s"},
                 {"message": f"at {now}."}, {"t": [now - 3600]}]
        for case in cases:
            self.assertTrue(any("epoch-like" in problem for problem in self.scan(case)), case)
        problems = leakscan.scan_text(f"started {now}\n", self.secret)  # stderr / client text, unparsed
        self.assertTrue(any("epoch-like" in problem for problem in problems), problems)

    def test_pre_injection_values_are_found(self):
        if not self.secret["strings"] and not self.secret["ints"]:
            self.skipTest("this episode has no pre-injection-only values")
        value = sorted(self.secret["strings"])[0] if self.secret["strings"] else sorted(self.secret["ints"])[0]
        self.assertTrue(self.scan({"x": value}))

    def test_shape_rejects_verdicts_and_eval_rewards(self):
        final = {"type": "final", "steps_remaining": 1}
        info = {"reward": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "inspection_steps": 0}
        good = {"observation": final, "reward": 1.0, "terminated": True, "truncated": False, "info": info}
        self.assertEqual(leakscan.check_shape(good, "step", "train"), [])
        self.assertTrue(leakscan.check_shape(dict(good, info=dict(info, tp=1)), "step", "train"))
        self.assertTrue(leakscan.check_shape(good, "step", "eval"))
        error = {"observation": {"type": "error", "error": {"code": "unknown_id", "message": "No such id x1."},
                                 "steps_remaining": 5}, "reward": 0.0, "terminated": False, "truncated": False,
                 "info": {}}
        self.assertTrue(leakscan.check_shape(error, "step", "train"))

    def test_auc(self):
        self.assertEqual(leakscan.auc([2, 3], [0, 1]), 1.0)
        self.assertEqual(leakscan.auc([0, 1], [2, 3]), 0.0)
        self.assertEqual(leakscan.auc([1, 1], [1, 1]), 0.5)

    def test_surface_statistics_flag_a_planted_separator(self):
        episodes = []
        for seed in range(25):
            episode = generate(seed, {"errors": {"duplicate_invoice": 1}})
            affected = set(episode["private"]["truth"]["issues"][0]["affected_ids"])
            for invoice in episode["public"]["invoices"]:
                count = 7 if invoice["doc_id"] in affected else 1
                invoice["line_items"] = invoice["line_items"][:1] * count
            episodes.append(episode)
        rows = {(r["type"], r["feature"]): r for r in leakscan.surface_statistics(episodes)}
        self.assertEqual(rows[("duplicate_invoice", "line_items")]["status"], "FAIL")
        self.assertEqual(rows[("duplicate_invoice", "id_rank")]["status"], "ok")


class SignificanceTest(unittest.TestCase):
    """Q6: a row fails when it is large AND significant, whatever the sample size."""

    def test_perfect_separation_of_a_few_samples_is_significant(self):
        stat = leakscan.separation(list(range(1000, 1010)), list(range(0, 200)))
        self.assertEqual(stat["auc"], 1.0)
        self.assertLess(stat["p"], 1e-6)
        self.assertTrue(leakscan.verdict(stat, tests=300))

    def test_a_small_effect_is_not_a_finding_however_many_samples(self):
        from auditgym.rng import Rng

        rng = Rng(5, "small-effect")
        negative = [rng.below(1000) for _ in range(4000)]
        positive = [rng.below(1000) + 100 for _ in range(4000)]  # AUC about 0.55: visible, tiny
        stat = leakscan.separation(positive, negative)
        self.assertGreater(abs(stat["auc"] - 0.5), 0.03)
        self.assertLess(stat["p"], 1e-6)  # significant ...
        self.assertFalse(leakscan.verdict(stat, tests=100))  # ... but below MIN_EFFECT

    def test_noise_of_a_large_effect_size_is_not_significant(self):
        stat = leakscan.separation([1, 5, 2, 9, 4, 8, 3, 7], [2, 6, 1, 9, 5, 3, 8, 4, 6, 2, 7, 1, 9, 3])
        self.assertFalse(leakscan.verdict(stat, tests=1))

    def test_bonferroni_correction(self):
        stat = {"auc": 0.8, "p": 1e-4}
        self.assertTrue(leakscan.verdict(stat, tests=10))     # 1e-4 < 0.01 / 10
        self.assertFalse(leakscan.verdict(stat, tests=1000))  # 1e-4 > 0.01 / 1000

    def test_ties_and_empty_sides(self):
        stat = leakscan.separation([3] * 10, [3] * 30)
        self.assertEqual((stat["auc"], stat["z"], stat["p"]), (0.5, 0.0, 1.0))
        self.assertIsNone(leakscan.separation([], [1]))
        self.assertIsNone(leakscan.separation([1], []))

    def test_z_matches_the_exact_distribution_for_a_small_case(self):
        # n1 = 5, n0 = 5, perfect separation: exact two-sided p = 2 / C(10, 5) = 0.0079; the normal
        # approximation says 0.0090, the same order and on the right side of 0.05.
        stat = leakscan.separation([6, 7, 8, 9, 10], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(stat["p"], 0.0090, places=3)
        self.assertGreater(stat["p"], 2 / 252)

    def test_doc_id_that_starts_with_z_is_found_with_ten_affected_documents(self):
        """The old rule needed 20 affected documents; ten affected ids all sorting last are no chance."""
        episodes = []
        for seed in range(5):
            episode = generate(seed, {"errors": {"duplicate_invoice": 1}})
            truth = episode["private"]["truth"]["issues"][0]
            rename = {old: "z" + old[1:] for old in truth["affected_ids"]}
            for table in ("invoices", "payments"):
                for record in episode["public"][table]:
                    record["doc_id"] = rename.get(record["doc_id"], record["doc_id"])
            truth["affected_ids"] = [rename[i] for i in truth["affected_ids"]]
            truth["evidence_sets"] = [[rename.get(i, i) for i in group] for group in truth["evidence_sets"]]
            episodes.append(episode)
        rows = {(r["type"], r["table"], r["feature"]): r for r in leakscan.surface_statistics(episodes)}
        row = rows[("duplicate_invoice", "invoices", "id_rank")]
        self.assertEqual((row["affected"], row["status"]), (10, "FAIL"))


class ViewsTest(unittest.TestCase):
    """Q6: the ledger and the duplicate pairs are scanned as well as the documents."""

    DENSE = {"errors": {"duplicate_invoice": 3, "allocation_gap": 2, "fx_error": 2, "period_cutoff": 2},
             "size": {"invoices": [80, 100]}, "currencies": ["USD", "EUR"]}

    def episodes(self, count=12):
        return [generate(900 + k, self.DENSE) for k in range(count)]

    def test_the_three_views_produce_rows(self):
        rows = leakscan.surface_statistics(self.episodes())
        tables = {row["table"] for row in rows}
        self.assertTrue({"invoices", "payments", "pairs"} <= tables, tables)
        self.assertTrue(any(table.startswith("ledger:") for table in tables), tables)
        pair_features = {row["feature"] for row in rows if row["table"] == "pairs"}
        self.assertTrue({"gap_days", "id_order", "same_items", "later_paid"} <= pair_features, pair_features)
        self.assertTrue(all(row["status"] in ("ok", "few", "explained") or row["status"].startswith("pending")
                            for row in rows), [r for r in rows if r["status"] == "FAIL"])

    def test_a_planted_ledger_separator_is_found(self):
        episodes = self.episodes()
        for episode in episodes:
            cited = {i for issue in episode["private"]["truth"]["issues"] if issue["type"] == "duplicate_invoice"
                     for group in issue["evidence_sets"] for i in group}
            for row in episode["public"]["ledger"]:
                if row["row_id"] in cited:
                    row["timestamp"] = row["timestamp"][:17] + "59" + row["timestamp"][19:]  # :59 seconds
        rows = [r for r in leakscan.surface_statistics(episodes)
                if r["type"] == "duplicate_invoice" and r["table"].startswith("ledger:") and r["feature"] == "second"]
        self.assertTrue(any(r["status"] == "FAIL" for r in rows), rows)

    def test_a_planted_pair_separator_is_found(self):
        episodes = self.episodes()
        for episode in episodes:
            late = {sorted(issue["affected_ids"])[-1] for issue in episode["private"]["truth"]["issues"]
                    if issue["type"] == "duplicate_invoice"}
            for invoice in episode["public"]["invoices"]:
                if invoice["doc_id"] in late:
                    invoice["line_items"] = [dict(item, description="copy") for item in invoice["line_items"]]
        rows = {r["feature"]: r for r in leakscan.surface_statistics(episodes) if r["table"] == "pairs"}
        self.assertEqual(rows["same_items"]["status"], "FAIL", rows["same_items"])

    def test_the_ledger_view_compares_like_with_like(self):
        """Amounts of converted documents are not round: the stratum includes the currency kind, so a
        foreign-currency evidence row is not compared with a base-currency posting."""
        episode = generate(900, self.DENSE)
        strata = {stratum for stratum, _ in leakscan.ledger_features(episode["public"]).values()}
        self.assertTrue(any(stratum.endswith("/fx") for stratum in strata), strata)
        self.assertTrue(any(stratum.endswith("/base") for stratum in strata), strata)

    def test_pair_features_describe_the_pair_without_the_truth(self):
        episode = generate(900, self.DENSE)
        pairs = leakscan.pair_features(episode["public"])
        duplicates = {frozenset(i["affected_ids"]) for i in episode["private"]["truth"]["issues"]
                      if i["type"] == "duplicate_invoice"}
        self.assertTrue(duplicates <= set(pairs))
        for values in pairs.values():
            self.assertGreaterEqual(values["gap_days"], 0)


class LeakscanRunTest(unittest.TestCase):
    def test_two_episodes_pass(self):
        out = io.StringIO()
        summary = leakscan.run(2, out=out)
        self.assertEqual(summary["failures"], [], out.getvalue())
        # A pending defect is allowed only with its docs/AUDIT.md number, and it is not a pass.
        for key, reason in leakscan.PENDING.items():
            self.assertRegex(reason, r"^pending: AUDIT [A-Z]\d+", key)
        self.assertTrue(all(item.startswith("pending: AUDIT") for item in summary["pending"]), summary["pending"])

    def test_strict_fails_while_something_is_pending(self):
        from unittest import mock

        quiet = {"failures": [], "pending": ["pending: AUDIT L6 [x]"]}
        with mock.patch.object(leakscan, "run", return_value=quiet):
            self.assertEqual(leakscan.main(["--episodes", "1"]), 0)
            self.assertEqual(leakscan.main(["--episodes", "1", "--strict"]), 1)
        with mock.patch.object(leakscan, "run", return_value={"failures": [], "pending": []}):
            self.assertEqual(leakscan.main(["--episodes", "1", "--strict"]), 0)


if __name__ == "__main__":
    unittest.main()
