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


class LeakscanRunTest(unittest.TestCase):
    def test_two_episodes_pass(self):
        out = io.StringIO()
        summary = leakscan.run(2, out=out)
        self.assertEqual(summary["failures"], [], out.getvalue())
        # C1 and C2 are fixed in the generator (DECISIONS C1, C2): nothing is waved through as pending.
        self.assertEqual(leakscan.PENDING, {})
        self.assertEqual(summary["pending"], [])


if __name__ == "__main__":
    unittest.main()
