"""Decoys against shortcuts (docs/DESIGN.md 3.1): resent unposted copies (A) and invoices of the period
for a neighbouring period's service (B). They come at the expected rate, only with distractors,
never reach the truth as A (B is an ordinary target of every operator but period_cutoff), are
checked by verify, and defeat the shortcuts measured by tools/tricks.py."""
import copy
import datetime as dt
import unittest
from collections import Counter, defaultdict

from auditgym import levels
from auditgym.config import normalize_config
from auditgym.generate import generate
from auditgym.money import parse_offset, parse_time
from auditgym.rng import Rng
from auditgym.verify import invoice_signature, verify
from auditgym.world import RESENT_COPIES, SHIFTED_SERVICE, build_world
from auditgym.world import invoice_signature as world_signature
from tools import tricks

SEEDS = range(90_000, 90_150)


def worlds(config: dict, seeds=SEEDS):
    normalized = normalize_config(config)
    for seed in seeds:
        yield build_world(Rng(seed, "decoy-test").child("world"), normalized)


def public_decoys(public: dict) -> tuple[set, set, set]:
    """From the open records only: (resent copies, their sources, invoices of the period for another
    period's service), as doc_ids."""
    policy = public["policy"]
    start, end = parse_time(policy["period_start"]), parse_time(policy["period_end"])
    zone = parse_offset(policy["timezone"])
    first = start.astimezone(zone).date().isoformat()
    last = (end - dt.timedelta(seconds=1)).astimezone(zone).date().isoformat()
    posted = {row["invoice_id"] for row in public["ledger"] if row["payment_id"] is None and row["invoice_id"]}
    groups = defaultdict(list)
    for invoice in public["invoices"]:
        groups[invoice_signature(invoice)].append(invoice)
    copies, sources = set(), set()
    for group in groups.values():
        unposted = [inv for inv in group if inv["invoice_id"] not in posted]
        if len(group) == 2 and len(unposted) == 1:
            copies.add(unposted[0]["doc_id"])
            sources.update(inv["doc_id"] for inv in group if inv is not unposted[0])
    shifted = {inv["doc_id"] for inv in public["invoices"]
               if start <= parse_time(inv["issued_at"]) < end
               and all(i["service_end"] < first or i["service_start"] > last for i in inv["line_items"])}
    return copies, sources, shifted


def _with_decoys(seed_from: int = 90_000) -> dict:
    for seed in range(seed_from, seed_from + 50):
        episode = generate(seed, {})
        copies, _, shifted = public_decoys(episode["public"])
        if copies and shifted:
            return episode
    raise AssertionError("no default episode with both decoys")


class FrequencyTest(unittest.TestCase):
    def test_counts_per_story_with_distractors(self):
        resent, shifted = Counter(), Counter()
        for world in worlds({}):
            kinds = Counter(inv.get("decoy") for inv in world["invoices"])
            resent[kinds["resent_copy"]] += 1
            shifted[kinds["shifted_service"]] += 1
            self.assertEqual(kinds["resent_copy"], sum(1 for inv in world["invoices"] if inv.get("has_copy")))
        self.assertEqual(set(resent), set(range(RESENT_COPIES[0], RESENT_COPIES[1] + 1)))
        self.assertEqual(set(shifted), set(range(SHIFTED_SERVICE[0], SHIFTED_SERVICE[1] + 1)))
        n = len(SEEDS)
        self.assertAlmostEqual(sum(k * v for k, v in resent.items()) / n, 1.5, delta=0.15)
        self.assertAlmostEqual(sum(k * v for k, v in shifted.items()) / n, 2.0, delta=0.2)

    def test_shape_of_each_decoy(self):
        for world in worlds({}, range(90_000, 90_040)):
            start, end = world["period_start"], world["period_end"]
            by_signature = defaultdict(list)
            for invoice in world["invoices"]:
                by_signature[world_signature(invoice)].append(invoice)
            for invoice in world["invoices"]:
                if invoice.get("decoy") == "resent_copy":
                    twins = by_signature[world_signature(invoice)]
                    self.assertEqual(len(twins), 2)
                    source = next(inv for inv in twins if inv is not invoice)
                    self.assertTrue(source.get("has_copy"))
                    self.assertIsNotNone(source["posted_at"])
                    self.assertIsNone(invoice["posted_at"])
                    self.assertTrue(end <= invoice["issued_at"] < end + dt.timedelta(days=4))
                    self.assertFalse(any(invoice["key"] in (p["invoice_refs"] or []) for p in world["payments"]))
                elif invoice.get("decoy") == "shifted_service":
                    self.assertTrue(start <= invoice["issued_at"] < end)
                    self.assertTrue(start <= invoice["posted_at"] < end)
                    self.assertTrue(all(i["service_start"] != world["period_start"].date().isoformat()
                                        for i in invoice["line_items"]))

    def test_no_decoys_without_distractors(self):
        for world in worlds({"distractors": False}, range(90_000, 90_030)):
            self.assertFalse(any(inv.get("decoy") or inv.get("has_copy") for inv in world["invoices"]))

    def test_smallest_company_still_has_room(self):
        config = {"size": {"invoices": [10, 10], "background_pairs": [100, 120]},
                  "errors": {"period_cutoff": 1}, "currencies": []}
        for seed in range(30):
            episode = generate(seed, config)
            self.assertEqual(len(episode["public"]["invoices"]), 10)

    def test_same_seed_same_decoys(self):
        first = [(inv["key"], inv.get("decoy"), inv["issued_at"]) for inv in next(worlds({}, [90_001]))["invoices"]]
        again = [(inv["key"], inv.get("decoy"), inv["issued_at"]) for inv in next(worlds({}, [90_001]))["invoices"]]
        self.assertEqual(first, again)


class TruthTest(unittest.TestCase):
    def test_resent_copies_never_reach_the_truth_and_shifted_invoices_are_ordinary_targets(self):
        """A resent copy (A) and a duplicate of its source never appear in the truth. An invoice for a
        neighbouring period's service (B) carries errors about as often as any posted invoice of the
        period: excluding it would make it a guaranteed clean document (a shortcut of its own)."""
        configs = [{}, {"interaction": "I1"}, {"interaction": "I2"}, levels.load("max"), levels.load("hard")]
        seen, types = Counter(), Counter()
        for config in configs:
            for seed in range(90_000, 90_040):
                episode = generate(seed, config)
                public = episode["public"]
                copies, sources, shifted = public_decoys(public)
                start, end = (parse_time(public["policy"][k]) for k in ("period_start", "period_end"))
                posted = {r["invoice_id"] for r in public["ledger"] if r["invoice_id"] and not r["payment_id"]}
                regular = {inv["doc_id"] for inv in public["invoices"] if inv["invoice_id"] in posted
                           and start <= parse_time(inv["issued_at"]) < end} - shifted
                hit = set()
                for issue in episode["private"]["truth"]["issues"]:
                    affected = set(issue["affected_ids"])
                    self.assertFalse(affected & copies, (seed, issue["type"]))
                    if issue["type"] == "duplicate_invoice":
                        self.assertFalse(affected & sources, seed)
                    self.assertFalse(issue["type"] == "period_cutoff" and affected & shifted, seed)
                    types.update(issue["type"] for _ in affected & shifted)
                    hit |= affected
                seen.update(copies=len(copies), shifted=len(shifted), shifted_hit=len(shifted & hit),
                            regular=len(regular), regular_hit=len(regular & hit))
        self.assertGreater(seen["copies"], 100)
        self.assertGreater(seen["shifted"], 150)
        shifted_share = seen["shifted_hit"] / seen["shifted"]
        regular_share = seen["regular_hit"] / seen["regular"]
        self.assertGreater(seen["shifted_hit"], 20)
        self.assertLess(abs(shifted_share - regular_share), 0.06, (shifted_share, regular_share))
        self.assertGreaterEqual(len(types), 2, types)


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.episode = copy.deepcopy(_with_decoys())
        self.public = self.episode["public"]
        self.copies, self.sources, self.shifted = public_decoys(self.public)
        self.assertEqual(verify(self.episode), [])

    def _count(self):
        for table in ("ledger", "invoices", "payments"):
            self.public["manifest"]["counts"][table] = len(self.public[table])

    def test_posting_a_resent_copy_is_caught(self):
        doc = sorted(self.copies)[0]
        invoice = next(inv for inv in self.public["invoices"] if inv["doc_id"] == doc)
        moment = (parse_time(self.public["policy"]["period_end"]) - dt.timedelta(hours=1)).isoformat()
        for suffix, account, debit, credit in (("dr", "receivable", 1, 0), ("cr", "revenue", 0, 1)):
            amount = invoice["total_minor"]
            self.public["ledger"].append({
                "row_id": f"forged{suffix}", "timestamp": moment, "account": account,
                "debit_minor": amount * debit, "credit_minor": amount * credit,
                "currency": self.public["policy"]["base_currency"], "entity_id": invoice["entity_id"],
                "invoice_id": invoice["invoice_id"], "payment_id": None})
        self._count()
        problems = verify(self.episode)
        self.assertTrue(any("outside the period but posted" in p for p in problems), problems)

    def test_posting_a_copy_is_an_issue_the_truth_must_list(self):
        from auditgym.verify import predicate_issues

        doc = sorted(self.copies)[0]
        invoice = next(inv for inv in self.public["invoices"] if inv["doc_id"] == doc)
        source_doc = next(d for d in self.sources if any(
            inv["doc_id"] == d and invoice_signature(inv) == invoice_signature(invoice) for inv in self.public["invoices"]))
        source = next(inv for inv in self.public["invoices"] if inv["doc_id"] == source_doc)
        rows = [dict(r, row_id="forged" + r["row_id"], invoice_id=invoice["invoice_id"])
                for r in self.public["ledger"] if r["invoice_id"] == source["invoice_id"] and r["payment_id"] is None]
        self.public["ledger"].extend(rows)
        found, _ = predicate_issues(self.public)
        targets = {(i["type"], tuple(i["affected_ids"])) for i in found}
        self.assertIn(("duplicate_invoice", tuple(sorted([doc, source_doc]))), targets)
        self.assertIn(("period_cutoff", (doc,)), targets)
        self._count()
        self.assertTrue(verify(self.episode))
        # Even with the postings shaped right, the truth would have to list both new issues.
        from auditgym.verify import _check_truth

        problems = _check_truth(self.public, self.episode["private"])
        for kind in ("duplicate_invoice", "period_cutoff"):
            self.assertTrue(any(f"the {kind} definition holds" in p and "no issue lists it" in p for p in problems),
                            problems)

    def test_paying_a_resent_copy_is_caught(self):
        doc = sorted(self.copies)[0]
        invoice = next(inv for inv in self.public["invoices"] if inv["doc_id"] == doc)
        moment = (parse_time(invoice["issued_at"]) + dt.timedelta(hours=2)).isoformat()
        self.public["payments"].append({"doc_id": "forgedpaydoc", "payment_id": "forgedpay", "timestamp": moment,
                                        "currency": invoice["currency"], "amount_minor": invoice["total_minor"],
                                        "invoice_refs": [invoice["invoice_id"]]})
        self._count()
        problems = verify(self.episode)
        self.assertTrue(any("is referred to by a payment" in p for p in problems), problems)

    def test_moving_a_copy_into_the_period_or_far_away_is_caught(self):
        doc = sorted(self.copies)[0]
        invoice = next(inv for inv in self.public["invoices"] if inv["doc_id"] == doc)
        invoice["issued_at"] = (parse_time(self.public["policy"]["period_end"]) + dt.timedelta(days=9)).isoformat()
        problems = verify(self.episode)
        self.assertTrue(any("is not dated just after the period" in p for p in problems), problems)

    def test_unposting_a_shifted_service_invoice_is_caught(self):
        doc = sorted(self.shifted)[0]
        invoice = next(inv for inv in self.public["invoices"] if inv["doc_id"] == doc)
        self.public["ledger"] = [r for r in self.public["ledger"]
                                 if not (r["invoice_id"] == invoice["invoice_id"] and r["payment_id"] is None)]
        self._count()
        problems = verify(self.episode)
        self.assertTrue(any("for another period's service is not posted" in p for p in problems), problems)


class TricksTest(unittest.TestCase):
    """The fast version of `python -m tools.tricks --check` (the full check runs 300 seeds per config)."""

    def test_shortcuts_fall_and_definitions_stay_exact(self):
        for label, config in (("default", {}), ("default + I2", {"interaction": "I2"})):
            stats = tricks.measure(config, 60, seed_base=91_000)
            self.assertEqual(stats["solver_mismatches"], 0, label)
            for method in ("duplicates (definition)", "cutoff (definition)"):
                s = stats[method]
                self.assertEqual((s["hits"], s["predicted"]), (s["truth"], s["truth"]), (label, method))
            for method in ("duplicates by key", "cutoff by service"):
                s = stats[method]
                self.assertLess(tricks.ratio(s["hits"], s["predicted"]), tricks.SHORTCUT_MAX_PRECISION, (label, method))

    def test_without_distractors_the_key_shortcut_is_exact(self):
        """The decoys are what defeats the shortcut: without distractors it was enough."""
        stats = tricks.measure({"distractors": False}, 30, seed_base=91_000)
        s = stats["duplicates by key"]
        self.assertEqual((s["hits"], s["predicted"]), (s["truth"], s["truth"]))


if __name__ == "__main__":
    unittest.main()
