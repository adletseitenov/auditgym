"""allocation_gap_mode (docs/DECISIONS.md S): a partial, a fully missing and a mixed allocation gap."""
import unittest
from collections import defaultdict

from auditgym import Env
from auditgym.config import ConfigError, normalize_config
from auditgym.generate import generate
from auditgym.solver import solve
from auditgym.verify import verify
from policies.runner import make_policy, run_episode
from tools import fuzz

SIZE = {"invoices": [40, 56], "background_pairs": [10, 40]}
SEEDS = range(30)


def make(mode=None, gap=2, **extra):
    config = {"errors": {"allocation_gap": gap}, "size": SIZE, "currencies": ["USD"], **extra}
    if mode is not None:
        config["allocation_gap_mode"] = mode
    return config


def gap_facts(episode, kind="allocation_gap"):
    """(issue, closing rows of the payment on its invoice, unallocated rows, payment, invoice) per issue of `kind`."""
    public, found = episode["public"], []
    docs = {d["doc_id"]: d for d in public["invoices"] + public["payments"]}
    for issue in (i for i in episode["private"]["truth"]["issues"] if i["type"] == kind):
        payment = next(docs[d] for d in issue["affected_ids"] if "payment_id" in docs[d])
        invoice = next(docs[d] for d in issue["affected_ids"] if "line_items" in docs[d])
        rows = [r for r in public["ledger"] if r["payment_id"] == payment["payment_id"]]
        closing = [r for r in rows if r["account"] == "receivable" and r["invoice_id"] == invoice["invoice_id"]]
        found.append((issue, closing, [r for r in rows if r["account"] == "unallocated"], payment, invoice))
    return found


class ConfigTest(unittest.TestCase):
    def test_default_is_partial_and_not_part_of_the_normalized_config(self):
        self.assertEqual(normalize_config({}), normalize_config({"allocation_gap_mode": "partial"}))
        self.assertNotIn("allocation_gap_mode", normalize_config({"allocation_gap_mode": "partial"}))

    def test_other_modes_are_kept(self):
        for mode in ("missing", "mixed"):
            self.assertEqual(normalize_config({"allocation_gap_mode": mode})["allocation_gap_mode"], mode)

    def test_bad_values_are_config_errors(self):
        for bad in ("full", "", "Partial", None, 1, True, ["missing"], {"a": 1}):
            with self.assertRaises(ConfigError) as caught:
                normalize_config({"allocation_gap_mode": bad})
            self.assertEqual(caught.exception.code, "bad_gap_mode")

    def test_partial_is_the_old_episode_byte_for_byte(self):
        for seed in range(5):
            self.assertEqual(generate(seed, make()), generate(seed, make("partial")))


class MissingTest(unittest.TestCase):
    def test_the_whole_closing_moves_to_unallocated(self):
        for seed in SEEDS:
            episode = generate(seed, make("missing"))
            self.assertEqual(verify(episode), [])
            for issue, closing, unallocated, payment, invoice in gap_facts(episode):
                self.assertEqual(issue["type"], "allocation_gap")
                self.assertEqual(closing, [])  # the payment is posted but not applied to the invoice
                self.assertEqual(len(unallocated), 1)
                self.assertEqual(unallocated[0]["credit_minor"], issue["expected_amount_minor"])
                self.assertGreater(issue["expected_amount_minor"], 0)
                self.assertEqual(payment["invoice_refs"], [invoice["invoice_id"]])
                debit = [r["row_id"] for r in episode["public"]["ledger"] if r["invoice_id"] == invoice["invoice_id"]
                         and not r["payment_id"] and r["account"] == "receivable"]
                self.assertEqual(issue["evidence_sets"], [sorted(issue["affected_ids"]) + sorted(
                    [unallocated[0]["row_id"], debit[0]])])

    def test_books_balance(self):
        for seed in SEEDS:
            for currencies in (["USD"], []):
                ledger = generate(seed, make("missing", currencies=currencies))["public"]["ledger"]
                self.assertEqual(sum(r["debit_minor"] - r["credit_minor"] for r in ledger), 0)
                per_payment = defaultdict(int)
                for row in ledger:
                    if row["payment_id"]:
                        per_payment[row["payment_id"]] += row["debit_minor"] - row["credit_minor"]
                self.assertEqual({v for v in per_payment.values()}, {0})

    def test_solver_finds_the_missing_allocation(self):
        for seed in SEEDS:
            episode = generate(seed, make("missing", gap=1))
            truth = episode["private"]["truth"]["issues"][0]
            hits = [f for f in solve(episode["public"]) if f["type"] == "allocation_gap"]
            self.assertEqual([sorted(f["affected_ids"]) for f in hits], [sorted(truth["affected_ids"])])
            self.assertEqual(hits[0]["claimed_amount_minor"], truth["expected_amount_minor"])
            self.assertTrue(any(set(group) <= set(hits[0]["evidence_ids"]) for group in truth["evidence_sets"]))

    def test_oracle_and_solver_score_in_the_environment(self):
        for mode in ("missing", "mixed"):
            config = make(mode, gap=3, currencies=["USD", "EUR"])
            for seed in (1, 2, 3):
                episode = generate(seed, config)
                with Env(mode="train") as env:
                    self.assertEqual(run_episode(env, make_policy("oracle", truth=episode["private"]["truth"]),
                                                 seed, config)["reward"], 1.0)
                    info = run_episode(env, make_policy("solver_policy"), seed, config)["info"]
                    self.assertEqual((info["precision"], info["recall"], info["f1"]), (1.0, 1.0, 1.0))

    def test_every_other_error_type_alongside_still_verifies(self):
        config = make("missing", gap=2, size={"invoices": [64, 80], "background_pairs": [50, 100]})
        config["errors"] = {"duplicate_invoice": 1, "allocation_gap": 2, "fx_error": 1, "period_cutoff": 1}
        for seed in range(15):
            self.assertEqual(verify(generate(seed, config)), [])

    def test_i1_stays_partial(self):
        for seed in range(10):
            episode = generate(seed, make("missing", gap=0, interaction="I1"))
            self.assertEqual(verify(episode), [])
            for _, closing, unallocated, *_ in gap_facts(episode, "allocation_gap"):
                self.assertEqual((len(closing), len(unallocated)), (1, 1))


class MixedTest(unittest.TestCase):
    def test_both_variants_occur_and_verify(self):
        seen = set()
        for seed in SEEDS:
            episode = generate(seed, make("mixed", gap=3))
            self.assertEqual(verify(episode), [])
            for issue, closing, unallocated, *_ in gap_facts(episode):
                seen.add("partial" if closing else "missing")
                self.assertEqual(len(unallocated), 1)
                self.assertEqual(unallocated[0]["credit_minor"], issue["expected_amount_minor"])
        self.assertEqual(seen, {"partial", "missing"})

    def test_deterministic(self):
        self.assertEqual(generate(4, make("mixed", gap=3)), generate(4, make("mixed", gap=3)))


class SeparationTest(unittest.TestCase):
    """The affected invoice of a gap is not told apart by what a gap needs anyway (a paid invoice)."""

    def test_distractor_overpayments_and_partials_stay_clean(self):
        for seed in range(20):
            episode = generate(seed, make("missing", gap=1, size={"invoices": [60, 80], "background_pairs": [20, 60]}))
            found = [f for f in solve(episode["public"]) if f["type"] == "allocation_gap"]
            self.assertEqual(len(found), 1, seed)

    def test_fuzz_grid_covers_the_modes(self):
        modes = {config.get("allocation_gap_mode") for _, config in fuzz.grid()}
        self.assertTrue({"missing", "mixed"} <= modes)


if __name__ == "__main__":
    unittest.main()
