"""Task E8: a policy that does not analyse the documents is not better than silence.

"Blind" policies here choose what to accuse by position, id or chance and copy the fields a finding
needs (amount, currency, linked ledger rows) mechanically. They must not do better than chance on
books with errors, and must not beat silence on a mix like the closed evaluation (CASE §06: 20
episodes, five clean, five with interacting errors).

Before the fix, pages followed the document date, so the first and last invoices were the cut-off
errors: "accuse the two first and two last invoices" scored -0.33 on cut-off books where silence
scores -1.0, and +0.40 on dense ones (AUDIT R6). Pages now follow evaluator.order_key.
"""
import json
import math
import pathlib
import shutil
import tempfile
import unittest

from auditgym.evaluator import Evaluator, order_key
from auditgym.rng import Rng
from auditgym.storage import write_episode
from auditgym.generate import generate
from tests.env.helpers import FakeClock, full_walk, strict_json
from tools import leakscan, metamorph

SEEDS = range(8_000_001, 8_000_005)
WIDER = {"invoices": [48, 100], "background_pairs": [150, 300]}


def _configs() -> list[tuple[str, dict]]:
    return metamorph.supported_configs() + [(f"dense{n}", c) for n, c in enumerate(leakscan._dense_configs())]


def _page_all(evaluator: Evaluator, table: str) -> list[dict]:
    records, cursor = [], None
    while True:
        observation = evaluator.step({"action": "list_records", "table": table, "cursor": cursor})[0]
        records += observation["records"]
        cursor = observation["next_cursor"]
        if cursor is None:
            return records


def _invoice_finding(invoice: dict, rows: dict, kind: str) -> dict:
    evidence = [invoice["doc_id"]] + rows.get(invoice["invoice_id"], [])[:9]
    return {"type": kind, "affected_ids": [invoice["doc_id"]], "claimed_amount_minor": abs(invoice["total_minor"]),
            "currency": invoice["currency"], "evidence_ids": evidence, "explanation": ""}


def _picks(name: str, invoices: list[dict], seed: int) -> list[dict]:
    if name == "positional":
        return invoices[:2] + invoices[-2:] if len(invoices) > 4 else invoices
    if name == "random":
        return Rng(seed, "blind-random").sample(invoices, min(4, len(invoices)))
    if name == "id_order":  # smallest and largest ids
        ordered = sorted(invoices, key=lambda invoice: invoice["doc_id"])
        return ordered[:2] + ordered[-2:] if len(ordered) > 4 else ordered
    raise ValueError(name)


def _play(name: str, seed: int, config) -> tuple[float, int, int]:
    """(reward, picked documents that some issue affects, picked documents) of one blind policy."""
    evaluator = Evaluator(clock=FakeClock())
    first, _ = evaluator.reset(seed, config)
    episode_id = first["manifest"]["episode_id"]
    if name == "silence":
        return evaluator.step({"action": "submit", "findings": {"episode_id": episode_id, "findings": []}})[1], 0, 0
    invoices = _page_all(evaluator, "invoices")
    rows: dict[str, list[str]] = {}
    for row in _page_all(evaluator, "ledger"):
        if row["invoice_id"] and not row["payment_id"]:
            rows.setdefault(row["invoice_id"], []).append(row["row_id"])
    picks = _picks(name, invoices, seed)
    findings = [_invoice_finding(invoice, rows, kind) for invoice in picks
                for kind in ("period_cutoff", "duplicate_invoice", "fx_error")][:30]
    affected = {i for issue in evaluator._episode["private"]["truth"]["issues"] for i in issue["affected_ids"]}
    hits = sum(invoice["doc_id"] in affected for invoice in picks)
    reward = evaluator.step({"action": "submit", "findings": {"episode_id": episode_id, "findings": findings}})[1]
    return reward, hits, len(picks)


class PageOrderTest(unittest.TestCase):
    def test_pages_follow_the_id_derived_order(self):
        episode = generate(8_000_101, {"errors": {"period_cutoff": 2, "duplicate_invoice": 2}})
        evaluator = Evaluator(clock=FakeClock())
        evaluator.reset(8_000_101, {"errors": {"period_cutoff": 2, "duplicate_invoice": 2}})
        for table in ("ledger", "invoices", "payments", "fx"):
            served = _page_all(evaluator, table)
            expected = sorted(episode["public"][table], key=lambda record, table=table: order_key(table, record))
            self.assertEqual(strict_json(served), strict_json(expected), table)
        fx = _page_all(evaluator, "fx")
        self.assertEqual([r["date"] for r in fx], sorted(r["date"] for r in fx))

    def test_reordered_files_give_the_same_observations(self):
        episode = generate(8_000_102, {"errors": {"period_cutoff": 1}})
        with tempfile.TemporaryDirectory() as directory:
            original, shuffled = pathlib.Path(directory) / "a", pathlib.Path(directory) / "b"
            write_episode(episode, original)
            shutil.copytree(original, shuffled)
            for name in ("ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"):
                path = shuffled / "public" / name
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                head, body = (lines[:1], lines[1:]) if name.endswith(".csv") else ([], lines)
                body.reverse()
                path.write_text("".join(head + body), encoding="utf-8")
            transcripts = []
            for source in (original, shuffled):
                evaluator = Evaluator(clock=FakeClock())
                transcripts.append(strict_json([evaluator.load(source)] + full_walk(evaluator)))
        self.assertEqual(transcripts[0], transcripts[1])


class BlindPolicyTest(unittest.TestCase):
    """E8 done-criterion, measured on every supported config and the dense ones."""

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for label, config in _configs():
            for seed in SEEDS:
                for name in ("silence", "positional", "random", "id_order"):
                    cls.results.setdefault(name, []).append((label, seed) + _play(name, seed, config))

    def test_position_and_id_order_find_errors_only_by_chance(self):
        for name in ("positional", "id_order"):
            hits = sum(row[3] for row in self.results[name])
            chance = sum(row[3] for row in self.results["random"])
            picks = sum(row[4] for row in self.results[name])
            # Generous bound: three standard deviations of a binomial at the chance rate.
            rate = chance / max(picks, 1)
            bound = chance + 3 * math.sqrt(picks * rate * (1 - rate)) + 3
            self.assertLessEqual(hits, bound, f"{name}: {hits} affected documents in {picks} picks, chance {chance}")

    def test_blind_policies_lose_to_silence_on_clean_books(self):
        for name in ("positional", "random", "id_order"):
            for label, seed, reward, _, _ in self.results[name]:
                if label.startswith("clean"):
                    self.assertEqual(reward, -1.0, (name, label, seed))

    def test_blind_policies_do_not_beat_silence_on_a_closed_run_mix(self):
        """CASE §06: 20 episodes, five clean, five with interacting errors, ten others."""
        configs = dict(_configs())
        clean = [configs["clean"], configs["clean-multi"]]
        interacting = [c for label, c in configs.items() if label.startswith(("I1", "I2"))]
        other = [c for label, c in configs.items() if not label.startswith(("clean", "I1", "I2"))]
        mix = [clean[n % 2] for n in range(5)] + [interacting[n % len(interacting)] for n in range(5)] \
            + [other[n % len(other)] for n in range(10)]
        means = {}
        for name in ("silence", "positional", "random", "id_order"):
            rewards = [_play(name, 8_000_201 + number, config)[0] for number, config in enumerate(mix)]
            means[name] = sum(rewards) / len(rewards)
        for name in ("positional", "random", "id_order"):
            self.assertLess(means[name], means["silence"], json.dumps(means))


if __name__ == "__main__":
    unittest.main()
