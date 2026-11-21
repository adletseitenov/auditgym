"""The organizers' OFFICIAL matcher and reward checks of 20.11 (CHECKS.md, Block 3, tests 1-20, and
the open examples M01-M12), run through the same entry a jury `score_episode` would call:
`auditgym.judge.judge(episode, payload, inspection_steps)`. See docs/DECISIONS.md, section R.

Fixture: one error A of the supported type `duplicate_invoice`, canonical records R1.. (invoices),
amount 10000 minor units KZT, allowed evidence set {D1, D2} (ledger rows), D3 an existing extra
evidence id, U absent from the books, tolerance 0 (M08: tolerance 1). Expected values are written
as the page gives them (fractions where the page gives fractions); rewards compare to 1e-9.
"""
import unittest
from fractions import Fraction

from auditgym.judge import judge
from auditgym.matcher import match

EP = "ep_official"
T = "duplicate_invoice"
OTHER_TYPES = ("allocation_gap", "fx_error", "period_cutoff")
TOLERANCE = 1e-9


def _row(row_id):
    return {"row_id": row_id, "timestamp": "2026-10-10T10:00:00+05:00", "account": "cash",
            "debit_minor": 1, "credit_minor": 0, "currency": "KZT", "entity_id": "e",
            "invoice_id": None, "payment_id": None}


def _doc(doc_id):
    return {"doc_id": doc_id, "invoice_id": "inv" + doc_id, "entity_id": "e",
            "issued_at": "2026-10-03T10:00:00+05:00", "due_at": "2026-11-02T10:00:00+05:00",
            "currency": "KZT", "total_minor": 10000, "line_items": []}


def issue(issue_id, affected, evidence_sets=(("D1", "D2"),), amount=10000, currency="KZT", kind=T):
    return {"issue_id": issue_id, "type": kind, "affected_ids": list(affected),
            "expected_amount_minor": amount, "currency": currency,
            "evidence_sets": [list(s) for s in evidence_sets]}


def episode(issues, tolerance=0, records=5):
    ledger = [_row(f"D{k}") for k in range(1, 13)]  # D1..D12 exist; U does not
    invoices = [_doc(f"R{k}") for k in range(1, records + 1)]
    policy = {"base_currency": "KZT", "evidence_superset_allowed": True,
              "amount_tolerance_minor": {t: tolerance for t in (T, *OTHER_TYPES)}}
    public = {"manifest": {"episode_id": EP, "base_currency": "KZT"}, "policy": policy,
              "ledger": ledger, "invoices": invoices, "payments": [], "fx": []}
    return {"public": public, "private": {"truth": {"episode_id": EP, "issues": issues}}}


def finding(affected=("R1",), evidence=("D1", "D2"), amount=10000, currency="KZT", kind=T, explanation="ok"):
    return {"type": kind, "affected_ids": list(affected), "claimed_amount_minor": amount,
            "currency": currency, "evidence_ids": list(evidence), "explanation": explanation}


def submit(findings):
    return {"episode_id": EP, "findings": findings}


def formula(tp, fp, fn, steps):
    f1 = Fraction(1) if tp + fp + fn == 0 else Fraction(2 * tp, 2 * tp + fp + fn)
    return float(max(Fraction(-1), min(Fraction(1), 2 * f1 - 1 - Fraction(2, 1000) * steps)))


class OfficialCase(unittest.TestCase):
    def check(self, ep, findings, tp, fp, fn, reward, steps=10, precision=None, recall=None, f1=None):
        result = judge(ep, submit(findings), steps)
        self.assertIsNone(result["structure_error"], result)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (tp, fp, fn))
        for name, want in (("precision", precision), ("recall", recall), ("f1", f1)):
            if want is not None:
                self.assertAlmostEqual(result[name], float(want), delta=TOLERANCE, msg=name)
        self.assertAlmostEqual(result["reward"], float(reward), delta=TOLERANCE)
        self.assertAlmostEqual(result["reward"], formula(tp, fp, fn, steps), delta=TOLERANCE)
        return result

    def structure(self, ep, payload, code=None, steps=10):
        result = judge(ep, payload, steps)
        self.assertEqual(result["reward"], -1.0)
        self.assertIsNotNone(result["structure_error"], result)
        if code:
            self.assertEqual(result["structure_error"], code)
        self.assertEqual((result["tp"], result["fp"]), (0, 0))
        return result


class Block3Test(OfficialCase):
    def a(self, **kw):
        return episode([issue("A", ["R1"], **kw)])

    def test_01_one_correct_finding(self):
        self.check(self.a(), [finding()], 1, 0, 0, 0.98, precision=1, recall=1, f1=1)

    def test_02_empty_findings(self):
        self.check(self.a(), [], 0, 0, 1, -1, precision=0, recall=0, f1=0)

    def test_03_empty_truth_empty_findings(self):
        self.check(episode([]), [], 0, 0, 0, 0.98, precision=0, recall=0, f1=1)

    def test_04_empty_truth_one_false_finding(self):
        self.check(episode([]), [finding()], 0, 1, 0, -1, f1=0)

    def test_05_two_identical_findings(self):
        result = self.check(self.a(), [finding(), finding()], 1, 1, 0, Fraction(47, 150),
                            precision=Fraction(1, 2), recall=1, f1=Fraction(2, 3))
        self.assertAlmostEqual(result["reward"], 0.3133333333, delta=1e-9)
        self.assertLess(result["reward"], judge(self.a(), submit([finding()]), 10)["reward"])

    def test_06_wrong_type(self):
        self.check(self.a(), [finding(kind="fx_error")], 0, 1, 1, -1, f1=0)

    def test_07_other_existing_record(self):
        self.check(self.a(), [finding(affected=["R3"])], 0, 1, 1, -1, f1=0)

    def test_08_amount_off_by_one_at_tolerance_zero(self):
        self.check(self.a(), [finding(amount=10001)], 0, 1, 1, -1, f1=0)

    def test_09_other_currency(self):
        self.check(self.a(), [finding(currency="USD")], 0, 1, 1, -1, f1=0)

    def test_10_only_d1(self):
        self.check(self.a(), [finding(evidence=["D1"])], 0, 1, 1, -1, f1=0)

    def test_11_missing_evidence_id_instead_of_d2(self):
        self.check(self.a(), [finding(evidence=["D1", "U"])], 0, 1, 1, -1, f1=0)

    def test_12_mass_flagging_five_records_four_types(self):
        ep = episode([issue("A", ["R1"])], records=5)
        every = [finding(affected=[f"R{r}"], kind=k) for r in range(1, 6) for k in (T, *OTHER_TYPES)]
        self.assertEqual(len(every), 20)
        result = self.check(ep, every, 1, 19, 0, Fraction(-871, 1050),
                            precision=Fraction(1, 20), recall=1, f1=Fraction(2, 21))
        self.assertAlmostEqual(result["reward"], -0.8295238095, delta=1e-9)

    def test_13_two_errors_one_found(self):
        ep = episode([issue("A", ["R1"]), issue("B", ["R2"])])
        self.check(ep, [finding()], 1, 0, 1, Fraction(47, 150),
                   precision=1, recall=Fraction(1, 2), f1=Fraction(2, 3))

    def test_14_two_errors_one_found_one_false(self):
        ep = episode([issue("A", ["R1"]), issue("B", ["R2"])])
        self.check(ep, [finding(), finding(affected=["R3"])], 1, 1, 1, -0.02,
                   precision=Fraction(1, 2), recall=Fraction(1, 2), f1=Fraction(1, 2))

    def test_15_zero_inspection_steps(self):
        self.check(self.a(), [finding()], 1, 0, 0, 1, steps=0, f1=1)

    def test_16_fifty_inspection_steps(self):
        result = self.check(self.a(), [finding()], 1, 0, 0, 0.90, steps=50, f1=1)
        self.assertAlmostEqual(judge(self.a(), submit([finding()]), 10)["reward"] - result["reward"], 0.08,
                               delta=1e-9)

    def test_17_hundred_inspection_steps(self):
        self.check(self.a(), [finding()], 1, 0, 0, 0.80, steps=100, f1=1)

    def test_18_thirty_one_findings(self):
        self.structure(self.a(), submit([finding()] * 31), "too_many_findings")
        self.check(self.a(), [finding()] * 30, 1, 29, 0, formula(1, 29, 0, 10))  # 30 is fine

    def test_19_eleven_different_evidence_ids(self):
        ids = [f"D{k}" for k in range(1, 12)]
        self.structure(self.a(), submit([finding(evidence=ids)]), "too_many_evidence_ids")
        self.check(self.a(), [finding(evidence=ids[:10])], 1, 0, 0, 0.98)  # 10 existing ids: a superset, TP

    def test_20_explanation_of_1001_ascii_characters(self):
        self.structure(self.a(), submit([finding(explanation="a" * 1001)]), "explanation_too_long")
        self.check(self.a(), [finding(explanation="a" * 1000)], 1, 0, 0, 0.98)

    def test_18_to_20_are_logged_by_the_evaluator(self):
        """The structure error is written to the evaluator's journal and ends the episode with -1."""
        import json
        import pathlib
        import tempfile

        from auditgym.evaluator import Evaluator
        from tests.env.helpers import TINY

        for label, change in (("31 findings", lambda f: [f] * 31),
                              ("11 evidence ids", lambda f: [dict(f, evidence_ids=[f"x{k}" for k in range(11)])]),
                              ("1001 characters", lambda f: [dict(f, explanation="a" * 1001)])):
            with self.subTest(label), tempfile.TemporaryDirectory() as scratch:
                path = pathlib.Path(scratch) / "results.jsonl"
                evaluator = Evaluator(mode="train", results_path=path)
                evaluator.load(str(TINY))
                good = {"type": T, "affected_ids": ["k7m2qa", "z5c1ty"], "claimed_amount_minor": 5000000,
                        "currency": "KZT", "evidence_ids": ["k7m2qa", "z5c1ty", "a1", "a3"], "explanation": "x"}
                observation, reward, terminated, _, _ = evaluator.step(
                    {"action": "submit", "findings": {"episode_id": "ep_tiny0001", "findings": change(good)}})
                row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
                self.assertTrue(terminated)
                self.assertEqual(row["reward"], -1.0)
                self.assertIsNotNone(row["structure_error"])


class ConventionsTest(OfficialCase):
    """Exact comparison rules of «Точные соглашения matcher»."""

    def a(self, **kw):
        return episode([issue("A", ["R1"], **kw)])

    def test_ids_are_compared_exactly(self):
        for wrong in ("r1", " R1", "R1 ", "R01"):
            with self.subTest(wrong):
                self.check(self.a(), [finding(affected=[wrong])], 0, 1, 1, -1)
        self.check(self.a(), [finding(evidence=["d1", "D2"])], 0, 1, 1, -1)

    def test_ids_are_not_coerced_from_numbers(self):
        for bad in (finding(affected=[1]), finding(evidence=["D1", 2])):
            self.structure(self.a(), submit([bad]), "id_not_string")

    def test_type_is_exact(self):
        for wrong in ("Duplicate_Invoice", " duplicate_invoice", "duplicate-invoice", "fraud"):
            with self.subTest(wrong):
                self.check(self.a(), [finding(kind=wrong)], 0, 1, 1, -1)  # an FP, not a structure error
        self.structure(self.a(), submit([finding(kind=7)]), "type_not_string")

    def test_amount_must_be_a_json_integer(self):
        for bad in ("10000", True, False, 10000.0, 10000.5, None, [10000]):
            with self.subTest(repr(bad)):
                self.structure(self.a(), submit([finding(amount=bad)]))

    def test_limits_count_the_raw_input_in_code_points(self):
        self.check(self.a(), [finding(explanation="\U0001F600" * 1000)], 1, 0, 0, 0.98)
        self.structure(self.a(), submit([finding(explanation="\U0001F600" * 1001)]), "explanation_too_long")
        self.check(self.a(), [finding(explanation="ж" * 1000)], 1, 0, 0, 0.98)

    def test_one_non_existent_evidence_id_makes_a_false_positive_in_a_valid_answer(self):
        self.check(self.a(), [finding(), finding(evidence=["D1", "D2", "U"])], 1, 1, 0, formula(1, 1, 0, 10))

    def test_a_structure_error_anywhere_voids_the_whole_answer(self):
        self.structure(self.a(), submit([finding(), finding(amount="1")]), "amount_not_int")

    def test_per_type_tolerance_is_taken_from_policy(self):
        ep = episode([issue("A", ["R1"])], tolerance=0)
        ep["public"]["policy"]["amount_tolerance_minor"][T] = 5
        self.check(ep, [finding(amount=10005)], 1, 0, 0, 0.98)
        self.check(ep, [finding(amount=9995)], 1, 0, 0, 0.98)
        self.check(ep, [finding(amount=10006)], 0, 1, 1, -1)

    def test_missing_tolerance_means_zero(self):
        ep = self.a()
        del ep["public"]["policy"]["amount_tolerance_minor"]
        self.check(ep, [finding(amount=10001)], 0, 1, 1, -1)
        self.check(ep, [finding()], 1, 0, 0, 0.98)

    def test_f1_special_case_only_for_both_empty(self):
        self.check(episode([]), [finding()], 0, 1, 0, -1, f1=0)
        self.check(episode([]), [], 0, 0, 0, 0.98, f1=1)

    def test_reward_is_clipped_and_not_below_minus_one(self):
        self.check(self.a(), [], 0, 0, 1, -1, steps=100)
        self.check(self.a(), [finding()], 1, 0, 0, 1.0, steps=0)


class OpenExamplesTest(OfficialCase):
    """M01-M12: single error A, affected_ids={R1,R2}, amount 10000 KZT, evidence {D1,D2}, tau 0."""

    def a(self, tolerance=0):
        return episode([issue("A", ["R1", "R2"])], tolerance=tolerance)

    def f(self, **kw):
        kw.setdefault("affected", ["R1", "R2"])
        return finding(**kw)

    def test_baseline_correct_finding(self):
        self.check(self.a(), [self.f()], 1, 0, 0, 0.98)

    def test_m01_affected_ids_repeat_is_the_same_set(self):
        self.check(self.a(), [self.f(affected=["R2", "R1", "R1"])], 1, 0, 0, 0.98)
        pairs = match(self.a()["private"]["truth"], [self.normalized(self.f(affected=["R2", "R1", "R1"]))],
                      self.a()["public"])["pairs"]
        self.assertEqual(pairs, [[0, "A"]])

    def test_m01_a_repeated_finding_is_still_a_separate_fp(self):
        self.check(self.a(), [self.f(affected=["R2", "R1", "R1"]), self.f()], 1, 1, 0, Fraction(47, 150))

    def test_m02_extra_existing_unaffected_record(self):
        self.check(self.a(), [self.f(affected=["R1", "R2", "R3"])], 0, 1, 1, -1)

    def test_m03_superset_with_existing_available_evidence(self):
        self.check(self.a(), [self.f(evidence=["D2", "D1", "D3"])], 1, 0, 0, 0.98)

    def test_m03_unbounded_existing_superset_including_many_ids(self):
        ids = ["D2", "D1"] + [f"D{k}" for k in range(3, 11)]
        self.check(self.a(), [self.f(evidence=ids)], 1, 0, 0, 0.98)

    def test_m04_unavailable_id_beside_the_minimal_set(self):
        self.check(self.a(), [self.f(evidence=["D1", "D2", "U"])], 0, 1, 1, -1)

    def test_m05_raw_limit_before_repeats_collapse(self):
        self.structure(self.a(), submit([self.f(evidence=["D1"] * 10 + ["D2"])]), "too_many_evidence_ids")
        self.structure(self.a(), submit([self.f(evidence=["D1"] * 11)]), "too_many_evidence_ids")
        self.check(self.a(), [self.f(evidence=["D1"] * 9 + ["D2"])], 1, 0, 0, 0.98)

    def test_m06_currency_is_exact(self):
        for wrong in ("kzt", "Kzt", " KZT", "KZT ", "KZT\n"):
            with self.subTest(repr(wrong)):
                self.check(self.a(), [self.f(currency=wrong)], 0, 1, 1, -1)

    def test_m07_amount_one_off_with_tau_zero(self):
        self.check(self.a(), [self.f(amount=10001)], 0, 1, 1, -1)
        self.check(self.a(), [self.f(amount=9999)], 0, 1, 1, -1)

    def test_m08_tolerance_one_is_inclusive(self):
        ep = self.a(tolerance=1)
        self.check(ep, [self.f(amount=10001)], 1, 0, 0, 0.98)
        self.check(ep, [self.f(amount=9999)], 1, 0, 0, 0.98)
        self.check(ep, [self.f(amount=10002)], 0, 1, 1, -1)
        self.check(ep, [self.f(amount=9998)], 0, 1, 1, -1)

    def test_m09_amount_string_or_bool(self):
        self.structure(self.a(), submit([self.f(amount="10000")]), "amount_not_int")
        self.structure(self.a(), submit([self.f(amount=True)]), "amount_not_int")

    # M10-M12 give the graph of valid edges explicitly: the two issues share type, ids, amount,
    # currency and differ only in their allowed evidence sets.
    def graph(self, sets_a, sets_b):
        return episode([issue("A", ["R1", "R2"], evidence_sets=sets_a),
                        issue("B", ["R1", "R2"], evidence_sets=sets_b)])

    def test_m10_maximum_matching_beats_greedy(self):
        ep = self.graph([("D1", "D2")], [("D1", "D3")])
        f0 = self.f(evidence=["D1", "D2", "D3"])  # fits A and B
        f1 = self.f(evidence=["D1", "D2"])        # fits A only
        for order in ([f0, f1], [f1, f0]):
            self.check(ep, order, 2, 0, 0, formula(2, 0, 0, 10))
        pairs = match(ep["private"]["truth"], [self.normalized(f0), self.normalized(f1)], ep["public"])["pairs"]
        self.assertEqual(pairs, [[0, "B"], [1, "A"]])  # F1 -> A, F0 -> B

    def test_m11_deterministic_choice_when_both_fit_both(self):
        ep = self.graph([("D1", "D2")], [("D1", "D2")])
        f0, f1 = self.f(explanation="first"), self.f(explanation="second")
        result = self.check(ep, [f0, f1], 2, 0, 0, formula(2, 0, 0, 10))
        self.assertEqual(self.check(ep, [f1, f0], 2, 0, 0, formula(2, 0, 0, 10)), result)
        pairs = match(ep["private"]["truth"], [self.normalized(f0), self.normalized(f1)], ep["public"])["pairs"]
        self.assertEqual(pairs, [[0, "A"], [1, "B"]])  # A -> F0, B -> F1
        # the truth order does not matter either: issue_ids are sorted by code points
        ep["private"]["truth"]["issues"].reverse()
        pairs = match(ep["private"]["truth"], [self.normalized(f0), self.normalized(f1)], ep["public"])["pairs"]
        self.assertEqual(pairs, [[0, "A"], [1, "B"]])

    def test_m11_ties_are_ordered_by_code_points_of_issue_id(self):
        ep = episode([issue("b", ["R1", "R2"]), issue("B", ["R1", "R2"]), issue("a", ["R1", "R2"])])
        fs = [self.normalized(self.f()) for _ in range(3)]
        pairs = match(ep["private"]["truth"], fs, ep["public"])["pairs"]
        self.assertEqual(pairs, [[0, "B"], [1, "a"], [2, "b"]])  # "B" < "a" < "b"

    def test_m11_smallest_issue_wins_a_single_finding(self):
        ep = self.graph([("D1", "D2")], [("D1", "D2")])
        f = self.normalized(self.f())
        self.assertEqual(match(ep["private"]["truth"], [f], ep["public"])["pairs"], [[0, "A"]])

    def test_m12_two_identical_correct_findings(self):
        self.check(self.a(), [self.f(), self.f()], 1, 1, 0, Fraction(47, 150))
        self.assertEqual(judge(self.a(), submit([self.f(), self.f()]), 10)["reward"],
                         judge(episode([issue("A", ["R1"])]), submit([finding(), finding()]), 10)["reward"])

    @staticmethod
    def normalized(raw):
        return {**raw, "affected_ids": sorted(set(raw["affected_ids"])),
                "evidence_ids": sorted(set(raw["evidence_ids"]))}


class TieBreakIsLexicographicallySmallestTest(unittest.TestCase):
    """Among maximum matchings: the lexicographically smallest sorted sequence of (issue_id, index)."""

    def test_against_brute_force(self):
        import itertools
        import random

        rng = random.Random(20)
        for _ in range(300):
            n_issues, n_findings = rng.randint(1, 4), rng.randint(1, 5)
            ids = sorted(rng.sample(["A", "B", "Z", "a", "b"], n_issues))
            edges = {(i, f) for i in ids for f in range(n_findings) if rng.random() < 0.5}
            issues = [issue(i, [f"R{k + 1}"]) for k, i in enumerate(ids)]
            # one distinct finding shape per edge set is not needed: patch the edge predicate instead
            from unittest import mock

            from auditgym import matcher

            def fake(finding, iss, policy, known):
                return (iss["issue_id"], finding["n"]) in edges

            findings = [{"n": f} for f in range(n_findings)]
            with mock.patch.object(matcher, "is_match", fake):
                got = match({"issues": issues}, findings, {"policy": {}})["pairs"]
            best = None
            for size in range(len(ids), -1, -1):
                options = []
                for chosen in itertools.combinations(ids, size):
                    for used in itertools.permutations(range(n_findings), size):
                        if all((i, f) in edges for i, f in zip(chosen, used)):
                            options.append(sorted(zip(chosen, used)))
                if options:
                    best = min(options)
                    break
            self.assertEqual(sorted((i, f) for f, i in got), best, (edges, ids))


if __name__ == "__main__":
    unittest.main()
